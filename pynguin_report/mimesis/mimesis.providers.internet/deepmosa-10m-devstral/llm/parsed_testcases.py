####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_url_default. Retrieved 2/7 statements.
# Partially parsed test_url_with_scheme. Retrieved 2/7 statements.
# Partially parsed test_url_with_port_range. Retrieved 7/10 statements.
# Partially parsed test_url_with_tld_type. Retrieved 1/5 statements.
# Partially parsed test_url_with_subdomains. Retrieved 3/5 statements.


def test_case_0():
    var_0 = []
    var_1 = 'https://'
    var_2 = '/'
    var_3 = '.'

def test_case_0():
    var_0 = []
    var_1 = 'http://'
    var_2 = '/'

def test_case_0():
    var_0 = []
    var_1 = ':'
    var_2 = 0
    var_3 = -1
    var_4 = ':'
    var_5 = url.split(var_4)[var_3]
    var_6 = '/'
    var_7 = var_5.split(var_6)[var_2]
    var_8 = int(var_7)
    var_9 = 0
    var_10 = bool(0 <= var_8)
    assert var_10 is True
    var_11 = bool(var_8 <= 1023)
    assert var_11 is True

def test_case_0():
    var_0 = []
    var_1 = '/'
    var_2 = '.'

def test_case_0():
    var_0 = []
    var_1 = 'api'
    var_2 = 'v1'
    var_3 = [var_1, var_2]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_url. Retrieved 10/23 statements.


def test_case_0():
    var_0 = []
    var_1 = 'https://'
    var_2 = '/'
    var_3 = 'http://'
    var_4 = 1
    var_5 = '//'
    var_6 = url_with_port.split(var_5)[var_4]
    var_7 = ':'
    var_8 = bool(':' in var_6)
    assert var_8 is True
    var_9 = 'api'
    var_10 = 'v1'
    var_11 = [var_9, var_10]
    var_12 = [var_9, var_10]



# Parsed testcases at query #3
#--------------------------

# Failed to parse test_query_parameters_default_length.
# Partially parsed test_query_parameters_custom_length. Retrieved 1/8 statements.
# Partially parsed test_query_parameters_max_length. Retrieved 1/8 statements.
# Partially parsed test_query_parameters_exceeds_max_length. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 5
    var_2 = bool(var_1)
    assert var_2 is True

def test_case_0():
    var_0 = []
    var_1 = 32
    var_2 = bool(var_1)
    assert var_2 is True

def test_case_0():
    var_0 = []
    var_1 = 33
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_query_parameters_length_exceeds_maximum. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 33
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_query_parameters_length_above_32. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 33
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #6
#--------------------------

# Failed to parse test_url_with_port_range.




# Parsed testcases at query #7
#--------------------------

# Failed to parse test_url_with_port_range.




# Parsed testcases at query #8
#--------------------------

# Partially parsed test_query_parameters_length_above_32. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 33
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_query_parameters_raises_value_error_when_length_exceeds_32. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 33



# Parsed testcases at query #10
#--------------------------

# Failed to parse test_url_with_port_range.




# Parsed testcases at query #11
#--------------------------

# Partially parsed test_query_parameters_length_exceeds_maximum. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 33
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #12
#--------------------------

# Failed to parse test_url_with_port_range_includes_port.




# Parsed testcases at query #13
#--------------------------

# Partially parsed test_query_parameters_length_exceeds_maximum. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 33
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #14
#--------------------------

# Failed to parse test_url_with_port_range.




# Parsed testcases at query #15
#--------------------------

# Failed to parse test_url_with_port_range.




# Parsed testcases at query #16
#--------------------------

# Partially parsed test_query_parameters_raises_value_error_when_length_exceeds_32. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 33



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_url. Retrieved 5/10 statements.
# Partially parsed test_url_with_scheme. Retrieved 1/5 statements.
# Partially parsed test_url_with_port. Retrieved 3/6 statements.
# Partially parsed test_url_with_subdomains. Retrieved 4/8 statements.
# Partially parsed test_url_with_tld_type. Retrieved 3/6 statements.


def test_case_0():
    var_0 = []
    var_1 = 'https://'
    var_2 = '/'
    var_3 = 1
    var_4 = '//'
    var_5 = url.split(var_4)[var_3]
    var_6 = '.'
    var_7 = bool('.' in var_5)
    assert var_7 is True

def test_case_0():
    var_0 = []
    var_1 = 'http://'

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = '//'
    var_3 = url.split(var_2)[var_1]
    var_4 = ':'
    var_5 = bool(':' in var_3)
    assert var_5 is True

def test_case_0():
    var_0 = []
    var_1 = 'api'
    var_2 = 'www'
    var_3 = [var_1, var_2]
    var_4 = [var_1, var_2]

def test_case_0():
    var_0 = []
    var_1 = -1
    var_2 = '.'
    var_3 = url.split(var_2)[var_1]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_query_parameters_default_length. Retrieved 2/12 statements.
# Partially parsed test_query_parameters_custom_length. Retrieved 1/8 statements.
# Partially parsed test_query_parameters_max_length. Retrieved 1/8 statements.
# Partially parsed test_query_parameters_exceeds_max_length. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 10

def test_case_0():
    var_0 = []
    var_1 = 5
    var_2 = bool(var_1)
    assert var_2 is True

def test_case_0():
    var_0 = []
    var_1 = 32
    var_2 = bool(var_1)
    assert var_2 is True

def test_case_0():
    var_0 = []
    var_1 = 33
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_slug_with_default_parts_count. Retrieved 1/8 statements.
# Partially parsed test_slug_with_specific_parts_count. Retrieved 2/7 statements.
# Partially parsed test_slug_with_minimum_parts_count. Retrieved 2/7 statements.
# Partially parsed test_slug_with_maximum_parts_count. Retrieved 2/7 statements.
# Partially parsed test_slug_with_invalid_parts_count_raises_value_error. Retrieved 1/4 statements.
# Partially parsed test_slug_with_exceeding_parts_count_raises_value_error. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = '-'

def test_case_0():
    var_0 = []
    var_1 = 5
    var_2 = '-'

def test_case_0():
    var_0 = []
    var_1 = 2
    var_2 = '-'

def test_case_0():
    var_0 = []
    var_1 = 12
    var_2 = '-'

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = []
    var_1 = 13
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_query_parameters_default_length.
# Partially parsed test_query_parameters_custom_length. Retrieved 1/5 statements.
# Partially parsed test_query_parameters_max_length. Retrieved 1/5 statements.
# Partially parsed test_query_parameters_exceeds_max_length. Retrieved 1/4 statements.
# Partially parsed test_query_parameters_unique_keys. Retrieved 1/7 statements.
# Partially parsed test_query_parameters_values_are_strings. Retrieved 1/5 statements.


def test_case_0():
    var_0 = []
    var_1 = 5

def test_case_0():
    var_0 = []
    var_1 = 32

def test_case_0():
    var_0 = []
    var_1 = 33
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = []
    var_1 = 10

def test_case_0():
    var_0 = []
    var_1 = 5
    var_2 = bool(var_1)
    assert var_2 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_slug_raises_value_error_when_parts_count_less_than_2. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 42
    var_1 = []
    var_2 = 1
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_slug_parts_count_gt_12_raises_value_error. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 13



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_slug_parts_count_greater_than_12. Retrieved 2/6 statements.


import locale as module_0

def test_case_0():
    var_0 = []
    var_1 = 13
    var_2 = module_0.str(var_1)
    assert var_2 == "Slug's parts count must be <= 12"



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_query_parameters_length_exceeds_maximum. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 33
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_slug. Retrieved 2/10 statements.


def test_case_0():
    var_0 = []
    var_1 = 5
    var_2 = '-'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_slug_default_parts_count. Retrieved 1/8 statements.
# Partially parsed test_slug_custom_parts_count. Retrieved 2/7 statements.
# Partially parsed test_slug_maximum_parts_count. Retrieved 2/7 statements.
# Partially parsed test_slug_minimum_parts_count. Retrieved 2/7 statements.
# Partially parsed test_slug_value_error_parts_count_exceeds_maximum. Retrieved 1/4 statements.
# Partially parsed test_slug_value_error_parts_count_below_minimum. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = '-'

def test_case_0():
    var_0 = []
    var_1 = 5
    var_2 = '-'

def test_case_0():
    var_0 = []
    var_1 = 12
    var_2 = '-'

def test_case_0():
    var_0 = []
    var_1 = 2
    var_2 = '-'

def test_case_0():
    var_0 = []
    var_1 = 13
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #11
#--------------------------

# Failed to parse test_query_parameters_default_length.
# Partially parsed test_query_parameters_custom_length. Retrieved 1/8 statements.
# Partially parsed test_query_parameters_max_length. Retrieved 1/8 statements.
# Partially parsed test_query_parameters_exceeds_max_length. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 5
    var_2 = bool(var_1)
    assert var_2 is True

def test_case_0():
    var_0 = []
    var_1 = 32
    var_2 = bool(var_1)
    assert var_2 is True

def test_case_0():
    var_0 = []
    var_1 = 33
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_slug_default_parts_count. Retrieved 1/6 statements.
# Partially parsed test_slug_custom_parts_count. Retrieved 2/7 statements.
# Partially parsed test_slug_maximum_parts_count. Retrieved 2/7 statements.
# Partially parsed test_slug_minimum_parts_count. Retrieved 2/7 statements.
# Partially parsed test_slug_invalid_parts_count_above_max. Retrieved 1/4 statements.
# Partially parsed test_slug_invalid_parts_count_below_min. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = '-'
    var_2 = 2

def test_case_0():
    var_0 = []
    var_1 = 5
    var_2 = '-'

def test_case_0():
    var_0 = []
    var_1 = 12
    var_2 = '-'

def test_case_0():
    var_0 = []
    var_1 = 2
    var_2 = '-'

def test_case_0():
    var_0 = []
    var_1 = 13
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_slug_raises_value_error_when_parts_count_less_than_2. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 1



# Parsed testcases at query #14
#--------------------------

# Failed to parse test_query_parameters_default_length.
# Partially parsed test_query_parameters_specific_length. Retrieved 1/8 statements.
# Partially parsed test_query_parameters_max_length. Retrieved 1/8 statements.
# Partially parsed test_query_parameters_exceeds_max_length. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 5
    var_2 = bool(var_1)
    assert var_2 is True

def test_case_0():
    var_0 = []
    var_1 = 32
    var_2 = bool(var_1)
    assert var_2 is True

def test_case_0():
    var_0 = []
    var_1 = 33
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_slug_parts_count_less_than_2. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 1



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_slug_default_parts_count. Retrieved 1/5 statements.
# Partially parsed test_slug_custom_parts_count. Retrieved 2/6 statements.
# Partially parsed test_slug_min_parts_count. Retrieved 2/6 statements.
# Partially parsed test_slug_max_parts_count. Retrieved 2/6 statements.
# Partially parsed test_slug_raises_value_error_for_parts_count_less_than_2. Retrieved 1/4 statements.
# Partially parsed test_slug_raises_value_error_for_parts_count_greater_than_12. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = '-'
    var_2 = '-'
    var_3 = 1

def test_case_0():
    var_0 = []
    var_1 = 5
    var_2 = '-'

def test_case_0():
    var_0 = []
    var_1 = 2
    var_2 = '-'

def test_case_0():
    var_0 = []
    var_1 = 12
    var_2 = '-'

def test_case_0():
    var_0 = []
    var_1 = 1

def test_case_0():
    var_0 = []
    var_1 = 13



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_slug_default_parts_count. Retrieved 1/8 statements.
# Partially parsed test_slug_custom_parts_count. Retrieved 2/7 statements.
# Partially parsed test_slug_maximum_parts_count. Retrieved 2/7 statements.
# Partially parsed test_slug_minimum_parts_count. Retrieved 2/7 statements.
# Partially parsed test_slug_invalid_parts_count_above_maximum. Retrieved 1/4 statements.
# Partially parsed test_slug_invalid_parts_count_below_minimum. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = '-'

def test_case_0():
    var_0 = []
    var_1 = 5
    var_2 = '-'

def test_case_0():
    var_0 = []
    var_1 = 12
    var_2 = '-'

def test_case_0():
    var_0 = []
    var_1 = 2
    var_2 = '-'

def test_case_0():
    var_0 = []
    var_1 = 13
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_slug_raises_value_error_when_parts_count_less_than_2. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 1



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_query_parameters_with_length_33. Retrieved 2/6 statements.


import locale as module_0

def test_case_0():
    var_0 = []
    var_1 = 33
    var_2 = module_0.str(var_1)
    var_3 = 'Maximum allowed length of query parameters is 32.'
    var_4 = bool('Maximum allowed length of query parameters is 32.' in var_2)
    assert var_4 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_slug_default_parts_count. Retrieved 1/6 statements.
# Partially parsed test_slug_custom_parts_count. Retrieved 2/7 statements.
# Partially parsed test_slug_maximum_parts_count. Retrieved 2/7 statements.
# Partially parsed test_slug_invalid_parts_count. Retrieved 1/4 statements.
# Partially parsed test_slug_minimum_parts_count. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = '-'
    var_2 = '-'
    var_3 = 1

def test_case_0():
    var_0 = []
    var_1 = 5
    var_2 = '-'

def test_case_0():
    var_0 = []
    var_1 = 12
    var_2 = '-'

def test_case_0():
    var_0 = []
    var_1 = 13
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_slug_parts_count_greater_than_12. Retrieved 2/6 statements.


import locale as module_0

def test_case_0():
    var_0 = []
    var_1 = 13
    var_2 = module_0.str(var_1)
    var_3 = "Slug's parts count must be <= 12"
    var_4 = bool("Slug's parts count must be <= 12" in var_2)
    assert var_4 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_query_parameters_raises_value_error_for_length_greater_than_32. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 33



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_slug_parts_count_above_12_raises_value_error. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 13



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_slug_default_parts_count. Retrieved 1/8 statements.
# Partially parsed test_slug_custom_parts_count. Retrieved 2/7 statements.
# Partially parsed test_slug_maximum_parts_count. Retrieved 2/7 statements.
# Partially parsed test_slug_invalid_parts_count_above_max. Retrieved 1/4 statements.
# Partially parsed test_slug_invalid_parts_count_below_min. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = '-'

def test_case_0():
    var_0 = []
    var_1 = 5
    var_2 = '-'

def test_case_0():
    var_0 = []
    var_1 = 12
    var_2 = '-'

def test_case_0():
    var_0 = []
    var_1 = 13
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_query_parameters_length_above_32_raises_value_error. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 33



