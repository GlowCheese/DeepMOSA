####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_uri_default_params. Retrieved 3/4 statements.
# Partially parsed test_uri_with_scheme. Retrieved 2/5 statements.
# Partially parsed test_uri_with_tld_type. Retrieved 1/7 statements.
# Partially parsed test_uri_with_subdomains. Retrieved 6/8 statements.
# Partially parsed test_uri_path_format. Retrieved 12/14 statements.


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.uri()
    var_2 = 'https://'

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 'http://'

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 'www'
    var_2 = 'api'
    var_3 = 'blog'
    var_4 = [var_1, var_2, var_3]
    var_5 = var_0.uri(subdomains=var_4)

import mimesis.providers.internet as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 3
    var_2 = var_0.uri(query_params_count=var_1)
    var_3 = 1
    var_4 = '?'
    var_5 = uri.split(var_4)[var_3]
    var_6 = '&'
    var_7 = module_1.split(var_6)
    var_8 = len(var_7)
    assert var_8 == 3

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 0
    var_2 = var_0.uri(query_params_count=var_1)

import mimesis.providers.internet as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 32
    var_2 = var_0.uri(query_params_count=var_1)
    var_3 = 1
    var_4 = '?'
    var_5 = uri.split(var_4)[var_3]
    var_6 = '&'
    var_7 = module_1.split(var_6)
    var_8 = len(var_7)
    assert var_8 == 32

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 33
    var_2 = var_0.uri(query_params_count=var_1)

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.uri()
    var_2 = 1
    var_3 = 0
    var_4 = '//'
    var_5 = uri.split(var_4)[var_2]
    var_6 = '?'
    var_7 = var_3.split(var_6)[var_3]
    var_8 = '/'
    var_9 = var_5.split(var_8)[var_2:]
    var_10 = '-'
    var_11 = ''



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_slug_contains_only_valid_characters. Retrieved 5/7 statements.


import mimesis.providers.internet as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 5
    var_2 = var_0.slug(var_1)
    var_3 = '-'
    var_4 = module_1.split(var_3)
    var_5 = len(var_4)
    assert var_5 == 5

import mimesis.providers.internet as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.slug()
    var_2 = '-'
    var_3 = module_1.split(var_2)
    var_4 = len(var_3)

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 1
    var_2 = var_0.slug(var_1)

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 13
    var_2 = var_0.slug(var_1)

import mimesis.providers.internet as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 3
    var_2 = var_0.slug(var_1)
    var_3 = '-'
    var_4 = module_1.split(var_3)

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.slug()
    var_2 = var_0.slug()



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_query_parameters_default_length. Retrieved 3/4 statements.
# Partially parsed test_query_parameters_custom_length. Retrieved 4/5 statements.
# Partially parsed test_query_parameters_max_length. Retrieved 4/5 statements.
# Partially parsed test_query_parameters_unique_keys. Retrieved 3/8 statements.
# Partially parsed test_query_parameters_values_are_strings. Retrieved 3/6 statements.


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.query_parameters()
    var_2 = len(var_1)

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 5
    var_2 = var_0.query_parameters(var_1)
    var_3 = len(var_2)
    assert var_3 == 5

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 32
    var_2 = var_0.query_parameters(var_1)
    var_3 = len(var_2)
    assert var_3 == 32

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 33
    var_2 = var_0.query_parameters(var_1)

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 0
    var_2 = var_0.query_parameters(var_1)

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = -1
    var_2 = var_0.query_parameters(var_1)

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 10
    var_2 = var_0.query_parameters(var_1)

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 5
    var_2 = var_0.query_parameters(var_1)



# Parsed testcases at query #4
#--------------------------




import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 33
    var_2 = var_0.query_parameters(var_1)



# Parsed testcases at query #5
#--------------------------




import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = False
    var_2 = 33
    var_3 = var_0.query_parameters(var_2)
    var_4 = True



# Parsed testcases at query #6
#--------------------------




import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 33
    var_2 = var_0.query_parameters(var_1)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_query_parameters_length_gt_32_raises_value_error. Retrieved 4/7 statements.


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 33
    var_2 = 'test'
    var_3 = [var_2]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_query_parameters_max_length_exceeded. Retrieved 6/7 statements.


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 33
    var_2 = False
    var_3 = 33
    var_4 = var_0.query_parameters(var_3)
    var_5 = True



# Parsed testcases at query #9
#--------------------------




import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 31
    var_2 = var_0.query_parameters(var_1)
    var_3 = len(var_2)



# Parsed testcases at query #10
#--------------------------




import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 32
    var_2 = var_0.query_parameters(var_1)
    var_3 = len(var_2)
    assert var_3 == 32



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_slug_default_parts_count. Retrieved 5/6 statements.
# Partially parsed test_slug_specific_parts_count. Retrieved 6/7 statements.


import mimesis.providers.internet as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.slug()
    var_2 = '-'
    var_3 = module_1.split(var_2)
    var_4 = len(var_3)

import mimesis.providers.internet as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 5
    var_2 = var_0.slug(var_1)
    var_3 = '-'
    var_4 = module_1.split(var_3)
    var_5 = len(var_4)
    assert var_5 == 5

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 13
    var_2 = var_0.slug(var_1)

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 1
    var_2 = var_0.slug(var_1)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_url_default_scheme. Retrieved 3/4 statements.
# Partially parsed test_url_custom_scheme. Retrieved 2/5 statements.
# Partially parsed test_url_with_port. Retrieved 1/3 statements.
# Partially parsed test_url_with_tld_type. Retrieved 1/7 statements.
# Partially parsed test_url_with_subdomains. Retrieved 6/8 statements.
# Partially parsed test_url_without_slash. Retrieved 3/4 statements.


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.url()
    var_2 = 'https://'

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 'http://'

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 'www'
    var_2 = 'api'
    var_3 = 'blog'
    var_4 = [var_1, var_2, var_3]
    var_5 = var_0.url(subdomains=var_4)

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.url()
    var_2 = '/'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_query_parameters_default_length. Retrieved 3/4 statements.
# Partially parsed test_query_parameters_custom_length. Retrieved 4/5 statements.
# Partially parsed test_query_parameters_max_length. Retrieved 4/5 statements.
# Partially parsed test_query_parameters_zero_length. Retrieved 4/5 statements.


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.query_parameters()
    var_2 = len(var_1)

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 5
    var_2 = var_0.query_parameters(var_1)
    var_3 = len(var_2)
    assert var_3 == 5

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 32
    var_2 = var_0.query_parameters(var_1)
    var_3 = len(var_2)
    assert var_3 == 32

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 0
    var_2 = var_0.query_parameters(var_1)
    var_3 = len(var_2)
    assert var_3 == 0

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 33
    var_2 = var_0.query_parameters(var_1)

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = -1
    var_2 = var_0.query_parameters(var_1)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_url_generates_correct_format. Retrieved 4/6 statements.
# Partially parsed test_url_with_port_range_includes_port. Retrieved 2/5 statements.
# Partially parsed test_url_with_tld_type_uses_correct_tld. Retrieved 1/7 statements.
# Partially parsed test_url_with_subdomains_includes_subdomain. Retrieved 6/8 statements.
# Partially parsed test_url_with_http_scheme_uses_http. Retrieved 2/5 statements.
# Partially parsed test_url_with_https_scheme_uses_https. Retrieved 2/5 statements.


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.url()
    var_2 = 'https://'
    var_3 = '/'

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = '/'

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 'www'
    var_2 = 'api'
    var_3 = [var_1, var_2]
    var_4 = var_0.url(subdomains=var_3)
    var_5 = [var_1, var_2]

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 'http://'

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 'https://'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_url_with_port_range. Retrieved 2/5 statements.


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = '/'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_query_parameters_default_length. Retrieved 3/4 statements.
# Partially parsed test_query_parameters_custom_length. Retrieved 4/5 statements.
# Partially parsed test_query_parameters_max_length. Retrieved 4/5 statements.


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.query_parameters()
    var_2 = len(var_1)

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 5
    var_2 = var_0.query_parameters(var_1)
    var_3 = len(var_2)
    assert var_3 == 5

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 32
    var_2 = var_0.query_parameters(var_1)
    var_3 = len(var_2)
    assert var_3 == 32

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 33
    var_2 = var_0.query_parameters(var_1)

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 0
    var_2 = var_0.query_parameters(var_1)

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = -1
    var_2 = var_0.query_parameters(var_1)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_query_parameters_length_greater_than_32_raises_value_error. Retrieved 5/8 statements.


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 0
    var_2 = var_0.random
    var_3 = 33
    var_4 = var_0.query_parameters(var_3)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_url_with_port_range. Retrieved 1/3 statements.


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()



# Parsed testcases at query #9
#--------------------------




import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 33
    var_2 = var_0.query_parameters(var_1)



# Parsed testcases at query #10
#--------------------------




import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 33
    var_2 = var_0.query_parameters(var_1)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_query_parameters_with_default_length. Retrieved 8/9 statements.
# Partially parsed test_query_parameters_with_specific_length. Retrieved 4/5 statements.
# Partially parsed test_query_parameters_with_max_length. Retrieved 4/5 statements.


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.query_parameters()
    var_2 = len(var_1)
    var_3 = 1
    var_4 = var_2 >= var_3
    var_5 = len(var_1)
    var_6 = 10
    var_7 = var_5 <= var_6

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 5
    var_2 = var_0.query_parameters(var_1)
    var_3 = len(var_2)
    assert var_3 == 5

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 32
    var_2 = var_0.query_parameters(var_1)
    var_3 = len(var_2)
    assert var_3 == 32

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 33
    var_2 = var_0.query_parameters(var_1)

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 0
    var_2 = var_0.query_parameters(var_1)

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = -1
    var_2 = var_0.query_parameters(var_1)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_query_parameters_default_length. Retrieved 3/4 statements.
# Partially parsed test_query_parameters_custom_length. Retrieved 4/5 statements.
# Partially parsed test_query_parameters_max_length. Retrieved 4/5 statements.
# Partially parsed test_query_parameters_keys_and_values. Retrieved 3/8 statements.


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.query_parameters()
    var_2 = len(var_1)

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 5
    var_2 = var_0.query_parameters(var_1)
    var_3 = len(var_2)
    assert var_3 == 5

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 32
    var_2 = var_0.query_parameters(var_1)
    var_3 = len(var_2)
    assert var_3 == 32

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 33
    var_2 = var_0.query_parameters(var_1)

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 0
    var_2 = var_0.query_parameters(var_1)

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = -1
    var_2 = var_0.query_parameters(var_1)

import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 3
    var_2 = var_0.query_parameters(var_1)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_query_parameters_length_gt_32_raises_value_error. Retrieved 4/6 statements.


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 33
    var_2 = 33
    var_3 = var_0.query_parameters(var_2)



