####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_currency_of_creates_valid_currency. Retrieved 4/6 statements.
# Partially parsed test_currency_of_creates_valid_currency_with_zero_decimals. Retrieved 3/5 statements.
# Partially parsed test_currency_of_creates_valid_currency_with_negative_decimals. Retrieved 3/5 statements.
# Partially parsed test_currency_of_raises_error_for_non_string_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_non_alphabetic_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_non_uppercase_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_non_string_name. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_empty_name. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_name_with_leading_space. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_name_with_trailing_space. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_non_integer_decimals. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_decimals_less_than_minus_one. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_non_currencytype_type. Retrieved 4/6 statements.
# Partially parsed test_currency_equality. Retrieved 4/10 statements.
# Partially parsed test_currency_hash_equality. Retrieved 4/14 statements.
# Partially parsed test_currency_quantize_with_positive_decimals. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_with_zero_decimals. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_with_negative_decimals. Retrieved 7/15 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = module_0.make_quantizer(var_2)

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1

def test_case_0():
    var_0 = 123
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'US1'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'usd'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 123
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = ''
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = ' US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars '
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = '2'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = -2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'MONEY'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '1.005'
    var_4 = '1.00'
    var_5 = '1.015'
    var_6 = '1.02'

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '0.5'
    var_4 = '0'
    var_5 = '1.5'
    var_6 = '2'

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1
    var_3 = '1.0000000000005'
    var_4 = '1.000000000000'
    var_5 = '1.0000000000015'
    var_6 = '1.000000000002'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_constructor_singleton_preserves_state. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__registry
    var_2 = len(var_1)
    assert var_2 == 0
    var_3 = var_0._CurrencyRegistry__currencies
    var_4 = len(var_3)
    assert var_4 == 0
    var_5 = var_0._CurrencyRegistry__codes
    var_6 = len(var_5)
    assert var_6 == 0
    var_7 = var_0._CurrencyRegistry__codenames
    var_8 = len(var_7)
    assert var_8 == 0

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_constructor_singleton_persistence. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #4
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #5
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_constructor_singleton_preserves_state. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_constructor_singleton_preserves_state. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_constructor_singleton_preserves_state. Retrieved 2/3 statements.
# Partially parsed test_constructor_get_with_default_returns_default. Retrieved 5/8 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = [CurrencyRegistry() for _ in var_1]
    var_3 = 0
    var_4 = var_2[var_3]

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = var_0.has(var_1)
    assert var_2 is False

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = var_0[var_1]

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = var_0.get(var_1)
    assert var_2 is None

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'EUR'
    var_2 = 'Euro'
    var_3 = 2
    var_4 = 'USD'



# Parsed testcases at query #9
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_constructor_singleton_preserves_state. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_currency_registry_context_manager. Retrieved 1/3 statements.
# Partially parsed test_currency_registry_get. Retrieved 6/9 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = var_0.has(var_1)
    assert var_2 is False

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = var_0[var_1]

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = var_0.get(var_1)
    assert var_2 is None
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_constructor_singleton_persistence. Retrieved 4/6 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'test'
    var_2 = 'dummy'
    var_3 = module_0.CurrencyRegistry()



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_constructor_singleton_preserves_state. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #14
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_currency_registry_constructor_context_closed_initially. Retrieved 4/8 statements.
# Partially parsed test_currency_registry_constructor_private_attributes. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'TEST'
    var_2 = 'Test Currency'
    var_3 = 2

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = []



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_currency_of_creates_valid_currency. Retrieved 5/9 statements.
# Partially parsed test_currency_of_with_zero_decimals. Retrieved 3/5 statements.
# Partially parsed test_currency_of_with_negative_decimals. Retrieved 3/5 statements.
# Partially parsed test_currency_of_raises_on_non_string_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_non_alpha_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_non_uppercase_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_non_string_name. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_empty_name. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_name_with_leading_space. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_name_with_trailing_space. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_non_integer_decimals. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_decimals_less_than_minus_one. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_non_currencytype_type. Retrieved 4/6 statements.
# Partially parsed test_currency_equality. Retrieved 3/8 statements.
# Partially parsed test_currency_inequality_due_to_name. Retrieved 4/9 statements.
# Partially parsed test_currency_quantize_with_positive_decimals. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_with_zero_decimals. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_with_negative_decimals. Retrieved 7/15 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = module_0.make_quantizer(var_2)
    var_4 = module_0.make_quantizer(var_2)

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1

def test_case_0():
    var_0 = 123
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'US1'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'usd'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 123
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = ''
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = ' US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars '
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = '2'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = -2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'MONEY'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '1.005'
    var_4 = '1.00'
    var_5 = '1.015'
    var_6 = '1.02'

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '0.5'
    var_4 = '0'
    var_5 = '1.5'
    var_6 = '2'

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1
    var_3 = '1.0000000000005'
    var_4 = '1.000000000000'
    var_5 = '1.0000000000015'
    var_6 = '1.000000000002'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_constructor_singleton_preserves_state. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #18
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_constructor_singleton_persistence. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #20
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__codes
    var_2 = bool(var_1)
    assert var_2 is False



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_currency_registry_constructor_context_closed_initially. Retrieved 4/8 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'TEST'
    var_2 = 'Test Currency'
    var_3 = 2

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = '_CurrencyRegistry__registry'
    var_2 = hasattr(var_0, var_1)
    var_3 = '_CurrencyRegistry__currencies'
    var_4 = hasattr(var_0, var_3)
    var_5 = '_CurrencyRegistry__codes'
    var_6 = hasattr(var_0, var_5)
    var_7 = '_CurrencyRegistry__codenames'
    var_8 = hasattr(var_0, var_7)
    var_9 = '_CurrencyRegistry__ctx_open'
    var_10 = hasattr(var_0, var_9)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_constructor_singleton_preserves_state. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_currency_of_creates_valid_instance. Retrieved 5/10 statements.
# Partially parsed test_currency_of_with_zero_decimals. Retrieved 3/5 statements.
# Partially parsed test_currency_of_with_negative_decimals. Retrieved 3/5 statements.
# Partially parsed test_currency_of_raises_on_invalid_code_type. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_non_alphabetic_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_non_uppercase_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_invalid_name_type. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_empty_name. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_name_with_leading_space. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_name_with_trailing_space. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_invalid_decimals_type. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_decimals_less_than_minus_one. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_invalid_currency_type. Retrieved 4/6 statements.
# Partially parsed test_currency_equality. Retrieved 4/10 statements.
# Partially parsed test_currency_hash_equality. Retrieved 4/14 statements.
# Partially parsed test_currency_quantize_positive_decimals. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_zero_decimals. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_negative_decimals. Retrieved 7/15 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = module_0.make_quantizer(var_2)
    var_4 = module_0.make_quantizer(var_2)

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1

def test_case_0():
    var_0 = 123
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'US1'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'usd'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 123
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = ''
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = ' US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars '
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = '2'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = -2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'MONEY'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '1.005'
    var_4 = '1.00'
    var_5 = '1.015'
    var_6 = '1.02'

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '0.5'
    var_4 = '0'
    var_5 = '1.5'
    var_6 = '2'

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1
    var_3 = '1.0000000000005'
    var_4 = '1.000000000000'
    var_5 = '1.0000000000015'
    var_6 = '1.000000000002'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_currency_of_creates_valid_currency. Retrieved 5/9 statements.
# Partially parsed test_currency_of_with_zero_decimals. Retrieved 3/7 statements.
# Partially parsed test_currency_of_with_negative_decimals. Retrieved 4/8 statements.
# Partially parsed test_currency_of_raises_on_non_string_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_non_alphabetic_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_non_uppercase_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_non_string_name. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_empty_name. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_name_with_leading_space. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_name_with_trailing_space. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_non_integer_decimals. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_decimals_less_than_minus_one. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_non_currencytype_type. Retrieved 4/6 statements.
# Partially parsed test_currency_equality. Retrieved 3/8 statements.
# Partially parsed test_currency_inequality_due_to_name. Retrieved 4/9 statements.
# Partially parsed test_currency_inequality_due_to_code. Retrieved 4/9 statements.
# Partially parsed test_currency_inequality_due_to_decimals. Retrieved 4/9 statements.
# Partially parsed test_currency_inequality_due_to_type. Retrieved 3/9 statements.
# Partially parsed test_currency_quantize_positive_decimals. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_zero_decimals. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_negative_decimals. Retrieved 7/15 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = module_0.make_quantizer(var_2)
    var_4 = module_0.make_quantizer(var_2)

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1
    var_3 = -1

def test_case_0():
    var_0 = 123
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'US1'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'usd'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 123
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = ''
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = ' US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars '
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = '2'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = -2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'MONEY'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 3

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '1.005'
    var_4 = '1.00'
    var_5 = '1.015'
    var_6 = '1.02'

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '0.5'
    var_4 = '0'
    var_5 = '1.5'
    var_6 = '2'

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1
    var_3 = '1.0000000000005'
    var_4 = '1.000000000000'
    var_5 = '1.0000000000015'
    var_6 = '1.000000000002'



# Parsed testcases at query #25
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_currency_of_creates_valid_instance. Retrieved 5/10 statements.
# Partially parsed test_currency_of_creates_instance_with_zero_decimals. Retrieved 3/5 statements.
# Partially parsed test_currency_of_creates_instance_with_negative_decimals. Retrieved 3/5 statements.
# Partially parsed test_currency_of_raises_error_for_non_string_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_non_alphabetic_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_non_uppercase_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_non_string_name. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_empty_name. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_name_with_leading_or_trailing_spaces. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_non_integer_decimals. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_decimals_less_than_minus_one. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_non_currencytype_type. Retrieved 4/6 statements.
# Partially parsed test_currency_equality_based_on_hashcache. Retrieved 4/14 statements.
# Partially parsed test_currency_quantize_with_positive_decimals. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_with_zero_decimals. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_with_negative_decimals. Retrieved 7/15 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = module_0.make_quantizer(var_2)
    var_4 = module_0.make_quantizer(var_2)

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1

def test_case_0():
    var_0 = 123
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'US1'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'usd'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 123
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = ''
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = ' US Dollars '
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = '2'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = -2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'MONEY'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '1.005'
    var_4 = '1.00'
    var_5 = '1.015'
    var_6 = '1.02'

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '0.5'
    var_4 = '0'
    var_5 = '1.5'
    var_6 = '2'

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1
    var_3 = '1.0000000000005'
    var_4 = '1.000000000000'
    var_5 = '1.0000000000015'
    var_6 = '1.000000000002'



# Parsed testcases at query #27
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #28
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0.__enter__
    var_2 = callable(var_1)
    var_3 = var_0.__exit__
    var_4 = callable(var_3)
    var_5 = var_0.__register
    var_6 = callable(var_5)
    var_7 = var_0.__len__
    var_8 = callable(var_7)
    var_9 = var_0.__contains__
    var_10 = callable(var_9)
    var_11 = var_0.__getitem__
    var_12 = callable(var_11)
    var_13 = var_0.has
    var_14 = callable(var_13)
    var_15 = var_0.get
    var_16 = callable(var_15)

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_constructor_singleton_persistence. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #30
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_eq_same_currency_objects. Retrieved 3/7 statements.
# Partially parsed test_eq_different_currency_codes. Retrieved 5/9 statements.
# Partially parsed test_eq_different_names. Retrieved 4/8 statements.
# Partially parsed test_eq_different_decimals. Retrieved 6/10 statements.
# Partially parsed test_eq_different_types. Retrieved 6/10 statements.
# Partially parsed test_eq_with_non_currency_object. Retrieved 3/5 statements.
# Partially parsed test_eq_with_none. Retrieved 3/5 statements.
# Partially parsed test_eq_same_hash_different_fields. Retrieved 5/12 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'BTC'
    var_4 = 'Bitcoin'
    var_5 = 8

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'AAA'
    var_1 = 'Currency A'
    var_2 = 2
    var_3 = module_0.make_quantizer(var_2)
    var_4 = module_0.make_quantizer(var_2)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_constructor_singleton_preserves_state. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #3
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_constructor_initializes_registry. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = []

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #5
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #6
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_currency_registry_constructor_registry_empty. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = []



# Parsed testcases at query #8
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #9
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_currency_registry_enter_exit. Retrieved 1/3 statements.
# Partially parsed test_currency_registry_register_outside_context. Retrieved 2/4 statements.
# Partially parsed test_currency_registry_get_empty. Retrieved 6/9 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = None

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = var_0.has(var_1)
    assert var_2 is False

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = var_0[var_1]

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = var_0.get(var_1)
    assert var_2 is None
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_currency_of_creates_valid_currency. Retrieved 4/13 statements.
# Partially parsed test_currency_of_with_zero_decimals. Retrieved 4/13 statements.
# Partially parsed test_currency_of_with_negative_decimals. Retrieved 4/11 statements.
# Partially parsed test_currency_of_raises_on_non_string_code. Retrieved 3/7 statements.
# Partially parsed test_currency_of_raises_on_non_alphabetic_code. Retrieved 3/7 statements.
# Partially parsed test_currency_of_raises_on_non_uppercase_code. Retrieved 3/7 statements.
# Partially parsed test_currency_of_raises_on_non_string_name. Retrieved 3/7 statements.
# Partially parsed test_currency_of_raises_on_empty_name. Retrieved 3/7 statements.
# Partially parsed test_currency_of_raises_on_name_with_leading_space. Retrieved 3/7 statements.
# Partially parsed test_currency_of_raises_on_name_with_trailing_space. Retrieved 3/7 statements.
# Partially parsed test_currency_of_raises_on_non_integer_decimals. Retrieved 3/7 statements.
# Partially parsed test_currency_of_raises_on_decimals_less_than_minus_one. Retrieved 3/7 statements.
# Partially parsed test_currency_of_raises_on_non_currencytype_type. Retrieved 4/7 statements.
# Partially parsed test_currency_equality. Retrieved 3/10 statements.
# Partially parsed test_currency_inequality_due_to_name. Retrieved 4/11 statements.
# Partially parsed test_currency_quantize_with_positive_decimals. Retrieved 7/17 statements.
# Partially parsed test_currency_quantize_with_zero_decimals. Retrieved 7/17 statements.
# Partially parsed test_currency_quantize_with_negative_decimals. Retrieved 7/17 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = 2
    var_3 = '0.01'

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '0'

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Weird Crypto'
    var_2 = -1
    var_3 = -1

def test_case_0():
    var_0 = 123
    var_1 = 'US Dollar'
    var_2 = 2

def test_case_0():
    var_0 = 'US1'
    var_1 = 'US Dollar'
    var_2 = 2

def test_case_0():
    var_0 = 'usd'
    var_1 = 'US Dollar'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 123
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = ''
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = ' US Dollar'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar '
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = '2'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = -2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = 2
    var_3 = 'MONEY'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = 2
    var_3 = 'UX Dollar'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = 2
    var_3 = '1.005'
    var_4 = '1.00'
    var_5 = '1.015'
    var_6 = '1.02'

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '0.5'
    var_4 = '0'
    var_5 = '1.5'
    var_6 = '2'

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Weird Crypto'
    var_2 = -1
    var_3 = '1.0000000000005'
    var_4 = '1.000000000000'
    var_5 = '1.0000000000015'
    var_6 = '1.000000000002'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_currency_registry_constructor_context_closed_initially. Retrieved 4/8 statements.
# Partially parsed test_currency_registry_constructor_singleton_preserves_state. Retrieved 6/10 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'TEST'
    var_2 = 'Test Currency'
    var_3 = 2

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = module_0.CurrencyRegistry()
    var_5 = len(var_4)
    assert var_5 == 1



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_constructor_singleton_preserves_state. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #14
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_currency_registry_constructor_context_closed_initially. Retrieved 4/8 statements.
# Partially parsed test_currency_registry_constructor_reinitialization_preserves_singleton. Retrieved 4/6 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'TEST'
    var_2 = 'Test Currency'
    var_3 = 2

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'DUMMY'
    var_3 = var_0._CurrencyRegistry__registry[var_2]



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_currency_of_creates_valid_currency. Retrieved 4/13 statements.
# Partially parsed test_currency_of_with_zero_decimals. Retrieved 4/13 statements.
# Partially parsed test_currency_of_with_negative_decimals. Retrieved 4/11 statements.
# Partially parsed test_currency_of_raises_error_for_non_string_code. Retrieved 3/7 statements.
# Partially parsed test_currency_of_raises_error_for_non_alpha_code. Retrieved 3/7 statements.
# Partially parsed test_currency_of_raises_error_for_non_uppercase_code. Retrieved 3/7 statements.
# Partially parsed test_currency_of_raises_error_for_non_string_name. Retrieved 3/7 statements.
# Partially parsed test_currency_of_raises_error_for_empty_name. Retrieved 3/7 statements.
# Partially parsed test_currency_of_raises_error_for_name_with_leading_space. Retrieved 3/7 statements.
# Partially parsed test_currency_of_raises_error_for_name_with_trailing_space. Retrieved 3/7 statements.
# Partially parsed test_currency_of_raises_error_for_non_integer_decimals. Retrieved 3/7 statements.
# Partially parsed test_currency_of_raises_error_for_decimals_less_than_minus_one. Retrieved 3/7 statements.
# Partially parsed test_currency_of_raises_error_for_non_currencytype_type. Retrieved 4/7 statements.
# Partially parsed test_currency_equality. Retrieved 4/11 statements.
# Partially parsed test_currency_hash_equality. Retrieved 4/15 statements.
# Partially parsed test_currency_quantize_with_positive_decimals. Retrieved 7/17 statements.
# Partially parsed test_currency_quantize_with_zero_decimals. Retrieved 7/17 statements.
# Partially parsed test_currency_quantize_with_negative_decimals. Retrieved 7/17 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '0'

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1
    var_3 = -1

def test_case_0():
    var_0 = 123
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'US1'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'usd'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 123
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = ''
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = ' US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars '
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = '2'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = -2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'MONEY'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '1.005'
    var_4 = '1.00'
    var_5 = '1.015'
    var_6 = '1.02'

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '0.5'
    var_4 = '0'
    var_5 = '1.5'
    var_6 = '2'

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1
    var_3 = '1.0000000000005'
    var_4 = '1.000000000000'
    var_5 = '1.0000000000015'
    var_6 = '1.000000000002'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_currency_of_creates_valid_currency. Retrieved 5/10 statements.
# Partially parsed test_currency_of_creates_currency_with_zero_decimals. Retrieved 3/5 statements.
# Partially parsed test_currency_of_creates_currency_with_negative_decimals. Retrieved 3/5 statements.
# Partially parsed test_currency_of_raises_error_for_non_string_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_non_alphabetic_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_non_uppercase_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_non_string_name. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_empty_name. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_name_with_leading_space. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_name_with_trailing_space. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_non_integer_decimals. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_decimals_less_than_minus_one. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_non_currencytype_type. Retrieved 4/6 statements.
# Partially parsed test_currency_equality. Retrieved 4/10 statements.
# Partially parsed test_currency_hash_equality. Retrieved 4/14 statements.
# Partially parsed test_currency_quantize_positive_decimals. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_zero_decimals. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_negative_decimals. Retrieved 7/15 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = module_0.make_quantizer(var_2)
    var_4 = module_0.make_quantizer(var_2)

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1

def test_case_0():
    var_0 = 123
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'US1'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'usd'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 123
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = ''
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = ' US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars '
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = '2'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = -2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'MONEY'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '1.005'
    var_4 = '1.00'
    var_5 = '1.015'
    var_6 = '1.02'

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '0.5'
    var_4 = '0'
    var_5 = '1.5'
    var_6 = '2'

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1
    var_3 = '1.0000000000005'
    var_4 = '1.000000000000'
    var_5 = '1.0000000000015'
    var_6 = '1.000000000002'



# Parsed testcases at query #18
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #19
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_constructor_singleton_preserves_state. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_currency_of_creates_valid_currency. Retrieved 5/10 statements.
# Partially parsed test_currency_of_with_zero_decimals. Retrieved 3/5 statements.
# Partially parsed test_currency_of_with_negative_decimals. Retrieved 3/5 statements.
# Partially parsed test_currency_of_raises_error_for_non_string_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_non_alphabetic_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_non_uppercase_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_non_string_name. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_empty_name. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_name_with_leading_space. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_name_with_trailing_space. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_non_integer_decimals. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_decimals_less_than_minus_one. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_non_currencytype_type. Retrieved 4/6 statements.
# Partially parsed test_currency_equality. Retrieved 4/10 statements.
# Partially parsed test_currency_hash_equality. Retrieved 4/14 statements.
# Partially parsed test_currency_quantize_with_positive_decimals. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_with_zero_decimals. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_with_negative_decimals. Retrieved 7/15 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = module_0.make_quantizer(var_2)
    var_4 = module_0.make_quantizer(var_2)

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1

def test_case_0():
    var_0 = 123
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'US1'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'usd'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 123
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = ''
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = ' US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars '
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = '2'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = -2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'MONEY'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '1.005'
    var_4 = '1.00'
    var_5 = '1.015'
    var_6 = '1.02'

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '0.5'
    var_4 = '0'
    var_5 = '1.5'
    var_6 = '2'

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1
    var_3 = '1.0000000000005'
    var_4 = '1.000000000000'
    var_5 = '1.0000000000015'
    var_6 = '1.000000000002'



# Parsed testcases at query #22
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_constructor_singleton_preserves_state. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_constructor_singleton_persistence. Retrieved 4/5 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'test'
    var_2 = 'dummy'
    var_3 = module_0.CurrencyRegistry()



# Parsed testcases at query #25
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #26
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_currency_registry_initial_get_with_default. Retrieved 5/8 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = var_0.has(var_1)
    assert var_2 is False

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = var_0[var_1]

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = var_0.get(var_1)
    assert var_2 is None

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'EUR'
    var_2 = 'Euro'
    var_3 = 2
    var_4 = 'USD'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_currency_of_creates_valid_instance. Retrieved 5/10 statements.
# Partially parsed test_currency_of_raises_on_invalid_code_type. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_code_not_alpha. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_code_not_upper. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_invalid_name_type. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_empty_name. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_name_with_leading_space. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_name_with_trailing_space. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_invalid_decimals_type. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_decimals_less_than_minus_one. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_invalid_currency_type. Retrieved 4/6 statements.
# Partially parsed test_currency_of_with_zero_decimals. Retrieved 3/5 statements.
# Partially parsed test_currency_of_with_negative_decimals. Retrieved 3/5 statements.
# Partially parsed test_currency_of_positive_decimals_quantizer. Retrieved 4/6 statements.
# Partially parsed test_currency_equality. Retrieved 3/9 statements.
# Partially parsed test_currency_inequality. Retrieved 5/11 statements.
# Partially parsed test_currency_quantize_positive_decimals. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_zero_decimals. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_negative_decimals. Retrieved 7/15 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = module_0.make_quantizer(var_2)
    var_4 = module_0.make_quantizer(var_2)

def test_case_0():
    var_0 = 123
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'US1'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'usd'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 123
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = ''
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = ' US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars '
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = '2'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = -2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'MONEY'

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1

import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'Euro'
    var_2 = 2
    var_3 = module_0.make_quantizer(var_2)

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '1.005'
    var_4 = '1.00'
    var_5 = '1.015'
    var_6 = '1.02'

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '0.5'
    var_4 = '0'
    var_5 = '1.5'
    var_6 = '2'

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1
    var_3 = '1.0000000000005'
    var_4 = '1.000000000000'
    var_5 = '1.0000000000015'
    var_6 = '1.000000000002'



# Parsed testcases at query #29
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #30
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_currency_registry_constructor_context_closed_initially. Retrieved 4/8 statements.
# Partially parsed test_currency_registry_constructor_private_registry_empty. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'TEST'
    var_2 = 'Test Currency'
    var_3 = 2

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = []

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_currency_registry_constructor_context_closed_initially. Retrieved 4/8 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'TEST'
    var_2 = 'Test Currency'
    var_3 = 2

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = '_CurrencyRegistry__registry'
    var_2 = hasattr(var_0, var_1)
    var_3 = '_CurrencyRegistry__currencies'
    var_4 = hasattr(var_0, var_3)
    var_5 = '_CurrencyRegistry__codes'
    var_6 = hasattr(var_0, var_5)
    var_7 = '_CurrencyRegistry__codenames'
    var_8 = hasattr(var_0, var_7)
    var_9 = '_CurrencyRegistry__ctx_open'
    var_10 = hasattr(var_0, var_9)



# Parsed testcases at query #33
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #34
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



