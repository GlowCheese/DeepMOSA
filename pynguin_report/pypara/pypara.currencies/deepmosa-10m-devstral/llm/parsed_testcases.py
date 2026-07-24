####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_currency_constructor_with_valid_inputs. Retrieved 5/9 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 3/8 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 3/8 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 2
    var_4 = module_0.make_quantizer(var_3)

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1



# Parsed testcases at query #2
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True
    var_3 = len(var_0)
    assert var_3 == 0
    var_4 = var_0.all
    var_5 = bool(var_0.all == [])
    assert var_5 is True
    var_6 = var_0.codes
    var_7 = bool(var_0.codes == [])
    assert var_7 is True
    var_8 = var_0.codenames
    var_9 = bool(var_0.codenames == [])
    assert var_9 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_currency_constructor_creates_frozen_instance. Retrieved 4/13 statements.
# Partially parsed test_currency_constructor_sets_attributes_correctly. Retrieved 4/16 statements.
# Partially parsed test_currency_constructor_enables_ordering. Retrieved 6/20 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = [var_3]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = [var_3]
    var_6 = [var_3]
    var_7 = [var_3]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = [var_3]
    var_6 = 'EUR'
    var_7 = 'Euro'
    var_8 = [var_3]
    var_9 = [var_3]



# Parsed testcases at query #4
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True



# Parsed testcases at query #5
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True
    var_3 = len(var_0)
    assert var_3 == 0
    var_4 = 'USD'
    var_5 = var_0.has(var_4)
    var_6 = bool(not var_5)
    assert var_6 is True
    var_7 = var_0.get(var_4)
    assert var_7 is None



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_currency_constructor_with_valid_inputs. Retrieved 5/9 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 2
    var_4 = module_0.make_quantizer(var_3)



# Parsed testcases at query #7
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__codes
    var_2 = bool(var_0._CurrencyRegistry__codes == [])
    assert var_2 is True



# Parsed testcases at query #8
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #9
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_currency_registry_constructor. Retrieved 2/4 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0.__instance
    var_2 = bool(var_0.__instance is not None)
    assert var_2 is True
    var_3 = []
    var_4 = [var_3]
    var_5 = var_0.__registry
    var_6 = var_0.__currencies
    var_7 = bool(var_0.__currencies == [])
    assert var_7 is True
    var_8 = var_0.__codes
    var_9 = bool(var_0.__codes == [])
    assert var_9 is True
    var_10 = var_0.__codenames
    var_11 = bool(var_0.__codenames == [])
    assert var_11 is True
    var_12 = var_0.__ctx_open
    assert var_12 is False



# Parsed testcases at query #11
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__codes
    var_2 = bool(var_0._CurrencyRegistry__codes == [])
    assert var_2 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_currency_constructor_with_valid_parameters. Retrieved 5/9 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 2
    var_4 = module_0.make_quantizer(var_3)



# Parsed testcases at query #13
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True
    var_3 = len(var_0)
    assert var_3 == 0
    var_4 = 'USD'
    var_5 = bool('USD' not in var_0)
    assert var_5 is True
    var_6 = 'USD'
    var_7 = var_0.has(var_6)
    assert var_7 is False
    var_8 = var_0.get(var_6)
    assert var_8 is None
    var_9 = var_0.all
    var_10 = bool(var_0.all == [])
    assert var_10 is True
    var_11 = var_0.codes
    var_12 = bool(var_0.codes == [])
    assert var_12 is True
    var_13 = var_0.codenames
    var_14 = bool(var_0.codenames == [])
    assert var_14 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_currency_registry_constructor. Retrieved 4/5 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True
    var_3 = len(var_0)
    assert var_3 == 0
    var_4 = bool(not var_0.__ctx_open)
    assert var_4 is True
    var_5 = []
    var_6 = [var_5]
    var_7 = var_0.__registry
    var_8 = var_0.__currencies
    var_9 = bool(var_0.__currencies == [])
    assert var_9 is True
    var_10 = var_0.__codes
    var_11 = bool(var_0.__codes == [])
    assert var_11 is True
    var_12 = var_0.__codenames
    var_13 = bool(var_0.__codenames == [])
    assert var_13 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_currency_constructor_creates_frozen_instance. Retrieved 3/7 statements.
# Partially parsed test_currency_constructor_with_valid_inputs. Retrieved 4/6 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 3/5 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 3/5 statements.
# Partially parsed test_currency_constructor_with_invalid_code_type. Retrieved 3/6 statements.
# Partially parsed test_currency_constructor_with_invalid_code_case. Retrieved 3/6 statements.
# Partially parsed test_currency_constructor_with_invalid_code_chars. Retrieved 3/6 statements.
# Partially parsed test_currency_constructor_with_empty_name. Retrieved 3/6 statements.
# Partially parsed test_currency_constructor_with_whitespace_name. Retrieved 3/6 statements.
# Partially parsed test_currency_constructor_with_invalid_decimals_type. Retrieved 3/6 statements.
# Partially parsed test_currency_constructor_with_invalid_decimals_value. Retrieved 3/6 statements.
# Partially parsed test_currency_constructor_with_invalid_type. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

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
    var_0 = 'usd'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD1'
    var_1 = 'US Dollars'
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



# Parsed testcases at query #16
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True



# Parsed testcases at query #17
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__codes
    var_2 = bool(var_0._CurrencyRegistry__codes == [])
    assert var_2 is True



# Parsed testcases at query #18
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_currency_constructor_with_valid_parameters. Retrieved 5/9 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 2
    var_4 = module_0.make_quantizer(var_3)



# Parsed testcases at query #20
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_currency_constructor_creates_immutable_instance. Retrieved 3/7 statements.
# Partially parsed test_currency_constructor_stores_code. Retrieved 3/5 statements.
# Partially parsed test_currency_constructor_stores_name. Retrieved 3/5 statements.
# Partially parsed test_currency_constructor_stores_decimals. Retrieved 3/5 statements.
# Partially parsed test_currency_constructor_stores_type. Retrieved 3/5 statements.
# Partially parsed test_currency_constructor_stores_quantizer. Retrieved 4/6 statements.
# Partially parsed test_currency_constructor_stores_hashcache. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

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
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = module_0.make_quantizer(var_2)

import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = module_0.make_quantizer(var_2)



# Parsed testcases at query #22
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__codes
    var_2 = bool(var_0._CurrencyRegistry__codes == [])
    assert var_2 is True



# Parsed testcases at query #23
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__codes
    var_2 = bool(var_0._CurrencyRegistry__codes == [])
    assert var_2 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_currency_constructor_with_valid_inputs. Retrieved 6/12 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 3/10 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 3/10 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 2
    var_4 = module_0.make_quantizer(var_3)
    var_5 = module_0.make_quantizer(var_3)

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1



# Parsed testcases at query #25
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_currency_equality_same_attributes. Retrieved 3/7 statements.
# Partially parsed test_currency_equality_different_name. Retrieved 4/8 statements.
# Partially parsed test_currency_equality_different_type. Retrieved 3/7 statements.
# Partially parsed test_currency_equality_different_decimals. Retrieved 4/8 statements.
# Partially parsed test_currency_equality_different_code. Retrieved 5/9 statements.
# Partially parsed test_currency_equality_non_currency_object. Retrieved 3/5 statements.


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

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 0

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



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_currency_registry_constructor. Retrieved 1/2 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #3
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_currency_constructor_with_valid_parameters. Retrieved 4/8 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = module_0.make_quantizer(var_2)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_currency_registry_constructor. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0
    var_2 = var_0.all
    var_3 = bool(var_0.all == [])
    assert var_3 is True
    var_4 = var_0.codes
    var_5 = bool(var_0.codes == [])
    assert var_5 is True
    var_6 = var_0.codenames
    var_7 = bool(var_0.codenames == [])
    assert var_7 is True



# Parsed testcases at query #6
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__codes
    var_2 = bool(var_0._CurrencyRegistry__codes == [])
    assert var_2 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_currency_constructor_with_valid_inputs. Retrieved 4/8 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 3/8 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 3/8 statements.


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



# Parsed testcases at query #8
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_currency_constructor_with_valid_parameters. Retrieved 4/8 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = module_0.make_quantizer(var_2)



# Parsed testcases at query #10
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__codes
    var_2 = bool(var_0._CurrencyRegistry__codes == [])
    assert var_2 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_currency_registry_constructor. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0
    var_2 = var_0.codes
    var_3 = bool(var_0.codes == [])
    assert var_3 is True
    var_4 = var_0.codenames
    var_5 = bool(var_0.codenames == [])
    assert var_5 is True
    var_6 = var_0.all
    var_7 = bool(var_0.all == [])
    assert var_7 is True



# Parsed testcases at query #12
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_currency_constructor_with_valid_parameters. Retrieved 5/9 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 2
    var_4 = module_0.make_quantizer(var_3)



# Parsed testcases at query #14
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #15
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True
    var_3 = len(var_0)
    assert var_3 == 0
    var_4 = 'USD'
    var_5 = var_0.has(var_4)
    var_6 = bool(not var_5)
    assert var_6 is True
    var_7 = var_0.get(var_4)
    assert var_7 is None
    var_8 = var_0.all
    var_9 = bool(var_0.all == [])
    assert var_9 is True
    var_10 = var_0.codes
    var_11 = bool(var_0.codes == [])
    assert var_11 is True
    var_12 = var_0.codenames
    var_13 = bool(var_0.codenames == [])
    assert var_13 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_currency_constructor_with_valid_parameters. Retrieved 5/9 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 2
    var_4 = module_0.make_quantizer(var_3)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_currency_registry_constructor. Retrieved 5/6 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0
    var_2 = 'USD'
    var_3 = bool('USD' not in var_0)
    assert var_3 is True
    var_4 = 'USD'
    var_5 = var_0.has(var_4)
    var_6 = bool(not var_5)
    assert var_6 is True
    var_7 = var_0.get(var_4)
    assert var_7 is None
    var_8 = var_0.all
    var_9 = bool(var_0.all == [])
    assert var_9 is True
    var_10 = var_0.codes
    var_11 = bool(var_0.codes == [])
    assert var_11 is True
    var_12 = var_0.codenames
    var_13 = bool(var_0.codenames == [])
    assert var_13 is True



# Parsed testcases at query #18
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_currency_constructor_with_valid_inputs. Retrieved 4/16 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 4/16 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = [var_3]
    var_6 = [var_3]
    var_7 = [var_3]

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '1'
    var_4 = [var_3]
    var_5 = [var_3]
    var_6 = [var_3]
    var_7 = [var_3]

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1
    var_3 = -1
    var_4 = -1



# Parsed testcases at query #20
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #21
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True



# Parsed testcases at query #22
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_currency_constructor_creates_immutable_instance. Retrieved 4/18 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.0001'
    var_4 = [var_3]



# Parsed testcases at query #24
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #25
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__codes
    var_2 = bool(var_0._CurrencyRegistry__codes == [])
    assert var_2 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_currency_constructor_with_valid_inputs. Retrieved 5/9 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 3/8 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 3/8 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 2
    var_4 = module_0.make_quantizer(var_3)

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1



