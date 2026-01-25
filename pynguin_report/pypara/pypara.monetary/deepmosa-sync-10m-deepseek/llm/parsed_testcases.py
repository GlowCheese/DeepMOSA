####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_is_equal_returns_true_for_same_money_objects. Retrieved 4/14 statements.
# Partially parsed test_is_equal_returns_false_for_different_money_objects. Retrieved 5/15 statements.
# Partially parsed test_is_equal_returns_false_for_non_money_objects. Retrieved 5/11 statements.
# Failed to parse test_is_equal_returns_true_for_undefined_money_objects.
# Partially parsed test_is_equal_returns_false_when_comparing_defined_with_undefined_money. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 'EUR'
    var_6 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 'not a money object'

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_positive_returns_same_price_if_defined. Retrieved 4/9 statements.
# Failed to parse test_positive_returns_itself_if_undefined.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_ccy_or_returns_ccy_when_defined. Retrieved 5/14 statements.
# Partially parsed test_ccy_or_returns_default_when_undefined. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 'EUR'

def test_case_0():
    var_0 = 'USD'
    var_1 = None
    var_2 = 'EUR'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test___le___with_same_currency_and_lesser_qty. Retrieved 5/12 statements.
# Partially parsed test___le___with_same_currency_and_equal_qty. Retrieved 4/11 statements.
# Partially parsed test___le___with_same_currency_and_greater_qty. Retrieved 5/12 statements.
# Partially parsed test___le___with_different_currency_raises_error. Retrieved 5/15 statements.
# Partially parsed test___le___with_non_SomePrice_object_returns_false. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.0'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '20.0'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.0'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '20.0'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '10.0'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '10.0'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = [var_2]
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.0'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_gt_with_defined_prices. Retrieved 5/17 statements.
# Partially parsed test_gt_with_undefined_price. Retrieved 4/13 statements.
# Partially parsed test_gt_with_incompatible_currencies. Retrieved 5/17 statements.
# Failed to parse test_gt_with_both_undefined_prices.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '50'
    var_7 = [var_6]
    var_8 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 'EUR'
    var_7 = [var_1]
    var_8 = [var_3, var_4, var_4]
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_lt_with_defined_money_objects. Retrieved 5/17 statements.
# Partially parsed test_lt_with_undefined_money_object. Retrieved 4/13 statements.
# Partially parsed test_lt_with_incompatible_currencies. Retrieved 5/17 statements.
# Failed to parse test_lt_with_two_undefined_money_objects.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '2'
    var_7 = [var_6]
    var_8 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 'EUR'
    var_7 = [var_1]
    var_8 = [var_3, var_4, var_4]
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_qty_map_defined_money. Retrieved 6/18 statements.
# Partially parsed test_qty_map_undefined_money. Retrieved 3/13 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = [var_1]
    var_7 = '42'
    var_8 = [var_7]
    var_9 = '2'
    var_10 = [var_9]

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = [var_1]
    var_3 = [var_1]
    var_4 = '42'
    var_5 = [var_4]
    var_6 = [var_4]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_add_defined_prices_with_same_currency. Retrieved 7/20 statements.
# Partially parsed test_add_defined_prices_with_different_currencies. Retrieved 7/19 statements.
# Partially parsed test_add_defined_price_with_undefined_price. Retrieved 4/12 statements.
# Partially parsed test_add_undefined_price_with_defined_price. Retrieved 5/13 statements.
# Failed to parse test_add_two_undefined_prices.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '20'
    var_7 = [var_6]
    var_8 = 2
    var_9 = [var_3, var_4, var_8]
    var_10 = '30'
    var_11 = [var_10]
    var_12 = [var_3, var_4, var_8]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 'EUR'
    var_7 = '20'
    var_8 = [var_7]
    var_9 = 2
    var_10 = [var_3, var_4, var_9]
    var_11 = bool(False)
    assert var_11 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '20'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = 2
    var_6 = [var_3, var_4, var_5]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_convert_same_currency. Retrieved 6/12 statements.
# Partially parsed test_convert_with_valid_rate. Retrieved 10/24 statements.
# Partially parsed test_convert_with_strict_and_no_rate. Retrieved 11/22 statements.
# Partially parsed test_convert_with_non_strict_and_no_rate. Retrieved 11/21 statements.
# Partially parsed test_convert_with_custom_asof_date. Retrieved 10/25 statements.
# Partially parsed test_convert_without_fx_rate_service. Retrieved 8/18 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '100.00'
    var_4 = [var_3]
    var_5 = 2023
    var_6 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 2023
    var_8 = 1
    var_9 = '0.85'
    var_10 = [var_9]
    var_11 = '85.00'
    var_12 = [var_11]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 2023
    var_8 = 1
    var_9 = None
    var_10 = lambda c1, c2, d, strict: var_9
    var_11 = True
    var_12 = bool(False)
    assert var_12 is True
    var_13 = bool(True)
    assert var_13 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 2023
    var_8 = 1
    var_9 = None
    var_10 = lambda c1, c2, d, strict: var_9
    var_11 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 2023
    var_8 = 1
    var_9 = '0.85'
    var_10 = [var_9]
    var_11 = '85.00'
    var_12 = [var_11]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 2023
    var_8 = 1
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool(True)
    assert var_10 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test___ge___with_compatible_currency. Retrieved 6/12 statements.
# Partially parsed test___ge___with_incompatible_currency. Retrieved 7/16 statements.
# Partially parsed test___ge___with_non_somemoney_instance. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100.00'
    var_2 = [var_1]
    var_3 = '50.00'
    var_4 = [var_3]
    var_5 = 2023
    var_6 = 10
    var_7 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = '50.00'
    var_5 = [var_4]
    var_6 = 2023
    var_7 = 10
    var_8 = 1
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool(True)
    assert var_10 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = '100.00'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 10
    var_5 = 1
    var_6 = []



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_constructor_with_valid_arguments. Retrieved 5/9 statements.
# Partially parsed test_constructor_with_zero_quantity. Retrieved 5/9 statements.
# Partially parsed test_constructor_with_negative_quantity. Retrieved 5/9 statements.
# Partially parsed test_constructor_with_non_quantized_quantity. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '0.00'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '-100.00'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.123'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_floor_divide_defined_money. Retrieved 5/16 statements.
# Partially parsed test_floor_divide_undefined_money. Retrieved 1/7 statements.
# Partially parsed test_floor_divide_by_zero. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '3'
    var_7 = [var_6]
    var_8 = [var_6]
    var_9 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = '3'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '0'
    var_7 = [var_6]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_add_defined_prices_with_same_currency. Retrieved 7/19 statements.
# Partially parsed test_add_defined_price_with_undefined_price. Retrieved 4/13 statements.
# Partially parsed test_add_undefined_price_with_defined_price. Retrieved 5/14 statements.
# Failed to parse test_add_undefined_prices.
# Partially parsed test_add_defined_prices_with_different_currencies. Retrieved 7/18 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]
    var_7 = 2
    var_8 = '3'
    var_9 = [var_8]

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '2'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 2
    var_6 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 'EUR'
    var_6 = '2'
    var_7 = [var_6]
    var_8 = 2
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_multiply_defined_price. Retrieved 6/14 statements.
# Partially parsed test_multiply_undefined_price. Retrieved 1/4 statements.
# Partially parsed test_multiply_zero_quantity. Retrieved 5/13 statements.
# Partially parsed test_multiply_negative_quantity. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]
    var_7 = '20'
    var_8 = [var_7]

def test_case_0():
    var_0 = '2'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'USD'
    var_1 = '0'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]
    var_7 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '-5'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '3'
    var_6 = [var_5]
    var_7 = '-15'
    var_8 = [var_7]



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_dov_or_with_defined_price. Retrieved 5/14 statements.
# Partially parsed test_dov_or_with_undefined_price. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 2001
    var_7 = [var_6, var_4, var_4]
    var_8 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = None
    var_1 = 2001
    var_2 = 1
    var_3 = [var_1, var_2, var_2]



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_divide_defined_price. Retrieved 6/17 statements.
# Partially parsed test_divide_undefined_price. Retrieved 1/4 statements.
# Partially parsed test_divide_by_zero. Retrieved 5/14 statements.
# Partially parsed test_divide_with_negative_divisor. Retrieved 6/17 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2020
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '2'
    var_7 = [var_6]
    var_8 = '5'
    var_9 = [var_8]
    var_10 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = '2'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2020
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '0'
    var_7 = [var_6]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2020
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '-2'
    var_7 = [var_6]
    var_8 = '-5'
    var_9 = [var_8]
    var_10 = [var_3, var_4, var_4]



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_truediv_with_defined_price. Retrieved 6/16 statements.
# Partially parsed test_truediv_with_undefined_price. Retrieved 1/4 statements.
# Partially parsed test_truediv_with_zero_division. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '2'
    var_7 = [var_6]
    var_8 = '5'
    var_9 = [var_8]
    var_10 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = '2'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '0'
    var_7 = [var_6]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_lt_defined_money_with_same_currency. Retrieved 5/16 statements.
# Partially parsed test_lt_defined_money_with_different_currency_raises_error. Retrieved 5/16 statements.
# Partially parsed test_lt_undefined_money_with_defined_money. Retrieved 4/12 statements.
# Failed to parse test_lt_undefined_money_with_undefined_money.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 'EUR'
    var_6 = [var_1]
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_lte_method. Retrieved 6/30 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '20'
    var_7 = [var_6]
    var_8 = [var_3, var_4, var_4]
    var_9 = 'EUR'
    var_10 = [var_1]
    var_11 = [var_3, var_4, var_4]
    var_12 = bool(False)
    assert var_12 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test___gt___with_same_currency. Retrieved 5/12 statements.
# Partially parsed test___gt___with_different_currency. Retrieved 6/16 statements.
# Partially parsed test___gt___with_non_price_object. Retrieved 4/8 statements.
# Partially parsed test___gt___with_equal_quantities. Retrieved 4/11 statements.
# Partially parsed test___gt___with_lower_quantity. Retrieved 5/12 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100.00'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '50.00'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = '50.00'
    var_7 = [var_6]
    var_8 = bool(False)
    assert var_8 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = '100.00'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '100.00'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '50.00'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '100.00'
    var_6 = [var_5]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_floor_divide_with_defined_money. Retrieved 5/15 statements.
# Partially parsed test_floor_divide_with_undefined_money. Retrieved 1/4 statements.
# Partially parsed test_floor_divide_by_zero. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '3'
    var_7 = [var_6]
    var_8 = [var_6]

def test_case_0():
    var_0 = '3'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '0'
    var_7 = [var_6]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_scalar_add_defined_money. Retrieved 6/15 statements.
# Partially parsed test_scalar_add_undefined_money. Retrieved 1/4 statements.
# Partially parsed test_scalar_add_with_zero. Retrieved 5/14 statements.
# Partially parsed test_scalar_add_with_negative_value. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '5'
    var_7 = [var_6]
    var_8 = '15'
    var_9 = [var_8]

def test_case_0():
    var_0 = '5'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '0'
    var_7 = [var_6]
    var_8 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '-5'
    var_7 = [var_6]
    var_8 = '5'
    var_9 = [var_8]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test___sub___with_same_currency. Retrieved 6/16 statements.
# Partially parsed test___sub___with_different_currencies. Retrieved 7/17 statements.
# Partially parsed test___sub___with_undefined_money. Retrieved 5/13 statements.
# Partially parsed test___sub___with_different_dates. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.50'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = '50.25'
    var_7 = [var_6]
    var_8 = [var_6]

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = 'EUR'
    var_3 = '100.50'
    var_4 = [var_3]
    var_5 = 2023
    var_6 = 1
    var_7 = '50.25'
    var_8 = [var_7]
    var_9 = bool(False)
    assert var_9 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.50'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = [var_2]

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.50'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = '50.25'
    var_7 = [var_6]
    var_8 = [var_6]



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_gt_method_with_defined_money. Retrieved 5/16 statements.
# Partially parsed test_gt_method_with_undefined_money. Retrieved 4/12 statements.
# Failed to parse test_gt_method_with_both_undefined_money.
# Partially parsed test_gt_method_with_incompatible_currencies. Retrieved 6/18 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '5'
    var_7 = [var_6]
    var_8 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '5'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 'EUR'
    var_7 = '5'
    var_8 = [var_7]
    var_9 = [var_3, var_4, var_4]
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool(True)
    assert var_11 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_gt_with_defined_prices. Retrieved 5/14 statements.
# Partially parsed test_gt_with_undefined_price. Retrieved 4/10 statements.
# Failed to parse test_gt_with_undefined_prices.
# Partially parsed test_gt_with_incompatible_currencies. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '5'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = 'EUR'
    var_6 = '5'
    var_7 = [var_6]
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_or_else_returns_self_when_defined. Retrieved 7/19 statements.
# Partially parsed test_or_else_returns_fallback_when_undefined. Retrieved 4/13 statements.
# Partially parsed test_or_else_returns_fallback_when_undefined_with_custom_fallback. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 'EUR'
    var_7 = '2'
    var_8 = [var_7]
    var_9 = 2
    var_10 = [var_3, var_4, var_9]

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'EUR'
    var_1 = '2'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 2
    var_6 = [var_3, var_4, var_5]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_negative_method. Retrieved 5/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '-10'
    var_6 = [var_5]



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_multiply_defined_money. Retrieved 6/17 statements.
# Partially parsed test_multiply_undefined_money. Retrieved 1/4 statements.
# Partially parsed test_multiply_with_zero. Retrieved 5/16 statements.
# Partially parsed test_multiply_with_negative. Retrieved 6/17 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2020
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '2'
    var_7 = [var_6]
    var_8 = '20'
    var_9 = [var_8]
    var_10 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = '5'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'EUR'
    var_1 = '15'
    var_2 = [var_1]
    var_3 = 2020
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '0'
    var_7 = [var_6]
    var_8 = [var_6]
    var_9 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'GBP'
    var_1 = '8'
    var_2 = [var_1]
    var_3 = 2020
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '-3'
    var_7 = [var_6]
    var_8 = '-24'
    var_9 = [var_8]
    var_10 = [var_3, var_4, var_4]



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_is_equal_returns_true_for_same_money_objects. Retrieved 4/10 statements.
# Partially parsed test_is_equal_returns_false_for_different_money_objects. Retrieved 5/12 statements.
# Partially parsed test_is_equal_returns_false_for_non_money_objects. Retrieved 5/10 statements.
# Failed to parse test_is_equal_returns_true_for_undefined_money_objects.
# Partially parsed test_is_equal_returns_false_for_defined_and_undefined_money_objects. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '1'
    var_3 = [var_2]
    var_4 = 2019
    var_5 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 'Not a money object'

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_neg_defined_money. Retrieved 5/14 statements.
# Failed to parse test_neg_undefined_money.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '-10'
    var_7 = [var_6]



