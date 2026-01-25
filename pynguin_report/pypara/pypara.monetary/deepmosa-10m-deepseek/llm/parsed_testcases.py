####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_qty_or_none_returns_qty_when_defined. Retrieved 4/12 statements.
# Partially parsed test_qty_or_none_returns_none_when_undefined. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = [var_1]

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = [var_1]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_lt_with_defined_prices_same_currency. Retrieved 5/16 statements.
# Partially parsed test_lt_with_defined_prices_different_currencies. Retrieved 5/16 statements.
# Partially parsed test_lt_with_undefined_price_and_defined_price. Retrieved 4/12 statements.
# Failed to parse test_lt_with_two_undefined_prices.


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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_positive_defined_money. Retrieved 4/13 statements.
# Failed to parse test_positive_undefined_money.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = [var_1]
    var_7 = [var_3, var_4, var_4]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test___ge___with_same_currency. Retrieved 6/13 statements.
# Partially parsed test___ge___with_different_currency. Retrieved 7/17 statements.
# Partially parsed test___ge___with_equal_quantities. Retrieved 5/12 statements.
# Partially parsed test___ge___with_non_money_object. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = '50.00'
    var_7 = [var_6]

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = 'EUR'
    var_3 = '100.00'
    var_4 = [var_3]
    var_5 = 2023
    var_6 = 1
    var_7 = '50.00'
    var_8 = [var_7]
    var_9 = bool(False)
    assert var_9 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = [var_2]

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_times_with_defined_price. Retrieved 6/16 statements.
# Partially parsed test_times_with_undefined_price. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '2'
    var_7 = [var_6]
    var_8 = '20'
    var_9 = [var_8]
    var_10 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = '2'
    var_1 = [var_0]



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_gt_with_defined_money_and_greater_quantity. Retrieved 5/13 statements.
# Partially parsed test_gt_with_defined_money_and_lesser_quantity. Retrieved 5/13 statements.
# Partially parsed test_gt_with_defined_money_and_equal_quantity. Retrieved 4/12 statements.
# Partially parsed test_gt_with_undefined_money_and_defined_money. Retrieved 4/10 statements.
# Partially parsed test_gt_with_defined_money_and_undefined_money. Retrieved 4/10 statements.
# Failed to parse test_gt_with_undefined_money_and_undefined_money.
# Partially parsed test_gt_with_incompatible_currencies. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '2'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '1'
    var_6 = [var_5]

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
    var_5 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

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
    var_8 = bool(True)
    assert var_8 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_ccy_or_returns_ccy_when_price_is_defined. Retrieved 5/13 statements.
# Partially parsed test_ccy_or_returns_default_when_price_is_undefined. Retrieved 3/9 statements.


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



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_gte_defined_price_greater_than_defined_price_same_currency. Retrieved 5/16 statements.
# Partially parsed test_gte_defined_price_equal_to_defined_price_same_currency. Retrieved 4/15 statements.
# Partially parsed test_gte_defined_price_less_than_defined_price_same_currency. Retrieved 5/16 statements.
# Partially parsed test_gte_undefined_price_greater_than_defined_price. Retrieved 4/12 statements.
# Failed to parse test_gte_undefined_price_greater_than_undefined_price.
# Partially parsed test_gte_defined_price_greater_than_undefined_price. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '2'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '1'
    var_6 = [var_5]

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
    var_5 = '2'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1



# Parsed testcases at query #9
#--------------------------

# Partially parsed test___bool__with_defined_price. Retrieved 4/11 statements.
# Failed to parse test___bool__with_undefined_price.
# Partially parsed test___bool__with_zero_quantity. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '0'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_floor_divide_defined_money. Retrieved 5/16 statements.
# Partially parsed test_floor_divide_undefined_money. Retrieved 1/4 statements.
# Partially parsed test_floor_divide_by_zero. Retrieved 5/15 statements.
# Partially parsed test_floor_divide_negative_divisor. Retrieved 6/17 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2020
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '3'
    var_7 = [var_6]
    var_8 = [var_6]
    var_9 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = '5'
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
    var_6 = '-3'
    var_7 = [var_6]
    var_8 = '-4'
    var_9 = [var_8]
    var_10 = [var_3, var_4, var_4]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_convert_with_valid_currency_and_date. Retrieved 6/18 statements.
# Partially parsed test_convert_with_undefined_price. Retrieved 1/6 statements.
# Partially parsed test_convert_with_invalid_currency. Retrieved 5/15 statements.
# Partially parsed test_convert_without_date. Retrieved 6/17 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 'EUR'
    var_7 = [var_3, var_4, var_4]
    var_8 = '0'
    var_9 = [var_8]
    var_10 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'EUR'

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 'XYZ'
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 'EUR'
    var_7 = '0'
    var_8 = [var_7]
    var_9 = [var_3, var_4, var_4]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_lt_undefined_price. Retrieved 4/10 statements.
# Partially parsed test_lt_defined_prices_with_different_currencies. Retrieved 5/15 statements.
# Partially parsed test_lt_defined_prices_with_same_currency. Retrieved 5/15 statements.
# Failed to parse test_lt_undefined_prices.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

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
    var_8 = bool(True)
    assert var_8 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_qty_or_none_returns_none_for_undefined_price.
# Partially parsed test_qty_or_none_returns_qty_for_defined_price. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = [var_1]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_eq. Retrieved 7/30 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = [var_1]
    var_7 = [var_3, var_4, var_4]
    var_8 = 'EUR'
    var_9 = [var_1]
    var_10 = [var_3, var_4, var_4]
    var_11 = '2'
    var_12 = [var_11]
    var_13 = [var_3, var_4, var_4]
    var_14 = [var_1]
    var_15 = 2
    var_16 = [var_3, var_4, var_15]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test___truediv___with_defined_money. Retrieved 6/17 statements.
# Partially parsed test___truediv___with_undefined_money. Retrieved 1/7 statements.
# Partially parsed test___truediv___with_division_by_zero. Retrieved 5/14 statements.


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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_none_money_constructor. Retrieved 9/71 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = 1
    var_5 = []
    var_6 = []
    var_7 = 2
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = []
    var_13 = []
    var_14 = lambda x: x
    var_15 = lambda x: x
    var_16 = []
    var_17 = []
    var_18 = 'USD'
    var_19 = '1'
    var_20 = [var_19]
    var_21 = [var_19]
    var_22 = [var_19]
    var_23 = '0'
    var_24 = [var_23]
    var_25 = [var_19]
    var_26 = [var_19]
    var_27 = lambda x: x
    var_28 = [var_19]
    var_29 = [var_19]
    var_30 = 'EUR'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_add_defined_money_with_same_currency. Retrieved 7/17 statements.
# Partially parsed test_add_defined_money_with_different_currency. Retrieved 7/17 statements.
# Partially parsed test_add_undefined_money_with_defined_money. Retrieved 5/13 statements.
# Partially parsed test_add_defined_money_with_undefined_money. Retrieved 4/12 statements.
# Failed to parse test_add_undefined_money_with_undefined_money.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '20'
    var_6 = [var_5]
    var_7 = 2
    var_8 = '30'
    var_9 = [var_8]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '10'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = '20'
    var_7 = [var_6]
    var_8 = 2
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool(True)
    assert var_10 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = '20'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = 2
    var_6 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_1]



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_positive_defined_price. Retrieved 4/9 statements.
# Failed to parse test_positive_undefined_price.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_ccy_or_returns_ccy_when_defined. Retrieved 5/13 statements.
# Partially parsed test_ccy_or_returns_default_when_undefined. Retrieved 3/9 statements.


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



# Parsed testcases at query #8
#--------------------------

# Failed to parse test_as_boolean_returns_false_for_undefined_money.
# Partially parsed test_as_boolean_returns_false_for_zero_quantity. Retrieved 4/9 statements.
# Partially parsed test_as_boolean_returns_true_for_defined_non_zero_money. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '0'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_with_qty_updates_quantity_for_defined_money. Retrieved 5/14 statements.
# Partially parsed test_with_qty_returns_self_for_undefined_money. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]
    var_7 = [var_5]

def test_case_0():
    var_0 = '2'
    var_1 = [var_0]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_lt_with_defined_prices_same_currency. Retrieved 5/17 statements.
# Partially parsed test_lt_with_undefined_price. Retrieved 4/13 statements.
# Failed to parse test_lt_with_undefined_prices.
# Partially parsed test_lt_with_different_currencies. Retrieved 5/18 statements.
# Partially parsed test_lt_with_equal_prices. Retrieved 4/16 statements.


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
    var_10 = bool(True)
    assert var_10 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = [var_1]
    var_7 = [var_3, var_4, var_4]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_qty_or_none_defined_price. Retrieved 4/13 statements.
# Partially parsed test_qty_or_none_undefined_price. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = [var_1]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_gt_with_defined_prices. Retrieved 5/12 statements.
# Partially parsed test_gt_with_undefined_price. Retrieved 4/10 statements.
# Failed to parse test_gt_with_both_undefined_prices.
# Partially parsed test_gt_with_different_currencies. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2019
    var_2 = 1
    var_3 = '2'
    var_4 = [var_3]
    var_5 = '1'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = 2019
    var_2 = 1
    var_3 = '1'
    var_4 = [var_3]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2019
    var_3 = 1
    var_4 = '2'
    var_5 = [var_4]
    var_6 = '1'
    var_7 = [var_6]
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_floor_divide_defined_money. Retrieved 5/13 statements.
# Partially parsed test_floor_divide_undefined_money. Retrieved 1/4 statements.
# Partially parsed test_floor_divide_by_zero. Retrieved 5/12 statements.
# Partially parsed test_floor_divide_negative_divisor. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '3'
    var_6 = [var_5]
    var_7 = [var_5]

def test_case_0():
    var_0 = '3'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '0'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '-3'
    var_6 = [var_5]
    var_7 = '-4'
    var_8 = [var_7]



# Parsed testcases at query #14
#--------------------------

# Partially parsed test___add___with_same_currency. Retrieved 4/13 statements.
# Partially parsed test___add___with_different_currencies. Retrieved 4/13 statements.
# Partially parsed test___add___with_undefined_money. Retrieved 2/8 statements.
# Partially parsed test___add___with_different_dates. Retrieved 5/17 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.00'
    var_2 = [var_1]
    var_3 = '20.00'
    var_4 = [var_3]
    var_5 = '30.00'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '10.00'
    var_3 = [var_2]
    var_4 = '20.00'
    var_5 = [var_4]
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.00'
    var_2 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.00'
    var_2 = [var_1]
    var_3 = '20.00'
    var_4 = [var_3]
    var_5 = 1
    var_6 = '30.00'
    var_7 = [var_6]



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_lt_defined_less_than_defined. Retrieved 5/14 statements.
# Partially parsed test_lt_defined_not_less_than_defined. Retrieved 5/14 statements.
# Partially parsed test_lt_undefined_less_than_defined. Retrieved 4/10 statements.
# Partially parsed test_lt_defined_not_less_than_undefined. Retrieved 4/10 statements.
# Failed to parse test_lt_undefined_not_less_than_undefined.
# Partially parsed test_lt_incompatible_currencies. Retrieved 5/15 statements.


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
    var_1 = '2'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '1'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

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
    var_8 = bool(True)
    assert var_8 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_qty_or_zero_with_defined_money. Retrieved 4/11 statements.
# Partially parsed test_qty_or_zero_with_undefined_money. Retrieved 1/4 statements.
# Partially parsed test_qty_or_zero_with_zero_quantity. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_1]

def test_case_0():
    var_0 = '0'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'USD'
    var_1 = '0'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_1]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test___eq__. Retrieved 7/31 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = [var_1]
    var_7 = [var_3, var_4, var_4]
    var_8 = 'EUR'
    var_9 = [var_1]
    var_10 = [var_3, var_4, var_4]
    var_11 = '2'
    var_12 = [var_11]
    var_13 = [var_3, var_4, var_4]
    var_14 = [var_1]
    var_15 = 2
    var_16 = [var_3, var_4, var_15]



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_as_integer_defined. Retrieved 4/11 statements.
# Failed to parse test_as_integer_undefined.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.5'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_with_qty_updates_quantity_correctly. Retrieved 4/13 statements.
# Partially parsed test_with_qty_returns_new_instance. Retrieved 4/10 statements.
# Partially parsed test_with_qty_handles_zero_quantity. Retrieved 4/11 statements.
# Partially parsed test_with_qty_handles_negative_quantity. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100.00'
    var_2 = [var_1]
    var_3 = '2023-10-01'
    var_4 = '200.00'
    var_5 = [var_4]
    var_6 = [var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '100.00'
    var_2 = [var_1]
    var_3 = '2023-10-01'
    var_4 = '200.00'
    var_5 = [var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '100.00'
    var_2 = [var_1]
    var_3 = '2023-10-01'
    var_4 = '0.00'
    var_5 = [var_4]
    var_6 = [var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '100.00'
    var_2 = [var_1]
    var_3 = '2023-10-01'
    var_4 = '-50.00'
    var_5 = [var_4]
    var_6 = [var_4]



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_subtract_defined_money_objects. Retrieved 6/20 statements.
# Partially parsed test_subtract_undefined_money_object. Retrieved 4/13 statements.
# Partially parsed test_subtract_incompatible_currencies. Retrieved 6/19 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '3'
    var_7 = [var_6]
    var_8 = [var_3, var_4, var_4]
    var_9 = '7'
    var_10 = [var_9]
    var_11 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
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
    var_7 = '3'
    var_8 = [var_7]
    var_9 = [var_3, var_4, var_4]
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool(True)
    assert var_11 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_gt_defined_price_greater_than_undefined. Retrieved 4/10 statements.
# Partially parsed test_gt_undefined_price_not_greater_than_defined. Retrieved 4/10 statements.
# Partially parsed test_gt_defined_price_greater_than_defined_with_different_currency. Retrieved 5/15 statements.
# Partially parsed test_gt_defined_price_greater_than_defined_with_same_currency. Retrieved 5/14 statements.
# Failed to parse test_gt_undefined_price_not_greater_than_undefined.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

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
    var_1 = '2'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '1'
    var_6 = [var_5]



# Parsed testcases at query #22
#--------------------------

# Partially parsed test___truediv___with_valid_divisor. Retrieved 6/14 statements.
# Partially parsed test___truediv___with_zero_divisor. Retrieved 5/11 statements.
# Partially parsed test___truediv___with_negative_divisor. Retrieved 6/14 statements.
# Partially parsed test___truediv___with_integer_divisor. Retrieved 6/13 statements.
# Partially parsed test___truediv___with_float_divisor. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.0'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '2.0'
    var_6 = [var_5]
    var_7 = '5.0'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.0'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '0.0'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.0'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '-2.0'
    var_6 = [var_5]
    var_7 = '-5.0'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.0'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = 2
    var_6 = '5.0'
    var_7 = [var_6]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.0'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = 2.0
    var_6 = '5.0'
    var_7 = [var_6]



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_neg_returns_negative_of_defined_price. Retrieved 5/14 statements.
# Failed to parse test_neg_returns_itself_for_undefined_price.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '-1'
    var_7 = [var_6]



# Parsed testcases at query #24
#--------------------------

# Partially parsed test___lt___with_same_currency_and_lesser_qty. Retrieved 5/12 statements.
# Partially parsed test___lt___with_same_currency_and_greater_qty. Retrieved 5/12 statements.
# Partially parsed test___lt___with_same_currency_and_equal_qty. Retrieved 4/11 statements.
# Partially parsed test___lt___with_different_currency. Retrieved 5/15 statements.
# Partially parsed test___lt___with_non_SomePrice_object. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100.00'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '200.00'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '200.00'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '100.00'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '100.00'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = [var_2]
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = '100.00'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1



# Parsed testcases at query #25
#--------------------------

# Partially parsed test___bool__. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_round_defined_price. Retrieved 6/12 statements.
# Partially parsed test_round_undefined_price. Retrieved 1/3 statements.
# Partially parsed test_round_zero_ndigits. Retrieved 6/12 statements.
# Partially parsed test_round_negative_ndigits. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1.2345'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 2
    var_6 = '1.23'
    var_7 = [var_6]

def test_case_0():
    var_0 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = '1.2345'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 0
    var_6 = '1'
    var_7 = [var_6]

def test_case_0():
    var_0 = 'USD'
    var_1 = '123.45'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = -1
    var_6 = '120'
    var_7 = [var_6]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_some_price_add_with_same_currency. Retrieved 7/17 statements.
# Partially parsed test_some_price_add_with_different_currency. Retrieved 7/17 statements.
# Partially parsed test_some_price_add_with_undefined_price. Retrieved 4/10 statements.
# Partially parsed test_some_price_add_with_later_date. Retrieved 6/15 statements.
# Partially parsed test_some_price_add_with_earlier_date. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '5.25'
    var_6 = [var_5]
    var_7 = 2
    var_8 = '15.75'
    var_9 = [var_8]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '10.50'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = '5.25'
    var_7 = [var_6]
    var_8 = 2
    var_9 = bool(False)
    assert var_9 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = 2
    var_6 = '5.25'
    var_7 = [var_6]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '5.25'
    var_6 = [var_5]
    var_7 = 2



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_as_float_defined_money. Retrieved 4/11 statements.
# Failed to parse test_as_float_undefined_money.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1.23'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_convert_same_currency. Retrieved 6/13 statements.
# Partially parsed test_convert_different_currency. Retrieved 10/27 statements.
# Partially parsed test_convert_strict_mode_no_rate. Retrieved 13/24 statements.
# Partially parsed test_convert_non_strict_mode_no_rate. Retrieved 11/21 statements.
# Partially parsed test_convert_with_asof_date. Retrieved 13/30 statements.


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
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = '0.85'
    var_8 = [var_7]
    var_9 = '100.00'
    var_10 = [var_9]
    var_11 = '85.00'
    var_12 = [var_11]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = None
    var_6 = lambda c1, c2, d, s: var_5
    var_7 = '100.00'
    var_8 = [var_7]
    var_9 = 2023
    var_10 = 1
    var_11 = 2023
    var_12 = 1
    var_13 = True
    var_14 = bool(False)
    assert var_14 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = None
    var_6 = lambda c1, c2, d, s: var_5
    var_7 = '100.00'
    var_8 = [var_7]
    var_9 = 2023
    var_10 = 1
    var_11 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = '0.85'
    var_8 = [var_7]
    var_9 = '100.00'
    var_10 = [var_9]
    var_11 = 2022
    var_12 = 12
    var_13 = 31
    var_14 = '85.00'
    var_15 = [var_14]



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_lte_comparison_with_defined_prices. Retrieved 5/22 statements.
# Partially parsed test_lte_comparison_with_undefined_prices. Retrieved 4/14 statements.
# Partially parsed test_lte_comparison_with_incompatible_currencies. Retrieved 5/17 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]
    var_7 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

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
    var_8 = bool(True)
    assert var_8 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_convert_handles_attribute_error_when_fx_rate_service_default_is_not_none. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '100'
    var_4 = [var_3]
    var_5 = []
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_truediv_with_defined_price. Retrieved 7/15 statements.
# Partially parsed test_truediv_with_undefined_price. Retrieved 1/4 statements.
# Partially parsed test_truediv_with_zero_divisor. Retrieved 6/12 statements.
# Partially parsed test_truediv_with_negative_divisor. Retrieved 7/15 statements.
# Partially parsed test_truediv_with_non_decimal_divisor. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '10'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = '2'
    var_7 = [var_6]
    var_8 = '5'
    var_9 = [var_8]

def test_case_0():
    var_0 = '2'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '10'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = '0'
    var_7 = [var_6]

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '10'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = '-2'
    var_7 = [var_6]
    var_8 = '-5'
    var_9 = [var_8]

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '10'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = '5'
    var_7 = [var_6]



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_with_qty_updates_quantity_for_defined_price. Retrieved 5/12 statements.
# Partially parsed test_with_qty_returns_same_for_undefined_price. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]
    var_7 = [var_5]

def test_case_0():
    var_0 = '2'
    var_1 = [var_0]



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_is_equal_with_same_money_objects. Retrieved 4/14 statements.
# Partially parsed test_is_equal_with_different_money_objects. Retrieved 5/15 statements.
# Partially parsed test_is_equal_with_undefined_money. Retrieved 4/11 statements.
# Failed to parse test_is_equal_with_two_undefined_money_objects.


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



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_or_else_returns_itself_if_defined. Retrieved 7/18 statements.
# Partially parsed test_or_else_returns_fallback_if_undefined. Retrieved 5/14 statements.


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

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 'USD'
    var_4 = [var_1]
    var_5 = 2019
    var_6 = 1



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_as_integer_returns_integer_for_defined_money. Retrieved 4/12 statements.
# Partially parsed test_as_integer_raises_exception_for_undefined_money. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '42.99'
    var_2 = [var_1]
    var_3 = 2020
    var_4 = 1
    var_5 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = None
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_qty_or_returns_quantity_when_money_is_defined. Retrieved 6/14 statements.
# Partially parsed test_qty_or_returns_default_when_money_is_undefined. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '0'
    var_6 = [var_5]
    var_7 = '1.00'
    var_8 = [var_7]

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = [var_1]
    var_3 = '0'
    var_4 = [var_3]
    var_5 = [var_3]



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_neg_positive_quantity. Retrieved 6/12 statements.
# Partially parsed test_neg_negative_quantity. Retrieved 6/12 statements.
# Partially parsed test_neg_zero_quantity. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = '-100.00'
    var_7 = [var_6]

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '-50.25'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = '50.25'
    var_7 = [var_6]

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '0.00'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = [var_2]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_qty_map_defined_price. Retrieved 6/18 statements.
# Partially parsed test_qty_map_undefined_price. Retrieved 3/13 statements.


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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_qty_or_zero_returns_zero_for_undefined_money. Retrieved 1/4 statements.
# Partially parsed test_qty_or_zero_returns_qty_for_defined_money. Retrieved 5/12 statements.


def test_case_0():
    var_0 = '0'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '1.00'
    var_6 = [var_5]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_with_qty_updates_quantity_correctly. Retrieved 7/13 statements.
# Partially parsed test_with_qty_preserves_currency_and_date. Retrieved 7/13 statements.
# Partially parsed test_with_qty_quantizes_quantity_correctly. Retrieved 8/15 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 10
    var_6 = 1
    var_7 = '200.00'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 10
    var_6 = 1
    var_7 = '200.00'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 10
    var_6 = 1
    var_7 = '200.555'
    var_8 = [var_7]
    var_9 = '200.56'
    var_10 = [var_9]



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_fmap. Retrieved 8/16 statements.
# Partially parsed test_fmap_returns_self_when_function_returns_none. Retrieved 8/14 statements.
# Partially parsed test_fmap_returns_self_when_function_raises_exception. Retrieved 9/15 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '0.01'
    var_3 = [var_2]
    var_4 = '100.00'
    var_5 = [var_4]
    var_6 = 2023
    var_7 = 1
    var_8 = lambda x: SomeMoney(x.ccy, x.qty * var_1, x.dov)
    var_9 = '200.00'
    var_10 = [var_9]

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '0.01'
    var_3 = [var_2]
    var_4 = '100.00'
    var_5 = [var_4]
    var_6 = 2023
    var_7 = 1
    var_8 = None
    var_9 = lambda x: var_8

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '0.01'
    var_3 = [var_2]
    var_4 = '100.00'
    var_5 = [var_4]
    var_6 = 2023
    var_7 = 1
    var_8 = 0
    var_9 = var_7 / var_8
    var_10 = lambda x: var_9



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_dov_or_returns_dov_when_default_provided. Retrieved 8/14 statements.
# Partially parsed test_dov_or_returns_default_when_dov_is_none. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = 2022
    var_7 = 12
    var_8 = 31

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = None
    var_5 = 2022
    var_6 = 12
    var_7 = 31



# Parsed testcases at query #7
#--------------------------

# Partially parsed test___bool__. Retrieved 4/14 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_abs_defined_price. Retrieved 4/10 statements.
# Partially parsed test_abs_negative_defined_price. Retrieved 5/11 statements.
# Failed to parse test_abs_undefined_price.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.5'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '-10.5'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '10.5'
    var_6 = [var_5]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_lte_with_defined_prices. Retrieved 5/16 statements.
# Partially parsed test_lte_with_undefined_price. Retrieved 4/11 statements.
# Partially parsed test_lte_with_incompatible_currencies. Retrieved 5/17 statements.
# Partially parsed test_lte_with_equal_prices. Retrieved 4/15 statements.
# Partially parsed test_lte_with_greater_price. Retrieved 5/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '20'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '20'
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
    var_6 = [var_1]
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '30'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '20'
    var_6 = [var_5]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_is_equal_with_defined_prices. Retrieved 4/10 statements.
# Partially parsed test_is_equal_with_different_defined_prices. Retrieved 7/16 statements.
# Failed to parse test_is_equal_with_undefined_prices.
# Partially parsed test_is_equal_with_defined_and_undefined_prices. Retrieved 4/10 statements.


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
    var_4 = '2'
    var_5 = [var_4]
    var_6 = 2019
    var_7 = 1
    var_8 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_divide_defined_price. Retrieved 6/14 statements.
# Partially parsed test_divide_by_zero. Retrieved 5/11 statements.
# Partially parsed test_divide_undefined_price. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]
    var_7 = '5'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '0'
    var_6 = [var_5]

def test_case_0():
    var_0 = '2'
    var_1 = [var_0]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test___add__. Retrieved 8/17 statements.
# Partially parsed test___add__incompatible_currency. Retrieved 8/18 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = '200.00'
    var_5 = [var_4]
    var_6 = 2023
    var_7 = 10
    var_8 = 1
    var_9 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = '200.00'
    var_5 = [var_4]
    var_6 = 2023
    var_7 = 10
    var_8 = 1
    var_9 = 2
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test___add___with_undefined_price. Retrieved 3/9 statements.
# Partially parsed test___add___with_same_currency. Retrieved 6/19 statements.
# Partially parsed test___add___with_different_currency. Retrieved 6/16 statements.
# Partially parsed test___add___with_later_date. Retrieved 6/19 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100.00'
    var_2 = [var_1]
    var_3 = '2023-01-01'

def test_case_0():
    var_0 = 'USD'
    var_1 = '100.00'
    var_2 = [var_1]
    var_3 = '2023-01-01'
    var_4 = '50.00'
    var_5 = [var_4]
    var_6 = '2023-01-02'
    var_7 = '150.00'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '100.00'
    var_2 = [var_1]
    var_3 = '2023-01-01'
    var_4 = 'EUR'
    var_5 = '50.00'
    var_6 = [var_5]
    var_7 = '2023-01-02'
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = '100.00'
    var_2 = [var_1]
    var_3 = '2023-01-02'
    var_4 = '50.00'
    var_5 = [var_4]
    var_6 = '2023-01-01'
    var_7 = '150.00'
    var_8 = [var_7]



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_qty_map_with_defined_price. Retrieved 6/19 statements.
# Partially parsed test_qty_map_with_undefined_price. Retrieved 3/14 statements.


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



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_positive_defined_price. Retrieved 4/13 statements.
# Failed to parse test_positive_undefined_price.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = [var_1]
    var_7 = [var_3, var_4, var_4]



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_scalar_add_with_defined_money. Retrieved 6/17 statements.
# Partially parsed test_scalar_add_with_undefined_money. Retrieved 1/6 statements.
# Partially parsed test_scalar_add_with_zero. Retrieved 5/16 statements.
# Partially parsed test_scalar_add_with_negative_value. Retrieved 6/17 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2020
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '5'
    var_7 = [var_6]
    var_8 = '15'
    var_9 = [var_8]
    var_10 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = '5'
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
    var_8 = [var_1]
    var_9 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2020
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '-5'
    var_7 = [var_6]
    var_8 = '5'
    var_9 = [var_8]
    var_10 = [var_3, var_4, var_4]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_dov_or_none_with_defined_money. Retrieved 4/12 statements.
# Partially parsed test_dov_or_none_with_undefined_money. Retrieved 1/4 statements.
# Partially parsed test_dov_or_none_with_undefined_dov. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = None



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_with_qty_returns_same_instance_if_price_is_undefined. Retrieved 1/4 statements.
# Partially parsed test_with_qty_returns_new_instance_with_updated_qty_if_price_is_defined. Retrieved 5/12 statements.


def test_case_0():
    var_0 = '10'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'USD'
    var_1 = 2019
    var_2 = 1
    var_3 = '1'
    var_4 = [var_3]
    var_5 = '10'
    var_6 = [var_5]
    var_7 = [var_5]



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_subtract_same_currency. Retrieved 7/18 statements.
# Partially parsed test_subtract_different_currency. Retrieved 8/18 statements.
# Partially parsed test_subtract_with_undefined_money. Retrieved 6/12 statements.
# Partially parsed test_subtract_with_different_dates. Retrieved 7/18 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 10
    var_6 = 1
    var_7 = '50.00'
    var_8 = [var_7]
    var_9 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = 'EUR'
    var_3 = '100.00'
    var_4 = [var_3]
    var_5 = 2023
    var_6 = 10
    var_7 = 1
    var_8 = '50.00'
    var_9 = [var_8]

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 10
    var_6 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 10
    var_6 = '50.00'
    var_7 = [var_6]
    var_8 = 1
    var_9 = [var_6]



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_subtract_defined_money_with_same_currency. Retrieved 5/18 statements.
# Partially parsed test_subtract_defined_money_with_different_currency_raises_error. Retrieved 6/18 statements.
# Partially parsed test_subtract_undefined_money_with_defined_money_returns_defined. Retrieved 4/12 statements.
# Partially parsed test_subtract_defined_money_with_undefined_money_returns_defined. Retrieved 4/12 statements.
# Failed to parse test_subtract_undefined_money_with_undefined_money_returns_undefined.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2020
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '5'
    var_7 = [var_6]
    var_8 = [var_3, var_4, var_4]
    var_9 = [var_6]
    var_10 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2020
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

def test_case_0():
    var_0 = 'USD'
    var_1 = '5'
    var_2 = [var_1]
    var_3 = 2020
    var_4 = 1
    var_5 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2020
    var_4 = 1
    var_5 = [var_3, var_4, var_4]



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_with_dov_updates_date_for_defined_price. Retrieved 5/15 statements.
# Partially parsed test_with_dov_returns_self_for_undefined_price. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 2020
    var_7 = [var_6, var_4, var_4]
    var_8 = [var_6, var_4, var_4]

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2020
    var_4 = 1
    var_5 = [var_3, var_4, var_4]



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_convert_same_currency. Retrieved 7/16 statements.
# Partially parsed test_convert_different_currency. Retrieved 11/27 statements.
# Partially parsed test_convert_with_strict_set_to_true_and_rate_not_found. Retrieved 13/24 statements.
# Partially parsed test_convert_with_strict_set_to_false_and_rate_not_found. Retrieved 10/20 statements.
# Partially parsed test_convert_with_no_fx_rate_service_set. Retrieved 12/23 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '100.00'
    var_4 = [var_3]
    var_5 = 2023
    var_6 = 10
    var_7 = 1
    var_8 = [var_3]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 2023
    var_8 = 10
    var_9 = 1
    var_10 = '0.85'
    var_11 = [var_10]
    var_12 = '85.00'
    var_13 = [var_12]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 2023
    var_8 = 10
    var_9 = 1
    var_10 = 2023
    var_11 = 10
    var_12 = 1
    var_13 = True
    var_14 = bool(False)
    assert var_14 is True
    var_15 = bool(True)
    assert var_15 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 2023
    var_8 = 10
    var_9 = 1
    var_10 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 2023
    var_8 = 10
    var_9 = 1
    var_10 = 2023
    var_11 = 10
    var_12 = 1
    var_13 = bool(False)
    assert var_13 is True
    var_14 = bool(True)
    assert var_14 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_positive_method_returns_same_money_for_defined_money. Retrieved 4/10 statements.
# Failed to parse test_positive_method_returns_itself_for_undefined_money.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_qty_or_none_returns_qty_when_defined. Retrieved 4/12 statements.
# Partially parsed test_qty_or_none_returns_none_when_undefined. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = [var_1]

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = [var_1]



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_positive_defined_price. Retrieved 4/14 statements.
# Failed to parse test_positive_undefined_price.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = [var_1]
    var_7 = [var_3, var_4, var_4]



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_le_method_with_same_currency. Retrieved 6/12 statements.
# Partially parsed test_le_method_with_different_currency. Retrieved 7/16 statements.
# Partially parsed test_le_method_with_non_price_object. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100.00'
    var_2 = [var_1]
    var_3 = '200.00'
    var_4 = [var_3]
    var_5 = 2023
    var_6 = 10
    var_7 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = '200.00'
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
    var_6 = 'Not a Price'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_qty_or_else_with_defined_price. Retrieved 5/15 statements.
# Partially parsed test_qty_or_else_with_defined_price_returns_qty. Retrieved 6/14 statements.
# Partially parsed test_qty_or_else_with_undefined_price_returns_default. Retrieved 3/9 statements.
# Partially parsed test_qty_or_else_with_undefined_price_returns_default_value. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '42'
    var_7 = [var_6]
    var_8 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = True
    var_7 = lambda : var_6
    var_8 = [var_1]

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = [var_1]
    var_3 = '42'
    var_4 = [var_3]
    var_5 = [var_3]

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = [var_1]
    var_3 = False
    var_4 = lambda : var_3



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_subtract_defined_money_objects. Retrieved 6/16 statements.
# Partially parsed test_subtract_undefined_money_object. Retrieved 4/10 statements.
# Partially parsed test_subtract_incompatible_currencies. Retrieved 6/16 statements.
# Partially parsed test_subtract_scalar_from_defined_money. Retrieved 6/14 statements.
# Partially parsed test_subtract_scalar_from_undefined_money. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '4'
    var_6 = [var_5]
    var_7 = '6'
    var_8 = [var_7]

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
    var_6 = '4'
    var_7 = [var_6]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '4'
    var_6 = [var_5]
    var_7 = '6'
    var_8 = [var_7]

def test_case_0():
    var_0 = '4'
    var_1 = [var_0]



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_gte_defined_price_greater_than_undefined. Retrieved 4/10 statements.
# Partially parsed test_gte_undefined_price_not_greater_than_defined. Retrieved 4/10 statements.
# Failed to parse test_gte_undefined_price_greater_than_undefined.
# Partially parsed test_gte_defined_price_greater_than_defined_same_currency. Retrieved 5/14 statements.
# Partially parsed test_gte_defined_price_not_greater_than_defined_same_currency. Retrieved 5/14 statements.
# Partially parsed test_gte_defined_price_equal_to_defined_same_currency. Retrieved 4/13 statements.
# Partially parsed test_gte_raises_incompatible_currency_error. Retrieved 5/15 statements.


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

def test_case_0():
    var_0 = 'USD'
    var_1 = '20'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '10'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '5'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '10'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = 'EUR'
    var_6 = [var_1]
    var_7 = bool(True)
    assert var_7 is True
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_gt_defined_price_greater_than_undefined. Retrieved 4/12 statements.
# Partially parsed test_gt_undefined_price_not_greater_than_defined. Retrieved 4/12 statements.
# Failed to parse test_gt_undefined_price_not_greater_than_undefined.
# Partially parsed test_gt_defined_price_greater_than_defined_with_same_currency. Retrieved 5/16 statements.
# Partially parsed test_gt_defined_price_not_greater_than_defined_with_same_currency. Retrieved 5/16 statements.
# Partially parsed test_gt_defined_price_equal_to_defined_with_same_currency. Retrieved 4/15 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '2'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '1'
    var_6 = [var_5]

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
    var_5 = [var_1]



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_qty_map_with_defined_money. Retrieved 6/18 statements.
# Partially parsed test_qty_map_with_undefined_money. Retrieved 3/13 statements.


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



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_constructor_with_valid_arguments. Retrieved 5/9 statements.
# Partially parsed test_constructor_with_zero_quantity. Retrieved 5/10 statements.
# Partially parsed test_constructor_with_negative_quantity. Retrieved 5/10 statements.
# Partially parsed test_constructor_with_future_date. Retrieved 6/10 statements.
# Partially parsed test_constructor_with_minimal_currency_decimals. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1

def test_case_0():
    var_0 = 'EUR'
    var_1 = 2
    var_2 = '0.00'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1

def test_case_0():
    var_0 = 'JPY'
    var_1 = 0
    var_2 = '-500'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1

def test_case_0():
    var_0 = 'GBP'
    var_1 = 2
    var_2 = '50.00'
    var_3 = [var_2]
    var_4 = 2050
    var_5 = 12
    var_6 = 31

def test_case_0():
    var_0 = 'BTC'
    var_1 = 8
    var_2 = '0.00000001'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_dov_or_none_with_defined_price. Retrieved 4/12 statements.
# Partially parsed test_dov_or_none_with_undefined_price. Retrieved 1/3 statements.
# Partially parsed test_dov_or_none_with_undefined_dov. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = None



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_multiply_with_defined_price. Retrieved 6/13 statements.
# Partially parsed test_multiply_with_undefined_price. Retrieved 1/4 statements.
# Partially parsed test_multiply_with_zero. Retrieved 5/12 statements.
# Partially parsed test_multiply_with_negative_scalar. Retrieved 6/13 statements.


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
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '0'
    var_6 = [var_5]
    var_7 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '-3'
    var_6 = [var_5]
    var_7 = '-30'
    var_8 = [var_7]



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_add_with_same_currency. Retrieved 7/15 statements.
# Partially parsed test_add_with_different_currency. Retrieved 8/18 statements.
# Partially parsed test_add_with_one_undefined_price. Retrieved 5/11 statements.
# Failed to parse test_add_with_both_undefined_prices.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = '200'
    var_4 = [var_3]
    var_5 = 2023
    var_6 = 10
    var_7 = 1
    var_8 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '100'
    var_3 = [var_2]
    var_4 = '200'
    var_5 = [var_4]
    var_6 = 2023
    var_7 = 10
    var_8 = 1
    var_9 = 2
    var_10 = bool(False)
    assert var_10 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 10
    var_5 = 1



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_convert_with_default_fx_rate_service_not_none. Retrieved 9/19 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '100.00'
    var_4 = [var_3]
    var_5 = 2023
    var_6 = 1
    var_7 = 'EUR'
    var_8 = 'Euro'
    var_9 = 2
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_abs_defined_price. Retrieved 4/11 statements.
# Partially parsed test_abs_negative_price. Retrieved 5/12 statements.
# Partially parsed test_abs_zero_price. Retrieved 4/11 statements.
# Failed to parse test_abs_undefined_price.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.5'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '-10.5'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '10.5'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '0'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_with_dov_defined_money. Retrieved 5/13 statements.
# Partially parsed test_with_dov_undefined_money. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 2020
    var_7 = [var_6, var_4, var_4]

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_with_ccy_updates_ccy_for_defined_price. Retrieved 5/16 statements.
# Partially parsed test_with_ccy_returns_itself_for_undefined_price. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 'EUR'
    var_6 = [var_1]

def test_case_0():
    var_0 = 'EUR'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_qty_or_else_returns_qty_when_defined. Retrieved 6/15 statements.
# Partially parsed test_qty_or_else_returns_default_when_undefined. Retrieved 3/10 statements.
# Partially parsed test_qty_or_else_returns_non_decimal_default_when_undefined. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '42'
    var_6 = [var_5]
    var_7 = '1.00'
    var_8 = [var_7]

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = [var_1]
    var_3 = '42'
    var_4 = [var_3]
    var_5 = [var_3]

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = [var_1]
    var_3 = False
    var_4 = lambda : var_3



# Parsed testcases at query #41
#--------------------------

# Partially parsed test___add__. Retrieved 8/22 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = '200.00'
    var_5 = [var_4]
    var_6 = 2023
    var_7 = 10
    var_8 = 1
    var_9 = 2
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #42
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
    var_9 = '2.00'
    var_10 = [var_9]

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = [var_1]
    var_3 = [var_1]
    var_4 = '42'
    var_5 = [var_4]
    var_6 = [var_4]



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_times_returns_undefined_price_when_price_is_undefined. Retrieved 1/4 statements.
# Partially parsed test_times_returns_money_with_multiplied_quantity. Retrieved 6/15 statements.
# Partially parsed test_times_returns_money_with_negative_quantity_when_multiplier_is_negative. Retrieved 6/15 statements.
# Partially parsed test_times_returns_zero_money_when_multiplied_by_zero. Retrieved 5/14 statements.


def test_case_0():
    var_0 = '10'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2020
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]
    var_7 = '20'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2020
    var_4 = 1
    var_5 = '-1'
    var_6 = [var_5]
    var_7 = '-10'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2020
    var_4 = 1
    var_5 = '0'
    var_6 = [var_5]
    var_7 = [var_5]



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_subtraction_with_defined_prices. Retrieved 6/17 statements.
# Partially parsed test_subtraction_with_undefined_price. Retrieved 4/12 statements.
# Partially parsed test_subtraction_with_incompatible_currencies. Retrieved 6/16 statements.
# Partially parsed test_subtraction_with_undefined_price_as_first_operand. Retrieved 5/13 statements.
# Failed to parse test_subtraction_with_both_undefined_prices.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '4'
    var_6 = [var_5]
    var_7 = '6'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = 'EUR'
    var_6 = '4'
    var_7 = [var_6]
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = '4'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '-4'
    var_6 = [var_5]



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_as_integer_with_defined_money. Retrieved 4/9 statements.
# Failed to parse test_as_integer_with_undefined_money.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.5'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1



