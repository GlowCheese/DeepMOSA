####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_is_equal. Retrieved 10/65 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]
    var_6 = [var_1]
    var_7 = '200'
    var_8 = [var_7]
    var_9 = [var_1]
    var_10 = 'EUR'
    var_11 = [var_1]
    var_12 = [var_1]
    var_13 = [var_1]
    var_14 = 2
    var_15 = [var_1]
    var_16 = [var_1]
    var_17 = 'not a money object'
    var_18 = 100
    var_19 = None



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_dov_or. Retrieved 9/19 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '0.01'
    var_2 = [var_1]
    var_3 = '100.00'
    var_4 = [var_3]
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = [var_5, var_6, var_7]
    var_9 = 2024
    var_10 = 12
    var_11 = 31
    var_12 = [var_9, var_10, var_11]
    var_13 = [var_5, var_6, var_7]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_someprice_int. Retrieved 4/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '123.456'
    var_2 = [var_1]
    var_3 = 2024
    var_4 = 1
    var_5 = [var_3, var_4, var_4]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_money_float_conversion. Retrieved 14/37 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '123.45'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'MonetaryOperationException'
    var_7 = 'EUR'
    var_8 = '-99.99'
    var_9 = [var_8]
    var_10 = 2020
    var_11 = 6
    var_12 = 15
    var_13 = 'GBP'
    var_14 = '0'
    var_15 = [var_14]
    var_16 = 2021
    var_17 = 12
    var_18 = 31



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_price_is_equal. Retrieved 10/45 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]
    var_6 = '200'
    var_7 = [var_6]
    var_8 = 'EUR'
    var_9 = [var_1]
    var_10 = [var_1]
    var_11 = 2
    var_12 = 'not a price'
    var_13 = 100
    var_14 = None



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_ccy_or_none. Retrieved 5/19 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = None



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_price_floordiv. Retrieved 15/58 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '3'
    var_6 = [var_5]
    var_7 = [var_5]
    var_8 = [var_1]
    var_9 = '0'
    var_10 = [var_9]
    var_11 = '5'
    var_12 = [var_11]
    var_13 = 'EUR'
    var_14 = '20'
    var_15 = [var_14]
    var_16 = '-3'
    var_17 = [var_16]
    var_18 = '-7'
    var_19 = [var_18]
    var_20 = 'GBP'
    var_21 = '7'
    var_22 = [var_21]
    var_23 = '2.5'
    var_24 = [var_23]
    var_25 = '2'
    var_26 = [var_25]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_price_negative. Retrieved 8/38 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '-100'
    var_6 = [var_5]
    var_7 = '-50'
    var_8 = [var_7]
    var_9 = '50'
    var_10 = [var_9]
    var_11 = '0'
    var_12 = [var_11]
    var_13 = [var_11]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_with_dov. Retrieved 8/25 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 2020
    var_6 = 6
    var_7 = 15
    var_8 = '1.00'
    var_9 = [var_8]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_price_add_defined_prices_same_currency. Retrieved 7/23 statements.
# Partially parsed test_price_add_defined_with_undefined. Retrieved 4/15 statements.
# Partially parsed test_price_add_undefined_with_defined. Retrieved 4/15 statements.
# Failed to parse test_price_add_two_undefined.
# Partially parsed test_price_add_negative_quantities. Retrieved 7/21 statements.
# Partially parsed test_price_add_decimal_quantities. Retrieved 7/21 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '5'
    var_6 = [var_5]
    var_7 = 2
    var_8 = '15'
    var_9 = [var_8]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '5'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '-3'
    var_6 = [var_5]
    var_7 = 2
    var_8 = '7'
    var_9 = [var_8]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.5'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '5.25'
    var_6 = [var_5]
    var_7 = 2
    var_8 = '15.75'
    var_9 = [var_8]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_price_bool. Retrieved 11/37 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '0'
    var_6 = [var_5]
    var_7 = '-5'
    var_8 = [var_7]
    var_9 = 'EUR'
    var_10 = '100'
    var_11 = [var_10]
    var_12 = 2020
    var_13 = 6
    var_14 = 15



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_price_float. Retrieved 8/39 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '123.45'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '-50.25'
    var_6 = [var_5]
    var_7 = '0'
    var_8 = [var_7]
    var_9 = '999999.99'
    var_10 = [var_9]
    var_11 = '0.001'
    var_12 = [var_11]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_dov_or_none. Retrieved 5/20 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = None



# Parsed testcases at query #14
#--------------------------

# Failed to parse test_noneprice_constructor.




# Parsed testcases at query #15
#--------------------------

# Partially parsed test_price_add. Retrieved 10/41 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '5'
    var_6 = [var_5]
    var_7 = 2
    var_8 = '15'
    var_9 = [var_8]
    var_10 = '20'
    var_11 = [var_10]
    var_12 = 3
    var_13 = 'EUR'
    var_14 = [var_1]
    var_15 = bool(False)
    assert var_15 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_somemoney_constructor. Retrieved 10/25 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.50'
    var_3 = [var_2]
    var_4 = 2024
    var_5 = 1
    var_6 = 15
    var_7 = [var_4, var_5, var_6]
    var_8 = 'EUR'
    var_9 = '50.25'
    var_10 = [var_9]
    var_11 = 6
    var_12 = 30
    var_13 = [var_4, var_11, var_12]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_dimap_with_defined_money. Retrieved 7/16 statements.
# Partially parsed test_dimap_with_undefined_money. Retrieved 5/10 statements.
# Partially parsed test_dimap_applies_function_to_defined_money. Retrieved 8/20 statements.
# Partially parsed test_dimap_uses_combinator_for_undefined_money. Retrieved 4/14 statements.
# Partially parsed test_dimap_with_date_extraction. Retrieved 8/20 statements.
# Partially parsed test_dimap_undefined_money_with_date_combinator. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = lambda x: x.ccy.code
    var_6 = 'EUR'
    var_7 = lambda : var_6

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = [var_1]
    var_3 = lambda x: x.ccy.code
    var_4 = 'EUR'
    var_5 = lambda : var_4

def test_case_0():
    var_0 = 'USD'
    var_1 = '42'
    var_2 = [var_1]
    var_3 = 2020
    var_4 = 5
    var_5 = 15
    var_6 = lambda x: x.qty
    var_7 = '0'
    var_8 = [var_7]
    var_9 = '42.00'
    var_10 = [var_9]

def test_case_0():
    var_0 = None
    var_1 = '100'
    var_2 = [var_1]
    var_3 = '50'
    var_4 = [var_3]
    var_5 = '999'
    var_6 = [var_5]
    var_7 = [var_5]

def test_case_0():
    var_0 = 'EUR'
    var_1 = '5'
    var_2 = [var_1]
    var_3 = 2021
    var_4 = 12
    var_5 = 25
    var_6 = lambda x: x.dov
    var_7 = 2000
    var_8 = 1

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = [var_1]
    var_3 = lambda x: x.dov
    var_4 = 2000
    var_5 = 1



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_someprice_add. Retrieved 15/47 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '0.01'
    var_2 = [var_1]
    var_3 = 'EUR'
    var_4 = [var_1]
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 2024
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = '50.00'
    var_11 = [var_10]
    var_12 = 2
    var_13 = [var_7, var_8, var_12]
    var_14 = [var_10]
    var_15 = [var_7, var_8, var_8]
    var_16 = '150.00'
    var_17 = [var_16]
    var_18 = [var_7, var_8, var_12]
    var_19 = bool(False)
    assert var_19 is True
    var_20 = '25.00'
    var_21 = [var_20]
    var_22 = 3
    var_23 = [var_7, var_8, var_22]
    var_24 = [var_7, var_8, var_22]
    var_25 = '75.00'
    var_26 = [var_25]
    var_27 = 2023
    var_28 = 12
    var_29 = 31
    var_30 = [var_27, var_28, var_29]
    var_31 = [var_7, var_8, var_8]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_money_floordiv. Retrieved 12/45 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '3'
    var_6 = [var_5]
    var_7 = [var_5]
    var_8 = [var_1]
    var_9 = '0'
    var_10 = [var_9]
    var_11 = '5'
    var_12 = [var_11]
    var_13 = [var_1]
    var_14 = '-3'
    var_15 = [var_14]
    var_16 = '-4'
    var_17 = [var_16]
    var_18 = '10.5'
    var_19 = [var_18]
    var_20 = '2.5'
    var_21 = [var_20]
    var_22 = '4'
    var_23 = [var_22]



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_convert_with_valid_rate. Retrieved 12/37 statements.
# Partially parsed test_convert_with_custom_asof_date. Retrieved 14/40 statements.
# Partially parsed test_convert_with_no_rate_non_strict. Retrieved 10/30 statements.
# Partially parsed test_convert_with_no_rate_strict. Retrieved 11/31 statements.
# Partially parsed test_convert_with_no_default_service. Retrieved 10/28 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 2023
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = '0.85'
    var_11 = [var_10]
    var_12 = [var_7, var_8, var_8]
    var_13 = '85.00'
    var_14 = [var_13]
    var_15 = 2023
    var_16 = 1
    var_17 = [var_15, var_16, var_16]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'GBP'
    var_4 = 'British Pound'
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 2023
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = '0.73'
    var_11 = [var_10]
    var_12 = 6
    var_13 = [var_7, var_12, var_8]
    var_14 = 2023
    var_15 = 6
    var_16 = 1
    var_17 = [var_14, var_15, var_16]
    var_18 = '73.00'
    var_19 = [var_18]
    var_20 = [var_14, var_15, var_16]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = '100.00'
    var_7 = [var_6]
    var_8 = 2023
    var_9 = 1
    var_10 = [var_8, var_9, var_9]
    var_11 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'CHF'
    var_4 = 'Swiss Franc'
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 2023
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = False
    var_11 = True
    var_12 = True
    var_13 = bool(var_12)
    assert var_13 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 2023
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = False
    var_11 = True
    var_12 = bool(var_11)
    assert var_12 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_qty_or_none. Retrieved 6/20 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '1.00'
    var_6 = [var_5]
    var_7 = None
    var_8 = [var_1]



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_price_or_else_returns_self_when_defined. Retrieved 7/20 statements.
# Partially parsed test_price_or_else_returns_fallback_when_undefined. Retrieved 4/14 statements.
# Partially parsed test_price_or_else_fallback_not_called_when_defined. Retrieved 7/18 statements.
# Partially parsed test_price_or_else_fallback_called_when_undefined. Retrieved 2/10 statements.


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
    var_5 = 2
    var_6 = []
    var_7 = len(var_6)
    assert var_7 == 0

def test_case_0():
    var_0 = []
    var_1 = len(var_0)
    assert var_1 == 1



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_price_ge_defined_prices_same_currency. Retrieved 5/23 statements.
# Failed to parse test_price_ge_undefined_with_undefined.
# Partially parsed test_price_ge_undefined_with_defined. Retrieved 4/13 statements.
# Partially parsed test_price_ge_defined_with_undefined. Retrieved 4/13 statements.
# Partially parsed test_price_ge_equal_prices. Retrieved 4/17 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '5'
    var_6 = [var_5]
    var_7 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_price_floordiv. Retrieved 11/47 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '3'
    var_6 = [var_5]
    var_7 = [var_5]
    var_8 = [var_1]
    var_9 = '0'
    var_10 = [var_9]
    var_11 = '5'
    var_12 = [var_11]
    var_13 = '-10'
    var_14 = [var_13]
    var_15 = [var_5]
    var_16 = '-4'
    var_17 = [var_16]
    var_18 = '7.5'
    var_19 = [var_18]
    var_20 = '2.5'
    var_21 = [var_20]
    var_22 = [var_5]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_qty_or. Retrieved 11/38 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 0
    var_6 = [var_5]
    var_7 = [var_1]
    var_8 = None
    var_9 = [var_1]
    var_10 = [var_5]
    var_11 = '0'
    var_12 = [var_11]
    var_13 = '5'
    var_14 = [var_13]
    var_15 = 100
    var_16 = [var_15]
    var_17 = [var_13]
    var_18 = [var_1]
    var_19 = 42
    var_20 = [var_19]
    var_21 = '42'
    var_22 = [var_21]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_round_defined_money. Retrieved 6/16 statements.
# Partially parsed test_round_undefined_money. Retrieved 1/3 statements.
# Partially parsed test_round_zero_digits. Retrieved 6/16 statements.
# Partially parsed test_round_negative_quantity. Retrieved 6/16 statements.
# Partially parsed test_round_preserves_currency. Retrieved 5/15 statements.
# Partially parsed test_round_preserves_date. Retrieved 5/14 statements.
# Partially parsed test_round_already_rounded. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1.567'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 2
    var_6 = '1.57'
    var_7 = [var_6]

def test_case_0():
    var_0 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = '1.567'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 0
    var_6 = '2'
    var_7 = [var_6]

def test_case_0():
    var_0 = 'USD'
    var_1 = '-1.567'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 2
    var_6 = '-1.57'
    var_7 = [var_6]

def test_case_0():
    var_0 = 'EUR'
    var_1 = '1.567'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 2

def test_case_0():
    var_0 = 2019
    var_1 = 1
    var_2 = 'USD'
    var_3 = '1.567'
    var_4 = [var_3]
    var_5 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = '1.50'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 2
    var_6 = [var_1]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_abs_defined_positive_price. Retrieved 4/17 statements.
# Partially parsed test_abs_defined_negative_price. Retrieved 5/18 statements.
# Partially parsed test_abs_defined_zero_price. Retrieved 4/17 statements.
# Failed to parse test_abs_undefined_price.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '-10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '10'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '0'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_somemoney_lt. Retrieved 7/22 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = 'EUR'
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '100.00'
    var_7 = [var_6]
    var_8 = '200.00'
    var_9 = [var_8]
    var_10 = [var_6]
    var_11 = [var_6]
    var_12 = bool(False)
    assert var_12 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_dov_or_none. Retrieved 5/20 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = None



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_money_add. Retrieved 9/59 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '5'
    var_6 = [var_5]
    var_7 = '15.00'
    var_8 = [var_7]
    var_9 = [var_1]
    var_10 = [var_1]
    var_11 = [var_1]
    var_12 = '-3'
    var_13 = [var_12]
    var_14 = '7.00'
    var_15 = [var_14]
    var_16 = [var_1]
    var_17 = [var_5]
    var_18 = 15



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_scalar_add_with_defined_price. Retrieved 6/20 statements.
# Partially parsed test_scalar_add_with_undefined_price. Retrieved 1/6 statements.
# Partially parsed test_scalar_add_with_negative_scalar. Retrieved 6/18 statements.
# Partially parsed test_scalar_add_with_zero_scalar. Retrieved 5/17 statements.
# Partially parsed test_scalar_add_with_decimal_scalar. Retrieved 7/21 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '5'
    var_6 = [var_5]
    var_7 = '15'
    var_8 = [var_7]

def test_case_0():
    var_0 = '5'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '-3'
    var_6 = [var_5]
    var_7 = '7'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '0'
    var_6 = [var_5]
    var_7 = [var_1]

def test_case_0():
    var_0 = 'EUR'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2020
    var_4 = 6
    var_5 = 15
    var_6 = '25.50'
    var_7 = [var_6]
    var_8 = '125.50'
    var_9 = [var_8]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_floordiv_with_valid_divisor. Retrieved 9/21 statements.
# Partially parsed test_floordiv_with_zero_divisor. Retrieved 8/18 statements.
# Partially parsed test_floordiv_with_decimal_divisor. Retrieved 9/22 statements.
# Partially parsed test_floordiv_preserves_date_of_valuation. Retrieved 9/19 statements.
# Partially parsed test_floordiv_with_string_numeric. Retrieved 9/20 statements.


def test_case_0():
    var_0 = 'Currency'
    var_1 = ()
    var_2 = 'quantizer'
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = '10'
    var_6 = [var_5]
    var_7 = 2024
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = 3
    var_11 = '3'
    var_12 = [var_11]
    var_13 = [var_7, var_8, var_8]

def test_case_0():
    var_0 = 'Currency'
    var_1 = ()
    var_2 = 'quantizer'
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = '10'
    var_6 = [var_5]
    var_7 = 2024
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = 0

def test_case_0():
    var_0 = 'Currency'
    var_1 = ()
    var_2 = 'quantizer'
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = '10.5'
    var_6 = [var_5]
    var_7 = 2024
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = '2.5'
    var_11 = [var_10]
    var_12 = '4'
    var_13 = [var_12]
    var_14 = [var_7, var_8, var_8]

def test_case_0():
    var_0 = 'Currency'
    var_1 = ()
    var_2 = 'quantizer'
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = 2024
    var_6 = 6
    var_7 = 15
    var_8 = [var_5, var_6, var_7]
    var_9 = '100'
    var_10 = [var_9]
    var_11 = 7

def test_case_0():
    var_0 = 'Currency'
    var_1 = ()
    var_2 = 'quantizer'
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = '20'
    var_6 = [var_5]
    var_7 = 2024
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = '3'
    var_11 = '6'
    var_12 = [var_11]



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_with_qty. Retrieved 18/52 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '200'
    var_6 = [var_5]
    var_7 = [var_5]
    var_8 = '300'
    var_9 = [var_8]
    var_10 = 'EUR'
    var_11 = '50'
    var_12 = [var_11]
    var_13 = 2020
    var_14 = 6
    var_15 = 15
    var_16 = '0'
    var_17 = [var_16]
    var_18 = [var_16]
    var_19 = 'GBP'
    var_20 = '75'
    var_21 = [var_20]
    var_22 = 2021
    var_23 = 3
    var_24 = 20
    var_25 = '-100'
    var_26 = [var_25]
    var_27 = [var_25]



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_floor_divide_defined_price. Retrieved 5/19 statements.
# Partially parsed test_floor_divide_undefined_price. Retrieved 1/6 statements.
# Partially parsed test_floor_divide_by_zero. Retrieved 5/15 statements.
# Partially parsed test_floor_divide_negative_quantity. Retrieved 6/18 statements.
# Partially parsed test_floor_divide_preserves_currency. Retrieved 5/17 statements.


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
    var_1 = '-10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '3'
    var_6 = [var_5]
    var_7 = '-4'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'EUR'
    var_1 = '20'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '4'
    var_6 = [var_5]



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_money_lte. Retrieved 6/31 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]
    var_7 = [var_5]
    var_8 = '3'
    var_9 = [var_8]



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_dov_or_none. Retrieved 5/20 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = None



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_positive. Retrieved 6/35 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '5'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]
    var_6 = '-5'
    var_7 = [var_6]
    var_8 = [var_6]
    var_9 = '0'
    var_10 = [var_9]
    var_11 = [var_9]



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_truediv_basic_division. Retrieved 6/15 statements.
# Partially parsed test_truediv_with_decimal. Retrieved 7/16 statements.
# Partially parsed test_truediv_with_float. Retrieved 7/15 statements.
# Partially parsed test_truediv_by_zero_returns_nomoney. Retrieved 6/13 statements.
# Partially parsed test_truediv_quantizes_result. Retrieved 7/19 statements.
# Partially parsed test_truediv_preserves_date. Retrieved 7/14 statements.
# Partially parsed test_truediv_with_integer. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = 2024
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = '50.00'
    var_8 = [var_7]
    var_9 = [var_4, var_5, var_5]

def test_case_0():
    var_0 = 'EUR'
    var_1 = 2
    var_2 = '75.50'
    var_3 = [var_2]
    var_4 = 2024
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = '2.5'
    var_8 = [var_7]
    var_9 = '30.20'
    var_10 = [var_9]

def test_case_0():
    var_0 = 'GBP'
    var_1 = 2
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = 2024
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = 4.0
    var_8 = '25.00'
    var_9 = [var_8]

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = 2024
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = 0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = 2024
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = 3
    var_8 = '0.01'
    var_9 = [var_8]

def test_case_0():
    var_0 = 'JPY'
    var_1 = 0
    var_2 = 2024
    var_3 = 6
    var_4 = 15
    var_5 = [var_2, var_3, var_4]
    var_6 = '1000'
    var_7 = [var_6]
    var_8 = 5

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '50.00'
    var_3 = [var_2]
    var_4 = 2024
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = 5
    var_8 = '10.00'
    var_9 = [var_8]



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_money_int_conversion. Retrieved 7/31 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '42.75'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '-42.75'
    var_6 = [var_5]
    var_7 = '0'
    var_8 = [var_7]
    var_9 = '100'
    var_10 = [var_9]



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_convert_with_valid_rate. Retrieved 11/35 statements.
# Partially parsed test_convert_with_custom_asof_date. Retrieved 11/35 statements.
# Partially parsed test_convert_no_rate_non_strict. Retrieved 10/30 statements.
# Partially parsed test_convert_no_rate_strict_raises_error. Retrieved 11/32 statements.
# Partially parsed test_convert_no_fx_service_raises_error. Retrieved 10/28 statements.
# Partially parsed test_convert_uses_money_dov_when_asof_not_provided. Retrieved 12/36 statements.


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
    var_9 = [var_7, var_8, var_8]
    var_10 = '85.00'
    var_11 = [var_10]
    var_12 = 2023
    var_13 = 1
    var_14 = [var_12, var_13, var_13]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'GBP'
    var_4 = 'British Pounds'
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 2023
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = 6
    var_11 = 15
    var_12 = [var_7, var_10, var_11]
    var_13 = '80.00'
    var_14 = [var_13]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = '100.00'
    var_7 = [var_6]
    var_8 = 2023
    var_9 = 1
    var_10 = [var_8, var_9, var_9]
    var_11 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'CAD'
    var_4 = 'Canadian Dollars'
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 2023
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = False
    var_11 = True
    var_12 = True
    var_13 = bool(var_12)
    assert var_13 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'AUD'
    var_4 = 'Australian Dollars'
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 2023
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = False
    var_11 = True
    var_12 = bool(var_11)
    assert var_12 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'CHF'
    var_4 = 'Swiss Francs'
    var_5 = 2023
    var_6 = 3
    var_7 = 15
    var_8 = [var_5, var_6, var_7]
    var_9 = '100.00'
    var_10 = [var_9]
    var_11 = []
    var_12 = len(var_11)
    var_13 = bool(var_12 > 0)
    assert var_13 is True
    var_14 = var_11[0]
    var_15 = '92.00'
    var_16 = [var_15]

def test_case_0():
    pass



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_as_float. Retrieved 19/47 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '123.456'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'MonetaryOperationException'
    var_7 = 'EUR'
    var_8 = '100'
    var_9 = [var_8]
    var_10 = 2020
    var_11 = 6
    var_12 = 15
    var_13 = 'GBP'
    var_14 = '-50.75'
    var_15 = [var_14]
    var_16 = 2021
    var_17 = 3
    var_18 = 10
    var_19 = 'JPY'
    var_20 = '0'
    var_21 = [var_20]
    var_22 = 2022
    var_23 = 12
    var_24 = 31



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_ccy_or_none. Retrieved 5/32 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = None
    var_6 = [var_1]
    var_7 = [var_1]



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_money_truediv_defined_money_positive_divisor. Retrieved 6/20 statements.
# Partially parsed test_money_truediv_defined_money_negative_divisor. Retrieved 6/18 statements.
# Partially parsed test_money_truediv_defined_money_zero_divisor. Retrieved 5/15 statements.
# Partially parsed test_money_truediv_undefined_money. Retrieved 1/6 statements.
# Partially parsed test_money_truediv_decimal_result. Retrieved 6/18 statements.
# Partially parsed test_money_truediv_with_one. Retrieved 6/18 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]
    var_7 = '5.00'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '-2'
    var_6 = [var_5]
    var_7 = '-5.00'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '0'
    var_6 = [var_5]

def test_case_0():
    var_0 = '2'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'USD'
    var_1 = '7'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]
    var_7 = '3.50'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '1'
    var_6 = [var_5]
    var_7 = '10.00'
    var_8 = [var_7]



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_multiply_defined_money_positive_scalar. Retrieved 6/21 statements.
# Partially parsed test_multiply_defined_money_negative_scalar. Retrieved 6/17 statements.
# Partially parsed test_multiply_defined_money_zero_scalar. Retrieved 6/17 statements.
# Partially parsed test_multiply_defined_money_decimal_scalar. Retrieved 6/17 statements.
# Partially parsed test_multiply_undefined_money_returns_itself. Retrieved 1/5 statements.
# Partially parsed test_multiply_defined_money_integer_scalar. Retrieved 6/16 statements.
# Partially parsed test_multiply_defined_money_float_scalar. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]
    var_7 = '20.00'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '-3'
    var_6 = [var_5]
    var_7 = '-30.00'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '0'
    var_6 = [var_5]
    var_7 = '0.00'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '1.5'
    var_6 = [var_5]
    var_7 = '15.00'
    var_8 = [var_7]

def test_case_0():
    var_0 = '5'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'USD'
    var_1 = '5'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 4
    var_6 = '20.00'
    var_7 = [var_6]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 2.5
    var_6 = '25.00'
    var_7 = [var_6]



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_price_gt. Retrieved 7/46 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]
    var_6 = '5'
    var_7 = [var_6]
    var_8 = '3'
    var_9 = [var_8]
    var_10 = '8'
    var_11 = [var_10]
    var_12 = [var_6]
    var_13 = [var_6]



