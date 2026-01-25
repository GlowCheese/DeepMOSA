####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = 'Unit test for method credit_card_number of class Payment.'
    var_1 = module_0.Payment()
    var_2 = ' '
    var_3 = ''
    var_4 = var_1.credit_card_number()
    var_5 = var_1.credit_card_number()
    var_6 = ' '
    var_7 = ''
    var_8 = var_1.credit_card_number()



# Parsed testcases at query #2
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = ' '
    var_3 = ''
    var_4 = '4'
    var_5 = '22'
    var_6 = '23'
    var_7 = '24'
    var_8 = '25'
    var_9 = '26'
    var_10 = '27'
    var_11 = '51'
    var_12 = '52'
    var_13 = '53'
    var_14 = '54'
    var_15 = '55'
    var_16 = (var_5, var_6, var_7, var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_15)
    var_17 = '34'
    var_18 = '37'
    var_19 = (var_17, var_18)
    var_20 = 'InvalidCardType'
    var_21 = var_0.credit_card_number(var_20)



# Parsed testcases at query #3
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = len(var_1)
    assert var_2 == 19
    var_3 = ' '
    var_4 = ''
    var_5 = len(var_1)
    assert var_5 == 19
    var_6 = len(var_1)
    assert var_6 == 17
    var_7 = 'unsupported_type'
    var_8 = var_0.credit_card_number(var_7)



# Parsed testcases at query #4
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = ' '
    var_3 = ''
    var_4 = '4'
    var_5 = '2'
    var_6 = '5'
    var_7 = (var_5, var_6)
    var_8 = '34'
    var_9 = '37'
    var_10 = (var_8, var_9)
    var_11 = 'InvalidType'
    var_12 = var_0.credit_card_number(var_11)



# Parsed testcases at query #5
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = 'Unit test for method credit_card_number of class Payment.'
    var_1 = module_0.Payment()
    var_2 = ' '
    var_3 = ''
    var_4 = 'Unknown'
    var_5 = var_1.credit_card_number(var_4)



# Parsed testcases at query #6
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = ' '
    var_3 = ''
    var_4 = '4'
    var_5 = '2'
    var_6 = '5'
    var_7 = (var_5, var_6)
    var_8 = '34'
    var_9 = '37'
    var_10 = (var_8, var_9)
    var_11 = 'InvalidType'
    var_12 = var_0.credit_card_number(var_11)



# Parsed testcases at query #7
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = ' '
    var_3 = ''
    var_4 = 'InvalidType'
    var_5 = var_0.credit_card_number(var_4)



# Parsed testcases at query #8
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = ' '
    var_2 = ''
    var_3 = '4'
    var_4 = '2'
    var_5 = '5'
    var_6 = (var_4, var_5)
    var_7 = '34'
    var_8 = '37'
    var_9 = (var_7, var_8)



# Parsed testcases at query #9
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = ' '
    var_3 = ''



# Parsed testcases at query #10
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = 'Test the credit_card_number method of the Payment class.'
    var_1 = 12345
    var_2 = module_0.Payment()
    var_3 = var_2.credit_card_number()
    var_4 = len(var_3)
    assert var_4 == 19
    var_5 = len(var_3)
    assert var_5 == 19
    var_6 = len(var_3)
    assert var_6 == 17
    var_7 = 'UnknownType'
    var_8 = var_2.credit_card_number(var_7)



# Parsed testcases at query #11
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = ' '
    var_3 = ''
    var_4 = '4'
    var_5 = '34'
    var_6 = '37'
    var_7 = 'InvalidCardType'
    var_8 = var_0.credit_card_number(var_7)



# Parsed testcases at query #12
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = len(var_1)
    assert var_2 == 19
    var_3 = 'Unsupported_Card_Type'
    var_4 = var_0.credit_card_number(var_3)
    var_5 = 'All tests passed for method credit_card_number of class Payment.'
    var_6 = print(var_5)



# Parsed testcases at query #13
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = 'Unit test for method credit_card_number of class Payment.'
    var_1 = 42
    var_2 = module_0.Payment()
    var_3 = ' '
    var_4 = ''
    var_5 = '4'
    var_6 = '22'
    var_7 = '23'
    var_8 = '24'
    var_9 = '25'
    var_10 = '26'
    var_11 = '27'
    var_12 = '51'
    var_13 = '52'
    var_14 = '53'
    var_15 = '54'
    var_16 = '55'
    var_17 = (var_6, var_7, var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_15, var_16)
    var_18 = '34'
    var_19 = '37'
    var_20 = (var_18, var_19)
    var_21 = var_2.credit_card_number()
    var_22 = 'unsupported_type'
    var_23 = var_2.credit_card_number(var_22)



# Parsed testcases at query #14
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = ' '
    var_3 = ''
    var_4 = '4'
    var_5 = '2'
    var_6 = '5'
    var_7 = (var_5, var_6)
    var_8 = '34'
    var_9 = '37'
    var_10 = (var_8, var_9)
    var_11 = 'InvalidCardType'
    var_12 = var_0.credit_card_number(var_11)



# Parsed testcases at query #15
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = 'Test function for method credit_card_number of class Payment.'
    var_1 = module_0.Payment()
    var_2 = var_1.credit_card_number()
    var_3 = ' '
    var_4 = ''
    var_5 = '4'
    var_6 = '51'
    var_7 = '52'
    var_8 = '53'
    var_9 = '54'
    var_10 = '55'
    var_11 = '2221'
    var_12 = '2222'
    var_13 = '2223'
    var_14 = '2224'
    var_15 = (var_6, var_7, var_8, var_9, var_10, var_11, var_12, var_13, var_14)
    var_16 = '34'
    var_17 = '37'
    var_18 = (var_16, var_17)



# Parsed testcases at query #16
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = 'Unit test for method credit_card_number of class Payment.'
    var_1 = module_0.Payment()
    var_2 = ' '
    var_3 = ''
    var_4 = '4'
    var_5 = '5'
    var_6 = '2'
    var_7 = (var_5, var_6)
    var_8 = '34'
    var_9 = '37'
    var_10 = (var_8, var_9)
    var_11 = var_1.credit_card_number()
    var_12 = 'InvalidCardType'
    var_13 = var_1.credit_card_number(var_12)



# Parsed testcases at query #17
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = ' '
    var_3 = ''
    var_4 = '4'
    var_5 = '22'
    var_6 = '23'
    var_7 = '24'
    var_8 = '25'
    var_9 = '26'
    var_10 = '27'
    var_11 = '51'
    var_12 = '52'
    var_13 = '53'
    var_14 = '54'
    var_15 = '55'
    var_16 = (var_5, var_6, var_7, var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_15)
    var_17 = '34'
    var_18 = '37'
    var_19 = (var_17, var_18)
    var_20 = 'InvalidCardType'
    var_21 = var_0.credit_card_number(var_20)



# Parsed testcases at query #18
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = ' '
    var_2 = ''
    var_3 = var_0.credit_card_number()
    var_4 = ' '
    var_5 = ''
    var_6 = 'invalid_card_type'
    var_7 = var_0.credit_card_number(var_6)



# Parsed testcases at query #19
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Payment()



# Parsed testcases at query #20
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = ' '
    var_3 = ''
    var_4 = 'InvalidCardType'
    var_5 = var_0.credit_card_number(var_4)



# Parsed testcases at query #21
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = ' '
    var_3 = ''
    var_4 = '4'
    var_5 = '22'
    var_6 = '51'
    var_7 = '52'
    var_8 = '53'
    var_9 = '54'
    var_10 = '55'
    var_11 = (var_5, var_6, var_7, var_8, var_9, var_10)
    var_12 = '34'
    var_13 = '37'
    var_14 = (var_12, var_13)
    var_15 = 'InvalidCardType'
    var_16 = var_0.credit_card_number(var_15)



# Parsed testcases at query #22
#--------------------------


import mimesis.providers.payment as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = '4\\d{3} \\d{4} \\d{4} \\d{4}'
    var_2 = 'Visa card number format is incorrect'
    var_3 = '(222[1-9]|22[3-9]\\d|2[3-9]\\d{2}|[3-9]\\d{3}|5[1-5]\\d{2}) \\d{4} \\d{4} \\d{4}'
    var_4 = 'MasterCard card number format is incorrect'
    var_5 = '(34|37) \\d{6} \\d{5}'
    var_6 = 'American Express card number format is incorrect'
    var_7 = var_0.credit_card_number()
    var_8 = module_1.match(var_1, var_7)
    var_9 = 'Default card number format is incorrect'
    var_10 = 'UnsupportedType'
    var_11 = var_0.credit_card_number(var_10)



# Parsed testcases at query #23
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = ' '
    var_2 = ''
    var_3 = '4'
    var_4 = '5'
    var_5 = '2'
    var_6 = '34'
    var_7 = '37'
    var_8 = 'InvalidCardType'
    var_9 = var_0.credit_card_number(var_8)



# Parsed testcases at query #24
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = ' '
    var_3 = ''
    var_4 = '4'
    var_5 = '2'
    var_6 = '5'
    var_7 = (var_5, var_6)
    var_8 = '34'
    var_9 = '37'
    var_10 = (var_8, var_9)
    var_11 = 'InvalidCardType'
    var_12 = var_0.credit_card_number(var_11)



# Parsed testcases at query #25
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = ' '
    var_3 = ''
    var_4 = '4'
    var_5 = '5'
    var_6 = '2'
    var_7 = '34'
    var_8 = '37'
    var_9 = 'UnknownType'
    var_10 = var_0.credit_card_number(var_9)



# Parsed testcases at query #26
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = ' '
    var_3 = ''
    var_4 = '4'
    var_5 = '22'
    var_6 = '51'
    var_7 = '52'
    var_8 = '53'
    var_9 = '54'
    var_10 = '55'
    var_11 = (var_5, var_6, var_7, var_8, var_9, var_10)
    var_12 = '34'
    var_13 = '37'
    var_14 = (var_12, var_13)
    var_15 = 'InvalidCardType'
    var_16 = var_0.credit_card_number(var_15)



# Parsed testcases at query #27
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = ' '
    var_3 = ''
    var_4 = '4'
    var_5 = '2'
    var_6 = '5'
    var_7 = (var_5, var_6)
    var_8 = '34'
    var_9 = '37'
    var_10 = (var_8, var_9)
    var_11 = 'InvalidCardType'
    var_12 = var_0.credit_card_number(var_11)



# Parsed testcases at query #28
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = ' '
    var_2 = ''



# Parsed testcases at query #29
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = ' '
    var_2 = ''
    var_3 = '2221'
    var_4 = '2222'
    var_5 = '2720'
    var_6 = '5100'
    var_7 = '5599'
    var_8 = (var_3, var_4, var_5, var_6, var_7)



# Parsed testcases at query #30
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = ' '
    var_2 = ''
    var_3 = '4'
    var_4 = '22'
    var_5 = '5'
    var_6 = '34'
    var_7 = '37'
    var_8 = 'InvalidCardType'
    var_9 = var_0.credit_card_number(var_8)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = ' '
    var_3 = ''



# Parsed testcases at query #2
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = 'Test method credit_card_number of class Payment.'
    var_1 = module_0.Payment()
    var_2 = var_1.credit_card_number()
    var_3 = ' '
    var_4 = ''
    var_5 = '4'
    var_6 = '5'
    var_7 = '2'
    var_8 = (var_6, var_7)
    var_9 = '34'
    var_10 = '37'
    var_11 = (var_9, var_10)
    var_12 = 'InvalidType'
    var_13 = var_1.credit_card_number(var_12)
    var_14 = 0
    var_15 = 2
    var_16 = 10
    var_17 = 9
    var_18 = var_14 + var_5
    var_19 = 0
    var_20 = 2
    var_21 = 10
    var_22 = 9
    var_23 = var_19 + var_5
    var_24 = 0
    var_25 = 2
    var_26 = 10
    var_27 = 9
    var_28 = var_24 + var_5



# Parsed testcases at query #3
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = '4'
    var_2 = '2'
    var_3 = '5'
    var_4 = '34'
    var_5 = '37'
    var_6 = '^\\d{4} \\d{4} \\d{4} \\d{4}$'
    var_7 = '^\\d{4} \\d{6} \\d{5}$'



# Parsed testcases at query #4
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = ' '
    var_3 = ''
    var_4 = '4'
    var_5 = '5'
    var_6 = '2'
    var_7 = '34'
    var_8 = '37'
    var_9 = 'InvalidCardType'
    var_10 = var_0.credit_card_number(var_9)



# Parsed testcases at query #5
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = ' '
    var_3 = ''
    var_4 = '4'
    var_5 = '2'
    var_6 = '5'
    var_7 = (var_5, var_6)
    var_8 = '34'
    var_9 = '37'
    var_10 = (var_8, var_9)
    var_11 = 'InvalidCardType'
    var_12 = var_0.credit_card_number(var_11)



# Parsed testcases at query #6
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = ' '
    var_2 = ''
    var_3 = '4'
    var_4 = '22'
    var_5 = '5'
    var_6 = '34'
    var_7 = '37'



# Parsed testcases at query #7
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = ' '
    var_3 = ''
    var_4 = '4'
    var_5 = '2'
    var_6 = '5'
    var_7 = (var_5, var_6)
    var_8 = '34'
    var_9 = '37'
    var_10 = (var_8, var_9)
    var_11 = 'InvalidCardType'
    var_12 = var_0.credit_card_number(var_11)



# Parsed testcases at query #8
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = '^4\\d{3} \\d{4} \\d{4} \\d{4}$'
    var_2 = '^(2221|2720|5[1-5]\\d{2}) \\d{4} \\d{4} \\d{4}$'
    var_3 = '^(34|37)\\d{2} \\d{6} \\d{5}$'
    var_4 = 'InvalidType'
    var_5 = var_0.credit_card_number(var_4)



# Parsed testcases at query #9
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = ' '
    var_3 = ''
    var_4 = '4'
    var_5 = '5'
    var_6 = '2'
    var_7 = '34'
    var_8 = '37'



# Parsed testcases at query #10
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = '4\\d{3} \\d{4} \\d{4} \\d{4}'
    var_2 = '(2221 \\d{4} \\d{4} \\d{4})|(2720 \\d{4} \\d{4} \\d{4})|(51\\d{2} \\d{4} \\d{4} \\d{4})|(55\\d{2} \\d{4} \\d{4} \\d{4})'
    var_3 = '(34|37)\\d{2} \\d{6} \\d{5}'
    var_4 = 'InvalidCardType'
    var_5 = var_0.credit_card_number(var_4)



# Parsed testcases at query #11
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = 'Test method credit_card_number of class Payment.'
    var_1 = module_0.Payment()
    var_2 = var_1.credit_card_number()
    var_3 = ' '
    var_4 = ''
    var_5 = '4'
    var_6 = '5'
    var_7 = '2'
    var_8 = (var_5, var_6, var_7)
    var_9 = (var_6, var_7)
    var_10 = '34'
    var_11 = '37'
    var_12 = (var_10, var_11)
    var_13 = 'InvalidCardType'
    var_14 = var_1.credit_card_number(var_13)



# Parsed testcases at query #12
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = ' '
    var_3 = ''
    var_4 = '4'
    var_5 = '2'
    var_6 = '5'
    var_7 = (var_5, var_6)
    var_8 = '34'
    var_9 = '37'
    var_10 = (var_8, var_9)
    var_11 = 'InvalidCardType'
    var_12 = var_0.credit_card_number(var_11)



# Parsed testcases at query #13
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = ' '
    var_3 = ''
    var_4 = '4'
    var_5 = '5'
    var_6 = '2'
    var_7 = '34'
    var_8 = '37'
    var_9 = 'InvalidType'
    var_10 = var_0.credit_card_number(var_9)



# Parsed testcases at query #14
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = ' '
    var_3 = ''
    var_4 = '4'
    var_5 = '2'
    var_6 = '5'
    var_7 = (var_5, var_6)
    var_8 = '34'
    var_9 = '37'
    var_10 = (var_8, var_9)
    var_11 = 'InvalidCardType'
    var_12 = var_0.credit_card_number(var_11)



# Parsed testcases at query #15
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = len(var_1)
    assert var_2 == 19



# Parsed testcases at query #16
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = ' '
    var_3 = ''
    var_4 = '4'
    var_5 = '5'
    var_6 = '2'
    var_7 = '34'
    var_8 = '37'
    var_9 = 'UnsupportedCardType'
    var_10 = var_0.credit_card_number(var_9)



# Parsed testcases at query #17
#--------------------------


import mimesis.providers.payment as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = '^\\d{4} \\d{4} \\d{4} \\d{4}$'
    var_3 = module_1.match(var_2, var_1)
    var_4 = '4'
    var_5 = '2221'
    var_6 = '2720'
    var_7 = '5100'
    var_8 = '5599'
    var_9 = '^\\d{4} \\d{6} \\d{5}$'
    var_10 = '34'
    var_11 = '37'
    var_12 = 999
    var_13 = var_0.credit_card_number(var_3)



# Parsed testcases at query #18
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = ' '
    var_2 = ''
    var_3 = '4'
    var_4 = '222'
    var_5 = '272'
    var_6 = '51'
    var_7 = '55'
    var_8 = '34'
    var_9 = '37'
    var_10 = 'InvalidType'
    var_11 = var_0.credit_card_number(var_10)



# Parsed testcases at query #19
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = ' '
    var_2 = ''
    var_3 = '4'
    var_4 = '22'
    var_5 = '51'
    var_6 = '55'
    var_7 = '34'
    var_8 = '37'
    var_9 = var_0.credit_card_number()
    var_10 = 'InvalidCardType'
    var_11 = var_0.credit_card_number(var_10)



# Parsed testcases at query #20
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()



# Parsed testcases at query #21
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = ' '
    var_3 = ''
    var_4 = '4'
    var_5 = '5'
    var_6 = '2'
    var_7 = (var_5, var_6)
    var_8 = '34'
    var_9 = '37'
    var_10 = (var_8, var_9)
    var_11 = 'InvalidType'
    var_12 = var_0.credit_card_number(var_11)



# Parsed testcases at query #22
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = 'Unit test for method credit_card_number of class Payment.'
    var_1 = module_0.Payment()
    var_2 = ' '
    var_3 = ''
    var_4 = '4'
    var_5 = '5'
    var_6 = '2'
    var_7 = '34'
    var_8 = '37'
    var_9 = var_1.credit_card_number()



# Parsed testcases at query #23
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = 'Unit test for method credit_card_number of class Payment.'
    var_1 = module_0.Payment()
    var_2 = '4'
    var_3 = '5'
    var_4 = '2'
    var_5 = '34'
    var_6 = '37'
    var_7 = var_1.credit_card_number()
    var_8 = len(var_7)
    assert var_8 == 19
    var_9 = 'InvalidType'
    var_10 = var_1.credit_card_number(var_9)



# Parsed testcases at query #24
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = '4'
    var_2 = '4 4'
    var_3 = '2221'
    var_4 = '2720'
    var_5 = '5100'
    var_6 = '5599'
    var_7 = '2221 '
    var_8 = '2720 '
    var_9 = '5100 '
    var_10 = '5599 '
    var_11 = '34'
    var_12 = '37'
    var_13 = '34 '
    var_14 = '37 '
    var_15 = var_0.credit_card_number()
    var_16 = len(var_15)
    assert var_16 == 19



# Parsed testcases at query #25
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = ' '
    var_2 = ''
    var_3 = '4'
    var_4 = '2'
    var_5 = '5'
    var_6 = '34'
    var_7 = '37'



# Parsed testcases at query #26
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = ' '
    var_3 = ''
    var_4 = '4'
    var_5 = '22'
    var_6 = '23'
    var_7 = '24'
    var_8 = '25'
    var_9 = '26'
    var_10 = '27'
    var_11 = '51'
    var_12 = '52'
    var_13 = '53'
    var_14 = '54'
    var_15 = '55'
    var_16 = (var_5, var_6, var_7, var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_15)
    var_17 = '34'
    var_18 = '37'
    var_19 = (var_17, var_18)
    var_20 = 'unsupported_type'
    var_21 = var_0.credit_card_number(var_20)



# Parsed testcases at query #27
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = 'Unit test for method credit_card_number of class Payment.'
    var_1 = module_0.Payment()
    var_2 = ' '
    var_3 = ''
    var_4 = '4'
    var_5 = '5'
    var_6 = '2'
    var_7 = '34'
    var_8 = '37'
    var_9 = 'InvalidCardType'
    var_10 = var_1.credit_card_number(var_9)



# Parsed testcases at query #28
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = 'Test method credit_card_number of class Payment.'
    var_1 = module_0.Payment()
    var_2 = var_1.credit_card_number()
    var_3 = len(var_2)
    assert var_3 == 19
    var_4 = ' '
    var_5 = ''
    var_6 = len(var_2)
    assert var_6 == 19
    var_7 = len(var_2)
    assert var_7 == 17
    var_8 = 'UnsupportedCardType'
    var_9 = var_1.credit_card_number(var_8)



# Parsed testcases at query #29
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = ' '
    var_3 = ''



# Parsed testcases at query #30
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()



# Parsed testcases at query #31
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = 'Test method credit_card_number of class Payment.'
    var_1 = module_0.Payment()
    var_2 = var_1.credit_card_number()
    var_3 = ' '
    var_4 = ''
    var_5 = '4'
    var_6 = '5'
    var_7 = '2'
    var_8 = (var_6, var_7)
    var_9 = '34'
    var_10 = '37'
    var_11 = (var_9, var_10)
    var_12 = 'invalid_card_type'
    var_13 = var_1.credit_card_number(var_12)



# Parsed testcases at query #32
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = len(var_1)
    assert var_2 == 19
    var_3 = 'InvalidCardType'
    var_4 = var_0.credit_card_number(var_3)



# Parsed testcases at query #33
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = ' '
    var_3 = ''
    var_4 = '4'
    var_5 = '5'
    var_6 = '2'
    var_7 = (var_5, var_6)
    var_8 = '34'
    var_9 = '37'
    var_10 = (var_8, var_9)
    var_11 = 'InvalidType'
    var_12 = var_0.credit_card_number(var_11)



# Parsed testcases at query #34
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = ' '
    var_3 = ''
    var_4 = '4'
    var_5 = '2221'
    var_6 = '2720'
    var_7 = '5100'
    var_8 = '5599'
    var_9 = (var_5, var_6, var_7, var_8)
    var_10 = '34'
    var_11 = '37'
    var_12 = (var_10, var_11)
    var_13 = 'InvalidType'
    var_14 = var_0.credit_card_number(var_13)



# Parsed testcases at query #35
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = ' '
    var_2 = ''
    var_3 = '4'
    var_4 = '5'
    var_5 = '2'
    var_6 = '34'
    var_7 = '37'



