####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = ' '
    var_1 = ''
    var_2 = '4'
    var_3 = '2'
    var_4 = '5'
    var_5 = '34'
    var_6 = '37'
    var_7 = 'INVALID_TYPE'



# Parsed testcases at query #2
#--------------------------


import mimesis.providers.payment as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = '^\\d{4} \\d{4} \\d{4} \\d{4}$'
    var_3 = module_1.match(var_2, var_1)
    var_4 = '4'
    var_5 = module_1.match(var_2, var_1)
    var_6 = '2'
    var_7 = '5'
    var_8 = '^\\d{4} \\d{6} \\d{5}$'
    var_9 = module_1.match(var_8, var_1)
    var_10 = '3'
    var_11 = 'INVALID'
    var_12 = var_0.credit_card_number(var_3)



# Parsed testcases at query #3
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
    var_7 = '34'
    var_8 = '37'
    var_9 = 'InvalidCardType'
    var_10 = var_0.credit_card_number(var_9)



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
    var_7 = '34'
    var_8 = '37'
    var_9 = 'INVALID'
    var_10 = var_0.credit_card_number(var_3)



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
    var_7 = '34'
    var_8 = '37'
    var_9 = 'InvalidCardType'
    var_10 = var_0.credit_card_number(var_9)



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
    var_7 = '34'
    var_8 = '37'
    var_9 = 'InvalidCardType'
    var_10 = var_0.credit_card_number(var_9)



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = ' '
    var_1 = ''
    var_2 = '4'
    var_3 = '2'
    var_4 = '5'
    var_5 = (var_3, var_4)
    var_6 = '34'
    var_7 = '37'
    var_8 = (var_6, var_7)
    var_9 = 'InvalidCardType'



# Parsed testcases at query #8
#--------------------------


import mimesis.providers.payment as module_0

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = len(var_1)
    assert var_2 == 19
    var_3 = '4'
    var_4 = len(var_1)
    assert var_4 == 19
    var_5 = '2'
    var_6 = '5'
    var_7 = len(var_1)
    assert var_7 == 19
    var_8 = '3'
    var_9 = 'INVALID'
    var_10 = var_0.credit_card_number(var_2)



# Parsed testcases at query #9
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
    var_11 = 'INVALID_CARD_TYPE'
    var_12 = var_0.credit_card_number(var_11)



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = '^\\d{4} \\d{4} \\d{4} \\d{4}$'
    var_1 = '4'
    var_2 = '2'
    var_3 = '5'
    var_4 = '^\\d{4} \\d{6} \\d{5}$'
    var_5 = '34'
    var_6 = '37'
    var_7 = 'InvalidCardType'



# Parsed testcases at query #11
#--------------------------


import mimesis.providers.payment as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = '^\\d{4} \\d{4} \\d{4} \\d{4}$'
    var_3 = module_1.match(var_2, var_1)
    var_4 = '4'
    var_5 = module_1.match(var_2, var_1)
    var_6 = '2'
    var_7 = '5'
    var_8 = '^\\d{4} \\d{6} \\d{5}$'
    var_9 = module_1.match(var_8, var_1)
    var_10 = '3'
    var_11 = var_0.credit_card_number(var_2)
    var_12 = var_0.credit_card_number()
    var_13 = ' '
    var_14 = ''
    var_15 = 0
    var_16 = 2
    var_17 = 9



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = '^4\\d{3} \\d{4} \\d{4} \\d{4}$'
    var_1 = ' '
    var_2 = ''
    var_3 = '^(2[2-7]\\d{2}|5[1-5]\\d{2}) \\d{4} \\d{4} \\d{4}$'
    var_4 = '^(34|37) \\d{6} \\d{5}$'
    var_5 = 'InvalidCardType'



# Parsed testcases at query #13
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
    var_11 = 'invalid_type'
    var_12 = var_0.credit_card_number(var_11)



# Parsed testcases at query #14
#--------------------------


import mimesis.providers.payment as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = '^\\d{4} \\d{4} \\d{4} \\d{4}$'
    var_3 = module_1.match(var_2, var_1)
    var_4 = '4'
    var_5 = module_1.match(var_2, var_1)
    var_6 = module_1.match(var_2, var_1)
    var_7 = '2'
    var_8 = '5'
    var_9 = '^\\d{4} \\d{6} \\d{5}$'
    var_10 = module_1.match(var_9, var_1)
    var_11 = '34'
    var_12 = '37'
    var_13 = 'InvalidCardType'
    var_14 = var_0.credit_card_number(var_13)



# Parsed testcases at query #15
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
    var_7 = '34'
    var_8 = '37'
    var_9 = 'INVALID'
    var_10 = var_0.credit_card_number(var_3)



# Parsed testcases at query #16
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



# Parsed testcases at query #17
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
    var_7 = '34'
    var_8 = '37'
    var_9 = 'InvalidCardType'
    var_10 = var_0.credit_card_number(var_9)



# Parsed testcases at query #18
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



# Parsed testcases at query #19
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
    var_7 = '34'
    var_8 = '37'
    var_9 = 'InvalidCardType'
    var_10 = var_0.credit_card_number(var_9)



# Parsed testcases at query #20
#--------------------------


import mimesis.providers.payment as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = ' '
    var_3 = ''
    var_4 = '^\\d{4} \\d{4} \\d{4} \\d{4}$'
    var_5 = module_1.match(var_4, var_1)
    var_6 = '4'
    var_7 = '2'
    var_8 = '5'
    var_9 = '34'
    var_10 = '37'
    var_11 = 'INVALID_TYPE'
    var_12 = var_0.credit_card_number(var_11)



# Parsed testcases at query #21
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
    var_7 = '34'
    var_8 = '37'
    var_9 = 'INVALID'
    var_10 = var_0.credit_card_number(var_3)



# Parsed testcases at query #22
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
    var_7 = '34'
    var_8 = '37'
    var_9 = 'INVALID'
    var_10 = var_0.credit_card_number(var_3)



# Parsed testcases at query #23
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
    var_7 = '34'
    var_8 = '37'
    var_9 = 'INVALID'
    var_10 = var_0.credit_card_number(var_3)
    var_11 = var_0.credit_card_number()
    var_12 = 0
    var_13 = 2
    var_14 = 9



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
    var_7 = '34'
    var_8 = '37'
    var_9 = 'InvalidCardType'
    var_10 = var_0.credit_card_number(var_9)



# Parsed testcases at query #25
#--------------------------


import mimesis.providers.payment as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = '^\\d{4} \\d{4} \\d{4} \\d{4}$'
    var_3 = module_1.match(var_2, var_1)
    var_4 = '4'
    var_5 = module_1.match(var_2, var_1)
    var_6 = '2'
    var_7 = '5'
    var_8 = '^\\d{4} \\d{6} \\d{5}$'
    var_9 = module_1.match(var_8, var_1)
    var_10 = '34'
    var_11 = '37'
    var_12 = 'InvalidCardType'
    var_13 = var_0.credit_card_number(var_12)



# Parsed testcases at query #26
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
    var_7 = '34'
    var_8 = '37'
    var_9 = 'InvalidCardType'
    var_10 = var_0.credit_card_number(var_9)



# Parsed testcases at query #27
#--------------------------


import mimesis.providers.payment as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = '^\\d{4} \\d{4} \\d{4} \\d{4}$'
    var_3 = module_1.match(var_2, var_1)
    var_4 = '4'
    var_5 = module_1.match(var_2, var_1)
    var_6 = '2'
    var_7 = '5'
    var_8 = '^\\d{4} \\d{6} \\d{5}$'
    var_9 = module_1.match(var_8, var_1)
    var_10 = '3'
    var_11 = 'INVALID'
    var_12 = var_0.credit_card_number(var_3)



# Parsed testcases at query #28
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
    var_7 = '34'
    var_8 = '37'
    var_9 = 'InvalidCardType'
    var_10 = var_0.credit_card_number(var_9)



# Parsed testcases at query #29
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
    var_11 = 'INVALID_TYPE'
    var_12 = var_0.credit_card_number(var_11)



# Parsed testcases at query #30
#--------------------------


import mimesis.providers.payment as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = '^\\d{4} \\d{4} \\d{4} \\d{4}$'
    var_3 = module_1.match(var_2, var_1)
    var_4 = '4'
    var_5 = module_1.match(var_2, var_1)
    var_6 = '2'
    var_7 = '5'
    var_8 = '^\\d{4} \\d{6} \\d{5}$'
    var_9 = module_1.match(var_8, var_1)
    var_10 = '3'
    var_11 = 'InvalidCardType'
    var_12 = var_0.credit_card_number(var_11)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import mimesis.providers.payment as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = '^\\d{4} \\d{4} \\d{4} \\d{4}$'
    var_3 = module_1.match(var_2, var_1)
    var_4 = '4'
    var_5 = module_1.match(var_2, var_1)
    var_6 = '2'
    var_7 = '5'
    var_8 = '^\\d{4} \\d{6} \\d{5}$'
    var_9 = module_1.match(var_8, var_1)
    var_10 = '34'
    var_11 = '37'
    var_12 = 'InvalidCardType'
    var_13 = var_0.credit_card_number(var_12)



# Parsed testcases at query #2
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



# Parsed testcases at query #3
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
    var_7 = '34'
    var_8 = '37'
    var_9 = 'InvalidCardType'
    var_10 = var_0.credit_card_number(var_9)



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
    var_7 = '34'
    var_8 = '37'
    var_9 = 'INVALID'
    var_10 = var_0.credit_card_number(var_3)



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
    var_11 = 'InvalidCardType'
    var_12 = var_0.credit_card_number(var_11)



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = '^\\d{4} \\d{4} \\d{4} \\d{4}$'
    var_1 = '4'
    var_2 = '2'
    var_3 = '5'
    var_4 = '^\\d{4} \\d{6} \\d{5}$'
    var_5 = '3'
    var_6 = 'InvalidCardType'



# Parsed testcases at query #8
#--------------------------


import mimesis.providers.payment as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = '^\\d{4} \\d{4} \\d{4} \\d{4}$'
    var_3 = module_1.match(var_2, var_1)
    var_4 = '4'
    var_5 = module_1.match(var_2, var_1)
    var_6 = '2'
    var_7 = '5'
    var_8 = '^\\d{4} \\d{6} \\d{5}$'
    var_9 = module_1.match(var_8, var_1)
    var_10 = '34'
    var_11 = '37'
    var_12 = 'InvalidCardType'
    var_13 = var_0.credit_card_number(var_12)



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = '^\\d{4} \\d{4} \\d{4} \\d{4}$'
    var_1 = '4'
    var_2 = '2'
    var_3 = '5'
    var_4 = '^\\d{4} \\d{6} \\d{5}$'
    var_5 = '3'
    var_6 = 'InvalidCardType'



# Parsed testcases at query #10
#--------------------------


import mimesis.providers.payment as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = '^\\d{4} \\d{4} \\d{4} \\d{4}$'
    var_3 = module_1.match(var_2, var_1)
    var_4 = '4'
    var_5 = module_1.match(var_2, var_1)
    var_6 = '2'
    var_7 = '5'
    var_8 = (var_6, var_7)
    var_9 = '^\\d{4} \\d{6} \\d{5}$'
    var_10 = module_1.match(var_9, var_1)
    var_11 = '34'
    var_12 = '37'
    var_13 = (var_11, var_12)
    var_14 = var_0.credit_card_number(var_2)



# Parsed testcases at query #11
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
    var_7 = '3'
    var_8 = 'InvalidCardType'
    var_9 = var_0.credit_card_number(var_8)
    var_10 = 0
    var_11 = 2
    var_12 = 9



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
    var_5 = '2'
    var_6 = '5'
    var_7 = (var_5, var_6)
    var_8 = '34'
    var_9 = '37'
    var_10 = (var_8, var_9)
    var_11 = 'InvalidCardType'
    var_12 = var_0.credit_card_number(var_11)



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = ' '
    var_1 = ''
    var_2 = '4'
    var_3 = '2'
    var_4 = '5'
    var_5 = (var_3, var_4)
    var_6 = '34'
    var_7 = '37'
    var_8 = (var_6, var_7)
    var_9 = 'UNKNOWN'



# Parsed testcases at query #15
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
    var_7 = '34'
    var_8 = '37'
    var_9 = 'INVALID'
    var_10 = var_0.credit_card_number(var_3)
    var_11 = var_0.credit_card_number()
    var_12 = 0
    var_13 = 2
    var_14 = 9



# Parsed testcases at query #16
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
    var_11 = 'INVALID'
    var_12 = var_0.credit_card_number(var_2)



# Parsed testcases at query #17
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
    var_7 = '34'
    var_8 = '37'
    var_9 = 'INVALID'
    var_10 = var_0.credit_card_number(var_3)
    var_11 = var_0.credit_card_number()
    var_12 = ' '
    var_13 = ''
    var_14 = 0
    var_15 = 2
    var_16 = 9



# Parsed testcases at query #18
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
    var_7 = '34'
    var_8 = '37'
    var_9 = 'InvalidCardType'
    var_10 = var_0.credit_card_number(var_9)
    var_11 = var_0.credit_card_number()
    var_12 = 0
    var_13 = 2
    var_14 = 9



# Parsed testcases at query #19
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
    var_7 = '34'
    var_8 = '37'
    var_9 = 'InvalidCardType'
    var_10 = var_0.credit_card_number(var_9)



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = '^\\d{4} \\d{4} \\d{4} \\d{4}$'
    var_1 = '4'
    var_2 = '2'
    var_3 = '5'
    var_4 = '^\\d{4} \\d{6} \\d{5}$'
    var_5 = '3'
    var_6 = 'INVALID'
    var_7 = ' '
    var_8 = ''
    var_9 = 0
    var_10 = 2
    var_11 = 9



# Parsed testcases at query #21
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
    var_7 = '34'
    var_8 = '37'
    var_9 = 'InvalidCardType'
    var_10 = var_0.credit_card_number(var_9)



# Parsed testcases at query #22
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
    var_7 = '3'
    var_8 = 'InvalidCardType'
    var_9 = var_0.credit_card_number(var_8)



# Parsed testcases at query #23
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
    var_7 = '3'
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
    var_7 = '34'
    var_8 = '37'
    var_9 = 'InvalidCardType'
    var_10 = var_0.credit_card_number(var_9)



# Parsed testcases at query #25
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



# Parsed testcases at query #26
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
    var_7 = '34'
    var_8 = '37'
    var_9 = 'InvalidCardType'
    var_10 = var_0.credit_card_number(var_9)



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
    var_7 = '3'
    var_8 = 'InvalidCardType'
    var_9 = var_0.credit_card_number(var_8)



# Parsed testcases at query #28
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
    var_7 = '3'
    var_8 = 'INVALID'
    var_9 = var_0.credit_card_number(var_2)



# Parsed testcases at query #29
#--------------------------


import mimesis.providers.payment as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = '^\\d{4} \\d{4} \\d{4} \\d{4}$'
    var_3 = module_1.match(var_2, var_1)
    var_4 = '4'
    var_5 = module_1.match(var_2, var_1)
    var_6 = '2'
    var_7 = '5'
    var_8 = '^\\d{4} \\d{6} \\d{5}$'
    var_9 = module_1.match(var_8, var_1)
    var_10 = '34'
    var_11 = '37'
    var_12 = 'InvalidCardType'
    var_13 = var_0.credit_card_number(var_12)



# Parsed testcases at query #30
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
    var_7 = '34'
    var_8 = '37'
    var_9 = 'InvalidCardType'
    var_10 = var_0.credit_card_number(var_9)
    var_11 = var_0.credit_card_number()
    var_12 = 0
    var_13 = 2
    var_14 = 10



