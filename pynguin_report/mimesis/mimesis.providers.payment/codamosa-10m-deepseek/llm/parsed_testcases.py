####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import mimesis.providers.payment as module_0


def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = len(var_1)
    assert var_2 == 19
    var_3 = len(var_1)
    assert var_3 == 19
    var_4 = len(var_1)
    assert var_4 == 17
    var_5 = 'invalid'
    var_6 = var_0.credit_card_number(var_5)



# Parsed testcases at query #2
#--------------------------



def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = len(var_1)
    assert var_2 == 19
    var_3 = len(var_1)
    assert var_3 == 19
    var_4 = len(var_1)
    assert var_4 == 19
    var_5 = len(var_1)
    assert var_5 == 17
    var_6 = 'InvalidCardType'
    var_7 = var_0.credit_card_number(var_6)



# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = len(var_1)
    assert var_2 == 19
    var_3 = module_0.Payment()
    var_4 = len(var_1)
    assert var_4 == 19
    var_5 = module_0.Payment()
    var_6 = len(var_1)
    assert var_6 == 17
    var_7 = module_0.Payment()
    var_8 = 'Invalid'
    var_9 = var_7.credit_card_number(var_8)
    var_10 = module_0.Payment()
    var_11 = None
    var_12 = var_10.credit_card_number(var_11)
    var_13 = len(var_12)
    assert var_13 == 19
    var_14 = module_0.Payment()
    var_15 = len(var_12)
    assert var_15 == 19
    var_16 = module_0.Payment()
    var_17 = len(var_12)
    assert var_17 == 19
    var_18 = module_0.Payment()
    var_19 = len(var_12)
    assert var_19 == 17
    var_20 = module_0.Payment()
    var_21 = len(var_12)
    assert var_21 == 19
    var_22 = module_0.Payment()
    var_23 = len(var_12)
    assert var_23 == 19
    var_24 = module_0.Payment()
    var_25 = len(var_12)
    assert var_25 == 17
    var_26 = module_0.Payment()
    var_27 = len(var_12)
    assert var_27 == 19
    var_28 = module_0.Payment()
    var_29 = len(var_12)
    assert var_29 == 19
    var_30 = module_0.Payment()
    var_31 = len(var_12)
    assert var_31 == 17
    var_32 = module_0.Payment()
    var_33 = len(var_12)
    assert var_33 == 19
    var_34 = module_0.Payment()
    var_35 = len(var_12)
    assert var_35 == 19
    var_36 = module_0.Payment()
    var_37 = len(var_12)
    assert var_37 == 17
    var_38 = module_0.Payment()
    var_39 = len(var_12)
    assert var_39 == 19
    var_40 = module_0.Payment()
    var_41 = len(var_12)
    assert var_41 == 19
    var_42 = module_0.Payment()
    var_43 = len(var_12)
    assert var_43 == 17
    var_44 = module_0.Payment()
    var_45 = len(var_12)
    assert var_45 == 19
    var_46 = module_0.Payment()
    var_47 = len(var_12)
    assert var_47 == 19
    var_48 = module_0.Payment()
    var_49 = len(var_12)
    assert var_49 == 17
    var_50 = module_0.Payment()
    var_51 = len(var_12)
    assert var_51 == 19
    var_52 = module_0.Payment()
    var_53 = len(var_12)
    assert var_53 == 19
    var_54 = module_0.Payment()
    var_55 = len(var_12)
    assert var_55 == 17
    var_56 = module_0.Payment()
    var_57 = len(var_12)
    assert var_57 == 19



# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = len(var_1)
    assert var_2 == 19
    var_3 = ' '
    var_4 = module_0.Payment()
    var_5 = len(var_1)
    assert var_5 == 19
    var_6 = module_0.Payment()
    var_7 = len(var_1)
    assert var_7 == 19
    var_8 = module_0.Payment()
    var_9 = len(var_1)
    assert var_9 == 17
    var_10 = module_0.Payment()
    var_11 = 'invalid_card_type'
    var_12 = var_10.credit_card_number(var_11)
    var_13 = 12345
    var_14 = module_0.Payment()
    var_15 = module_0.Payment()
    var_16 = module_0.Payment()
    var_17 = module_0.Payment()
    var_18 = module_0.Payment()
    var_19 = module_0.Payment()
    var_20 = len(var_1)
    assert var_20 == 19
    var_21 = len(var_1)
    assert var_21 == 19
    var_22 = len(var_1)
    assert var_22 == 17
    var_23 = 'invalid_locale'
    var_24 = module_0.Payment()
    var_25 = len(var_1)
    assert var_25 == 19
    var_26 = module_0.Payment()
    var_27 = len(var_1)
    assert var_27 == 19
    var_28 = module_0.Payment()
    var_29 = len(var_1)
    assert var_29 == 17
    var_30 = module_0.Payment()
    var_31 = len(var_1)
    assert var_31 == 19
    var_32 = module_0.Payment()
    var_33 = len(var_1)
    assert var_33 == 19
    var_34 = module_0.Payment()
    var_35 = len(var_1)
    assert var_35 == 17



# Parsed testcases at query #5
#--------------------------



def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = len(var_1)
    assert var_2 == 19
    var_3 = len(var_1)
    assert var_3 == 19
    var_4 = len(var_1)
    assert var_4 == 17
    var_5 = 'invalid'
    var_6 = var_0.credit_card_number(var_5)



# Parsed testcases at query #6
#--------------------------



def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = len(var_1)
    assert var_2 == 19
    var_3 = module_0.Payment()
    var_4 = len(var_1)
    assert var_4 == 19
    var_5 = module_0.Payment()
    var_6 = len(var_1)
    assert var_6 == 19
    var_7 = module_0.Payment()
    var_8 = len(var_1)
    assert var_8 == 17
    var_9 = module_0.Payment()
    var_10 = 'Invalid'
    var_11 = var_9.credit_card_number(var_10)
    var_12 = module_0.Payment()
    var_13 = ' '
    var_14 = ''
    var_15 = module_0.Payment()
    var_16 = module_0.Payment()
    var_17 = module_0.Payment()
    var_18 = -1
    var_19 = module_0.Payment()
    var_20 = -1
    var_21 = module_0.Payment()
    var_22 = -1



# Parsed testcases at query #7
#--------------------------



def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = len(var_1)
    assert var_2 == 19
    var_3 = '4'
    var_4 = len(var_1)
    assert var_4 == 19
    var_5 = '5'
    var_6 = '2'
    var_7 = len(var_1)
    assert var_7 == 17
    var_8 = '34'
    var_9 = '37'
    var_10 = 'InvalidCardType'
    var_11 = var_0.credit_card_number(var_10)
    var_12 = None
    var_13 = var_0.credit_card_number(var_12)
    var_14 = len(var_13)
    assert var_14 == 19
    var_15 = '4'
    var_16 = '5'
    var_17 = '2'
    var_18 = '34'
    var_19 = '37'
    var_20 = set()
    var_21 = var_0.credit_card_number()
    var_22 = len(var_20)
    var_23 = 12345
    var_24 = module_0.Payment()
    var_25 = module_0.Payment()
    var_26 = var_24.credit_card_number()
    var_27 = var_25.credit_card_number()
    var_28 = module_0.Payment()
    var_29 = 67890
    var_30 = module_0.Payment()
    var_31 = var_28.credit_card_number()
    var_32 = var_30.credit_card_number()
    var_33 = '4'
    var_34 = '5'
    var_35 = '2'
    var_36 = '34'
    var_37 = '37'
    var_38 = 'All tests passed!'
    var_39 = print(var_38)



# Parsed testcases at query #8
#--------------------------



def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = len(var_1)
    assert var_2 == 19
    var_3 = module_0.Payment()
    var_4 = len(var_1)
    assert var_4 == 19
    var_5 = module_0.Payment()
    var_6 = len(var_1)
    assert var_6 == 19
    var_7 = module_0.Payment()
    var_8 = len(var_1)
    assert var_8 == 17
    var_9 = module_0.Payment()
    var_10 = 'InvalidCardType'
    var_11 = var_9.credit_card_number(var_10)



# Parsed testcases at query #9
#--------------------------



def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = len(var_1)
    assert var_2 == 19
    var_3 = '4'
    var_4 = len(var_1)
    assert var_4 == 19
    var_5 = '5'
    var_6 = '2'
    var_7 = len(var_1)
    assert var_7 == 17
    var_8 = '34'
    var_9 = '37'
    var_10 = 'Invalid'
    var_11 = var_0.credit_card_number(var_10)
    var_12 = None
    var_13 = var_0.credit_card_number(var_12)
    var_14 = len(var_13)
    assert var_14 == 19
    var_15 = var_0.credit_card_number()
    var_16 = len(var_15)
    assert var_16 == 19
    var_17 = 42
    var_18 = module_0.Payment()
    var_19 = var_18.credit_card_number()
    assert var_19 == '4455 5299 1152 2450'
    assert var_19 == '2720 5299 1152 2450'
    assert var_19 == '3411 111111 11111'
    var_20 = module_0.Payment()
    var_21 = module_0.Payment()
    var_22 = module_0.Payment()
    var_23 = var_22.credit_card_number(var_12)
    assert var_23 == '4455 5299 1152 2450'
    var_24 = module_0.Payment()
    var_25 = var_24.credit_card_number()
    assert var_25 == '4455 5299 1152 2450'
    var_26 = 43
    var_27 = module_0.Payment()
    var_28 = var_27.credit_card_number()
    assert var_28 == '4455 5299 1152 2450'
    assert var_28 == '2720 5299 1152 2450'
    assert var_28 == '3411 111111 11111'
    var_29 = module_0.Payment()
    var_30 = module_0.Payment()
    var_31 = module_0.Payment()
    var_32 = var_31.credit_card_number(var_12)
    assert var_32 == '4455 5299 1152 2450'
    var_33 = module_0.Payment()
    var_34 = var_33.credit_card_number()
    assert var_34 == '4455 5299 1152 2450'
    var_35 = 44
    var_36 = module_0.Payment()
    var_37 = var_36.credit_card_number()
    assert var_37 == '4455 5299 1152 2450'
    assert var_37 == '2720 5299 1152 2450'
    assert var_37 == '3411 111111 11111'
    var_38 = module_0.Payment()
    var_39 = module_0.Payment()
    var_40 = module_0.Payment()
    var_41 = var_40.credit_card_number(var_12)
    assert var_41 == '4455 5299 1152 2450'
    var_42 = module_0.Payment()
    var_43 = var_42.credit_card_number()
    assert var_43 == '4455 5299 1152 2450'
    var_44 = 45
    var_45 = module_0.Payment()
    var_46 = var_45.credit_card_number()
    assert var_46 == '4455 5299 1152 2450'
    assert var_46 == '2720 5299 1152 2450'
    assert var_46 == '3411 111111 11111'
    var_47 = module_0.Payment()
    var_48 = module_0.Payment()
    var_49 = module_0.Payment()
    var_50 = var_49.credit_card_number(var_12)
    assert var_50 == '4455 5299 1152 2450'
    var_51 = module_0.Payment()
    var_52 = var_51.credit_card_number()
    assert var_52 == '4455 5299 1152 2450'
    var_53 = 46
    var_54 = module_0.Payment()



# Parsed testcases at query #10
#--------------------------



def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = len(var_1)
    assert var_2 == 19
    var_3 = len(var_1)
    assert var_3 == 19
    var_4 = len(var_1)
    assert var_4 == 19
    var_5 = len(var_1)
    assert var_5 == 17
    var_6 = 'Invalid'
    var_7 = var_0.credit_card_number(var_6)



# Parsed testcases at query #11
#--------------------------



def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = len(var_1)
    assert var_2 == 19
    var_3 = len(var_1)
    assert var_3 == 19
    var_4 = '4'
    var_5 = len(var_1)
    assert var_5 == 19
    var_6 = '5'
    var_7 = '2'
    var_8 = len(var_1)
    assert var_8 == 17
    var_9 = '34'
    var_10 = '37'
    var_11 = 'InvalidCardType'
    var_12 = var_0.credit_card_number(var_11)



# Parsed testcases at query #12
#--------------------------




# Parsed testcases at query #13
#--------------------------



def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = len(var_1)
    assert var_2 == 19
    var_3 = len(var_1)
    assert var_3 == 19
    var_4 = len(var_1)
    assert var_4 == 19
    var_5 = len(var_1)
    assert var_5 == 17
    var_6 = 'invalid'
    var_7 = var_0.credit_card_number(var_6)



# Parsed testcases at query #14
#--------------------------



def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = len(var_1)
    assert var_2 == 19
    var_3 = ' '
    var_4 = len(var_1)
    assert var_4 == 19
    var_5 = len(var_1)
    assert var_5 == 19
    var_6 = len(var_1)
    assert var_6 == 17
    var_7 = 'invalid'
    var_8 = var_0.credit_card_number(var_7)



# Parsed testcases at query #15
#--------------------------



def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = len(var_1)
    assert var_2 == 19
    var_3 = module_0.Payment()
    var_4 = len(var_1)
    assert var_4 == 19
    var_5 = module_0.Payment()
    var_6 = len(var_1)
    assert var_6 == 19
    var_7 = module_0.Payment()
    var_8 = len(var_1)
    assert var_8 == 17
    var_9 = module_0.Payment()
    var_10 = 'Invalid'
    var_11 = var_9.credit_card_number(var_10)



# Parsed testcases at query #16
#--------------------------



def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = len(var_1)
    assert var_2 == 19
    var_3 = ' '
    var_4 = ''
    var_5 = module_0.Payment()
    var_6 = len(var_1)
    assert var_6 == 19
    var_7 = '4'
    var_8 = module_0.Payment()
    var_9 = len(var_1)
    assert var_9 == 19
    var_10 = '2'
    var_11 = '5'
    var_12 = module_0.Payment()
    var_13 = len(var_1)
    assert var_13 == 17
    var_14 = '34'
    var_15 = '37'
    var_16 = module_0.Payment()
    var_17 = 'unsupported'
    var_18 = var_16.credit_card_number(var_17)



# Parsed testcases at query #17
#--------------------------


import builtins as module_1


def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = len(var_1)
    assert var_2 == 19
    var_3 = len(var_1)
    assert var_3 == 19
    var_4 = len(var_1)
    assert var_4 == 19
    var_5 = len(var_1)
    assert var_5 == 17
    var_6 = 'invalid'
    var_7 = var_0.credit_card_number(var_6)
    var_8 = None
    var_9 = var_0.credit_card_number(var_8)
    var_10 = len(var_9)
    assert var_10 == 19
    var_11 = ''
    var_12 = var_0.credit_card_number(var_11)
    var_13 = 123
    var_14 = var_0.credit_card_number(var_13)
    var_15 = True
    var_16 = var_0.credit_card_number(var_15)
    var_17 = []
    var_18 = var_0.credit_card_number(var_17)
    var_19 = {}
    var_20 = var_0.credit_card_number(var_19)
    var_21 = ()
    var_22 = var_0.credit_card_number(var_21)
    var_23 = set()
    var_24 = var_0.credit_card_number(var_23)
    var_25 = frozenset()
    var_26 = var_0.credit_card_number(var_25)
    var_27 = b''
    var_28 = var_0.credit_card_number(var_27)
    var_29 = bytearray()
    var_30 = var_0.credit_card_number(var_29)
    var_31 = b''
    var_32 = memoryview(var_31)
    var_33 = var_0.credit_card_number(var_32)
    var_34 = 1
    var_35 = 2
    var_36 = complex(var_34, var_35)
    var_37 = var_0.credit_card_number(var_36)
    var_38 = 10
    var_39 = range(var_38)
    var_40 = var_0.credit_card_number(var_39)
    var_41 = 10
    var_42 = slice(var_41)
    var_43 = var_0.credit_card_number(var_42)
    var_44 = module_1.object()
    var_45 = var_0.credit_card_number(var_44)
    var_46 = lambda x: x
    var_47 = var_0.credit_card_number(var_46)
    var_48 = var_0.credit_card_number(var_0)
    var_49 = 10
    var_50 = range(var_49)
    var_51 = var_0.credit_card_number(var_43)
    var_52 = var_0.credit_card_number(var_49)
    var_53 = var_0.credit_card_number(var_49)
    var_54 = var_0.credit_card_number(var_49)
    var_55 = var_0.credit_card_number(var_49)
    var_56 = var_0.credit_card_number(var_49)
    var_57 = var_0.credit_card_number(var_49)
    var_58 = var_0.credit_card_number(var_49)
    var_59 = property()
    var_60 = var_0.credit_card_number(var_59)
    var_61 = None
    var_62 = lambda : var_61
    var_63 = staticmethod(var_62)
    var_64 = var_0.credit_card_number(var_63)
    var_65 = None
    var_66 = lambda cls: var_65
    var_67 = classmethod(var_66)
    var_68 = var_0.credit_card_number(var_67)
    var_69 = var_0.credit_card_number(var_65)
    var_70 = 1
    var_71 = var_0.credit_card_number(var_65)



# Parsed testcases at query #18
#--------------------------



def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = len(var_1)
    assert var_2 == 19
    var_3 = ' '
    var_4 = len(var_1)
    assert var_4 == 19
    var_5 = len(var_1)
    assert var_5 == 19
    var_6 = len(var_1)
    assert var_6 == 17
    var_7 = 'invalid'
    var_8 = var_0.credit_card_number(var_7)



# Parsed testcases at query #19
#--------------------------



def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = len(var_1)
    assert var_2 == 19
    var_3 = '4'
    var_4 = len(var_1)
    assert var_4 == 19
    var_5 = '5'
    var_6 = '2'
    var_7 = len(var_1)
    assert var_7 == 17
    var_8 = '34'
    var_9 = '37'
    var_10 = 'InvalidCardType'
    var_11 = var_0.credit_card_number(var_10)
    var_12 = None
    var_13 = var_0.credit_card_number(var_12)
    var_14 = len(var_13)
    assert var_14 == 19
    var_15 = var_0.credit_card_number()
    var_16 = len(var_15)
    assert var_16 == 19
    var_17 = 42
    var_18 = module_0.Payment()
    var_19 = module_0.Payment()
    var_20 = var_18.credit_card_number()
    var_21 = var_19.credit_card_number()
    var_22 = module_0.Payment()
    var_23 = 43
    var_24 = module_0.Payment()
    var_25 = var_22.credit_card_number()
    var_26 = var_24.credit_card_number()
    var_27 = 10
    var_28 = range(var_27)
    var_29 = [payment.credit_card_number() for _ in var_28]
    var_30 = set(var_29)
    var_31 = len(var_30)
    assert var_31 == 10
    var_32 = module_0.Payment()
    var_33 = module_0.Payment()
    var_34 = module_0.Payment()
    var_35 = module_0.Payment()
    var_36 = var_35.credit_card_number()
    var_37 = var_35.credit_card_number()
    var_38 = 'VISA'
    var_39 = var_35.credit_card_number(var_38)
    var_40 = 1
    var_41 = var_35.credit_card_number(var_40)
    var_42 = len(var_15)
    assert var_42 == 19
    var_43 = len(var_15)
    var_44 = module_0.Payment()
    var_45 = module_0.Payment()
    var_46 = module_0.Payment()
    var_47 = len(var_15)
    assert var_47 == 19
    var_48 = len(var_15)
    assert var_48 == 17
    var_49 = len(var_15)
    assert var_49 == 19
    var_50 = var_44.credit_card_number(var_12)
    var_51 = len(var_50)
    assert var_51 == 19
    var_52 = len(var_50)
    var_53 = module_0.Payment()
    var_54 = module_0.Payment()
    var_55 = module_0.Payment()
    var_56 = module_0.Payment()
    var_57 = module_0.Payment()
    var_58 = module_0.Payment()
    var_59 = module_0.Payment()
    var_60 = module_0.Payment()



####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import re as module_1


def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = len(var_1)
    assert var_2 == 19
    var_3 = ' '
    var_4 = len(var_1)
    assert var_4 == 19
    var_5 = len(var_1)
    assert var_5 == 19
    var_6 = len(var_1)
    assert var_6 == 17
    var_7 = 'InvalidCardType'
    var_8 = var_0.credit_card_number(var_7)
    var_9 = len(var_1)
    assert var_9 == 19
    var_10 = len(var_1)
    assert var_10 == 19
    var_11 = len(var_1)
    assert var_11 == 17
    var_12 = '4'
    var_13 = '2'
    var_14 = '5'
    var_15 = '3'
    var_16 = ''
    var_17 = module_1.split(var_8)
    var_18 = len(var_17)
    assert var_18 == 4
    var_19 = 4
    var_20 = module_1.split(var_8)
    var_21 = len(var_20)
    assert var_21 == 4
    var_22 = module_1.split(var_8)
    var_23 = len(var_22)
    assert var_23 == 3
    var_24 = 0
    var_25 = var_22[var_24]
    var_26 = len(var_25)
    assert var_26 == 4
    var_27 = 1
    var_28 = var_22[var_27]
    var_29 = len(var_28)
    assert var_29 == 6
    var_30 = 2
    var_31 = var_22[var_30]
    var_32 = len(var_31)
    assert var_32 == 5



# Parsed testcases at query #2
#--------------------------



def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = len(var_1)
    assert var_2 == 19
    var_3 = ' '
    var_4 = len(var_1)
    assert var_4 == 19
    var_5 = len(var_1)
    assert var_5 == 19
    var_6 = len(var_1)
    assert var_6 == 17
    var_7 = 'invalid'
    var_8 = var_0.credit_card_number(var_7)



# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = len(var_1)
    assert var_2 == 19
    var_3 = len(var_1)
    assert var_3 == 19
    var_4 = len(var_1)
    assert var_4 == 17
    var_5 = 'Invalid'
    var_6 = var_0.credit_card_number(var_5)



# Parsed testcases at query #4
#--------------------------



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
    var_7 = 'Invalid'
    var_8 = var_0.credit_card_number(var_7)
    var_9 = None
    var_10 = var_0.credit_card_number(var_9)
    var_11 = len(var_10)
    assert var_11 == 19
    var_12 = len(var_10)
    var_13 = var_0.credit_card_number()
    var_14 = len(var_13)
    assert var_14 == 19
    var_15 = ' '
    var_16 = ''
    var_17 = 42
    var_18 = module_0.Payment()
    var_19 = var_18.credit_card_number()
    var_20 = module_0.Payment()
    var_21 = var_20.credit_card_number()
    var_22 = module_0.Payment()
    var_23 = var_22.credit_card_number()
    var_24 = 43
    var_25 = module_0.Payment()
    var_26 = var_25.credit_card_number()
    var_27 = len(var_13)
    assert var_27 == 19
    var_28 = len(var_13)
    assert var_28 == 19
    var_29 = len(var_13)
    assert var_29 == 17
    var_30 = var_20.credit_card_number(var_9)
    var_31 = len(var_30)
    assert var_31 == 19
    var_32 = len(var_30)
    var_33 = var_20.credit_card_number()
    var_34 = len(var_33)
    assert var_34 == 19
    var_35 = ' '
    var_36 = ''
    var_37 = module_0.Payment()
    var_38 = var_37.credit_card_number()
    var_39 = module_0.Payment()
    var_40 = var_39.credit_card_number()
    var_41 = module_0.Payment()
    var_42 = var_41.credit_card_number()
    var_43 = module_0.Payment()
    var_44 = var_43.credit_card_number()
    var_45 = len(var_33)
    assert var_45 == 19
    var_46 = len(var_33)
    assert var_46 == 19
    var_47 = len(var_33)
    assert var_47 == 17
    var_48 = var_39.credit_card_number(var_9)
    var_49 = len(var_48)
    assert var_49 == 19
    var_50 = len(var_48)
    var_51 = var_39.credit_card_number()
    var_52 = len(var_51)
    assert var_52 == 19
    var_53 = ' '
    var_54 = ''
    var_55 = module_0.Payment()
    var_56 = var_55.credit_card_number()
    var_57 = module_0.Payment()
    var_58 = var_57.credit_card_number()
    var_59 = module_0.Payment()
    var_60 = var_59.credit_card_number()
    var_61 = module_0.Payment()
    var_62 = var_61.credit_card_number()
    var_63 = len(var_51)
    assert var_63 == 19
    var_64 = len(var_51)
    assert var_64 == 19
    var_65 = len(var_51)
    assert var_65 == 17
    var_66 = var_57.credit_card_number(var_9)
    var_67 = len(var_66)
    assert var_67 == 19
    var_68 = len(var_66)
    var_69 = var_57.credit_card_number()
    var_70 = len(var_69)
    assert var_70 == 19
    var_71 = ' '
    var_72 = ''
    var_73 = module_0.Payment()
    var_74 = var_73.credit_card_number()
    var_75 = module_0.Payment()
    var_76 = var_75.credit_card_number()



# Parsed testcases at query #5
#--------------------------



def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = len(var_1)
    assert var_2 == 19
    var_3 = len(var_1)
    assert var_3 == 19
    var_4 = len(var_1)
    assert var_4 == 17
    var_5 = 'invalid'
    var_6 = var_0.credit_card_number(var_5)



# Parsed testcases at query #6
#--------------------------



def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = len(var_1)
    assert var_2 == 19
    var_3 = len(var_1)
    assert var_3 == 19
    var_4 = len(var_1)
    assert var_4 == 19
    var_5 = len(var_1)
    assert var_5 == 17
    var_6 = 'InvalidCardType'
    var_7 = var_0.credit_card_number(var_6)



# Parsed testcases at query #7
#--------------------------



def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = len(var_1)
    assert var_2 == 19
    var_3 = len(var_1)
    assert var_3 == 19
    var_4 = len(var_1)
    assert var_4 == 17
    var_5 = 'Invalid'
    var_6 = var_0.credit_card_number(var_5)
    var_7 = None
    var_8 = var_0.credit_card_number(var_7)
    var_9 = len(var_8)
    assert var_9 == 19
    var_10 = len(var_8)
    var_11 = set()
    var_12 = var_0.credit_card_number()
    var_13 = len(var_11)
    assert var_13 == 100
    var_14 = 12345
    var_15 = module_0.Payment()
    var_16 = module_0.Payment()
    var_17 = var_15.credit_card_number()
    var_18 = var_16.credit_card_number()
    var_19 = module_0.Payment()
    var_20 = 54321
    var_21 = module_0.Payment()
    var_22 = var_19.credit_card_number()
    var_23 = var_21.credit_card_number()
    var_24 = module_0.Payment()
    var_25 = '4455 5299 1152 2450'
    var_26 = module_0.Payment()
    var_27 = '2720 5299 1152 2450'
    var_28 = module_0.Payment()
    var_29 = '3411 115224 508'
    var_30 = module_0.Payment()
    var_31 = '4455 5299 1152 2450'
    var_32 = module_0.Payment()
    var_33 = '2720 5299 1152 2450'
    var_34 = module_0.Payment()
    var_35 = '3411 115224 508'
    var_36 = 0
    var_37 = module_0.Payment()
    var_38 = '4000 0000 0000 0002'
    var_39 = module_0.Payment()
    var_40 = '2221 0000 0000 0009'
    var_41 = module_0.Payment()
    var_42 = '3400 000000 00004'
    var_43 = 2
    var_44 = 32
    var_45 = var_43 ** var_44
    var_46 = 1
    var_47 = var_45 - var_46
    var_48 = module_0.Payment()
    var_49 = '4999 9999 9999 9997'
    var_50 = var_43 ** var_44
    var_51 = var_50 - var_46
    var_52 = module_0.Payment()
    var_53 = '5599 9999 9999 9995'
    var_54 = var_43 ** var_44
    var_55 = var_54 - var_46
    var_56 = module_0.Payment()
    var_57 = '3799 999999 99998'
    var_58 = 123456789
    var_59 = module_0.Payment()
    var_60 = '4455 5299 1152 2450'
    var_61 = module_0.Payment()
    var_62 = '2720 5299 1152 2450'
    var_63 = module_0.Payment()
    var_64 = '3411 115224 508'
    var_65 = 987654321
    var_66 = module_0.Payment()
    var_67 = '4455 5299 1152 2450'
    var_68 = module_0.Payment()



# Parsed testcases at query #8
#--------------------------


import re as module_2

import mimesis.shortcuts as module_1


def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = len(var_1)
    assert var_2 == 19
    var_3 = len(var_1)
    assert var_3 == 19
    var_4 = len(var_1)
    assert var_4 == 19
    var_5 = len(var_1)
    assert var_5 == 17
    var_6 = 'Invalid'
    var_7 = var_0.credit_card_number(var_6)
    var_8 = ' '
    var_9 = ''
    var_10 = -1
    var_11 = result.replace(var_8, var_9)[:var_10]
    var_12 = module_1.luhn_checksum(var_11)
    var_13 = -1
    var_14 = result.replace(var_8, var_9)[var_13]
    var_15 = -1
    var_16 = result.replace(var_8, var_9)[:var_15]
    var_17 = module_1.luhn_checksum(var_16)
    var_18 = -1
    var_19 = result.replace(var_8, var_9)[var_18]
    var_20 = -1
    var_21 = result.replace(var_8, var_9)[:var_20]
    var_22 = module_1.luhn_checksum(var_21)
    var_23 = -1
    var_24 = result.replace(var_8, var_9)[var_23]
    var_25 = '^\\d{4} \\d{4} \\d{4} \\d{4}$'
    var_26 = module_2.match(var_25, var_1)
    var_27 = module_2.match(var_25, var_1)
    var_28 = '^\\d{4} \\d{6} \\d{5}$'
    var_29 = module_2.match(var_28, var_1)



# Parsed testcases at query #9
#--------------------------



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
    assert var_6 == 19
    var_7 = len(var_1)
    assert var_7 == 17
    var_8 = 'invalid'
    var_9 = var_0.credit_card_number(var_8)



# Parsed testcases at query #10
#--------------------------



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
    var_7 = (var_5, var_6)
    var_8 = len(var_1)
    assert var_8 == 17
    var_9 = '34'
    var_10 = '37'
    var_11 = (var_9, var_10)
    var_12 = 'Invalid'
    var_13 = var_0.credit_card_number(var_12)
    var_14 = None
    var_15 = var_0.credit_card_number(var_14)
    var_16 = len(var_15)
    assert var_16 == 19
    var_17 = len(var_15)
    assert var_17 == 19
    var_18 = '4'
    var_19 = len(var_15)
    assert var_19 == 19
    var_20 = '2'
    var_21 = '5'
    var_22 = (var_20, var_21)
    var_23 = len(var_15)
    assert var_23 == 17
    var_24 = '34'
    var_25 = '37'
    var_26 = (var_24, var_25)
    var_27 = set()
    var_28 = var_0.credit_card_number()
    var_29 = len(var_27)
    assert var_29 == 100
    var_30 = 12345
    var_31 = module_0.Payment()
    var_32 = module_0.Payment()
    var_33 = var_31.credit_card_number()
    var_34 = var_32.credit_card_number()
    var_35 = module_0.Payment()
    var_36 = 54321
    var_37 = module_0.Payment()
    var_38 = var_35.credit_card_number()
    var_39 = var_37.credit_card_number()
    var_40 = module_0.Payment()
    var_41 = module_0.Payment()
    var_42 = module_0.Payment()
    var_43 = module_0.Payment()
    var_44 = module_0.Payment()
    var_45 = module_0.Payment()
    var_46 = module_0.Payment()
    var_47 = set()
    var_48 = var_46.credit_card_number(var_23)
    var_49 = len(var_47)
    assert var_49 == 1
    var_50 = module_0.Payment()
    var_51 = module_0.Payment()
    var_52 = module_0.Payment()
    var_53 = module_0.Payment()
    var_54 = []
    var_55 = module_0.Payment()
    var_56 = []
    var_57 = module_0.Payment()
    var_58 = module_0.Payment()
    var_59 = []
    var_60 = []
    var_61 = module_0.Payment()
    var_62 = module_0.Payment()
    var_63 = []
    var_64 = []
    var_65 = module_0.Payment()
    var_66 = []
    var_67 = var_64



# Parsed testcases at query #11
#--------------------------



def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = len(var_1)
    assert var_2 == 19
    var_3 = module_0.Payment()
    var_4 = len(var_1)
    assert var_4 == 19
    var_5 = module_0.Payment()
    var_6 = len(var_1)
    assert var_6 == 19
    var_7 = module_0.Payment()
    var_8 = len(var_1)
    assert var_8 == 17
    var_9 = module_0.Payment()
    var_10 = 'unsupported_card_type'
    var_11 = var_9.credit_card_number(var_10)
    var_12 = module_0.Payment()
    var_13 = len(var_1)
    assert var_13 == 19
    var_14 = module_0.Payment()
    var_15 = len(var_1)
    assert var_15 == 19
    var_16 = module_0.Payment()
    var_17 = len(var_1)
    assert var_17 == 17
    var_18 = module_0.Payment()
    var_19 = module_0.Payment()
    var_20 = module_0.Payment()
    var_21 = module_0.Payment()
    var_22 = module_0.Payment()
    var_23 = module_0.Payment()
    var_24 = module_0.Payment()
    var_25 = module_0.Payment()
    var_26 = module_0.Payment()
    var_27 = module_0.Payment()
    var_28 = module_0.Payment()
    var_29 = module_0.Payment()
    var_30 = module_0.Payment()



# Parsed testcases at query #12
#--------------------------



def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = len(var_1)
    assert var_2 == 19
    var_3 = '4'
    var_4 = module_0.Payment()
    var_5 = len(var_1)
    assert var_5 == 19
    var_6 = '5'
    var_7 = '2'
    var_8 = module_0.Payment()
    var_9 = len(var_1)
    assert var_9 == 17
    var_10 = '34'
    var_11 = '37'
    var_12 = module_0.Payment()
    var_13 = 'InvalidCardType'
    var_14 = var_12.credit_card_number(var_13)
    var_15 = module_0.Payment()
    var_16 = None
    var_17 = var_15.credit_card_number(var_16)
    var_18 = len(var_17)
    assert var_18 == 19
    var_19 = module_0.Payment()
    var_20 = var_19.credit_card_number()
    var_21 = len(var_20)
    assert var_21 == 19
    var_22 = module_0.Payment()
    var_23 = len(var_20)
    assert var_23 == 19
    var_24 = module_0.Payment()
    var_25 = len(var_20)
    assert var_25 == 19
    var_26 = module_0.Payment()
    var_27 = len(var_20)
    assert var_27 == 17
    var_28 = module_0.Payment()
    var_29 = ' '
    var_30 = ''
    var_31 = -1
    var_32 = var_20[:var_31]
    var_33 = module_1.luhn_checksum(var_32)
    var_34 = -1
    var_35 = var_20[var_34]
    var_36 = int(var_35)
    var_37 = module_0.Payment()
    var_38 = -1
    var_39 = var_20[:var_38]
    var_40 = module_1.luhn_checksum(var_39)
    var_41 = -1
    var_42 = var_20[var_41]
    var_43 = int(var_42)
    var_44 = module_0.Payment()
    var_45 = -1
    var_46 = var_20[:var_45]
    var_47 = module_1.luhn_checksum(var_46)
    var_48 = -1
    var_49 = var_20[var_48]
    var_50 = int(var_49)
    var_51 = module_0.Payment()
    var_52 = '^\\d{4} \\d{4} \\d{4} \\d{4}$'
    var_53 = module_2.match(var_52, var_20)
    var_54 = module_0.Payment()
    var_55 = module_2.match(var_52, var_20)
    var_56 = module_0.Payment()
    var_57 = '^\\d{4} \\d{6} \\d{5}$'
    var_58 = module_2.match(var_57, var_20)
    var_59 = module_0.Payment()
    var_60 = len(var_20)
    assert var_60 == 16
    var_61 = module_0.Payment()
    var_62 = len(var_20)
    assert var_62 == 16
    var_63 = module_0.Payment()
    var_64 = len(var_20)
    assert var_64 == 15
    var_65 = module_0.Payment()
    var_66 = module_0.Payment()
    var_67 = module_0.Payment()
    var_68 = module_0.Payment()
    var_69 = module_0.Payment()
    var_70 = module_0.Payment()
    var_71 = module_0.Payment()



# Parsed testcases at query #13
#--------------------------



def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = len(var_1)
    assert var_2 == 19
    var_3 = module_0.Payment()
    var_4 = len(var_1)
    assert var_4 == 19
    var_5 = module_0.Payment()
    var_6 = len(var_1)
    assert var_6 == 19
    var_7 = module_0.Payment()
    var_8 = len(var_1)
    assert var_8 == 17
    var_9 = module_0.Payment()
    var_10 = 'unsupported'
    var_11 = var_9.credit_card_number(var_10)



# Parsed testcases at query #14
#--------------------------


import re as module_1


def test_case_0():
    var_0 = module_0.Payment()
    var_1 = var_0.credit_card_number()
    var_2 = len(var_1)
    assert var_2 == 19
    var_3 = len(var_1)
    assert var_3 == 19
    var_4 = len(var_1)
    assert var_4 == 17
    var_5 = 'Invalid'
    var_6 = var_0.credit_card_number(var_5)
    var_7 = None
    var_8 = var_0.credit_card_number(var_7)
    var_9 = len(var_8)
    assert var_9 == 19
    var_10 = len(var_8)
    var_11 = len(var_8)
    assert var_11 == 19
    var_12 = len(var_8)
    assert var_12 == 19
    var_13 = len(var_8)
    assert var_13 == 17
    var_14 = ' '
    var_15 = ''
    var_16 = -1
    var_17 = -1
    var_18 = -1
    var_19 = module_1.split(var_14)
    var_20 = len(var_19)
    assert var_20 == 4
    var_21 = 4
    var_22 = module_1.split(var_14)
    var_23 = len(var_22)
    assert var_23 == 4
    var_24 = module_1.split(var_14)
    var_25 = len(var_24)
    assert var_25 == 3
    var_26 = 0
    var_27 = var_24[var_26]
    var_28 = len(var_27)
    assert var_28 == 4
    var_29 = 1
    var_30 = var_24[var_29]
    var_31 = len(var_30)
    assert var_31 == 6
    var_32 = 2
    var_33 = var_24[var_32]
    var_34 = len(var_33)
    assert var_34 == 5
    var_35 = var_8[:var_21]
    var_36 = int(var_35)
    var_37 = var_8[:var_21]
    var_38 = int(var_37)
    var_39 = var_8[:var_32]
    var_40 = int(var_39)



