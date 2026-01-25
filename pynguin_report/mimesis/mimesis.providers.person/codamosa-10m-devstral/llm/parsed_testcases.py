####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()
    var_2 = len(var_1)



# Parsed testcases at query #2
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()
    var_2 = 'invalid_gender'
    var_3 = var_0.patronymic(var_2)



# Parsed testcases at query #3
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()
    var_2 = 'invalid_gender'
    var_3 = var_0.patronymic(var_2)
    var_4 = None
    var_5 = var_0.patronymic(var_4)



# Parsed testcases at query #4
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()



# Parsed testcases at query #5
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()
    var_2 = None
    var_3 = var_0.surname(var_2)
    var_4 = 'invalid_gender'
    var_5 = var_0.surname(var_4)



# Parsed testcases at query #6
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()
    var_2 = var_0.patronymic()
    assert var_2 is None



# Parsed testcases at query #7
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()
    var_2 = 'invalid_gender'
    var_3 = var_0.patronymic(var_2)



# Parsed testcases at query #8
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()
    var_2 = 10
    var_3 = range(var_2)
    var_4 = [person.surname() for _ in var_3]
    var_5 = set(var_4)
    var_6 = len(var_5)



# Parsed testcases at query #9
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()
    var_2 = len(var_1)



# Parsed testcases at query #10
#--------------------------




# Parsed testcases at query #11
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()
    var_2 = len(var_1)



# Parsed testcases at query #12
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()
    var_2 = None
    var_3 = var_0.surname(var_2)



# Parsed testcases at query #13
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()
    var_2 = len(var_1)



# Parsed testcases at query #14
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()
    var_2 = var_0.patronymic()
    assert var_2 is None



# Parsed testcases at query #15
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()



# Parsed testcases at query #16
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()



# Parsed testcases at query #17
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()
    var_2 = len(var_1)
    var_3 = None
    var_4 = var_0.nationality(var_3)
    var_5 = len(var_4)



# Parsed testcases at query #18
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()



# Parsed testcases at query #19
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.email()
    var_2 = -1
    var_3 = '@'
    var_4 = email.split(var_3)[var_2]
    var_5 = 'example.com'
    var_6 = 'test.org'
    var_7 = [var_5, var_6]
    var_8 = var_0.email(var_7)
    var_9 = -1
    var_10 = email.split(var_3)[var_9]
    var_11 = True
    var_12 = var_0.email(unique=var_11)
    var_13 = var_0.email(unique=var_11)
    var_14 = 42
    var_15 = module_0.Person()
    var_16 = True
    var_17 = var_15.email(unique=var_16)



# Parsed testcases at query #20
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()



# Parsed testcases at query #21
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()
    var_2 = len(var_1)



# Parsed testcases at query #22
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()



# Parsed testcases at query #23
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()
    var_2 = None
    var_3 = var_0.surname(var_2)



# Parsed testcases at query #24
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()
    var_2 = len(var_1)
    var_3 = 'nationality'
    var_4 = [var_3]



# Parsed testcases at query #25
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()



# Parsed testcases at query #26
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()
    var_2 = 10
    var_3 = range(var_2)
    var_4 = [person.surname() for _ in var_3]
    var_5 = set(var_4)
    var_6 = len(var_5)



# Parsed testcases at query #27
#--------------------------




# Parsed testcases at query #28
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()



# Parsed testcases at query #29
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()
    var_2 = 'invalid'
    var_3 = var_0.patronymic(var_2)



# Parsed testcases at query #30
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()
    var_2 = 'invalid'
    var_3 = var_0.patronymic(var_2)
    assert var_3 is None



# Parsed testcases at query #31
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()
    var_2 = 'invalid_gender'
    var_3 = var_0.patronymic(var_2)



# Parsed testcases at query #32
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()
    var_2 = len(var_1)
    var_3 = None
    var_4 = var_0.surname(var_3)
    var_5 = len(var_4)
    var_6 = 'surnames'
    var_7 = [var_6]



# Parsed testcases at query #33
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()
    var_2 = None
    var_3 = len(var_1)



# Parsed testcases at query #34
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()
    var_2 = len(var_1)



# Parsed testcases at query #35
#--------------------------


import mimesis.providers.person as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.username()
    var_2 = '_'
    var_3 = module_1.split(var_2)
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = 0
    var_6 = username.split(var_2)[var_5]
    var_7 = 1
    var_8 = username.split(var_2)[var_7]
    var_9 = 'l_l_d'
    var_10 = var_0.username(var_9)
    var_11 = module_1.split(var_2)
    var_12 = len(var_11)
    assert var_12 == 3
    var_13 = var_11[var_5]
    var_14 = var_11[var_7]
    var_15 = 2
    var_16 = var_11[var_15]
    var_17 = 'U.U.d'
    var_18 = var_0.username(var_17)
    var_19 = '.'
    var_20 = module_1.split(var_19)
    var_21 = len(var_20)
    assert var_21 == 3
    var_22 = var_20[var_5]
    var_23 = var_20[var_7]
    var_24 = var_20[var_15]
    var_25 = 'C-d'
    var_26 = var_0.username(var_25)
    var_27 = '-'
    var_28 = module_1.split(var_27)
    var_29 = len(var_28)
    assert var_29 == 2
    var_30 = var_28[var_5][var_5]
    var_31 = var_28[var_5][var_7:]
    var_32 = var_28[var_7]
    var_33 = 1900
    var_34 = 2000
    var_35 = (var_33, var_34)
    var_36 = var_0.username(drange=var_35)
    var_37 = -1
    var_38 = username.split(var_2)[var_37]
    var_39 = int(var_38)
    var_40 = '123'
    var_41 = var_0.username(var_40)
    var_42 = 1900
    var_43 = (var_42,)
    var_44 = var_0.username(drange=var_43)



# Parsed testcases at query #36
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()



# Parsed testcases at query #37
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()



# Parsed testcases at query #38
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.email()
    var_2 = 1
    var_3 = '@'
    var_4 = email.split(var_3)[var_2]
    var_5 = 'example.com'
    var_6 = 'test.org'
    var_7 = [var_5, var_6]
    var_8 = var_0.email(var_7)
    var_9 = email.split(var_3)[var_2]
    var_10 = True
    var_11 = var_0.email(unique=var_10)
    var_12 = True
    var_13 = var_0.email(unique=var_12)
    var_14 = 42
    var_15 = module_0.Person()
    var_16 = True
    var_17 = var_15.email(unique=var_16)



# Parsed testcases at query #39
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()



# Parsed testcases at query #40
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()



# Parsed testcases at query #41
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()



# Parsed testcases at query #42
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()
    var_2 = len(var_1)



# Parsed testcases at query #43
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()



# Parsed testcases at query #44
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()
    var_2 = 'invalid_gender'
    var_3 = var_0.patronymic(var_2)



# Parsed testcases at query #45
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()
    var_2 = None
    var_3 = var_0.surname(var_2)



# Parsed testcases at query #46
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()
    var_2 = len(var_1)



# Parsed testcases at query #47
#--------------------------


import mimesis.providers.person as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.username()
    var_2 = '_'
    var_3 = module_1.split(var_2)
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = 0
    var_6 = username.split(var_2)[var_5]
    var_7 = 1
    var_8 = username.split(var_2)[var_7]
    var_9 = 'l_d'
    var_10 = var_0.username(var_9)
    var_11 = module_1.split(var_2)
    var_12 = len(var_11)
    assert var_12 == 2
    var_13 = username.split(var_2)[var_5]
    var_14 = username.split(var_2)[var_7]
    var_15 = 'U_d'
    var_16 = var_0.username(var_15)
    var_17 = module_1.split(var_2)
    var_18 = len(var_17)
    assert var_18 == 2
    var_19 = username.split(var_2)[var_5]
    var_20 = username.split(var_2)[var_7]
    var_21 = 'C_d'
    var_22 = var_0.username(var_21)
    var_23 = module_1.split(var_2)
    var_24 = len(var_23)
    assert var_24 == 2
    var_25 = username.split(var_2)[var_5]
    var_26 = username.split(var_2)[var_7]
    var_27 = 'l.l.d'
    var_28 = var_0.username(var_27)
    var_29 = '.'
    var_30 = module_1.split(var_29)
    var_31 = len(var_30)
    assert var_31 == 3
    var_32 = var_30[var_5]
    var_33 = var_30[var_7]
    var_34 = 2
    var_35 = var_30[var_34]
    var_36 = 'l-l-d'
    var_37 = var_0.username(var_36)
    var_38 = '-'
    var_39 = module_1.split(var_38)
    var_40 = len(var_39)
    assert var_40 == 3
    var_41 = var_39[var_5]
    var_42 = var_39[var_7]
    var_43 = var_39[var_34]
    var_44 = 'l_l_d'
    var_45 = var_0.username(var_44)
    var_46 = module_1.split(var_2)
    var_47 = len(var_46)
    assert var_47 == 3
    var_48 = var_46[var_5]
    var_49 = var_46[var_7]
    var_50 = var_46[var_34]
    var_51 = 1900
    var_52 = 2000
    var_53 = (var_51, var_52)
    var_54 = var_0.username(var_9, var_53)
    var_55 = username.split(var_2)[var_7]
    var_56 = int(var_55)
    var_57 = 'd_d'
    var_58 = var_0.username(var_57)
    var_59 = 'l_d'
    var_60 = 1900
    var_61 = (var_60,)
    var_62 = var_0.username(var_59, var_61)



# Parsed testcases at query #48
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()
    var_2 = len(var_1)



# Parsed testcases at query #49
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import mimesis.providers.person as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.username()
    var_2 = '_'
    var_3 = module_1.split(var_2)
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = 0
    var_6 = username.split(var_2)[var_5]
    var_7 = 1
    var_8 = username.split(var_2)[var_7]
    var_9 = 'C_C_d'
    var_10 = var_0.username(var_9)
    var_11 = module_1.split(var_2)
    var_12 = len(var_11)
    assert var_12 == 3
    var_13 = username.split(var_2)[var_5][var_5]
    var_14 = username.split(var_2)[var_7][var_5]
    var_15 = 2
    var_16 = username.split(var_2)[var_15]
    var_17 = 1900
    var_18 = 2021
    var_19 = (var_17, var_18)
    var_20 = var_0.username(drange=var_19)
    var_21 = module_1.split(var_2)
    var_22 = len(var_21)
    assert var_22 == 2
    var_23 = username.split(var_2)[var_5]
    var_24 = username.split(var_2)[var_7]
    var_25 = int(var_24)
    var_26 = '123'
    var_27 = var_0.username(var_26)
    var_28 = 1900
    var_29 = (var_28,)
    var_30 = var_0.username(drange=var_29)



# Parsed testcases at query #2
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()
    var_2 = len(var_1)



# Parsed testcases at query #3
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()
    var_2 = 'en'
    var_3 = module_0.Person()
    var_4 = var_3.patronymic()
    assert var_4 is None



# Parsed testcases at query #4
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()
    var_2 = None
    var_3 = var_0.surname(var_2)



# Parsed testcases at query #5
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()
    var_2 = len(var_1)



# Parsed testcases at query #6
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()



# Parsed testcases at query #7
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()



# Parsed testcases at query #8
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.email()
    var_2 = 1
    var_3 = '@'
    var_4 = email.split(var_3)[var_2]
    var_5 = 'example.com'
    var_6 = 'test.org'
    var_7 = [var_5, var_6]
    var_8 = var_0.email(var_7)
    var_9 = email.split(var_3)[var_2]
    var_10 = True
    var_11 = var_0.email(unique=var_10)
    var_12 = True
    var_13 = var_0.email(unique=var_12)
    var_14 = 42
    var_15 = module_0.Person()
    var_16 = True
    var_17 = var_15.email(unique=var_16)



# Parsed testcases at query #9
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()
    var_2 = 'en'
    var_3 = module_0.Person()
    var_4 = var_3.patronymic()
    assert var_4 is None



# Parsed testcases at query #10
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()



# Parsed testcases at query #11
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()
    var_2 = len(var_1)



# Parsed testcases at query #12
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()



# Parsed testcases at query #13
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()
    var_2 = var_0.nationality()
    var_3 = var_0.nationality()



# Parsed testcases at query #14
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()
    var_2 = 'invalid_gender'
    var_3 = var_0.patronymic(var_2)



# Parsed testcases at query #15
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()
    var_2 = '-'
    var_3 = "'"
    var_4 = 'invalid_gender'
    var_5 = var_0.surname(var_4)



# Parsed testcases at query #16
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()
    var_2 = 'invalid_gender'
    var_3 = var_0.patronymic(var_2)



# Parsed testcases at query #17
#--------------------------


import mimesis.providers.person as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.username()
    var_2 = '_'
    var_3 = module_1.split(var_2)
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = 0
    var_6 = username.split(var_2)[var_5]
    var_7 = 1
    var_8 = username.split(var_2)[var_7]
    var_9 = 'U.U.d'
    var_10 = var_0.username(var_9)
    var_11 = '.'
    var_12 = module_1.split(var_11)
    var_13 = len(var_12)
    assert var_13 == 3
    var_14 = var_12[var_5]
    var_15 = var_12[var_7]
    var_16 = 2
    var_17 = var_12[var_16]
    var_18 = 'C-C-d'
    var_19 = var_0.username(var_18)
    var_20 = '-'
    var_21 = module_1.split(var_20)
    var_22 = len(var_21)
    assert var_22 == 3
    var_23 = var_21[var_5][var_5]
    var_24 = var_21[var_5][var_7:]
    var_25 = var_21[var_7][var_5]
    var_26 = var_21[var_7][var_7:]
    var_27 = var_21[var_16]
    var_28 = 1950
    var_29 = 2000
    var_30 = (var_28, var_29)
    var_31 = var_0.username(drange=var_30)
    var_32 = -1
    var_33 = username.split(var_2)[var_32]
    var_34 = int(var_33)
    var_35 = 'd.d.d'
    var_36 = var_0.username(var_35)
    var_37 = 1900
    var_38 = 2000
    var_39 = 2020
    var_40 = (var_37, var_38, var_39)
    var_41 = var_0.username(drange=var_40)



# Parsed testcases at query #18
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()
    var_2 = len(var_1)



# Parsed testcases at query #19
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()



# Parsed testcases at query #20
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()
    var_2 = len(var_1)
    var_3 = 'nationality'
    var_4 = [var_3]



# Parsed testcases at query #21
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()
    var_2 = 10
    var_3 = range(var_2)
    var_4 = [person.surname() for _ in var_3]
    var_5 = set(var_4)
    var_6 = len(var_5)



# Parsed testcases at query #22
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()
    var_2 = len(var_1)
    var_3 = None
    var_4 = var_0.surname(var_3)



# Parsed testcases at query #23
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()
    var_2 = len(var_1)
    var_3 = 'nationality'
    var_4 = [var_3]



# Parsed testcases at query #24
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()
    var_2 = len(var_1)
    var_3 = 10
    var_4 = range(var_3)
    var_5 = [person.surname() for _ in var_4]
    var_6 = set(var_5)
    var_7 = len(var_6)



# Parsed testcases at query #25
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()
    var_2 = 'en'
    var_3 = module_0.Person()
    var_4 = var_3.patronymic()
    assert var_4 is None
    var_5 = 'ru'
    var_6 = module_0.Person()
    var_7 = var_6.patronymic()



# Parsed testcases at query #26
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()



# Parsed testcases at query #27
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()
    var_2 = len(var_1)
    var_3 = None
    var_4 = var_0.nationality(var_3)
    var_5 = len(var_4)



# Parsed testcases at query #28
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()
    var_2 = len(var_1)
    var_3 = None
    var_4 = var_0.nationality(var_3)
    var_5 = len(var_4)



# Parsed testcases at query #29
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()
    var_2 = None
    var_3 = var_0.surname(var_2)



# Parsed testcases at query #30
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()



# Parsed testcases at query #31
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()



# Parsed testcases at query #32
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()
    var_2 = var_0.patronymic()
    assert var_2 is None



# Parsed testcases at query #33
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()
    var_2 = len(var_1)



# Parsed testcases at query #34
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()
    var_2 = len(var_1)



# Parsed testcases at query #35
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()
    var_2 = None
    var_3 = var_0.surname(var_2)



# Parsed testcases at query #36
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()
    var_2 = 'en'
    var_3 = module_0.Person()
    var_4 = var_3.patronymic()
    assert var_4 is None



# Parsed testcases at query #37
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()
    var_2 = len(var_1)
    var_3 = None
    var_4 = var_0.nationality(var_3)
    var_5 = len(var_4)



# Parsed testcases at query #38
#--------------------------


import mimesis.providers.person as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.username()
    var_2 = len(var_1)
    var_3 = 'l_d'
    var_4 = var_0.username(var_3)
    var_5 = '_'
    var_6 = module_1.split(var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 0
    var_9 = var_6[var_8]
    var_10 = 1
    var_11 = var_6[var_10]
    var_12 = 'U.U'
    var_13 = var_0.username(var_12)
    var_14 = '.'
    var_15 = module_1.split(var_14)
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = var_15[var_8]
    var_18 = var_15[var_10]
    var_19 = 'C-C'
    var_20 = var_0.username(var_19)
    var_21 = '-'
    var_22 = module_1.split(var_21)
    var_23 = len(var_22)
    assert var_23 == 2
    var_24 = var_22[var_8]
    var_25 = var_22[var_10]
    var_26 = 'd'
    var_27 = 1900
    var_28 = 2000
    var_29 = (var_27, var_28)
    var_30 = var_0.username(var_26, var_29)
    var_31 = int(var_30)
    var_32 = '...'
    var_33 = var_0.username(var_32)
    var_34 = 'd'
    var_35 = 1900
    var_36 = (var_35,)
    var_37 = var_0.username(var_34, var_36)
    var_38 = 'C.l-d_U'
    var_39 = var_0.username(var_38)
    var_40 = module_1.split(var_14)
    var_41 = len(var_40)
    assert var_41 == 2
    var_42 = var_40[var_8]
    var_43 = var_40[var_10]
    var_44 = module_1.split(var_21)
    var_45 = len(var_44)
    assert var_45 == 2
    var_46 = var_44[var_8]
    var_47 = var_44[var_10]



# Parsed testcases at query #39
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()
    var_2 = 10
    var_3 = range(var_2)
    var_4 = [person.surname() for _ in var_3]
    var_5 = set(var_4)
    var_6 = len(var_5)



# Parsed testcases at query #40
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()
    var_2 = len(var_1)



# Parsed testcases at query #41
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()
    var_2 = 'en'
    var_3 = module_0.Person()
    var_4 = var_3.patronymic()
    assert var_4 is None



# Parsed testcases at query #42
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()
    var_2 = 'invalid_gender'
    var_3 = var_0.patronymic(var_2)



# Parsed testcases at query #43
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()
    var_2 = var_0.surname()
    var_3 = var_0.surname()



# Parsed testcases at query #44
#--------------------------


def test_case_0():
    var_0 = 'en'
    var_1 = 'ru'
    var_2 = 'ич'
    var_3 = 'ович'
    var_4 = 'на'
    var_5 = 'овна'
    var_6 = 'uk'
    var_7 = 'івна'



# Parsed testcases at query #45
#--------------------------


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()
    var_2 = len(var_1)



