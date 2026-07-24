####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 10
    var_4 = var_2[var_3]
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = 4
    var_9 = 5
    var_10 = [var_5, var_6, var_7, var_8, var_9]
    var_11 = range(var_7)
    var_12 = module_0.LazyList(var_11)
    var_13 = var_12[var_6]



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = -1
    var_4 = -2
    var_5 = 0



# Parsed testcases at query #3
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 10
    var_4 = var_2[var_3]
    var_5 = -11
    var_6 = var_2[var_5]
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = 4
    var_11 = 5
    var_12 = [var_7, var_8, var_9, var_10, var_11]
    var_13 = range(var_11)
    var_14 = module_0.LazyList(var_13)
    var_15 = var_14[var_10]



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = -1
    var_4 = -2
    var_5 = 0



# Parsed testcases at query #5
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 10
    var_4 = var_2[var_3]
    var_5 = range(var_3)
    var_6 = module_0.LazyList(var_5)
    var_7 = 5
    var_8 = range(var_7)
    var_9 = module_0.LazyList(var_8)
    var_10 = list(var_9)



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = -5
    var_3 = 1
    var_4 = -3
    var_5 = 3
    var_6 = 10
    var_7 = 2
    var_8 = -2
    var_9 = -1
    var_10 = 11
    var_11 = 100



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 10
    var_4 = -10



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 11
    var_3 = 2
    var_4 = 0
    var_5 = -2
    var_6 = 5
    var_7 = -5
    var_8 = -10



# Parsed testcases at query #9
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = 3
    var_7 = lambda x: x == var_6
    var_8 = 1
    var_9 = 2
    var_10 = 4
    var_11 = [var_8, var_9, var_6, var_10, var_0]
    var_12 = module_0.drop_until(var_7, var_11)
    var_13 = list(var_12)
    var_14 = 0
    var_15 = lambda x: x % var_9 == var_14
    var_16 = 6
    var_17 = 7
    var_18 = [var_8, var_6, var_0, var_16, var_17]
    var_19 = module_0.drop_until(var_15, var_18)
    var_20 = list(var_19)
    var_21 = lambda x: x > var_2
    var_22 = range(var_0)
    var_23 = module_0.drop_until(var_21, var_22)
    var_24 = list(var_23)
    var_25 = lambda x: x == var_8
    var_26 = [var_8, var_9, var_6]
    var_27 = module_0.drop_until(var_25, var_26)
    var_28 = list(var_27)
    var_29 = lambda x: x > var_14
    var_30 = []
    var_31 = module_0.drop_until(var_29, var_30)
    var_32 = list(var_31)
    var_33 = lambda x: x.val == var_9



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = -1
    var_4 = -2
    var_5 = 0



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 11
    var_3 = 2
    var_4 = 10
    var_5 = -10
    var_6 = 0
    var_7 = 0



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = -1
    var_4 = -2
    var_5 = 0



# Parsed testcases at query #13
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = 1
    var_2 = 2
    var_3 = 4
    var_4 = 5
    var_5 = [var_1, var_2, var_0, var_3, var_4]
    var_6 = module_0.take(var_0, var_5)
    var_7 = list(var_6)
    var_8 = [var_1, var_2, var_0]
    var_9 = module_0.take(var_4, var_8)
    var_10 = list(var_9)
    var_11 = 0
    var_12 = [var_1, var_2, var_0]
    var_13 = module_0.take(var_11, var_12)
    var_14 = list(var_13)
    var_15 = []
    var_16 = module_0.take(var_4, var_15)
    var_17 = list(var_16)
    var_18 = 10
    var_19 = range(var_18)
    var_20 = -1
    var_21 = 1
    var_22 = 2
    var_23 = 3
    var_24 = [var_21, var_22, var_23]
    var_25 = module_0.take(var_20, var_24)
    var_26 = list(var_25)



# Parsed testcases at query #14
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = lambda x: x * var_1
    var_7 = module_0.MapList(var_6, var_5)
    var_8 = 10
    var_9 = var_7[var_8]
    var_10 = -10
    var_11 = var_7[var_10]



# Parsed testcases at query #15
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = [var_1, var_2, var_3, var_4]
    var_6 = module_0.drop(var_0, var_5)
    var_7 = list(var_6)
    var_8 = [var_1, var_2, var_3, var_4]
    var_9 = module_0.drop(var_4, var_8)
    var_10 = list(var_9)
    var_11 = [var_1, var_2, var_3, var_4]
    var_12 = module_0.drop(var_2, var_11)
    var_13 = list(var_12)
    var_14 = 10
    var_15 = [var_1, var_2, var_3, var_4]
    var_16 = module_0.drop(var_14, var_15)
    var_17 = list(var_16)
    var_18 = 5
    var_19 = []
    var_20 = module_0.drop(var_18, var_19)
    var_21 = list(var_20)
    var_22 = -1
    var_23 = 1
    var_24 = 2
    var_25 = 3
    var_26 = [var_23, var_24, var_25]
    var_27 = module_0.drop(var_22, var_26)
    var_28 = list(var_27)
    var_29 = range(var_14)
    var_30 = 'hello'
    var_31 = module_0.drop(var_25, var_30)
    var_32 = list(var_31)



# Parsed testcases at query #16
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 10
    var_4 = var_2[var_3]
    var_5 = -11
    var_6 = var_2[var_5]
    var_7 = 1
    var_8 = 4
    var_9 = 9
    var_10 = 16
    var_11 = 25
    var_12 = [var_7, var_8, var_9, var_10, var_11]
    var_13 = 5
    var_14 = range(var_13)
    var_15 = module_0.LazyList(var_14)
    var_16 = list(var_15)



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = -1
    var_4 = -2
    var_5 = 0



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -5
    var_3 = 1
    var_4 = -5
    var_5 = 5
    var_6 = 2
    var_7 = 3
    var_8 = -1
    var_9 = -2
    var_10 = -1
    var_11 = 100



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -5
    var_3 = 1
    var_4 = -5
    var_5 = 5
    var_6 = 2
    var_7 = 3
    var_8 = -1
    var_9 = -2
    var_10 = -1
    var_11 = -5
    var_12 = -1
    var_13 = -5
    var_14 = -2
    var_15 = -1



# Parsed testcases at query #20
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = lambda x: x * var_1
    var_7 = module_0.MapList(var_6, var_5)
    var_8 = 10
    var_9 = var_7[var_8]



# Parsed testcases at query #21
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = 3
    var_7 = lambda x: x == var_6
    var_8 = range(var_2)
    var_9 = module_0.drop_until(var_7, var_8)
    var_10 = list(var_9)
    var_11 = lambda x: x > var_0
    var_12 = []
    var_13 = module_0.drop_until(var_11, var_12)
    var_14 = list(var_13)
    var_15 = 0
    var_16 = lambda x: x >= var_15
    var_17 = range(var_2)
    var_18 = module_0.drop_until(var_16, var_17)
    var_19 = list(var_18)
    var_20 = 100
    var_21 = lambda x: x > var_20
    var_22 = range(var_2)
    var_23 = module_0.drop_until(var_21, var_22)
    var_24 = list(var_23)
    var_25 = lambda x: x == var_15
    var_26 = range(var_2)
    var_27 = module_0.drop_until(var_25, var_26)
    var_28 = list(var_27)
    var_29 = 9
    var_30 = lambda x: x == var_29
    var_31 = range(var_2)
    var_32 = module_0.drop_until(var_30, var_31)
    var_33 = list(var_32)
    var_34 = 'b'
    var_35 = lambda x: x.startswith(var_34)
    var_36 = 'apple'
    var_37 = 'banana'
    var_38 = 'cherry'
    var_39 = [var_36, var_37, var_38]
    var_40 = module_0.drop_until(var_35, var_39)
    var_41 = list(var_40)



# Parsed testcases at query #22
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = 3
    var_7 = lambda x: x == var_6
    var_8 = 1
    var_9 = 2
    var_10 = 4
    var_11 = [var_8, var_9, var_6, var_10, var_0]
    var_12 = module_0.drop_until(var_7, var_11)
    var_13 = list(var_12)
    var_14 = 0
    var_15 = lambda x: x % var_9 == var_14
    var_16 = 6
    var_17 = 7
    var_18 = [var_8, var_6, var_0, var_16, var_17]
    var_19 = module_0.drop_until(var_15, var_18)
    var_20 = list(var_19)
    var_21 = lambda x: x == var_8
    var_22 = [var_8, var_9, var_6]
    var_23 = module_0.drop_until(var_21, var_22)
    var_24 = list(var_23)
    var_25 = lambda x: x > var_2
    var_26 = range(var_0)
    var_27 = module_0.drop_until(var_25, var_26)
    var_28 = list(var_27)
    var_29 = lambda x: x > var_14
    var_30 = []
    var_31 = module_0.drop_until(var_29, var_30)
    var_32 = list(var_31)
    var_33 = lambda x: x.val == var_9



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 10
    var_4 = -11



# Parsed testcases at query #24
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 10
    var_4 = var_2[var_3]
    var_5 = -11
    var_6 = var_2[var_5]
    var_7 = 1
    var_8 = 3
    var_9 = 5
    var_10 = 7
    var_11 = 9
    var_12 = [var_7, var_8, var_9, var_10, var_11]
    var_13 = module_0.LazyList(var_12)
    var_14 = range(var_9)
    var_15 = module_0.LazyList(var_14)
    var_16 = var_15.list
    var_17 = len(var_16)
    assert var_17 == 0
    var_18 = 4
    var_19 = var_15[var_18]
    var_20 = var_15.list
    var_21 = len(var_20)
    assert var_21 == 5



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 10
    var_4 = -11



# Parsed testcases at query #26
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = 3
    var_7 = lambda x: x == var_6
    var_8 = 1
    var_9 = 2
    var_10 = 4
    var_11 = [var_8, var_9, var_6, var_10, var_0]
    var_12 = module_0.drop_until(var_7, var_11)
    var_13 = list(var_12)
    var_14 = 'b'
    var_15 = lambda x: x.startswith(var_14)
    var_16 = 'a'
    var_17 = 'c'
    var_18 = [var_16, var_14, var_17]
    var_19 = module_0.drop_until(var_15, var_18)
    var_20 = list(var_19)
    var_21 = 0
    var_22 = lambda x: x > var_21
    var_23 = []
    var_24 = module_0.drop_until(var_22, var_23)
    var_25 = list(var_24)
    var_26 = lambda x: x > var_2
    var_27 = range(var_0)
    var_28 = module_0.drop_until(var_26, var_27)
    var_29 = list(var_28)
    var_30 = lambda x: x == var_8
    var_31 = [var_8, var_9, var_6]
    var_32 = module_0.drop_until(var_30, var_31)
    var_33 = list(var_32)
    var_34 = lambda x: x.val > var_8



# Parsed testcases at query #27
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = module_0.LazyList(var_6)
    var_8 = list(var_7)
    var_9 = 10
    var_10 = var_2[var_9]
    var_11 = -11
    var_12 = var_2[var_11]



# Parsed testcases at query #28
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = range(var_0)
    var_4 = module_0.LazyList(var_3)
    var_5 = 5
    var_6 = range(var_5)
    var_7 = module_0.LazyList(var_6)
    var_8 = list(var_7)



# Parsed testcases at query #29
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = 3
    var_7 = lambda x: x == var_6
    var_8 = 1
    var_9 = 2
    var_10 = 4
    var_11 = [var_8, var_9, var_6, var_10, var_0]
    var_12 = module_0.drop_until(var_7, var_11)
    var_13 = list(var_12)
    var_14 = lambda x: x > var_0
    var_15 = []
    var_16 = module_0.drop_until(var_14, var_15)
    var_17 = list(var_16)
    var_18 = lambda x: x > var_2
    var_19 = range(var_0)
    var_20 = module_0.drop_until(var_18, var_19)
    var_21 = list(var_20)
    var_22 = lambda x: x == var_8
    var_23 = [var_8, var_9, var_6]
    var_24 = module_0.drop_until(var_22, var_23)
    var_25 = list(var_24)
    var_26 = lambda x: x.val > var_9
    var_27 = lambda x: x == var_6



# Parsed testcases at query #30
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 10
    var_4 = var_2[var_3]
    var_5 = -11
    var_6 = var_2[var_5]
    var_7 = 1
    var_8 = 4
    var_9 = 9
    var_10 = 16
    var_11 = 25
    var_12 = [var_7, var_8, var_9, var_10, var_11]
    var_13 = module_0.LazyList(var_12)
    var_14 = 5
    var_15 = range(var_14)
    var_16 = module_0.LazyList(var_15)
    var_17 = list(var_16)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = 3
    var_3 = 0
    var_4 = lambda x: x % var_2 == var_3
    var_5 = module_0.split_by(var_1, criterion=var_4)
    var_6 = list(var_5)
    var_7 = range(var_0)
    var_8 = True
    var_9 = lambda x: x % var_2 == var_3
    var_10 = module_0.split_by(var_7, var_8, criterion=var_9)
    var_11 = list(var_10)
    var_12 = ' Split by: '
    var_13 = ' '
    var_14 = module_0.split_by(var_12, separator=var_13)
    var_15 = list(var_14)
    var_16 = module_0.split_by(var_12, var_8, separator=var_13)
    var_17 = list(var_16)
    var_18 = 10
    var_19 = range(var_18)
    var_20 = module_0.split_by(var_19)
    var_21 = list(var_20)
    var_22 = 10
    var_23 = range(var_22)
    var_24 = 3
    var_25 = 0
    var_26 = lambda x: x % var_24 == var_25
    var_27 = module_0.split_by(var_23, criterion=var_26, separator=var_25)
    var_28 = list(var_27)
    var_29 = []
    var_30 = lambda x: x % var_24 == var_25
    var_31 = module_0.split_by(var_29, criterion=var_30)
    var_32 = list(var_31)
    var_33 = []
    var_34 = lambda x: x % var_24 == var_25
    var_35 = module_0.split_by(var_33, var_8, criterion=var_34)
    var_36 = list(var_35)
    var_37 = 6
    var_38 = 9
    var_39 = [var_25, var_24, var_37, var_38]
    var_40 = lambda x: x % var_24 == var_25
    var_41 = module_0.split_by(var_39, criterion=var_40)
    var_42 = list(var_41)
    var_43 = [var_25, var_24, var_37, var_38]
    var_44 = lambda x: x % var_24 == var_25
    var_45 = module_0.split_by(var_43, var_8, criterion=var_44)
    var_46 = list(var_45)
    var_47 = 2
    var_48 = 4
    var_49 = 5
    var_50 = [var_8, var_47, var_48, var_49]
    var_51 = lambda x: x % var_24 == var_25
    var_52 = module_0.split_by(var_50, criterion=var_51)
    var_53 = list(var_52)
    var_54 = [var_8, var_47, var_48, var_49]
    var_55 = lambda x: x % var_24 == var_25
    var_56 = module_0.split_by(var_54, var_8, criterion=var_55)
    var_57 = list(var_56)



# Parsed testcases at query #2
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 10
    var_4 = var_2[var_3]
    var_5 = -11
    var_6 = var_2[var_5]
    var_7 = range(var_5)
    var_8 = 2
    var_9 = 0
    var_10 = 5
    var_11 = range(var_10)
    var_12 = module_0.LazyList(var_11)
    var_13 = 4
    var_14 = var_12[var_13]



# Parsed testcases at query #3
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.drop(var_0, var_4)
    var_6 = list(var_5)
    var_7 = [var_1, var_2, var_3]
    var_8 = module_0.drop(var_3, var_7)
    var_9 = list(var_8)
    var_10 = 4
    var_11 = 5
    var_12 = [var_1, var_2, var_3, var_10, var_11]
    var_13 = module_0.drop(var_2, var_12)
    var_14 = list(var_13)
    var_15 = 10
    var_16 = [var_1, var_2, var_3]
    var_17 = module_0.drop(var_15, var_16)
    var_18 = list(var_17)
    var_19 = []
    var_20 = module_0.drop(var_11, var_19)
    var_21 = list(var_20)
    var_22 = -1
    var_23 = 1
    var_24 = 2
    var_25 = 3
    var_26 = [var_23, var_24, var_25]
    var_27 = module_0.drop(var_22, var_26)
    var_28 = list(var_27)
    var_29 = [var_23, var_24, var_25, var_10, var_11]
    var_30 = 'hello'
    var_31 = module_0.drop(var_25, var_30)
    var_32 = list(var_31)



# Parsed testcases at query #4
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = 3
    var_3 = 0
    var_4 = lambda x: x % var_2 == var_3
    var_5 = module_0.split_by(var_1, criterion=var_4)
    var_6 = list(var_5)
    var_7 = range(var_0)
    var_8 = True
    var_9 = lambda x: x % var_2 == var_3
    var_10 = module_0.split_by(var_7, var_8, criterion=var_9)
    var_11 = list(var_10)
    var_12 = ' Split by: '
    var_13 = ' '
    var_14 = module_0.split_by(var_12, separator=var_13)
    var_15 = list(var_14)
    var_16 = module_0.split_by(var_12, var_8, separator=var_13)
    var_17 = list(var_16)
    var_18 = []
    var_19 = lambda x: x > var_3
    var_20 = module_0.split_by(var_18, criterion=var_19)
    var_21 = list(var_20)
    var_22 = []
    var_23 = module_0.split_by(var_22, separator=var_3)
    var_24 = list(var_23)
    var_25 = 2
    var_26 = [var_8, var_25, var_2]
    var_27 = lambda x: x > var_3
    var_28 = module_0.split_by(var_26, criterion=var_27)
    var_29 = list(var_28)
    var_30 = [var_8, var_25, var_2]
    var_31 = False
    var_32 = lambda x: x > var_31
    var_33 = module_0.split_by(var_30, var_31, criterion=var_32)
    var_34 = list(var_33)
    var_35 = [var_8, var_25, var_2]
    var_36 = lambda x: x > var_0
    var_37 = module_0.split_by(var_35, criterion=var_36)
    var_38 = list(var_37)
    var_39 = 1
    var_40 = 2
    var_41 = 3
    var_42 = [var_39, var_40, var_41]
    var_43 = 0
    var_44 = lambda x: x > var_43
    var_45 = module_0.split_by(var_42, criterion=var_44, separator=var_43)
    var_46 = list(var_45)
    var_47 = 1
    var_48 = 2
    var_49 = 3
    var_50 = [var_47, var_48, var_49]
    var_51 = module_0.split_by(var_50)
    var_52 = list(var_51)



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 1
    var_2 = 10
    var_3 = 2
    var_4 = -1
    var_5 = 0
    var_6 = 1000000



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 1
    var_2 = 10
    var_3 = 2
    var_4 = 0
    var_5 = -1
    var_6 = 20
    var_7 = 3



# Parsed testcases at query #2
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = 3
    var_3 = 0
    var_4 = lambda x: x % var_2 == var_3
    var_5 = module_0.split_by(var_1, criterion=var_4)
    var_6 = list(var_5)
    var_7 = range(var_0)
    var_8 = True
    var_9 = lambda x: x % var_2 == var_3
    var_10 = module_0.split_by(var_7, var_8, criterion=var_9)
    var_11 = list(var_10)
    var_12 = ' Split by: '
    var_13 = ' '
    var_14 = module_0.split_by(var_12, separator=var_13)
    var_15 = list(var_14)
    var_16 = module_0.split_by(var_12, var_8, separator=var_13)
    var_17 = list(var_16)
    var_18 = []
    var_19 = lambda x: x > var_3
    var_20 = module_0.split_by(var_18, criterion=var_19)
    var_21 = list(var_20)
    var_22 = []
    var_23 = module_0.split_by(var_22, var_8, separator=var_3)
    var_24 = list(var_23)
    var_25 = 2
    var_26 = [var_8, var_25, var_2]
    var_27 = lambda x: x > var_3
    var_28 = module_0.split_by(var_26, criterion=var_27)
    var_29 = list(var_28)
    var_30 = [var_8, var_25, var_2]
    var_31 = lambda x: x > var_3
    var_32 = module_0.split_by(var_30, var_8, criterion=var_31)
    var_33 = list(var_32)
    var_34 = [var_8, var_25, var_2]
    var_35 = lambda x: x > var_0
    var_36 = module_0.split_by(var_34, criterion=var_35)
    var_37 = list(var_36)
    var_38 = [var_8, var_25, var_2]
    var_39 = module_0.split_by(var_38, var_8, separator=var_3)
    var_40 = list(var_39)
    var_41 = 1
    var_42 = 2
    var_43 = 3
    var_44 = [var_41, var_42, var_43]
    var_45 = 0
    var_46 = lambda x: x > var_45
    var_47 = module_0.split_by(var_44, criterion=var_46, separator=var_45)
    var_48 = list(var_47)
    var_49 = 1
    var_50 = 2
    var_51 = 3
    var_52 = [var_49, var_50, var_51]
    var_53 = module_0.split_by(var_52)
    var_54 = list(var_53)



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 2
    var_2 = 0
    var_3 = 10
    var_4 = -1



# Parsed testcases at query #4
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = 1
    var_2 = 2
    var_3 = 4
    var_4 = 5
    var_5 = [var_1, var_2, var_0, var_3, var_4]
    var_6 = module_0.take(var_0, var_5)
    var_7 = list(var_6)
    var_8 = 0
    var_9 = [var_1, var_2, var_0]
    var_10 = module_0.take(var_8, var_9)
    var_11 = list(var_10)
    var_12 = [var_1, var_2, var_0]
    var_13 = module_0.take(var_4, var_12)
    var_14 = list(var_13)
    var_15 = 10
    var_16 = range(var_15)
    var_17 = -1
    var_18 = 1
    var_19 = 2
    var_20 = 3
    var_21 = [var_18, var_19, var_20]
    var_22 = module_0.take(var_17, var_21)
    var_23 = list(var_22)
    var_24 = []
    var_25 = module_0.take(var_21, var_24)
    var_26 = list(var_25)
    var_27 = 'hello'
    var_28 = module_0.take(var_17, var_27)
    var_29 = list(var_28)



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = -1
    var_4 = -2
    var_5 = 0
    var_6 = 5
    var_7 = 100



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 1
    var_2 = 10
    var_3 = 2
    var_4 = 0
    var_5 = -1
    var_6 = 3



# Parsed testcases at query #7
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = 10
    var_2 = range(var_1)
    var_3 = module_0.chunk(var_0, var_2)
    var_4 = list(var_3)
    var_5 = 20
    var_6 = range(var_1)
    var_7 = module_0.chunk(var_5, var_6)
    var_8 = list(var_7)
    var_9 = 1
    var_10 = 5
    var_11 = range(var_10)
    var_12 = module_0.chunk(var_9, var_11)
    var_13 = list(var_12)
    var_14 = []
    var_15 = module_0.chunk(var_0, var_14)
    var_16 = list(var_15)
    var_17 = range(var_10)
    var_18 = module_0.chunk(var_10, var_17)
    var_19 = list(var_18)
    var_20 = 2
    var_21 = 'a'
    var_22 = 'b'
    var_23 = 'c'
    var_24 = 'd'
    var_25 = 'e'
    var_26 = [var_21, var_22, var_23, var_24, var_25]
    var_27 = module_0.chunk(var_20, var_26)
    var_28 = list(var_27)
    var_29 = range(var_1)
    var_30 = 0
    var_31 = 10
    var_32 = range(var_31)
    var_33 = module_0.chunk(var_30, var_32)
    var_34 = list(var_33)
    var_35 = -1
    var_36 = 10
    var_37 = range(var_36)
    var_38 = module_0.chunk(var_35, var_37)
    var_39 = list(var_38)



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 0
    var_4 = 0
    var_5 = 5
    var_6 = -6
    var_7 = 3
    var_8 = 3
    var_9 = 100



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = -1
    var_4 = -2
    var_5 = 0
    var_6 = 5



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 0
    var_3 = 10
    var_4 = 2
    var_5 = 3
    var_6 = -1



# Parsed testcases at query #11
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = 1
    var_2 = 2
    var_3 = 4
    var_4 = 5
    var_5 = [var_1, var_2, var_0, var_3, var_4]
    var_6 = module_0.drop(var_0, var_5)
    var_7 = list(var_6)
    var_8 = 0
    var_9 = [var_1, var_2, var_0]
    var_10 = module_0.drop(var_8, var_9)
    var_11 = list(var_10)
    var_12 = [var_1, var_2, var_0]
    var_13 = module_0.drop(var_4, var_12)
    var_14 = list(var_13)
    var_15 = 10
    var_16 = range(var_15)
    var_17 = range(var_4)
    var_18 = module_0.drop(var_15, var_17)
    var_19 = list(var_18)
    var_20 = -1
    var_21 = 1
    var_22 = 2
    var_23 = 3
    var_24 = [var_21, var_22, var_23]
    var_25 = module_0.drop(var_20, var_24)
    var_26 = list(var_25)
    var_27 = []
    var_28 = module_0.drop(var_24, var_27)
    var_29 = list(var_28)



# Parsed testcases at query #12
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = 3
    var_7 = lambda x: x == var_6
    var_8 = 1
    var_9 = 2
    var_10 = 4
    var_11 = [var_8, var_9, var_6, var_10, var_0]
    var_12 = module_0.drop_until(var_7, var_11)
    var_13 = list(var_12)
    var_14 = 0
    var_15 = lambda x: x % var_9 == var_14
    var_16 = 6
    var_17 = 7
    var_18 = [var_8, var_6, var_0, var_16, var_17]
    var_19 = module_0.drop_until(var_15, var_18)
    var_20 = list(var_19)
    var_21 = lambda x: x > var_2
    var_22 = range(var_0)
    var_23 = module_0.drop_until(var_21, var_22)
    var_24 = list(var_23)
    var_25 = lambda x: x == var_8
    var_26 = [var_8, var_9, var_6]
    var_27 = module_0.drop_until(var_25, var_26)
    var_28 = list(var_27)
    var_29 = lambda x: x > var_14
    var_30 = []
    var_31 = module_0.drop_until(var_29, var_30)
    var_32 = list(var_31)
    var_33 = lambda x: x.value == var_9



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -5
    var_3 = 1
    var_4 = 11
    var_5 = 5
    var_6 = -3
    var_7 = 3
    var_8 = 2
    var_9 = -1
    var_10 = -5
    var_11 = -2
    var_12 = -1
    var_13 = -1
    var_14 = 100



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 2
    var_2 = 1
    var_3 = 10
    var_4 = 0
    var_5 = -1
    var_6 = 20
    var_7 = 3



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 1
    var_2 = 10
    var_3 = 2
    var_4 = 0
    var_5 = -1
    var_6 = 1000



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 2



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = -5
    var_3 = 1
    var_4 = -3
    var_5 = 3
    var_6 = 10
    var_7 = 2
    var_8 = 11
    var_9 = 100
    var_10 = -1
    var_11 = -2
    var_12 = -1



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = 2
    var_3 = 10
    var_4 = -11



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = -1
    var_4 = -2
    var_5 = 0
    var_6 = 5
    var_7 = -1



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 5
    var_2 = 0
    var_3 = 2
    var_4 = 10
    var_5 = -11



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 5
    var_2 = 15
    var_3 = 0
    var_4 = 20
    var_5 = 2
    var_6 = 20
    var_7 = -20



# Parsed testcases at query #22
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 10
    var_4 = var_2[var_3]
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = 4
    var_9 = 5
    var_10 = [var_5, var_6, var_7, var_8, var_9]
    var_11 = module_0.LazyList(var_10)
    var_12 = [var_5, var_6, var_7]
    var_13 = module_0.LazyList(var_12)
    var_14 = list(var_13)



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = -1
    var_4 = -2
    var_5 = 0



# Parsed testcases at query #24
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = 3
    var_3 = 0
    var_4 = lambda x: x % var_2 == var_3
    var_5 = module_0.split_by(var_1, criterion=var_4)
    var_6 = list(var_5)
    var_7 = range(var_0)
    var_8 = True
    var_9 = lambda x: x % var_2 == var_3
    var_10 = module_0.split_by(var_7, var_8, criterion=var_9)
    var_11 = list(var_10)
    var_12 = ' Split by: '
    var_13 = ' '
    var_14 = module_0.split_by(var_12, separator=var_13)
    var_15 = list(var_14)
    var_16 = module_0.split_by(var_12, var_8, separator=var_13)
    var_17 = list(var_16)
    var_18 = []
    var_19 = lambda x: x % var_2 == var_3
    var_20 = module_0.split_by(var_18, criterion=var_19)
    var_21 = list(var_20)
    var_22 = []
    var_23 = module_0.split_by(var_22, var_8, separator=var_13)
    var_24 = list(var_23)
    var_25 = 6
    var_26 = 9
    var_27 = [var_2, var_25, var_26]
    var_28 = lambda x: x % var_2 == var_3
    var_29 = module_0.split_by(var_27, criterion=var_28)
    var_30 = list(var_29)
    var_31 = [var_2, var_25, var_26]
    var_32 = lambda x: x % var_2 == var_3
    var_33 = module_0.split_by(var_31, var_8, criterion=var_32)
    var_34 = list(var_33)
    var_35 = 2
    var_36 = 4
    var_37 = 5
    var_38 = [var_8, var_35, var_36, var_37]
    var_39 = lambda x: x % var_2 == var_3
    var_40 = module_0.split_by(var_38, criterion=var_39)
    var_41 = list(var_40)
    var_42 = [var_8, var_35, var_36, var_37]
    var_43 = module_0.split_by(var_42, var_8, separator=var_13)
    var_44 = list(var_43)
    var_45 = 1
    var_46 = 2
    var_47 = 3
    var_48 = [var_45, var_46, var_47]
    var_49 = 0
    var_50 = lambda x: x % var_47 == var_49
    var_51 = ' '
    var_52 = module_0.split_by(var_48, criterion=var_50, separator=var_51)
    var_53 = list(var_52)
    var_54 = 1
    var_55 = 2
    var_56 = 3
    var_57 = [var_54, var_55, var_56]
    var_58 = module_0.split_by(var_57)
    var_59 = list(var_58)



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 2
    var_2 = 0
    var_3 = 10
    var_4 = -1



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -5
    var_3 = 1
    var_4 = -5
    var_5 = 5
    var_6 = 2
    var_7 = 3
    var_8 = -1
    var_9 = -2
    var_10 = -1



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -5
    var_3 = 1
    var_4 = -5
    var_5 = 5
    var_6 = 2
    var_7 = 3
    var_8 = -1
    var_9 = -2
    var_10 = -1
    var_11 = 100



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 1
    var_2 = 0
    var_3 = 10
    var_4 = 2
    var_5 = -1



# Parsed testcases at query #29
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = lambda x: x * var_1
    var_7 = module_0.MapList(var_6, var_5)



# Parsed testcases at query #30
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)



