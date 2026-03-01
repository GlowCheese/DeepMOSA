####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = 5
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = module_0.drop(var_0, var_6)
    var_8 = list(var_7)
    var_9 = [var_1, var_2, var_3, var_4, var_5]
    var_10 = module_0.drop(var_3, var_9)
    var_11 = list(var_10)
    var_12 = [var_1, var_2, var_3, var_4, var_5]
    var_13 = module_0.drop(var_5, var_12)
    var_14 = list(var_13)
    var_15 = 10
    var_16 = [var_1, var_2, var_3, var_4, var_5]
    var_17 = module_0.drop(var_15, var_16)
    var_18 = list(var_17)
    var_19 = []
    var_20 = module_0.drop(var_3, var_19)
    var_21 = list(var_20)
    var_22 = range(var_15)
    var_23 = -1
    var_24 = 1
    var_25 = 2
    var_26 = 3
    var_27 = [var_24, var_25, var_26]
    var_28 = module_0.drop(var_23, var_27)
    var_29 = list(var_28)



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
    var_12 = 'Split by.'
    var_13 = '.'
    var_14 = module_0.split_by(var_12, separator=var_13)
    var_15 = list(var_14)
    var_16 = module_0.split_by(var_12, var_8, separator=var_13)
    var_17 = list(var_16)
    var_18 = 'a.b.c'
    var_19 = module_0.split_by(var_18, separator=var_13)
    var_20 = list(var_19)
    var_21 = module_0.split_by(var_18, var_8, separator=var_13)
    var_22 = list(var_21)
    var_23 = 2
    var_24 = [var_8, var_23, var_2]
    var_25 = lambda x: x > var_0
    var_26 = module_0.split_by(var_24, criterion=var_25)
    var_27 = list(var_26)
    var_28 = [var_8, var_23, var_2]
    var_29 = module_0.split_by(var_28, separator=var_0)
    var_30 = list(var_29)
    var_31 = [var_8, var_23, var_2]
    var_32 = lambda x: x < var_0
    var_33 = module_0.split_by(var_31, criterion=var_32)
    var_34 = list(var_33)
    var_35 = [var_8, var_23, var_2]
    var_36 = lambda x: x < var_0
    var_37 = module_0.split_by(var_35, var_8, criterion=var_36)
    var_38 = list(var_37)
    var_39 = []
    var_40 = lambda x: x % var_2 == var_3
    var_41 = module_0.split_by(var_39, criterion=var_40)
    var_42 = list(var_41)
    var_43 = []
    var_44 = module_0.split_by(var_43, var_8, separator=var_13)
    var_45 = list(var_44)
    var_46 = 1
    var_47 = 2
    var_48 = 3
    var_49 = [var_46, var_47, var_48]
    var_50 = lambda x: x > var_46
    var_51 = '.'
    var_52 = module_0.split_by(var_49, criterion=var_50, separator=var_51)
    var_53 = list(var_52)
    var_54 = 1
    var_55 = 2
    var_56 = 3
    var_57 = [var_54, var_55, var_56]
    var_58 = module_0.split_by(var_57)
    var_59 = list(var_58)



# Parsed testcases at query #3
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



# Parsed testcases at query #4
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = 5
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = module_0.drop(var_0, var_6)
    var_8 = list(var_7)
    var_9 = [var_1, var_2, var_3, var_4, var_5]
    var_10 = module_0.drop(var_3, var_9)
    var_11 = list(var_10)
    var_12 = [var_1, var_2, var_3, var_4, var_5]
    var_13 = module_0.drop(var_5, var_12)
    var_14 = list(var_13)
    var_15 = 10
    var_16 = [var_1, var_2, var_3, var_4, var_5]
    var_17 = module_0.drop(var_15, var_16)
    var_18 = list(var_17)
    var_19 = []
    var_20 = module_0.drop(var_3, var_19)
    var_21 = list(var_20)
    var_22 = range(var_15)
    var_23 = -1
    var_24 = 1
    var_25 = 2
    var_26 = 3
    var_27 = [var_24, var_25, var_26]
    var_28 = module_0.drop(var_23, var_27)
    var_29 = list(var_28)



# Parsed testcases at query #5
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = 10
    var_2 = range(var_1)
    var_3 = module_0.chunk(var_0, var_2)
    var_4 = list(var_3)
    var_5 = 1
    var_6 = 5
    var_7 = range(var_6)
    var_8 = module_0.chunk(var_5, var_7)
    var_9 = list(var_8)
    var_10 = range(var_6)
    var_11 = module_0.chunk(var_6, var_10)
    var_12 = list(var_11)
    var_13 = []
    var_14 = module_0.chunk(var_0, var_13)
    var_15 = list(var_14)
    var_16 = 2
    var_17 = [var_5, var_16, var_0]
    var_18 = module_0.chunk(var_0, var_17)
    var_19 = list(var_18)
    var_20 = range(var_6)
    var_21 = module_0.chunk(var_1, var_20)
    var_22 = list(var_21)
    var_23 = 0
    var_24 = 10
    var_25 = range(var_24)
    var_26 = module_0.chunk(var_23, var_25)
    var_27 = list(var_26)
    var_28 = -1
    var_29 = 10
    var_30 = range(var_29)
    var_31 = module_0.chunk(var_28, var_30)
    var_32 = list(var_31)
    var_33 = 3.5
    var_34 = 10
    var_35 = range(var_34)
    var_36 = module_0.chunk(var_33, var_35)
    var_37 = list(var_36)



# Parsed testcases at query #6
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
    var_15 = lambda x: x < var_14
    var_16 = [var_8, var_9, var_6]
    var_17 = module_0.drop_until(var_15, var_16)
    var_18 = list(var_17)
    var_19 = lambda x: x > var_14
    var_20 = []
    var_21 = module_0.drop_until(var_19, var_20)
    var_22 = list(var_21)
    var_23 = lambda x: x >= var_14
    var_24 = [var_8, var_9, var_6]
    var_25 = module_0.drop_until(var_23, var_24)
    var_26 = list(var_25)
    var_27 = lambda x: x < var_14
    var_28 = [var_8, var_9, var_6]
    var_29 = module_0.drop_until(var_27, var_28)
    var_30 = list(var_29)
    var_31 = lambda x: x == var_8
    var_32 = [var_8, var_9, var_6]
    var_33 = module_0.drop_until(var_31, var_32)
    var_34 = list(var_33)
    var_35 = lambda x: x == var_6
    var_36 = [var_8, var_9, var_6]
    var_37 = module_0.drop_until(var_35, var_36)
    var_38 = list(var_37)
    var_39 = lambda x: x.value > var_8



# Parsed testcases at query #7
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = range(var_1)
    var_3 = module_0.take(var_0, var_2)
    var_4 = list(var_3)
    var_5 = 0
    var_6 = range(var_1)
    var_7 = module_0.take(var_5, var_6)
    var_8 = list(var_7)
    var_9 = range(var_0)
    var_10 = module_0.take(var_1, var_9)
    var_11 = list(var_10)
    var_12 = 3
    var_13 = 1
    var_14 = 2
    var_15 = 4
    var_16 = [var_13, var_14, var_12, var_15, var_0]
    var_17 = module_0.take(var_12, var_16)
    var_18 = list(var_17)
    var_19 = [var_13, var_14, var_12]
    var_20 = module_0.take(var_5, var_19)
    var_21 = list(var_20)
    var_22 = [var_13, var_14, var_12]
    var_23 = module_0.take(var_1, var_22)
    var_24 = list(var_23)
    var_25 = 'hello'
    var_26 = module_0.take(var_12, var_25)
    var_27 = list(var_26)
    var_28 = module_0.take(var_5, var_25)
    var_29 = list(var_28)
    var_30 = module_0.take(var_1, var_25)
    var_31 = list(var_30)
    var_32 = -1
    var_33 = 10
    var_34 = range(var_33)
    var_35 = module_0.take(var_32, var_34)
    var_36 = list(var_35)



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 100
    var_4 = -100



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -5
    var_3 = 1
    var_4 = -5
    var_5 = 5
    var_6 = -5
    var_7 = 2
    var_8 = 3
    var_9 = -1
    var_10 = -1



# Parsed testcases at query #10
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
    var_15 = lambda x: x > var_14
    var_16 = []
    var_17 = module_0.drop_until(var_15, var_16)
    var_18 = list(var_17)
    var_19 = lambda x: x > var_2
    var_20 = range(var_0)
    var_21 = module_0.drop_until(var_19, var_20)
    var_22 = list(var_21)
    var_23 = lambda x: x == var_8
    var_24 = [var_8, var_9, var_6]
    var_25 = module_0.drop_until(var_23, var_24)
    var_26 = list(var_25)
    var_27 = lambda x: x.value > var_9
    var_28 = 'c'
    var_29 = lambda x: x == var_28
    var_30 = 'abcdef'
    var_31 = module_0.drop_until(var_29, var_30)
    var_32 = list(var_31)



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 5
    var_4 = 10
    var_5 = -10



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 2
    var_2 = 0
    var_3 = 10
    var_4 = -1



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -5
    var_3 = 1
    var_4 = 5
    var_5 = -3
    var_6 = 3
    var_7 = 2
    var_8 = 11
    var_9 = 100
    var_10 = -1
    var_11 = -2
    var_12 = -1



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = list(var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 10
    var_6 = -10



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 2
    var_2 = 0
    var_3 = 10
    var_4 = -1



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 10
    var_4 = -10



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 10
    var_4 = -10



# Parsed testcases at query #18
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = []
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = 0
    var_6 = lambda x: x >= var_5
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = [var_7, var_8, var_9]
    var_11 = module_0.drop_until(var_6, var_10)
    var_12 = list(var_11)
    var_13 = 10
    var_14 = lambda x: x > var_13
    var_15 = [var_7, var_8, var_9]
    var_16 = module_0.drop_until(var_14, var_15)
    var_17 = list(var_16)
    var_18 = lambda x: x > var_0
    var_19 = 4
    var_20 = 6
    var_21 = 7
    var_22 = 8
    var_23 = [var_7, var_8, var_9, var_19, var_0, var_20, var_21, var_22]
    var_24 = module_0.drop_until(var_18, var_23)
    var_25 = list(var_24)
    var_26 = lambda x: x > var_5
    var_27 = [var_7, var_8, var_9]
    var_28 = module_0.drop_until(var_26, var_27)
    var_29 = list(var_28)
    var_30 = lambda x: x > var_0
    var_31 = [var_7, var_8, var_9, var_19, var_0, var_20]
    var_32 = module_0.drop_until(var_30, var_31)
    var_33 = list(var_32)
    var_34 = 'a'
    var_35 = lambda x: x.startswith(var_34)
    var_36 = 'b'
    var_37 = 'c'
    var_38 = 'd'
    var_39 = 'e'
    var_40 = [var_36, var_37, var_34, var_38, var_39]
    var_41 = module_0.drop_until(var_35, var_40)
    var_42 = list(var_41)
    var_43 = range(var_13)
    var_44 = lambda x: x == var_0
    var_45 = 'l'
    var_46 = lambda x: x == var_45
    var_47 = 'hello world'
    var_48 = module_0.drop_until(var_46, var_47)
    var_49 = list(var_48)



# Parsed testcases at query #19
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
    var_8 = lambda x: x ** var_1
    var_9 = module_0.MapList(var_8, var_5)
    var_10 = 'a'
    var_11 = 'b'
    var_12 = 'c'
    var_13 = [var_10, var_11, var_12]
    var_14 = lambda x: x.upper()
    var_15 = module_0.MapList(var_14, var_13)
    var_16 = (var_0, var_1)
    var_17 = (var_2, var_3)
    var_18 = 6
    var_19 = (var_4, var_18)
    var_20 = [var_16, var_17, var_19]
    var_21 = 0
    var_22 = lambda x: x[var_21] + x[var_0]
    var_23 = module_0.MapList(var_22, var_20)



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
    var_10 = -10
    var_11 = var_7[var_10]



# Parsed testcases at query #21
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
    var_8 = lambda x: str(x)
    var_9 = module_0.MapList(var_8, var_5)



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
    var_8 = range(var_2)
    var_9 = module_0.drop_until(var_7, var_8)
    var_10 = list(var_9)
    var_11 = 0
    var_12 = lambda x: x < var_11
    var_13 = range(var_2)
    var_14 = module_0.drop_until(var_12, var_13)
    var_15 = list(var_14)
    var_16 = lambda x: x > var_0
    var_17 = []
    var_18 = module_0.drop_until(var_16, var_17)
    var_19 = list(var_18)
    var_20 = 100
    var_21 = lambda x: x > var_20
    var_22 = range(var_2)
    var_23 = module_0.drop_until(var_21, var_22)
    var_24 = list(var_23)
    var_25 = lambda x: x == var_11
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



# Parsed testcases at query #23
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = 100
    var_7 = lambda x: x > var_6
    var_8 = range(var_2)
    var_9 = module_0.drop_until(var_7, var_8)
    var_10 = list(var_9)
    var_11 = 0
    var_12 = lambda x: x >= var_11
    var_13 = range(var_2)
    var_14 = module_0.drop_until(var_12, var_13)
    var_15 = list(var_14)
    var_16 = lambda x: x > var_0
    var_17 = []
    var_18 = module_0.drop_until(var_16, var_17)
    var_19 = list(var_18)
    var_20 = range(var_2)
    var_21 = [TestObj(i) for i in var_20]
    var_22 = lambda x: x.val > var_0
    var_23 = module_0.drop_until(var_22, var_21)
    var_24 = list(var_23)
    var_25 = 'c'
    var_26 = lambda x: x == var_25
    var_27 = 'abcdefg'
    var_28 = module_0.drop_until(var_26, var_27)
    var_29 = list(var_28)



# Parsed testcases at query #24
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
    var_15 = lambda x: x < var_14
    var_16 = [var_8, var_9, var_6]
    var_17 = module_0.drop_until(var_15, var_16)
    var_18 = list(var_17)
    var_19 = lambda x: x > var_14
    var_20 = []
    var_21 = module_0.drop_until(var_19, var_20)
    var_22 = list(var_21)
    var_23 = lambda x: x >= var_14
    var_24 = [var_8, var_9, var_6]
    var_25 = module_0.drop_until(var_23, var_24)
    var_26 = list(var_25)
    var_27 = lambda x: x < var_14
    var_28 = [var_8, var_9, var_6]
    var_29 = module_0.drop_until(var_27, var_28)
    var_30 = list(var_29)
    var_31 = lambda x: x == var_8
    var_32 = [var_8, var_9, var_6]
    var_33 = module_0.drop_until(var_31, var_32)
    var_34 = list(var_33)
    var_35 = lambda x: x == var_6
    var_36 = [var_8, var_9, var_6]
    var_37 = module_0.drop_until(var_35, var_36)
    var_38 = list(var_37)
    var_39 = lambda x: x.val > var_8



# Parsed testcases at query #25
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
    var_7 = 20
    var_8 = 30
    var_9 = 40
    var_10 = 50
    var_11 = [var_5, var_7, var_8, var_9, var_10]
    var_12 = 100
    var_13 = range(var_12)
    var_14 = module_0.LazyList(var_13)
    var_15 = var_14.list
    var_16 = len(var_15)
    assert var_16 == 0
    var_17 = 5
    var_18 = var_14[var_17]
    var_19 = var_14.list
    var_20 = len(var_19)
    assert var_20 == 6
    var_21 = var_14[var_5]
    var_22 = var_14.list
    var_23 = len(var_22)
    assert var_23 == 11
    var_24 = range(var_12)
    var_25 = module_0.LazyList(var_24)
    var_26 = var_25.list
    var_27 = len(var_26)
    assert var_27 == 0
    var_28 = var_25[var_17:var_5]
    var_29 = var_25.list
    var_30 = len(var_29)
    assert var_30 == 10
    var_31 = range(var_12)
    var_32 = module_0.LazyList(var_31)
    var_33 = var_32.list
    var_34 = len(var_33)
    assert var_34 == 6
    var_35 = var_32[var_5]
    var_36 = var_32.list
    var_37 = len(var_36)
    assert var_37 == 11
    var_38 = range(var_12)
    var_39 = module_0.LazyList(var_38)
    var_40 = var_39.list
    var_41 = len(var_40)
    assert var_41 == 6
    var_42 = var_39[var_17:var_5]
    var_43 = var_39.list
    var_44 = len(var_43)
    assert var_44 == 10



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
    var_8 = range(var_2)
    var_9 = module_0.drop_until(var_7, var_8)
    var_10 = list(var_9)
    var_11 = 0
    var_12 = lambda x: x < var_11
    var_13 = range(var_2)
    var_14 = module_0.drop_until(var_12, var_13)
    var_15 = list(var_14)
    var_16 = lambda x: x > var_0
    var_17 = []
    var_18 = module_0.drop_until(var_16, var_17)
    var_19 = list(var_18)
    var_20 = lambda x: x >= var_11
    var_21 = range(var_2)
    var_22 = module_0.drop_until(var_20, var_21)
    var_23 = list(var_22)
    var_24 = lambda x: x < var_11
    var_25 = range(var_2)
    var_26 = module_0.drop_until(var_24, var_25)
    var_27 = list(var_26)
    var_28 = 'b'
    var_29 = lambda x: x.startswith(var_28)
    var_30 = 'apple'
    var_31 = 'banana'
    var_32 = 'cherry'
    var_33 = [var_30, var_31, var_32]
    var_34 = module_0.drop_until(var_29, var_33)
    var_35 = list(var_34)
    var_36 = range(var_2)
    var_37 = lambda x: x == var_0



# Parsed testcases at query #27
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = []
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = 10
    var_6 = lambda x: x > var_5
    var_7 = range(var_0)
    var_8 = module_0.drop_until(var_6, var_7)
    var_9 = list(var_8)
    var_10 = 0
    var_11 = lambda x: x >= var_10
    var_12 = range(var_0)
    var_13 = module_0.drop_until(var_11, var_12)
    var_14 = list(var_13)
    var_15 = lambda x: x > var_0
    var_16 = range(var_5)
    var_17 = module_0.drop_until(var_15, var_16)
    var_18 = list(var_17)
    var_19 = lambda x: x == var_10
    var_20 = range(var_0)
    var_21 = module_0.drop_until(var_19, var_20)
    var_22 = list(var_21)
    var_23 = 4
    var_24 = lambda x: x == var_23
    var_25 = range(var_0)
    var_26 = module_0.drop_until(var_24, var_25)
    var_27 = list(var_26)
    var_28 = 2
    var_29 = lambda x: x % var_28 == var_10
    var_30 = 1
    var_31 = 3
    var_32 = 6
    var_33 = 7
    var_34 = 8
    var_35 = [var_30, var_31, var_0, var_32, var_33, var_34]
    var_36 = module_0.drop_until(var_29, var_35)
    var_37 = list(var_36)
    var_38 = 'c'
    var_39 = lambda x: x == var_38
    var_40 = 'abcdef'
    var_41 = module_0.drop_until(var_39, var_40)
    var_42 = list(var_41)



# Parsed testcases at query #28
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
    var_12 = []
    var_13 = lambda x: x * var_1
    var_14 = module_0.MapList(var_13, var_12)
    var_15 = 0
    var_16 = var_14[var_15]



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
    var_8 = 10
    var_9 = var_7[var_8]
    var_10 = -10
    var_11 = var_7[var_10]



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
    var_13 = 2
    var_14 = 3
    var_15 = [var_7, var_13, var_14]
    var_16 = module_0.LazyList(var_15)
    var_17 = var_16[var_13]



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 10
    var_4 = -11



# Parsed testcases at query #32
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 10
    var_4 = var_2[var_3]
    var_5 = 1
    var_6 = 4
    var_7 = 9
    var_8 = 16
    var_9 = 25
    var_10 = [var_5, var_6, var_7, var_8, var_9]
    var_11 = 5
    var_12 = range(var_11)
    var_13 = module_0.LazyList(var_12)
    var_14 = var_13[var_6]
    var_15 = len(var_13)
    assert var_15 == 5
    var_16 = []
    var_17 = module_0.LazyList(var_16)
    var_18 = 0
    var_19 = var_17[var_18]



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 5
    var_2 = 15
    var_3 = 0
    var_4 = 20
    var_5 = 2
    var_6 = 20
    var_7 = -21



# Parsed testcases at query #34
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = []
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = 0
    var_6 = lambda x: x > var_5
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = 4
    var_11 = [var_7, var_8, var_9, var_10]
    var_12 = module_0.drop_until(var_6, var_11)
    var_13 = list(var_12)
    var_14 = 10
    var_15 = lambda x: x > var_14
    var_16 = [var_7, var_8, var_9, var_10]
    var_17 = module_0.drop_until(var_15, var_16)
    var_18 = list(var_17)
    var_19 = lambda x: x > var_0
    var_20 = 6
    var_21 = 7
    var_22 = 8
    var_23 = [var_7, var_8, var_9, var_20, var_21, var_22]
    var_24 = module_0.drop_until(var_19, var_23)
    var_25 = list(var_24)
    var_26 = lambda x: x > var_5
    var_27 = [var_7, var_8, var_9, var_10]
    var_28 = module_0.drop_until(var_26, var_27)
    var_29 = list(var_28)
    var_30 = lambda x: x > var_0
    var_31 = [var_7, var_8, var_9, var_10, var_20]
    var_32 = module_0.drop_until(var_30, var_31)
    var_33 = list(var_32)
    var_34 = lambda x: x > var_8
    var_35 = [var_7, var_8, var_9, var_10, var_0]
    var_36 = module_0.drop_until(var_34, var_35)
    var_37 = list(var_36)
    var_38 = 'a'
    var_39 = lambda x: x.startswith(var_38)
    var_40 = 'b'
    var_41 = 'c'
    var_42 = 'apple'
    var_43 = 'banana'
    var_44 = [var_40, var_41, var_42, var_43]
    var_45 = module_0.drop_until(var_39, var_44)
    var_46 = list(var_45)
    var_47 = range(var_14)
    var_48 = lambda x: x > var_0



# Parsed testcases at query #35
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 10
    var_4 = var_2[var_3]
    var_5 = 1
    var_6 = 4
    var_7 = 9
    var_8 = 16
    var_9 = 25
    var_10 = [var_5, var_6, var_7, var_8, var_9]
    var_11 = 5
    var_12 = range(var_11)
    var_13 = module_0.LazyList(var_12)
    var_14 = list(var_13)



# Parsed testcases at query #36
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 10
    var_4 = var_2[var_3]
    var_5 = range(var_3)
    var_6 = 2
    var_7 = 0
    var_8 = 'hello'
    var_9 = module_0.LazyList(var_8)



# Parsed testcases at query #37
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
    var_7 = 3
    var_8 = range(var_7)
    var_9 = module_0.LazyList(var_8)
    var_10 = list(var_9)



# Parsed testcases at query #38
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
    var_13 = iter(var_12)
    var_14 = module_0.LazyList(var_13)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
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
    var_22 = range(var_14)
    var_23 = range(var_3, var_14)
    var_24 = list(var_23)
    var_25 = -1
    var_26 = 1
    var_27 = 2
    var_28 = 3
    var_29 = [var_26, var_27, var_28]
    var_30 = module_0.drop(var_25, var_29)
    var_31 = list(var_30)



# Parsed testcases at query #2
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



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
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
    var_18 = -1
    var_19 = 1
    var_20 = 2
    var_21 = 3
    var_22 = 4
    var_23 = [var_19, var_20, var_21, var_22]
    var_24 = module_0.drop(var_18, var_23)
    var_25 = list(var_24)
    var_26 = 5
    var_27 = []
    var_28 = module_0.drop(var_26, var_27)
    var_29 = list(var_28)
    var_30 = [var_19, var_20, var_21, var_22]



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



# Parsed testcases at query #3
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
    var_6 = module_0.drop(var_0, var_5)
    var_7 = list(var_6)
    var_8 = [var_1, var_2, var_0, var_3, var_4]
    var_9 = module_0.drop(var_4, var_8)
    var_10 = list(var_9)
    var_11 = 0
    var_12 = [var_1, var_2, var_0, var_3, var_4]
    var_13 = module_0.drop(var_11, var_12)
    var_14 = list(var_13)
    var_15 = []
    var_16 = module_0.drop(var_0, var_15)
    var_17 = list(var_16)
    var_18 = 10
    var_19 = [var_1, var_2, var_0]
    var_20 = module_0.drop(var_18, var_19)
    var_21 = list(var_20)
    var_22 = range(var_18)
    var_23 = -1
    var_24 = 1
    var_25 = 2
    var_26 = 3
    var_27 = [var_24, var_25, var_26]
    var_28 = module_0.drop(var_23, var_27)
    var_29 = list(var_28)



# Parsed testcases at query #5
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = 10
    var_2 = range(var_1)
    var_3 = module_0.chunk(var_0, var_2)
    var_4 = list(var_3)
    var_5 = 9
    var_6 = range(var_5)
    var_7 = module_0.chunk(var_0, var_6)
    var_8 = list(var_7)
    var_9 = []
    var_10 = module_0.chunk(var_0, var_9)
    var_11 = list(var_10)
    var_12 = 1
    var_13 = 2
    var_14 = [var_12, var_13, var_0]
    var_15 = module_0.chunk(var_12, var_14)
    var_16 = list(var_15)
    var_17 = 5
    var_18 = range(var_17)
    var_19 = module_0.chunk(var_1, var_18)
    var_20 = list(var_19)
    var_21 = 0
    var_22 = 5
    var_23 = range(var_22)
    var_24 = module_0.chunk(var_21, var_23)
    var_25 = list(var_24)
    var_26 = -1
    var_27 = 5
    var_28 = range(var_27)
    var_29 = module_0.chunk(var_26, var_28)
    var_30 = list(var_29)



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 11
    var_3 = 2
    var_4 = 10
    var_5 = -10



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = -1
    var_4 = -2
    var_5 = 0
    var_6 = 100



# Parsed testcases at query #8
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
    var_13 = 100
    var_14 = range(var_13)
    var_15 = module_0.LazyList(var_14)
    var_16 = var_15.list
    var_17 = len(var_16)
    assert var_17 == 0
    var_18 = var_15[var_5]
    var_19 = var_15.list
    var_20 = len(var_19)
    assert var_20 == 11
    var_21 = 5
    var_22 = var_15[var_21]
    var_23 = var_15.list
    var_24 = len(var_23)
    assert var_24 == 11
    var_25 = 20
    var_26 = var_15[var_25]
    var_27 = var_15.list
    var_28 = len(var_27)
    assert var_28 == 21
    var_29 = range(var_13)
    var_30 = module_0.LazyList(var_29)
    var_31 = var_30.list
    var_32 = len(var_31)
    assert var_32 == 0
    var_33 = var_30[var_5:var_25]
    var_34 = var_30.list
    var_35 = len(var_34)
    assert var_35 == 20



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
    var_8 = range(var_2)
    var_9 = module_0.drop_until(var_7, var_8)
    var_10 = list(var_9)
    var_11 = 0
    var_12 = lambda x: x < var_11
    var_13 = range(var_2)
    var_14 = module_0.drop_until(var_12, var_13)
    var_15 = list(var_14)
    var_16 = lambda x: x > var_0
    var_17 = []
    var_18 = module_0.drop_until(var_16, var_17)
    var_19 = list(var_18)
    var_20 = lambda x: x >= var_11
    var_21 = range(var_2)
    var_22 = module_0.drop_until(var_20, var_21)
    var_23 = list(var_22)
    var_24 = lambda x: x > var_2
    var_25 = range(var_2)
    var_26 = module_0.drop_until(var_24, var_25)
    var_27 = list(var_26)
    var_28 = range(var_2)
    var_29 = [Custom(i) for i in var_28]
    var_30 = lambda x: x.val > var_0
    var_31 = module_0.drop_until(var_30, var_29)
    var_32 = list(var_31)
    var_33 = 6
    var_34 = range(var_33, var_2)
    var_35 = [Custom(i) for i in var_34]
    var_36 = 'c'
    var_37 = lambda x: x == var_36
    var_38 = 'abcdef'
    var_39 = module_0.drop_until(var_37, var_38)
    var_40 = list(var_39)



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 1
    var_2 = 0
    var_3 = 10
    var_4 = 2
    var_5 = -1
    var_6 = 3



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 11
    var_3 = 2
    var_4 = 10
    var_5 = -11



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 1
    var_2 = 10
    var_3 = 2
    var_4 = 10
    var_5 = -10



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 11
    var_3 = 2
    var_4 = 10
    var_5 = -11



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 10
    var_4 = -11



# Parsed testcases at query #15
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



# Parsed testcases at query #16
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = lambda acc, x: acc + x
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = [var_1, var_2, var_3, var_4]
    var_6 = 0
    var_7 = lambda acc, x: acc * x
    var_8 = [var_1, var_2, var_3, var_4]
    var_9 = lambda acc, x: acc + x
    var_10 = [var_1, var_2, var_3, var_4]
    var_11 = module_0.scanl(var_9, var_10)
    var_12 = list(var_11)
    var_13 = lambda acc, x: acc * x
    var_14 = [var_1, var_2, var_3, var_4]
    var_15 = module_0.scanl(var_13, var_14)
    var_16 = list(var_15)
    var_17 = lambda acc, x: x + acc
    var_18 = 'a'
    var_19 = 'b'
    var_20 = 'c'
    var_21 = 'd'
    var_22 = [var_18, var_19, var_20, var_21]
    var_23 = module_0.scanl(var_17, var_22)
    var_24 = list(var_23)
    var_25 = lambda acc, x: acc + x
    var_26 = []
    var_27 = lambda acc, x: acc + x
    var_28 = []
    var_29 = module_0.scanl(var_27, var_28)
    var_30 = list(var_29)
    var_31 = lambda acc, x: acc + x
    var_32 = 5
    var_33 = [var_32]
    var_34 = lambda acc, x: acc + x
    var_35 = [var_32]
    var_36 = module_0.scanl(var_34, var_35)
    var_37 = list(var_36)
    var_38 = lambda acc, x: acc + x
    var_39 = -1
    var_40 = -2
    var_41 = -3
    var_42 = -4
    var_43 = [var_39, var_40, var_41, var_42]
    var_44 = lambda acc, x: acc + str(x)
    var_45 = [var_1, var_2, var_3]
    var_46 = ''
    var_47 = lambda acc, x: acc + x
    var_48 = 6
    var_49 = lambda acc, x: acc + x
    var_50 = 1
    var_51 = 2
    var_52 = 3
    var_53 = [var_50, var_51, var_52]
    var_54 = 0
    var_55 = list(var_6)



# Parsed testcases at query #17
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
    var_23 = module_0.split_by(var_22, separator=var_13)
    var_24 = list(var_23)
    var_25 = range(var_0)
    var_26 = lambda x: x > var_0
    var_27 = module_0.split_by(var_25, criterion=var_26)
    var_28 = list(var_27)
    var_29 = range(var_0)
    var_30 = module_0.split_by(var_29, separator=var_0)
    var_31 = list(var_30)
    var_32 = 6
    var_33 = 9
    var_34 = [var_3, var_2, var_32, var_33]
    var_35 = lambda x: x % var_2 == var_3
    var_36 = module_0.split_by(var_34, criterion=var_35)
    var_37 = list(var_36)
    var_38 = [var_3, var_2, var_32, var_33]
    var_39 = lambda x: x % var_2 == var_3
    var_40 = module_0.split_by(var_38, var_8, criterion=var_39)
    var_41 = list(var_40)
    var_42 = [var_8, var_8, var_8]
    var_43 = module_0.split_by(var_42, separator=var_8)
    var_44 = list(var_43)
    var_45 = [var_8, var_8, var_8]
    var_46 = module_0.split_by(var_45, var_8, separator=var_8)
    var_47 = list(var_46)
    var_48 = 10
    var_49 = range(var_48)
    var_50 = module_0.split_by(var_49)
    var_51 = list(var_50)
    var_52 = 10
    var_53 = range(var_52)
    var_54 = 3
    var_55 = 0
    var_56 = lambda x: x % var_54 == var_55
    var_57 = ' '
    var_58 = module_0.split_by(var_53, criterion=var_56, separator=var_57)
    var_59 = list(var_58)



# Parsed testcases at query #18
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
    var_23 = lambda x: x % var_2 == var_3
    var_24 = module_0.split_by(var_22, var_8, criterion=var_23)
    var_25 = list(var_24)
    var_26 = 6
    var_27 = 9
    var_28 = [var_2, var_26, var_27]
    var_29 = lambda x: x % var_2 == var_3
    var_30 = module_0.split_by(var_28, criterion=var_29)
    var_31 = list(var_30)
    var_32 = [var_2, var_26, var_27]
    var_33 = lambda x: x % var_2 == var_3
    var_34 = module_0.split_by(var_32, var_8, criterion=var_33)
    var_35 = list(var_34)
    var_36 = 2
    var_37 = 4
    var_38 = 5
    var_39 = [var_8, var_36, var_37, var_38]
    var_40 = lambda x: x % var_2 == var_3
    var_41 = module_0.split_by(var_39, criterion=var_40)
    var_42 = list(var_41)
    var_43 = [var_8, var_36, var_37, var_38]
    var_44 = lambda x: x % var_2 == var_3
    var_45 = module_0.split_by(var_43, var_8, criterion=var_44)
    var_46 = list(var_45)
    var_47 = '..a..b..'
    var_48 = '.'
    var_49 = module_0.split_by(var_47, separator=var_48)
    var_50 = list(var_49)
    var_51 = module_0.split_by(var_47, var_8, separator=var_48)
    var_52 = list(var_51)
    var_53 = 1
    var_54 = 2
    var_55 = 3
    var_56 = [var_53, var_54, var_55]
    var_57 = 0
    var_58 = lambda x: x % var_55 == var_57
    var_59 = module_0.split_by(var_56, criterion=var_58, separator=var_55)
    var_60 = list(var_59)
    var_61 = 1
    var_62 = 2
    var_63 = 3
    var_64 = [var_61, var_62, var_63]
    var_65 = module_0.split_by(var_64)
    var_66 = list(var_65)



# Parsed testcases at query #19
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



# Parsed testcases at query #20
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = lambda x: x > var_0
    var_7 = []
    var_8 = module_0.drop_until(var_6, var_7)
    var_9 = list(var_8)
    var_10 = 0
    var_11 = lambda x: x >= var_10
    var_12 = range(var_2)
    var_13 = module_0.drop_until(var_11, var_12)
    var_14 = list(var_13)
    var_15 = 100
    var_16 = lambda x: x > var_15
    var_17 = range(var_2)
    var_18 = module_0.drop_until(var_16, var_17)
    var_19 = list(var_18)
    var_20 = range(var_2)
    var_21 = [TestObj(i) for i in var_20]
    var_22 = lambda x: x.val > var_0
    var_23 = module_0.drop_until(var_22, var_21)
    var_24 = list(var_23)
    var_25 = 'c'
    var_26 = lambda x: x == var_25
    var_27 = 'abcdefg'
    var_28 = module_0.drop_until(var_26, var_27)
    var_29 = list(var_28)
    var_30 = range(var_2)
    var_31 = lambda x: x > var_0



# Parsed testcases at query #21
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
    var_7 = []
    var_8 = lambda x: x > var_3
    var_9 = module_0.split_by(var_7, criterion=var_8)
    var_10 = list(var_9)
    var_11 = 1
    var_12 = 2
    var_13 = [var_11, var_12, var_2]
    var_14 = 5
    var_15 = lambda x: x > var_14
    var_16 = module_0.split_by(var_13, criterion=var_15)
    var_17 = list(var_16)
    var_18 = ' Split by: '
    var_19 = True
    var_20 = ' '
    var_21 = module_0.split_by(var_18, var_19, separator=var_20)
    var_22 = list(var_21)
    var_23 = 4
    var_24 = [var_19, var_12, var_2, var_19, var_23, var_19]
    var_25 = module_0.split_by(var_24, separator=var_19)
    var_26 = list(var_25)
    var_27 = []
    var_28 = module_0.split_by(var_27, separator=var_19)
    var_29 = list(var_28)
    var_30 = [var_19, var_19, var_12, var_19, var_19]
    var_31 = True
    var_32 = module_0.split_by(var_30, var_31, separator=var_31)
    var_33 = list(var_32)
    var_34 = [var_31, var_31, var_12, var_31, var_31]
    var_35 = False
    var_36 = module_0.split_by(var_34, var_35, separator=var_31)
    var_37 = list(var_36)
    var_38 = 1
    var_39 = 2
    var_40 = 3
    var_41 = [var_38, var_39, var_40]
    var_42 = 0
    var_43 = lambda x: x > var_42
    var_44 = module_0.split_by(var_41, criterion=var_43, separator=var_38)
    var_45 = list(var_44)
    var_46 = 1
    var_47 = 2
    var_48 = 3
    var_49 = [var_46, var_47, var_48]
    var_50 = module_0.split_by(var_49)
    var_51 = list(var_50)



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
    var_15 = lambda x: x < var_14
    var_16 = [var_8, var_9, var_6]
    var_17 = module_0.drop_until(var_15, var_16)
    var_18 = list(var_17)
    var_19 = lambda x: x > var_0
    var_20 = []
    var_21 = module_0.drop_until(var_19, var_20)
    var_22 = list(var_21)
    var_23 = lambda x: x >= var_14
    var_24 = [var_8, var_9, var_6]
    var_25 = module_0.drop_until(var_23, var_24)
    var_26 = list(var_25)
    var_27 = lambda x: x < var_14
    var_28 = [var_8, var_9, var_6]
    var_29 = module_0.drop_until(var_27, var_28)
    var_30 = list(var_29)
    var_31 = lambda x: x == var_8
    var_32 = [var_8, var_9, var_6]
    var_33 = module_0.drop_until(var_31, var_32)
    var_34 = list(var_33)
    var_35 = lambda x: x == var_6
    var_36 = [var_8, var_9, var_6]
    var_37 = module_0.drop_until(var_35, var_36)
    var_38 = list(var_37)
    var_39 = lambda x: x.value > var_9



# Parsed testcases at query #23
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
    var_42 = 1
    var_43 = 2
    var_44 = 3
    var_45 = [var_42, var_43, var_44]
    var_46 = 0
    var_47 = lambda x: x % var_44 == var_46
    var_48 = ' '
    var_49 = module_0.split_by(var_45, criterion=var_47, separator=var_48)
    var_50 = list(var_49)
    var_51 = 1
    var_52 = 2
    var_53 = 3
    var_54 = [var_51, var_52, var_53]
    var_55 = module_0.split_by(var_54)
    var_56 = list(var_55)



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
    var_12 = 'Split by:'
    var_13 = ' '
    var_14 = module_0.split_by(var_12, separator=var_13)
    var_15 = list(var_14)
    var_16 = ' Split by: '
    var_17 = module_0.split_by(var_16, var_8, separator=var_13)
    var_18 = list(var_17)
    var_19 = []
    var_20 = lambda x: x % var_2 == var_3
    var_21 = module_0.split_by(var_19, criterion=var_20)
    var_22 = list(var_21)
    var_23 = []
    var_24 = lambda x: x % var_2 == var_3
    var_25 = module_0.split_by(var_23, var_8, criterion=var_24)
    var_26 = list(var_25)
    var_27 = 6
    var_28 = 9
    var_29 = [var_3, var_2, var_27, var_28]
    var_30 = lambda x: x % var_2 == var_3
    var_31 = module_0.split_by(var_29, criterion=var_30)
    var_32 = list(var_31)
    var_33 = [var_3, var_2, var_27, var_28]
    var_34 = lambda x: x % var_2 == var_3
    var_35 = module_0.split_by(var_33, var_8, criterion=var_34)
    var_36 = list(var_35)
    var_37 = 2
    var_38 = 4
    var_39 = 5
    var_40 = [var_8, var_37, var_38, var_39]
    var_41 = lambda x: x % var_2 == var_3
    var_42 = module_0.split_by(var_40, criterion=var_41)
    var_43 = list(var_42)
    var_44 = [var_8, var_37, var_38, var_39]
    var_45 = lambda x: x % var_2 == var_3
    var_46 = module_0.split_by(var_44, var_8, criterion=var_45)
    var_47 = list(var_46)
    var_48 = 1
    var_49 = 2
    var_50 = 3
    var_51 = [var_48, var_49, var_50]
    var_52 = 0
    var_53 = lambda x: x % var_50 == var_52
    var_54 = module_0.split_by(var_51, criterion=var_53, separator=var_52)
    var_55 = list(var_54)
    var_56 = 1
    var_57 = 2
    var_58 = 3
    var_59 = [var_56, var_57, var_58]
    var_60 = module_0.split_by(var_59)
    var_61 = list(var_60)



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 1
    var_2 = 10
    var_3 = 2
    var_4 = 10
    var_5 = -10



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
    var_14 = 0
    var_15 = lambda x: x < var_14
    var_16 = [var_8, var_9, var_6]
    var_17 = module_0.drop_until(var_15, var_16)
    var_18 = list(var_17)
    var_19 = lambda x: x > var_0
    var_20 = []
    var_21 = module_0.drop_until(var_19, var_20)
    var_22 = list(var_21)
    var_23 = lambda x: x >= var_14
    var_24 = [var_8, var_9, var_6]
    var_25 = module_0.drop_until(var_23, var_24)
    var_26 = list(var_25)
    var_27 = lambda x: x < var_14
    var_28 = [var_8, var_9, var_6]
    var_29 = module_0.drop_until(var_27, var_28)
    var_30 = list(var_29)
    var_31 = lambda x: x == var_8
    var_32 = [var_8, var_9, var_6]
    var_33 = module_0.drop_until(var_31, var_32)
    var_34 = list(var_33)
    var_35 = lambda x: x == var_6
    var_36 = [var_8, var_9, var_6]
    var_37 = module_0.drop_until(var_35, var_36)
    var_38 = list(var_37)
    var_39 = lambda x: x.value == var_9



# Parsed testcases at query #27
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
    var_34 = module_0.split_by(var_33, separator=var_25)
    var_35 = list(var_34)
    var_36 = 6
    var_37 = 9
    var_38 = [var_25, var_24, var_36, var_37]
    var_39 = lambda x: x % var_24 == var_25
    var_40 = module_0.split_by(var_38, criterion=var_39)
    var_41 = list(var_40)
    var_42 = [var_25, var_24, var_36, var_37]
    var_43 = lambda x: x % var_24 == var_25
    var_44 = module_0.split_by(var_42, var_8, criterion=var_43)
    var_45 = list(var_44)
    var_46 = 2
    var_47 = 4
    var_48 = 5
    var_49 = [var_8, var_46, var_47, var_48]
    var_50 = lambda x: x % var_24 == var_25
    var_51 = module_0.split_by(var_49, criterion=var_50)
    var_52 = list(var_51)
    var_53 = [var_8, var_46, var_47, var_48]
    var_54 = lambda x: x % var_24 == var_25
    var_55 = module_0.split_by(var_53, var_8, criterion=var_54)
    var_56 = list(var_55)



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 100
    var_4 = -100



# Parsed testcases at query #29
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = []
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = 0
    var_6 = lambda x: x >= var_5
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = [var_7, var_8, var_9]
    var_11 = module_0.drop_until(var_6, var_10)
    var_12 = list(var_11)
    var_13 = 10
    var_14 = lambda x: x > var_13
    var_15 = [var_7, var_8, var_9]
    var_16 = module_0.drop_until(var_14, var_15)
    var_17 = list(var_16)
    var_18 = lambda x: x > var_8
    var_19 = 4
    var_20 = [var_7, var_8, var_9, var_19, var_0]
    var_21 = module_0.drop_until(var_18, var_20)
    var_22 = list(var_21)
    var_23 = lambda x: x == var_7
    var_24 = [var_7, var_8, var_9]
    var_25 = module_0.drop_until(var_23, var_24)
    var_26 = list(var_25)
    var_27 = lambda x: x == var_9
    var_28 = [var_7, var_8, var_9]
    var_29 = module_0.drop_until(var_27, var_28)
    var_30 = list(var_29)
    var_31 = lambda x: x % var_8 == var_5
    var_32 = [var_7, var_8, var_9, var_19, var_0]
    var_33 = module_0.drop_until(var_31, var_32)
    var_34 = list(var_33)
    var_35 = lambda x: x.value > var_7



# Parsed testcases at query #30
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
    var_7 = ' Split by: '
    var_8 = '.'
    var_9 = module_0.split_by(var_7, separator=var_8)
    var_10 = list(var_9)
    var_11 = range(var_0)
    var_12 = True
    var_13 = lambda x: x % var_2 == var_3
    var_14 = module_0.split_by(var_11, var_12, criterion=var_13)
    var_15 = list(var_14)
    var_16 = module_0.split_by(var_7, var_12, separator=var_8)
    var_17 = list(var_16)
    var_18 = []
    var_19 = lambda x: x % var_2 == var_3
    var_20 = module_0.split_by(var_18, criterion=var_19)
    var_21 = list(var_20)
    var_22 = []
    var_23 = lambda x: x % var_2 == var_3
    var_24 = module_0.split_by(var_22, var_12, criterion=var_23)
    var_25 = list(var_24)
    var_26 = 6
    var_27 = 9
    var_28 = [var_3, var_2, var_26, var_27]
    var_29 = lambda x: x % var_2 == var_3
    var_30 = module_0.split_by(var_28, criterion=var_29)
    var_31 = list(var_30)
    var_32 = [var_3, var_2, var_26, var_27]
    var_33 = lambda x: x % var_2 == var_3
    var_34 = module_0.split_by(var_32, var_12, criterion=var_33)
    var_35 = list(var_34)
    var_36 = 2
    var_37 = 4
    var_38 = 5
    var_39 = [var_12, var_36, var_37, var_38]
    var_40 = lambda x: x % var_2 == var_3
    var_41 = module_0.split_by(var_39, criterion=var_40)
    var_42 = list(var_41)
    var_43 = 1
    var_44 = 2
    var_45 = 3
    var_46 = [var_43, var_44, var_45]
    var_47 = 0
    var_48 = lambda x: x % var_45 == var_47
    var_49 = module_0.split_by(var_46, criterion=var_48, separator=var_45)
    var_50 = list(var_49)
    var_51 = 1
    var_52 = 2
    var_53 = 3
    var_54 = [var_51, var_52, var_53]
    var_55 = module_0.split_by(var_54)
    var_56 = list(var_55)



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 10
    var_4 = -10



# Parsed testcases at query #32
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



# Parsed testcases at query #33
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
    var_15 = lambda x: x < var_14
    var_16 = [var_8, var_9, var_6]
    var_17 = module_0.drop_until(var_15, var_16)
    var_18 = list(var_17)
    var_19 = lambda x: x > var_14
    var_20 = []
    var_21 = module_0.drop_until(var_19, var_20)
    var_22 = list(var_21)
    var_23 = lambda x: x >= var_14
    var_24 = [var_8, var_9, var_6]
    var_25 = module_0.drop_until(var_23, var_24)
    var_26 = list(var_25)
    var_27 = lambda x: x < var_14
    var_28 = [var_8, var_9, var_6]
    var_29 = module_0.drop_until(var_27, var_28)
    var_30 = list(var_29)
    var_31 = lambda x: x == var_8
    var_32 = [var_8, var_9, var_6]
    var_33 = module_0.drop_until(var_31, var_32)
    var_34 = list(var_33)
    var_35 = lambda obj: obj.value > var_8



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 5
    var_4 = 10
    var_5 = -10



# Parsed testcases at query #35
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



# Parsed testcases at query #36
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
    var_11 = 0
    var_12 = lambda x: x < var_11
    var_13 = range(var_2)
    var_14 = module_0.drop_until(var_12, var_13)
    var_15 = list(var_14)
    var_16 = lambda x: x > var_0
    var_17 = []
    var_18 = module_0.drop_until(var_16, var_17)
    var_19 = list(var_18)
    var_20 = lambda x: x >= var_11
    var_21 = range(var_2)
    var_22 = module_0.drop_until(var_20, var_21)
    var_23 = list(var_22)
    var_24 = lambda x: x > var_2
    var_25 = range(var_2)
    var_26 = module_0.drop_until(var_24, var_25)
    var_27 = list(var_26)
    var_28 = 2
    var_29 = lambda x: x % var_28 == var_11
    var_30 = 1
    var_31 = 6
    var_32 = 7
    var_33 = 8
    var_34 = [var_30, var_6, var_0, var_31, var_32, var_33]
    var_35 = module_0.drop_until(var_29, var_34)
    var_36 = list(var_35)
    var_37 = 'c'
    var_38 = lambda x: x == var_37
    var_39 = 'abcdef'
    var_40 = module_0.drop_until(var_38, var_39)
    var_41 = list(var_40)



# Parsed testcases at query #37
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
    var_27 = [var_3, var_2, var_25, var_26]
    var_28 = lambda x: x % var_2 == var_3
    var_29 = module_0.split_by(var_27, criterion=var_28)
    var_30 = list(var_29)
    var_31 = [var_3, var_2, var_25, var_26]
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



# Parsed testcases at query #38
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



# Parsed testcases at query #39
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = 4
    var_5 = 5
    var_6 = [var_2, var_0, var_3, var_4, var_5]
    var_7 = module_0.MapList(var_1, var_6)
    var_8 = 5
    var_9 = var_7[var_8]
    var_10 = -6
    var_11 = var_7[var_10]



# Parsed testcases at query #40
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 100
    var_4 = -100



# Parsed testcases at query #41
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
    var_7 = []
    var_8 = lambda x: x % var_2 == var_3
    var_9 = module_0.split_by(var_7, criterion=var_8)
    var_10 = list(var_9)
    var_11 = 1
    var_12 = 2
    var_13 = 4
    var_14 = 5
    var_15 = [var_11, var_12, var_2, var_13, var_14]
    var_16 = lambda x: x == var_2
    var_17 = module_0.split_by(var_15, criterion=var_16)
    var_18 = list(var_17)
    var_19 = ' Split by: '
    var_20 = ' '
    var_21 = module_0.split_by(var_19, separator=var_20)
    var_22 = list(var_21)
    var_23 = [var_11, var_12, var_2, var_13, var_14]
    var_24 = module_0.split_by(var_23, separator=var_2)
    var_25 = list(var_24)
    var_26 = True
    var_27 = '.'
    var_28 = module_0.split_by(var_19, var_26, separator=var_27)
    var_29 = list(var_28)
    var_30 = [var_26, var_12, var_2, var_13, var_14]
    var_31 = True
    var_32 = module_0.split_by(var_30, var_31, separator=var_2)
    var_33 = list(var_32)
    var_34 = module_0.split_by(var_19, separator=var_27)
    var_35 = list(var_34)
    var_36 = [var_31, var_12, var_2, var_13, var_14]
    var_37 = module_0.split_by(var_36, separator=var_2)
    var_38 = list(var_37)
    var_39 = 1
    var_40 = 2
    var_41 = 3
    var_42 = [var_39, var_40, var_41]
    var_43 = lambda x: x == var_40
    var_44 = module_0.split_by(var_42, criterion=var_43, separator=var_40)
    var_45 = list(var_44)
    var_46 = 1
    var_47 = 2
    var_48 = 3
    var_49 = [var_46, var_47, var_48]
    var_50 = module_0.split_by(var_49)
    var_51 = list(var_50)



# Parsed testcases at query #42
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = []
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = 0
    var_6 = lambda x: x > var_5
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = 4
    var_11 = [var_7, var_8, var_9, var_10]
    var_12 = module_0.drop_until(var_6, var_11)
    var_13 = list(var_12)
    var_14 = 10
    var_15 = lambda x: x > var_14
    var_16 = [var_7, var_8, var_9, var_10]
    var_17 = module_0.drop_until(var_15, var_16)
    var_18 = list(var_17)
    var_19 = lambda x: x > var_8
    var_20 = [var_7, var_8, var_9, var_10]
    var_21 = module_0.drop_until(var_19, var_20)
    var_22 = list(var_21)
    var_23 = lambda x: x > var_5
    var_24 = [var_7, var_8, var_9, var_10]
    var_25 = module_0.drop_until(var_23, var_24)
    var_26 = list(var_25)
    var_27 = lambda x: x > var_9
    var_28 = [var_7, var_8, var_9, var_10]
    var_29 = module_0.drop_until(var_27, var_28)
    var_30 = list(var_29)
    var_31 = lambda x: x % var_8 == var_5
    var_32 = 6
    var_33 = 7
    var_34 = 8
    var_35 = [var_7, var_9, var_0, var_32, var_33, var_34]
    var_36 = module_0.drop_until(var_31, var_35)
    var_37 = list(var_36)
    var_38 = 'c'
    var_39 = lambda x: x == var_38
    var_40 = 'abcdef'
    var_41 = module_0.drop_until(var_39, var_40)
    var_42 = list(var_41)
    var_43 = lambda x: x > var_0
    var_44 = range(var_14)



# Parsed testcases at query #43
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 10
    var_4 = -10



# Parsed testcases at query #44
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
    var_23 = module_0.split_by(var_22, separator=var_13)
    var_24 = list(var_23)
    var_25 = 6
    var_26 = 9
    var_27 = [var_2, var_25, var_26]
    var_28 = lambda x: x % var_2 == var_3
    var_29 = module_0.split_by(var_27, criterion=var_28)
    var_30 = list(var_29)
    var_31 = [var_2, var_25, var_26]
    var_32 = False
    var_33 = lambda x: x % var_2 == var_32
    var_34 = module_0.split_by(var_31, var_32, criterion=var_33)
    var_35 = list(var_34)
    var_36 = 2
    var_37 = 4
    var_38 = 5
    var_39 = [var_8, var_36, var_37, var_38]
    var_40 = lambda x: x % var_2 == var_32
    var_41 = module_0.split_by(var_39, criterion=var_40)
    var_42 = list(var_41)
    var_43 = 1
    var_44 = 2
    var_45 = 3
    var_46 = [var_43, var_44, var_45]
    var_47 = 0
    var_48 = lambda x: x % var_45 == var_47
    var_49 = ' '
    var_50 = module_0.split_by(var_46, criterion=var_48, separator=var_49)
    var_51 = list(var_50)
    var_52 = 1
    var_53 = 2
    var_54 = 3
    var_55 = [var_52, var_53, var_54]
    var_56 = module_0.split_by(var_55)
    var_57 = list(var_56)



# Parsed testcases at query #45
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
    var_23 = module_0.split_by(var_22, separator=var_13)
    var_24 = list(var_23)
    var_25 = 6
    var_26 = 9
    var_27 = [var_2, var_25, var_26]
    var_28 = lambda x: x % var_2 == var_3
    var_29 = module_0.split_by(var_27, criterion=var_28)
    var_30 = list(var_29)
    var_31 = [var_2, var_25, var_26]
    var_32 = False
    var_33 = lambda x: x % var_2 == var_32
    var_34 = module_0.split_by(var_31, var_32, criterion=var_33)
    var_35 = list(var_34)
    var_36 = 2
    var_37 = 4
    var_38 = 5
    var_39 = [var_8, var_36, var_37, var_38]
    var_40 = lambda x: x % var_2 == var_32
    var_41 = module_0.split_by(var_39, criterion=var_40)
    var_42 = list(var_41)
    var_43 = 1
    var_44 = 2
    var_45 = 3
    var_46 = [var_43, var_44, var_45]
    var_47 = 0
    var_48 = lambda x: x % var_45 == var_47
    var_49 = ' '
    var_50 = module_0.split_by(var_46, criterion=var_48, separator=var_49)
    var_51 = list(var_50)
    var_52 = 1
    var_53 = 2
    var_54 = 3
    var_55 = [var_52, var_53, var_54]
    var_56 = module_0.split_by(var_55)
    var_57 = list(var_56)



# Parsed testcases at query #46
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
    var_43 = lambda x: x % var_2 == var_3
    var_44 = module_0.split_by(var_42, var_8, criterion=var_43)
    var_45 = list(var_44)
    var_46 = 1
    var_47 = 2
    var_48 = 3
    var_49 = [var_46, var_47, var_48]
    var_50 = 0
    var_51 = lambda x: x % var_48 == var_50
    var_52 = ' '
    var_53 = module_0.split_by(var_49, criterion=var_51, separator=var_52)
    var_54 = list(var_53)
    var_55 = 1
    var_56 = 2
    var_57 = 3
    var_58 = [var_55, var_56, var_57]
    var_59 = module_0.split_by(var_58)
    var_60 = list(var_59)



# Parsed testcases at query #47
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



# Parsed testcases at query #48
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 11
    var_3 = 2
    var_4 = 5
    var_5 = 10
    var_6 = -10



# Parsed testcases at query #49
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
    var_12 = []
    var_13 = lambda x: x % var_2 == var_3
    var_14 = module_0.split_by(var_12, criterion=var_13)
    var_15 = list(var_14)
    var_16 = 2
    var_17 = [var_8, var_16, var_2]
    var_18 = lambda x: x == var_16
    var_19 = module_0.split_by(var_17, criterion=var_18)
    var_20 = list(var_19)
    var_21 = ' Split by: '
    var_22 = ' '
    var_23 = module_0.split_by(var_21, separator=var_22)
    var_24 = list(var_23)
    var_25 = module_0.split_by(var_21, var_8, separator=var_22)
    var_26 = list(var_25)
    var_27 = []
    var_28 = module_0.split_by(var_27, separator=var_22)
    var_29 = list(var_28)
    var_30 = 4
    var_31 = [var_8, var_16, var_2, var_16, var_30]
    var_32 = module_0.split_by(var_31, separator=var_16)
    var_33 = list(var_32)
    var_34 = 10
    var_35 = range(var_34)
    var_36 = 3
    var_37 = 0
    var_38 = lambda x: x % var_36 == var_37
    var_39 = ' '
    var_40 = module_0.split_by(var_35, criterion=var_38, separator=var_39)
    var_41 = list(var_40)
    var_42 = 10
    var_43 = range(var_42)
    var_44 = module_0.split_by(var_43)
    var_45 = list(var_44)



# Parsed testcases at query #50
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
    var_23 = module_0.split_by(var_22, separator=var_13)
    var_24 = list(var_23)
    var_25 = 6
    var_26 = 9
    var_27 = [var_2, var_25, var_26]
    var_28 = lambda x: x % var_2 == var_3
    var_29 = module_0.split_by(var_27, criterion=var_28)
    var_30 = list(var_29)
    var_31 = [var_2, var_25, var_26]
    var_32 = False
    var_33 = lambda x: x % var_2 == var_32
    var_34 = module_0.split_by(var_31, var_32, criterion=var_33)
    var_35 = list(var_34)
    var_36 = 2
    var_37 = 4
    var_38 = 5
    var_39 = [var_8, var_36, var_37, var_38]
    var_40 = lambda x: x % var_2 == var_32
    var_41 = module_0.split_by(var_39, criterion=var_40)
    var_42 = list(var_41)
    var_43 = 1
    var_44 = 2
    var_45 = 3
    var_46 = [var_43, var_44, var_45]
    var_47 = 0
    var_48 = lambda x: x % var_45 == var_47
    var_49 = ' '
    var_50 = module_0.split_by(var_46, criterion=var_48, separator=var_49)
    var_51 = list(var_50)
    var_52 = 1
    var_53 = 2
    var_54 = 3
    var_55 = [var_52, var_53, var_54]
    var_56 = module_0.split_by(var_55)
    var_57 = list(var_56)



# Parsed testcases at query #51
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 10
    var_4 = -11



# Parsed testcases at query #52
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



# Parsed testcases at query #53
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



# Parsed testcases at query #54
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 10
    var_4 = -11



# Parsed testcases at query #55
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
    var_15 = lambda x: x < var_14
    var_16 = [var_8, var_9, var_6]
    var_17 = module_0.drop_until(var_15, var_16)
    var_18 = list(var_17)
    var_19 = lambda x: x > var_14
    var_20 = []
    var_21 = module_0.drop_until(var_19, var_20)
    var_22 = list(var_21)
    var_23 = lambda x: x >= var_14
    var_24 = [var_8, var_9, var_6]
    var_25 = module_0.drop_until(var_23, var_24)
    var_26 = list(var_25)
    var_27 = lambda x: x < var_14
    var_28 = [var_8, var_9, var_6]
    var_29 = module_0.drop_until(var_27, var_28)
    var_30 = list(var_29)
    var_31 = lambda x: x.val > var_9



# Parsed testcases at query #56
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = 2
    var_7 = 0
    var_8 = lambda x: x % var_6 == var_7
    var_9 = 1
    var_10 = 3
    var_11 = 6
    var_12 = 7
    var_13 = 8
    var_14 = [var_9, var_10, var_0, var_11, var_12, var_13]
    var_15 = module_0.drop_until(var_8, var_14)
    var_16 = list(var_15)
    var_17 = lambda x: x > var_0
    var_18 = []
    var_19 = module_0.drop_until(var_17, var_18)
    var_20 = list(var_19)
    var_21 = lambda x: x > var_7
    var_22 = [var_9, var_6, var_10]
    var_23 = module_0.drop_until(var_21, var_22)
    var_24 = list(var_23)
    var_25 = lambda x: x > var_2
    var_26 = range(var_0)
    var_27 = module_0.drop_until(var_25, var_26)
    var_28 = list(var_27)
    var_29 = 4
    var_30 = lambda x: x.val > var_6
    var_31 = range(var_2)
    var_32 = lambda x: x == var_0



# Parsed testcases at query #57
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
    var_27 = [var_3, var_2, var_25, var_26]
    var_28 = lambda x: x % var_2 == var_3
    var_29 = module_0.split_by(var_27, criterion=var_28)
    var_30 = list(var_29)
    var_31 = [var_3, var_2, var_25, var_26]
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



# Parsed testcases at query #58
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = []
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = 0
    var_6 = lambda x: x >= var_5
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = [var_7, var_8, var_9]
    var_11 = module_0.drop_until(var_6, var_10)
    var_12 = list(var_11)
    var_13 = 10
    var_14 = lambda x: x > var_13
    var_15 = [var_7, var_8, var_9]
    var_16 = module_0.drop_until(var_14, var_15)
    var_17 = list(var_16)
    var_18 = lambda x: x > var_0
    var_19 = 4
    var_20 = 6
    var_21 = 7
    var_22 = 8
    var_23 = [var_7, var_8, var_9, var_19, var_0, var_20, var_21, var_22]
    var_24 = module_0.drop_until(var_18, var_23)
    var_25 = list(var_24)
    var_26 = lambda x: x > var_5
    var_27 = [var_7, var_8, var_9]
    var_28 = module_0.drop_until(var_26, var_27)
    var_29 = list(var_28)
    var_30 = lambda x: x > var_0
    var_31 = [var_7, var_8, var_9, var_19, var_0, var_20]
    var_32 = module_0.drop_until(var_30, var_31)
    var_33 = list(var_32)
    var_34 = lambda x: x.value > var_8



# Parsed testcases at query #59
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 100
    var_4 = -100



# Parsed testcases at query #60
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
    var_18 = 8
    var_19 = [var_8, var_6, var_0, var_16, var_17, var_18]
    var_20 = module_0.drop_until(var_15, var_19)
    var_21 = list(var_20)
    var_22 = lambda x: x > var_0
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
    var_34 = 'c'
    var_35 = lambda x: x == var_34
    var_36 = 'abcdef'
    var_37 = module_0.drop_until(var_35, var_36)
    var_38 = list(var_37)
    var_39 = lambda x: x.val > var_9



# Parsed testcases at query #61
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = lambda x: x == var_0
    var_7 = range(var_2)
    var_8 = module_0.drop_until(var_6, var_7)
    var_9 = list(var_8)
    var_10 = lambda x: x < var_0
    var_11 = range(var_2)
    var_12 = module_0.drop_until(var_10, var_11)
    var_13 = list(var_12)
    var_14 = lambda x: x > var_0
    var_15 = []
    var_16 = module_0.drop_until(var_14, var_15)
    var_17 = list(var_16)
    var_18 = 0
    var_19 = lambda x: x >= var_18
    var_20 = range(var_2)
    var_21 = module_0.drop_until(var_19, var_20)
    var_22 = list(var_21)
    var_23 = lambda x: x > var_2
    var_24 = range(var_2)
    var_25 = module_0.drop_until(var_23, var_24)
    var_26 = list(var_25)
    var_27 = 'c'
    var_28 = lambda x: x == var_27
    var_29 = 'abcdefg'
    var_30 = module_0.drop_until(var_28, var_29)
    var_31 = list(var_30)
    var_32 = 'z'
    var_33 = lambda x: x == var_32
    var_34 = module_0.drop_until(var_33, var_29)
    var_35 = list(var_34)
    var_36 = range(var_2)
    var_37 = [TestObj(i) for i in var_36]
    var_38 = lambda x: x.val > var_0
    var_39 = module_0.drop_until(var_38, var_37)
    var_40 = list(var_39)



# Parsed testcases at query #62
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
    var_41 = [var_8, var_3, var_3, var_25]
    var_42 = module_0.split_by(var_41, separator=var_3)
    var_43 = list(var_42)
    var_44 = [var_8, var_3, var_3, var_25]
    var_45 = module_0.split_by(var_44, var_8, separator=var_3)
    var_46 = list(var_45)
    var_47 = 1
    var_48 = 2
    var_49 = 3
    var_50 = [var_47, var_48, var_49]
    var_51 = 0
    var_52 = lambda x: x > var_51
    var_53 = module_0.split_by(var_50, criterion=var_52, separator=var_51)
    var_54 = list(var_53)
    var_55 = 1
    var_56 = 2
    var_57 = 3
    var_58 = [var_55, var_56, var_57]
    var_59 = module_0.split_by(var_58)
    var_60 = list(var_59)



# Parsed testcases at query #63
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 11
    var_3 = 2
    var_4 = 10
    var_5 = -10



# Parsed testcases at query #64
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 100
    var_4 = -100



# Parsed testcases at query #65
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = 4
    var_5 = 5
    var_6 = [var_2, var_0, var_3, var_4, var_5]
    var_7 = module_0.MapList(var_1, var_6)
    var_8 = 10
    var_9 = var_7[var_8]
    var_10 = -10
    var_11 = var_7[var_10]



# Parsed testcases at query #66
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = 0
    var_7 = lambda x: x == var_6
    var_8 = range(var_2)
    var_9 = module_0.drop_until(var_7, var_8)
    var_10 = list(var_9)
    var_11 = 9
    var_12 = lambda x: x == var_11
    var_13 = range(var_2)
    var_14 = module_0.drop_until(var_12, var_13)
    var_15 = list(var_14)
    var_16 = lambda x: x > var_2
    var_17 = range(var_2)
    var_18 = module_0.drop_until(var_16, var_17)
    var_19 = list(var_18)
    var_20 = lambda x: x > var_0
    var_21 = []
    var_22 = module_0.drop_until(var_20, var_21)
    var_23 = list(var_22)
    var_24 = 'a'
    var_25 = lambda x: x.startswith(var_24)
    var_26 = 'b'
    var_27 = 'c'
    var_28 = 'apple'
    var_29 = 'banana'
    var_30 = [var_26, var_27, var_28, var_29]
    var_31 = module_0.drop_until(var_25, var_30)
    var_32 = list(var_31)



# Parsed testcases at query #67
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 11
    var_3 = 2
    var_4 = 10
    var_5 = -11



# Parsed testcases at query #68
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 11
    var_3 = 2
    var_4 = 10
    var_5 = -10



# Parsed testcases at query #69
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
    var_22 = 0
    var_23 = lambda x: x == var_22
    var_24 = range(var_0)
    var_25 = module_0.drop_until(var_23, var_24)
    var_26 = list(var_25)
    var_27 = range(var_0)
    var_28 = [Custom(i) for i in var_27]
    var_29 = lambda x: x.val == var_9
    var_30 = module_0.drop_until(var_29, var_28)
    var_31 = list(var_30)
    var_32 = 'c'
    var_33 = lambda x: x == var_32
    var_34 = 'abcdef'
    var_35 = module_0.drop_until(var_33, var_34)
    var_36 = list(var_35)



# Parsed testcases at query #70
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
    var_7 = []
    var_8 = lambda x: x % var_2 == var_3
    var_9 = module_0.split_by(var_7, criterion=var_8)
    var_10 = list(var_9)
    var_11 = 1
    var_12 = 2
    var_13 = [var_11, var_12, var_2]
    var_14 = lambda x: x == var_12
    var_15 = module_0.split_by(var_13, criterion=var_14)
    var_16 = list(var_15)
    var_17 = ' Split by: '
    var_18 = ' '
    var_19 = module_0.split_by(var_17, separator=var_18)
    var_20 = list(var_19)
    var_21 = []
    var_22 = module_0.split_by(var_21, separator=var_18)
    var_23 = list(var_22)
    var_24 = [var_11, var_12, var_2]
    var_25 = module_0.split_by(var_24, separator=var_12)
    var_26 = list(var_25)
    var_27 = True
    var_28 = module_0.split_by(var_17, var_27, separator=var_18)
    var_29 = list(var_28)
    var_30 = [var_27, var_12, var_2, var_12]
    var_31 = True
    var_32 = module_0.split_by(var_30, var_31, separator=var_12)
    var_33 = list(var_32)
    var_34 = 1
    var_35 = 2
    var_36 = 3
    var_37 = [var_34, var_35, var_36]
    var_38 = lambda x: x == var_35
    var_39 = module_0.split_by(var_37, criterion=var_38, separator=var_35)
    var_40 = list(var_39)
    var_41 = 1
    var_42 = 2
    var_43 = 3
    var_44 = [var_41, var_42, var_43]
    var_45 = module_0.split_by(var_44)
    var_46 = list(var_45)



# Parsed testcases at query #71
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
    var_8 = lambda x: x ** var_1
    var_9 = module_0.MapList(var_8, var_5)



# Parsed testcases at query #72
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
    var_15 = lambda x: x < var_14
    var_16 = [var_8, var_9, var_6]
    var_17 = module_0.drop_until(var_15, var_16)
    var_18 = list(var_17)
    var_19 = lambda x: x > var_14
    var_20 = []
    var_21 = module_0.drop_until(var_19, var_20)
    var_22 = list(var_21)
    var_23 = lambda x: x >= var_14
    var_24 = [var_8, var_9, var_6]
    var_25 = module_0.drop_until(var_23, var_24)
    var_26 = list(var_25)
    var_27 = lambda x: x < var_14
    var_28 = [var_8, var_9, var_6]
    var_29 = module_0.drop_until(var_27, var_28)
    var_30 = list(var_29)
    var_31 = lambda x: x == var_8
    var_32 = [var_8, var_9, var_6]
    var_33 = module_0.drop_until(var_31, var_32)
    var_34 = list(var_33)
    var_35 = lambda x: x == var_6
    var_36 = [var_8, var_9, var_6]
    var_37 = module_0.drop_until(var_35, var_36)
    var_38 = list(var_37)
    var_39 = lambda x: x.val > var_9



# Parsed testcases at query #73
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
    var_15 = lambda x: x < var_14
    var_16 = [var_8, var_9, var_6]
    var_17 = module_0.drop_until(var_15, var_16)
    var_18 = list(var_17)
    var_19 = lambda x: x > var_14
    var_20 = []
    var_21 = module_0.drop_until(var_19, var_20)
    var_22 = list(var_21)
    var_23 = lambda x: x >= var_14
    var_24 = [var_8, var_9, var_6]
    var_25 = module_0.drop_until(var_23, var_24)
    var_26 = list(var_25)
    var_27 = lambda x: x < var_14
    var_28 = [var_8, var_9, var_6]
    var_29 = module_0.drop_until(var_27, var_28)
    var_30 = list(var_29)
    var_31 = lambda x: x == var_8
    var_32 = [var_8, var_9, var_6]
    var_33 = module_0.drop_until(var_31, var_32)
    var_34 = list(var_33)
    var_35 = lambda x: x == var_6
    var_36 = [var_8, var_9, var_6]
    var_37 = module_0.drop_until(var_35, var_36)
    var_38 = list(var_37)
    var_39 = lambda x: x.value > var_8



# Parsed testcases at query #74
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



# Parsed testcases at query #75
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
    var_23 = lambda x: x > var_3
    var_24 = module_0.split_by(var_22, var_8, criterion=var_23)
    var_25 = list(var_24)
    var_26 = 2
    var_27 = [var_8, var_26, var_2]
    var_28 = lambda x: x > var_3
    var_29 = module_0.split_by(var_27, criterion=var_28)
    var_30 = list(var_29)
    var_31 = [var_8, var_26, var_2]
    var_32 = lambda x: x > var_3
    var_33 = module_0.split_by(var_31, var_8, criterion=var_32)
    var_34 = list(var_33)
    var_35 = [var_8, var_26, var_2]
    var_36 = lambda x: x > var_0
    var_37 = module_0.split_by(var_35, criterion=var_36)
    var_38 = list(var_37)
    var_39 = [var_8, var_26, var_2]
    var_40 = lambda x: x > var_0
    var_41 = module_0.split_by(var_39, var_8, criterion=var_40)
    var_42 = list(var_41)
    var_43 = 1
    var_44 = 2
    var_45 = 3
    var_46 = [var_43, var_44, var_45]
    var_47 = 0
    var_48 = lambda x: x > var_47
    var_49 = module_0.split_by(var_46, criterion=var_48, separator=var_47)
    var_50 = list(var_49)
    var_51 = 1
    var_52 = 2
    var_53 = 3
    var_54 = [var_51, var_52, var_53]
    var_55 = module_0.split_by(var_54)
    var_56 = list(var_55)



# Parsed testcases at query #76
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
    var_20 = range(var_2)
    var_21 = list(var_20)
    var_22 = lambda x: x < var_15
    var_23 = range(var_2)
    var_24 = module_0.drop_until(var_22, var_23)
    var_25 = list(var_24)
    var_26 = lambda x: x == var_15
    var_27 = range(var_2)
    var_28 = module_0.drop_until(var_26, var_27)
    var_29 = list(var_28)
    var_30 = range(var_2)
    var_31 = list(var_30)
    var_32 = 2
    var_33 = lambda x: x % var_32 == var_15
    var_34 = 1
    var_35 = 6
    var_36 = 7
    var_37 = 8
    var_38 = [var_34, var_6, var_0, var_35, var_36, var_37]
    var_39 = module_0.drop_until(var_33, var_38)
    var_40 = list(var_39)
    var_41 = 'c'
    var_42 = lambda x: x == var_41
    var_43 = 'abcdef'
    var_44 = module_0.drop_until(var_42, var_43)
    var_45 = list(var_44)
    var_46 = range(var_2)
    var_47 = lambda x: x > var_0



# Parsed testcases at query #77
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 11
    var_3 = 2
    var_4 = 100
    var_5 = -100



# Parsed testcases at query #78
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 10
    var_4 = -10



# Parsed testcases at query #79
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
    var_21 = lambda x: x > var_14
    var_22 = []
    var_23 = module_0.drop_until(var_21, var_22)
    var_24 = list(var_23)
    var_25 = lambda x: x > var_2
    var_26 = range(var_0)
    var_27 = module_0.drop_until(var_25, var_26)
    var_28 = list(var_27)
    var_29 = lambda x: x == var_8
    var_30 = [var_8, var_9, var_6]
    var_31 = module_0.drop_until(var_29, var_30)
    var_32 = list(var_31)
    var_33 = lambda s: len(s) > var_6
    var_34 = 'a'
    var_35 = 'ab'
    var_36 = 'abc'
    var_37 = 'abcd'
    var_38 = 'abcde'
    var_39 = [var_34, var_35, var_36, var_37, var_38]
    var_40 = module_0.drop_until(var_33, var_39)
    var_41 = list(var_40)
    var_42 = 'key'
    var_43 = lambda x: x[var_42] == var_9
    var_44 = {var_42: var_8}
    var_45 = {var_42: var_9}
    var_46 = {var_42: var_6}
    var_47 = [var_44, var_45, var_46]
    var_48 = module_0.drop_until(var_43, var_47)
    var_49 = list(var_48)
    var_50 = range(var_2)
    var_51 = lambda x: x == var_0



# Parsed testcases at query #80
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = list(var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 10
    var_6 = -11



# Parsed testcases at query #81
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
    var_8 = 5
    var_9 = var_7[var_8]
    var_10 = -6
    var_11 = var_7[var_10]



# Parsed testcases at query #82
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
    var_8 = 'a'
    var_9 = 'b'
    var_10 = 'c'
    var_11 = [var_8, var_9, var_10]
    var_12 = lambda x: x.upper()
    var_13 = module_0.MapList(var_12, var_11)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
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
    var_18 = []
    var_19 = lambda x: x % var_2 == var_3
    var_20 = module_0.split_by(var_18, criterion=var_19)
    var_21 = list(var_20)
    var_22 = []
    var_23 = lambda x: x % var_2 == var_3
    var_24 = module_0.split_by(var_22, var_8, criterion=var_23)
    var_25 = list(var_24)
    var_26 = 6
    var_27 = 9
    var_28 = [var_2, var_26, var_27]
    var_29 = lambda x: x % var_2 == var_3
    var_30 = module_0.split_by(var_28, criterion=var_29)
    var_31 = list(var_30)
    var_32 = [var_2, var_26, var_27]
    var_33 = lambda x: x % var_2 == var_3
    var_34 = module_0.split_by(var_32, var_8, criterion=var_33)
    var_35 = list(var_34)
    var_36 = 2
    var_37 = 4
    var_38 = 5
    var_39 = [var_8, var_36, var_37, var_38]
    var_40 = lambda x: x % var_2 == var_3
    var_41 = module_0.split_by(var_39, criterion=var_40)
    var_42 = list(var_41)
    var_43 = [var_8, var_36, var_37, var_38]
    var_44 = lambda x: x % var_2 == var_3
    var_45 = module_0.split_by(var_43, var_8, criterion=var_44)
    var_46 = list(var_45)
    var_47 = 1
    var_48 = 2
    var_49 = 3
    var_50 = [var_47, var_48, var_49]
    var_51 = 0
    var_52 = lambda x: x % var_49 == var_51
    var_53 = module_0.split_by(var_50, criterion=var_52, separator=var_49)
    var_54 = list(var_53)
    var_55 = 1
    var_56 = 2
    var_57 = 3
    var_58 = [var_55, var_56, var_57]
    var_59 = module_0.split_by(var_58)
    var_60 = list(var_59)



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
    var_18 = 'a..b..c'
    var_19 = '.'
    var_20 = module_0.split_by(var_18, separator=var_19)
    var_21 = list(var_20)
    var_22 = module_0.split_by(var_18, var_8, separator=var_19)
    var_23 = list(var_22)
    var_24 = 2
    var_25 = [var_8, var_24, var_2]
    var_26 = lambda x: x > var_0
    var_27 = module_0.split_by(var_25, criterion=var_26)
    var_28 = list(var_27)
    var_29 = [var_8, var_24, var_2]
    var_30 = module_0.split_by(var_29, separator=var_0)
    var_31 = list(var_30)
    var_32 = [var_8, var_8, var_8]
    var_33 = module_0.split_by(var_32, separator=var_8)
    var_34 = list(var_33)
    var_35 = [var_8, var_8, var_8]
    var_36 = module_0.split_by(var_35, var_8, separator=var_8)
    var_37 = list(var_36)
    var_38 = []
    var_39 = 5
    var_40 = lambda x: x > var_39
    var_41 = module_0.split_by(var_38, criterion=var_40)
    var_42 = list(var_41)
    var_43 = []
    var_44 = module_0.split_by(var_43, separator=var_39)
    var_45 = list(var_44)
    var_46 = 1
    var_47 = 2
    var_48 = 3
    var_49 = [var_46, var_47, var_48]
    var_50 = 5
    var_51 = lambda x: x > var_50
    var_52 = module_0.split_by(var_49, criterion=var_51, separator=var_50)
    var_53 = list(var_52)
    var_54 = 1
    var_55 = 2
    var_56 = 3
    var_57 = [var_54, var_55, var_56]
    var_58 = module_0.split_by(var_57)
    var_59 = list(var_58)



# Parsed testcases at query #3
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
    var_11 = 0
    var_12 = lambda x: x < var_11
    var_13 = range(var_2)
    var_14 = module_0.drop_until(var_12, var_13)
    var_15 = list(var_14)
    var_16 = lambda x: x > var_0
    var_17 = []
    var_18 = module_0.drop_until(var_16, var_17)
    var_19 = list(var_18)
    var_20 = lambda x: x >= var_11
    var_21 = range(var_2)
    var_22 = module_0.drop_until(var_20, var_21)
    var_23 = list(var_22)
    var_24 = range(var_2)
    var_25 = list(var_24)
    var_26 = lambda x: x < var_11
    var_27 = range(var_2)
    var_28 = module_0.drop_until(var_26, var_27)
    var_29 = list(var_28)
    var_30 = 2
    var_31 = lambda x: x % var_30 == var_11
    var_32 = 1
    var_33 = 6
    var_34 = 7
    var_35 = 8
    var_36 = [var_32, var_6, var_0, var_33, var_34, var_35]
    var_37 = module_0.drop_until(var_31, var_36)
    var_38 = list(var_37)
    var_39 = 'c'
    var_40 = lambda x: x == var_39
    var_41 = 'abcdef'
    var_42 = module_0.drop_until(var_40, var_41)
    var_43 = list(var_42)



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 1
    var_2 = 0
    var_3 = 10
    var_4 = 2
    var_5 = -1



# Parsed testcases at query #5
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
    var_11 = 0
    var_12 = lambda x: x < var_11
    var_13 = range(var_2)
    var_14 = module_0.drop_until(var_12, var_13)
    var_15 = list(var_14)
    var_16 = lambda x: x > var_0
    var_17 = []
    var_18 = module_0.drop_until(var_16, var_17)
    var_19 = list(var_18)
    var_20 = 100
    var_21 = lambda x: x > var_20
    var_22 = range(var_2)
    var_23 = module_0.drop_until(var_21, var_22)
    var_24 = list(var_23)
    var_25 = lambda x: x == var_11
    var_26 = range(var_2)
    var_27 = module_0.drop_until(var_25, var_26)
    var_28 = list(var_27)
    var_29 = range(var_2)
    var_30 = [Custom(i) for i in var_29]
    var_31 = lambda x: x.val > var_0
    var_32 = module_0.drop_until(var_31, var_30)
    var_33 = list(var_32)
    var_34 = 'c'
    var_35 = lambda x: x == var_34
    var_36 = 'abcdefg'
    var_37 = module_0.drop_until(var_35, var_36)
    var_38 = list(var_37)



# Parsed testcases at query #6
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = 5
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = module_0.drop(var_0, var_6)
    var_8 = list(var_7)
    var_9 = [var_1, var_2, var_3, var_4, var_5]
    var_10 = module_0.drop(var_5, var_9)
    var_11 = list(var_10)
    var_12 = [var_1, var_2, var_3, var_4, var_5]
    var_13 = module_0.drop(var_2, var_12)
    var_14 = list(var_13)
    var_15 = []
    var_16 = module_0.drop(var_3, var_15)
    var_17 = list(var_16)
    var_18 = 10
    var_19 = [var_1, var_2, var_3]
    var_20 = module_0.drop(var_18, var_19)
    var_21 = list(var_20)
    var_22 = -1
    var_23 = 1
    var_24 = 2
    var_25 = 3
    var_26 = [var_23, var_24, var_25]
    var_27 = module_0.drop(var_22, var_26)
    var_28 = list(var_27)
    var_29 = [var_23, var_24, var_25, var_26, var_27]
    var_30 = 'hello'
    var_31 = module_0.drop(var_24, var_30)
    var_32 = list(var_31)



# Parsed testcases at query #7
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
    var_17 = module_0.take(var_4, var_16)
    var_18 = list(var_17)
    var_19 = range(var_15)
    var_20 = module_0.take(var_8, var_19)
    var_21 = list(var_20)
    var_22 = 100
    var_23 = [var_1, var_2, var_0]
    var_24 = module_0.take(var_22, var_23)
    var_25 = list(var_24)
    var_26 = []
    var_27 = module_0.take(var_4, var_26)
    var_28 = list(var_27)
    var_29 = -1
    var_30 = 1
    var_31 = 2
    var_32 = 3
    var_33 = [var_30, var_31, var_32]
    var_34 = module_0.take(var_29, var_33)
    var_35 = list(var_34)



# Parsed testcases at query #8
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
    var_9 = [x * var_8 for x in var_7]
    var_10 = module_0.LazyList(var_9)
    var_11 = 5
    var_12 = range(var_11)
    var_13 = module_0.LazyList(var_12)
    var_14 = list(var_13)



# Parsed testcases at query #9
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = 10
    var_2 = range(var_1)
    var_3 = module_0.chunk(var_0, var_2)
    var_4 = list(var_3)
    var_5 = 5
    var_6 = range(var_1)
    var_7 = module_0.chunk(var_5, var_6)
    var_8 = list(var_7)
    var_9 = []
    var_10 = module_0.chunk(var_0, var_9)
    var_11 = list(var_10)
    var_12 = range(var_5)
    var_13 = module_0.chunk(var_1, var_12)
    var_14 = list(var_13)
    var_15 = 1
    var_16 = range(var_5)
    var_17 = module_0.chunk(var_15, var_16)
    var_18 = list(var_17)
    var_19 = 2
    var_20 = 'a'
    var_21 = 'b'
    var_22 = 'c'
    var_23 = 'd'
    var_24 = 'e'
    var_25 = [var_20, var_21, var_22, var_23, var_24]
    var_26 = module_0.chunk(var_19, var_25)
    var_27 = list(var_26)
    var_28 = 0
    var_29 = 10
    var_30 = range(var_29)
    var_31 = module_0.chunk(var_28, var_30)
    var_32 = list(var_31)
    var_33 = -1
    var_34 = 10
    var_35 = range(var_34)
    var_36 = module_0.chunk(var_33, var_35)
    var_37 = list(var_36)



# Parsed testcases at query #10
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



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 5
    var_2 = 0
    var_3 = 2
    var_4 = 10
    var_5 = -11



# Parsed testcases at query #12
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



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 0
    var_3 = 10
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = -1



# Parsed testcases at query #14
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
    var_7 = 5
    var_8 = range(var_7)
    var_9 = 2
    var_10 = [x * var_9 for x in var_8]
    var_11 = module_0.LazyList(var_10)
    var_12 = 100
    var_13 = range(var_12)
    var_14 = module_0.LazyList(var_13)
    var_15 = var_14.list
    var_16 = len(var_15)
    assert var_16 == 100



# Parsed testcases at query #15
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
    var_11 = 0
    var_12 = lambda x: x < var_11
    var_13 = range(var_2)
    var_14 = module_0.drop_until(var_12, var_13)
    var_15 = list(var_14)
    var_16 = lambda x: x > var_0
    var_17 = []
    var_18 = module_0.drop_until(var_16, var_17)
    var_19 = list(var_18)
    var_20 = lambda x: x >= var_11
    var_21 = range(var_2)
    var_22 = module_0.drop_until(var_20, var_21)
    var_23 = list(var_22)
    var_24 = range(var_2)
    var_25 = list(var_24)
    var_26 = lambda x: x < var_11
    var_27 = range(var_2)
    var_28 = module_0.drop_until(var_26, var_27)
    var_29 = list(var_28)
    var_30 = 'c'
    var_31 = lambda x: x == var_30
    var_32 = 'abcdef'
    var_33 = module_0.drop_until(var_31, var_32)
    var_34 = list(var_33)
    var_35 = 'z'
    var_36 = lambda x: x == var_35
    var_37 = module_0.drop_until(var_36, var_32)
    var_38 = list(var_37)
    var_39 = range(var_2)
    var_40 = [Custom(i) for i in var_39]
    var_41 = lambda x: x.val > var_0
    var_42 = module_0.drop_until(var_41, var_40)
    var_43 = list(var_42)



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 0
    var_3 = 10
    var_4 = 2
    var_5 = -1
    var_6 = 3



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 10
    var_4 = -10



# Parsed testcases at query #18
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
    var_26 = 0
    var_27 = lambda x: x >= var_26
    var_28 = [var_8, var_9, var_6]
    var_29 = module_0.drop_until(var_27, var_28)
    var_30 = list(var_29)
    var_31 = range(var_0)
    var_32 = [CustomObj(i) for i in var_31]
    var_33 = lambda x: x.val == var_9
    var_34 = module_0.drop_until(var_33, var_32)
    var_35 = list(var_34)



# Parsed testcases at query #19
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
    var_14 = lambda x: x > var_2
    var_15 = range(var_0)
    var_16 = module_0.drop_until(var_14, var_15)
    var_17 = list(var_16)
    var_18 = 0
    var_19 = lambda x: x > var_18
    var_20 = []
    var_21 = module_0.drop_until(var_19, var_20)
    var_22 = list(var_21)
    var_23 = lambda x: x == var_8
    var_24 = [var_8, var_9, var_6]
    var_25 = module_0.drop_until(var_23, var_24)
    var_26 = list(var_25)
    var_27 = lambda x: x > var_2
    var_28 = [var_8, var_9, var_6]
    var_29 = module_0.drop_until(var_27, var_28)
    var_30 = list(var_29)
    var_31 = range(var_0)
    var_32 = [CustomObj(i) for i in var_31]
    var_33 = lambda x: x.val == var_9
    var_34 = module_0.drop_until(var_33, var_32)
    var_35 = list(var_34)



# Parsed testcases at query #20
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
    var_7 = 5
    var_8 = 15
    var_9 = range(var_7, var_8)
    var_10 = 3
    var_11 = range(var_10)
    var_12 = module_0.LazyList(var_11)
    var_13 = 2
    var_14 = var_12[var_13]



# Parsed testcases at query #21
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = []
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = 0
    var_6 = lambda x: x > var_5
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = 4
    var_11 = [var_7, var_8, var_9, var_10]
    var_12 = module_0.drop_until(var_6, var_11)
    var_13 = list(var_12)
    var_14 = 10
    var_15 = lambda x: x > var_14
    var_16 = [var_7, var_8, var_9, var_10]
    var_17 = module_0.drop_until(var_15, var_16)
    var_18 = list(var_17)
    var_19 = lambda x: x > var_8
    var_20 = [var_7, var_8, var_9, var_10, var_0]
    var_21 = module_0.drop_until(var_19, var_20)
    var_22 = list(var_21)
    var_23 = lambda x: x > var_5
    var_24 = [var_7, var_8, var_9, var_10]
    var_25 = module_0.drop_until(var_23, var_24)
    var_26 = list(var_25)
    var_27 = lambda x: x > var_10
    var_28 = [var_7, var_8, var_9, var_10, var_0]
    var_29 = module_0.drop_until(var_27, var_28)
    var_30 = list(var_29)
    var_31 = lambda x: x % var_8 == var_5
    var_32 = 6
    var_33 = 7
    var_34 = 8
    var_35 = [var_7, var_9, var_0, var_32, var_33, var_34]
    var_36 = module_0.drop_until(var_31, var_35)
    var_37 = list(var_36)
    var_38 = 'c'
    var_39 = lambda x: x == var_38
    var_40 = 'abcdef'
    var_41 = module_0.drop_until(var_39, var_40)
    var_42 = list(var_41)
    var_43 = range(var_14)
    var_44 = lambda x: x > var_0



# Parsed testcases at query #22
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
    var_7 = []
    var_8 = lambda x: x % var_2 == var_3
    var_9 = module_0.split_by(var_7, criterion=var_8)
    var_10 = list(var_9)
    var_11 = 1
    var_12 = 2
    var_13 = [var_11, var_12, var_2]
    var_14 = lambda x: x == var_12
    var_15 = module_0.split_by(var_13, criterion=var_14)
    var_16 = list(var_15)
    var_17 = ' Split by: '
    var_18 = ' '
    var_19 = module_0.split_by(var_17, separator=var_18)
    var_20 = list(var_19)
    var_21 = []
    var_22 = module_0.split_by(var_21, separator=var_18)
    var_23 = list(var_22)
    var_24 = [var_11, var_12, var_2]
    var_25 = module_0.split_by(var_24, separator=var_12)
    var_26 = list(var_25)
    var_27 = True
    var_28 = module_0.split_by(var_17, var_27, separator=var_18)
    var_29 = list(var_28)
    var_30 = 4
    var_31 = [var_27, var_12, var_2, var_12, var_30]
    var_32 = True
    var_33 = module_0.split_by(var_31, var_32, separator=var_12)
    var_34 = list(var_33)
    var_35 = 1
    var_36 = 2
    var_37 = 3
    var_38 = [var_35, var_36, var_37]
    var_39 = lambda x: x == var_36
    var_40 = module_0.split_by(var_38, criterion=var_39, separator=var_36)
    var_41 = list(var_40)
    var_42 = 1
    var_43 = 2
    var_44 = 3
    var_45 = [var_42, var_43, var_44]
    var_46 = module_0.split_by(var_45)
    var_47 = list(var_46)



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 5
    var_2 = 0
    var_3 = 2
    var_4 = 5
    var_5 = 10
    var_6 = 5
    var_7 = -10



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
    var_18 = 'a..b..c'
    var_19 = '.'
    var_20 = module_0.split_by(var_18, separator=var_19)
    var_21 = list(var_20)
    var_22 = module_0.split_by(var_18, var_8, separator=var_19)
    var_23 = list(var_22)
    var_24 = 'abc'
    var_25 = module_0.split_by(var_24, separator=var_19)
    var_26 = list(var_25)
    var_27 = module_0.split_by(var_24, var_8, separator=var_19)
    var_28 = list(var_27)
    var_29 = '...'
    var_30 = module_0.split_by(var_29, separator=var_19)
    var_31 = list(var_30)
    var_32 = module_0.split_by(var_29, var_8, separator=var_19)
    var_33 = list(var_32)
    var_34 = []
    var_35 = module_0.split_by(var_34, separator=var_19)
    var_36 = list(var_35)
    var_37 = []
    var_38 = module_0.split_by(var_37, var_8, separator=var_19)
    var_39 = list(var_38)
    var_40 = []
    var_41 = lambda x: x
    var_42 = '.'
    var_43 = module_0.split_by(var_40, criterion=var_41, separator=var_42)
    var_44 = list(var_43)



# Parsed testcases at query #25
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
    var_7 = []
    var_8 = lambda x: x % var_2 == var_3
    var_9 = module_0.split_by(var_7, criterion=var_8)
    var_10 = list(var_9)
    var_11 = range(var_0)
    var_12 = True
    var_13 = lambda x: x % var_2 == var_3
    var_14 = module_0.split_by(var_11, var_12, criterion=var_13)
    var_15 = list(var_14)
    var_16 = ' Split by: '
    var_17 = ' '
    var_18 = module_0.split_by(var_16, separator=var_17)
    var_19 = list(var_18)
    var_20 = '.'
    var_21 = module_0.split_by(var_16, var_12, separator=var_20)
    var_22 = list(var_21)
    var_23 = ''
    var_24 = module_0.split_by(var_23, separator=var_17)
    var_25 = list(var_24)
    var_26 = 'a.b.c'
    var_27 = module_0.split_by(var_26, separator=var_20)
    var_28 = list(var_27)
    var_29 = 10
    var_30 = range(var_29)
    var_31 = 3
    var_32 = 0
    var_33 = lambda x: x % var_31 == var_32
    var_34 = '.'
    var_35 = module_0.split_by(var_30, criterion=var_33, separator=var_34)
    var_36 = list(var_35)
    var_37 = 10
    var_38 = range(var_37)
    var_39 = module_0.split_by(var_38)
    var_40 = list(var_39)



# Parsed testcases at query #26
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = []
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = 0
    var_6 = lambda x: x >= var_5
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = [var_7, var_8, var_9]
    var_11 = module_0.drop_until(var_6, var_10)
    var_12 = list(var_11)
    var_13 = 10
    var_14 = lambda x: x > var_13
    var_15 = [var_7, var_8, var_9]
    var_16 = module_0.drop_until(var_14, var_15)
    var_17 = list(var_16)
    var_18 = lambda x: x > var_0
    var_19 = 4
    var_20 = 6
    var_21 = 7
    var_22 = 8
    var_23 = [var_7, var_8, var_9, var_19, var_0, var_20, var_21, var_22]
    var_24 = module_0.drop_until(var_18, var_23)
    var_25 = list(var_24)
    var_26 = lambda x: x == var_7
    var_27 = [var_7, var_8, var_9]
    var_28 = module_0.drop_until(var_26, var_27)
    var_29 = list(var_28)
    var_30 = lambda x: x == var_9
    var_31 = [var_7, var_8, var_9]
    var_32 = module_0.drop_until(var_30, var_31)
    var_33 = list(var_32)
    var_34 = lambda x: x.value > var_7



# Parsed testcases at query #27
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



# Parsed testcases at query #28
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
    var_8 = module_0.LazyList(var_7)
    var_9 = 3
    var_10 = range(var_9)
    var_11 = module_0.LazyList(var_10)
    var_12 = list(var_11)



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
    var_8 = 10
    var_9 = var_7[var_8]
    var_10 = -10
    var_11 = var_7[var_10]



# Parsed testcases at query #30
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
    var_15 = lambda x: x < var_14
    var_16 = [var_8, var_9, var_6]
    var_17 = module_0.drop_until(var_15, var_16)
    var_18 = list(var_17)
    var_19 = lambda x: x > var_0
    var_20 = []
    var_21 = module_0.drop_until(var_19, var_20)
    var_22 = list(var_21)
    var_23 = lambda x: x >= var_14
    var_24 = [var_8, var_9, var_6]
    var_25 = module_0.drop_until(var_23, var_24)
    var_26 = list(var_25)
    var_27 = lambda x: x < var_14
    var_28 = [var_8, var_9, var_6]
    var_29 = module_0.drop_until(var_27, var_28)
    var_30 = list(var_29)
    var_31 = lambda x: x == var_8
    var_32 = [var_8, var_9, var_6]
    var_33 = module_0.drop_until(var_31, var_32)
    var_34 = list(var_33)
    var_35 = lambda x: x.val > var_8



# Parsed testcases at query #31
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
    var_8 = lambda x: x ** var_1
    var_9 = module_0.MapList(var_8, var_5)
    var_10 = 'a'
    var_11 = 'b'
    var_12 = 'c'
    var_13 = [var_10, var_11, var_12]
    var_14 = lambda x: x.upper()
    var_15 = module_0.MapList(var_14, var_13)



# Parsed testcases at query #32
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



# Parsed testcases at query #33
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
    var_23 = module_0.split_by(var_22, separator=var_13)
    var_24 = list(var_23)
    var_25 = 6
    var_26 = 9
    var_27 = [var_2, var_25, var_26]
    var_28 = lambda x: x % var_2 == var_3
    var_29 = module_0.split_by(var_27, criterion=var_28)
    var_30 = list(var_29)
    var_31 = [var_2, var_25, var_26]
    var_32 = False
    var_33 = lambda x: x % var_2 == var_32
    var_34 = module_0.split_by(var_31, var_32, criterion=var_33)
    var_35 = list(var_34)
    var_36 = 2
    var_37 = 4
    var_38 = 5
    var_39 = [var_8, var_36, var_37, var_38]
    var_40 = lambda x: x % var_2 == var_32
    var_41 = module_0.split_by(var_39, criterion=var_40)
    var_42 = list(var_41)
    var_43 = 1
    var_44 = 2
    var_45 = 3
    var_46 = [var_43, var_44, var_45]
    var_47 = 0
    var_48 = lambda x: x % var_45 == var_47
    var_49 = ' '
    var_50 = module_0.split_by(var_46, criterion=var_48, separator=var_49)
    var_51 = list(var_50)
    var_52 = 1
    var_53 = 2
    var_54 = 3
    var_55 = [var_52, var_53, var_54]
    var_56 = module_0.split_by(var_55)
    var_57 = list(var_56)



# Parsed testcases at query #34
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



# Parsed testcases at query #35
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
    var_7 = 1
    var_8 = 2
    var_9 = 4
    var_10 = 5
    var_11 = [var_7, var_8, var_2, var_9, var_10]
    var_12 = lambda x: x == var_2
    var_13 = module_0.split_by(var_11, criterion=var_12)
    var_14 = list(var_13)
    var_15 = [var_7, var_8, var_2, var_9, var_10]
    var_16 = True
    var_17 = lambda x: x == var_2
    var_18 = module_0.split_by(var_15, var_16, criterion=var_17)
    var_19 = list(var_18)
    var_20 = ' Split by: '
    var_21 = ' '
    var_22 = module_0.split_by(var_20, separator=var_21)
    var_23 = list(var_22)
    var_24 = True
    var_25 = module_0.split_by(var_20, var_24, separator=var_21)
    var_26 = list(var_25)
    var_27 = 1
    var_28 = 2
    var_29 = 3
    var_30 = [var_27, var_28, var_29]
    var_31 = module_0.split_by(var_30)
    var_32 = list(var_31)
    var_33 = 1
    var_34 = 2
    var_35 = 3
    var_36 = [var_33, var_34, var_35]
    var_37 = lambda x: x == var_34
    var_38 = module_0.split_by(var_36, criterion=var_37, separator=var_34)
    var_39 = list(var_38)



# Parsed testcases at query #36
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
    var_7 = []
    var_8 = lambda x: x % var_2 == var_3
    var_9 = module_0.split_by(var_7, criterion=var_8)
    var_10 = list(var_9)
    var_11 = 1
    var_12 = 2
    var_13 = 4
    var_14 = [var_11, var_12, var_2, var_13]
    var_15 = 5
    var_16 = lambda x: x % var_15 == var_3
    var_17 = module_0.split_by(var_14, criterion=var_16)
    var_18 = list(var_17)
    var_19 = ' Split by: '
    var_20 = True
    var_21 = ' '
    var_22 = module_0.split_by(var_19, var_20, separator=var_21)
    var_23 = list(var_22)
    var_24 = [var_20, var_12, var_2, var_12, var_13]
    var_25 = module_0.split_by(var_24, separator=var_12)
    var_26 = list(var_25)
    var_27 = [var_20, var_12, var_2, var_12, var_13]
    var_28 = True
    var_29 = module_0.split_by(var_27, var_28, separator=var_12)
    var_30 = list(var_29)
    var_31 = [var_28, var_12, var_2, var_12, var_13]
    var_32 = False
    var_33 = module_0.split_by(var_31, var_32, separator=var_12)
    var_34 = list(var_33)
    var_35 = [var_12, var_28, var_12, var_2, var_12]
    var_36 = False
    var_37 = module_0.split_by(var_35, var_36, separator=var_12)
    var_38 = list(var_37)
    var_39 = 1
    var_40 = 2
    var_41 = 3
    var_42 = [var_39, var_40, var_41]
    var_43 = lambda x: x > var_39
    var_44 = module_0.split_by(var_42, criterion=var_43, separator=var_40)
    var_45 = list(var_44)
    var_46 = 1
    var_47 = 2
    var_48 = 3
    var_49 = [var_46, var_47, var_48]
    var_50 = module_0.split_by(var_49)
    var_51 = list(var_50)



# Parsed testcases at query #37
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 10
    var_4 = -10



# Parsed testcases at query #38
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = 4
    var_5 = 5
    var_6 = [var_2, var_0, var_3, var_4, var_5]
    var_7 = module_0.MapList(var_1, var_6)
    var_8 = lambda x: x ** var_0 + var_2
    var_9 = 0
    var_10 = [var_9, var_2, var_0, var_3, var_4]
    var_11 = module_0.MapList(var_8, var_10)
    var_12 = lambda s: s.upper()
    var_13 = 'a'
    var_14 = 'b'
    var_15 = 'c'
    var_16 = 'd'
    var_17 = [var_13, var_14, var_15, var_16]
    var_18 = module_0.MapList(var_12, var_17)



# Parsed testcases at query #39
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 10
    var_4 = -11



# Parsed testcases at query #40
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 10
    var_4 = -11



# Parsed testcases at query #41
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
    var_19 = 5
    var_20 = lambda x: x > var_19
    var_21 = module_0.split_by(var_18, criterion=var_20)
    var_22 = list(var_21)
    var_23 = []
    var_24 = module_0.split_by(var_23, separator=var_19)
    var_25 = list(var_24)
    var_26 = 2
    var_27 = [var_8, var_26, var_2]
    var_28 = lambda x: x > var_3
    var_29 = module_0.split_by(var_27, criterion=var_28)
    var_30 = list(var_29)
    var_31 = [var_8, var_26, var_2]
    var_32 = False
    var_33 = lambda x: x > var_32
    var_34 = module_0.split_by(var_31, var_32, criterion=var_33)
    var_35 = list(var_34)
    var_36 = [var_8, var_26, var_2]
    var_37 = lambda x: x > var_19
    var_38 = module_0.split_by(var_36, criterion=var_37)
    var_39 = list(var_38)
    var_40 = 1
    var_41 = 2
    var_42 = 3
    var_43 = [var_40, var_41, var_42]
    var_44 = 5
    var_45 = lambda x: x > var_44
    var_46 = module_0.split_by(var_43, criterion=var_45, separator=var_44)
    var_47 = list(var_46)
    var_48 = 1
    var_49 = 2
    var_50 = 3
    var_51 = [var_48, var_49, var_50]
    var_52 = module_0.split_by(var_51)
    var_53 = list(var_52)



# Parsed testcases at query #42
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 10
    var_4 = -11



# Parsed testcases at query #43
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
    var_7 = []
    var_8 = lambda x: x % var_2 == var_3
    var_9 = module_0.split_by(var_7, criterion=var_8)
    var_10 = list(var_9)
    var_11 = range(var_0)
    var_12 = True
    var_13 = lambda x: x % var_2 == var_3
    var_14 = module_0.split_by(var_11, var_12, criterion=var_13)
    var_15 = list(var_14)
    var_16 = ' Split by: '
    var_17 = ' '
    var_18 = module_0.split_by(var_16, separator=var_17)
    var_19 = list(var_18)
    var_20 = module_0.split_by(var_16, var_12, separator=var_17)
    var_21 = list(var_20)
    var_22 = []
    var_23 = module_0.split_by(var_22, separator=var_17)
    var_24 = list(var_23)
    var_25 = 10
    var_26 = range(var_25)
    var_27 = module_0.split_by(var_26)
    var_28 = list(var_27)
    var_29 = 10
    var_30 = range(var_29)
    var_31 = 3
    var_32 = 0
    var_33 = lambda x: x % var_31 == var_32
    var_34 = ' '
    var_35 = module_0.split_by(var_30, criterion=var_33, separator=var_34)
    var_36 = list(var_35)



# Parsed testcases at query #44
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
    var_16 = var_15[var_8]



# Parsed testcases at query #45
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
    var_8 = module_0.LazyList(var_7)
    var_9 = range(var_5)
    var_10 = module_0.LazyList(var_9)
    var_11 = 5
    var_12 = var_10[var_11]
    var_13 = range(var_5)
    var_14 = module_0.LazyList(var_13)
    var_15 = list(var_14)



# Parsed testcases at query #46
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 10
    var_3 = 2
    var_4 = 0
    var_5 = -1



# Parsed testcases at query #47
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
    var_8 = []
    var_9 = lambda x: x * var_1
    var_10 = module_0.MapList(var_9, var_8)
    var_11 = 'a'
    var_12 = 'b'
    var_13 = 'c'
    var_14 = [var_11, var_12, var_13]
    var_15 = lambda x: x.upper()
    var_16 = module_0.MapList(var_15, var_14)



# Parsed testcases at query #48
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 10
    var_3 = 2
    var_4 = -1



# Parsed testcases at query #49
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



# Parsed testcases at query #50
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
    var_14 = module_0.split_by(var_12, var_8, separator=var_13)
    var_15 = list(var_14)
    var_16 = module_0.split_by(var_12, separator=var_13)
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
    var_27 = [var_3, var_2, var_25, var_26]
    var_28 = lambda x: x % var_2 == var_3
    var_29 = module_0.split_by(var_27, criterion=var_28)
    var_30 = list(var_29)
    var_31 = [var_3, var_2, var_25, var_26]
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



# Parsed testcases at query #51
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



# Parsed testcases at query #52
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
    var_8 = []
    var_9 = lambda x: x * var_1
    var_10 = module_0.MapList(var_9, var_8)
    var_11 = [var_0, var_1, var_2, var_3, var_4]
    var_12 = lambda x: x * x
    var_13 = module_0.MapList(var_12, var_11)



# Parsed testcases at query #53
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 11
    var_3 = 2
    var_4 = 5
    var_5 = 0
    var_6 = 0
    var_7 = 10
    var_8 = -10



# Parsed testcases at query #54
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = []
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = 0
    var_6 = lambda x: x >= var_5
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = 4
    var_11 = [var_7, var_8, var_9, var_10]
    var_12 = module_0.drop_until(var_6, var_11)
    var_13 = list(var_12)
    var_14 = 10
    var_15 = lambda x: x > var_14
    var_16 = [var_7, var_8, var_9, var_10]
    var_17 = module_0.drop_until(var_15, var_16)
    var_18 = list(var_17)
    var_19 = lambda x: x > var_0
    var_20 = 6
    var_21 = 7
    var_22 = 8
    var_23 = [var_7, var_8, var_9, var_10, var_0, var_20, var_21, var_22]
    var_24 = module_0.drop_until(var_19, var_23)
    var_25 = list(var_24)
    var_26 = lambda x: x == var_7
    var_27 = [var_7, var_8, var_9, var_10]
    var_28 = module_0.drop_until(var_26, var_27)
    var_29 = list(var_28)
    var_30 = lambda obj: obj.val > var_8
    var_31 = range(var_14)
    var_32 = lambda x: x == var_0



# Parsed testcases at query #55
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = lambda x: x * var_1
    var_6 = module_0.MapList(var_5, var_4)
    var_7 = 4
    var_8 = var_6[var_7]
    var_9 = -5
    var_10 = var_6[var_9]



# Parsed testcases at query #56
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
    var_8 = lambda x: x * x



# Parsed testcases at query #57
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



# Parsed testcases at query #58
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
    var_18 = 8
    var_19 = [var_8, var_6, var_0, var_16, var_17, var_18]
    var_20 = module_0.drop_until(var_15, var_19)
    var_21 = list(var_20)
    var_22 = lambda x: x > var_14
    var_23 = []
    var_24 = module_0.drop_until(var_22, var_23)
    var_25 = list(var_24)
    var_26 = lambda x: x >= var_14
    var_27 = [var_8, var_9, var_6]
    var_28 = module_0.drop_until(var_26, var_27)
    var_29 = list(var_28)
    var_30 = lambda x: x < var_14
    var_31 = [var_8, var_9, var_6]
    var_32 = module_0.drop_until(var_30, var_31)
    var_33 = list(var_32)
    var_34 = lambda x: x == var_8
    var_35 = [var_8, var_9, var_6]
    var_36 = module_0.drop_until(var_34, var_35)
    var_37 = list(var_36)
    var_38 = lambda x: x.value > var_9



# Parsed testcases at query #59
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 10
    var_4 = -11



# Parsed testcases at query #60
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
    var_8 = 5
    var_9 = var_7[var_8]
    var_10 = -6
    var_11 = var_7[var_10]



# Parsed testcases at query #61
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 10
    var_4 = var_2[var_3]
    var_5 = 20
    var_6 = 30
    var_7 = 40
    var_8 = 50
    var_9 = [var_3, var_5, var_6, var_7, var_8]
    var_10 = 5
    var_11 = range(var_10)
    var_12 = module_0.LazyList(var_11)
    var_13 = 4
    var_14 = var_12[var_13]
    var_15 = len(var_12)
    assert var_15 == 5



# Parsed testcases at query #62
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = []
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = 0
    var_6 = lambda x: x >= var_5
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = [var_7, var_8, var_9]
    var_11 = module_0.drop_until(var_6, var_10)
    var_12 = list(var_11)
    var_13 = 10
    var_14 = lambda x: x > var_13
    var_15 = [var_7, var_8, var_9]
    var_16 = module_0.drop_until(var_14, var_15)
    var_17 = list(var_16)
    var_18 = lambda x: x > var_0
    var_19 = 4
    var_20 = 6
    var_21 = 7
    var_22 = 8
    var_23 = [var_7, var_8, var_9, var_19, var_0, var_20, var_21, var_22]
    var_24 = module_0.drop_until(var_18, var_23)
    var_25 = list(var_24)
    var_26 = lambda x: x == var_7
    var_27 = [var_7, var_8, var_9, var_19]
    var_28 = module_0.drop_until(var_26, var_27)
    var_29 = list(var_28)
    var_30 = lambda x: x == var_19
    var_31 = [var_7, var_8, var_9, var_19]
    var_32 = module_0.drop_until(var_30, var_31)
    var_33 = list(var_32)
    var_34 = lambda x: x % var_8 == var_5
    var_35 = [var_7, var_9, var_0, var_20, var_21, var_22]
    var_36 = module_0.drop_until(var_34, var_35)
    var_37 = list(var_36)
    var_38 = lambda x: x.value > var_8



# Parsed testcases at query #63
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



# Parsed testcases at query #64
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
    var_12 = []
    var_13 = lambda x: x % var_2 == var_3
    var_14 = module_0.split_by(var_12, criterion=var_13)
    var_15 = list(var_14)
    var_16 = 2
    var_17 = [var_8, var_16, var_2]
    var_18 = lambda x: x == var_16
    var_19 = module_0.split_by(var_17, criterion=var_18)
    var_20 = list(var_19)
    var_21 = ' Split by: '
    var_22 = ' '
    var_23 = module_0.split_by(var_21, separator=var_22)
    var_24 = list(var_23)
    var_25 = module_0.split_by(var_21, var_8, separator=var_22)
    var_26 = list(var_25)
    var_27 = 4
    var_28 = [var_8, var_16, var_2, var_16, var_27]
    var_29 = module_0.split_by(var_28, separator=var_16)
    var_30 = list(var_29)
    var_31 = [var_8, var_16, var_2, var_16, var_27]
    var_32 = module_0.split_by(var_31, var_8, separator=var_16)
    var_33 = list(var_32)
    var_34 = 10
    var_35 = range(var_34)
    var_36 = module_0.split_by(var_35)
    var_37 = list(var_36)
    var_38 = 10
    var_39 = range(var_38)
    var_40 = 3
    var_41 = 0
    var_42 = lambda x: x % var_40 == var_41
    var_43 = 2
    var_44 = module_0.split_by(var_39, criterion=var_42, separator=var_43)
    var_45 = list(var_44)



# Parsed testcases at query #65
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
    var_7 = []
    var_8 = lambda x: x % var_2 == var_3
    var_9 = module_0.split_by(var_7, criterion=var_8)
    var_10 = list(var_9)
    var_11 = 1
    var_12 = 2
    var_13 = 4
    var_14 = 5
    var_15 = [var_11, var_12, var_2, var_13, var_14]
    var_16 = lambda x: x == var_2
    var_17 = module_0.split_by(var_15, criterion=var_16)
    var_18 = list(var_17)
    var_19 = ' Split by: '
    var_20 = True
    var_21 = '.'
    var_22 = module_0.split_by(var_19, var_20, separator=var_21)
    var_23 = list(var_22)
    var_24 = [var_20, var_12, var_2, var_13, var_14]
    var_25 = module_0.split_by(var_24, separator=var_2)
    var_26 = list(var_25)
    var_27 = []
    var_28 = module_0.split_by(var_27, separator=var_2)
    var_29 = list(var_28)
    var_30 = [var_20, var_12, var_2, var_13, var_14]
    var_31 = True
    var_32 = module_0.split_by(var_30, var_31, separator=var_2)
    var_33 = list(var_32)
    var_34 = [var_2, var_2, var_2]
    var_35 = True
    var_36 = module_0.split_by(var_34, var_35, separator=var_2)
    var_37 = list(var_36)
    var_38 = [var_2, var_2, var_2]
    var_39 = False
    var_40 = module_0.split_by(var_38, var_39, separator=var_2)
    var_41 = list(var_40)
    var_42 = 1
    var_43 = 2
    var_44 = 3
    var_45 = [var_42, var_43, var_44]
    var_46 = lambda x: x == var_43
    var_47 = module_0.split_by(var_45, criterion=var_46, separator=var_43)
    var_48 = list(var_47)
    var_49 = 1
    var_50 = 2
    var_51 = 3
    var_52 = [var_49, var_50, var_51]
    var_53 = module_0.split_by(var_52)
    var_54 = list(var_53)



# Parsed testcases at query #66
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
    var_14 = 3
    var_15 = var_13[var_14]



# Parsed testcases at query #67
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = []
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = lambda x: x > var_0
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = module_0.drop_until(var_5, var_9)
    var_11 = list(var_10)
    var_12 = 5
    var_13 = lambda x: x > var_12
    var_14 = 10
    var_15 = range(var_14)
    var_16 = module_0.drop_until(var_13, var_15)
    var_17 = list(var_16)
    var_18 = 100
    var_19 = lambda x: x > var_18
    var_20 = range(var_14)
    var_21 = module_0.drop_until(var_19, var_20)
    var_22 = list(var_21)
    var_23 = 9
    var_24 = lambda x: x == var_23
    var_25 = range(var_14)
    var_26 = module_0.drop_until(var_24, var_25)
    var_27 = list(var_26)
    var_28 = range(var_12)
    var_29 = [TestObj(i) for i in var_28]
    var_30 = lambda x: x.val >= var_8
    var_31 = module_0.drop_until(var_30, var_29)
    var_32 = list(var_31)



# Parsed testcases at query #68
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
    var_11 = range(var_9)
    var_12 = module_0.LazyList(var_11)
    var_13 = list(var_12)



# Parsed testcases at query #69
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
    var_23 = module_0.split_by(var_22, separator=var_13)
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
    var_43 = lambda x: x % var_2 == var_3
    var_44 = module_0.split_by(var_42, var_8, criterion=var_43)
    var_45 = list(var_44)
    var_46 = 1
    var_47 = 2
    var_48 = 3
    var_49 = [var_46, var_47, var_48]
    var_50 = 0
    var_51 = lambda x: x % var_48 == var_50
    var_52 = ' '
    var_53 = module_0.split_by(var_49, criterion=var_51, separator=var_52)
    var_54 = list(var_53)
    var_55 = 1
    var_56 = 2
    var_57 = 3
    var_58 = [var_55, var_56, var_57]
    var_59 = module_0.split_by(var_58)
    var_60 = list(var_59)



# Parsed testcases at query #70
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
    var_13 = module_0.LazyList(var_12)
    var_14 = 2
    var_15 = var_13[var_14]
    var_16 = []
    var_17 = module_0.LazyList(var_16)
    var_18 = 0
    var_19 = var_17[var_18]
    var_20 = range(var_18)
    var_21 = module_0.LazyList(var_20)
    var_22 = var_21[var_11]



# Parsed testcases at query #71
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
    var_13 = 100
    var_14 = range(var_13)
    var_15 = module_0.LazyList(var_14)
    var_16 = var_15.list
    var_17 = len(var_16)
    assert var_17 == 0
    var_18 = var_15[var_11]
    var_19 = var_15.list
    var_20 = len(var_19)
    assert var_20 == 6
    var_21 = var_15[var_5]
    var_22 = var_15.list
    var_23 = len(var_22)
    assert var_23 == 11
    var_24 = range(var_13)
    var_25 = module_0.LazyList(var_24)
    var_26 = var_25.list
    var_27 = len(var_26)
    assert var_27 == 0
    var_28 = var_25[var_11:var_5]
    var_29 = var_25.list
    var_30 = len(var_29)
    assert var_30 == 10



# Parsed testcases at query #72
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



# Parsed testcases at query #73
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 0
    var_3 = 10
    var_4 = 2
    var_5 = 3
    var_6 = -1



# Parsed testcases at query #74
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 0
    var_3 = 10
    var_4 = 2
    var_5 = 3
    var_6 = -1



# Parsed testcases at query #75
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



# Parsed testcases at query #76
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
    var_34 = range(var_2)
    var_35 = [Custom(i) for i in var_34]
    var_36 = lambda x: x.value > var_0
    var_37 = module_0.drop_until(var_36, var_35)
    var_38 = list(var_37)



# Parsed testcases at query #77
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 10
    var_4 = -11



# Parsed testcases at query #78
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



# Parsed testcases at query #79
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 100
    var_4 = -100



# Parsed testcases at query #80
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
    var_8 = lambda x: x ** var_1
    var_9 = module_0.MapList(var_8, var_5)



# Parsed testcases at query #81
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



# Parsed testcases at query #82
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



# Parsed testcases at query #83
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
    var_7 = 'hello'
    var_8 = module_0.LazyList(var_7)
    var_9 = 5
    var_10 = range(var_9)
    var_11 = module_0.LazyList(var_10)
    var_12 = 4
    var_13 = var_11[var_12]
    var_14 = len(var_11)
    assert var_14 == 5



# Parsed testcases at query #84
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 10
    var_4 = var_2[var_3]
    var_5 = 20
    var_6 = 30
    var_7 = 40
    var_8 = 50
    var_9 = [var_3, var_5, var_6, var_7, var_8]
    var_10 = 5
    var_11 = range(var_10)
    var_12 = module_0.LazyList(var_11)
    var_13 = 4
    var_14 = var_12[var_13]
    var_15 = len(var_12)
    assert var_15 == 5



# Parsed testcases at query #85
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = 2
    var_7 = 0
    var_8 = lambda x: x % var_6 == var_7
    var_9 = 1
    var_10 = 3
    var_11 = 6
    var_12 = 7
    var_13 = 8
    var_14 = [var_9, var_10, var_0, var_11, var_12, var_13]
    var_15 = module_0.drop_until(var_8, var_14)
    var_16 = list(var_15)
    var_17 = 'c'
    var_18 = lambda x: x == var_17
    var_19 = 'a'
    var_20 = 'b'
    var_21 = 'd'
    var_22 = [var_19, var_20, var_17, var_21]
    var_23 = module_0.drop_until(var_18, var_22)
    var_24 = list(var_23)
    var_25 = lambda x: x == var_7
    var_26 = range(var_2)
    var_27 = module_0.drop_until(var_25, var_26)
    var_28 = list(var_27)
    var_29 = 100
    var_30 = lambda x: x > var_29
    var_31 = range(var_2)
    var_32 = module_0.drop_until(var_30, var_31)
    var_33 = list(var_32)
    var_34 = lambda x: x > var_0
    var_35 = []
    var_36 = module_0.drop_until(var_34, var_35)
    var_37 = list(var_36)
    var_38 = range(var_2)
    var_39 = [CustomObj(i) for i in var_38]
    var_40 = lambda x: x.val > var_0
    var_41 = module_0.drop_until(var_40, var_39)
    var_42 = list(var_41)



# Parsed testcases at query #86
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
    var_9 = lambda x: x * x



# Parsed testcases at query #87
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 10
    var_4 = -10



# Parsed testcases at query #88
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
    var_15 = lambda x: x < var_14
    var_16 = [var_8, var_9, var_6]
    var_17 = module_0.drop_until(var_15, var_16)
    var_18 = list(var_17)
    var_19 = lambda x: x > var_0
    var_20 = []
    var_21 = module_0.drop_until(var_19, var_20)
    var_22 = list(var_21)
    var_23 = lambda x: x >= var_14
    var_24 = [var_8, var_9, var_6]
    var_25 = module_0.drop_until(var_23, var_24)
    var_26 = list(var_25)
    var_27 = lambda x: x < var_14
    var_28 = [var_8, var_9, var_6]
    var_29 = module_0.drop_until(var_27, var_28)
    var_30 = list(var_29)
    var_31 = lambda x: x == var_8
    var_32 = [var_8, var_9, var_6]
    var_33 = module_0.drop_until(var_31, var_32)
    var_34 = list(var_33)
    var_35 = lambda x: x == var_6
    var_36 = [var_8, var_9, var_6]
    var_37 = module_0.drop_until(var_35, var_36)
    var_38 = list(var_37)
    var_39 = lambda x: x.value > var_8



# Parsed testcases at query #89
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = []
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = 0
    var_6 = lambda x: x >= var_5
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = [var_7, var_8, var_9]
    var_11 = module_0.drop_until(var_6, var_10)
    var_12 = list(var_11)
    var_13 = 10
    var_14 = lambda x: x > var_13
    var_15 = [var_7, var_8, var_9]
    var_16 = module_0.drop_until(var_14, var_15)
    var_17 = list(var_16)
    var_18 = lambda x: x > var_0
    var_19 = 4
    var_20 = 6
    var_21 = 7
    var_22 = 8
    var_23 = [var_7, var_8, var_9, var_19, var_0, var_20, var_21, var_22]
    var_24 = module_0.drop_until(var_18, var_23)
    var_25 = list(var_24)
    var_26 = lambda x: x > var_5
    var_27 = [var_7, var_8, var_9]
    var_28 = module_0.drop_until(var_26, var_27)
    var_29 = list(var_28)
    var_30 = lambda x: x == var_9
    var_31 = [var_7, var_8, var_9]
    var_32 = module_0.drop_until(var_30, var_31)
    var_33 = list(var_32)
    var_34 = lambda x: x.value > var_8



# Parsed testcases at query #90
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 10
    var_4 = -11



# Parsed testcases at query #91
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = lambda x: x * x
    var_7 = module_0.MapList(var_6, var_5)
    var_8 = 5
    var_9 = var_7[var_8]
    var_10 = -6
    var_11 = var_7[var_10]



# Parsed testcases at query #92
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



# Parsed testcases at query #93
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
    var_15 = lambda x: x < var_14
    var_16 = -1
    var_17 = -2
    var_18 = -3
    var_19 = [var_16, var_17, var_18]
    var_20 = module_0.drop_until(var_15, var_19)
    var_21 = list(var_20)
    var_22 = lambda x: x > var_0
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
    var_34 = 'b'
    var_35 = lambda x: x.startswith(var_34)
    var_36 = 'apple'
    var_37 = 'banana'
    var_38 = 'cherry'
    var_39 = [var_36, var_37, var_38]
    var_40 = module_0.drop_until(var_35, var_39)
    var_41 = list(var_40)
    var_42 = lambda x: len(x) > var_6
    var_43 = 'a'
    var_44 = 'ab'
    var_45 = 'abc'
    var_46 = 'abcd'
    var_47 = [var_43, var_44, var_45, var_46]
    var_48 = module_0.drop_until(var_42, var_47)
    var_49 = list(var_48)
    var_50 = lambda x: x.val == var_9



# Parsed testcases at query #94
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



# Parsed testcases at query #95
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 100
    var_4 = -100



# Parsed testcases at query #96
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
    var_15 = lambda x: x < var_14
    var_16 = [var_8, var_9, var_6]
    var_17 = module_0.drop_until(var_15, var_16)
    var_18 = list(var_17)
    var_19 = lambda x: x > var_14
    var_20 = []
    var_21 = module_0.drop_until(var_19, var_20)
    var_22 = list(var_21)
    var_23 = lambda x: x == var_8
    var_24 = [var_8, var_9, var_6]
    var_25 = module_0.drop_until(var_23, var_24)
    var_26 = list(var_25)
    var_27 = lambda x: x > var_2
    var_28 = [var_8, var_9, var_6]
    var_29 = module_0.drop_until(var_27, var_28)
    var_30 = list(var_29)
    var_31 = 'c'
    var_32 = lambda x: x == var_31
    var_33 = 'abcdef'
    var_34 = module_0.drop_until(var_32, var_33)
    var_35 = list(var_34)
    var_36 = 'z'
    var_37 = lambda x: x == var_36
    var_38 = 'abc'
    var_39 = module_0.drop_until(var_37, var_38)
    var_40 = list(var_39)
    var_41 = lambda x: x.val > var_9



# Parsed testcases at query #97
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 100
    var_4 = -100



# Parsed testcases at query #98
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = []
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = 0
    var_6 = lambda x: x > var_5
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = 4
    var_11 = [var_7, var_8, var_9, var_10]
    var_12 = module_0.drop_until(var_6, var_11)
    var_13 = list(var_12)
    var_14 = 10
    var_15 = lambda x: x > var_14
    var_16 = [var_7, var_8, var_9, var_10]
    var_17 = module_0.drop_until(var_15, var_16)
    var_18 = list(var_17)
    var_19 = lambda x: x > var_8
    var_20 = [var_7, var_8, var_9, var_10, var_0]
    var_21 = module_0.drop_until(var_19, var_20)
    var_22 = list(var_21)
    var_23 = lambda x: x > var_5
    var_24 = [var_7, var_8, var_9, var_10]
    var_25 = module_0.drop_until(var_23, var_24)
    var_26 = list(var_25)
    var_27 = lambda x: x > var_10
    var_28 = [var_7, var_8, var_9, var_10, var_0]
    var_29 = module_0.drop_until(var_27, var_28)
    var_30 = list(var_29)
    var_31 = lambda x: x % var_8 == var_5
    var_32 = 6
    var_33 = 7
    var_34 = 8
    var_35 = [var_7, var_9, var_0, var_32, var_33, var_34]
    var_36 = module_0.drop_until(var_31, var_35)
    var_37 = list(var_36)
    var_38 = 'c'
    var_39 = lambda x: x == var_38
    var_40 = 'abcdef'
    var_41 = module_0.drop_until(var_39, var_40)
    var_42 = list(var_41)
    var_43 = range(var_14)
    var_44 = lambda x: x > var_0



# Parsed testcases at query #99
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 5
    var_4 = range(var_3)
    var_5 = 2
    var_6 = [x * var_5 for x in var_4]
    var_7 = module_0.LazyList(var_6)
    var_8 = range(var_0)
    var_9 = module_0.LazyList(var_8)
    var_10 = 100
    var_11 = var_9[var_10]
    var_12 = -100
    var_13 = var_9[var_12]
    var_14 = range(var_12)
    var_15 = module_0.LazyList(var_14)



# Parsed testcases at query #100
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
    var_11 = []
    var_12 = module_0.LazyList(var_11)
    var_13 = 0
    var_14 = var_12[var_13]
    var_15 = range(var_7)
    var_16 = module_0.LazyList(var_15)
    var_17 = var_16[var_6]



# Parsed testcases at query #101
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = []
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = 0
    var_6 = lambda x: x >= var_5
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = [var_7, var_8, var_9]
    var_11 = module_0.drop_until(var_6, var_10)
    var_12 = list(var_11)
    var_13 = 10
    var_14 = lambda x: x > var_13
    var_15 = [var_7, var_8, var_9]
    var_16 = module_0.drop_until(var_14, var_15)
    var_17 = list(var_16)
    var_18 = lambda x: x > var_8
    var_19 = 4
    var_20 = [var_7, var_8, var_9, var_19, var_0]
    var_21 = module_0.drop_until(var_18, var_20)
    var_22 = list(var_21)
    var_23 = lambda x: x == var_7
    var_24 = [var_7, var_8, var_9]
    var_25 = module_0.drop_until(var_23, var_24)
    var_26 = list(var_25)
    var_27 = lambda x: x == var_9
    var_28 = [var_7, var_8, var_9]
    var_29 = module_0.drop_until(var_27, var_28)
    var_30 = list(var_29)
    var_31 = lambda x: x % var_8 == var_5
    var_32 = 6
    var_33 = [var_7, var_9, var_0, var_8, var_19, var_32]
    var_34 = module_0.drop_until(var_31, var_33)
    var_35 = list(var_34)
    var_36 = 'c'
    var_37 = lambda x: x == var_36
    var_38 = 'abcdef'
    var_39 = module_0.drop_until(var_37, var_38)
    var_40 = list(var_39)
    var_41 = lambda x: x.value > var_8



# Parsed testcases at query #102
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 10
    var_4 = var_2[var_3]
    var_5 = 20
    var_6 = 30
    var_7 = 40
    var_8 = 50
    var_9 = [var_3, var_5, var_6, var_7, var_8]
    var_10 = 5
    var_11 = range(var_10)
    var_12 = module_0.LazyList(var_11)
    var_13 = list(var_12)



# Parsed testcases at query #103
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
    var_21 = lambda x: x > var_14
    var_22 = []
    var_23 = module_0.drop_until(var_21, var_22)
    var_24 = list(var_23)
    var_25 = lambda x: x >= var_14
    var_26 = [var_8, var_9, var_6]
    var_27 = module_0.drop_until(var_25, var_26)
    var_28 = list(var_27)
    var_29 = lambda x: x < var_14
    var_30 = [var_8, var_9, var_6]
    var_31 = module_0.drop_until(var_29, var_30)
    var_32 = list(var_31)
    var_33 = lambda x: x == var_8
    var_34 = [var_8, var_9, var_6]
    var_35 = module_0.drop_until(var_33, var_34)
    var_36 = list(var_35)
    var_37 = lambda obj: obj.value == var_9



