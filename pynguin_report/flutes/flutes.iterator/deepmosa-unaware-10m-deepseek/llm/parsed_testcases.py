####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = 10
    var_3 = 2
    var_4 = 7
    var_5 = -3
    var_6 = 1
    var_7 = 3
    var_8 = 11
    var_9 = -1
    var_10 = -2
    var_11 = -3
    var_12 = -5
    var_13 = -10
    var_14 = -5
    var_15 = -10
    var_16 = -10
    var_17 = 15
    var_18 = 4
    var_19 = 6



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 2
    var_2 = 7
    var_3 = 1
    var_4 = 10
    var_5 = 0
    var_6 = -1
    var_7 = 3
    var_8 = -3
    var_9 = -2



# Parsed testcases at query #3
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = 10
    var_2 = range(var_1)
    var_3 = module_0.chunk(var_0, var_2)
    var_4 = list(var_3)
    var_5 = 5
    var_6 = 1
    var_7 = 2
    var_8 = [var_6, var_7, var_0]
    var_9 = module_0.chunk(var_5, var_8)
    var_10 = list(var_9)
    var_11 = [var_6, var_7, var_0]
    var_12 = module_0.chunk(var_0, var_11)
    var_13 = list(var_12)
    var_14 = [var_6, var_7, var_0]
    var_15 = module_0.chunk(var_6, var_14)
    var_16 = list(var_15)
    var_17 = []
    var_18 = module_0.chunk(var_0, var_17)
    var_19 = list(var_18)
    var_20 = 4
    var_21 = [var_6, var_7, var_0, var_20]
    var_22 = iter(var_21)
    var_23 = module_0.chunk(var_7, var_22)
    var_24 = list(var_23)
    var_25 = 'abcde'
    var_26 = module_0.chunk(var_7, var_25)
    var_27 = list(var_26)
    var_28 = range(var_5)
    var_29 = 0
    var_30 = 1
    var_31 = 2
    var_32 = 3
    var_33 = [var_30, var_31, var_32]
    var_34 = module_0.chunk(var_29, var_33)
    var_35 = list(var_34)
    var_36 = -1
    var_37 = 1
    var_38 = 2
    var_39 = 3
    var_40 = [var_37, var_38, var_39]
    var_41 = module_0.chunk(var_36, var_40)
    var_42 = list(var_41)



# Parsed testcases at query #4
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
    var_10 = lambda x: x > var_2
    var_11 = range(var_0)
    var_12 = module_0.drop_until(var_10, var_11)
    var_13 = list(var_12)
    var_14 = 0
    var_15 = lambda x: x >= var_14
    var_16 = range(var_0)
    var_17 = module_0.drop_until(var_15, var_16)
    var_18 = list(var_17)
    var_19 = 'l'
    var_20 = lambda c: c == var_19
    var_21 = 'hello world'
    var_22 = module_0.drop_until(var_20, var_21)
    var_23 = list(var_22)
    var_24 = 2
    var_25 = lambda x: x % var_24 == var_14
    var_26 = 1
    var_27 = 3
    var_28 = 6
    var_29 = 7
    var_30 = 8
    var_31 = [var_26, var_27, var_0, var_28, var_29, var_30]
    var_32 = module_0.drop_until(var_25, var_31)
    var_33 = list(var_32)
    var_34 = range(var_2)
    var_35 = lambda x: x > var_29
    var_36 = lambda x: x == var_0
    var_37 = [var_0]
    var_38 = module_0.drop_until(var_36, var_37)
    var_39 = list(var_38)
    var_40 = lambda x: x == var_0
    var_41 = [var_27]
    var_42 = module_0.drop_until(var_40, var_41)
    var_43 = list(var_42)
    var_44 = 'hello'
    var_45 = 'world'
    var_46 = [var_26, var_24, var_44, var_45]



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 2
    var_2 = 7
    var_3 = 1
    var_4 = 10
    var_5 = 0
    var_6 = -1
    var_7 = 20
    var_8 = 3
    var_9 = -3
    var_10 = -2



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
    var_6 = lambda x: x > var_0
    var_7 = []
    var_8 = module_0.drop_until(var_6, var_7)
    var_9 = list(var_8)
    var_10 = lambda x: x > var_2
    var_11 = range(var_0)
    var_12 = module_0.drop_until(var_10, var_11)
    var_13 = list(var_12)
    var_14 = 0
    var_15 = lambda x: x >= var_14
    var_16 = range(var_0)
    var_17 = module_0.drop_until(var_15, var_16)
    var_18 = list(var_17)
    var_19 = 'l'
    var_20 = lambda c: c == var_19
    var_21 = 'hello world'
    var_22 = module_0.drop_until(var_20, var_21)
    var_23 = list(var_22)
    var_24 = 2
    var_25 = 1
    var_26 = lambda x: x % var_24 == var_25
    var_27 = 4
    var_28 = 6
    var_29 = 7
    var_30 = 8
    var_31 = 9
    var_32 = [var_24, var_27, var_28, var_29, var_30, var_31]
    var_33 = module_0.drop_until(var_26, var_32)
    var_34 = list(var_33)
    var_35 = range(var_2)
    var_36 = lambda x: x > var_29
    var_37 = lambda x: x == var_0
    var_38 = [var_0]
    var_39 = module_0.drop_until(var_37, var_38)
    var_40 = list(var_39)
    var_41 = lambda x: x == var_0
    var_42 = 3
    var_43 = [var_42]
    var_44 = module_0.drop_until(var_41, var_43)
    var_45 = list(var_44)
    var_46 = True
    var_47 = lambda x: var_46
    var_48 = [var_46, var_24, var_42]
    var_49 = module_0.drop_until(var_47, var_48)
    var_50 = list(var_49)
    var_51 = False
    var_52 = lambda x: var_51
    var_53 = [var_46, var_24, var_42]
    var_54 = module_0.drop_until(var_52, var_53)
    var_55 = list(var_54)



# Parsed testcases at query #7
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
    var_7 = 4
    var_8 = 5
    var_9 = [var_1, var_2, var_3, var_7, var_8]
    var_10 = module_0.drop(var_2, var_9)
    var_11 = list(var_10)
    var_12 = 10
    var_13 = [var_1, var_2, var_3]
    var_14 = module_0.drop(var_12, var_13)
    var_15 = list(var_14)
    var_16 = [var_1, var_2, var_3]
    var_17 = module_0.drop(var_3, var_16)
    var_18 = list(var_17)
    var_19 = range(var_8)
    var_20 = []
    var_21 = module_0.drop(var_8, var_20)
    var_22 = list(var_21)
    var_23 = -1
    var_24 = 1
    var_25 = 2
    var_26 = 3
    var_27 = [var_24, var_25, var_26]
    var_28 = module_0.drop(var_23, var_27)
    var_29 = list(var_28)
    var_30 = 'hello'
    var_31 = module_0.drop(var_25, var_30)
    var_32 = list(var_31)
    var_33 = range(var_12)
    var_34 = module_0.drop(var_8, var_33)
    var_35 = list(var_34)
    var_36 = [var_24, var_25, var_26]
    var_37 = module_0.drop(var_24, var_36)
    var_38 = '__iter__'
    var_39 = hasattr(var_37, var_38)
    var_40 = '__next__'
    var_41 = hasattr(var_37, var_40)
    var_42 = 42
    var_43 = [var_42]
    var_44 = module_0.drop(var_24, var_43)
    var_45 = list(var_44)
    var_46 = []
    var_47 = module_0.drop(var_23, var_46)
    var_48 = list(var_47)



# Parsed testcases at query #8
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = 1
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_1, var_0, var_2, var_3, var_4]
    var_6 = module_0.drop(var_0, var_5)
    var_7 = list(var_6)
    var_8 = 0
    var_9 = [var_1, var_0, var_2]
    var_10 = module_0.drop(var_8, var_9)
    var_11 = list(var_10)
    var_12 = 10
    var_13 = [var_1, var_0, var_2]
    var_14 = module_0.drop(var_12, var_13)
    var_15 = list(var_14)
    var_16 = []
    var_17 = module_0.drop(var_2, var_16)
    var_18 = list(var_17)
    var_19 = range(var_4)
    var_20 = 'hello'
    var_21 = module_0.drop(var_0, var_20)
    var_22 = list(var_21)
    var_23 = range(var_12)
    var_24 = module_0.drop(var_4, var_23)
    var_25 = list(var_24)
    var_26 = -1
    var_27 = 1
    var_28 = 2
    var_29 = 3
    var_30 = [var_27, var_28, var_29]
    var_31 = module_0.drop(var_26, var_30)
    var_32 = list(var_31)
    var_33 = 6
    var_34 = 1000
    var_35 = 1005
    var_36 = range(var_35)
    var_37 = module_0.drop(var_34, var_36)
    var_38 = list(var_37)
    var_39 = [var_27, var_26, var_28, var_29, var_30]
    var_40 = module_0.drop(var_30, var_39)
    var_41 = list(var_40)
    var_42 = [var_27, var_26, var_28, var_29, var_30]
    var_43 = iter(var_42)
    var_44 = next(var_43)
    var_45 = module_0.drop(var_27, var_43)
    var_46 = list(var_45)



# Parsed testcases at query #9
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
    var_7 = 'hello world'
    var_8 = ' '
    var_9 = module_0.split_by(var_7, separator=var_8)
    var_10 = list(var_9)
    var_11 = ' Split by: '
    var_12 = True
    var_13 = module_0.split_by(var_11, var_12, separator=var_8)
    var_14 = list(var_13)
    var_15 = 2
    var_16 = 4
    var_17 = [var_12, var_15, var_2, var_16]
    var_18 = lambda x: x % var_15 == var_3
    var_19 = module_0.split_by(var_17, var_12, criterion=var_18)
    var_20 = list(var_19)
    var_21 = []
    var_22 = lambda x: x
    var_23 = module_0.split_by(var_21, criterion=var_22)
    var_24 = list(var_23)
    var_25 = []
    var_26 = lambda x: x
    var_27 = module_0.split_by(var_25, var_12, criterion=var_26)
    var_28 = list(var_27)
    var_29 = [var_12, var_15, var_2]
    var_30 = lambda x: x > var_0
    var_31 = module_0.split_by(var_29, criterion=var_30)
    var_32 = list(var_31)
    var_33 = [var_3, var_3, var_3]
    var_34 = lambda x: x == var_3
    var_35 = module_0.split_by(var_33, criterion=var_34)
    var_36 = list(var_35)
    var_37 = [var_3, var_3, var_3]
    var_38 = lambda x: x == var_3
    var_39 = module_0.split_by(var_37, var_12, criterion=var_38)
    var_40 = list(var_39)
    var_41 = 5
    var_42 = [var_41]
    var_43 = lambda x: x == var_41
    var_44 = module_0.split_by(var_42, criterion=var_43)
    var_45 = list(var_44)
    var_46 = [var_41]
    var_47 = lambda x: x == var_3
    var_48 = module_0.split_by(var_46, criterion=var_47)
    var_49 = list(var_48)
    var_50 = 'a.b.c'
    var_51 = '.'
    var_52 = module_0.split_by(var_50, separator=var_51)
    var_53 = list(var_52)
    var_54 = 1
    var_55 = 2
    var_56 = 3
    var_57 = [var_54, var_55, var_56]
    var_58 = module_0.split_by(var_57)
    var_59 = list(var_58)
    var_60 = 1
    var_61 = 2
    var_62 = 3
    var_63 = [var_60, var_61, var_62]
    var_64 = lambda x: x
    var_65 = module_0.split_by(var_63, criterion=var_64, separator=var_60)
    var_66 = list(var_65)



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = 1
    var_3 = 2
    var_4 = 7
    var_5 = 10
    var_6 = 6
    var_7 = 9
    var_8 = 3
    var_9 = -1
    var_10 = -2
    var_11 = -1
    var_12 = -2
    var_13 = 20
    var_14 = 15
    var_15 = 4
    var_16 = -5
    var_17 = -10
    var_18 = -10
    var_19 = -5
    var_20 = -10
    var_21 = -5



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 5
    var_2 = 15
    var_3 = 1
    var_4 = 2
    var_5 = 20
    var_6 = 3
    var_7 = 0
    var_8 = -1
    var_9 = -3
    var_10 = 5
    var_11 = -6
    var_12 = 100



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
    var_6 = lambda x: x > var_0
    var_7 = []
    var_8 = module_0.drop_until(var_6, var_7)
    var_9 = list(var_8)
    var_10 = lambda x: x > var_2
    var_11 = range(var_0)
    var_12 = module_0.drop_until(var_10, var_11)
    var_13 = list(var_12)
    var_14 = 0
    var_15 = lambda x: x >= var_14
    var_16 = range(var_0)
    var_17 = module_0.drop_until(var_15, var_16)
    var_18 = list(var_17)
    var_19 = 'b'
    var_20 = lambda c: c == var_19
    var_21 = 'abcdef'
    var_22 = module_0.drop_until(var_20, var_21)
    var_23 = list(var_22)
    var_24 = 2
    var_25 = lambda s: len(s) > var_24
    var_26 = 'a'
    var_27 = 'ab'
    var_28 = 'abc'
    var_29 = 'abcd'
    var_30 = [var_26, var_27, var_28, var_29]
    var_31 = module_0.drop_until(var_25, var_30)
    var_32 = list(var_31)
    var_33 = 3
    var_34 = lambda x: x % var_33 == var_14
    var_35 = range(var_2)
    var_36 = lambda x: x == var_33
    var_37 = 1
    var_38 = 4
    var_39 = [var_37, var_24, var_33, var_38, var_0]
    var_40 = module_0.drop_until(var_36, var_39)
    var_41 = list(var_40)
    var_42 = lambda x: x == var_37
    var_43 = [var_37]
    var_44 = module_0.drop_until(var_42, var_43)
    var_45 = list(var_44)
    var_46 = lambda x: x == var_24
    var_47 = [var_37]
    var_48 = module_0.drop_until(var_46, var_47)
    var_49 = list(var_48)
    var_50 = None
    var_51 = lambda x: x is not var_50
    var_52 = [var_50, var_50, var_37, var_24, var_33]
    var_53 = module_0.drop_until(var_51, var_52)
    var_54 = list(var_53)



# Parsed testcases at query #13
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
    var_10 = lambda x: x * var_2
    var_11 = lambda x: x + var_0
    var_12 = []
    var_13 = module_0.MapList(var_11, var_12)
    var_14 = 'a'
    var_15 = 'b'
    var_16 = 'c'
    var_17 = [var_14, var_15, var_16]
    var_18 = '_test'
    var_19 = lambda x: x * var_1 + var_18
    var_20 = module_0.MapList(var_19, var_17)
    var_21 = [var_0, var_1, var_2]
    var_22 = 10
    var_23 = lambda x: x * var_22
    var_24 = module_0.MapList(var_23, var_21)
    var_25 = 0
    var_26 = var_24[var_25]



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 5
    var_2 = 15
    var_3 = 1
    var_4 = 2
    var_5 = 20
    var_6 = 3
    var_7 = 0
    var_8 = -1
    var_9 = 5
    var_10 = -6



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = 2
    var_3 = 7
    var_4 = 1
    var_5 = 10
    var_6 = 3
    var_7 = -1
    var_8 = -2
    var_9 = -5
    var_10 = -2
    var_11 = 11
    var_12 = -1
    var_13 = 4



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 5
    var_2 = 15
    var_3 = 1
    var_4 = 2
    var_5 = 0
    var_6 = -1
    var_7 = 20
    var_8 = 3
    var_9 = -5



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 5
    var_2 = 15
    var_3 = 1
    var_4 = 2
    var_5 = 20
    var_6 = 3
    var_7 = 0
    var_8 = -2
    var_9 = -2
    var_10 = 6
    var_11 = 4



# Parsed testcases at query #18
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 5
    var_4 = range(var_3)
    var_5 = module_0.LazyList(var_4)
    var_6 = range(var_0)
    var_7 = module_0.LazyList(var_6)
    var_8 = 100
    var_9 = range(var_8)
    var_10 = module_0.LazyList(var_9)
    var_11 = var_10.list
    var_12 = len(var_11)
    assert var_12 == 0
    var_13 = var_10[var_3]
    var_14 = var_10.list
    var_15 = len(var_14)
    assert var_15 == 6
    var_16 = var_10[var_0]
    var_17 = var_10.list
    var_18 = len(var_17)
    assert var_18 == 11
    var_19 = 3
    var_20 = range(var_19)
    var_21 = module_0.LazyList(var_20)
    var_22 = 2
    var_23 = var_21[var_22]
    var_24 = range(var_3)
    var_25 = module_0.LazyList(var_24)
    var_26 = []
    var_27 = module_0.LazyList(var_26)
    var_28 = 0
    var_29 = var_27[var_28]
    var_30 = range(var_28)
    var_31 = module_0.LazyList(var_30)
    var_32 = 20
    var_33 = range(var_32)
    var_34 = module_0.LazyList(var_33)



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 5
    var_2 = 15
    var_3 = 1
    var_4 = 11
    var_5 = 2
    var_6 = 25
    var_7 = 3
    var_8 = 0
    var_9 = -1
    var_10 = -5
    var_11 = 10
    var_12 = -11



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
    var_10 = lambda x: x > var_2
    var_11 = range(var_0)
    var_12 = module_0.drop_until(var_10, var_11)
    var_13 = list(var_12)
    var_14 = 0
    var_15 = lambda x: x >= var_14
    var_16 = range(var_0)
    var_17 = module_0.drop_until(var_15, var_16)
    var_18 = list(var_17)
    var_19 = 'l'
    var_20 = lambda c: c == var_19
    var_21 = 'hello'
    var_22 = module_0.drop_until(var_20, var_21)
    var_23 = list(var_22)
    var_24 = 2
    var_25 = lambda s: len(s) > var_24
    var_26 = 'a'
    var_27 = 'ab'
    var_28 = 'abc'
    var_29 = 'abcd'
    var_30 = [var_26, var_27, var_28, var_29]
    var_31 = module_0.drop_until(var_25, var_30)
    var_32 = list(var_31)
    var_33 = range(var_2)
    var_34 = 4
    var_35 = lambda x: x > var_34
    var_36 = None
    var_37 = lambda x: x is not var_36
    var_38 = 1
    var_39 = 3
    var_40 = [var_36, var_36, var_38, var_24, var_39]
    var_41 = module_0.drop_until(var_37, var_40)
    var_42 = list(var_41)
    var_43 = 'value'
    var_44 = lambda x: x[var_43] > var_39
    var_45 = {var_43: var_38}
    var_46 = {var_43: var_24}
    var_47 = {var_43: var_34}
    var_48 = {var_43: var_0}
    var_49 = [var_45, var_46, var_47, var_48]
    var_50 = module_0.drop_until(var_44, var_49)
    var_51 = list(var_50)



# Parsed testcases at query #21
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 5
    var_4 = range(var_3)
    var_5 = module_0.LazyList(var_4)
    var_6 = None
    var_7 = range(var_0)
    var_8 = module_0.LazyList(var_7)
    var_9 = range(var_0)
    var_10 = module_0.LazyList(var_9)
    var_11 = 100
    var_12 = range(var_11)
    var_13 = module_0.LazyList(var_12)
    var_14 = var_13[var_3]
    var_15 = var_13.list
    var_16 = len(var_15)
    assert var_16 == 6
    var_17 = range(var_0)
    var_18 = module_0.LazyList(var_17)
    var_19 = 3
    var_20 = 7
    var_21 = var_18[var_19:var_20]
    var_22 = var_18.list
    var_23 = len(var_22)
    assert var_23 == 7
    var_24 = range(var_19)
    var_25 = module_0.LazyList(var_24)
    var_26 = 5
    var_27 = var_25[var_26]
    var_28 = range(var_26)
    var_29 = module_0.LazyList(var_28)
    var_30 = 2
    var_31 = -2
    var_32 = var_29[var_30:var_31]
    var_33 = range(var_3)
    var_34 = module_0.LazyList(var_33)
    var_35 = 'a'
    var_36 = 'b'
    var_37 = 'c'
    var_38 = 'd'
    var_39 = 'e'
    var_40 = [var_35, var_36, var_37, var_38, var_39]
    var_41 = module_0.LazyList(var_40)



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 2
    var_2 = 8
    var_3 = 1
    var_4 = 10
    var_5 = 0
    var_6 = -2
    var_7 = 15
    var_8 = 20
    var_9 = 3
    var_10 = -1
    var_11 = 6
    var_12 = 5
    var_13 = -6



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 5
    var_2 = 15
    var_3 = 1
    var_4 = 2
    var_5 = 0
    var_6 = -1
    var_7 = -1
    var_8 = 20
    var_9 = 3



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 5
    var_2 = 15
    var_3 = 1
    var_4 = 20
    var_5 = 3
    var_6 = 25
    var_7 = 4
    var_8 = 2
    var_9 = 0
    var_10 = -1
    var_11 = -3
    var_12 = 5
    var_13 = -6



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
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
    var_8 = [var_0, var_1, var_2, var_3, var_4]
    var_9 = lambda x: x ** var_1
    var_10 = module_0.MapList(var_9, var_8)
    var_11 = lambda x: x * var_1
    var_12 = []
    var_13 = module_0.MapList(var_11, var_12)
    var_14 = 'a'
    var_15 = 'bb'
    var_16 = 'ccc'
    var_17 = [var_14, var_15, var_16]
    var_18 = lambda s: len(s)
    var_19 = module_0.MapList(var_18, var_17)
    var_20 = lambda x: x * var_2
    var_21 = lambda x: x + var_0
    var_22 = [var_0, var_1, var_2]
    var_23 = module_0.MapList(var_21, var_22)
    var_24 = lambda x: x * var_1
    var_25 = module_0.MapList(var_24, var_23)
    var_26 = 10
    var_27 = 20
    var_28 = 30
    var_29 = 40
    var_30 = 50
    var_31 = [var_26, var_27, var_28, var_29, var_30]
    var_32 = lambda x: x // var_26
    var_33 = module_0.MapList(var_32, var_31)
    var_34 = range(var_26)
    var_35 = list(var_34)
    var_36 = lambda x: x * var_1
    var_37 = module_0.MapList(var_36, var_35)



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 11
    var_3 = 2
    var_4 = 5
    var_5 = 0
    var_6 = -1
    var_7 = -1
    var_8 = -2



# Parsed testcases at query #3
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
    var_7 = 'a.b.c'
    var_8 = '.'
    var_9 = module_0.split_by(var_7, separator=var_8)
    var_10 = list(var_9)
    var_11 = '..a..b..'
    var_12 = True
    var_13 = module_0.split_by(var_11, var_12, separator=var_8)
    var_14 = list(var_13)
    var_15 = module_0.split_by(var_11, separator=var_8)
    var_16 = list(var_15)
    var_17 = []
    var_18 = lambda x: x
    var_19 = module_0.split_by(var_17, criterion=var_18)
    var_20 = list(var_19)
    var_21 = []
    var_22 = lambda x: x
    var_23 = module_0.split_by(var_21, var_12, criterion=var_22)
    var_24 = list(var_23)
    var_25 = 2
    var_26 = [var_12, var_25, var_2]
    var_27 = module_0.split_by(var_26, separator=var_3)
    var_28 = list(var_27)
    var_29 = [var_3, var_3, var_3]
    var_30 = module_0.split_by(var_29, separator=var_3)
    var_31 = list(var_30)
    var_32 = [var_3, var_3, var_3]
    var_33 = module_0.split_by(var_32, var_12, separator=var_3)
    var_34 = list(var_33)
    var_35 = 'sep'
    var_36 = [var_12, var_35, var_25, var_35, var_2]
    var_37 = module_0.split_by(var_36, separator=var_35)
    var_38 = list(var_37)
    var_39 = 1
    var_40 = 2
    var_41 = 3
    var_42 = [var_39, var_40, var_41]
    var_43 = lambda x: x
    var_44 = module_0.split_by(var_42, criterion=var_43, separator=var_39)
    var_45 = list(var_44)
    var_46 = 1
    var_47 = 2
    var_48 = 3
    var_49 = [var_46, var_47, var_48]
    var_50 = module_0.split_by(var_49)
    var_51 = list(var_50)
    var_52 = [var_12, var_25, var_48]
    var_53 = lambda x: var_12
    var_54 = module_0.split_by(var_52, criterion=var_53)
    var_55 = list(var_54)
    var_56 = [var_12, var_25, var_48]
    var_57 = False
    var_58 = lambda x: var_57
    var_59 = module_0.split_by(var_56, criterion=var_58)
    var_60 = list(var_59)
    var_61 = 'hello world'
    var_62 = ' '
    var_63 = lambda x: x == var_62
    var_64 = module_0.split_by(var_61, criterion=var_63)
    var_65 = list(var_64)
    var_66 = [var_12, var_57, var_57, var_25, var_57, var_48]
    var_67 = module_0.split_by(var_66, separator=var_57)
    var_68 = list(var_67)



# Parsed testcases at query #4
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
    var_10 = lambda x: x * var_1
    var_11 = []
    var_12 = module_0.MapList(var_10, var_11)
    var_13 = 0
    var_14 = var_12[var_13]
    var_15 = 10
    var_16 = lambda x: x + var_15
    var_17 = '!'
    var_18 = lambda x: str(x) + var_17
    var_19 = 'a'
    var_20 = 'b'
    var_21 = 'c'
    var_22 = [var_19, var_20, var_21]
    var_23 = module_0.MapList(var_18, var_22)
    var_24 = lambda x: (x, x ** var_1)
    var_25 = [var_13, var_1, var_2]
    var_26 = module_0.MapList(var_24, var_25)



# Parsed testcases at query #5
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
    var_7 = 'a.b.c'
    var_8 = '.'
    var_9 = module_0.split_by(var_7, separator=var_8)
    var_10 = list(var_9)
    var_11 = ' Split by: '
    var_12 = True
    var_13 = module_0.split_by(var_11, var_12, separator=var_8)
    var_14 = list(var_13)
    var_15 = []
    var_16 = lambda x: x
    var_17 = module_0.split_by(var_15, criterion=var_16)
    var_18 = list(var_17)
    var_19 = []
    var_20 = lambda x: x
    var_21 = module_0.split_by(var_19, var_12, criterion=var_20)
    var_22 = list(var_21)
    var_23 = 2
    var_24 = [var_12, var_23, var_2]
    var_25 = lambda x: var_12
    var_26 = module_0.split_by(var_24, criterion=var_25)
    var_27 = list(var_26)
    var_28 = [var_12, var_23, var_2]
    var_29 = lambda x: var_12
    var_30 = module_0.split_by(var_28, var_12, criterion=var_29)
    var_31 = list(var_30)
    var_32 = [var_12, var_23, var_2]
    var_33 = False
    var_34 = lambda x: var_33
    var_35 = module_0.split_by(var_32, criterion=var_34)
    var_36 = list(var_35)
    var_37 = 'a..b'
    var_38 = module_0.split_by(var_37, separator=var_8)
    var_39 = list(var_38)
    var_40 = module_0.split_by(var_37, var_12, separator=var_8)
    var_41 = list(var_40)
    var_42 = '.a.b.'
    var_43 = module_0.split_by(var_42, separator=var_8)
    var_44 = list(var_43)
    var_45 = module_0.split_by(var_42, var_12, separator=var_8)
    var_46 = list(var_45)
    var_47 = 1
    var_48 = 2
    var_49 = 3
    var_50 = [var_47, var_48, var_49]
    var_51 = module_0.split_by(var_50)
    var_52 = list(var_51)
    var_53 = 1
    var_54 = 2
    var_55 = 3
    var_56 = [var_53, var_54, var_55]
    var_57 = lambda x: x
    var_58 = module_0.split_by(var_56, criterion=var_57, separator=var_53)
    var_59 = list(var_58)
    var_60 = 4
    var_61 = 5
    var_62 = [var_12, var_23, var_55, var_60, var_61]
    var_63 = lambda x: x % var_23 == var_33
    var_64 = module_0.split_by(var_62, criterion=var_63)
    var_65 = list(var_64)
    var_66 = 'a'
    var_67 = 'sep'
    var_68 = 'b'
    var_69 = 'c'
    var_70 = [var_66, var_67, var_68, var_67, var_69]
    var_71 = module_0.split_by(var_70, separator=var_67)
    var_72 = list(var_71)



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
    var_6 = lambda x: x > var_0
    var_7 = []
    var_8 = module_0.drop_until(var_6, var_7)
    var_9 = list(var_8)
    var_10 = lambda x: x > var_2
    var_11 = range(var_0)
    var_12 = module_0.drop_until(var_10, var_11)
    var_13 = list(var_12)
    var_14 = 0
    var_15 = lambda x: x >= var_14
    var_16 = range(var_0)
    var_17 = module_0.drop_until(var_15, var_16)
    var_18 = list(var_17)
    var_19 = 'b'
    var_20 = lambda c: c == var_19
    var_21 = 'abcdef'
    var_22 = module_0.drop_until(var_20, var_21)
    var_23 = list(var_22)
    var_24 = 2
    var_25 = lambda x: x % var_24 == var_14
    var_26 = 1
    var_27 = 3
    var_28 = 6
    var_29 = 7
    var_30 = 8
    var_31 = [var_26, var_27, var_0, var_28, var_29, var_30]
    var_32 = module_0.drop_until(var_25, var_31)
    var_33 = list(var_32)
    var_34 = range(var_2)
    var_35 = lambda x: x > var_29
    var_36 = lambda x: x == var_0
    var_37 = [var_0]
    var_38 = module_0.drop_until(var_36, var_37)
    var_39 = list(var_38)
    var_40 = lambda x: x == var_0
    var_41 = [var_27]
    var_42 = module_0.drop_until(var_40, var_41)
    var_43 = list(var_42)
    var_44 = 9
    var_45 = lambda x: x == var_44
    var_46 = range(var_2)
    var_47 = module_0.drop_until(var_45, var_46)
    var_48 = list(var_47)
    var_49 = True
    var_50 = lambda x: var_49
    var_51 = [var_49, var_24, var_27]
    var_52 = module_0.drop_until(var_50, var_51)
    var_53 = list(var_52)
    var_54 = False
    var_55 = lambda x: var_54
    var_56 = [var_49, var_24, var_27]
    var_57 = module_0.drop_until(var_55, var_56)
    var_58 = list(var_57)



# Parsed testcases at query #7
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
    var_7 = 'a.b.c'
    var_8 = '.'
    var_9 = module_0.split_by(var_7, separator=var_8)
    var_10 = list(var_9)
    var_11 = ' Split by: '
    var_12 = True
    var_13 = ' '
    var_14 = module_0.split_by(var_11, var_12, separator=var_13)
    var_15 = list(var_14)
    var_16 = []
    var_17 = lambda x: x
    var_18 = module_0.split_by(var_16, criterion=var_17)
    var_19 = list(var_18)
    var_20 = []
    var_21 = lambda x: x
    var_22 = module_0.split_by(var_20, var_12, criterion=var_21)
    var_23 = list(var_22)
    var_24 = 2
    var_25 = [var_12, var_24, var_2]
    var_26 = lambda x: var_12
    var_27 = module_0.split_by(var_25, criterion=var_26)
    var_28 = list(var_27)
    var_29 = [var_12, var_24, var_2]
    var_30 = lambda x: var_12
    var_31 = module_0.split_by(var_29, var_12, criterion=var_30)
    var_32 = list(var_31)
    var_33 = [var_12, var_24, var_2]
    var_34 = False
    var_35 = lambda x: var_34
    var_36 = module_0.split_by(var_33, criterion=var_35)
    var_37 = list(var_36)
    var_38 = 'sep'
    var_39 = [var_12, var_38, var_24, var_38, var_2]
    var_40 = module_0.split_by(var_39, separator=var_38)
    var_41 = list(var_40)
    var_42 = 1
    var_43 = 2
    var_44 = 3
    var_45 = [var_42, var_43, var_44]
    var_46 = lambda x: x > var_42
    var_47 = module_0.split_by(var_45, criterion=var_46, separator=var_43)
    var_48 = list(var_47)
    var_49 = 1
    var_50 = 2
    var_51 = 3
    var_52 = [var_49, var_50, var_51]
    var_53 = module_0.split_by(var_52)
    var_54 = list(var_53)
    var_55 = '...'
    var_56 = module_0.split_by(var_55, var_12, separator=var_8)
    var_57 = list(var_56)
    var_58 = 'a..b'
    var_59 = module_0.split_by(var_58, separator=var_8)
    var_60 = list(var_59)
    var_61 = 'a.b.'
    var_62 = module_0.split_by(var_61, separator=var_8)
    var_63 = list(var_62)
    var_64 = '.a.b'
    var_65 = module_0.split_by(var_64, separator=var_8)
    var_66 = list(var_65)



# Parsed testcases at query #8
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = 10
    var_2 = range(var_1)
    var_3 = module_0.chunk(var_0, var_2)
    var_4 = list(var_3)
    var_5 = 2
    var_6 = 6
    var_7 = range(var_6)
    var_8 = module_0.chunk(var_5, var_7)
    var_9 = list(var_8)
    var_10 = 5
    var_11 = range(var_0)
    var_12 = module_0.chunk(var_10, var_11)
    var_13 = list(var_12)
    var_14 = 1
    var_15 = range(var_0)
    var_16 = module_0.chunk(var_14, var_15)
    var_17 = list(var_16)
    var_18 = []
    var_19 = module_0.chunk(var_0, var_18)
    var_20 = list(var_19)
    var_21 = 'abcdef'
    var_22 = module_0.chunk(var_5, var_21)
    var_23 = list(var_22)
    var_24 = range(var_10)
    var_25 = 0
    var_26 = 5
    var_27 = range(var_26)
    var_28 = module_0.chunk(var_25, var_27)
    var_29 = list(var_28)
    var_30 = -1
    var_31 = 5
    var_32 = range(var_31)
    var_33 = module_0.chunk(var_30, var_32)
    var_34 = list(var_33)



# Parsed testcases at query #9
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 5
    var_4 = range(var_3)
    var_5 = module_0.LazyList(var_4)
    var_6 = None
    var_7 = range(var_0)
    var_8 = module_0.LazyList(var_7)
    var_9 = 100
    var_10 = range(var_9)
    var_11 = module_0.LazyList(var_10)
    var_12 = var_11[var_0]
    var_13 = var_11.list
    var_14 = len(var_13)
    assert var_14 == 11
    var_15 = range(var_3)
    var_16 = module_0.LazyList(var_15)
    var_17 = range(var_3)
    var_18 = module_0.LazyList(var_17)
    var_19 = 10
    var_20 = var_18[var_19]
    var_21 = range(var_3)
    var_22 = module_0.LazyList(var_21)
    var_23 = var_22[:]
    var_24 = []
    var_25 = module_0.LazyList(var_24)
    var_26 = 0
    var_27 = var_25[var_26]
    var_28 = range(var_26)
    var_29 = module_0.LazyList(var_28)
    var_30 = 1
    var_31 = 2
    var_32 = 3
    var_33 = 4
    var_34 = [var_30, var_31, var_32, var_33, var_3]
    var_35 = module_0.LazyList(var_34)
    var_36 = var_35[var_31]
    assert var_36 == 3
    var_37 = 20
    var_38 = range(var_37)
    var_39 = module_0.LazyList(var_38)



# Parsed testcases at query #10
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
    var_7 = 'a.b.c'
    var_8 = '.'
    var_9 = module_0.split_by(var_7, separator=var_8)
    var_10 = list(var_9)
    var_11 = '..a..b..'
    var_12 = True
    var_13 = module_0.split_by(var_11, var_12, separator=var_8)
    var_14 = list(var_13)
    var_15 = module_0.split_by(var_11, separator=var_8)
    var_16 = list(var_15)
    var_17 = []
    var_18 = lambda x: x
    var_19 = module_0.split_by(var_17, criterion=var_18)
    var_20 = list(var_19)
    var_21 = []
    var_22 = lambda x: x
    var_23 = module_0.split_by(var_21, var_12, criterion=var_22)
    var_24 = list(var_23)
    var_25 = 2
    var_26 = [var_12, var_25, var_2]
    var_27 = module_0.split_by(var_26, separator=var_3)
    var_28 = list(var_27)
    var_29 = [var_3, var_3, var_3]
    var_30 = module_0.split_by(var_29, var_12, separator=var_3)
    var_31 = list(var_30)
    var_32 = [var_3, var_3, var_3]
    var_33 = module_0.split_by(var_32, separator=var_3)
    var_34 = list(var_33)
    var_35 = 'hello world'
    var_36 = ' '
    var_37 = lambda x: x == var_36
    var_38 = module_0.split_by(var_35, criterion=var_37)
    var_39 = list(var_38)
    var_40 = 1
    var_41 = 2
    var_42 = 3
    var_43 = [var_40, var_41, var_42]
    var_44 = lambda x: x > var_40
    var_45 = module_0.split_by(var_43, criterion=var_44, separator=var_41)
    var_46 = list(var_45)
    var_47 = 1
    var_48 = 2
    var_49 = 3
    var_50 = [var_47, var_48, var_49]
    var_51 = module_0.split_by(var_50)
    var_52 = list(var_51)



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 5
    var_2 = 15
    var_3 = 1
    var_4 = 2
    var_5 = 20
    var_6 = 3
    var_7 = 0
    var_8 = 100
    var_9 = -1



# Parsed testcases at query #12
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 5
    var_4 = range(var_3)
    var_5 = module_0.LazyList(var_4)
    var_6 = range(var_0)
    var_7 = module_0.LazyList(var_6)
    var_8 = 100
    var_9 = range(var_8)
    var_10 = module_0.LazyList(var_9)
    var_11 = 50
    var_12 = var_10[var_11]
    var_13 = var_10.list
    var_14 = len(var_13)
    assert var_14 == 51
    var_15 = 3
    var_16 = range(var_15)
    var_17 = module_0.LazyList(var_16)
    var_18 = 0
    var_19 = var_17[var_18]
    var_20 = 1
    var_21 = var_17[var_20]
    var_22 = 2
    var_23 = var_17[var_22]
    var_24 = range(var_3)
    var_25 = module_0.LazyList(var_24)
    var_26 = var_25[var_22:var_0]
    var_27 = []
    var_28 = module_0.LazyList(var_27)
    var_29 = 0
    var_30 = var_28[var_29]
    var_31 = range(var_3)
    var_32 = module_0.LazyList(var_31)



# Parsed testcases at query #13
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
    var_10 = lambda x: x > var_2
    var_11 = range(var_0)
    var_12 = module_0.drop_until(var_10, var_11)
    var_13 = list(var_12)
    var_14 = 0
    var_15 = lambda x: x >= var_14
    var_16 = range(var_0)
    var_17 = module_0.drop_until(var_15, var_16)
    var_18 = list(var_17)
    var_19 = 'b'
    var_20 = lambda x: x == var_19
    var_21 = 'a'
    var_22 = 'c'
    var_23 = 'd'
    var_24 = [var_21, var_19, var_22, var_23]
    var_25 = module_0.drop_until(var_20, var_24)
    var_26 = list(var_25)
    var_27 = lambda x: x
    var_28 = False
    var_29 = False
    var_30 = True
    var_31 = False
    var_32 = [var_28, var_29, var_30, var_31, var_30]
    var_33 = module_0.drop_until(var_27, var_32)
    var_34 = list(var_33)
    var_35 = 2
    var_36 = 3
    var_37 = lambda x: x.val > var_30
    var_38 = range(var_2)
    var_39 = 7
    var_40 = lambda x: x > var_39
    var_41 = [var_30, var_21, var_35, var_19, var_36]
    var_42 = 4
    var_43 = lambda x: x == var_42
    var_44 = range(var_0)
    var_45 = module_0.drop_until(var_43, var_44)
    var_46 = list(var_45)
    var_47 = lambda x: x >= var_31
    var_48 = -5
    var_49 = -3
    var_50 = -1
    var_51 = [var_48, var_49, var_50, var_31, var_35, var_42]
    var_52 = module_0.drop_until(var_47, var_51)
    var_53 = list(var_52)



# Parsed testcases at query #14
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
    var_7 = lambda x: x >= var_6
    var_8 = range(var_0)
    var_9 = module_0.drop_until(var_7, var_8)
    var_10 = list(var_9)
    var_11 = lambda x: x > var_2
    var_12 = range(var_0)
    var_13 = module_0.drop_until(var_11, var_12)
    var_14 = list(var_13)
    var_15 = lambda x: x > var_0
    var_16 = []
    var_17 = module_0.drop_until(var_15, var_16)
    var_18 = list(var_17)
    var_19 = 'b'
    var_20 = lambda x: x == var_19
    var_21 = 'abcdef'
    var_22 = module_0.drop_until(var_20, var_21)
    var_23 = list(var_22)
    var_24 = 2
    var_25 = lambda x: len(x) > var_24
    var_26 = 'a'
    var_27 = 'ab'
    var_28 = 'abc'
    var_29 = 'abcd'
    var_30 = [var_26, var_27, var_28, var_29]
    var_31 = module_0.drop_until(var_25, var_30)
    var_32 = list(var_31)
    var_33 = 3
    var_34 = lambda x: x % var_33 == var_6
    var_35 = 1
    var_36 = 4
    var_37 = 6
    var_38 = [var_35, var_24, var_33, var_36, var_0, var_37]
    var_39 = iter(var_38)
    var_40 = module_0.drop_until(var_34, var_39)
    var_41 = list(var_40)
    var_42 = 'value'
    var_43 = lambda x: x[var_42] > var_33
    var_44 = {var_42: var_35}
    var_45 = {var_42: var_24}
    var_46 = {var_42: var_33}
    var_47 = {var_42: var_36}
    var_48 = [var_44, var_45, var_46, var_47]
    var_49 = module_0.drop_until(var_43, var_48)
    var_50 = list(var_49)
    var_51 = lambda x: x > var_0
    var_52 = None
    var_53 = lambda x: x is not var_52
    var_54 = [var_52, var_52, var_35, var_24, var_33]
    var_55 = module_0.drop_until(var_53, var_54)
    var_56 = list(var_55)



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 5
    var_2 = 15
    var_3 = 1
    var_4 = 2
    var_5 = range(var_0)
    var_6 = list(var_5)
    var_7 = 0
    var_8 = 3
    var_9 = 6
    var_10 = -1
    var_11 = -2



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 5
    var_2 = 15
    var_3 = 1
    var_4 = 2
    var_5 = 11
    var_6 = 0
    var_7 = 0



# Parsed testcases at query #17
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = True
    var_7 = lambda x: var_6
    var_8 = []
    var_9 = module_0.drop_until(var_7, var_8)
    var_10 = list(var_9)
    var_11 = lambda x: x > var_2
    var_12 = range(var_0)
    var_13 = module_0.drop_until(var_11, var_12)
    var_14 = list(var_13)
    var_15 = 0
    var_16 = lambda x: x >= var_15
    var_17 = range(var_0)
    var_18 = module_0.drop_until(var_16, var_17)
    var_19 = list(var_18)
    var_20 = 'l'
    var_21 = lambda c: c == var_20
    var_22 = 'hello world'
    var_23 = module_0.drop_until(var_21, var_22)
    var_24 = list(var_23)
    var_25 = 2
    var_26 = lambda x: x % var_25 == var_15
    var_27 = 3
    var_28 = 6
    var_29 = 7
    var_30 = 8
    var_31 = [var_6, var_27, var_0, var_28, var_29, var_30]
    var_32 = module_0.drop_until(var_26, var_31)
    var_33 = list(var_32)
    var_34 = lambda x: x == var_27
    var_35 = 4
    var_36 = [var_6, var_25, var_27, var_35, var_0]
    var_37 = iter(var_36)
    var_38 = module_0.drop_until(var_34, var_37)
    var_39 = list(var_38)
    var_40 = lambda x: var_6
    var_41 = [var_6, var_25, var_27]
    var_42 = module_0.drop_until(var_40, var_41)
    var_43 = list(var_42)
    var_44 = False
    var_45 = lambda x: var_44
    var_46 = [var_6, var_25, var_27]
    var_47 = module_0.drop_until(var_45, var_46)
    var_48 = list(var_47)
    var_49 = lambda x: len(x) > var_27
    var_50 = 'a'
    var_51 = 'ab'
    var_52 = 'abc'
    var_53 = 'abcd'
    var_54 = 'abcde'
    var_55 = [var_50, var_51, var_52, var_53, var_54]
    var_56 = module_0.drop_until(var_49, var_55)
    var_57 = list(var_56)



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
    var_6 = lambda x: x > var_0
    var_7 = []
    var_8 = module_0.drop_until(var_6, var_7)
    var_9 = list(var_8)
    var_10 = lambda x: x > var_2
    var_11 = range(var_0)
    var_12 = module_0.drop_until(var_10, var_11)
    var_13 = list(var_12)
    var_14 = 0
    var_15 = lambda x: x >= var_14
    var_16 = range(var_0)
    var_17 = module_0.drop_until(var_15, var_16)
    var_18 = list(var_17)
    var_19 = 'l'
    var_20 = lambda c: c == var_19
    var_21 = 'hello world'
    var_22 = module_0.drop_until(var_20, var_21)
    var_23 = list(var_22)
    var_24 = 3
    var_25 = lambda s: len(s) > var_24
    var_26 = 'a'
    var_27 = 'ab'
    var_28 = 'abc'
    var_29 = 'abcd'
    var_30 = 'abcde'
    var_31 = [var_26, var_27, var_28, var_29, var_30]
    var_32 = module_0.drop_until(var_25, var_31)
    var_33 = list(var_32)
    var_34 = 2
    var_35 = lambda x: x % var_34 == var_14
    var_36 = range(var_2)
    var_37 = lambda x: x == var_24
    var_38 = 1
    var_39 = 4
    var_40 = [var_38, var_34, var_24, var_39, var_0]
    var_41 = module_0.drop_until(var_37, var_40)
    var_42 = list(var_41)
    var_43 = lambda x: x == var_38
    var_44 = [var_38]
    var_45 = module_0.drop_until(var_43, var_44)
    var_46 = list(var_45)
    var_47 = lambda x: x == var_34
    var_48 = [var_38]
    var_49 = module_0.drop_until(var_47, var_48)
    var_50 = list(var_49)



