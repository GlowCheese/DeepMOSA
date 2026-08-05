####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = 2
    var_3 = 6
    var_4 = 1
    var_5 = 10
    var_6 = -1
    var_7 = 3



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 2
    var_2 = 6
    var_3 = 1
    var_4 = 10
    var_5 = -1
    var_6 = 3



# Parsed testcases at query #3
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = 6
    var_6 = [var_0, var_1, var_2, var_3, var_4, var_5]
    var_7 = 0
    var_8 = lambda x: x % var_1 == var_7
    var_9 = module_0.split_by(var_6, criterion=var_8)
    var_10 = list(var_9)
    var_11 = 'a,b,c,d'
    var_12 = ','
    var_13 = module_0.split_by(var_11, separator=var_12)
    var_14 = list(var_13)
    var_15 = [var_0, var_1, var_1, var_2]
    var_16 = False
    var_17 = module_0.split_by(var_15, var_16, separator=var_1)
    var_18 = list(var_17)
    var_19 = True
    var_20 = module_0.split_by(var_15, var_19, separator=var_1)
    var_21 = list(var_20)
    var_22 = [var_19, var_1, var_2]
    var_23 = True
    var_24 = module_0.split_by(var_22, var_23, separator=var_2)
    var_25 = list(var_24)
    var_26 = [var_23, var_1, var_2]
    var_27 = True
    var_28 = module_0.split_by(var_26, var_27, separator=var_27)
    var_29 = list(var_28)
    var_30 = 1
    var_31 = 2
    var_32 = [var_30, var_31]
    var_33 = True
    var_34 = lambda x: var_33
    var_35 = module_0.split_by(var_32, criterion=var_34, separator=var_33)
    var_36 = list(var_35)
    var_37 = 1
    var_38 = 2
    var_39 = [var_37, var_38]
    var_40 = module_0.split_by(var_39)
    var_41 = list(var_40)
    var_42 = []
    var_43 = module_0.split_by(var_42, separator=var_27)
    var_44 = list(var_43)



# Parsed testcases at query #4
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = 10
    var_2 = range(var_1)
    var_3 = module_0.drop(var_0, var_2)
    var_4 = list(var_3)
    var_5 = 0
    var_6 = 1
    var_7 = 2
    var_8 = [var_6, var_7, var_0]
    var_9 = module_0.drop(var_5, var_8)
    var_10 = list(var_9)
    var_11 = [var_6, var_7, var_0]
    var_12 = module_0.drop(var_0, var_11)
    var_13 = list(var_12)
    var_14 = [var_6, var_7, var_0]
    var_15 = module_0.drop(var_1, var_14)
    var_16 = list(var_15)
    var_17 = 5
    var_18 = []
    var_19 = module_0.drop(var_17, var_18)
    var_20 = list(var_19)
    var_21 = ''
    var_22 = 'hello'
    var_23 = module_0.drop(var_7, var_22)
    var_24 = -1
    var_25 = 1
    var_26 = 2
    var_27 = 3
    var_28 = [var_25, var_26, var_27]
    var_29 = module_0.drop(var_24, var_28)
    var_30 = list(var_29)
    var_31 = 20
    var_32 = 30
    var_33 = [var_25, var_31, var_32]
    var_34 = iter(var_33)
    var_35 = module_0.drop(var_30, var_34)
    var_36 = next(var_35)
    assert var_36 == 20
    var_37 = next(var_35)
    assert var_37 == 30
    var_38 = next(var_35)



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 3
    var_1 = 5
    var_2 = 8
    var_3 = 1
    var_4 = 10



# Parsed testcases at query #6
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
    var_11 = 'a.b.c'
    var_12 = module_0.split_by(var_11, separator=var_8)
    var_13 = list(var_12)
    var_14 = module_0.split_by(var_11, separator=var_8)
    var_15 = list(var_14)
    var_16 = '.a.'
    var_17 = True
    var_18 = module_0.split_by(var_16, var_17, separator=var_8)
    var_19 = list(var_18)
    var_20 = False
    var_21 = module_0.split_by(var_16, var_20, separator=var_8)
    var_22 = list(var_21)
    var_23 = 1
    var_24 = 2
    var_25 = [var_23, var_24]
    var_26 = True
    var_27 = lambda x: var_26
    var_28 = module_0.split_by(var_25, criterion=var_27, separator=var_26)
    var_29 = list(var_28)
    var_30 = 1
    var_31 = 2
    var_32 = [var_30, var_31]
    var_33 = module_0.split_by(var_32)
    var_34 = list(var_33)
    var_35 = []
    var_36 = module_0.split_by(var_35, separator=var_8)
    var_37 = list(var_36)
    var_38 = 2
    var_39 = [var_17, var_38, var_32]
    var_40 = False
    var_41 = lambda x: var_40
    var_42 = module_0.split_by(var_39, criterion=var_41)
    var_43 = list(var_42)
    var_44 = [var_17, var_38, var_32]
    var_45 = lambda x: var_17
    var_46 = module_0.split_by(var_44, criterion=var_45)
    var_47 = list(var_46)
    var_48 = 4
    var_49 = 5
    var_50 = 6
    var_51 = [var_17, var_38, var_32, var_48, var_49, var_50]
    var_52 = module_0.split_by(var_51, separator=var_32)
    var_53 = list(var_52)



# Parsed testcases at query #7
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = 40
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.LazyList(var_4)
    var_6 = range(var_0)
    var_7 = 10
    var_8 = var_5[var_7]
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = [var_9, var_10, var_11]
    var_13 = iter(var_12)
    var_14 = module_0.LazyList(var_13)
    var_15 = 5
    var_16 = 6
    var_17 = 7
    var_18 = [var_15, var_16, var_17]
    var_19 = iter(var_18)
    var_20 = module_0.LazyList(var_19)
    var_21 = var_20.list
    var_22 = len(var_21)
    assert var_22 == 3
    var_23 = 0
    var_24 = 4
    var_25 = [var_23, var_9, var_10, var_11, var_24, var_15]
    var_26 = module_0.LazyList(var_25)
    var_27 = slice(var_23, var_15, var_10)
    var_28 = var_26[var_27]



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 2
    var_3 = 1
    var_4 = 10
    var_5 = 5



# Parsed testcases at query #9
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
    var_12 = 10
    var_13 = var_7[var_12]



# Parsed testcases at query #10
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = range(var_1)
    var_3 = module_0.take(var_0, var_2)
    var_4 = list(var_3)
    var_5 = range(var_0)
    var_6 = module_0.take(var_1, var_5)
    var_7 = list(var_6)
    var_8 = 0
    var_9 = range(var_0)
    var_10 = module_0.take(var_8, var_9)
    var_11 = list(var_10)
    var_12 = []
    var_13 = module_0.take(var_0, var_12)
    var_14 = list(var_13)
    var_15 = 2
    var_16 = 'a'
    var_17 = 'b'
    var_18 = 'tuple'
    var_19 = 'd'
    var_20 = [var_16, var_17, var_18, var_19]
    var_21 = module_0.take(var_15, var_20)
    var_22 = list(var_21)
    var_23 = -1
    var_24 = 5
    var_25 = range(var_24)
    var_26 = module_0.take(var_23, var_25)
    var_27 = list(var_26)
    var_28 = 3



# Parsed testcases at query #11
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x == var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_0, var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = 5
    var_9 = lambda x: x > var_8
    var_10 = 6
    var_11 = 7
    var_12 = 8
    var_13 = [var_2, var_3, var_10, var_11, var_12]
    var_14 = module_0.drop_until(var_9, var_13)
    var_15 = list(var_14)
    var_16 = 9
    var_17 = lambda x: x == var_16
    var_18 = 10
    var_19 = [var_0, var_2, var_3, var_16, var_18]
    var_20 = module_0.drop_until(var_17, var_19)
    var_21 = list(var_20)
    var_22 = lambda x: x > var_18
    var_23 = 4
    var_24 = [var_2, var_3, var_4, var_23]
    var_25 = module_0.drop_until(var_22, var_24)
    var_26 = list(var_25)
    var_27 = True
    var_28 = lambda x: var_27
    var_29 = []
    var_30 = module_0.drop_until(var_28, var_29)
    var_31 = list(var_30)
    var_32 = 'b'
    var_33 = lambda x: x == var_32
    var_34 = 'a'
    var_35 = 'c'
    var_36 = [var_34, var_32, var_35]
    var_37 = module_0.drop_until(var_33, var_36)
    var_38 = list(var_37)
    var_39 = lambda x: x % var_3 == var_0
    var_40 = [var_27, var_4, var_8, var_10, var_11]
    var_41 = module_0.drop_until(var_39, var_40)
    var_42 = list(var_41)



# Parsed testcases at query #12
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x == var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_0, var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = 5
    var_9 = lambda x: x > var_8
    var_10 = 6
    var_11 = 7
    var_12 = 8
    var_13 = [var_2, var_3, var_10, var_11, var_12]
    var_14 = module_0.drop_until(var_9, var_13)
    var_15 = list(var_14)
    var_16 = 9
    var_17 = lambda x: x == var_16
    var_18 = [var_2, var_3, var_4, var_16]
    var_19 = module_0.drop_until(var_17, var_18)
    var_20 = list(var_19)
    var_21 = 10
    var_22 = lambda x: x > var_21
    var_23 = 4
    var_24 = [var_2, var_3, var_4, var_23]
    var_25 = module_0.drop_until(var_22, var_24)
    var_26 = list(var_25)
    var_27 = True
    var_28 = lambda x: var_27
    var_29 = []
    var_30 = module_0.drop_until(var_28, var_29)
    var_31 = list(var_30)
    var_32 = 'b'
    var_33 = lambda x: x == var_32
    var_34 = 'a'
    var_35 = 'c'
    var_36 = [var_34, var_32, var_35]
    var_37 = module_0.drop_until(var_33, var_36)
    var_38 = list(var_37)
    var_39 = lambda x: x % var_3 == var_0
    var_40 = [var_27, var_4, var_8, var_23, var_10]
    var_41 = module_0.drop_until(var_39, var_40)
    var_42 = list(var_41)
    var_43 = lambda x: x == var_3
    var_44 = [var_27, var_3, var_4]
    var_45 = module_0.drop_until(var_43, var_44)
    var_46 = '__iter__'
    var_47 = hasattr(var_45, var_46)
    var_48 = next(var_45)
    assert var_48 == 2



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 11
    var_3 = 2
    var_4 = 0
    var_5 = 5
    var_6 = slice(var_4, var_5)
    var_7 = 8
    var_8 = slice(var_3, var_7, var_3)
    var_9 = 10
    var_10 = -11



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 2
    var_3 = 5
    var_4 = 15
    var_5 = 1



# Parsed testcases at query #15
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = 5
    var_9 = lambda x: x == var_8
    var_10 = 6
    var_11 = 7
    var_12 = [var_2, var_3, var_8, var_10, var_11]
    var_13 = module_0.drop_until(var_9, var_12)
    var_14 = list(var_13)
    var_15 = 9
    var_16 = lambda x: x == var_15
    var_17 = [var_2, var_3, var_4, var_15]
    var_18 = module_0.drop_until(var_16, var_17)
    var_19 = list(var_18)
    var_20 = 10
    var_21 = lambda x: x > var_20
    var_22 = [var_2, var_3, var_4]
    var_23 = module_0.drop_until(var_21, var_22)
    var_24 = list(var_23)
    var_25 = lambda x: x > var_0
    var_26 = []
    var_27 = module_0.drop_until(var_25, var_26)
    var_28 = list(var_27)
    var_29 = ''
    var_30 = 'b'
    var_31 = lambda x: x == var_30
    var_32 = 'abcde'
    var_33 = module_0.drop_until(var_31, var_32)
    var_34 = (var_2, var_0)
    var_35 = (var_3, var_0)
    var_36 = (var_4, var_2)
    var_37 = 4
    var_38 = (var_37, var_0)
    var_39 = [var_34, var_35, var_36, var_38]
    var_40 = lambda x: x[var_2] == var_2
    var_41 = module_0.drop_until(var_40, var_39)
    var_42 = list(var_41)
    var_43 = True
    var_44 = lambda x: var_43
    var_45 = [var_43, var_3, var_4]
    var_46 = module_0.drop_until(var_44, var_45)
    var_47 = list(var_46)
    var_48 = 20
    var_49 = 30
    var_50 = [var_20, var_48, var_49]
    var_51 = iter(var_50)
    var_52 = lambda x: x >= var_48
    var_53 = module_0.drop_until(var_52, var_51)
    var_54 = list(var_53)



# Parsed testcases at query #16
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = 4
    var_9 = 5
    var_10 = [var_5, var_6, var_7, var_8, var_9]
    var_11 = iter(var_10)
    var_12 = module_0.LazyList(var_11)
    var_13 = var_12.list
    var_14 = len(var_13)
    assert var_14 == 0
    var_15 = var_12[var_6]
    assert var_15 == 3
    var_16 = 10
    var_17 = var_12[var_16]
    var_18 = var_12[var_9:var_16]
    var_19 = var_12.list
    var_20 = len(var_19)
    assert var_20 == 5
    var_21 = [var_5, var_6, var_7]
    var_22 = module_0.LazyList(var_21)
    var_23 = 1
    var_24 = 2
    var_25 = 3
    var_26 = [var_23, var_24, var_25]
    var_27 = iter(var_26)
    var_28 = module_0.LazyList(var_27)
    var_29 = len(var_28)
    var_30 = [var_27, var_28]
    var_31 = iter(var_30)
    var_32 = module_0.LazyList(var_31)
    var_33 = list(var_32)



# Parsed testcases at query #17
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
    var_8 = '!'
    var_9 = lambda x: str(x) + var_8
    var_10 = module_0.MapList(var_9, var_5)
    var_11 = 0
    var_12 = slice(var_11, var_1)
    var_13 = var_10[var_12]
    var_14 = 10
    var_15 = var_7[var_14]



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 2
    var_3 = 1
    var_4 = 11
    var_5 = 5
    var_6 = 5



# Parsed testcases at query #19
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = 5
    var_9 = lambda x: x == var_8
    var_10 = 6
    var_11 = 7
    var_12 = [var_2, var_3, var_8, var_10, var_11]
    var_13 = module_0.drop_until(var_9, var_12)
    var_14 = list(var_13)
    var_15 = 9
    var_16 = lambda x: x == var_15
    var_17 = [var_2, var_3, var_4, var_15]
    var_18 = module_0.drop_until(var_16, var_17)
    var_19 = list(var_18)
    var_20 = 10
    var_21 = lambda x: x > var_20
    var_22 = 4
    var_23 = [var_2, var_3, var_4, var_22]
    var_24 = module_0.drop_until(var_21, var_23)
    var_25 = list(var_24)
    var_26 = True
    var_27 = lambda x: var_26
    var_28 = []
    var_29 = module_0.drop_until(var_27, var_28)
    var_30 = list(var_29)
    var_31 = 'b'
    var_32 = lambda char: char == var_31
    var_33 = 'abcde'
    var_34 = module_0.drop_until(var_32, var_33)
    var_35 = list(var_34)
    var_36 = 20
    var_37 = 30
    var_38 = [var_20, var_36, var_37]
    var_39 = iter(var_38)
    var_40 = 15
    var_41 = lambda x: x > var_40
    var_42 = module_0.drop_until(var_41, var_39)
    var_43 = next(var_42)
    assert var_43 == 20
    var_44 = next(var_42)
    assert var_44 == 30
    var_45 = next(var_42)
    var_46 = 'id'
    var_47 = {var_46: var_26}
    var_48 = {var_46: var_3}
    var_49 = {var_46: var_4}
    var_50 = [var_47, var_48, var_49]
    var_51 = lambda x: x[var_46] == var_3
    var_52 = module_0.drop_until(var_51, var_50)
    var_53 = list(var_52)



# Parsed testcases at query #20
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x == var_0
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_2, var_3]
    var_5 = module_0.drop_until(var_1, var_4)
    var_6 = list(var_5)
    var_7 = 5
    var_8 = lambda x: x > var_7
    var_9 = 6
    var_10 = 7
    var_11 = 8
    var_12 = [var_0, var_2, var_9, var_10, var_11]
    var_13 = module_0.drop_until(var_8, var_12)
    var_14 = list(var_13)
    var_15 = lambda x: x == var_3
    var_16 = [var_0, var_2, var_3]
    var_17 = module_0.drop_until(var_15, var_16)
    var_18 = list(var_17)
    var_19 = 10
    var_20 = lambda x: x > var_19
    var_21 = [var_0, var_2, var_3]
    var_22 = module_0.drop_until(var_20, var_21)
    var_23 = list(var_22)
    var_24 = True
    var_25 = lambda x: var_24
    var_26 = []
    var_27 = module_0.drop_until(var_25, var_26)
    var_28 = list(var_27)
    var_29 = 'target'
    var_30 = lambda s: s == var_29
    var_31 = 'a'
    var_32 = 'b'
    var_33 = 'c'
    var_34 = [var_31, var_32, var_29, var_33]
    var_35 = module_0.drop_until(var_30, var_34)
    var_36 = list(var_35)
    var_37 = True
    var_38 = lambda x: var_37
    var_39 = 20
    var_40 = 30
    var_41 = [var_19, var_39, var_40]
    var_42 = module_0.drop_until(var_38, var_41)
    var_43 = list(var_42)



# Parsed testcases at query #21
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = 5
    var_9 = lambda x: x >= var_8
    var_10 = 6
    var_11 = 7
    var_12 = [var_2, var_3, var_8, var_10, var_11]
    var_13 = module_0.drop_until(var_9, var_12)
    var_14 = list(var_13)
    var_15 = lambda x: x == var_4
    var_16 = [var_2, var_3, var_4]
    var_17 = module_0.drop_until(var_15, var_16)
    var_18 = list(var_17)
    var_19 = 10
    var_20 = lambda x: x > var_19
    var_21 = [var_2, var_3, var_4]
    var_22 = module_0.drop_until(var_20, var_21)
    var_23 = list(var_22)
    var_24 = ''
    var_25 = 'c'
    var_26 = lambda x: x == var_25
    var_27 = 'abcde'
    var_28 = module_0.drop_until(var_26, var_27)
    var_29 = True
    var_30 = lambda x: var_29
    var_31 = []
    var_32 = module_0.drop_until(var_30, var_31)
    var_33 = list(var_32)
    var_34 = lambda x: x % var_3 == var_0
    var_35 = 4
    var_36 = [var_29, var_4, var_8, var_35, var_10]
    var_37 = module_0.drop_until(var_34, var_36)
    var_38 = list(var_37)



# Parsed testcases at query #22
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = 40
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = iter(var_4)
    var_6 = module_0.LazyList(var_5)
    var_7 = var_6.list
    var_8 = len(var_7)
    assert var_8 == 3
    var_9 = 'a'
    var_10 = 'b'
    var_11 = 'c'
    var_12 = 'd'
    var_13 = 'e'
    var_14 = [var_9, var_10, var_11, var_12, var_13]
    var_15 = iter(var_14)
    var_16 = module_0.LazyList(var_15)
    var_17 = var_16.list
    var_18 = len(var_17)
    assert var_18 == 3
    var_19 = 1
    var_20 = 2
    var_21 = 3
    var_22 = [var_19, var_20, var_21]
    var_23 = iter(var_22)
    var_24 = module_0.LazyList(var_23)
    var_25 = var_24.list
    var_26 = len(var_25)
    assert var_26 == 3
    var_27 = 10
    var_28 = var_6[var_27]
    var_29 = [var_19, var_20]
    var_30 = iter(var_29)
    var_31 = module_0.LazyList(var_30)
    var_32 = list(var_31)



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 11
    var_3 = 2
    var_4 = 5
    var_5 = 0
    var_6 = slice(var_5, var_4, var_3)
    var_7 = None
    var_8 = 3
    var_9 = slice(var_7, var_7, var_8)
    var_10 = 10
    var_11 = -11



# Parsed testcases at query #24
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = 6
    var_6 = [var_0, var_1, var_2, var_3, var_4, var_5]
    var_7 = 0
    var_8 = lambda x: x % var_1 == var_7
    var_9 = module_0.split_by(var_6, criterion=var_8)
    var_10 = list(var_9)
    var_11 = 'a,b,c'
    var_12 = ','
    var_13 = module_0.split_by(var_11, separator=var_12)
    var_14 = list(var_13)
    var_15 = ',a,b,'
    var_16 = True
    var_17 = module_0.split_by(var_15, var_16, separator=var_12)
    var_18 = list(var_17)
    var_19 = 'a,,b'
    var_20 = False
    var_21 = module_0.split_by(var_19, var_20, separator=var_12)
    var_22 = list(var_21)
    var_23 = 1
    var_24 = 2
    var_25 = [var_23, var_24]
    var_26 = 0
    var_27 = lambda x: x > var_26
    var_28 = module_0.split_by(var_25, criterion=var_27, separator=var_23)
    var_29 = list(var_28)
    var_30 = 1
    var_31 = 2
    var_32 = [var_30, var_31]
    var_33 = module_0.split_by(var_32)
    var_34 = list(var_33)
    var_35 = []
    var_36 = module_0.split_by(var_35, separator=var_12)
    var_37 = list(var_36)
    var_38 = [var_16]
    var_39 = lambda x: x == var_16
    var_40 = module_0.split_by(var_38, criterion=var_39)
    var_41 = list(var_40)
    var_42 = [var_16]
    var_43 = True
    var_44 = module_0.split_by(var_42, var_43, separator=var_16)
    var_45 = list(var_44)



# Parsed testcases at query #25
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = 6
    var_6 = 7
    var_7 = 8
    var_8 = [var_0, var_1, var_2, var_3, var_4, var_5, var_6, var_7]
    var_9 = [var_0, var_1]
    var_10 = [var_3, var_4]
    var_11 = [var_6, var_7]
    var_12 = [var_9, var_10, var_11]
    var_13 = 0
    var_14 = lambda x: x % var_2 == var_13
    var_15 = module_0.split_by(var_8, criterion=var_14)
    var_16 = list(var_15)
    var_17 = 'A.B.C'
    var_18 = 'A'
    var_19 = [var_18]
    var_20 = 'B'
    var_21 = [var_20]
    var_22 = 'C'
    var_23 = [var_22]
    var_24 = [var_19, var_21, var_23]
    var_25 = '.'
    var_26 = module_0.split_by(var_17, separator=var_25)
    var_27 = list(var_26)
    var_28 = '.A..B.'
    var_29 = []
    var_30 = [var_18]
    var_31 = []
    var_32 = [var_20]
    var_33 = []
    var_34 = [var_29, var_30, var_31, var_32, var_33]
    var_35 = True
    var_36 = module_0.split_by(var_28, var_35, separator=var_25)
    var_37 = list(var_36)
    var_38 = '.A..B.'
    var_39 = [var_18]
    var_40 = [var_20]
    var_41 = [var_39, var_40]
    var_42 = False
    var_43 = module_0.split_by(var_38, var_42, separator=var_25)
    var_44 = list(var_43)
    var_45 = []
    var_46 = ','
    var_47 = module_0.split_by(var_45, separator=var_46)
    var_48 = list(var_47)
    var_49 = [var_35, var_1, var_2]
    var_50 = []
    var_51 = True
    var_52 = lambda x: var_51
    var_53 = module_0.split_by(var_49, criterion=var_52)
    var_54 = list(var_53)
    var_55 = [var_51, var_1, var_2]
    var_56 = [var_51, var_1, var_2]
    var_57 = [var_56]
    var_58 = False
    var_59 = lambda x: var_58
    var_60 = module_0.split_by(var_55, criterion=var_59)
    var_61 = list(var_60)
    var_62 = 1
    var_63 = 2
    var_64 = [var_62, var_63]
    var_65 = True
    var_66 = lambda x: var_65
    var_67 = module_0.split_by(var_64, criterion=var_66, separator=var_65)
    var_68 = list(var_67)
    var_69 = 1
    var_70 = 2
    var_71 = [var_69, var_70]
    var_72 = module_0.split_by(var_71)
    var_73 = list(var_72)
    var_74 = 'a'
    var_75 = [var_46, var_74]
    var_76 = []
    var_77 = [var_74]
    var_78 = [var_76, var_77]
    var_79 = True
    var_80 = module_0.split_by(var_75, var_79, separator=var_46)
    var_81 = list(var_80)



# Parsed testcases at query #26
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x == var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_0, var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = 5
    var_9 = lambda x: x > var_8
    var_10 = 6
    var_11 = 7
    var_12 = 8
    var_13 = [var_2, var_3, var_10, var_11, var_12]
    var_14 = module_0.drop_until(var_9, var_13)
    var_15 = list(var_14)
    var_16 = 9
    var_17 = lambda x: x == var_16
    var_18 = [var_2, var_3, var_4, var_16]
    var_19 = module_0.drop_until(var_17, var_18)
    var_20 = list(var_19)
    var_21 = 10
    var_22 = lambda x: x > var_21
    var_23 = 4
    var_24 = [var_2, var_3, var_4, var_23]
    var_25 = module_0.drop_until(var_22, var_24)
    var_26 = list(var_25)
    var_27 = True
    var_28 = lambda x: var_27
    var_29 = []
    var_30 = module_0.drop_until(var_28, var_29)
    var_31 = list(var_30)
    var_32 = 'b'
    var_33 = lambda x: x == var_32
    var_34 = 'a'
    var_35 = 'c'
    var_36 = [var_34, var_32, var_35]
    var_37 = module_0.drop_until(var_33, var_36)
    var_38 = list(var_37)
    var_39 = lambda x: x % var_3 == var_0
    var_40 = [var_27, var_4, var_8, var_10, var_11]
    var_41 = module_0.drop_until(var_39, var_40)
    var_42 = list(var_41)
    var_43 = lambda x: x == var_3
    var_44 = [var_27, var_3, var_4, var_23, var_8]
    var_45 = module_0.drop_until(var_43, var_44)
    var_46 = next(var_45)
    assert var_46 == 2
    var_47 = next(var_45)
    assert var_47 == 3



# Parsed testcases at query #27
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x == var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_0, var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = 5
    var_9 = lambda x: x > var_8
    var_10 = 6
    var_11 = 7
    var_12 = 8
    var_13 = [var_2, var_3, var_10, var_11, var_12]
    var_14 = module_0.drop_until(var_9, var_13)
    var_15 = list(var_14)
    var_16 = 9
    var_17 = lambda x: x == var_16
    var_18 = [var_2, var_3, var_4, var_16]
    var_19 = module_0.drop_until(var_17, var_18)
    var_20 = list(var_19)
    var_21 = 10
    var_22 = lambda x: x > var_21
    var_23 = 4
    var_24 = [var_2, var_3, var_4, var_23]
    var_25 = module_0.drop_until(var_22, var_24)
    var_26 = list(var_25)
    var_27 = True
    var_28 = lambda x: var_27
    var_29 = []
    var_30 = module_0.drop_until(var_28, var_29)
    var_31 = list(var_30)
    var_32 = 'b'
    var_33 = lambda x: x == var_32
    var_34 = 'a'
    var_35 = 'c'
    var_36 = [var_34, var_32, var_35]
    var_37 = module_0.drop_until(var_33, var_36)
    var_38 = list(var_37)
    var_39 = 100
    var_40 = lambda x: x >= var_39
    var_41 = 200
    var_42 = range(var_41)
    var_43 = module_0.drop_until(var_40, var_42)
    var_44 = next(var_43)
    assert var_44 == 100



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 2
    var_2 = 6
    var_3 = 1
    var_4 = 10
    var_5 = 0
    var_6 = 3
    var_7 = []



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 3
    var_1 = 5
    var_2 = 8
    var_3 = 0
    var_4 = 10
    var_5 = -1



# Parsed testcases at query #3
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = 10
    var_2 = range(var_1)
    var_3 = module_0.chunk(var_0, var_2)
    var_4 = list(var_3)
    var_5 = 1
    var_6 = 2
    var_7 = [var_5, var_6, var_0]
    var_8 = module_0.chunk(var_5, var_7)
    var_9 = list(var_8)
    var_10 = [var_5, var_6, var_0]
    var_11 = module_0.chunk(var_1, var_10)
    var_12 = list(var_11)
    var_13 = []
    var_14 = module_0.chunk(var_0, var_13)
    var_15 = list(var_14)



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 2
    var_3 = 5
    var_4 = 15
    var_5 = 5



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 5
    var_2 = 15
    var_3 = 0
    var_4 = 2
    var_5 = 3
    var_6 = slice(var_3, var_5)
    var_7 = None
    var_8 = slice(var_4, var_7)
    var_9 = slice(var_7, var_1)
    var_10 = slice(var_3, var_1, var_4)
    var_11 = slice(var_0, var_2)
    var_12 = slice(var_1, var_3)
    var_13 = 10
    var_14 = -11



# Parsed testcases at query #6
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = 40
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = iter(var_4)
    var_6 = module_0.LazyList(var_5)
    var_7 = range(var_0)
    var_8 = iter(var_7)
    var_9 = module_0.LazyList(var_8)
    var_10 = 10
    var_11 = var_6[var_10]
    var_12 = 5
    var_13 = 6
    var_14 = 7
    var_15 = [var_12, var_13, var_14]
    var_16 = iter(var_15)
    var_17 = module_0.LazyList(var_16)
    var_18 = 1
    var_19 = 2
    var_20 = 3
    var_21 = [var_18, var_19, var_20]
    var_22 = iter(var_21)
    var_23 = module_0.LazyList(var_22)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = 10
    var_2 = 20
    var_3 = 30
    var_4 = 40
    var_5 = [var_1, var_2, var_3, var_4]
    var_6 = module_0.drop(var_0, var_5)
    var_7 = list(var_6)
    var_8 = 0
    var_9 = 1
    var_10 = 3
    var_11 = [var_9, var_0, var_10]
    var_12 = module_0.drop(var_8, var_11)
    var_13 = list(var_12)
    var_14 = 5
    var_15 = [var_9, var_0, var_10]
    var_16 = module_0.drop(var_14, var_15)
    var_17 = list(var_16)
    var_18 = range(var_1)
    var_19 = module_0.drop(var_10, var_18)
    var_20 = list(var_19)
    var_21 = []
    var_22 = module_0.drop(var_0, var_21)
    var_23 = list(var_22)
    var_24 = -1
    var_25 = 1
    var_26 = 2
    var_27 = 3
    var_28 = [var_25, var_26, var_27]
    var_29 = module_0.drop(var_24, var_28)
    var_30 = list(var_29)



# Parsed testcases at query #2
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = 6
    var_6 = [var_0, var_1, var_2, var_3, var_4, var_5]
    var_7 = 0
    var_8 = lambda x: x % var_1 == var_7
    var_9 = module_0.split_by(var_6, criterion=var_8)
    var_10 = list(var_9)
    var_11 = 'abc.def.ghi'
    var_12 = 'abc.def.ghi'
    var_13 = '.'
    var_14 = module_0.split_by(var_12, separator=var_13)
    var_15 = list(var_14)
    var_16 = '.a.b.'
    var_17 = '.a.b.'
    var_18 = True
    var_19 = module_0.split_by(var_17, var_18, separator=var_13)
    var_20 = list(var_19)
    var_21 = '.a.b.'
    var_22 = False
    var_23 = module_0.split_by(var_17, var_22, separator=var_13)
    var_24 = list(var_23)
    var_25 = 'a,,b'
    var_26 = 'a,,b'
    var_27 = True
    var_28 = ','
    var_29 = module_0.split_by(var_26, var_27, separator=var_28)
    var_30 = list(var_29)
    var_31 = 1
    var_32 = 2
    var_33 = 3
    var_34 = [var_31, var_32, var_33]
    var_35 = lambda x: x > var_31
    var_36 = module_0.split_by(var_34, criterion=var_35, separator=var_32)
    var_37 = list(var_36)
    var_38 = 1
    var_39 = 2
    var_40 = 3
    var_41 = [var_38, var_39, var_40]
    var_42 = module_0.split_by(var_41)
    var_43 = list(var_42)
    var_44 = []
    var_45 = module_0.split_by(var_44, separator=var_28)
    var_46 = list(var_45)
    var_47 = [var_39, var_41, var_43]
    var_48 = lambda x: x % var_39 == var_22
    var_49 = False
    var_50 = module_0.split_by(var_47, var_49, criterion=var_48)
    var_51 = list(var_50)
    var_52 = lambda x: x % var_39 == var_49
    var_53 = True
    var_54 = module_0.split_by(var_47, var_53, criterion=var_52)
    var_55 = list(var_54)



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
    var_8 = 'a'
    var_9 = 'b'
    var_10 = 'c'
    var_11 = [var_8, var_9, var_10]
    var_12 = lambda x: x.upper()
    var_13 = module_0.MapList(var_12, var_11)
    var_14 = lambda x: x
    var_15 = []
    var_16 = module_0.MapList(var_14, var_15)
    var_17 = 0
    var_18 = var_16[var_17]
    var_19 = 10
    var_20 = var_7[var_19]



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 2
    var_2 = 1
    var_3 = 10
    var_4 = 0
    var_5 = -1
    var_6 = 3



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 2
    var_2 = 6
    var_3 = 1
    var_4 = 10
    var_5 = 0
    var_6 = -1
    var_7 = 3
    var_8 = 0
    var_9 = 5
    var_10 = list(var_1)



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -5
    var_3 = 5
    var_4 = 15
    var_5 = 2
    var_6 = 3



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 11
    var_3 = 2
    var_4 = 0
    var_5 = 5
    var_6 = slice(var_3, var_5)
    var_7 = slice(var_1, var_5, var_3)
    var_8 = slice(var_4, var_5)
    var_9 = 100
    var_10 = slice(var_4, var_5, var_3)



# Parsed testcases at query #8
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
    var_8 = lambda s: s.upper()
    var_9 = 'a'
    var_10 = 'b'
    var_11 = 'c'
    var_12 = [var_9, var_10, var_11]
    var_13 = module_0.MapList(var_8, var_12)
    var_14 = 10
    var_15 = var_7[var_14]



# Parsed testcases at query #9
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = 0
    var_2 = 1
    var_3 = 2
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = 7
    var_8 = 8
    var_9 = 9
    var_10 = [var_1, var_2, var_3, var_0, var_4, var_5, var_6, var_7, var_8, var_9]
    var_11 = module_0.chunk(var_0, var_10)
    var_12 = list(var_11)
    var_13 = 10
    var_14 = [var_2, var_3, var_0]
    var_15 = module_0.chunk(var_13, var_14)
    var_16 = list(var_15)
    var_17 = [var_2, var_3, var_0]
    var_18 = module_0.chunk(var_2, var_17)
    var_19 = list(var_18)
    var_20 = []
    var_21 = module_0.chunk(var_0, var_20)
    var_22 = list(var_21)
    var_23 = range(var_4)
    var_24 = module_0.chunk(var_3, var_23)
    var_25 = list(var_24)
    var_26 = 0
    var_27 = 1
    var_28 = 2
    var_29 = 3
    var_30 = [var_27, var_28, var_29]
    var_31 = module_0.chunk(var_26, var_30)
    var_32 = list(var_31)
    var_33 = -1
    var_34 = 1
    var_35 = 2
    var_36 = 3
    var_37 = [var_34, var_35, var_36]
    var_38 = module_0.chunk(var_33, var_37)
    var_39 = list(var_38)
    var_40 = 'abcde'
    var_41 = module_0.chunk(var_36, var_40)
    var_42 = list(var_41)
    var_43 = module_0.chunk(var_36, var_40)
    var_44 = list(var_43)



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 11
    var_3 = 2
    var_4 = 5
    var_5 = 0



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 0
    var_4 = 10
    var_5 = -11



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 5
    var_4 = 0
    var_5 = 11
    var_6 = 10
    var_7 = -6



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
    var_8 = 'a'
    var_9 = 'b'
    var_10 = 'c'
    var_11 = [var_8, var_9, var_10]
    var_12 = lambda x: x.upper()
    var_13 = 10
    var_14 = var_7[var_13]



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 2
    var_3 = 5
    var_4 = 15
    var_5 = 1
    var_6 = 3
    var_7 = slice(var_0, var_6)
    var_8 = slice(var_2, var_3)
    var_9 = slice(var_0, var_1, var_2)
    var_10 = None
    var_11 = slice(var_10, var_10, var_6)
    var_12 = -5
    var_13 = -2
    var_14 = slice(var_12, var_13)
    var_15 = 5
    var_16 = -6



# Parsed testcases at query #15
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = 6
    var_6 = 7
    var_7 = [var_0, var_1, var_2, var_3, var_4, var_5, var_6]
    var_8 = [var_0]
    var_9 = [var_2]
    var_10 = [var_4]
    var_11 = [var_6]
    var_12 = [var_8, var_9, var_10, var_11]
    var_13 = 0
    var_14 = lambda x: x % var_1 == var_13
    var_15 = module_0.split_by(var_7, criterion=var_14)
    var_16 = list(var_15)
    var_17 = [var_0, var_1, var_2]
    var_18 = []
    var_19 = []
    var_20 = []
    var_21 = []
    var_22 = [var_18, var_19, var_20, var_21]
    var_23 = 10
    var_24 = range(var_23)
    var_25 = lambda x: x % var_2 == var_13
    var_26 = module_0.split_by(var_24, criterion=var_25)
    var_27 = list(var_26)
    var_28 = 'a.b.c'
    var_29 = '.'
    var_30 = module_0.split_by(var_28, separator=var_29)
    var_31 = list(var_30)
    var_32 = '.a..b.'
    var_33 = module_0.split_by(var_32, separator=var_29)
    var_34 = list(var_33)
    var_35 = True
    var_36 = module_0.split_by(var_32, var_35, separator=var_29)
    var_37 = list(var_36)
    var_38 = 1
    var_39 = 2
    var_40 = [var_38, var_39]
    var_41 = True
    var_42 = lambda x: var_41
    var_43 = module_0.split_by(var_40, criterion=var_42, separator=var_41)
    var_44 = list(var_43)
    var_45 = 1
    var_46 = 2
    var_47 = [var_45, var_46]
    var_48 = module_0.split_by(var_47)
    var_49 = list(var_48)
    var_50 = []
    var_51 = module_0.split_by(var_50, separator=var_29)
    var_52 = list(var_51)
    var_53 = [var_49]
    var_54 = lambda x: x == var_49
    var_55 = True
    var_56 = module_0.split_by(var_53, var_55, criterion=var_54)
    var_57 = list(var_56)



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 0
    var_3 = 2
    var_4 = 5
    var_5 = 10
    var_6 = -6



# Parsed testcases at query #17
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x == var_0
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_2, var_3]
    var_5 = module_0.drop_until(var_1, var_4)
    var_6 = list(var_5)
    var_7 = 5
    var_8 = lambda x: x == var_7
    var_9 = 6
    var_10 = 7
    var_11 = [var_0, var_2, var_7, var_9, var_10]
    var_12 = module_0.drop_until(var_8, var_11)
    var_13 = list(var_12)
    var_14 = 9
    var_15 = lambda x: x > var_14
    var_16 = 10
    var_17 = [var_0, var_2, var_16]
    var_18 = module_0.drop_until(var_15, var_17)
    var_19 = list(var_18)
    var_20 = lambda x: x > var_16
    var_21 = [var_0, var_2, var_3]
    var_22 = module_0.drop_until(var_20, var_21)
    var_23 = list(var_22)
    var_24 = True
    var_25 = lambda x: var_24
    var_26 = []
    var_27 = module_0.drop_until(var_25, var_26)
    var_28 = list(var_27)
    var_29 = 'b'
    var_30 = lambda s: s == var_29
    var_31 = 'a'
    var_32 = 'c'
    var_33 = [var_31, var_29, var_32]
    var_34 = module_0.drop_until(var_30, var_33)
    var_35 = list(var_34)
    var_36 = 0
    var_37 = lambda x: x % var_2 == var_36
    var_38 = [var_24, var_3, var_7, var_9, var_10]
    var_39 = module_0.drop_until(var_37, var_38)
    var_40 = list(var_39)



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 5
    var_2 = 15
    var_3 = 0
    var_4 = 2
    var_5 = None
    var_6 = slice(var_5, var_5, var_4)
    var_7 = 10



# Parsed testcases at query #19
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = 6
    var_6 = [var_0, var_1, var_2, var_3, var_4, var_5]
    var_7 = 0
    var_8 = lambda x: x % var_1 == var_7
    var_9 = module_0.split_by(var_6, criterion=var_8)
    var_10 = list(var_9)
    var_11 = 'abc.def.ghi'
    var_12 = list(var_11)
    var_13 = '.'
    var_14 = module_0.split_by(var_12, separator=var_13)
    var_15 = list(var_14)
    var_16 = 'a..b'
    var_17 = list(var_16)
    var_18 = True
    var_19 = module_0.split_by(var_17, var_18, separator=var_13)
    var_20 = list(var_19)
    var_21 = 'a..b'
    var_22 = list(var_21)
    var_23 = False
    var_24 = module_0.split_by(var_22, var_23, separator=var_13)
    var_25 = list(var_24)
    var_26 = [var_18, var_1, var_2]
    var_27 = True
    var_28 = lambda x: var_27
    var_29 = module_0.split_by(var_26, criterion=var_28)
    var_30 = list(var_29)
    var_31 = [var_27, var_1, var_2]
    var_32 = False
    var_33 = lambda x: var_32
    var_34 = module_0.split_by(var_31, criterion=var_33)
    var_35 = list(var_34)
    var_36 = 1
    var_37 = 2
    var_38 = [var_36, var_37]
    var_39 = True
    var_40 = lambda x: var_39
    var_41 = module_0.split_by(var_38, criterion=var_40, separator=var_39)
    var_42 = list(var_41)
    var_43 = 1
    var_44 = 2
    var_45 = [var_43, var_44]
    var_46 = module_0.split_by(var_45)
    var_47 = list(var_46)
    var_48 = []
    var_49 = module_0.split_by(var_48, separator=var_13)
    var_50 = list(var_49)
    var_51 = '.a.'
    var_52 = list(var_51)
    var_53 = True
    var_54 = module_0.split_by(var_52, var_53, separator=var_13)
    var_55 = list(var_54)



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 11
    var_3 = 2
    var_4 = 0
    var_5 = 5
    var_6 = 15



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
    var_6 = lambda x: x == var_2
    var_7 = module_0.drop_until(var_6, var_5)
    var_8 = list(var_7)
    var_9 = [var_0, var_1, var_2]
    var_10 = lambda x: x == var_2
    var_11 = module_0.drop_until(var_10, var_9)
    var_12 = list(var_11)
    var_13 = [var_0, var_1, var_2]
    var_14 = 10
    var_15 = lambda x: x > var_14
    var_16 = module_0.drop_until(var_15, var_13)
    var_17 = list(var_16)
    var_18 = 20
    var_19 = 30
    var_20 = [var_14, var_18, var_19]
    var_21 = lambda x: x >= var_14
    var_22 = module_0.drop_until(var_21, var_20)
    var_23 = list(var_22)
    var_24 = []
    var_25 = True
    var_26 = lambda x: var_25
    var_27 = module_0.drop_until(var_26, var_24)
    var_28 = list(var_27)
    var_29 = 'abcdef'
    var_30 = 'd'
    var_31 = lambda x: x == var_30
    var_32 = ''
    var_33 = module_0.drop_until(var_31, var_29)
    var_34 = range(var_14)
    var_35 = 0
    var_36 = lambda x: x % var_4 == var_35
    var_37 = 7
    var_38 = lambda x: x == var_37



# Parsed testcases at query #22
#--------------------------




# Parsed testcases at query #23
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = 40
    var_4 = 50
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.LazyList(var_5)
    var_7 = range(var_0)
    var_8 = module_0.LazyList(var_7)
    var_9 = 1
    var_10 = 4
    var_11 = var_8[var_9:var_10]
    var_12 = list(var_11)
    var_13 = 3
    var_14 = var_8[:var_13]
    var_15 = list(var_14)
    var_16 = 7
    var_17 = var_8[var_16:]
    var_18 = list(var_17)
    var_19 = 2
    var_20 = var_8[::var_19]
    var_21 = list(var_20)
    var_22 = []
    var_23 = 10
    var_24 = var_6[var_23]
    var_25 = [var_9, var_19, var_13]
    var_26 = module_0.LazyList(var_25)
    var_27 = range(var_13)
    var_28 = module_0.LazyList(var_27)
    var_29 = var_28[:]
    var_30 = list(var_29)



# Parsed testcases at query #24
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = 5
    var_9 = lambda x: x >= var_8
    var_10 = 6
    var_11 = 7
    var_12 = [var_2, var_3, var_8, var_10, var_11]
    var_13 = module_0.drop_until(var_9, var_12)
    var_14 = list(var_13)
    var_15 = 9
    var_16 = lambda x: x == var_15
    var_17 = [var_2, var_3, var_4, var_15]
    var_18 = module_0.drop_until(var_16, var_17)
    var_19 = list(var_18)
    var_20 = 10
    var_21 = lambda x: x > var_20
    var_22 = 4
    var_23 = [var_2, var_3, var_4, var_22]
    var_24 = module_0.drop_until(var_21, var_23)
    var_25 = list(var_24)
    var_26 = True
    var_27 = lambda x: var_26
    var_28 = []
    var_29 = module_0.drop_until(var_27, var_28)
    var_30 = list(var_29)
    var_31 = 'b'
    var_32 = lambda char: char == var_31
    var_33 = 'abcde'
    var_34 = module_0.drop_until(var_32, var_33)
    var_35 = list(var_34)
    var_36 = lambda x: x < var_0
    var_37 = -1
    var_38 = -2
    var_39 = [var_26, var_3, var_37, var_38]
    var_40 = module_0.drop_until(var_36, var_39)
    var_41 = list(var_40)
    var_42 = lambda x: x == var_4
    var_43 = [var_26, var_3, var_4, var_22, var_8]
    var_44 = iter(var_43)
    var_45 = module_0.drop_until(var_42, var_44)
    var_46 = next(var_45)
    assert var_46 == 3
    var_47 = next(var_45)
    assert var_47 == 4



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 5
    var_2 = 15
    var_3 = 0
    var_4 = 2
    var_5 = 3
    var_6 = slice(var_3, var_5)
    var_7 = 1
    var_8 = 4
    var_9 = slice(var_7, var_8)
    var_10 = None
    var_11 = slice(var_10, var_10, var_4)
    var_12 = slice(var_3, var_0)
    var_13 = 10
    var_14 = -11



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 5
    var_2 = 15
    var_3 = 0
    var_4 = 2
    var_5 = 20
    var_6 = 1
    var_7 = 3
    var_8 = slice(var_3, var_0, var_7)
    var_9 = 10



# Parsed testcases at query #27
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 0
    var_3 = 2
    var_4 = 5
    var_5 = module_0.Range()



# Parsed testcases at query #28
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x == var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_0, var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = 5
    var_9 = lambda x: x > var_8
    var_10 = 10
    var_11 = range(var_10)
    var_12 = module_0.drop_until(var_9, var_11)
    var_13 = list(var_12)
    var_14 = lambda x: x > var_10
    var_15 = 4
    var_16 = [var_2, var_3, var_4, var_15]
    var_17 = module_0.drop_until(var_14, var_16)
    var_18 = list(var_17)
    var_19 = True
    var_20 = lambda x: var_19
    var_21 = []
    var_22 = module_0.drop_until(var_20, var_21)
    var_23 = list(var_22)
    var_24 = 9
    var_25 = lambda x: x == var_24
    var_26 = [var_19, var_3, var_24, var_0]
    var_27 = module_0.drop_until(var_25, var_26)
    var_28 = list(var_27)
    var_29 = 'b'
    var_30 = lambda x: x == var_29
    var_31 = 'a'
    var_32 = 'c'
    var_33 = [var_31, var_29, var_32]
    var_34 = module_0.drop_until(var_30, var_33)
    var_35 = list(var_34)
    var_36 = lambda x: x % var_3 == var_0 and x > var_3
    var_37 = [var_19, var_4, var_15, var_8]
    var_38 = module_0.drop_until(var_36, var_37)
    var_39 = list(var_38)



# Parsed testcases at query #29
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = 5
    var_9 = lambda x: x == var_8
    var_10 = 6
    var_11 = 7
    var_12 = [var_2, var_3, var_8, var_10, var_11]
    var_13 = module_0.drop_until(var_9, var_12)
    var_14 = list(var_13)
    var_15 = 10
    var_16 = lambda x: x > var_15
    var_17 = 11
    var_18 = [var_2, var_3, var_4, var_17]
    var_19 = module_0.drop_until(var_16, var_18)
    var_20 = list(var_19)
    var_21 = lambda x: x > var_15
    var_22 = 4
    var_23 = [var_2, var_3, var_4, var_22]
    var_24 = module_0.drop_until(var_21, var_23)
    var_25 = list(var_24)
    var_26 = True
    var_27 = lambda x: var_26
    var_28 = []
    var_29 = module_0.drop_until(var_27, var_28)
    var_30 = list(var_29)
    var_31 = 'b'
    var_32 = lambda x: x == var_31
    var_33 = 'a'
    var_34 = 'c'
    var_35 = [var_33, var_31, var_34]
    var_36 = module_0.drop_until(var_32, var_35)
    var_37 = list(var_36)
    var_38 = False
    var_39 = lambda x: var_38
    var_40 = [var_26, var_3, var_4]
    var_41 = module_0.drop_until(var_39, var_40)
    var_42 = list(var_41)
    var_43 = lambda x: x % var_3 == var_38
    var_44 = 8
    var_45 = [var_26, var_4, var_8, var_10, var_11, var_44]
    var_46 = module_0.drop_until(var_43, var_45)
    var_47 = list(var_46)



# Parsed testcases at query #30
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = 6
    var_6 = [var_0, var_1, var_2, var_3, var_4, var_5]
    var_7 = [var_0, var_1]
    var_8 = [var_3, var_4]
    var_9 = [var_7, var_8]
    var_10 = 0
    var_11 = lambda x: x % var_2 == var_10
    var_12 = module_0.split_by(var_6, criterion=var_11)
    var_13 = list(var_12)
    var_14 = 'a,b,c'
    var_15 = 'a'
    var_16 = [var_15]
    var_17 = 'b'
    var_18 = [var_17]
    var_19 = 'c'
    var_20 = [var_19]
    var_21 = [var_16, var_18, var_20]
    var_22 = ','
    var_23 = module_0.split_by(var_14, separator=var_22)
    var_24 = list(var_23)
    var_25 = ',a,,b,'
    var_26 = []
    var_27 = [var_15]
    var_28 = []
    var_29 = [var_17]
    var_30 = []
    var_31 = [var_26, var_27, var_28, var_29, var_30]
    var_32 = True
    var_33 = module_0.split_by(var_25, var_32, separator=var_22)
    var_34 = list(var_33)
    var_35 = [var_15]
    var_36 = [var_17]
    var_37 = [var_35, var_36]
    var_38 = False
    var_39 = module_0.split_by(var_25, var_38, separator=var_22)
    var_40 = list(var_39)
    var_41 = 1
    var_42 = 2
    var_43 = [var_41, var_42]
    var_44 = 0
    var_45 = lambda x: x > var_44
    var_46 = module_0.split_by(var_43, criterion=var_45, separator=var_41)
    var_47 = list(var_46)
    var_48 = 1
    var_49 = 2
    var_50 = [var_48, var_49]
    var_51 = module_0.split_by(var_50)
    var_52 = list(var_51)
    var_53 = []
    var_54 = module_0.split_by(var_53, separator=var_22)
    var_55 = list(var_54)
    var_56 = [var_32, var_32, var_32]
    var_57 = True
    var_58 = lambda x: var_57
    var_59 = module_0.split_by(var_56, criterion=var_58)
    var_60 = list(var_59)
    var_61 = [var_57, var_49, var_50]
    var_62 = False
    var_63 = lambda x: var_62
    var_64 = module_0.split_by(var_61, criterion=var_63)
    var_65 = list(var_64)
    var_66 = 'abc;'
    var_67 = ';'
    var_68 = True
    var_69 = module_0.split_by(var_66, var_68, separator=var_67)
    var_70 = list(var_69)



# Parsed testcases at query #31
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x == var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_0, var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = 5
    var_9 = lambda x: x > var_8
    var_10 = 6
    var_11 = 7
    var_12 = 8
    var_13 = [var_2, var_3, var_10, var_11, var_12]
    var_14 = module_0.drop_until(var_9, var_13)
    var_15 = list(var_14)
    var_16 = 9
    var_17 = lambda x: x == var_16
    var_18 = [var_2, var_3, var_4, var_16]
    var_19 = module_0.drop_until(var_17, var_18)
    var_20 = list(var_19)
    var_21 = 10
    var_22 = lambda x: x > var_21
    var_23 = 4
    var_24 = [var_2, var_3, var_4, var_23]
    var_25 = True
    var_26 = lambda x: var_25
    var_27 = []
    var_28 = module_0.drop_until(var_26, var_27)
    var_29 = list(var_28)
    var_30 = 'b'
    var_31 = lambda x: x == var_30
    var_32 = 'abcde'
    var_33 = module_0.drop_until(var_31, var_32)
    var_34 = list(var_33)
    var_35 = lambda x: x < var_0
    var_36 = [var_25, var_3, var_4]
    var_37 = module_0.drop_until(var_35, var_36)
    var_38 = list(var_37)



