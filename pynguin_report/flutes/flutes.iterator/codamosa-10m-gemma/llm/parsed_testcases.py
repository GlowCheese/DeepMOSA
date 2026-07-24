####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
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
    var_18 = 'iter'
    var_19 = [var_16, var_17, var_18]
    var_20 = module_0.take(var_15, var_19)
    var_21 = list(var_20)
    var_22 = -1
    var_23 = 5
    var_24 = range(var_23)
    var_25 = module_0.take(var_22, var_24)
    var_26 = list(var_25)
    var_27 = 3



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -5
    var_3 = 5
    var_4 = 15
    var_5 = 2
    var_6 = range(var_1, var_0, var_5)
    var_7 = len(var_6)
    var_8 = 3
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = 4



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
    var_6 = 20
    var_7 = 30
    var_8 = [var_1, var_6, var_7]
    var_9 = module_0.chunk(var_5, var_8)
    var_10 = list(var_9)
    var_11 = 2
    var_12 = [var_5, var_11, var_0]
    var_13 = module_0.chunk(var_1, var_12)
    var_14 = list(var_13)
    var_15 = []
    var_16 = module_0.chunk(var_0, var_15)
    var_17 = list(var_16)
    var_18 = 'abcde'
    var_19 = module_0.chunk(var_11, var_18)
    var_20 = list(var_19)
    var_21 = 0
    var_22 = 1
    var_23 = 2
    var_24 = 3
    var_25 = [var_22, var_23, var_24]
    var_26 = module_0.chunk(var_21, var_25)
    var_27 = list(var_26)
    var_28 = -1
    var_29 = 1
    var_30 = 2
    var_31 = 3
    var_32 = [var_29, var_30, var_31]
    var_33 = module_0.chunk(var_28, var_32)
    var_34 = list(var_33)
    var_35 = 4
    var_36 = [var_33, var_11, var_28, var_35]
    var_37 = iter(var_36)
    var_38 = module_0.chunk(var_11, var_37)
    var_39 = list(var_38)



# Parsed testcases at query #4
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = range(var_1)
    var_3 = module_0.take(var_0, var_2)
    var_4 = list(var_3)
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]
    var_9 = module_0.take(var_1, var_8)
    var_10 = list(var_9)
    var_11 = 0
    var_12 = [var_5, var_6, var_7]
    var_13 = module_0.take(var_11, var_12)
    var_14 = list(var_13)
    var_15 = []
    var_16 = module_0.take(var_0, var_15)
    var_17 = list(var_16)
    var_18 = -1
    var_19 = 1
    var_20 = 2
    var_21 = 3
    var_22 = [var_19, var_20, var_21]
    var_23 = module_0.take(var_18, var_22)
    var_24 = list(var_23)
    var_25 = 'hello'
    var_26 = module_0.take(var_7, var_25)
    var_27 = list(var_26)
    var_28 = 4
    var_29 = [var_23, var_24, var_7, var_28, var_18]
    var_30 = iter(var_29)
    var_31 = module_0.take(var_24, var_30)
    var_32 = list(var_31)
    var_33 = list(var_30)



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 3
    var_1 = 5
    var_2 = 8
    var_3 = 0
    var_4 = 6
    var_5 = 2
    var_6 = -1



# Parsed testcases at query #6
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = 0
    var_2 = 1
    var_3 = 3
    var_4 = 4
    var_5 = [var_1, var_2, var_0, var_3, var_4]
    var_6 = module_0.drop(var_0, var_5)
    var_7 = list(var_6)
    var_8 = 5
    var_9 = [var_1, var_2, var_0]
    var_10 = module_0.drop(var_8, var_9)
    var_11 = list(var_10)
    var_12 = [var_2, var_0, var_3]
    var_13 = module_0.drop(var_1, var_12)
    var_14 = list(var_13)
    var_15 = []
    var_16 = module_0.drop(var_0, var_15)
    var_17 = list(var_16)
    var_18 = 10
    var_19 = range(var_18)
    var_20 = module_0.drop(var_3, var_19)
    var_21 = list(var_20)
    var_22 = 'hello'
    var_23 = module_0.drop(var_0, var_22)
    var_24 = list(var_23)
    var_25 = -1
    var_26 = 1
    var_27 = 2
    var_28 = 3
    var_29 = [var_26, var_27, var_28]
    var_30 = module_0.drop(var_25, var_29)
    var_31 = list(var_30)
    var_32 = [var_27, var_25]
    var_33 = module_0.drop(var_27, var_32)
    var_34 = iter(var_33)
    var_35 = next(var_33)
    assert var_35 == 2
    var_36 = next(var_33)



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
    var_12 = [var_1, var_2, var_3]
    var_13 = module_0.drop(var_8, var_12)
    var_14 = list(var_13)
    var_15 = 10
    var_16 = [var_1, var_2, var_3]
    var_17 = module_0.drop(var_15, var_16)
    var_18 = list(var_17)
    var_19 = []
    var_20 = module_0.drop(var_1, var_19)
    var_21 = list(var_20)
    var_22 = range(var_15)
    var_23 = module_0.drop(var_3, var_22)
    var_24 = list(var_23)
    var_25 = -1
    var_26 = 1
    var_27 = 2
    var_28 = 3
    var_29 = [var_26, var_27, var_28]
    var_30 = module_0.drop(var_25, var_29)
    var_31 = list(var_30)
    var_32 = ''
    var_33 = 'hello'
    var_34 = module_0.drop(var_27, var_33)



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 2
    var_2 = 6
    var_3 = 1
    var_4 = 10
    var_5 = -1
    var_6 = 0



# Parsed testcases at query #9
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x == var_0
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = [var_0, var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = 5
    var_9 = lambda x: x > var_8
    var_10 = 6
    var_11 = 7
    var_12 = 8
    var_13 = [var_0, var_2, var_10, var_11, var_12]
    var_14 = module_0.drop_until(var_9, var_13)
    var_15 = list(var_14)
    var_16 = lambda x: x == var_4
    var_17 = [var_0, var_2, var_3, var_4]
    var_18 = module_0.drop_until(var_16, var_17)
    var_19 = list(var_18)
    var_20 = 10
    var_21 = lambda x: x > var_20
    var_22 = [var_0, var_2, var_3, var_4]
    var_23 = module_0.drop_until(var_21, var_22)
    var_24 = list(var_23)
    var_25 = 0
    var_26 = lambda x: x > var_25
    var_27 = []
    var_28 = module_0.drop_until(var_26, var_27)
    var_29 = list(var_28)
    var_30 = 'b'
    var_31 = lambda x: x == var_30
    var_32 = 'a'
    var_33 = 'c'
    var_34 = [var_32, var_30, var_33]
    var_35 = module_0.drop_until(var_31, var_34)
    var_36 = list(var_35)
    var_37 = lambda x: x % var_2 == var_25
    var_38 = [var_0, var_3, var_8, var_10, var_11, var_12]
    var_39 = module_0.drop_until(var_37, var_38)
    var_40 = list(var_39)



# Parsed testcases at query #10
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
    var_14 = [var_9, var_0, var_10]
    var_15 = module_0.drop(var_10, var_14)
    var_16 = list(var_15)
    var_17 = 5
    var_18 = [var_9, var_0]
    var_19 = module_0.drop(var_17, var_18)
    var_20 = list(var_19)
    var_21 = range(var_1)
    var_22 = module_0.drop(var_10, var_21)
    var_23 = list(var_22)
    var_24 = []
    var_25 = module_0.drop(var_9, var_24)
    var_26 = list(var_25)
    var_27 = -1
    var_28 = 1
    var_29 = 2
    var_30 = 3
    var_31 = [var_28, var_29, var_30]
    var_32 = module_0.drop(var_27, var_31)
    var_33 = list(var_32)
    var_34 = [var_9, var_27, var_10]
    var_35 = iter(var_34)
    var_36 = module_0.drop(var_9, var_35)
    var_37 = next(var_36)
    assert var_37 == 2
    var_38 = next(var_36)
    assert var_38 == 3
    var_39 = next(var_36)



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 2
    var_2 = 1
    var_3 = 10
    var_4 = 0
    var_5 = -1
    var_6 = 3



# Parsed testcases at query #12
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = iter(var_5)
    var_7 = module_0.LazyList(var_6)
    var_8 = list(var_7)
    var_9 = 10
    var_10 = 20
    var_11 = 30
    var_12 = 40
    var_13 = 50
    var_14 = [var_9, var_10, var_11, var_12, var_13]
    var_15 = iter(var_14)
    var_16 = module_0.LazyList(var_15)
    var_17 = var_16[var_1]
    var_18 = list(var_16)
    var_19 = [var_0, var_1]
    var_20 = iter(var_19)
    var_21 = module_0.LazyList(var_20)
    var_22 = iter(var_21)
    var_23 = next(var_22)
    var_24 = next(var_22)
    var_25 = next(var_22)
    var_26 = list(var_21)
    var_27 = []
    var_28 = iter(var_27)
    var_29 = module_0.LazyList(var_28)
    var_30 = list(var_29)
    var_31 = [var_25, var_1, var_2]
    var_32 = iter(var_31)
    var_33 = module_0.LazyList(var_32)
    var_34 = iter(var_33)
    var_35 = iter(var_33)
    var_36 = next(var_34)
    assert var_36 == 1
    var_37 = next(var_35)
    assert var_37 == 1
    var_38 = next(var_34)
    assert var_38 == 2



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -5
    var_3 = 5
    var_4 = 15
    var_5 = 2
    var_6 = 3
    var_7 = 11
    var_8 = 1
    var_9 = -1
    var_10 = -2



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 11
    var_3 = 2
    var_4 = 0
    var_5 = 5
    var_6 = 10
    var_7 = -11



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
    var_6 = module_0.LazyList(var_5)
    var_7 = list(var_6)
    var_8 = 10
    var_9 = var_6[var_8]
    var_10 = list(var_6)
    var_11 = []
    var_12 = module_0.LazyList(var_11)
    var_13 = list(var_12)
    var_14 = [var_8, var_8, var_8]
    var_15 = module_0.LazyList(var_14)
    var_16 = iter(var_15)
    var_17 = list(var_16)
    var_18 = iter(var_15)
    var_19 = list(var_18)



# Parsed testcases at query #16
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
    var_9 = [var_3, var_7, var_8]
    var_10 = lambda x: x == var_3
    var_11 = module_0.split_by(var_9, criterion=var_10)
    var_12 = list(var_11)
    var_13 = [var_7, var_8, var_3]
    var_14 = lambda x: x == var_3
    var_15 = module_0.split_by(var_13, criterion=var_14)
    var_16 = list(var_15)
    var_17 = 'abc.def.ghi'
    var_18 = '.'
    var_19 = module_0.split_by(var_17, separator=var_18)
    var_20 = list(var_19)
    var_21 = '.a.'
    var_22 = True
    var_23 = module_0.split_by(var_21, var_22, separator=var_18)
    var_24 = list(var_23)
    var_25 = False
    var_26 = module_0.split_by(var_21, var_25, separator=var_18)
    var_27 = list(var_26)
    var_28 = 'a,,b'
    var_29 = True
    var_30 = ','
    var_31 = module_0.split_by(var_28, var_29, separator=var_30)
    var_32 = list(var_31)
    var_33 = False
    var_34 = module_0.split_by(var_28, var_33, separator=var_30)
    var_35 = list(var_34)
    var_36 = 1
    var_37 = 2
    var_38 = 3
    var_39 = [var_36, var_37, var_38]
    var_40 = lambda x: x > var_36
    var_41 = module_0.split_by(var_39, criterion=var_40, separator=var_37)
    var_42 = list(var_41)



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 5
    var_2 = 15
    var_3 = 0
    var_4 = 2
    var_5 = 1
    var_6 = 20
    var_7 = slice(var_3, var_1, var_4)



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 10
    var_2 = -11
    var_3 = 1
    var_4 = 9
    var_5 = 2
    var_6 = 5
    var_7 = 0
    var_8 = -6



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
    var_6 = 0
    var_7 = lambda x: x == var_6
    var_8 = range(var_2)
    var_9 = module_0.drop_until(var_7, var_8)
    var_10 = list(var_9)
    var_11 = 100
    var_12 = lambda x: x > var_11
    var_13 = range(var_2)
    var_14 = module_0.drop_until(var_12, var_13)
    var_15 = list(var_14)
    var_16 = 'c'
    var_17 = lambda x: x == var_16
    var_18 = 'abcde'
    var_19 = module_0.drop_until(var_17, var_18)
    var_20 = list(var_19)
    var_21 = True
    var_22 = lambda x: var_21
    var_23 = []
    var_24 = module_0.drop_until(var_22, var_23)
    var_25 = list(var_24)
    var_26 = lambda x: var_21
    var_27 = 2
    var_28 = 3
    var_29 = [var_21, var_27, var_28]
    var_30 = module_0.drop_until(var_26, var_29)
    var_31 = list(var_30)
    var_32 = False
    var_33 = lambda x: var_32
    var_34 = [var_21, var_27, var_28]
    var_35 = module_0.drop_until(var_33, var_34)
    var_36 = list(var_35)
    var_37 = 'a'
    var_38 = (var_21, var_37)
    var_39 = 'b'
    var_40 = (var_27, var_39)
    var_41 = (var_28, var_16)
    var_42 = [var_38, var_40, var_41]
    var_43 = lambda x: x[var_32] == var_27
    var_44 = module_0.drop_until(var_43, var_42)
    var_45 = list(var_44)



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 2
    var_3 = 1
    var_4 = 10
    var_5 = -6
    var_6 = 5



# Parsed testcases at query #21
#--------------------------




# Parsed testcases at query #22
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
    var_7 = 10
    var_8 = var_6[var_7]
    var_9 = var_6.list
    var_10 = len(var_9)
    assert var_10 == 5
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = [var_11, var_12, var_13]
    var_15 = iter(var_14)
    var_16 = module_0.LazyList(var_15)
    var_17 = 4
    var_18 = 5
    var_19 = [var_11, var_12, var_13, var_17, var_18]
    var_20 = module_0.LazyList(var_19)
    var_21 = []
    var_22 = module_0.LazyList(var_21)
    var_23 = 0
    var_24 = var_22[var_23]



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
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
    var_12 = [var_1, var_2, var_3, var_7, var_8]
    var_13 = module_0.drop(var_8, var_12)
    var_14 = list(var_13)
    var_15 = 10
    var_16 = [var_1, var_2, var_3]
    var_17 = module_0.drop(var_15, var_16)
    var_18 = list(var_17)
    var_19 = range(var_15)
    var_20 = module_0.drop(var_3, var_19)
    var_21 = list(var_20)
    var_22 = []
    var_23 = module_0.drop(var_1, var_22)
    var_24 = list(var_23)
    var_25 = -1
    var_26 = 1
    var_27 = 2
    var_28 = 3
    var_29 = [var_26, var_27, var_28]
    var_30 = module_0.drop(var_25, var_29)
    var_31 = list(var_30)
    var_32 = ''
    var_33 = 'hello'
    var_34 = module_0.drop(var_27, var_33)



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
    var_7 = 1
    var_8 = 2
    var_9 = [var_3, var_7, var_8]
    var_10 = lambda x: x == var_3
    var_11 = module_0.split_by(var_9, criterion=var_10)
    var_12 = list(var_11)
    var_13 = [var_7, var_8, var_3]
    var_14 = lambda x: x == var_3
    var_15 = module_0.split_by(var_13, criterion=var_14)
    var_16 = list(var_15)
    var_17 = 'a,b,c'
    var_18 = ','
    var_19 = module_0.split_by(var_17, separator=var_18)
    var_20 = list(var_19)
    var_21 = ',a,b,'
    var_22 = module_0.split_by(var_21, separator=var_18)
    var_23 = list(var_22)
    var_24 = 'a,,b'
    var_25 = module_0.split_by(var_24, separator=var_18)
    var_26 = list(var_25)
    var_27 = 'a,b'
    var_28 = True
    var_29 = module_0.split_by(var_27, var_28, separator=var_18)
    var_30 = list(var_29)
    var_31 = ',a'
    var_32 = True
    var_33 = module_0.split_by(var_31, var_32, separator=var_18)
    var_34 = list(var_33)
    var_35 = ',,'
    var_36 = True
    var_37 = module_0.split_by(var_35, var_36, separator=var_18)
    var_38 = list(var_37)
    var_39 = 1
    var_40 = 2
    var_41 = [var_39, var_40]
    var_42 = True
    var_43 = lambda x: var_42
    var_44 = module_0.split_by(var_41, criterion=var_43, separator=var_42)
    var_45 = list(var_44)
    var_46 = 1
    var_47 = 2
    var_48 = [var_46, var_47]
    var_49 = module_0.split_by(var_48)
    var_50 = list(var_49)
    var_51 = []
    var_52 = module_0.split_by(var_51, separator=var_18)
    var_53 = list(var_52)
    var_54 = [var_36, var_8, var_48]
    var_55 = 9
    var_56 = module_0.split_by(var_54, separator=var_55)
    var_57 = list(var_56)



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
    var_7 = 1
    var_8 = 2
    var_9 = [var_3, var_7, var_8]
    var_10 = lambda x: x == var_3
    var_11 = module_0.split_by(var_9, criterion=var_10)
    var_12 = list(var_11)
    var_13 = [var_7, var_8, var_3]
    var_14 = lambda x: x == var_3
    var_15 = module_0.split_by(var_13, criterion=var_14)
    var_16 = list(var_15)
    var_17 = [var_7, var_8, var_2]
    var_18 = True
    var_19 = lambda x: var_18
    var_20 = module_0.split_by(var_17, criterion=var_19)
    var_21 = list(var_20)
    var_22 = 'a.b.c'
    var_23 = '.'
    var_24 = module_0.split_by(var_22, separator=var_23)
    var_25 = list(var_24)
    var_26 = '.a.'
    var_27 = True
    var_28 = module_0.split_by(var_26, var_27, separator=var_23)
    var_29 = list(var_28)
    var_30 = False
    var_31 = module_0.split_by(var_26, var_30, separator=var_23)
    var_32 = list(var_31)
    var_33 = [var_27, var_30, var_30, var_8]
    var_34 = True
    var_35 = module_0.split_by(var_33, var_34, separator=var_30)
    var_36 = list(var_35)
    var_37 = [var_34, var_30, var_30, var_8]
    var_38 = False
    var_39 = module_0.split_by(var_37, var_38, separator=var_30)
    var_40 = list(var_39)
    var_41 = 1
    var_42 = 2
    var_43 = 3
    var_44 = [var_41, var_42, var_43]
    var_45 = lambda x: x > var_41
    var_46 = module_0.split_by(var_44, criterion=var_45, separator=var_42)
    var_47 = list(var_46)
    var_48 = 1
    var_49 = 2
    var_50 = 3
    var_51 = [var_48, var_49, var_50]
    var_52 = module_0.split_by(var_51)
    var_53 = list(var_52)
    var_54 = []
    var_55 = module_0.split_by(var_54, separator=var_34)
    var_56 = list(var_55)
    var_57 = []
    var_58 = True
    var_59 = lambda x: var_58
    var_60 = module_0.split_by(var_57, criterion=var_59)
    var_61 = list(var_60)
    var_62 = [var_58]
    var_63 = module_0.split_by(var_62, separator=var_8)
    var_64 = list(var_63)
    var_65 = [var_58]
    var_66 = True
    var_67 = module_0.split_by(var_65, var_66, separator=var_58)
    var_68 = list(var_67)



# Parsed testcases at query #4
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 0
    var_6 = lambda s, x: x + s
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'c'
    var_10 = 'd'
    var_11 = [var_7, var_8, var_9, var_10]
    var_12 = module_0.scanl(var_6, var_11)
    var_13 = list(var_12)
    var_14 = 5
    var_15 = [var_14]
    var_16 = [var_14]
    var_17 = []
    var_18 = 10
    var_19 = []
    var_20 = list(var_1)
    var_21 = ' '
    var_22 = lambda acc, x: acc + var_21 + x
    var_23 = 'Hello'
    var_24 = 'World'
    var_25 = [var_23, var_24]
    var_26 = 'Start'
    var_27 = 1
    var_28 = 2
    var_29 = [var_27, var_28]
    var_30 = 0
    var_31 = 10
    var_32 = list(var_5)
    var_33 = [var_27, var_28, var_29]
    var_34 = '__iter__'



# Parsed testcases at query #5
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
    var_8 = 10
    var_9 = [var_1, var_2, var_0]
    var_10 = module_0.take(var_8, var_9)
    var_11 = list(var_10)
    var_12 = 0
    var_13 = [var_1, var_2, var_0]
    var_14 = module_0.take(var_12, var_13)
    var_15 = list(var_14)
    var_16 = []
    var_17 = module_0.take(var_4, var_16)
    var_18 = list(var_17)
    var_19 = 100
    var_20 = range(var_19)
    var_21 = module_0.take(var_4, var_20)
    var_22 = list(var_21)
    var_23 = -1
    var_24 = 1
    var_25 = 2
    var_26 = 3
    var_27 = [var_24, var_25, var_26]
    var_28 = module_0.take(var_23, var_27)
    var_29 = list(var_28)
    var_30 = ''
    var_31 = 'hello'
    var_32 = module_0.take(var_25, var_31)



# Parsed testcases at query #6
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
    var_7 = 0
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = 4
    var_12 = 5
    var_13 = [var_7, var_8, var_9, var_10, var_11, var_12]
    var_14 = module_0.LazyList(var_13)
    var_15 = 10
    var_16 = var_6[var_15]
    var_17 = [var_8, var_9, var_10]
    var_18 = module_0.LazyList(var_17)



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 11
    var_3 = 0
    var_4 = 2
    var_5 = 5
    var_6 = -1



# Parsed testcases at query #8
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = len(var_4)
    var_6 = 4
    var_7 = 5
    var_8 = [var_5, var_1, var_2, var_6, var_7]
    var_9 = module_0.LazyList(var_8)
    var_10 = len(var_9)
    assert var_10 == 5
    var_11 = 10
    var_12 = 20
    var_13 = 30
    var_14 = [var_11, var_12, var_13]
    var_15 = module_0.LazyList(var_14)
    var_16 = var_15[var_1]
    var_17 = len(var_15)
    assert var_17 == 3
    var_18 = []
    var_19 = module_0.LazyList(var_18)
    var_20 = len(var_19)
    assert var_20 == 0
    var_21 = range(var_11)
    var_22 = module_0.LazyList(var_21)
    var_23 = 0
    var_24 = var_22[var_23:var_7]
    var_25 = range(var_7)
    var_26 = module_0.LazyList(var_25)
    var_27 = var_26[var_23:var_7]
    var_28 = len(var_26)
    assert var_28 == 5



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 5
    var_2 = 15
    var_3 = 0
    var_4 = 2
    var_5 = -2
    var_6 = 1



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -5
    var_3 = 5
    var_4 = 15
    var_5 = -5
    var_6 = 2
    var_7 = 11



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 3
    var_1 = 5
    var_2 = 8
    var_3 = 1
    var_4 = 10



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 3
    var_1 = 5
    var_2 = 8
    var_3 = 1
    var_4 = 10



# Parsed testcases at query #13
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = len(var_2)
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = module_0.LazyList(var_7)
    var_9 = len(var_8)
    assert var_9 == 3
    var_10 = 20
    var_11 = 30
    var_12 = 40
    var_13 = [var_3, var_10, var_11, var_12]
    var_14 = module_0.LazyList(var_13)
    var_15 = var_14[var_5]
    var_16 = len(var_14)
    var_17 = var_14[var_6]
    var_18 = len(var_14)
    assert var_18 == 4
    var_19 = []
    var_20 = module_0.LazyList(var_19)
    var_21 = len(var_20)
    assert var_21 == 0
    var_22 = 5
    var_23 = range(var_22)
    var_24 = module_0.LazyList(var_23)
    var_25 = 0
    var_26 = slice(var_25, var_22)
    var_27 = var_24[var_26]
    var_28 = len(var_24)
    assert var_28 == 5



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 3
    var_1 = 1
    var_2 = 4
    var_3 = 0
    var_4 = 5
    var_5 = 2
    var_6 = -1



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
    var_6 = iter(var_5)
    var_7 = module_0.LazyList(var_6)
    var_8 = iter(var_7)
    var_9 = next(var_8)
    assert var_9 == 1
    var_10 = next(var_8)
    assert var_10 == 2
    var_11 = next(var_8)
    assert var_11 == 3
    var_12 = next(var_8)
    assert var_12 == 4
    var_13 = next(var_8)
    assert var_13 == 5
    var_14 = next(var_8)
    var_15 = 10
    var_16 = 20
    var_17 = 30
    var_18 = [var_15, var_16, var_17]
    var_19 = iter(var_18)
    var_20 = module_0.LazyList(var_19)
    var_21 = list(var_20)
    var_22 = list(var_20)
    var_23 = []
    var_24 = iter(var_23)
    var_25 = module_0.LazyList(var_24)
    var_26 = list(var_25)
    var_27 = [var_14, var_1]
    var_28 = iter(var_27)
    var_29 = module_0.LazyList(var_28)



# Parsed testcases at query #16
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.LazyList(var_5)
    var_7 = list(var_6)
    var_8 = 10
    var_9 = range(var_8)
    var_10 = module_0.LazyList(var_9)
    var_11 = var_10[var_1]
    var_12 = list(var_10)
    var_13 = 20
    var_14 = [var_8, var_13]
    var_15 = module_0.LazyList(var_14)
    var_16 = list(var_15)
    var_17 = []
    var_18 = module_0.LazyList(var_17)
    var_19 = list(var_18)
    var_20 = [var_0, var_1]
    var_21 = module_0.LazyList(var_20)
    var_22 = iter(var_21)
    var_23 = next(var_22)
    assert var_23 == 1
    var_24 = next(var_22)
    assert var_24 == 2
    var_25 = next(var_22)



# Parsed testcases at query #17
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = 10
    var_2 = range(var_1)
    var_3 = module_0.chunk(var_0, var_2)
    var_4 = list(var_3)
    var_5 = 1
    var_6 = 20
    var_7 = 30
    var_8 = [var_1, var_6, var_7]
    var_9 = module_0.chunk(var_5, var_8)
    var_10 = list(var_9)
    var_11 = 2
    var_12 = [var_5, var_11, var_0]
    var_13 = module_0.chunk(var_1, var_12)
    var_14 = list(var_13)
    var_15 = []
    var_16 = module_0.chunk(var_0, var_15)
    var_17 = list(var_16)
    var_18 = 'abcde'
    var_19 = module_0.chunk(var_11, var_18)
    var_20 = list(var_19)
    var_21 = 0
    var_22 = 1
    var_23 = 2
    var_24 = 3
    var_25 = [var_22, var_23, var_24]
    var_26 = module_0.chunk(var_21, var_25)
    var_27 = list(var_26)
    var_28 = -1
    var_29 = 1
    var_30 = 2
    var_31 = 3
    var_32 = [var_29, var_30, var_31]
    var_33 = module_0.chunk(var_28, var_32)
    var_34 = list(var_33)



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -5
    var_3 = 5
    var_4 = 15
    var_5 = 2
    var_6 = 11
    var_7 = -1
    var_8 = -2
    var_9 = 1



# Parsed testcases at query #19
#--------------------------


import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = 2
    var_3 = 0
    var_4 = lambda x: x % var_2 == var_3
    var_5 = module_0.split_by(var_1, criterion=var_4)
    var_6 = list(var_5)
    var_7 = 1
    var_8 = 3
    var_9 = 4
    var_10 = 5
    var_11 = 6
    var_12 = [var_7, var_2, var_8, var_9, var_10, var_11]
    var_13 = lambda x: x == var_8
    var_14 = module_0.split_by(var_12, criterion=var_13)
    var_15 = list(var_14)
    var_16 = [var_7, var_2, var_2, var_8]
    var_17 = True
    var_18 = lambda x: x == var_2
    var_19 = module_0.split_by(var_16, var_17, criterion=var_18)
    var_20 = list(var_19)
    var_21 = [var_17, var_3, var_3, var_2]
    var_22 = True
    var_23 = module_0.split_by(var_21, var_22, separator=var_3)
    var_24 = list(var_23)
    var_25 = 'a,b,c'
    var_26 = ','
    var_27 = module_0.split_by(var_25, separator=var_26)
    var_28 = list(var_27)
    var_29 = ',a,,b,'
    var_30 = True
    var_31 = module_0.split_by(var_29, var_30, separator=var_26)
    var_32 = list(var_31)
    var_33 = False
    var_34 = module_0.split_by(var_29, var_33, separator=var_26)
    var_35 = list(var_34)
    var_36 = 1
    var_37 = 2
    var_38 = 3
    var_39 = [var_36, var_37, var_38]
    var_40 = lambda x: x > var_36
    var_41 = module_0.split_by(var_39, criterion=var_40, separator=var_37)
    var_42 = list(var_41)
    var_43 = 1
    var_44 = 2
    var_45 = 3
    var_46 = [var_43, var_44, var_45]
    var_47 = module_0.split_by(var_46)
    var_48 = list(var_47)
    var_49 = []
    var_50 = module_0.split_by(var_49, separator=var_30)
    var_51 = list(var_50)
    var_52 = []
    var_53 = True
    var_54 = lambda x: var_53
    var_55 = module_0.split_by(var_52, criterion=var_54)
    var_56 = list(var_55)
    var_57 = [var_53, var_45, var_8]
    var_58 = module_0.split_by(var_57, separator=var_10)
    var_59 = list(var_58)
    var_60 = [var_53, var_45, var_8]
    var_61 = lambda x: x > var_10
    var_62 = module_0.split_by(var_60, criterion=var_61)
    var_63 = list(var_62)
    var_64 = [var_53, var_53, var_53]
    var_65 = True
    var_66 = module_0.split_by(var_64, var_65, separator=var_53)
    var_67 = list(var_66)
    var_68 = [var_65, var_65, var_65]
    var_69 = False
    var_70 = module_0.split_by(var_68, var_69, separator=var_65)
    var_71 = list(var_70)



# Parsed testcases at query #20
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
    var_7 = 1
    var_8 = 2
    var_9 = [var_7, var_8, var_2]
    var_10 = lambda x: x == var_7
    var_11 = module_0.split_by(var_9, criterion=var_10)
    var_12 = list(var_11)
    var_13 = [var_7, var_8, var_2]
    var_14 = lambda x: x == var_2
    var_15 = module_0.split_by(var_13, criterion=var_14)
    var_16 = list(var_15)
    var_17 = [var_7, var_8, var_2]
    var_18 = True
    var_19 = lambda x: var_18
    var_20 = module_0.split_by(var_17, criterion=var_19)
    var_21 = list(var_20)
    var_22 = [var_18, var_8, var_2]
    var_23 = False
    var_24 = lambda x: var_23
    var_25 = module_0.split_by(var_22, criterion=var_24)
    var_26 = list(var_25)
    var_27 = 'Split.By'
    var_28 = '.'
    var_29 = module_0.split_by(var_27, separator=var_28)
    var_30 = list(var_29)
    var_31 = '.A.'
    var_32 = module_0.split_by(var_31, separator=var_28)
    var_33 = list(var_32)
    var_34 = True
    var_35 = module_0.split_by(var_31, var_34, separator=var_28)
    var_36 = list(var_35)
    var_37 = 'A..B'
    var_38 = module_0.split_by(var_37, separator=var_28)
    var_39 = list(var_38)
    var_40 = True
    var_41 = module_0.split_by(var_37, var_40, separator=var_28)
    var_42 = list(var_41)
    var_43 = [var_40, var_23, var_8, var_23, var_2]
    var_44 = module_0.split_by(var_43, separator=var_23)
    var_45 = list(var_44)
    var_46 = 1
    var_47 = 2
    var_48 = 3
    var_49 = [var_46, var_47, var_48]
    var_50 = lambda x: x > var_46
    var_51 = module_0.split_by(var_49, criterion=var_50, separator=var_46)
    var_52 = list(var_51)
    var_53 = 1
    var_54 = 2
    var_55 = 3
    var_56 = [var_53, var_54, var_55]
    var_57 = module_0.split_by(var_56)
    var_58 = list(var_57)
    var_59 = []
    var_60 = module_0.split_by(var_59, separator=var_28)
    var_61 = list(var_60)
    var_62 = []
    var_63 = True
    var_64 = lambda x: var_63
    var_65 = module_0.split_by(var_62, criterion=var_64)
    var_66 = list(var_65)



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 2
    var_2 = 6
    var_3 = 1
    var_4 = 10
    var_5 = 0
    var_6 = -1



