####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = module_0.PMapValues(var_5)
    var_7 = module_0.PMapValues(var_5)



# Parsed testcases at query #2
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = module_0.PMapItems(var_7)



# Parsed testcases at query #3
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = module_0.PMapView(var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 'x'
    var_9 = 'y'
    var_10 = 10
    var_11 = 20
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = module_0.PMapView(var_12)
    var_14 = var_13._map
    var_15 = module_0.pmap()
    var_16 = var_15.__class__
    var_17 = isinstance(var_14, var_16)
    var_18 = module_0.pmap(var_12)
    var_19 = len(var_13)
    assert var_19 == 2
    var_20 = 1
    var_21 = 2
    var_22 = 3
    var_23 = [var_20, var_21, var_22]
    var_24 = module_0.PMapView(var_23)
    var_25 = len(var_6)
    var_26 = len(var_5)
    var_27 = reversed(var_6)



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = 'x'
    var_9 = 'y'
    var_10 = 'z'
    var_11 = 'hello'
    var_12 = None
    var_13 = True
    var_14 = {var_8: var_11, var_9: var_12, var_10: var_13}



# Parsed testcases at query #5
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = module_0.PMapValues(var_5)
    var_7 = module_0.PMapValues(var_5)
    var_8 = {var_0: var_2, var_1: var_3}
    var_9 = module_0.pmap(var_8)
    var_10 = module_0.PMapValues(var_9)



# Parsed testcases at query #6
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = module_0.PMapItems(var_7)



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = 3
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = 'c'
    var_9 = {var_0: var_2, var_1: var_3, var_8: var_6}
    var_10 = {var_0: var_2, var_1: var_3}
    var_11 = {var_1: var_3, var_0: var_2}
    var_12 = {var_0: var_2, var_1: var_6}
    var_13 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #8
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = module_0.PMapItems(var_7)



# Parsed testcases at query #9
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = module_0.PMapItems(var_7)



# Parsed testcases at query #10
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = module_0.PMapItems(var_7)
    var_9 = {}
    var_10 = module_0.pmap(var_9)
    var_11 = module_0.PMapItems(var_10)



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = 3
    var_7 = {var_0: var_3, var_5: var_6}
    var_8 = 'd'
    var_9 = 17
    var_10 = 35
    var_11 = {var_0: var_9, var_8: var_10}
    var_12 = lambda l, r: r
    var_13 = {var_0: var_9, var_1: var_3, var_5: var_6, var_8: var_10}
    var_14 = {var_0: var_6, var_1: var_3, var_5: var_6}
    var_15 = lambda l, r: l
    var_16 = {var_0: var_2, var_1: var_3, var_5: var_6, var_8: var_10}
    var_17 = {}
    var_18 = 'z'
    var_19 = 100
    var_20 = {var_18: var_19}



# Parsed testcases at query #12
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = module_0.PMapItems(var_7)
    var_9 = {}
    var_10 = module_0.pmap(var_9)
    var_11 = module_0.PMapItems(var_10)



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = 3
    var_8 = {var_0: var_2, var_1: var_7}
    var_9 = {var_0: var_2}
    var_10 = {var_0: var_2, var_1: var_3}
    var_11 = {var_0: var_2, var_1: var_3}
    var_12 = 'c'
    var_13 = {var_12: var_2, var_1: var_3}



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'a'
    var_1 = 'key'
    var_2 = 1
    var_3 = 'val'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = {var_0: var_2, var_1: var_3}
    var_8 = {var_0: var_2}
    var_9 = 2
    var_10 = {var_0: var_9, var_1: var_3}
    var_11 = 'b'
    var_12 = {var_11: var_2, var_1: var_3}
    var_13 = 20
    var_14 = None
    var_15 = [var_14]
    var_16 = var_15 * var_13
    var_17 = var_0 % var_13
    var_18 = var_16[var_17]



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = {var_0: var_2}
    var_8 = 99
    var_9 = {var_0: var_8, var_1: var_3}
    var_10 = {var_1: var_3, var_0: var_2}
    var_11 = 'c'
    var_12 = {var_0: var_2, var_11: var_3}



# Parsed testcases at query #16
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = module_0.PMapItems(var_7)



# Parsed testcases at query #17
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = module_0.PMapItems(var_7)



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_0.pmap(var_6)
    var_8 = 'c'
    var_9 = 3
    var_10 = {var_0: var_2, var_8: var_9}
    var_11 = module_0.pmap(var_10)
    var_12 = module_0.PMapItems(var_5)
    var_13 = module_0.PMapItems(var_7)
    var_14 = module_0.PMapItems(var_11)



# Parsed testcases at query #2
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = module_0.PMapItems(var_7)



# Parsed testcases at query #3
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_0.pmap(var_6)
    var_8 = 'c'
    var_9 = 3
    var_10 = {var_0: var_2, var_8: var_9}
    var_11 = module_0.pmap(var_10)
    var_12 = {var_1: var_3, var_0: var_2}
    var_13 = module_0.pmap(var_12)
    var_14 = module_0.PMapItems(var_5)
    var_15 = module_0.PMapItems(var_7)
    var_16 = module_0.PMapItems(var_11)
    var_17 = module_0.PMapItems(var_13)
    var_18 = module_0.PMapValues(var_5)
    var_19 = module_0.pmap()
    var_20 = module_0.PMapItems(var_19)
    var_21 = module_0.pmap()
    var_22 = module_0.PMapItems(var_21)



# Parsed testcases at query #4
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = module_0.PMapItems(var_7)



# Parsed testcases at query #5
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 3
    var_4 = module_0.m()
    var_5 = 'a'
    var_6 = 'd'
    var_7 = 17
    var_8 = 35
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = lambda l, r: r
    var_11 = 'b'
    var_12 = 'c'
    var_13 = {var_5: var_7, var_11: var_1, var_12: var_3, var_6: var_8}
    var_14 = module_0.m()
    var_15 = module_0.m()
    var_16 = module_0.m()
    var_17 = module_0.m()
    var_18 = {var_5: var_3}
    var_19 = lambda l, r: l
    var_20 = 10
    var_21 = module_0.m()
    var_22 = 20
    var_23 = module_0.m()
    var_24 = module_0.m()
    var_25 = module_0.m()
    var_26 = module_0.m()
    var_27 = 5
    var_28 = module_0.m()



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'a'
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 'c'
    var_4 = 3
    var_5 = (var_3, var_4)
    var_6 = 17
    var_7 = (var_0, var_6)
    var_8 = 'd'
    var_9 = 35
    var_10 = (var_8, var_9)
    var_11 = 'b'
    var_12 = 1
    var_13 = {var_0: var_12, var_11: var_1}
    var_14 = {var_0: var_12, var_11: var_1}
    var_15 = (var_0, var_1)
    var_16 = [var_15]
    var_17 = {var_0: var_12}
    var_18 = (var_0, var_4)
    var_19 = lambda l, r: l
    var_20 = {var_0: var_12}
    var_21 = 10
    var_22 = (var_0, var_21)
    var_23 = 20
    var_24 = (var_11, var_23)



# Parsed testcases at query #7
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = module_0.PMapItems(var_7)
    var_9 = 'd'
    var_10 = None
    var_11 = {var_9: var_10}
    var_12 = module_0.pmap(var_11)
    var_13 = module_0.PMapItems(var_12)



# Parsed testcases at query #8
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = module_0.PMapItems(var_7)
    var_9 = {}
    var_10 = module_0.pmap(var_9)
    var_11 = module_0.PMapItems(var_10)



# Parsed testcases at query #9
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_0.pmap(var_6)
    var_8 = 'c'
    var_9 = 3
    var_10 = {var_0: var_2, var_8: var_9}
    var_11 = module_0.pmap(var_10)
    var_12 = module_0.PMapItems(var_5)
    var_13 = module_0.PMapItems(var_7)
    var_14 = module_0.PMapItems(var_11)



# Parsed testcases at query #10
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = module_0.PMapItems(var_7)



# Parsed testcases at query #11
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = module_0.PMapItems(var_7)



# Parsed testcases at query #12
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_0.pmap(var_6)
    var_8 = 'c'
    var_9 = 3
    var_10 = {var_0: var_2, var_8: var_9}
    var_11 = module_0.pmap(var_10)
    var_12 = module_0.PMapItems(var_5)
    var_13 = module_0.PMapItems(var_7)
    var_14 = module_0.PMapItems(var_11)



# Parsed testcases at query #13
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_0.pmap(var_6)
    var_8 = 'c'
    var_9 = 3
    var_10 = {var_0: var_2, var_8: var_9}
    var_11 = module_0.pmap(var_10)
    var_12 = module_0.PMapItems(var_5)
    var_13 = module_0.PMapItems(var_7)
    var_14 = module_0.PMapItems(var_11)



# Parsed testcases at query #14
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: r
    var_4 = 3
    var_5 = module_0.m()
    var_6 = 'a'
    var_7 = 'd'
    var_8 = 17
    var_9 = 35
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = lambda l, r: l + r
    var_12 = module_0.m()
    var_13 = lambda l, r: l
    var_14 = module_0.m()
    var_15 = {var_6: var_4}
    var_16 = lambda l, r: r
    var_17 = 10
    var_18 = module_0.m()
    var_19 = 'c'
    var_20 = 20
    var_21 = 5
    var_22 = {var_19: var_20, var_6: var_21}
    var_23 = lambda l, r: r
    var_24 = module_0.m()
    var_25 = 100
    var_26 = module_0.m()
    var_27 = lambda l, r: r
    var_28 = 200
    var_29 = module_0.m()
    var_30 = lambda l, r: l + r
    var_31 = 50
    var_32 = module_0.m()



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = 3
    var_7 = {var_0: var_3, var_5: var_6}
    var_8 = 'd'
    var_9 = 17
    var_10 = 35
    var_11 = {var_0: var_9, var_8: var_10}
    var_12 = {var_0: var_2}
    var_13 = {var_0: var_3}
    var_14 = {var_0: var_6}
    var_15 = lambda l, r: l
    var_16 = 10
    var_17 = {var_0: var_16}
    var_18 = {var_0: var_3}
    var_19 = lambda l, r: l * r
    var_20 = {var_1: var_3}
    var_21 = 5
    var_22 = {var_0: var_21}
    var_23 = {}



