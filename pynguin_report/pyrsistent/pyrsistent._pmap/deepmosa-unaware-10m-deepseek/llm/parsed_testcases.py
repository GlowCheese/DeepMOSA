####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
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
    var_15 = len(var_13)
    assert var_15 == 2
    var_16 = 'key1'
    var_17 = 'value1'
    var_18 = (var_16, var_17)
    var_19 = 'key2'
    var_20 = 'value2'
    var_21 = (var_19, var_20)
    var_22 = [var_18, var_21]
    var_23 = var_13._map
    var_24 = len(var_13)
    assert var_24 == 2
    var_25 = {}
    var_26 = module_0.pmap(var_25)
    var_27 = module_0.PMapView(var_26)
    var_28 = len(var_27)
    assert var_28 == 0
    var_29 = {var_0: var_2}
    var_30 = module_0.pmap(var_29)
    var_31 = module_0.PMapView(var_30)
    var_32 = 'b'
    var_33 = 2
    var_34 = {var_32: var_33}
    var_35 = 1
    var_36 = 2
    var_37 = 3
    var_38 = [var_35, var_36, var_37]
    var_39 = module_0.PMapView(var_38)
    var_40 = 'not a mapping'
    var_41 = module_0.PMapView(var_40)
    var_42 = 123
    var_43 = module_0.PMapView(var_42)
    var_44 = {var_42: var_37, var_43: var_38}
    var_45 = module_0.pmap(var_44)
    var_46 = module_0.PMapView(var_45)
    var_47 = reversed(var_46)



# Parsed testcases at query #2
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
    var_15 = module_0.pmap(var_12)
    var_16 = len(var_13)
    assert var_16 == 2
    var_17 = 'p'
    var_18 = 'q'
    var_19 = 100
    var_20 = 200
    var_21 = {var_17: var_19, var_18: var_20}
    var_22 = var_13._map
    var_23 = {var_17: var_19, var_18: var_20}
    var_24 = module_0.pmap(var_23)
    var_25 = len(var_13)
    assert var_25 == 2
    var_26 = {var_0: var_2}
    var_27 = module_0.pmap(var_26)
    var_28 = module_0.PMapView(var_27)
    var_29 = 'b'
    var_30 = 2
    var_31 = {var_29: var_30}
    var_32 = 1
    var_33 = 2
    var_34 = 3
    var_35 = [var_32, var_33, var_34]
    var_36 = module_0.PMapView(var_35)
    var_37 = 'not a mapping'
    var_38 = module_0.PMapView(var_37)
    var_39 = 42
    var_40 = module_0.PMapView(var_39)
    var_41 = {var_39: var_34}
    var_42 = module_0.pmap(var_41)
    var_43 = module_0.PMapView(var_42)
    var_44 = {var_39: var_34, var_40: var_35}
    var_45 = module_0.pmap(var_44)
    var_46 = module_0.PMapView(var_45)
    var_47 = reversed(var_46)



# Parsed testcases at query #3
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = module_0.PMapItems(var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = module_0.pmap(var_9)
    var_11 = module_0.PMapItems(var_10)
    var_12 = 'int'
    var_13 = 'str'
    var_14 = 'list'
    var_15 = 'none'
    var_16 = 'dict'
    var_17 = 42
    var_18 = 'hello'
    var_19 = [var_6, var_7, var_8]
    var_20 = None
    var_21 = 'nested'
    var_22 = 'value'
    var_23 = {var_21: var_22}
    var_24 = {var_12: var_17, var_13: var_18, var_14: var_19, var_15: var_20, var_16: var_23}
    var_25 = module_0.pmap(var_24)
    var_26 = module_0.PMapItems(var_25)



# Parsed testcases at query #4
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = module_0.PMapItems(var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 'c'
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = module_0.pmap(var_9)
    var_11 = module_0.PMapItems(var_10)
    var_12 = None
    var_13 = 0
    var_14 = {var_3: var_12, var_4: var_13}
    var_15 = module_0.pmap(var_14)
    var_16 = module_0.PMapItems(var_15)
    var_17 = {var_4: var_5}
    var_18 = module_0.pmap(var_17)
    var_19 = {var_3: var_18}
    var_20 = module_0.pmap(var_19)
    var_21 = module_0.PMapItems(var_20)
    var_22 = {var_4: var_5}
    var_23 = module_0.pmap(var_22)
    var_24 = (var_3, var_23)
    var_25 = 4
    var_26 = {var_4: var_25}
    var_27 = module_0.pmap(var_26)
    var_28 = (var_3, var_27)
    var_29 = {var_6: var_3}
    var_30 = {var_7: var_4}
    var_31 = {var_3: var_29, var_4: var_30}
    var_32 = module_0.pmap(var_31)
    var_33 = module_0.PMapItems(var_32)



# Parsed testcases at query #5
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = module_0.PMapItems(var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = module_0.pmap(var_9)
    var_11 = module_0.PMapItems(var_10)
    var_12 = 'x'
    var_13 = 'y'
    var_14 = 5
    var_15 = {var_13: var_14}
    var_16 = module_0.pmap(var_15)
    var_17 = {var_12: var_16}
    var_18 = module_0.pmap(var_17)
    var_19 = module_0.PMapItems(var_18)
    var_20 = {var_13: var_14}
    var_21 = module_0.pmap(var_20)
    var_22 = (var_12, var_21)
    var_23 = 'key'



# Parsed testcases at query #6
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
    var_8 = 3
    var_9 = {var_0: var_2, var_1: var_8}
    var_10 = module_0.pmap(var_9)
    var_11 = 'c'
    var_12 = {var_0: var_2, var_11: var_3}
    var_13 = module_0.pmap(var_12)
    var_14 = {var_0: var_2, var_1: var_3}
    var_15 = {var_0: var_2, var_1: var_8}
    var_16 = {var_0: var_2, var_1: var_3, var_11: var_8}
    var_17 = {var_0: var_2, var_1: var_3}
    var_18 = 123
    var_19 = 100
    var_20 = range(var_19)
    var_21 = {i: i for i in var_20}
    var_22 = module_0.pmap(var_21)
    var_23 = range(var_19)
    var_24 = {i: i for i in var_23}
    var_25 = module_0.pmap(var_24)
    var_26 = 'x'
    var_27 = 'y'
    var_28 = {var_26: var_2, var_27: var_3}
    var_29 = module_0.pmap(var_28)
    var_30 = {var_26: var_2, var_27: var_3}
    var_31 = module_0.pmap(var_30)
    var_32 = hash(var_29)
    var_33 = hash(var_31)
    var_34 = 'z'
    var_35 = module_0.m()
    var_36 = {}
    var_37 = module_0.pmap(var_36)
    var_38 = {}
    var_39 = module_0.pmap(var_38)
    var_40 = {}



# Parsed testcases at query #7
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'c'
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = module_0.pmap(var_8)
    var_10 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_11 = module_0.pmap(var_10)
    var_12 = 4
    var_13 = 'd'



# Parsed testcases at query #8
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
    var_8 = 3
    var_9 = {var_0: var_2, var_1: var_8}
    var_10 = module_0.pmap(var_9)
    var_11 = 'c'
    var_12 = {var_0: var_2, var_11: var_3}
    var_13 = module_0.pmap(var_12)
    var_14 = {var_0: var_2, var_1: var_3}
    var_15 = {var_0: var_2, var_1: var_8}
    var_16 = {var_0: var_2, var_1: var_3, var_11: var_8}
    var_17 = {}
    var_18 = module_0.pmap(var_17)
    var_19 = {}
    var_20 = module_0.pmap(var_19)
    var_21 = {var_0: var_2, var_1: var_3}
    var_22 = 100
    var_23 = range(var_22)
    var_24 = {i: i for i in var_23}
    var_25 = module_0.pmap(var_24)
    var_26 = range(var_22)
    var_27 = {i: i for i in var_26}
    var_28 = module_0.pmap(var_27)
    var_29 = {var_0: var_2, var_1: var_3}
    var_30 = module_0.pmap(var_29)
    var_31 = {var_0: var_2, var_1: var_3}
    var_32 = module_0.pmap(var_31)
    var_33 = hash(var_30)
    var_34 = hash(var_32)
    var_35 = {var_0: var_2, var_1: var_3}
    var_36 = module_0.pmap(var_35)
    var_37 = {var_0: var_2, var_1: var_8}
    var_38 = module_0.pmap(var_37)
    var_39 = hash(var_36)
    var_40 = hash(var_38)
    var_41 = {var_0: var_2, var_1: var_3}
    var_42 = module_0.pmap(var_41)
    var_43 = 'x'
    var_44 = 'y'
    var_45 = 10
    var_46 = 20
    var_47 = {var_43: var_45, var_44: var_46}
    var_48 = module_0.pmap(var_47)
    var_49 = {var_43: var_45, var_44: var_46}
    var_50 = module_0.pmap(var_49)



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
    var_6 = {var_0: var_3}
    var_7 = module_0.pmap(var_6)
    var_8 = {var_0: var_2, var_1: var_3}
    var_9 = module_0.pmap(var_8)
    var_10 = {var_0: var_3}
    var_11 = module_0.pmap(var_10)
    var_12 = 'c'
    var_13 = 3
    var_14 = {var_0: var_13, var_12: var_2}
    var_15 = module_0.pmap(var_14)
    var_16 = {var_0: var_2, var_1: var_3}
    var_17 = module_0.pmap(var_16)
    var_18 = {}
    var_19 = module_0.pmap(var_18)
    var_20 = {}
    var_21 = module_0.pmap(var_20)
    var_22 = {var_0: var_2}
    var_23 = module_0.pmap(var_22)
    var_24 = lambda l, r: l
    var_25 = {var_0: var_3}
    var_26 = module_0.pmap(var_25)
    var_27 = {var_0: var_13}
    var_28 = module_0.pmap(var_27)
    var_29 = {var_0: var_2}
    var_30 = module_0.pmap(var_29)
    var_31 = lambda l, r: r
    var_32 = {var_0: var_3}
    var_33 = module_0.pmap(var_32)
    var_34 = {var_0: var_13}
    var_35 = module_0.pmap(var_34)
    var_36 = {var_0: var_2}
    var_37 = module_0.pmap(var_36)
    var_38 = {var_1: var_3}
    var_39 = module_0.pmap(var_38)
    var_40 = {var_12: var_13}
    var_41 = module_0.pmap(var_40)
    var_42 = 10
    var_43 = 5
    var_44 = {var_0: var_42, var_1: var_43}
    var_45 = module_0.pmap(var_44)
    var_46 = {var_0: var_13, var_1: var_3}
    var_47 = module_0.pmap(var_46)
    var_48 = [var_2]
    var_49 = {var_0: var_48, var_1: var_3}
    var_50 = module_0.pmap(var_49)
    var_51 = lambda l, r: l + r
    var_52 = [var_3, var_13]
    var_53 = {var_0: var_52}
    var_54 = module_0.pmap(var_53)
    var_55 = {var_0: var_2, var_1: var_3}
    var_56 = module_0.pmap(var_55)
    var_57 = {var_0: var_3, var_12: var_13}
    var_58 = {var_0: var_2, var_1: var_3}
    var_59 = module_0.pmap(var_58)
    var_60 = hash(var_59)
    var_61 = {var_0: var_3}
    var_62 = module_0.pmap(var_61)
    var_63 = hash(var_59)
    var_64 = {var_0: var_2, var_1: var_3}
    var_65 = module_0.pmap(var_64)
    var_66 = 'x'
    var_67 = {var_66: var_2}
    var_68 = module_0.pmap(var_67)
    var_69 = {var_0: var_68, var_1: var_3}
    var_70 = module_0.pmap(var_69)
    var_71 = 'y'
    var_72 = {var_66: var_3, var_71: var_13}
    var_73 = module_0.pmap(var_72)
    var_74 = {var_0: var_73}
    var_75 = module_0.pmap(var_74)



# Parsed testcases at query #10
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = module_0.PMapItems(var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 'c'
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = module_0.pmap(var_9)
    var_11 = module_0.PMapItems(var_10)
    var_12 = 'key'
    var_13 = 42
    var_14 = [var_3, var_4, var_5]
    var_15 = 'nested'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = {var_12: var_14, var_13: var_17}
    var_19 = module_0.pmap(var_18)
    var_20 = module_0.PMapItems(var_19)
    var_21 = {var_3: var_6, var_4: var_7}
    var_22 = module_0.pmap(var_21)
    var_23 = module_0.PMapItems(var_22)



# Parsed testcases at query #11
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = module_0.PMapItems(var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = module_0.pmap(var_9)
    var_11 = module_0.PMapItems(var_10)
    var_12 = 'x'
    var_13 = 'y'
    var_14 = [var_6, var_7]
    var_15 = 'z'
    var_16 = {var_15: var_8}
    var_17 = {var_12: var_14, var_13: var_16}
    var_18 = module_0.pmap(var_17)
    var_19 = module_0.PMapItems(var_18)
    var_20 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_21 = module_0.pmap(var_20)
    var_22 = module_0.PMapItems(var_21)
    var_23 = module_0.PMapView(var_10)
    var_24 = var_23._map
    var_25 = module_0.PMapItems(var_24)



# Parsed testcases at query #12
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = module_0.PMapItems(var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = module_0.pmap(var_9)
    var_11 = module_0.PMapItems(var_10)
    var_12 = 'x'
    var_13 = 'y'
    var_14 = None
    var_15 = 0
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = module_0.pmap(var_16)
    var_18 = module_0.PMapItems(var_17)
    var_19 = 'outer'
    var_20 = 'inner'
    var_21 = 'value'
    var_22 = {var_20: var_21}
    var_23 = module_0.pmap(var_22)
    var_24 = {var_19: var_23}
    var_25 = module_0.pmap(var_24)
    var_26 = module_0.PMapItems(var_25)
    var_27 = {var_20: var_21}
    var_28 = module_0.pmap(var_27)
    var_29 = (var_19, var_28)
    var_30 = 'wrong'
    var_31 = {var_20: var_30}
    var_32 = module_0.pmap(var_31)
    var_33 = (var_19, var_32)
    var_34 = 'key'
    var_35 = {var_34: var_21}
    var_36 = module_0.pmap(var_35)
    var_37 = module_0.PMapItems(var_36)



# Parsed testcases at query #13
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = module_0.PMapItems(var_1)
    var_3 = 1
    var_4 = 'a'
    var_5 = {var_3: var_4}
    var_6 = module_0.pmap(var_5)
    var_7 = module_0.PMapItems(var_6)
    var_8 = 2
    var_9 = 3
    var_10 = 'b'
    var_11 = 'c'
    var_12 = {var_3: var_4, var_8: var_10, var_9: var_11}
    var_13 = module_0.pmap(var_12)
    var_14 = module_0.PMapItems(var_13)
    var_15 = {var_3: var_4}
    var_16 = module_0.pmap(var_15)
    var_17 = module_0.PMapItems(var_16)
    var_18 = {var_3: var_4}
    var_19 = module_0.pmap(var_18)
    var_20 = module_0.PMapItems(var_19)
    var_21 = {var_3: var_4}
    var_22 = module_0.pmap(var_21)
    var_23 = module_0.PMapItems(var_22)
    var_24 = {var_3: var_4, var_8: var_10}
    var_25 = module_0.pmap(var_24)
    var_26 = {var_3: var_4, var_8: var_10}
    var_27 = module_0.pmap(var_26)
    var_28 = module_0.PMapItems(var_25)
    var_29 = module_0.PMapItems(var_27)



# Parsed testcases at query #14
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = module_0.PMapItems(var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = module_0.pmap(var_9)
    var_11 = module_0.PMapItems(var_10)
    var_12 = 'x'
    var_13 = 'y'
    var_14 = 'z'
    var_15 = [var_6, var_7]
    var_16 = 'nested'
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = None
    var_20 = {var_12: var_15, var_13: var_18, var_14: var_19}
    var_21 = module_0.pmap(var_20)
    var_22 = module_0.PMapItems(var_21)
    var_23 = 'key'
    var_24 = {var_23: var_17}
    var_25 = module_0.pmap(var_24)
    var_26 = module_0.PMapItems(var_25)
    var_27 = 'outer'
    var_28 = 'inner'
    var_29 = 42
    var_30 = {var_28: var_29}
    var_31 = module_0.pmap(var_30)
    var_32 = {var_27: var_31}
    var_33 = module_0.pmap(var_32)
    var_34 = module_0.PMapItems(var_33)
    var_35 = {var_28: var_29}
    var_36 = module_0.pmap(var_35)
    var_37 = (var_27, var_36)
    var_38 = len(var_10)
    var_39 = (var_3, var_6)
    var_40 = var_39 in var_11
    var_41 = 99
    var_42 = (var_12, var_41)
    var_43 = var_42 in var_11
    var_44 = len(var_10)



# Parsed testcases at query #15
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_3}
    var_7 = module_0.pmap(var_6)
    var_8 = {var_0: var_2, var_1: var_3}
    var_9 = module_0.pmap(var_8)
    var_10 = {var_0: var_3}
    var_11 = module_0.pmap(var_10)
    var_12 = 'c'
    var_13 = 3
    var_14 = {var_0: var_13, var_12: var_2}
    var_15 = {var_0: var_2}
    var_16 = module_0.pmap(var_15)
    var_17 = lambda l, r: l
    var_18 = {var_0: var_3}
    var_19 = module_0.pmap(var_18)
    var_20 = {var_0: var_13}
    var_21 = {var_0: var_2}
    var_22 = module_0.pmap(var_21)
    var_23 = lambda l, r: r
    var_24 = {var_0: var_3}
    var_25 = module_0.pmap(var_24)
    var_26 = {var_0: var_13}
    var_27 = {var_0: var_2}
    var_28 = module_0.pmap(var_27)
    var_29 = {var_1: var_3}
    var_30 = module_0.pmap(var_29)
    var_31 = {var_12: var_13}
    var_32 = {var_0: var_2, var_1: var_3}
    var_33 = module_0.pmap(var_32)
    var_34 = {}
    var_35 = {}
    var_36 = module_0.pmap(var_35)
    var_37 = [var_2]
    var_38 = [var_3]
    var_39 = {var_0: var_37, var_1: var_38}
    var_40 = module_0.pmap(var_39)
    var_41 = lambda l, r: l + r
    var_42 = [var_13]
    var_43 = 4
    var_44 = [var_43]
    var_45 = {var_0: var_42, var_12: var_44}
    var_46 = module_0.pmap(var_45)
    var_47 = {var_0: var_2, var_1: var_3}
    var_48 = module_0.pmap(var_47)
    var_49 = hash(var_48)
    var_50 = {var_0: var_3}
    var_51 = module_0.pmap(var_50)
    var_52 = hash(var_48)
    var_53 = {var_0: var_2, var_1: var_3}
    var_54 = module_0.pmap(var_53)
    var_55 = {var_0: var_3}
    var_56 = {var_0: var_13}
    var_57 = module_0.pmap(var_56)
    var_58 = {var_0: var_2, var_1: var_3}
    var_59 = module_0.pmap(var_58)
    var_60 = lambda l, r: str(l) + str(r)
    var_61 = {var_0: var_3}
    var_62 = module_0.pmap(var_61)



# Parsed testcases at query #16
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = module_0.PMapItems(var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 'c'
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = module_0.pmap(var_9)
    var_11 = module_0.PMapItems(var_10)
    var_12 = 'key'
    var_13 = 42
    var_14 = [var_3, var_4, var_5]
    var_15 = 'nested'
    var_16 = 'dict'
    var_17 = {var_15: var_16}
    var_18 = {var_12: var_14, var_13: var_17}
    var_19 = module_0.pmap(var_18)
    var_20 = module_0.PMapItems(var_19)
    var_21 = [var_3, var_4, var_5]
    var_22 = None
    var_23 = {var_3: var_22, var_4: var_22}
    var_24 = module_0.pmap(var_23)
    var_25 = module_0.PMapItems(var_24)



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
    var_9 = {}
    var_10 = module_0.pmap(var_9)
    var_11 = module_0.PMapItems(var_10)
    var_12 = 'x'
    var_13 = 'y'
    var_14 = [var_3, var_4]
    var_15 = 'z'
    var_16 = {var_15: var_5}
    var_17 = {var_12: var_14, var_13: var_16}
    var_18 = module_0.pmap(var_17)
    var_19 = module_0.PMapItems(var_18)
    var_20 = [var_3, var_4]
    var_21 = 'key'
    var_22 = {var_21: var_20}
    var_23 = module_0.pmap(var_22)
    var_24 = module_0.PMapItems(var_23)



# Parsed testcases at query #18
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_3}
    var_7 = module_0.pmap(var_6)
    var_8 = {var_0: var_2, var_1: var_3}
    var_9 = module_0.pmap(var_8)
    var_10 = {var_0: var_3}
    var_11 = module_0.pmap(var_10)
    var_12 = 'c'
    var_13 = 3
    var_14 = {var_0: var_13, var_12: var_2}
    var_15 = module_0.pmap(var_14)
    var_16 = {var_0: var_2}
    var_17 = module_0.pmap(var_16)
    var_18 = lambda l, r: l
    var_19 = {var_0: var_3}
    var_20 = module_0.pmap(var_19)
    var_21 = {var_0: var_13}
    var_22 = {var_0: var_2}
    var_23 = module_0.pmap(var_22)
    var_24 = lambda l, r: r
    var_25 = {var_0: var_3}
    var_26 = module_0.pmap(var_25)
    var_27 = {var_0: var_13}
    var_28 = module_0.pmap()
    var_29 = {var_0: var_2, var_1: var_3}
    var_30 = module_0.pmap(var_29)
    var_31 = {var_0: var_2, var_1: var_3}
    var_32 = module_0.pmap(var_31)
    var_33 = {var_0: var_2}
    var_34 = module_0.pmap(var_33)
    var_35 = lambda l, r: l + r
    var_36 = {var_1: var_3}
    var_37 = module_0.pmap(var_36)
    var_38 = [var_2]
    var_39 = {var_0: var_38, var_1: var_3}
    var_40 = module_0.pmap(var_39)
    var_41 = lambda l, r: l + r
    var_42 = [var_3, var_13]
    var_43 = {var_0: var_42}
    var_44 = module_0.pmap(var_43)
    var_45 = {var_0: var_2, var_1: var_3}
    var_46 = module_0.pmap(var_45)
    var_47 = {var_0: var_3}
    var_48 = module_0.pmap(var_47)
    var_49 = {var_0: var_2}
    var_50 = module_0.pmap(var_49)
    var_51 = {var_0: var_3}
    var_52 = {var_0: var_13}
    var_53 = module_0.pmap(var_52)
    var_54 = {var_0: var_2}
    var_55 = module_0.pmap(var_54)
    var_56 = None
    var_57 = lambda l, r: var_56
    var_58 = {var_0: var_3}
    var_59 = module_0.pmap(var_58)



# Parsed testcases at query #19
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = module_0.PMapItems(var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = module_0.pmap(var_9)
    var_11 = module_0.PMapItems(var_10)
    var_12 = 'x'
    var_13 = 'y'
    var_14 = 'nested'
    var_15 = 'value'
    var_16 = {var_14: var_15}
    var_17 = [var_6, var_7, var_8]
    var_18 = {var_12: var_16, var_13: var_17}
    var_19 = module_0.pmap(var_18)
    var_20 = module_0.PMapItems(var_19)
    var_21 = {var_3: var_6, var_4: var_7}
    var_22 = module_0.pmap(var_21)
    var_23 = module_0.PMapItems(var_22)



# Parsed testcases at query #20
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()
    var_4 = module_0.m()
    var_5 = module_0.m()
    var_6 = 'a'
    var_7 = 'c'
    var_8 = 3
    var_9 = 5
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = module_0.m()
    var_12 = lambda l, r: l
    var_13 = module_0.m()
    var_14 = {var_6: var_8}
    var_15 = module_0.m()
    var_16 = lambda l, r: r
    var_17 = module_0.m()
    var_18 = {var_6: var_8}
    var_19 = module_0.m()
    var_20 = 4
    var_21 = module_0.m()
    var_22 = module_0.m()
    var_23 = {}
    var_24 = module_0.m()
    var_25 = module_0.m()
    var_26 = module_0.m()
    var_27 = {var_7: var_8}
    var_28 = 10
    var_29 = 20
    var_30 = module_0.m()
    var_31 = 30
    var_32 = 40
    var_33 = module_0.m()
    var_34 = module_0.m()
    var_35 = hash(var_34)
    var_36 = module_0.m()
    var_37 = hash(var_34)
    var_38 = module_0.m()
    var_39 = {var_6: var_1}
    var_40 = module_0.m()
    var_41 = None
    var_42 = module_0.m()
    var_43 = module_0.m()
    var_44 = []
    var_45 = module_0.m()
    var_46 = module_0.m()



# Parsed testcases at query #21
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = module_0.PMapItems(var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = module_0.pmap(var_9)
    var_11 = module_0.PMapItems(var_10)
    var_12 = 'x'
    var_13 = 'y'
    var_14 = None
    var_15 = 0
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = module_0.pmap(var_16)
    var_18 = module_0.PMapItems(var_17)
    var_19 = 'outer'
    var_20 = 'inner'
    var_21 = 'value'
    var_22 = {var_20: var_21}
    var_23 = module_0.pmap(var_22)
    var_24 = {var_19: var_23}
    var_25 = module_0.pmap(var_24)
    var_26 = module_0.PMapItems(var_25)
    var_27 = {var_20: var_21}
    var_28 = module_0.pmap(var_27)
    var_29 = (var_19, var_28)
    var_30 = 'wrong'
    var_31 = {var_20: var_30}
    var_32 = module_0.pmap(var_31)
    var_33 = (var_19, var_32)



# Parsed testcases at query #22
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = module_0.PMapItems(var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = module_0.pmap(var_9)
    var_11 = module_0.PMapItems(var_10)
    var_12 = 'x'
    var_13 = 'y'
    var_14 = [var_6, var_7]
    var_15 = 'z'
    var_16 = {var_15: var_8}
    var_17 = {var_12: var_14, var_13: var_16}
    var_18 = module_0.pmap(var_17)
    var_19 = module_0.PMapItems(var_18)
    var_20 = module_0.PMapView(var_10)
    var_21 = var_20._map
    var_22 = module_0.PMapItems(var_21)



# Parsed testcases at query #23
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = module_0.PMapItems(var_1)
    var_3 = 1
    var_4 = 'a'
    var_5 = {var_3: var_4}
    var_6 = module_0.pmap(var_5)
    var_7 = module_0.PMapItems(var_6)
    var_8 = 2
    var_9 = 3
    var_10 = 'b'
    var_11 = 'c'
    var_12 = {var_3: var_4, var_8: var_10, var_9: var_11}
    var_13 = module_0.pmap(var_12)
    var_14 = module_0.PMapItems(var_13)
    var_15 = {var_3: var_4}
    var_16 = module_0.pmap(var_15)
    var_17 = module_0.PMapItems(var_16)
    var_18 = {var_3: var_4}
    var_19 = module_0.pmap(var_18)
    var_20 = module_0.PMapItems(var_19)
    var_21 = {var_3: var_4, var_8: var_10}
    var_22 = module_0.pmap(var_21)
    var_23 = module_0.PMapItems(var_22)
    var_24 = [var_3, var_8]
    var_25 = 'nested'
    var_26 = 'dict'
    var_27 = {var_25: var_26}
    var_28 = {var_4: var_3, var_10: var_24, var_11: var_27}
    var_29 = module_0.pmap(var_28)
    var_30 = module_0.PMapItems(var_29)
    var_31 = {var_3: var_4}
    var_32 = module_0.pmap(var_31)
    var_33 = module_0.PMapItems(var_32)



# Parsed testcases at query #24
#--------------------------


import pyrsistent._pmap as module_0
import pyrsistent._transformations as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_0.pmap(var_6)
    var_8 = 3
    var_9 = {var_0: var_2, var_1: var_8}
    var_10 = module_0.pmap(var_9)
    var_11 = {var_0: var_2, var_1: var_3}
    var_12 = {var_0: var_2, var_1: var_8}
    var_13 = 'c'
    var_14 = {var_0: var_2, var_1: var_3, var_13: var_8}
    var_15 = {}
    var_16 = module_0.pmap(var_15)
    var_17 = {}
    var_18 = module_0.pmap(var_17)
    var_19 = {var_1: var_2}
    var_20 = module_0.pmap(var_19)
    var_21 = {var_0: var_20}
    var_22 = module_0.pmap(var_21)
    var_23 = {var_1: var_2}
    var_24 = module_0.pmap(var_23)
    var_25 = {var_0: var_24}
    var_26 = module_0.pmap(var_25)
    var_27 = 123
    var_28 = {var_0: var_2, var_1: var_3}
    var_29 = 'd'
    var_30 = 'e'
    var_31 = 'f'
    var_32 = 4
    var_33 = 5
    var_34 = 6
    var_35 = {var_0: var_2, var_1: var_3, var_13: var_8, var_29: var_32, var_30: var_33, var_31: var_34}
    var_36 = module_0.pmap(var_35)
    var_37 = module_1.discard(var_31)
    var_38 = 'x'
    var_39 = 'y'
    var_40 = 10
    var_41 = 20
    var_42 = {var_38: var_40, var_39: var_41}
    var_43 = module_0.pmap(var_42)
    var_44 = {var_38: var_40, var_39: var_41}
    var_45 = module_0.pmap(var_44)
    var_46 = hash(var_43)
    var_47 = hash(var_45)
    var_48 = {var_0: var_2, var_1: var_3}
    var_49 = module_0.pmap(var_48)
    var_50 = {var_0: var_2, var_1: var_3}
    var_51 = module_0.pmap(var_50)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = module_0.PMapItems(var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = module_0.pmap(var_9)
    var_11 = module_0.PMapItems(var_10)
    var_12 = 'int'
    var_13 = 'str'
    var_14 = 'list'
    var_15 = 'none'
    var_16 = 42
    var_17 = 'hello'
    var_18 = [var_6, var_7]
    var_19 = None
    var_20 = {var_12: var_16, var_13: var_17, var_14: var_18, var_15: var_19}
    var_21 = module_0.pmap(var_20)
    var_22 = module_0.PMapItems(var_21)
    var_23 = 'x'
    var_24 = 10
    var_25 = {var_23: var_24}
    var_26 = module_0.pmap(var_25)
    var_27 = 'inner'
    var_28 = {var_27: var_26}
    var_29 = module_0.pmap(var_28)
    var_30 = module_0.PMapItems(var_29)
    var_31 = {var_23: var_24}
    var_32 = module_0.pmap(var_31)
    var_33 = (var_27, var_32)
    var_34 = 20
    var_35 = {var_23: var_34}
    var_36 = module_0.pmap(var_35)
    var_37 = (var_27, var_36)



# Parsed testcases at query #2
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_3}
    var_7 = module_0.pmap(var_6)
    var_8 = {var_0: var_2, var_1: var_3}
    var_9 = module_0.pmap(var_8)
    var_10 = {var_0: var_3}
    var_11 = module_0.pmap(var_10)
    var_12 = 'c'
    var_13 = 3
    var_14 = {var_0: var_13, var_12: var_2}
    var_15 = module_0.pmap(var_14)
    var_16 = {var_0: var_2}
    var_17 = module_0.pmap(var_16)
    var_18 = lambda l, r: l
    var_19 = {var_0: var_3}
    var_20 = module_0.pmap(var_19)
    var_21 = {var_0: var_13}
    var_22 = {var_0: var_2}
    var_23 = module_0.pmap(var_22)
    var_24 = lambda l, r: r
    var_25 = {var_0: var_3}
    var_26 = module_0.pmap(var_25)
    var_27 = {var_0: var_13}
    var_28 = module_0.pmap()
    var_29 = {var_0: var_2, var_1: var_3}
    var_30 = {var_0: var_2, var_1: var_3}
    var_31 = module_0.pmap(var_30)
    var_32 = {var_0: var_2}
    var_33 = module_0.pmap(var_32)
    var_34 = {var_1: var_3, var_12: var_13}
    var_35 = 10
    var_36 = 20
    var_37 = {var_0: var_35, var_1: var_36}
    var_38 = module_0.pmap(var_37)
    var_39 = 5
    var_40 = {var_0: var_39, var_1: var_35}
    var_41 = {var_0: var_2, var_1: var_3}
    var_42 = module_0.pmap(var_41)
    var_43 = {var_0: var_3}
    var_44 = module_0.pmap(var_43)
    var_45 = {var_0: var_2}
    var_46 = module_0.pmap(var_45)
    var_47 = {var_0: var_3}
    var_48 = {var_0: var_13}
    var_49 = module_0.pmap(var_48)
    var_50 = 4
    var_51 = dict(a=var_50)
    var_52 = {var_0: var_2}
    var_53 = module_0.pmap(var_52)
    var_54 = None
    var_55 = lambda l, r: var_54
    var_56 = {var_0: var_3}



# Parsed testcases at query #3
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = module_0.PMapItems(var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 'c'
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = module_0.pmap(var_9)
    var_11 = module_0.PMapItems(var_10)
    var_12 = 'x'
    var_13 = 'y'
    var_14 = 'z'
    var_15 = [var_3, var_4]
    var_16 = 'nested'
    var_17 = 'dict'
    var_18 = {var_16: var_17}
    var_19 = None
    var_20 = {var_12: var_15, var_13: var_18, var_14: var_19}
    var_21 = module_0.pmap(var_20)
    var_22 = module_0.PMapItems(var_21)
    var_23 = [var_3, var_4]
    var_24 = 'key'
    var_25 = {var_24: var_23}
    var_26 = module_0.pmap(var_25)
    var_27 = module_0.PMapItems(var_26)



# Parsed testcases at query #4
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = module_0.PMapItems(var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 'c'
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = module_0.pmap(var_9)
    var_11 = module_0.PMapItems(var_10)
    var_12 = 'key'
    var_13 = 42
    var_14 = [var_3, var_4, var_5]
    var_15 = 'nested'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = {var_12: var_14, var_13: var_17}
    var_19 = module_0.pmap(var_18)
    var_20 = module_0.PMapItems(var_19)
    var_21 = 'original'
    var_22 = {var_3: var_21}
    var_23 = module_0.pmap(var_22)
    var_24 = module_0.PMapItems(var_23)
    var_25 = 'updated'



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
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_0.pmap(var_6)
    var_8 = 3
    var_9 = {var_0: var_2, var_1: var_8}
    var_10 = module_0.pmap(var_9)
    var_11 = {var_0: var_2, var_1: var_3}
    var_12 = {var_0: var_2, var_1: var_8}
    var_13 = {var_0: var_2, var_1: var_3}
    var_14 = 'c'
    var_15 = {var_0: var_2, var_1: var_3, var_14: var_8}
    var_16 = module_0.pmap(var_15)
    var_17 = {}
    var_18 = module_0.pmap(var_17)
    var_19 = {}
    var_20 = module_0.pmap(var_19)
    var_21 = 'not a mapping'
    var_22 = 'x'
    var_23 = 'y'
    var_24 = 10
    var_25 = 20
    var_26 = {var_22: var_24, var_23: var_25}
    var_27 = module_0.pmap(var_26)
    var_28 = {var_22: var_24, var_23: var_25}
    var_29 = module_0.pmap(var_28)
    var_30 = hash(var_27)
    var_31 = hash(var_29)
    var_32 = {var_22: var_24, var_23: var_25}
    var_33 = module_0.pmap(var_32)
    var_34 = 21
    var_35 = {var_22: var_24, var_23: var_34}
    var_36 = module_0.pmap(var_35)
    var_37 = hash(var_33)
    var_38 = hash(var_36)
    var_39 = {var_0: var_2}
    var_40 = module_0.pmap(var_39)
    var_41 = {var_0: var_2}
    var_42 = module_0.pmap(var_41)
    var_43 = 'd'
    var_44 = 'e'
    var_45 = 4
    var_46 = 5
    var_47 = {var_0: var_2, var_1: var_3, var_14: var_8, var_43: var_45, var_44: var_46}
    var_48 = module_0.pmap(var_47)
    var_49 = {var_0: var_2, var_1: var_3, var_14: var_8, var_43: var_45, var_44: var_46}
    var_50 = module_0.pmap(var_49)
    var_51 = module_0.m()
    var_52 = module_0.m()



# Parsed testcases at query #6
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = module_0.PMapItems(var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = module_0.pmap(var_9)
    var_11 = module_0.PMapItems(var_10)
    var_12 = 'x'
    var_13 = 'y'
    var_14 = [var_6, var_7]
    var_15 = 'nested'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = {var_12: var_14, var_13: var_17}
    var_19 = module_0.pmap(var_18)
    var_20 = module_0.PMapItems(var_19)
    var_21 = 'key1'
    var_22 = 'key2'
    var_23 = 'val1'
    var_24 = 'val2'
    var_25 = {var_21: var_23, var_22: var_24}
    var_26 = module_0.PMapItems(var_25)



# Parsed testcases at query #7
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()
    var_4 = module_0.m()
    var_5 = module_0.m()
    var_6 = 'a'
    var_7 = 'c'
    var_8 = 3
    var_9 = 5
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = module_0.m()
    var_12 = lambda l, r: l
    var_13 = module_0.m()
    var_14 = {var_6: var_8}
    var_15 = module_0.m()
    var_16 = lambda l, r: r
    var_17 = module_0.m()
    var_18 = {var_6: var_8}
    var_19 = module_0.m()
    var_20 = {}
    var_21 = module_0.m()
    var_22 = module_0.m()
    var_23 = 'b'
    var_24 = {var_23: var_1, var_7: var_8}
    var_25 = 10
    var_26 = 20
    var_27 = module_0.m()
    var_28 = lambda x, y: x * y
    var_29 = {var_6: var_1, var_23: var_8}
    var_30 = module_0.m()
    var_31 = hash(var_30)
    var_32 = module_0.m()
    var_33 = hash(var_30)
    var_34 = module_0.m()
    var_35 = 'hello'
    var_36 = 'world'
    var_37 = module_0.m()
    var_38 = ' '
    var_39 = lambda x, y: x + var_38 + y
    var_40 = 'there'
    var_41 = {var_6: var_40}
    var_42 = [var_0, var_1]
    var_43 = 4
    var_44 = [var_8, var_43]
    var_45 = module_0.m()
    var_46 = lambda x, y: x + y
    var_47 = [var_8, var_43]
    var_48 = {var_6: var_47}
    var_49 = module_0.m()
    var_50 = (var_6, var_8)
    var_51 = (var_7, var_43)
    var_52 = [var_50, var_51]



# Parsed testcases at query #8
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = module_0.PMapItems(var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = module_0.pmap(var_9)
    var_11 = module_0.PMapItems(var_10)
    var_12 = 'x'
    var_13 = 'y'
    var_14 = 5
    var_15 = {var_13: var_14}
    var_16 = module_0.pmap(var_15)
    var_17 = {var_12: var_16}
    var_18 = module_0.pmap(var_17)
    var_19 = module_0.PMapItems(var_18)
    var_20 = {var_13: var_14}
    var_21 = module_0.pmap(var_20)
    var_22 = (var_12, var_21)
    var_23 = {var_3: var_6, var_4: var_7}
    var_24 = module_0.pmap(var_23)
    var_25 = module_0.PMapItems(var_24)
    var_26 = module_0.PMapItems(var_24)
    var_27 = 'k1'
    var_28 = 'k2'
    var_29 = 'v1'
    var_30 = 'v2'
    var_31 = {var_27: var_29, var_28: var_30}



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
    var_8 = 3
    var_9 = {var_0: var_2, var_1: var_8}
    var_10 = module_0.pmap(var_9)
    var_11 = 'c'
    var_12 = {var_0: var_2, var_11: var_3}
    var_13 = module_0.pmap(var_12)
    var_14 = {var_0: var_2, var_1: var_3, var_11: var_8}
    var_15 = module_0.pmap(var_14)
    var_16 = {}
    var_17 = module_0.pmap(var_16)
    var_18 = 'x'
    var_19 = 10
    var_20 = 'y'
    var_21 = 20
    var_22 = {var_18: var_19, var_20: var_21}
    var_23 = module_0.pmap(var_22)
    var_24 = {}
    var_25 = module_0.pmap(var_24)
    var_26 = {}
    var_27 = module_0.pmap(var_26)
    var_28 = {var_0: var_2, var_1: var_3}
    var_29 = {var_18: var_2, var_20: var_3}
    var_30 = module_0.pmap(var_29)
    var_31 = {var_18: var_2, var_20: var_3}
    var_32 = module_0.pmap(var_31)
    var_33 = hash(var_30)
    var_34 = hash(var_32)
    var_35 = {var_18: var_2, var_20: var_3}
    var_36 = module_0.pmap(var_35)
    var_37 = {var_18: var_2, var_20: var_8}
    var_38 = module_0.pmap(var_37)
    var_39 = hash(var_36)
    var_40 = hash(var_38)



# Parsed testcases at query #10
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = module_0.PMapItems(var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = module_0.pmap(var_9)
    var_11 = module_0.PMapItems(var_10)
    var_12 = 2.5
    var_13 = (var_6, var_7)
    var_14 = 'one'
    var_15 = [var_6, var_7, var_8]
    var_16 = 'nested'
    var_17 = 'dict'
    var_18 = {var_16: var_17}
    var_19 = {var_6: var_14, var_12: var_15, var_13: var_18}
    var_20 = module_0.pmap(var_19)
    var_21 = module_0.PMapItems(var_20)
    var_22 = 'x'
    var_23 = 'y'
    var_24 = 10
    var_25 = 20
    var_26 = {var_22: var_24, var_23: var_25}
    var_27 = module_0.pmap(var_26)
    var_28 = module_0.PMapItems(var_27)



# Parsed testcases at query #11
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = module_0.PMapItems(var_1)
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = module_0.pmap(var_5)
    var_7 = module_0.PMapItems(var_6)
    var_8 = 'x'
    var_9 = 'y'
    var_10 = 'z'
    var_11 = 10
    var_12 = 20
    var_13 = 30
    var_14 = {var_8: var_11, var_9: var_12, var_10: var_13}
    var_15 = module_0.pmap(var_14)
    var_16 = module_0.PMapItems(var_15)
    var_17 = 'key'
    var_18 = 'value'
    var_19 = {var_17: var_18}
    var_20 = module_0.pmap(var_19)
    var_21 = module_0.PMapItems(var_20)
    var_22 = 'b'
    var_23 = 2
    var_24 = {var_3: var_4, var_22: var_23}
    var_25 = module_0.pmap(var_24)
    var_26 = module_0.PMapItems(var_25)
    var_27 = 'num'
    var_28 = 5
    var_29 = {var_27: var_28}
    var_30 = module_0.pmap(var_29)
    var_31 = module_0.PMapItems(var_30)
    var_32 = 'old'
    var_33 = 'data'
    var_34 = {var_32: var_33}
    var_35 = module_0.pmap(var_34)
    var_36 = module_0.PMapItems(var_35)
    var_37 = 'new'
    var_38 = 'item'



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
    var_6 = {var_0: var_3}
    var_7 = module_0.pmap(var_6)
    var_8 = {var_0: var_2, var_1: var_3}
    var_9 = module_0.pmap(var_8)
    var_10 = {var_0: var_3}
    var_11 = module_0.pmap(var_10)
    var_12 = 'c'
    var_13 = 3
    var_14 = {var_1: var_13, var_12: var_2}
    var_15 = {var_0: var_2, var_1: var_3}
    var_16 = module_0.pmap(var_15)
    var_17 = lambda l, r: l
    var_18 = {var_0: var_3}
    var_19 = module_0.pmap(var_18)
    var_20 = {var_0: var_13}
    var_21 = {var_0: var_2, var_1: var_3}
    var_22 = module_0.pmap(var_21)
    var_23 = lambda l, r: r
    var_24 = {var_0: var_3}
    var_25 = module_0.pmap(var_24)
    var_26 = {var_0: var_13}
    var_27 = 10
    var_28 = 5
    var_29 = {var_0: var_27, var_1: var_28}
    var_30 = module_0.pmap(var_29)
    var_31 = {var_0: var_13, var_1: var_3}
    var_32 = module_0.pmap(var_31)
    var_33 = {var_0: var_2}
    var_34 = module_0.pmap(var_33)
    var_35 = {var_1: var_3, var_12: var_13}
    var_36 = module_0.pmap(var_35)
    var_37 = {var_0: var_2, var_1: var_3}
    var_38 = module_0.pmap(var_37)
    var_39 = {}
    var_40 = module_0.pmap(var_39)
    var_41 = {}
    var_42 = [var_2]
    var_43 = [var_3]
    var_44 = {var_0: var_42, var_1: var_43}
    var_45 = module_0.pmap(var_44)
    var_46 = lambda l, r: l + r
    var_47 = [var_13]
    var_48 = 4
    var_49 = [var_48]
    var_50 = {var_0: var_47, var_1: var_49}
    var_51 = module_0.pmap(var_50)
    var_52 = {var_0: var_2, var_1: var_3}
    var_53 = module_0.pmap(var_52)
    var_54 = hash(var_53)
    var_55 = {var_0: var_3}
    var_56 = module_0.pmap(var_55)
    var_57 = hash(var_53)
    var_58 = {var_0: var_2, var_1: var_3}
    var_59 = module_0.pmap(var_58)
    var_60 = {var_0: var_3}
    var_61 = dict(c=var_13)
    var_62 = {var_0: var_2, var_1: var_3}
    var_63 = module_0.pmap(var_62)
    var_64 = {var_0: var_2, var_1: var_3}
    var_65 = module_0.pmap(var_64)
    var_66 = None
    var_67 = lambda l, r: var_66
    var_68 = {var_0: var_3}
    var_69 = module_0.pmap(var_68)



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
    var_6 = {var_0: var_3}
    var_7 = module_0.pmap(var_6)
    var_8 = {var_0: var_2, var_1: var_3}
    var_9 = module_0.pmap(var_8)
    var_10 = {var_0: var_3}
    var_11 = module_0.pmap(var_10)
    var_12 = 'c'
    var_13 = 3
    var_14 = {var_0: var_13, var_12: var_2}
    var_15 = module_0.pmap(var_14)
    var_16 = {var_0: var_2}
    var_17 = module_0.pmap(var_16)
    var_18 = lambda l, r: l
    var_19 = {var_0: var_3}
    var_20 = module_0.pmap(var_19)
    var_21 = {var_0: var_13}
    var_22 = {var_0: var_2}
    var_23 = module_0.pmap(var_22)
    var_24 = lambda l, r: r
    var_25 = {var_0: var_3}
    var_26 = module_0.pmap(var_25)
    var_27 = {var_0: var_13}
    var_28 = {var_0: var_2}
    var_29 = module_0.pmap(var_28)
    var_30 = {var_1: var_3}
    var_31 = module_0.pmap(var_30)
    var_32 = {var_12: var_13}
    var_33 = module_0.pmap()
    var_34 = {var_0: var_2, var_1: var_3}
    var_35 = {var_0: var_2, var_1: var_3}
    var_36 = module_0.pmap(var_35)
    var_37 = [var_2]
    var_38 = [var_3]
    var_39 = {var_0: var_37, var_1: var_38}
    var_40 = module_0.pmap(var_39)
    var_41 = lambda l, r: l + r
    var_42 = [var_13]
    var_43 = 4
    var_44 = [var_43]
    var_45 = {var_0: var_42, var_12: var_44}
    var_46 = {var_0: var_2, var_1: var_3}
    var_47 = module_0.pmap(var_46)
    var_48 = {var_0: var_13, var_12: var_43}
    var_49 = module_0.pmap(var_48)
    var_50 = {var_0: var_2}
    var_51 = module_0.pmap(var_50)
    var_52 = {var_0: var_3}
    var_53 = {var_0: var_13}
    var_54 = module_0.pmap(var_53)



# Parsed testcases at query #14
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = module_0.PMapItems(var_1)
    var_3 = 1
    var_4 = 'a'
    var_5 = {var_3: var_4}
    var_6 = module_0.pmap(var_5)
    var_7 = module_0.PMapItems(var_6)
    var_8 = 2
    var_9 = 3
    var_10 = 'b'
    var_11 = 'c'
    var_12 = {var_3: var_4, var_8: var_10, var_9: var_11}
    var_13 = module_0.pmap(var_12)
    var_14 = module_0.PMapItems(var_13)
    var_15 = [var_3, var_8]
    var_16 = {var_4: var_3}
    var_17 = {var_3: var_15, var_8: var_16}
    var_18 = module_0.pmap(var_17)
    var_19 = module_0.PMapItems(var_18)
    var_20 = module_0.PMapItems(var_18)



# Parsed testcases at query #15
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = module_0.PMapItems(var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = module_0.pmap(var_9)
    var_11 = module_0.PMapItems(var_10)
    var_12 = 'x'
    var_13 = 'y'
    var_14 = [var_6, var_7]
    var_15 = 'z'
    var_16 = {var_15: var_8}
    var_17 = {var_12: var_14, var_13: var_16}
    var_18 = module_0.pmap(var_17)
    var_19 = module_0.PMapItems(var_18)
    var_20 = {var_3: var_6, var_4: var_7}
    var_21 = module_0.pmap(var_20)
    var_22 = module_0.PMapItems(var_21)
    var_23 = 'initial'
    var_24 = 'value'
    var_25 = {var_23: var_24}
    var_26 = module_0.pmap(var_25)
    var_27 = module_0.PMapItems(var_26)
    var_28 = 'key'
    var_29 = None
    var_30 = {var_28: var_29}
    var_31 = module_0.pmap(var_30)
    var_32 = module_0.PMapItems(var_31)



# Parsed testcases at query #16
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()
    var_4 = module_0.m()
    var_5 = module_0.m()
    var_6 = 'a'
    var_7 = 'c'
    var_8 = 3
    var_9 = {var_6: var_8, var_7: var_0}
    var_10 = module_0.m()
    var_11 = lambda l, r: l
    var_12 = module_0.m()
    var_13 = {var_6: var_8}
    var_14 = module_0.m()
    var_15 = lambda l, r: r
    var_16 = module_0.m()
    var_17 = {var_6: var_8}
    var_18 = module_0.m()
    var_19 = {}
    var_20 = {}
    var_21 = module_0.m()
    var_22 = 'b'
    var_23 = {var_22: var_1}
    var_24 = {var_7: var_8}
    var_25 = module_0.m()
    var_26 = lambda x, y: x * y
    var_27 = {var_6: var_8}
    var_28 = {var_6: var_1}
    var_29 = module_0.m()
    var_30 = hash(var_29)
    var_31 = {var_6: var_1}
    var_32 = hash(var_29)
    var_33 = module_0.m()
    var_34 = 'hello'
    var_35 = 'world'
    var_36 = module_0.m()
    var_37 = ' '
    var_38 = lambda x, y: x + var_37 + y
    var_39 = 'there'
    var_40 = {var_6: var_39}
    var_41 = 'test'
    var_42 = module_0.m()
    var_43 = lambda l, r: str(l) + str(r)
    var_44 = {var_6: var_1}
    var_45 = {var_6: var_8}



# Parsed testcases at query #17
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()
    var_4 = module_0.m()
    var_5 = module_0.m()
    var_6 = 'a'
    var_7 = 'c'
    var_8 = 3
    var_9 = {var_6: var_8, var_7: var_0}
    var_10 = module_0.m()
    var_11 = lambda l, r: l
    var_12 = module_0.m()
    var_13 = {var_6: var_8}
    var_14 = module_0.m()
    var_15 = lambda l, r: r
    var_16 = module_0.m()
    var_17 = {var_6: var_8}
    var_18 = module_0.m()
    var_19 = {}
    var_20 = {}
    var_21 = module_0.m()
    var_22 = 'b'
    var_23 = {var_22: var_1}
    var_24 = {var_7: var_8}
    var_25 = module_0.m()
    var_26 = 4
    var_27 = {var_6: var_8, var_22: var_26}
    var_28 = module_0.m()
    var_29 = hash(var_28)
    var_30 = {var_6: var_1}
    var_31 = hash(var_28)
    var_32 = module_0.m()
    var_33 = module_0.m()
    var_34 = dict(b=var_1)
    var_35 = module_0.m()
    var_36 = module_0.m()
    var_37 = None
    var_38 = lambda l, r: var_37
    var_39 = {var_6: var_1}



# Parsed testcases at query #18
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 3
    var_4 = module_0.m()
    var_5 = module_0.m()
    var_6 = module_0.m()
    var_7 = 'a'
    var_8 = 'c'
    var_9 = 5
    var_10 = 4
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = module_0.m()
    var_13 = lambda l, r: l
    var_14 = module_0.m()
    var_15 = {var_7: var_9}
    var_16 = module_0.m()
    var_17 = lambda l, r: r
    var_18 = module_0.m()
    var_19 = {var_7: var_9}
    var_20 = module_0.m()
    var_21 = module_0.m()
    var_22 = module_0.m()
    var_23 = module_0.m()
    var_24 = module_0.m()
    var_25 = module_0.m()
    var_26 = module_0.m()
    var_27 = module_0.m()
    var_28 = module_0.m()
    var_29 = module_0.m()
    var_30 = 'x'
    var_31 = {var_30: var_0}
    var_32 = module_0.m()
    var_33 = 'y'
    var_34 = {var_33: var_1}
    var_35 = module_0.m()
    var_36 = None
    var_37 = module_0.m()
    var_38 = lambda l, r: r if l is var_36 else l
    var_39 = module_0.m()



# Parsed testcases at query #19
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = module_0.PMapItems(var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = module_0.pmap(var_9)
    var_11 = module_0.PMapItems(var_10)
    var_12 = 'int'
    var_13 = 'str'
    var_14 = 'list'
    var_15 = 'dict'
    var_16 = 'none'
    var_17 = 42
    var_18 = 'hello'
    var_19 = [var_6, var_7, var_8]
    var_20 = 'nested'
    var_21 = 'value'
    var_22 = {var_20: var_21}
    var_23 = None
    var_24 = {var_12: var_17, var_13: var_18, var_14: var_19, var_15: var_22, var_16: var_23}
    var_25 = module_0.pmap(var_24)
    var_26 = module_0.PMapItems(var_25)



# Parsed testcases at query #20
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = module_0.PMapItems(var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 'c'
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = module_0.pmap(var_9)
    var_11 = module_0.PMapItems(var_10)
    var_12 = 'key'
    var_13 = 42
    var_14 = [var_3, var_4, var_5]
    var_15 = 'nested'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = {var_12: var_14, var_13: var_17}
    var_19 = module_0.pmap(var_18)
    var_20 = module_0.PMapItems(var_19)
    var_21 = None
    var_22 = {var_3: var_21, var_4: var_21}
    var_23 = module_0.pmap(var_22)
    var_24 = module_0.PMapItems(var_23)



# Parsed testcases at query #21
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
    var_12 = 'x'
    var_13 = 'y'
    var_14 = 'z'
    var_15 = [var_3, var_4]
    var_16 = 'nested'
    var_17 = 'dict'
    var_18 = {var_16: var_17}
    var_19 = None
    var_20 = {var_12: var_15, var_13: var_18, var_14: var_19}
    var_21 = module_0.pmap(var_20)
    var_22 = module_0.PMapItems(var_21)
    var_23 = 'key'
    var_24 = 'value'
    var_25 = {var_23: var_24}
    var_26 = module_0.pmap(var_25)
    var_27 = module_0.PMapItems(var_26)
    var_28 = 100
    var_29 = range(var_28)
    var_30 = {i: i * var_4 for i in var_29}
    var_31 = module_0.pmap(var_30)
    var_32 = module_0.PMapItems(var_31)



# Parsed testcases at query #22
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
    var_12 = 'x'
    var_13 = 'y'
    var_14 = 10
    var_15 = 20
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = module_0.PMapItems(var_16)



# Parsed testcases at query #23
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_3}
    var_7 = module_0.pmap(var_6)
    var_8 = {var_0: var_2, var_1: var_3}
    var_9 = module_0.pmap(var_8)
    var_10 = {var_0: var_3}
    var_11 = module_0.pmap(var_10)
    var_12 = 'c'
    var_13 = 3
    var_14 = {var_0: var_13, var_12: var_2}
    var_15 = module_0.pmap(var_14)
    var_16 = {var_0: var_2}
    var_17 = module_0.pmap(var_16)
    var_18 = lambda l, r: l
    var_19 = {var_0: var_3}
    var_20 = module_0.pmap(var_19)
    var_21 = {var_0: var_13}
    var_22 = {var_0: var_2}
    var_23 = module_0.pmap(var_22)
    var_24 = lambda l, r: r
    var_25 = {var_0: var_3}
    var_26 = module_0.pmap(var_25)
    var_27 = {var_0: var_13}
    var_28 = {var_0: var_2, var_1: var_3}
    var_29 = module_0.pmap(var_28)
    var_30 = {}
    var_31 = {}
    var_32 = module_0.pmap(var_31)
    var_33 = {var_0: var_2}
    var_34 = module_0.pmap(var_33)
    var_35 = {var_1: var_3}
    var_36 = module_0.pmap(var_35)
    var_37 = {var_12: var_13}
    var_38 = {var_0: var_2}
    var_39 = module_0.pmap(var_38)
    var_40 = {var_0: var_3}
    var_41 = {var_0: var_13}
    var_42 = module_0.pmap(var_41)
    var_43 = {var_0: var_2, var_1: var_3}
    var_44 = module_0.pmap(var_43)
    var_45 = [var_2]
    var_46 = [var_3]
    var_47 = {var_0: var_45, var_1: var_46}
    var_48 = module_0.pmap(var_47)
    var_49 = lambda l, r: l + r
    var_50 = [var_13]
    var_51 = {var_0: var_50}
    var_52 = 4
    var_53 = [var_52]
    var_54 = 5
    var_55 = [var_54]
    var_56 = {var_0: var_53, var_12: var_55}
    var_57 = module_0.pmap(var_56)
    var_58 = {var_0: var_2, var_1: var_3}
    var_59 = module_0.pmap(var_58)
    var_60 = {var_0: var_13, var_12: var_52}
    var_61 = module_0.pmap(var_60)



# Parsed testcases at query #24
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
    var_12 = 'x'
    var_13 = 'y'
    var_14 = 5
    var_15 = {var_13: var_14}
    var_16 = module_0.pmap(var_15)
    var_17 = {var_12: var_16}
    var_18 = module_0.pmap(var_17)
    var_19 = module_0.PMapItems(var_18)
    var_20 = {var_13: var_14}
    var_21 = module_0.pmap(var_20)
    var_22 = (var_12, var_21)



# Parsed testcases at query #25
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = module_0.PMapItems(var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = module_0.pmap(var_9)
    var_11 = module_0.PMapItems(var_10)
    var_12 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_13 = module_0.pmap(var_12)
    var_14 = module_0.PMapItems(var_13)
    var_15 = 'x'
    var_16 = 'y'
    var_17 = [var_6, var_7]
    var_18 = 'z'
    var_19 = {var_18: var_8}
    var_20 = {var_15: var_17, var_16: var_19}
    var_21 = module_0.pmap(var_20)
    var_22 = module_0.PMapItems(var_21)



