####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = [var_1, var_2]
    var_7 = [var_0, var_6]
    var_8 = module_0.freeze(var_7)
    var_9 = [var_1, var_2]
    var_10 = 'a'
    var_11 = 'b'
    var_12 = {var_10: var_0, var_11: var_1}
    var_13 = module_0.freeze(var_12)
    var_14 = {var_10: var_0, var_11: var_1}
    var_15 = module_1.pmap(var_14)
    var_16 = [var_0, var_1]
    var_17 = {var_10: var_16}
    var_18 = module_0.freeze(var_17)
    var_19 = [var_0, var_1]
    var_20 = {var_10: var_0, var_11: var_1}
    var_21 = {var_10: var_0, var_11: var_1}
    var_22 = module_1.pmap(var_21)
    var_23 = {var_0, var_1, var_2}
    var_24 = module_0.freeze(var_23)
    var_25 = {var_0, var_1, var_2}
    var_26 = module_2.pset(var_25)
    var_27 = (var_0, var_1, var_2)
    var_28 = module_0.freeze(var_27)
    var_29 = [var_1, var_2]
    var_30 = (var_0, var_29)
    var_31 = module_0.freeze(var_30)
    var_32 = [var_1, var_2]
    var_33 = 'c'
    var_34 = {var_11: var_1}
    var_35 = [var_0, var_34]
    var_36 = 4
    var_37 = {var_2, var_36}
    var_38 = {var_10: var_35, var_33: var_37}
    var_39 = {var_11: var_1}
    var_40 = module_1.pmap(var_39)
    var_41 = [var_0, var_40]
    var_42 = {var_2, var_36}
    var_43 = module_2.pset(var_42)
    var_44 = module_0.freeze(var_38)
    var_45 = [var_1, var_2]
    var_46 = [var_0, var_45]
    var_47 = False
    var_48 = module_0.freeze(var_46, var_47)
    var_49 = [var_1, var_2]
    var_50 = [var_0, var_49]
    var_51 = [var_0, var_1]
    var_52 = {var_10: var_51}
    var_53 = module_0.freeze(var_52, var_47)
    var_54 = [var_0, var_1]
    var_55 = {var_10: var_54}
    var_56 = module_1.pmap(var_55)
    var_57 = [var_0, var_1, var_2]
    var_58 = [var_0, var_1, var_2]
    var_59 = {var_10: var_0}
    var_60 = module_1.pmap(var_59)
    var_61 = module_0.freeze(var_60)
    var_62 = {var_10: var_0}
    var_63 = module_1.pmap(var_62)
    var_64 = {var_0, var_1}
    var_65 = module_2.pset(var_64)
    var_66 = module_0.freeze(var_65)
    var_67 = {var_0, var_1}
    var_68 = module_2.pset(var_67)
    var_69 = module_0.freeze(var_0)
    assert var_69 == 1
    var_70 = 'hello'
    var_71 = module_0.freeze(var_70)
    assert var_71 == 'hello'
    var_72 = None
    var_73 = module_0.freeze(var_72)
    assert var_73 is None



# Parsed testcases at query #2
#--------------------------


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 'a'
    var_7 = {var_6: var_0}
    var_8 = 'b'
    var_9 = {var_6: var_0, var_8: var_1}
    var_10 = module_0.pmap(var_9)
    var_11 = {var_0, var_1}
    var_12 = {var_0, var_1, var_2}
    var_13 = module_1.pset(var_12)
    var_14 = (var_0, var_1)
    var_15 = 'list'
    var_16 = 'dict'
    var_17 = [var_0, var_1, var_2]
    var_18 = {var_6: var_0}
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = [var_0, var_1, var_2, var_4]
    var_21 = 'c'
    var_22 = {var_6: var_0, var_21: var_2}
    var_23 = module_0.pmap(var_22)



# Parsed testcases at query #3
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_3]
    var_5 = module_0.freeze(var_4)
    var_6 = {var_1: var_2}
    var_7 = module_1.pmap(var_6)
    var_8 = [var_0, var_7]
    var_9 = []
    var_10 = (var_0, var_9)
    var_11 = module_0.freeze(var_10)
    var_12 = []
    var_13 = 2
    var_14 = [var_0, var_13]
    var_15 = set(var_14)
    var_16 = module_0.freeze(var_15)
    var_17 = [var_0, var_13]
    var_18 = module_2.pset(var_17)
    var_19 = [var_0, var_13]
    var_20 = {var_1: var_19}
    var_21 = module_0.freeze(var_20)
    var_22 = [var_0, var_13]
    var_23 = [var_0, var_13]
    var_24 = {var_1: var_23}
    var_25 = [var_0, var_13]
    var_26 = {var_1: var_2}
    var_27 = [var_0, var_26]
    var_28 = False
    var_29 = module_0.freeze(var_27, var_28)
    var_30 = {var_1: var_2}
    var_31 = [var_0, var_30]
    var_32 = [var_0, var_13]
    var_33 = True
    var_34 = [var_33, var_13]
    var_35 = [var_33, var_13]
    var_36 = [var_33, var_13]
    var_37 = [var_2]
    var_38 = (var_13, var_37)
    var_39 = (var_33, var_38)
    var_40 = module_0.freeze(var_39)
    var_41 = [var_2]
    var_42 = []
    var_43 = module_0.freeze(var_42)
    var_44 = []
    var_45 = {}
    var_46 = module_0.freeze(var_45)
    var_47 = {}
    var_48 = module_1.pmap(var_47)
    var_49 = set()
    var_50 = module_0.freeze(var_49)
    var_51 = set()
    var_52 = module_2.pset(var_51)
    var_53 = ()
    var_54 = module_0.freeze(var_53)
    var_55 = 42
    var_56 = module_0.freeze(var_55)
    assert var_56 == 42



# Parsed testcases at query #4
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_3]
    var_5 = module_0.freeze(var_4)
    var_6 = {var_1: var_2}
    var_7 = module_1.pmap(var_6)
    var_8 = [var_0, var_7]
    var_9 = []
    var_10 = (var_0, var_9)
    var_11 = module_0.freeze(var_10)
    var_12 = []
    var_13 = 2
    var_14 = [var_0, var_13]
    var_15 = set(var_14)
    var_16 = module_0.freeze(var_15)
    var_17 = [var_0, var_13]
    var_18 = module_2.pset(var_17)
    var_19 = 'b'
    var_20 = [var_13, var_2]
    var_21 = {var_1: var_0, var_19: var_20}
    var_22 = module_0.freeze(var_21)
    var_23 = [var_13, var_2]
    var_24 = [var_13, var_2]
    var_25 = {var_1: var_0, var_19: var_24}
    var_26 = [var_13, var_2]
    var_27 = {var_1: var_2}
    var_28 = [var_0, var_27]
    var_29 = False
    var_30 = module_0.freeze(var_28, var_29)
    var_31 = {var_1: var_2}
    var_32 = [var_0, var_31]
    var_33 = [var_0, var_13, var_2]
    var_34 = {var_1: var_0}
    var_35 = module_1.pmap(var_34)
    var_36 = module_0.freeze(var_35)
    var_37 = [var_0, var_13]
    var_38 = module_2.pset(var_37)
    var_39 = module_0.freeze(var_38)
    var_40 = 42
    var_41 = module_0.freeze(var_40)
    assert var_41 == 42
    var_42 = 'hello'
    var_43 = module_0.freeze(var_42)
    assert var_43 == 'hello'



# Parsed testcases at query #5
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 'nested'
    var_7 = 'value'
    var_8 = 5
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = 10
    var_12 = {var_7: var_11}
    var_13 = module_0.pmap(var_12)
    var_14 = {var_6: var_13}
    var_15 = module_0.pmap(var_14)
    var_16 = [var_0]
    var_17 = [var_0, var_1]
    var_18 = 'a'
    var_19 = {var_18: var_0}
    var_20 = module_0.pmap(var_19)
    var_21 = [var_0, var_1]
    var_22 = [var_2, var_4]
    var_23 = [var_0, var_1, var_2, var_4]



# Parsed testcases at query #6
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 'a'
    var_7 = {var_6: var_0}
    var_8 = 'b'
    var_9 = {var_6: var_0, var_8: var_1}
    var_10 = module_0.pmap(var_9)
    var_11 = 'list'
    var_12 = 'nested'
    var_13 = [var_0, var_1, var_2]
    var_14 = 'value'
    var_15 = {var_14: var_0}
    var_16 = {var_11: var_13, var_12: var_15}
    var_17 = [var_0, var_1, var_2, var_4]
    var_18 = {var_14: var_1}
    var_19 = module_0.pmap(var_18)
    var_20 = 'x'
    var_21 = 'y'
    var_22 = 10
    var_23 = {var_20: var_22, var_21: var_1}
    var_24 = module_0.pmap(var_23)
    var_25 = [var_0, var_1, var_2]
    var_26 = 42
    var_27 = 'hello'



# Parsed testcases at query #7
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 'nested'
    var_7 = 'value'
    var_8 = 50
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = 100
    var_12 = {var_7: var_11}
    var_13 = module_0.pmap(var_12)
    var_14 = {var_6: var_13}
    var_15 = module_0.pmap(var_14)
    var_16 = [var_0, var_1]
    var_17 = [var_2, var_4]
    var_18 = [var_0, var_1, var_2, var_4]
    var_19 = 'a'
    var_20 = {var_19: var_0}
    var_21 = 'b'
    var_22 = {var_19: var_0, var_21: var_1}
    var_23 = module_0.pmap(var_22)
    var_24 = [var_0, var_1, var_2]
    var_25 = (var_0, var_1, var_2)



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 3
    var_5 = [var_0, var_1, var_4]
    var_6 = 4
    var_7 = [var_0, var_1, var_4, var_6]
    var_8 = [var_0, var_1, var_4]
    var_9 = {var_2: var_8}
    var_10 = [var_0, var_1, var_4, var_6]
    var_11 = 'c'
    var_12 = [var_0, var_1, var_4]
    var_13 = {var_11: var_12}
    var_14 = [var_0, var_1, var_4, var_6]
    var_15 = [var_0, var_1, var_4]



# Parsed testcases at query #9
#--------------------------


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 'a'
    var_7 = 'b'
    var_8 = [var_0, var_1]
    var_9 = 'c'
    var_10 = {var_9: var_2}
    var_11 = {var_6: var_8, var_7: var_10}
    var_12 = [var_0, var_1, var_0]
    var_13 = {var_9: var_1}
    var_14 = module_0.pmap(var_13)
    var_15 = 'key'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = {var_0, var_1, var_2}
    var_19 = 'modified'
    var_20 = {var_15: var_19}
    var_21 = module_0.pmap(var_20)
    var_22 = 5
    var_23 = {var_0, var_1, var_2, var_22}
    var_24 = module_1.pset(var_23)
    var_25 = (var_0, var_1, var_2)
    var_26 = [var_0, var_1, var_2]



# Parsed testcases at query #10
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1
import pyrsistent._pmap as module_2

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 3
    var_4 = 4
    var_5 = 5
    var_6 = [var_4, var_5]
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = 6
    var_9 = 7
    var_10 = [var_8, var_9]
    var_11 = [var_0, var_7, var_10]
    var_12 = module_0.freeze(var_11)
    var_13 = var_12[var_0]
    var_14 = var_12[var_0][var_2]
    var_15 = [var_4, var_5]
    var_16 = 2
    var_17 = var_12[var_16]
    var_18 = [var_8, var_9]
    var_19 = 'x'
    var_20 = 'y'
    var_21 = 'z'
    var_22 = [var_16, var_3]
    var_23 = {var_1: var_4}
    var_24 = {var_19: var_0, var_20: var_22, var_21: var_23}
    var_25 = module_0.freeze(var_24)
    var_26 = var_25[var_20]
    var_27 = [var_16, var_3]
    var_28 = var_25[var_21]
    var_29 = {var_0, var_16, var_3}
    var_30 = module_0.freeze(var_29)
    var_31 = {var_0, var_16, var_3}
    var_32 = module_1.pset(var_31)
    var_33 = [var_16, var_3]
    var_34 = {var_1: var_4}
    var_35 = (var_0, var_33, var_34)
    var_36 = module_0.freeze(var_35)
    var_37 = var_36[var_0]
    var_38 = [var_16, var_3]
    var_39 = var_36[var_16]
    var_40 = [var_16, var_3]
    var_41 = {var_1: var_0, var_2: var_40}
    var_42 = var_36[var_2]
    var_43 = [var_16, var_3]
    var_44 = [var_16, var_3]
    var_45 = {var_1: var_44}
    var_46 = [var_0, var_45]
    var_47 = True
    var_48 = module_0.freeze(var_46, var_47)
    var_49 = False
    var_50 = module_0.freeze(var_46, var_49)
    var_51 = var_48[var_47][var_1]
    var_52 = var_50[var_47][var_1]
    var_53 = {var_1: var_16}
    var_54 = module_2.pmap(var_53)
    var_55 = [var_47, var_54]
    var_56 = var_36[var_47]
    var_57 = module_0.freeze(var_47)
    assert var_57 == 1
    var_58 = 'hello'
    var_59 = module_0.freeze(var_58)
    assert var_59 == 'hello'
    var_60 = (var_47, var_16, var_3)
    var_61 = module_0.freeze(var_60)



# Parsed testcases at query #11
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 'a'
    var_7 = [var_0, var_1]
    var_8 = {var_6: var_7}
    var_9 = 'b'
    var_10 = [var_2, var_4]
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = 'key1'
    var_14 = 'value1'
    var_15 = {var_13: var_14}
    var_16 = 'new_key'
    var_17 = 'new_value'
    var_18 = {var_13: var_14, var_16: var_17}
    var_19 = module_0.pmap(var_18)
    var_20 = [var_0, var_1]
    var_21 = [var_2, var_4]
    var_22 = 99
    var_23 = [var_0, var_1, var_2, var_4, var_22]



# Parsed testcases at query #12
#--------------------------


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 'a'
    var_7 = {var_6: var_0}
    var_8 = 'b'
    var_9 = {var_6: var_0, var_8: var_1}
    var_10 = module_0.pmap(var_9)
    var_11 = {var_0, var_1}
    var_12 = {var_0, var_1, var_2}
    var_13 = module_1.pset(var_12)
    var_14 = (var_0, var_1, var_2)
    var_15 = 99
    var_16 = 'list'
    var_17 = 'dict'
    var_18 = [var_0, var_1, var_2]
    var_19 = 'key'
    var_20 = 'value'
    var_21 = {var_19: var_20}
    var_22 = {var_16: var_18, var_17: var_21}
    var_23 = [var_0, var_1, var_2, var_4]
    var_24 = 'new_key'
    var_25 = 'new_value'
    var_26 = {var_19: var_20, var_24: var_25}
    var_27 = module_0.pmap(var_26)



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    pass



# Parsed testcases at query #14
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 'a'
    var_7 = {var_6: var_0}
    var_8 = 'b'
    var_9 = {var_6: var_0, var_8: var_1}
    var_10 = module_0.pmap(var_9)
    var_11 = 'list'
    var_12 = 'dict'
    var_13 = [var_0, var_1]
    var_14 = 'key'
    var_15 = 'val'
    var_16 = {var_14: var_15}
    var_17 = {var_11: var_13, var_12: var_16}
    var_18 = [var_0, var_1, var_0]
    var_19 = 'new_key'
    var_20 = 'value'
    var_21 = {var_14: var_15, var_19: var_20}
    var_22 = module_0.pmap(var_21)
    var_23 = [var_0]
    var_24 = 2
    var_25 = {var_14: var_20}
    var_26 = 'kwargs'
    var_27 = [var_0, var_0]
    var_28 = 'new_value'
    var_29 = {var_14: var_20, var_19: var_28}
    var_30 = module_0.pmap(var_29)



# Parsed testcases at query #15
#--------------------------


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 'a'
    var_7 = {var_6: var_0}
    var_8 = 'b'
    var_9 = {var_6: var_0, var_8: var_1}
    var_10 = module_0.pmap(var_9)
    var_11 = {var_0, var_1}
    var_12 = {var_0, var_1, var_2}
    var_13 = module_1.pset(var_12)
    var_14 = (var_0, var_1)
    var_15 = 'list'
    var_16 = 'dict'
    var_17 = [var_0, var_1, var_2]
    var_18 = {var_6: var_0, var_8: var_1}
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = [var_0, var_1, var_2, var_4]
    var_21 = 'c'
    var_22 = {var_6: var_0, var_8: var_1, var_21: var_2}
    var_23 = module_0.pmap(var_22)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_3]
    var_5 = module_0.freeze(var_4)
    var_6 = {var_1: var_2}
    var_7 = module_1.pmap(var_6)
    var_8 = [var_0, var_7]
    var_9 = []
    var_10 = (var_0, var_9)
    var_11 = module_0.freeze(var_10)
    var_12 = []
    var_13 = 2
    var_14 = [var_0, var_13]
    var_15 = set(var_14)
    var_16 = module_0.freeze(var_15)
    var_17 = [var_0, var_13]
    var_18 = module_2.pset(var_17)
    var_19 = [var_0, var_13]
    var_20 = {var_1: var_19}
    var_21 = module_0.freeze(var_20)
    var_22 = [var_0, var_13]
    var_23 = [var_0, var_13]
    var_24 = {var_1: var_23}
    var_25 = [var_0, var_13]
    var_26 = [var_0, var_13, var_2]
    var_27 = {var_1: var_0}
    var_28 = module_1.pmap(var_27)
    var_29 = module_0.freeze(var_28)
    var_30 = [var_0, var_13]
    var_31 = module_2.pset(var_30)
    var_32 = module_0.freeze(var_31)
    var_33 = (var_0, var_13, var_2)
    var_34 = module_0.freeze(var_33)
    var_35 = 42
    var_36 = module_0.freeze(var_35)
    assert var_36 == 42
    var_37 = 4
    var_38 = [var_2, var_37]
    var_39 = {var_1: var_38}
    var_40 = [var_0, var_39]
    var_41 = False
    var_42 = module_0.freeze(var_40, var_41)
    var_43 = [var_2, var_37]
    var_44 = {var_1: var_43}
    var_45 = [var_0, var_44]



# Parsed testcases at query #2
#--------------------------


import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_1, var_2]
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_5: var_0, var_6: var_1}
    var_8 = module_0.pmap(var_7)
    var_9 = module_1.thaw(var_8)
    var_10 = [var_0, var_1]
    var_11 = [var_0, var_1, var_2]
    var_12 = module_2.pset(var_11)
    var_13 = module_1.thaw(var_12)
    var_14 = [var_1, var_2]
    var_15 = [var_0, var_1, var_2]
    var_16 = False
    var_17 = {var_5: var_0, var_6: var_1}
    var_18 = module_0.pmap(var_17)
    var_19 = module_1.thaw(var_18, var_16)
    var_20 = module_1.thaw(var_0)
    assert var_20 == 1
    var_21 = 'hello'
    var_22 = module_1.thaw(var_21)
    assert var_22 == 'hello'
    var_23 = [var_0, var_1, var_2]
    var_24 = module_1.thaw(var_23)
    var_25 = {var_5: var_0, var_6: var_1}
    var_26 = module_1.thaw(var_25)
    var_27 = 'list'
    var_28 = 'set'
    var_29 = 'nested'
    var_30 = [var_0, var_1, var_2]
    var_31 = 4
    var_32 = 5
    var_33 = 6
    var_34 = [var_31, var_32, var_33]
    var_35 = module_2.pset(var_34)
    var_36 = 7
    var_37 = 8
    var_38 = [var_36, var_37]
    var_39 = [var_0, var_1, var_2]
    var_40 = {var_31, var_32, var_33}
    var_41 = [var_36, var_37]
    var_42 = {var_5: var_41}
    var_43 = {var_27: var_39, var_28: var_40, var_29: var_42}



# Parsed testcases at query #3
#--------------------------


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1
import pyrsistent._helpers as module_2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_1}
    var_6 = module_0.pmap(var_5)
    var_7 = 4
    var_8 = [var_2, var_7]
    var_9 = module_1.pset(var_8)
    var_10 = [var_0, var_6, var_9]
    var_11 = 'b'
    var_12 = {var_4: var_0, var_11: var_1}
    var_13 = module_0.pmap(var_12)
    var_14 = module_2.thaw(var_13)
    var_15 = [var_0, var_1]
    var_16 = [var_2, var_7]
    var_17 = module_1.pset(var_16)
    var_18 = [var_0, var_1, var_2]
    var_19 = module_1.pset(var_18)
    var_20 = module_2.thaw(var_19)
    var_21 = [var_1, var_2]
    var_22 = module_2.thaw(var_0)
    assert var_22 == 1
    var_23 = 'string'
    var_24 = module_2.thaw(var_23)
    assert var_24 == 'string'
    var_25 = [var_0, var_1, var_2]
    var_26 = False
    var_27 = module_2.thaw(var_25, var_26)
    var_28 = {var_4: var_0, var_11: var_1}
    var_29 = module_2.thaw(var_28, var_26)
    var_30 = [var_1, var_2]
    var_31 = [var_1, var_2]



# Parsed testcases at query #4
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 'a'
    var_7 = {var_6: var_0}
    var_8 = 'b'
    var_9 = {var_6: var_0, var_8: var_1}
    var_10 = module_0.pmap(var_9)
    var_11 = 'list'
    var_12 = 'dict'
    var_13 = [var_0, var_1, var_2]
    var_14 = {var_6: var_0, var_8: var_1}
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = [var_0, var_1, var_2, var_4]
    var_17 = 'c'
    var_18 = {var_6: var_0, var_8: var_1, var_17: var_2}
    var_19 = module_0.pmap(var_18)
    var_20 = 'y'
    var_21 = 20
    var_22 = {var_20: var_21}
    var_23 = 'x'
    var_24 = 10
    var_25 = {var_20: var_21, var_23: var_24}
    var_26 = module_0.pmap(var_25)
    var_27 = [var_0, var_1, var_2]



# Parsed testcases at query #5
#--------------------------


import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_1, var_2]
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_5: var_0, var_6: var_1}
    var_8 = module_0.pmap(var_7)
    var_9 = module_1.thaw(var_8)
    var_10 = [var_0, var_1]
    var_11 = [var_0, var_1, var_2]
    var_12 = module_2.pset(var_11)
    var_13 = module_1.thaw(var_12)
    var_14 = (var_0, var_1, var_2)
    var_15 = module_1.thaw(var_14)
    var_16 = [var_1, var_2]
    var_17 = [var_0, var_1, var_2]
    var_18 = False
    var_19 = {var_5: var_0, var_6: var_1}
    var_20 = module_0.pmap(var_19)
    var_21 = module_1.thaw(var_20, var_18)
    var_22 = [var_0, var_1, var_2]
    var_23 = module_2.pset(var_22)
    var_24 = module_1.thaw(var_23, var_18)
    var_25 = (var_0, var_1, var_2)
    var_26 = module_1.thaw(var_25, var_18)
    var_27 = 'list'
    var_28 = 'dict'
    var_29 = 'set'
    var_30 = 'tuple'
    var_31 = [var_0, var_1, var_2]
    var_32 = {var_5: var_0}
    var_33 = module_0.pmap(var_32)
    var_34 = [var_0, var_1]
    var_35 = module_2.pset(var_34)
    var_36 = (var_0, var_1)
    var_37 = [var_0, var_1, var_2]
    var_38 = {var_5: var_0}
    var_39 = {var_0, var_1}
    var_40 = (var_0, var_1)
    var_41 = {var_27: var_37, var_28: var_38, var_29: var_39, var_30: var_40}
    var_42 = module_1.thaw(var_0)
    assert var_42 == 1
    var_43 = 'string'
    var_44 = module_1.thaw(var_43)
    assert var_44 == 'string'
    var_45 = None
    var_46 = module_1.thaw(var_45)
    assert var_46 is None



# Parsed testcases at query #6
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 5
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = [var_1, var_2, var_3, var_5]
    var_7 = 'a'
    var_8 = {var_7: var_1}
    var_9 = 'b'
    var_10 = {var_7: var_1, var_9: var_2}
    var_11 = module_0.pmap(var_10)
    var_12 = 'list'
    var_13 = 'dict'
    var_14 = [var_1, var_2, var_3]
    var_15 = {var_7: var_1, var_9: var_2}
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = [var_1, var_2, var_3, var_5]
    var_18 = 'c'
    var_19 = {var_7: var_1, var_9: var_2, var_18: var_3}
    var_20 = module_0.pmap(var_19)
    var_21 = {var_7: var_1, var_9: var_2}
    var_22 = 'new_key'
    var_23 = 'new_value'
    var_24 = {var_7: var_1, var_9: var_2, var_22: var_23}
    var_25 = module_0.pmap(var_24)
    var_26 = 'original'
    var_27 = [var_26]
    var_28 = 'value'
    var_29 = {var_26: var_28}
    var_30 = 'existing'
    var_31 = 'kwarg'
    var_32 = {var_30: var_31}
    var_33 = 'modified'
    var_34 = [var_26, var_33]
    var_35 = {var_26: var_28, var_22: var_23}
    var_36 = module_0.pmap(var_35)
    var_37 = 'kwarg_key'
    var_38 = 'kwarg_value'
    var_39 = {var_30: var_31, var_37: var_38}
    var_40 = module_0.pmap(var_39)
    var_41 = [var_1, var_2, var_3]
    var_42 = (var_1, var_2, var_3)



# Parsed testcases at query #7
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 'a'
    var_7 = {var_6: var_0}
    var_8 = 'b'
    var_9 = {var_6: var_0, var_8: var_1}
    var_10 = module_0.pmap(var_9)
    var_11 = 'nested'
    var_12 = [var_0, var_1]
    var_13 = {var_11: var_12}
    var_14 = [var_0, var_1, var_0]
    var_15 = 'new_key'
    var_16 = 'new_value'
    var_17 = {var_6: var_0, var_8: var_1, var_15: var_16}
    var_18 = module_0.pmap(var_17)
    var_19 = (var_0, var_1)



# Parsed testcases at query #8
#--------------------------


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 'a'
    var_7 = {var_6: var_0}
    var_8 = 'b'
    var_9 = {var_6: var_0, var_8: var_1}
    var_10 = module_0.pmap(var_9)
    var_11 = {var_0, var_1}
    var_12 = {var_0, var_1, var_2}
    var_13 = module_1.pset(var_12)
    var_14 = (var_0, var_1, var_2)



# Parsed testcases at query #9
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 'nested'
    var_7 = 'value'
    var_8 = 5
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = 10
    var_12 = {var_7: var_11}
    var_13 = module_0.pmap(var_12)
    var_14 = {var_6: var_13}
    var_15 = module_0.pmap(var_14)
    var_16 = [var_0, var_1]
    var_17 = [var_2, var_4]
    var_18 = 99
    var_19 = [var_0, var_1, var_2, var_4, var_18]
    var_20 = 'key1'
    var_21 = 'value1'
    var_22 = {var_20: var_21}
    var_23 = 'new_key'
    var_24 = 'new_value'
    var_25 = {var_20: var_21, var_23: var_24}
    var_26 = module_0.pmap(var_25)



# Parsed testcases at query #10
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 'nested'
    var_7 = 'list'
    var_8 = 'value'
    var_9 = 'original'
    var_10 = {var_8: var_9}
    var_11 = [var_0, var_1, var_2]
    var_12 = {var_6: var_10, var_7: var_11}
    var_13 = 'modified'
    var_14 = {var_8: var_13}
    var_15 = module_0.pmap(var_14)
    var_16 = [var_0, var_1, var_2, var_4]
    var_17 = 'a'
    var_18 = 'b'
    var_19 = {var_17: var_0, var_18: var_1}
    var_20 = {var_17: var_1, var_18: var_1}
    var_21 = module_0.pmap(var_20)
    var_22 = [var_0, var_1]
    var_23 = {var_17: var_0}
    var_24 = {var_18: var_1}
    var_25 = [var_0, var_1, var_0]
    var_26 = 'new_key'
    var_27 = 'new_value'
    var_28 = {var_17: var_0, var_26: var_27}
    var_29 = module_0.pmap(var_28)
    var_30 = 'new_kwarg'
    var_31 = True
    var_32 = {var_18: var_1, var_30: var_31}
    var_33 = module_0.pmap(var_32)
    var_34 = (var_31, var_1, var_2)



# Parsed testcases at query #11
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_3]
    var_5 = module_0.freeze(var_4)
    var_6 = {var_1: var_2}
    var_7 = module_1.pmap(var_6)
    var_8 = [var_0, var_7]
    var_9 = []
    var_10 = (var_0, var_9)
    var_11 = module_0.freeze(var_10)
    var_12 = []
    var_13 = 2
    var_14 = [var_0, var_13]
    var_15 = set(var_14)
    var_16 = module_0.freeze(var_15)
    var_17 = [var_0, var_13]
    var_18 = module_2.pset(var_17)
    var_19 = [var_0, var_13]
    var_20 = {var_1: var_19}
    var_21 = module_0.freeze(var_20)
    var_22 = [var_0, var_13]
    var_23 = [var_0, var_13]
    var_24 = 4
    var_25 = [var_2, var_24]
    var_26 = {var_1: var_25}
    var_27 = [var_0, var_26]
    var_28 = False
    var_29 = module_0.freeze(var_27, var_28)
    var_30 = [var_2, var_24]
    var_31 = {var_1: var_30}
    var_32 = [var_0, var_31]
    var_33 = [var_0, var_13, var_2]
    var_34 = {var_1: var_0}
    var_35 = module_1.pmap(var_34)
    var_36 = module_0.freeze(var_35)
    var_37 = [var_0, var_13]
    var_38 = module_2.pset(var_37)
    var_39 = module_0.freeze(var_38)
    var_40 = 42
    var_41 = module_0.freeze(var_40)
    assert var_41 == 42



# Parsed testcases at query #12
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 'nested'
    var_7 = [var_0, var_1]
    var_8 = {var_6: var_7}
    var_9 = 5
    var_10 = [var_0, var_1, var_9]
    var_11 = 'a'
    var_12 = {var_11: var_0}
    var_13 = 'b'
    var_14 = {var_13: var_1}
    var_15 = {var_11: var_0, var_13: var_1}
    var_16 = module_0.pmap(var_15)
    var_17 = 'old_key'
    var_18 = 'old_value'
    var_19 = {var_17: var_18}
    var_20 = 'new_key'
    var_21 = 'new_value'
    var_22 = {var_17: var_18, var_20: var_21}
    var_23 = module_0.pmap(var_22)
    var_24 = [var_0, var_1]
    var_25 = 'key'
    var_26 = 'value'
    var_27 = {var_25: var_26}
    var_28 = [var_0, var_1, var_2]
    var_29 = 'modified'
    var_30 = True
    var_31 = {var_25: var_26, var_29: var_30}
    var_32 = module_0.pmap(var_31)
    var_33 = [var_30, var_1, var_2]



# Parsed testcases at query #13
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_3]
    var_5 = module_0.freeze(var_4)
    var_6 = {var_1: var_2}
    var_7 = module_1.pmap(var_6)
    var_8 = [var_0, var_7]
    var_9 = []
    var_10 = (var_0, var_9)
    var_11 = module_0.freeze(var_10)
    var_12 = []
    var_13 = 2
    var_14 = [var_0, var_13]
    var_15 = set(var_14)
    var_16 = module_0.freeze(var_15)
    var_17 = [var_0, var_13]
    var_18 = module_2.pset(var_17)
    var_19 = 'b'
    var_20 = [var_13, var_2]
    var_21 = {var_1: var_0, var_19: var_20}
    var_22 = module_0.freeze(var_21)
    var_23 = [var_13, var_2]
    var_24 = [var_13, var_2]
    var_25 = {var_1: var_0, var_19: var_24}
    var_26 = [var_13, var_2]
    var_27 = {var_1: var_2}
    var_28 = [var_0, var_27]
    var_29 = False
    var_30 = module_0.freeze(var_28, var_29)
    var_31 = {var_1: var_2}
    var_32 = [var_0, var_31]
    var_33 = {var_1: var_2}
    var_34 = [var_0, var_33]
    var_35 = {var_1: var_2}
    var_36 = module_1.pmap(var_35)
    var_37 = [var_0, var_36]
    var_38 = [var_13, var_2]
    var_39 = {var_1: var_0, var_19: var_38}
    var_40 = module_1.pmap(var_39)
    var_41 = module_0.freeze(var_40)
    var_42 = [var_13, var_2]
    var_43 = [var_0, var_13]
    var_44 = module_2.pset(var_43)
    var_45 = module_0.freeze(var_44)
    var_46 = [var_0, var_13]
    var_47 = module_2.pset(var_46)
    var_48 = 42
    var_49 = module_0.freeze(var_48)
    assert var_49 == 42



# Parsed testcases at query #14
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1
import pyrsistent._pmap as module_2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = set(var_2)
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1]
    var_6 = module_1.pset(var_5)
    var_7 = 'a'
    var_8 = 3
    var_9 = {var_7: var_8}
    var_10 = [var_0, var_9]
    var_11 = module_0.freeze(var_10)
    var_12 = {var_7: var_8}
    var_13 = module_2.pmap(var_12)
    var_14 = [var_0, var_13]
    var_15 = []
    var_16 = (var_0, var_15)
    var_17 = module_0.freeze(var_16)
    var_18 = []
    var_19 = 'b'
    var_20 = [var_1, var_8]
    var_21 = {var_7: var_0, var_19: var_20}
    var_22 = module_0.freeze(var_21)
    var_23 = [var_1, var_8]
    var_24 = [var_1, var_8]
    var_25 = {var_7: var_8}
    var_26 = [var_0, var_1, var_25]
    var_27 = {var_7: var_8}
    var_28 = module_2.pmap(var_27)
    var_29 = [var_0, var_1, var_28]
    var_30 = [var_1, var_8]
    var_31 = {var_7: var_0, var_19: var_30}
    var_32 = module_2.pmap(var_31)
    var_33 = module_0.freeze(var_32)
    var_34 = [var_1, var_8]
    var_35 = [var_0, var_1, var_8]
    var_36 = module_1.pset(var_35)
    var_37 = module_0.freeze(var_36)
    var_38 = [var_0, var_1, var_8]
    var_39 = module_1.pset(var_38)
    var_40 = {var_7: var_8}
    var_41 = [var_0, var_40]
    var_42 = False
    var_43 = module_0.freeze(var_41, var_42)
    var_44 = {var_7: var_8}
    var_45 = [var_0, var_44]
    var_46 = module_0.freeze(var_0)
    assert var_46 == 1
    var_47 = 'hello'
    var_48 = module_0.freeze(var_47)
    assert var_48 == 'hello'
    var_49 = (var_0, var_1, var_8)
    var_50 = module_0.freeze(var_49)



# Parsed testcases at query #15
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = [var_1, var_2]
    var_7 = [var_0, var_6]
    var_8 = module_0.freeze(var_7)
    var_9 = [var_1, var_2]
    var_10 = 'a'
    var_11 = 'b'
    var_12 = {var_10: var_0, var_11: var_1}
    var_13 = module_0.freeze(var_12)
    var_14 = {var_10: var_0, var_11: var_1}
    var_15 = module_1.pmap(var_14)
    var_16 = {var_11: var_1}
    var_17 = {var_10: var_16}
    var_18 = module_0.freeze(var_17)
    var_19 = {var_11: var_1}
    var_20 = module_1.pmap(var_19)
    var_21 = {var_10: var_20}
    var_22 = module_1.pmap(var_21)
    var_23 = {var_10: var_0, var_11: var_1}
    var_24 = {var_10: var_0, var_11: var_1}
    var_25 = module_1.pmap(var_24)
    var_26 = {var_0, var_1, var_2}
    var_27 = module_0.freeze(var_26)
    var_28 = {var_0, var_1, var_2}
    var_29 = module_2.pset(var_28)
    var_30 = (var_0, var_1, var_2)
    var_31 = module_0.freeze(var_30)
    var_32 = [var_1, var_2]
    var_33 = (var_0, var_32)
    var_34 = module_0.freeze(var_33)
    var_35 = [var_1, var_2]
    var_36 = [var_1, var_2]
    var_37 = [var_0, var_36]
    var_38 = False
    var_39 = module_0.freeze(var_37, var_38)
    var_40 = [var_1, var_2]
    var_41 = [var_0, var_40]
    var_42 = [var_0, var_1, var_2]
    var_43 = {var_10: var_0, var_11: var_1}
    var_44 = module_1.pmap(var_43)
    var_45 = module_0.freeze(var_44)
    var_46 = {var_0, var_1, var_2}
    var_47 = module_2.pset(var_46)
    var_48 = module_0.freeze(var_47)
    var_49 = 'c'
    var_50 = [var_0, var_1]
    var_51 = 4
    var_52 = (var_2, var_51)
    var_53 = 5
    var_54 = 6
    var_55 = {var_53, var_54}
    var_56 = {var_10: var_50, var_11: var_52, var_49: var_55}
    var_57 = [var_0, var_1]
    var_58 = (var_2, var_51)
    var_59 = {var_53, var_54}
    var_60 = module_2.pset(var_59)
    var_61 = module_0.freeze(var_56)



# Parsed testcases at query #16
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 'a'
    var_7 = {var_6: var_0}
    var_8 = 'b'
    var_9 = {var_6: var_0, var_8: var_1}
    var_10 = module_0.pmap(var_9)
    var_11 = 'list'
    var_12 = 'dict'
    var_13 = [var_0, var_1, var_2]
    var_14 = {var_6: var_0}
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = [var_0, var_1, var_2, var_4]
    var_17 = 'c'
    var_18 = {var_6: var_0, var_17: var_2}
    var_19 = module_0.pmap(var_18)
    var_20 = 'y'
    var_21 = 20
    var_22 = {var_20: var_21}
    var_23 = 'x'
    var_24 = 10
    var_25 = {var_20: var_21, var_23: var_24}
    var_26 = module_0.pmap(var_25)
    var_27 = [var_0, var_1]
    var_28 = {var_6: var_0}
    var_29 = {var_8: var_1}
    var_30 = 'kwargs'
    var_31 = [var_0, var_1, var_0]
    var_32 = 'new'
    var_33 = 'value'
    var_34 = {var_6: var_0, var_32: var_33}
    var_35 = module_0.pmap(var_34)
    var_36 = 'new_kwarg'
    var_37 = 'kwarg_value'
    var_38 = {var_8: var_1, var_36: var_37}
    var_39 = module_0.pmap(var_38)



# Parsed testcases at query #17
#--------------------------


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 'a'
    var_7 = {var_6: var_0}
    var_8 = 'b'
    var_9 = {var_6: var_0, var_8: var_1}
    var_10 = module_0.pmap(var_9)
    var_11 = 'list'
    var_12 = 'set'
    var_13 = [var_0, var_1, var_2]
    var_14 = {var_0, var_1, var_2}
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = [var_0, var_1, var_2, var_4]
    var_17 = {var_0, var_1, var_2, var_4}
    var_18 = module_1.pset(var_17)



# Parsed testcases at query #18
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 'a'
    var_7 = {var_6: var_0}
    var_8 = 'b'
    var_9 = {var_6: var_0, var_8: var_1}
    var_10 = module_0.pmap(var_9)
    var_11 = 'list'
    var_12 = 'dict'
    var_13 = [var_0, var_1]
    var_14 = 'key'
    var_15 = 'old'
    var_16 = {var_14: var_15}
    var_17 = {var_11: var_13, var_12: var_16}
    var_18 = [var_0, var_1, var_0]
    var_19 = 'value'
    var_20 = {var_14: var_19}
    var_21 = module_0.pmap(var_20)



