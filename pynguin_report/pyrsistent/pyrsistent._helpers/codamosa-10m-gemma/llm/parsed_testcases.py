####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 1
    var_1 = module_0.freeze(var_0)
    assert var_1 == 1
    var_2 = 'string'
    var_3 = module_0.freeze(var_2)
    assert var_3 == 'string'
    var_4 = True
    var_5 = module_0.freeze(var_4)
    assert var_5 is True
    var_6 = None
    var_7 = module_0.freeze(var_6)
    assert var_7 is None
    var_8 = 2
    var_9 = 3
    var_10 = [var_4, var_8, var_9]
    var_11 = module_0.freeze(var_10)
    var_12 = [var_4, var_8, var_9]
    var_13 = [var_8, var_9]
    var_14 = [var_4, var_13]
    var_15 = module_0.freeze(var_14)
    var_16 = [var_8, var_9]
    var_17 = 'a'
    var_18 = {var_17: var_4}
    var_19 = [var_18]
    var_20 = module_0.freeze(var_19)
    var_21 = {var_17: var_4}
    var_22 = module_1.pmap(var_21)
    var_23 = [var_22]
    var_24 = 'b'
    var_25 = {var_17: var_4, var_24: var_8}
    var_26 = module_0.freeze(var_25)
    var_27 = {var_17: var_4, var_24: var_8}
    var_28 = module_1.pmap(var_27)
    var_29 = [var_4, var_8]
    var_30 = 'c'
    var_31 = {var_30: var_9}
    var_32 = {var_17: var_29, var_24: var_31}
    var_33 = module_0.freeze(var_32)
    var_34 = [var_4, var_8]
    var_35 = {var_30: var_9}
    var_36 = module_1.pmap(var_35)
    var_37 = [var_4]
    var_38 = [var_8]
    var_39 = {var_17: var_37, var_24: var_38}
    var_40 = [var_4]
    var_41 = [var_8]
    var_42 = {var_4, var_8, var_9}
    var_43 = module_0.freeze(var_42)
    var_44 = [var_4, var_8, var_9]
    var_45 = module_2.pset(var_44)
    var_46 = {var_4, var_8}
    var_47 = module_0.freeze(var_46)
    var_48 = (var_4, var_8)
    var_49 = module_0.freeze(var_48)
    var_50 = [var_8, var_9]
    var_51 = (var_4, var_50)
    var_52 = module_0.freeze(var_51)
    var_53 = [var_8, var_9]
    var_54 = {var_17: var_8}
    var_55 = (var_4, var_54)
    var_56 = module_0.freeze(var_55)
    var_57 = {var_17: var_8}
    var_58 = module_1.pmap(var_57)
    var_59 = (var_4, var_58)
    var_60 = [var_4, var_8]
    var_61 = {var_17: var_60}
    var_62 = module_1.pmap(var_61)
    var_63 = module_0.freeze(var_62)
    var_64 = [var_4, var_8]
    var_65 = False
    var_66 = module_0.freeze(var_62, var_65)
    var_67 = 'd'
    var_68 = 4
    var_69 = {var_67: var_68}
    var_70 = [var_9, var_69]
    var_71 = (var_8, var_70)
    var_72 = {var_17: var_71}
    var_73 = [var_4, var_72]



# Parsed testcases at query #2
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 1
    var_1 = module_0.freeze(var_0)
    assert var_1 == 1
    var_2 = 'string'
    var_3 = module_0.freeze(var_2)
    assert var_3 == 'string'
    var_4 = True
    var_5 = module_0.freeze(var_4)
    assert var_5 is True
    var_6 = None
    var_7 = module_0.freeze(var_6)
    assert var_7 is None
    var_8 = 2
    var_9 = 3
    var_10 = [var_4, var_8, var_9]
    var_11 = module_0.freeze(var_10)
    var_12 = [var_4, var_8, var_9]
    var_13 = [var_8, var_9]
    var_14 = [var_4, var_13]
    var_15 = module_0.freeze(var_14)
    var_16 = [var_8, var_9]
    var_17 = 'a'
    var_18 = {var_17: var_4}
    var_19 = [var_18]
    var_20 = module_0.freeze(var_19)
    var_21 = {var_17: var_4}
    var_22 = module_1.pmap(var_21)
    var_23 = [var_22]
    var_24 = 'b'
    var_25 = {var_17: var_4, var_24: var_8}
    var_26 = module_0.freeze(var_25)
    var_27 = {var_17: var_4, var_24: var_8}
    var_28 = module_1.pmap(var_27)
    var_29 = [var_4, var_8]
    var_30 = 'c'
    var_31 = {var_30: var_9}
    var_32 = {var_17: var_29, var_24: var_31}
    var_33 = module_0.freeze(var_32)
    var_34 = [var_4, var_8]
    var_35 = {var_30: var_9}
    var_36 = module_1.pmap(var_35)
    var_37 = [var_4]
    var_38 = [var_8]
    var_39 = {var_17: var_37, var_24: var_38}
    var_40 = [var_4]
    var_41 = [var_8]
    var_42 = {var_4, var_8, var_9}
    var_43 = module_0.freeze(var_42)
    var_44 = [var_4, var_8, var_9]
    var_45 = module_2.pset(var_44)
    var_46 = [var_8]
    var_47 = (var_4, var_46)
    var_48 = {var_47}
    var_49 = module_0.freeze(var_48)
    var_50 = [var_8]
    var_51 = (var_4, var_50)
    var_52 = {var_51}
    var_53 = module_2.pset(var_52)
    var_54 = (var_4, var_8)
    var_55 = module_0.freeze(var_54)
    var_56 = [var_8, var_9]
    var_57 = (var_4, var_56)
    var_58 = module_0.freeze(var_57)
    var_59 = [var_8, var_9]
    var_60 = {var_17: var_8}
    var_61 = (var_4, var_60)
    var_62 = module_0.freeze(var_61)
    var_63 = {var_17: var_8}
    var_64 = module_1.pmap(var_63)
    var_65 = (var_4, var_64)
    var_66 = 'x'
    var_67 = 10
    var_68 = {var_66: var_67}
    var_69 = module_1.pmap(var_68)
    var_70 = [var_4, var_69]
    var_71 = 'y'
    var_72 = 5
    var_73 = [var_72]
    var_74 = [var_4, var_8]
    var_75 = False
    var_76 = {var_17: var_4}
    var_77 = module_1.pmap(var_76)
    var_78 = module_0.freeze(var_77, var_75)
    var_79 = 'key'
    var_80 = 'deep'
    var_81 = True
    var_82 = {var_80: var_81}
    var_83 = [var_4, var_8, var_82]
    var_84 = {var_79: var_83}
    var_85 = {var_67: var_67}
    var_86 = (var_67, var_85)
    var_87 = {var_81, var_8, var_9}
    var_88 = [var_84, var_86, var_87]



# Parsed testcases at query #3
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1
import pyrsistent._pmap as module_2

def test_case_0():
    var_0 = 1
    var_1 = module_0.thaw(var_0)
    assert var_1 == 1
    var_2 = 'string'
    var_3 = module_0.thaw(var_2)
    assert var_3 == 'string'
    var_4 = True
    var_5 = module_0.thaw(var_4)
    assert var_5 is True
    var_6 = None
    var_7 = module_0.thaw(var_6)
    assert var_7 is None
    var_8 = 2
    var_9 = 3
    var_10 = [var_4, var_8, var_9]
    var_11 = module_1.pset(var_10)
    var_12 = module_0.thaw(var_11)
    var_13 = {var_4, var_8, var_9}
    var_14 = module_0.thaw(var_13)
    var_15 = [var_4, var_8]
    var_16 = [var_4, var_8]
    var_17 = module_0.thaw(var_16)
    var_18 = 'a'
    var_19 = 'b'
    var_20 = {var_18: var_4, var_19: var_8}
    var_21 = module_2.pmap(var_20)
    var_22 = module_0.thaw(var_21)
    var_23 = {var_18: var_4, var_19: var_8}
    var_24 = module_0.thaw(var_23)
    var_25 = [var_8]
    var_26 = {var_18: var_8}
    var_27 = module_2.pmap(var_26)
    var_28 = (var_4, var_27)
    var_29 = module_0.thaw(var_28)
    var_30 = 'key'
    var_31 = 'list'
    var_32 = [var_4, var_8]
    var_33 = module_1.pset(var_32)
    var_34 = 4
    var_35 = [var_9, var_34]
    var_36 = 'simple'
    var_37 = 5
    var_38 = {var_36: var_37}
    var_39 = {var_4, var_8}
    var_40 = [var_9, var_34]
    var_41 = {var_30: var_39, var_31: var_40}
    var_42 = {var_36: var_37}
    var_43 = [var_41, var_42]
    var_44 = [var_4]
    var_45 = False
    var_46 = [var_4]
    var_47 = [var_4, var_8]
    var_48 = {var_18: var_47}



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
    var_8 = 'c'
    var_9 = 'extra'
    var_10 = 'extra_key'
    var_11 = {var_6: var_0, var_9: var_1, var_10: var_8}
    var_12 = module_0.pmap(var_11)
    var_13 = 'inner'
    var_14 = 10
    var_15 = 20
    var_16 = [var_14, var_15]
    var_17 = {var_13: var_16}
    var_18 = 30
    var_19 = [var_14, var_15, var_18]
    var_20 = 'key'
    var_21 = 'value'
    var_22 = {var_20: var_21}
    var_23 = {var_20: var_21}
    var_24 = module_0.pmap(var_23)
    var_25 = 5
    var_26 = 'string'
    var_27 = True



# Parsed testcases at query #5
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = [var_0, var_1]
    var_6 = {var_3: var_0}
    var_7 = [var_0, var_1]
    var_8 = [var_0, var_1]
    var_9 = 'key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = 0
    var_13 = 3
    var_14 = [var_1, var_13]
    var_15 = {var_3: var_14}
    var_16 = 4
    var_17 = 5
    var_18 = (var_16, var_17)
    var_19 = [var_0, var_15, var_18]
    var_20 = [var_1]
    var_21 = 'x'
    var_22 = {var_21: var_13}
    var_23 = [var_1]
    var_24 = {var_21: var_13}
    var_25 = module_0.pmap(var_24)



# Parsed testcases at query #6
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 1
    var_1 = module_0.freeze(var_0)
    assert var_1 == 1
    var_2 = 'string'
    var_3 = module_0.freeze(var_2)
    assert var_3 == 'string'
    var_4 = True
    var_5 = module_0.freeze(var_4)
    assert var_5 is True
    var_6 = None
    var_7 = module_0.freeze(var_6)
    assert var_7 is None
    var_8 = 2
    var_9 = 3
    var_10 = [var_4, var_8, var_9]
    var_11 = module_0.freeze(var_10)
    var_12 = [var_4, var_8, var_9]
    var_13 = [var_8, var_9]
    var_14 = [var_4, var_13]
    var_15 = module_0.freeze(var_14)
    var_16 = [var_8, var_9]
    var_17 = 'a'
    var_18 = {var_17: var_4}
    var_19 = [var_18]
    var_20 = module_0.freeze(var_19)
    var_21 = {var_17: var_4}
    var_22 = module_1.pmap(var_21)
    var_23 = [var_22]
    var_24 = 'b'
    var_25 = {var_17: var_4, var_24: var_8}
    var_26 = module_0.freeze(var_25)
    var_27 = {var_17: var_4, var_24: var_8}
    var_28 = module_1.pmap(var_27)
    var_29 = [var_4, var_8]
    var_30 = 'c'
    var_31 = {var_30: var_9}
    var_32 = {var_17: var_29, var_24: var_31}
    var_33 = module_0.freeze(var_32)
    var_34 = [var_4, var_8]
    var_35 = {var_30: var_9}
    var_36 = module_1.pmap(var_35)
    var_37 = [var_4]
    var_38 = [var_8]
    var_39 = {var_17: var_37, var_24: var_38}
    var_40 = [var_4]
    var_41 = [var_8]
    var_42 = {var_4, var_8, var_9}
    var_43 = module_0.freeze(var_42)
    var_44 = [var_4, var_8, var_9]
    var_45 = module_2.pset(var_44)
    var_46 = [var_8]
    var_47 = (var_4, var_46)
    var_48 = {var_47}
    var_49 = module_0.freeze(var_48)
    var_50 = [var_8]
    var_51 = (var_4, var_50)
    var_52 = {var_51}
    var_53 = module_2.pset(var_52)
    var_54 = (var_4, var_8)
    var_55 = module_0.freeze(var_54)
    var_56 = [var_8, var_9]
    var_57 = (var_4, var_56)
    var_58 = module_0.freeze(var_57)
    var_59 = [var_8, var_9]
    var_60 = {var_17: var_8}
    var_61 = (var_4, var_60)
    var_62 = module_0.freeze(var_61)
    var_63 = {var_17: var_8}
    var_64 = module_1.pmap(var_63)
    var_65 = (var_4, var_64)
    var_66 = [var_4, var_8]
    var_67 = {var_17: var_4}
    var_68 = module_1.pmap(var_67)
    var_69 = module_0.freeze(var_68)
    var_70 = False
    var_71 = module_0.freeze(var_68, var_70)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = [var_0, var_1, var_3]



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = 3
    var_6 = [var_0, var_1, var_5]
    var_7 = [var_0, var_1, var_5]
    var_8 = 'key'
    var_9 = [var_1, var_5]
    var_10 = {var_8: var_9}
    var_11 = 4
    var_12 = 5
    var_13 = (var_11, var_12)
    var_14 = [var_0, var_10, var_13]
    var_15 = 0
    var_16 = [var_0]
    var_17 = 'x'
    var_18 = {var_17: var_1}
    var_19 = 'b'
    var_20 = 10
    var_21 = 20
    var_22 = [var_20, var_21]
    var_23 = 'list'
    var_24 = 'set'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = [var_0, var_1]
    var_4 = [var_0, var_1]
    var_5 = 3
    var_6 = [var_0, var_1, var_5]
    var_7 = 'a'
    var_8 = {var_7: var_0}
    var_9 = [var_0, var_1, var_5]
    var_10 = [var_0, var_1, var_5]
    var_11 = 'c'
    var_12 = 'b'
    var_13 = {var_12: var_5}
    var_14 = [var_0, var_1, var_13]
    var_15 = 4
    var_16 = 5
    var_17 = (var_15, var_16)
    var_18 = {var_7: var_14, var_11: var_17}
    var_19 = [var_1]
    var_20 = 'x'
    var_21 = 10
    var_22 = {var_20: var_21}
    var_23 = [var_1]
    var_24 = {var_20: var_21}



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
    var_6 = 'a'
    var_7 = {var_6: var_0}
    var_8 = 'new_key'
    var_9 = 'new_val'
    var_10 = {var_6: var_0, var_8: var_9}
    var_11 = module_0.pmap(var_10)
    var_12 = [var_1, var_2]
    var_13 = {var_6: var_12}
    var_14 = [var_0, var_13]
    var_15 = [var_0]
    var_16 = 'x'
    var_17 = 10
    var_18 = {var_16: var_17}
    var_19 = 'b'
    var_20 = [var_0]
    var_21 = {var_16: var_17}
    var_22 = module_0.pmap(var_21)
    var_23 = [var_0, var_1, var_2]



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = 3
    var_6 = [var_0, var_1, var_5]
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = [var_0, var_1]
    var_11 = 'inner'
    var_12 = [var_1, var_5]
    var_13 = {var_11: var_12}
    var_14 = [var_0, var_13]
    var_15 = [var_0, var_1, var_5]
    var_16 = 4
    var_17 = [var_0, var_1, var_5, var_16]
    var_18 = [var_1]
    var_19 = (var_0, var_18)
    var_20 = {var_0, var_1, var_5}



# Parsed testcases at query #11
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = 3
    var_6 = [var_0, var_1, var_5]
    var_7 = 4
    var_8 = [var_0, var_1, var_5, var_7]
    var_9 = [var_1, var_5]
    var_10 = {var_3: var_9}
    var_11 = 5
    var_12 = (var_7, var_11)
    var_13 = [var_0, var_10, var_12]
    var_14 = [var_1]
    var_15 = [var_1]
    var_16 = 'x'
    var_17 = {var_16: var_1}
    var_18 = module_0.pmap(var_17)
    var_19 = [var_0, var_18]



# Parsed testcases at query #12
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = [var_0, var_1]
    var_6 = 'a'
    var_7 = {var_6: var_0}
    var_8 = 'c'
    var_9 = 'b'
    var_10 = {var_9: var_1}
    var_11 = [var_0, var_10]
    var_12 = (var_2, var_3)
    var_13 = {var_6: var_11, var_8: var_12}
    var_14 = {var_9: var_1}
    var_15 = module_0.pmap(var_14)
    var_16 = [var_0, var_15]
    var_17 = (var_2, var_3)
    var_18 = [var_0, var_1]
    var_19 = [var_0, var_1]



# Parsed testcases at query #13
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
    var_8 = [var_0, var_1]
    var_9 = [var_0]
    var_10 = 'x'
    var_11 = 10
    var_12 = {var_10: var_11}
    var_13 = [var_0]
    var_14 = {var_10: var_11}
    var_15 = module_0.pmap(var_14)
    var_16 = [var_0, var_1]
    var_17 = {var_6: var_0}
    var_18 = [var_0, var_1]
    var_19 = 'key'
    var_20 = 'val'
    var_21 = {var_19: var_20}
    var_22 = [var_18, var_21]
    var_23 = [var_0, var_1]
    var_24 = 123
    var_25 = 'string'



# Parsed testcases at query #14
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 1
    var_1 = module_0.freeze(var_0)
    assert var_1 == 1
    var_2 = 'string'
    var_3 = module_0.freeze(var_2)
    assert var_3 == 'string'
    var_4 = True
    var_5 = module_0.freeze(var_4)
    assert var_5 is True
    var_6 = None
    var_7 = module_0.freeze(var_6)
    assert var_7 is None
    var_8 = 2
    var_9 = 3
    var_10 = [var_4, var_8, var_9]
    var_11 = module_0.freeze(var_10)
    var_12 = [var_4, var_8, var_9]
    var_13 = [var_8, var_9]
    var_14 = [var_4, var_13]
    var_15 = module_0.freeze(var_14)
    var_16 = [var_8, var_9]
    var_17 = 'a'
    var_18 = {var_17: var_8}
    var_19 = [var_4, var_18]
    var_20 = module_0.freeze(var_19)
    var_21 = {var_17: var_8}
    var_22 = module_1.pmap(var_21)
    var_23 = [var_4, var_22]
    var_24 = 'b'
    var_25 = {var_17: var_4, var_24: var_8}
    var_26 = module_0.freeze(var_25)
    var_27 = {var_17: var_4, var_24: var_8}
    var_28 = module_1.pmap(var_27)
    var_29 = [var_4, var_8]
    var_30 = 'c'
    var_31 = {var_30: var_9}
    var_32 = {var_17: var_29, var_24: var_31}
    var_33 = module_0.freeze(var_32)
    var_34 = [var_4, var_8]
    var_35 = {var_30: var_9}
    var_36 = module_1.pmap(var_35)
    var_37 = [var_4]
    var_38 = [var_8]
    var_39 = {var_17: var_37, var_24: var_38}
    var_40 = [var_4]
    var_41 = [var_8]
    var_42 = {var_4, var_8, var_9}
    var_43 = module_0.freeze(var_42)
    var_44 = [var_4, var_8, var_9]
    var_45 = module_2.pset(var_44)
    var_46 = [var_8]
    var_47 = (var_4, var_46)
    var_48 = {var_47}
    var_49 = module_0.freeze(var_48)
    var_50 = [var_8]
    var_51 = (var_4, var_50)
    var_52 = {var_51}
    var_53 = module_2.pset(var_52)
    var_54 = (var_4, var_8)
    var_55 = module_0.freeze(var_54)
    var_56 = [var_8, var_9]
    var_57 = (var_4, var_56)
    var_58 = module_0.freeze(var_57)
    var_59 = [var_8, var_9]
    var_60 = [var_8]
    var_61 = {var_17: var_60}
    var_62 = (var_4, var_61)
    var_63 = module_0.freeze(var_62)
    var_64 = [var_8]
    var_65 = {var_17: var_8}
    var_66 = module_1.pmap(var_65)
    var_67 = [var_4, var_66]
    var_68 = [var_4]
    var_69 = [var_4]
    var_70 = {var_17: var_69}
    var_71 = module_1.pmap(var_70)
    var_72 = False
    var_73 = module_0.freeze(var_71, var_72)



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = [var_0, var_0]
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_2, var_3]
    var_5 = 'a'
    var_6 = 'c'
    var_7 = 'b'
    var_8 = {var_7: var_2}
    var_9 = [var_0, var_8]
    var_10 = 4
    var_11 = (var_3, var_10)
    var_12 = {var_5: var_9, var_6: var_11}
    var_13 = 'inner'
    var_14 = [var_0]
    var_15 = {var_13: var_14}
    var_16 = [var_0, var_2]
    var_17 = [var_2, var_3]
    var_18 = {var_5: var_17}
    var_19 = 5
    var_20 = {var_7: var_19}
    var_21 = (var_10, var_20)
    var_22 = [var_0, var_18, var_21]



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 1
    var_1 = module_0.freeze(var_0)
    assert var_1 == 1
    var_2 = 'string'
    var_3 = module_0.freeze(var_2)
    assert var_3 == 'string'
    var_4 = True
    var_5 = module_0.freeze(var_4)
    assert var_5 is True
    var_6 = None
    var_7 = module_0.freeze(var_6)
    assert var_7 is None
    var_8 = 2
    var_9 = 3
    var_10 = [var_4, var_8, var_9]
    var_11 = module_0.freeze(var_10)
    var_12 = [var_8, var_9]
    var_13 = [var_4, var_12]
    var_14 = module_0.freeze(var_13)
    var_15 = [var_8, var_9]
    var_16 = 'a'
    var_17 = {var_16: var_8}
    var_18 = [var_4, var_17]
    var_19 = module_0.freeze(var_18)
    var_20 = {var_16: var_8}
    var_21 = module_1.pmap(var_20)
    var_22 = [var_4, var_21]
    var_23 = 'b'
    var_24 = {var_16: var_4, var_23: var_8}
    var_25 = module_0.freeze(var_24)
    var_26 = [var_4, var_8]
    var_27 = 'c'
    var_28 = {var_27: var_9}
    var_29 = {var_16: var_26, var_23: var_28}
    var_30 = module_0.freeze(var_29)
    var_31 = [var_4, var_8]
    var_32 = {var_27: var_9}
    var_33 = module_1.pmap(var_32)
    var_34 = [var_4]
    var_35 = [var_8]
    var_36 = {var_16: var_34, var_23: var_35}
    var_37 = [var_4]
    var_38 = [var_8]
    var_39 = {var_4, var_8, var_9}
    var_40 = module_0.freeze(var_39)
    var_41 = {var_4, var_8}
    var_42 = module_0.freeze(var_41)
    var_43 = [var_4, var_8]
    var_44 = module_2.pset(var_43)
    var_45 = (var_4, var_8)
    var_46 = module_0.freeze(var_45)
    var_47 = [var_8, var_9]
    var_48 = (var_4, var_47)
    var_49 = module_0.freeze(var_48)
    var_50 = [var_8, var_9]
    var_51 = [var_4, var_8]
    var_52 = {var_16: var_51}
    var_53 = module_1.pmap(var_52)
    var_54 = False
    var_55 = module_0.freeze(var_53, var_54)
    var_56 = 'key'
    var_57 = 4
    var_58 = (var_9, var_57)
    var_59 = [var_4, var_8, var_58]
    var_60 = {var_56: var_59}
    var_61 = {var_4, var_8, var_9}
    var_62 = 5
    var_63 = '6'
    var_64 = 7
    var_65 = {var_63: var_64}
    var_66 = (var_62, var_65)
    var_67 = [var_60, var_61, var_66]
    var_68 = (var_9, var_57)
    var_69 = [var_4, var_8, var_68]
    var_70 = {var_4, var_8, var_9}
    var_71 = module_2.pset(var_70)
    var_72 = {var_63: var_64}
    var_73 = module_1.pmap(var_72)
    var_74 = (var_62, var_73)
    var_75 = module_0.freeze(var_67)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = [var_0, var_1, var_3]
    var_5 = 'a'
    var_6 = {var_5: var_0}
    var_7 = 'b'
    var_8 = {var_5: var_0, var_7: var_1}
    var_9 = module_0.pmap(var_8)



# Parsed testcases at query #2
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1
import pyrsistent._pmap as module_2

def test_case_0():
    var_0 = 1
    var_1 = module_0.thaw(var_0)
    assert var_1 == 1
    var_2 = 'string'
    var_3 = module_0.thaw(var_2)
    assert var_3 == 'string'
    var_4 = True
    var_5 = module_0.thaw(var_4)
    assert var_5 is True
    var_6 = None
    var_7 = module_0.thaw(var_6)
    assert var_7 is None
    var_8 = 2
    var_9 = 3
    var_10 = [var_4, var_8, var_9]
    var_11 = module_1.pset(var_10)
    var_12 = module_0.thaw(var_11)
    var_13 = module_0.thaw(var_11)
    var_14 = 4
    var_15 = [var_9, var_14]
    var_16 = [var_4, var_8, var_15]
    var_17 = 'a'
    var_18 = 'b'
    var_19 = 'c'
    var_20 = {var_19: var_8}
    var_21 = {var_17: var_4, var_18: var_20}
    var_22 = module_2.pmap(var_21)
    var_23 = module_0.thaw(var_22)
    var_24 = module_0.thaw(var_22)
    var_25 = [var_4, var_8]
    var_26 = [var_9, var_14]
    var_27 = tuple(var_26)
    var_28 = [var_4, var_8]
    var_29 = {var_17: var_28}
    var_30 = (var_9, var_14)
    var_31 = [var_29, var_30]
    var_32 = [var_4]
    var_33 = {var_17: var_8}
    var_34 = module_2.pmap(var_33)
    var_35 = [var_4]
    var_36 = False
    var_37 = [var_4]
    var_38 = [var_4]
    var_39 = [var_4]
    var_40 = [var_4]
    var_41 = [var_4]



# Parsed testcases at query #3
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1
import pyrsistent._pmap as module_2

def test_case_0():
    var_0 = 1
    var_1 = module_0.thaw(var_0)
    assert var_1 == 1
    var_2 = 'hello'
    var_3 = module_0.thaw(var_2)
    assert var_3 == 'hello'
    var_4 = True
    var_5 = module_0.thaw(var_4)
    assert var_5 is True
    var_6 = 2
    var_7 = 3
    var_8 = [var_4, var_6, var_7]
    var_9 = module_1.pset(var_8)
    var_10 = module_0.thaw(var_9)
    var_11 = {var_4, var_6, var_7}
    var_12 = module_0.thaw(var_11)
    var_13 = [var_4, var_6, var_7]
    var_14 = [var_4, var_6, var_7]
    var_15 = module_0.thaw(var_14)
    var_16 = [var_6, var_7]
    var_17 = [var_4, var_16]
    var_18 = module_0.thaw(var_17)
    var_19 = [var_4]
    var_20 = 'a'
    var_21 = {var_20: var_6}
    var_22 = module_2.pmap(var_21)
    var_23 = 'b'
    var_24 = {var_20: var_4, var_23: var_6}
    var_25 = module_2.pmap(var_24)
    var_26 = module_0.thaw(var_25)
    var_27 = {var_20: var_4, var_23: var_6}
    var_28 = module_0.thaw(var_27)
    var_29 = [var_4, var_6]
    var_30 = 'c'
    var_31 = {var_30: var_7}
    var_32 = (var_4, var_6, var_7)
    var_33 = module_0.thaw(var_32)
    var_34 = [var_6, var_7]
    var_35 = {var_20: var_4}
    var_36 = module_2.pmap(var_35)
    var_37 = (var_36,)
    var_38 = module_0.thaw(var_37)
    var_39 = 'list'
    var_40 = 'tuple'
    var_41 = 'dict'
    var_42 = 'inner'
    var_43 = {var_42: var_6}
    var_44 = module_2.pmap(var_43)
    var_45 = [var_4, var_44]
    var_46 = 4
    var_47 = [var_7, var_46]
    var_48 = module_1.pset(var_47)
    var_49 = (var_48,)
    var_50 = 'deep'
    var_51 = 5
    var_52 = [var_51]
    var_53 = {var_42: var_6}
    var_54 = [var_4, var_53]
    var_55 = {var_7, var_46}
    var_56 = (var_55,)
    var_57 = [var_51]
    var_58 = [var_57]
    var_59 = {var_50: var_58}
    var_60 = {var_39: var_54, var_40: var_56, var_41: var_59}
    var_61 = {var_23: var_4}
    var_62 = module_2.pmap(var_61)
    var_63 = {var_20: var_62}
    var_64 = False
    var_65 = module_0.thaw(var_63, var_64)
    var_66 = {var_23: var_4}
    var_67 = module_2.pmap(var_66)
    var_68 = {var_20: var_67}
    var_69 = [var_4]
    var_70 = [var_4]



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 3
    var_4 = {var_2: var_3}
    var_5 = [var_0, var_1, var_4]
    var_6 = 'key'
    var_7 = [var_0, var_1]
    var_8 = {var_6: var_7}
    var_9 = {var_2: var_0}
    var_10 = 'c'
    var_11 = 'b'
    var_12 = {var_11: var_3}
    var_13 = [var_0, var_1, var_12]
    var_14 = 4
    var_15 = 5
    var_16 = (var_14, var_15)
    var_17 = {var_2: var_13, var_10: var_16}



# Parsed testcases at query #5
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 1
    var_1 = module_0.thaw(var_0)
    assert var_1 == 1
    var_2 = 'string'
    var_3 = module_0.thaw(var_2)
    assert var_3 == 'string'
    var_4 = True
    var_5 = module_0.thaw(var_4)
    assert var_5 is True
    var_6 = None
    var_7 = module_0.thaw(var_6)
    assert var_7 is None
    var_8 = 2
    var_9 = 3
    var_10 = [var_8, var_9]
    var_11 = 'a'
    var_12 = 4
    var_13 = {var_11: var_12}
    var_14 = module_1.pmap(var_13)
    var_15 = [var_8, var_9]
    var_16 = {var_11: var_12}
    var_17 = [var_4, var_15, var_16]
    var_18 = [var_8, var_9]
    var_19 = {var_11: var_12}
    var_20 = [var_4, var_18, var_19]
    var_21 = module_0.thaw(var_20)
    var_22 = 'b'
    var_23 = 'd'
    var_24 = 'c'
    var_25 = {var_24: var_8}
    var_26 = module_1.pmap(var_25)
    var_27 = [var_9, var_12]
    var_28 = {var_24: var_8}
    var_29 = [var_9, var_12]
    var_30 = {var_11: var_4, var_22: var_28, var_23: var_29}
    var_31 = {var_24: var_8}
    var_32 = [var_9, var_12]
    var_33 = {var_11: var_4, var_22: var_31, var_23: var_32}
    var_34 = module_0.thaw(var_33)
    var_35 = [var_4, var_8, var_9]
    var_36 = module_2.pset(var_35)
    var_37 = module_0.thaw(var_36)
    var_38 = [var_8, var_9]
    var_39 = {var_11: var_12}
    var_40 = module_1.pmap(var_39)
    var_41 = [var_8, var_9]
    var_42 = {var_11: var_12}
    var_43 = (var_4, var_41, var_42)
    var_44 = [var_8, var_9]
    var_45 = [var_4, var_44]
    var_46 = False
    var_47 = module_0.thaw(var_45, var_46)
    var_48 = {var_22: var_4}
    var_49 = {var_11: var_48}
    var_50 = module_0.thaw(var_49, var_46)
    var_51 = 'list'
    var_52 = 'tuple'
    var_53 = 'dict'
    var_54 = [var_4]
    var_55 = module_2.pset(var_54)
    var_56 = 'inner'
    var_57 = {var_56: var_8}
    var_58 = module_1.pmap(var_57)
    var_59 = [var_55, var_58]
    var_60 = [var_9]
    var_61 = 'nested'
    var_62 = 'deep'
    var_63 = {var_62: var_12}
    var_64 = module_1.pmap(var_63)
    var_65 = {var_61: var_64}
    var_66 = {var_4}
    var_67 = {var_56: var_8}
    var_68 = [var_66, var_67]
    var_69 = [var_9]
    var_70 = (var_69,)
    var_71 = {var_62: var_12}
    var_72 = {var_61: var_71}
    var_73 = {var_51: var_68, var_52: var_70, var_53: var_72}



# Parsed testcases at query #6
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]
    var_5 = 'a'
    var_6 = {var_5: var_0}
    var_7 = {var_5: var_0}
    var_8 = module_0.pmap(var_7)
    var_9 = 10
    var_10 = 'nested'
    var_11 = [var_0]
    var_12 = {var_10: var_11}
    var_13 = [var_0]
    var_14 = 'b'
    var_15 = {var_14: var_2}
    var_16 = [var_1, var_15]
    var_17 = {var_5: var_16}
    var_18 = 4
    var_19 = 5
    var_20 = (var_18, var_19)
    var_21 = [var_0, var_17, var_20]



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]
    var_5 = 10
    var_6 = [var_5]
    var_7 = 5
    var_8 = 'a'
    var_9 = 'c'
    var_10 = 'b'
    var_11 = {var_10: var_2}
    var_12 = [var_0, var_1, var_11]
    var_13 = 4
    var_14 = (var_13, var_7)
    var_15 = {var_8: var_12, var_9: var_14}
    var_16 = 'inner'
    var_17 = [var_0, var_1]
    var_18 = {var_16: var_17}
    var_19 = [var_0]
    var_20 = [var_1]
    var_21 = [var_2]
    var_22 = [var_0, var_1, var_2]



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 'key'
    var_6 = 1
    var_7 = [var_6, var_4]
    var_8 = {var_5: var_7}
    var_9 = 0
    var_10 = 'hello'
    var_11 = ' world'
    var_12 = 10
    var_13 = None



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = [var_0, var_1]
    var_6 = 'key'
    var_7 = 'val'
    var_8 = {var_6: var_7}
    var_9 = 3
    var_10 = [var_0, var_1, var_9]
    var_11 = [var_0, var_1, var_9]
    var_12 = [var_0]
    var_13 = 'inner'
    var_14 = {var_13: var_1}
    var_15 = [var_1, var_9]
    var_16 = {var_3: var_15}
    var_17 = 4
    var_18 = 5
    var_19 = (var_17, var_18)
    var_20 = [var_0, var_16, var_19]
    var_21 = 0



# Parsed testcases at query #10
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1
import pyrsistent._pmap as module_2

def test_case_0():
    var_0 = 1
    var_1 = module_0.thaw(var_0)
    assert var_1 == 1
    var_2 = 'string'
    var_3 = module_0.thaw(var_2)
    assert var_3 == 'string'
    var_4 = True
    var_5 = module_0.thaw(var_4)
    assert var_5 is True
    var_6 = None
    var_7 = module_0.thaw(var_6)
    assert var_7 is None
    var_8 = 2
    var_9 = 3
    var_10 = [var_4, var_8, var_9]
    var_11 = module_1.pset(var_10)
    var_12 = module_0.thaw(var_11)
    var_13 = module_0.thaw(var_11)
    var_14 = [var_8, var_9]
    var_15 = 'a'
    var_16 = 4
    var_17 = {var_15: var_16}
    var_18 = module_2.pmap(var_17)
    var_19 = [var_8, var_9]
    var_20 = {var_15: var_16}
    var_21 = [var_4, var_19, var_20]
    var_22 = 'b'
    var_23 = 'd'
    var_24 = 'c'
    var_25 = {var_24: var_8}
    var_26 = module_2.pmap(var_25)
    var_27 = [var_9, var_16]
    var_28 = {var_24: var_8}
    var_29 = [var_9, var_16]
    var_30 = {var_15: var_4, var_22: var_28, var_23: var_29}
    var_31 = [var_8]
    var_32 = {var_15: var_9}
    var_33 = module_2.pmap(var_32)
    var_34 = [var_8]
    var_35 = {var_15: var_9}
    var_36 = (var_4, var_34, var_35)
    var_37 = {var_15: var_8}
    var_38 = [var_4, var_37]
    var_39 = module_0.thaw(var_38)
    var_40 = 'x'
    var_41 = [var_4, var_8]
    var_42 = {var_40: var_41}
    var_43 = module_0.thaw(var_42)
    var_44 = 'list'
    var_45 = 'tuple'
    var_46 = [var_4]
    var_47 = module_1.pset(var_46)
    var_48 = 'inner'
    var_49 = 5
    var_50 = [var_49]
    var_51 = 'key'
    var_52 = {var_51: var_4}
    var_53 = module_2.pmap(var_52)
    var_54 = (var_53,)
    var_55 = {var_4}
    var_56 = [var_49]
    var_57 = {var_48: var_56}
    var_58 = [var_55, var_57]
    var_59 = {var_51: var_4}
    var_60 = (var_59,)
    var_61 = {var_44: var_58, var_45: var_60}



# Parsed testcases at query #11
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1
import pyrsistent._pmap as module_2

def test_case_0():
    var_0 = 1
    var_1 = module_0.thaw(var_0)
    assert var_1 == 1
    var_2 = 'string'
    var_3 = module_0.thaw(var_2)
    assert var_3 == 'string'
    var_4 = True
    var_5 = module_0.thaw(var_4)
    assert var_5 is True
    var_6 = None
    var_7 = module_0.thaw(var_6)
    assert var_7 is None
    var_8 = 2
    var_9 = 3
    var_10 = [var_4, var_8, var_9]
    var_11 = module_1.pset(var_10)
    var_12 = module_0.thaw(var_11)
    var_13 = module_0.thaw(var_11)
    var_14 = [var_4, var_8, var_9]
    var_15 = [var_8, var_9]
    var_16 = 'a'
    var_17 = 4
    var_18 = {var_16: var_17}
    var_19 = module_2.pmap(var_18)
    var_20 = 'b'
    var_21 = {var_16: var_4, var_20: var_8}
    var_22 = module_2.pmap(var_21)
    var_23 = module_0.thaw(var_22)
    var_24 = module_0.thaw(var_22)
    var_25 = 'inner'
    var_26 = {var_25: var_4}
    var_27 = module_2.pmap(var_26)
    var_28 = [var_4, var_8]
    var_29 = [var_8, var_9]
    var_30 = {var_16: var_17}
    var_31 = module_2.pmap(var_30)
    var_32 = [var_4, var_8]
    var_33 = 'c'
    var_34 = {var_33: var_9}
    var_35 = {var_16: var_32, var_20: var_34}
    var_36 = module_0.thaw(var_35)
    var_37 = [var_8, var_9]
    var_38 = {var_16: var_17}
    var_39 = [var_4, var_37, var_38]
    var_40 = module_0.thaw(var_39)
    var_41 = [var_4, var_8]
    var_42 = False
    var_43 = {var_16: var_4}
    var_44 = module_2.pmap(var_43)
    var_45 = [var_44]
    var_46 = module_0.thaw(var_45, var_42)
    var_47 = var_46[var_42]



# Parsed testcases at query #12
#--------------------------


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = [var_0, var_1]
    var_4 = [var_0, var_1]
    var_5 = 0
    var_6 = 'a'
    var_7 = [var_0, var_1]
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = 'inner'
    var_12 = [var_0]
    var_13 = {var_11: var_12}
    var_14 = [var_0, var_1]
    var_15 = [var_0, var_1]
    var_16 = 3
    var_17 = [var_1, var_16]
    var_18 = (var_0, var_17)
    var_19 = 'c'
    var_20 = 'b'
    var_21 = {var_20: var_1}
    var_22 = [var_0, var_21]
    var_23 = 4
    var_24 = {var_16, var_23}
    var_25 = {var_6: var_22, var_19: var_24}
    var_26 = {var_20: var_1}
    var_27 = module_0.pmap(var_26)
    var_28 = [var_0, var_27]
    var_29 = {var_16, var_23}
    var_30 = module_1.pset(var_29)



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = [var_0, var_1]
    var_6 = {var_3: var_0}
    var_7 = 3
    var_8 = [var_0, var_1, var_7]
    var_9 = [var_0, var_1, var_7]
    var_10 = [var_1, var_7]
    var_11 = {var_3: var_10}
    var_12 = 4
    var_13 = 5
    var_14 = (var_12, var_13)
    var_15 = [var_0, var_11, var_14]
    var_16 = [var_0]
    var_17 = 'c'
    var_18 = {var_17: var_1}
    var_19 = 'b'



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = [var_1, var_2]
    var_5 = [var_1, var_2]
    var_6 = 'a'
    var_7 = 'b'
    var_8 = [var_1, var_2]
    var_9 = 'c'
    var_10 = 3
    var_11 = {var_9: var_10}
    var_12 = {var_6: var_8, var_7: var_11}
    var_13 = [var_2, var_10]
    var_14 = [var_2, var_10]
    var_15 = [var_1, var_2, var_10]
    var_16 = [var_1]
    var_17 = {var_6: var_16}
    var_18 = [var_1]
    var_19 = [var_2]
    var_20 = (var_1, var_19)
    var_21 = [var_2]
    var_22 = (var_1, var_21)
    var_23 = tuple_test(var_22)[var_1]



# Parsed testcases at query #15
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = [var_1, var_2, var_3]
    var_6 = [var_1]
    var_7 = 'key'
    var_8 = 'val'
    var_9 = {var_7: var_8}
    var_10 = [var_1]
    var_11 = {var_7: var_8}
    var_12 = module_0.pmap(var_11)
    var_13 = 'internal'
    var_14 = 'a'
    var_15 = [var_2, var_3]
    var_16 = {var_14: var_15}
    var_17 = 4
    var_18 = 5
    var_19 = {var_18}
    var_20 = (var_17, var_19)
    var_21 = [var_1, var_16, var_20]
    var_22 = [var_1, var_2]
    var_23 = [var_1, var_2, var_3]
    var_24 = 'x'
    var_25 = {var_24: var_1}
    var_26 = module_0.pmap(var_25)
    var_27 = [var_26]
    var_28 = 0



