####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
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
    var_2 = 'hello'
    var_3 = module_0.freeze(var_2)
    assert var_3 == 'hello'
    var_4 = None
    var_5 = module_0.freeze(var_4)
    assert var_5 is None
    var_6 = 2
    var_7 = 3
    var_8 = [var_0, var_6, var_7]
    var_9 = module_0.freeze(var_8)
    var_10 = [var_0, var_6, var_7]
    var_11 = [var_6, var_7]
    var_12 = [var_0, var_11]
    var_13 = module_0.freeze(var_12)
    var_14 = [var_6, var_7]
    var_15 = 'a'
    var_16 = 'b'
    var_17 = {var_15: var_0, var_16: var_6}
    var_18 = module_0.freeze(var_17)
    var_19 = {var_15: var_0, var_16: var_6}
    var_20 = module_1.pmap(var_19)
    var_21 = [var_0, var_6]
    var_22 = 'c'
    var_23 = {var_22: var_7}
    var_24 = {var_15: var_21, var_16: var_23}
    var_25 = module_0.freeze(var_24)
    var_26 = [var_0, var_6]
    var_27 = {var_22: var_7}
    var_28 = module_1.pmap(var_27)
    var_29 = {var_15: var_0, var_16: var_6}
    var_30 = {var_15: var_0, var_16: var_6}
    var_31 = module_1.pmap(var_30)
    var_32 = {var_0, var_6, var_7}
    var_33 = module_0.freeze(var_32)
    var_34 = {var_0, var_6, var_7}
    var_35 = module_2.pset(var_34)
    var_36 = (var_0, var_6, var_7)
    var_37 = module_0.freeze(var_36)
    var_38 = [var_6, var_7]
    var_39 = (var_0, var_38)
    var_40 = module_0.freeze(var_39)
    var_41 = [var_6, var_7]
    var_42 = {var_16: var_6}
    var_43 = [var_0, var_42]
    var_44 = 4
    var_45 = 5
    var_46 = [var_44, var_45]
    var_47 = (var_7, var_46)
    var_48 = {var_15: var_43, var_22: var_47}
    var_49 = {var_16: var_6}
    var_50 = module_1.pmap(var_49)
    var_51 = [var_0, var_50]
    var_52 = [var_44, var_45]
    var_53 = module_0.freeze(var_48)
    var_54 = [var_0, var_6, var_7]
    var_55 = True
    var_56 = [var_55, var_6, var_7]
    var_57 = False
    var_58 = {var_15: var_55, var_16: var_6}
    var_59 = module_1.pmap(var_58)
    var_60 = True
    var_61 = module_0.freeze(var_59, var_60)
    var_62 = {var_15: var_60, var_16: var_6}
    var_63 = module_1.pmap(var_62)
    var_64 = module_0.freeze(var_59, var_57)
    var_65 = {var_60, var_6, var_7}
    var_66 = module_2.pset(var_65)
    var_67 = module_0.freeze(var_66)



# Parsed testcases at query #2
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
    var_8 = 'original'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = 'changed'
    var_12 = {var_7: var_11}
    var_13 = module_0.pmap(var_12)
    var_14 = {var_6: var_13}
    var_15 = module_0.pmap(var_14)
    var_16 = 'c'
    var_17 = 'old'
    var_18 = {var_16: var_17}
    var_19 = 'a'
    var_20 = 'b'
    var_21 = 'kwargs'
    var_22 = 'new'
    var_23 = {var_16: var_22}
    var_24 = module_0.pmap(var_23)
    var_25 = {var_19: var_0, var_20: var_1, var_21: var_24}
    var_26 = module_0.pmap(var_25)
    var_27 = [var_0, var_1, var_2]
    var_28 = [var_0, var_1, var_2]
    var_29 = 'key'



# Parsed testcases at query #3
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 1
    var_1 = module_0.freeze(var_0)
    assert var_1 == 1
    var_2 = 'hello'
    var_3 = module_0.freeze(var_2)
    assert var_3 == 'hello'
    var_4 = None
    var_5 = module_0.freeze(var_4)
    assert var_5 is None
    var_6 = 2
    var_7 = 3
    var_8 = [var_0, var_6, var_7]
    var_9 = module_0.freeze(var_8)
    var_10 = [var_0, var_6, var_7]
    var_11 = [var_6, var_7]
    var_12 = [var_0, var_11]
    var_13 = module_0.freeze(var_12)
    var_14 = [var_6, var_7]
    var_15 = 'a'
    var_16 = {var_15: var_0}
    var_17 = module_0.freeze(var_16)
    var_18 = {var_15: var_0}
    var_19 = module_1.pmap(var_18)
    var_20 = [var_0, var_6]
    var_21 = {var_15: var_20}
    var_22 = module_0.freeze(var_21)
    var_23 = [var_0, var_6]
    var_24 = {var_15: var_0}
    var_25 = {var_15: var_0}
    var_26 = module_1.pmap(var_25)
    var_27 = {var_0, var_6, var_7}
    var_28 = module_0.freeze(var_27)
    var_29 = {var_0, var_6, var_7}
    var_30 = module_2.pset(var_29)
    var_31 = (var_0, var_6, var_7)
    var_32 = module_0.freeze(var_31)
    var_33 = [var_6, var_7]
    var_34 = (var_0, var_33)
    var_35 = module_0.freeze(var_34)
    var_36 = [var_6, var_7]
    var_37 = 'c'
    var_38 = 'b'
    var_39 = {var_38: var_6}
    var_40 = [var_0, var_39]
    var_41 = 4
    var_42 = 5
    var_43 = [var_41, var_42]
    var_44 = (var_7, var_43)
    var_45 = {var_15: var_40, var_37: var_44}
    var_46 = {var_38: var_6}
    var_47 = module_1.pmap(var_46)
    var_48 = [var_0, var_47]
    var_49 = [var_41, var_42]
    var_50 = module_0.freeze(var_45)
    var_51 = [var_0, var_6]
    var_52 = True
    var_53 = [var_52, var_6]
    var_54 = {var_15: var_52}
    var_55 = module_1.pmap(var_54)
    var_56 = True
    var_57 = module_0.freeze(var_55, var_56)
    var_58 = {var_15: var_56}
    var_59 = module_1.pmap(var_58)
    var_60 = [var_56, var_6]
    var_61 = False
    var_62 = [var_56, var_6]
    var_63 = {var_15: var_56}
    var_64 = module_1.pmap(var_63)
    var_65 = module_0.freeze(var_64, var_61)
    var_66 = {var_15: var_56}
    var_67 = module_1.pmap(var_66)
    var_68 = [var_56, var_6]
    var_69 = [var_56, var_6]
    var_70 = {var_15: var_56}
    var_71 = module_1.pmap(var_70)
    var_72 = module_0.freeze(var_71, var_61)
    var_73 = {var_15: var_56}
    var_74 = module_1.pmap(var_73)



# Parsed testcases at query #4
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
    var_12 = 'dict'
    var_13 = [var_0, var_1, var_2]
    var_14 = {var_6: var_0}
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = [var_0, var_1, var_2, var_4]
    var_17 = 'c'
    var_18 = {var_6: var_0, var_17: var_2}
    var_19 = module_0.pmap(var_18)
    var_20 = {var_0, var_1, var_2}
    var_21 = {var_0, var_1, var_2, var_4}
    var_22 = module_1.pset(var_21)



# Parsed testcases at query #5
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
    var_40 = module_0.freeze(var_0)
    assert var_40 == 1
    var_41 = 'hello'
    var_42 = module_0.freeze(var_41)
    assert var_42 == 'hello'



# Parsed testcases at query #6
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
    var_13 = [var_0, var_1]
    var_14 = {var_2, var_4}
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = 10
    var_17 = [var_0, var_1, var_16]
    var_18 = 20
    var_19 = {var_2, var_4, var_18}
    var_20 = module_1.pset(var_19)
    var_21 = [var_0, var_1]
    var_22 = 5
    var_23 = [var_0, var_1, var_22]



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
    var_12 = 'value'
    var_13 = 5
    var_14 = {var_12: var_13}
    var_15 = {var_11: var_14}
    var_16 = 10
    var_17 = {var_12: var_16}
    var_18 = module_0.pmap(var_17)
    var_19 = {var_11: var_18}
    var_20 = module_0.pmap(var_19)
    var_21 = [var_0]
    var_22 = [var_1, var_2]
    var_23 = [var_0, var_1, var_2]



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}



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
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = 'c'
    var_10 = {var_6: var_0, var_7: var_1, var_9: var_2}
    var_11 = module_0.pmap(var_10)
    var_12 = 'list'
    var_13 = 'dict'
    var_14 = [var_0, var_1]
    var_15 = 'key'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = {var_12: var_14, var_13: var_17}
    var_19 = [var_0, var_1, var_0]
    var_20 = 'new_value'
    var_21 = {var_15: var_20}
    var_22 = module_0.pmap(var_21)
    var_23 = {var_6: var_0}
    var_24 = 'new_key'
    var_25 = {var_6: var_0, var_24: var_20}
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
    var_7 = 'value'
    var_8 = 5
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = 10
    var_12 = {var_7: var_11}
    var_13 = module_0.pmap(var_12)
    var_14 = {var_6: var_13}
    var_15 = module_0.pmap(var_14)
    var_16 = 'x'
    var_17 = 'y'
    var_18 = 30
    var_19 = {var_16: var_11, var_17: var_18}
    var_20 = 20
    var_21 = {var_16: var_20, var_17: var_18}
    var_22 = module_0.pmap(var_21)
    var_23 = [var_0, var_1, var_2]
    var_24 = [var_0, var_1, var_2]
    var_25 = [var_0, var_1, var_2, var_4]



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
    var_18 = [var_0, var_1, var_2, var_4]
    var_19 = 'a'
    var_20 = {var_19: var_0}
    var_21 = 'b'
    var_22 = {var_19: var_0, var_21: var_1}
    var_23 = module_0.pmap(var_22)
    var_24 = [var_0, var_1, var_2]



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
    var_6 = 'a'
    var_7 = [var_0, var_1]
    var_8 = {var_6: var_7}
    var_9 = [var_2, var_4]
    var_10 = [var_2, var_4]
    var_11 = 'old_key'
    var_12 = 'old_value'
    var_13 = {var_11: var_12}
    var_14 = 'new_key'
    var_15 = 'new_value'
    var_16 = {var_11: var_12, var_14: var_15}
    var_17 = module_0.pmap(var_16)
    var_18 = [var_0, var_1, var_2]
    var_19 = [var_0, var_1, var_2]
    var_20 = [var_0, var_1, var_2, var_4]



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
    var_8 = 'b'
    var_9 = {var_6: var_0, var_8: var_1}
    var_10 = module_0.pmap(var_9)
    var_11 = 'list'
    var_12 = 'nested'
    var_13 = [var_0, var_1, var_2]
    var_14 = 'value'
    var_15 = 5
    var_16 = {var_14: var_15}
    var_17 = {var_11: var_13, var_12: var_16}
    var_18 = [var_0, var_1, var_2, var_4]
    var_19 = 10
    var_20 = {var_14: var_19}
    var_21 = module_0.pmap(var_20)
    var_22 = 'x'
    var_23 = 'y'
    var_24 = {var_22: var_0, var_23: var_1}
    var_25 = {var_22: var_19, var_23: var_1}
    var_26 = module_0.pmap(var_25)
    var_27 = [var_0, var_1, var_2]



# Parsed testcases at query #14
#--------------------------


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 4
    var_1 = 5
    var_2 = 'result'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_3, var_8: var_4}
    var_10 = {var_3, var_4, var_5}
    var_11 = 'result'
    var_12 = 0
    var_13 = 4
    var_14 = [var_3, var_4, var_5, var_13]
    var_15 = 'd'
    var_16 = {var_7: var_3, var_8: var_4, var_15: var_13}
    var_17 = module_0.pmap(var_16)
    var_18 = 5
    var_19 = {var_3, var_4, var_5, var_18}
    var_20 = module_1.pset(var_19)
    var_21 = [var_3, var_4, var_5]
    var_22 = [var_3, var_4, var_5]

import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 4
    var_1 = 5
    var_2 = 'result'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_3, var_8: var_4}
    var_10 = {var_3, var_4, var_5}
    var_11 = 'result'
    var_12 = 0
    var_13 = 4
    var_14 = [var_3, var_4, var_5, var_13]
    var_15 = 'd'
    var_16 = {var_7: var_3, var_8: var_4, var_15: var_13}
    var_17 = module_0.pmap(var_16)
    var_18 = 5
    var_19 = {var_3, var_4, var_5, var_18}
    var_20 = module_1.pset(var_19)
    var_21 = [var_3, var_4, var_5]
    var_22 = [var_3, var_4, var_5]



# Parsed testcases at query #15
#--------------------------


import pyrsistent._pset as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_0, var_1, var_3, var_4]
    var_7 = [var_0, var_1, var_3]
    var_8 = [var_0, var_1, var_3, var_4]
    var_9 = 'values'
    var_10 = [var_0, var_1, var_3]
    var_11 = {var_9: var_10}
    var_12 = 5
    var_13 = [var_0, var_1, var_3, var_12]
    var_14 = (var_0, var_1, var_3)
    var_15 = {var_0, var_1, var_3}
    var_16 = {var_0, var_1, var_3, var_4}
    var_17 = module_0.pset(var_16)
    var_18 = [var_0, var_1]
    var_19 = 'key'
    var_20 = 0
    var_21 = {var_19: var_20}
    var_22 = {var_0, var_1, var_3}
    var_23 = 10
    var_24 = {var_19: var_23}
    var_25 = module_1.pmap(var_24)
    var_26 = {var_0, var_1, var_3, var_12}
    var_27 = module_0.pset(var_26)
    var_28 = (var_23, var_25, var_27)



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
    var_13 = [var_0, var_1]
    var_14 = 'key'
    var_15 = 'value'
    var_16 = {var_14: var_15}
    var_17 = {var_11: var_13, var_12: var_16}
    var_18 = [var_0, var_1, var_0]
    var_19 = 'new_key'
    var_20 = 'new_value'
    var_21 = {var_14: var_15, var_19: var_20}
    var_22 = module_0.pmap(var_21)
    var_23 = 10
    var_24 = 20
    var_25 = 11
    var_26 = 21
    var_27 = {var_6: var_25, var_8: var_26}
    var_28 = module_0.pmap(var_27)



# Parsed testcases at query #17
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 1
    var_1 = module_0.freeze(var_0)
    assert var_1 == 1
    var_2 = 'hello'
    var_3 = module_0.freeze(var_2)
    assert var_3 == 'hello'
    var_4 = None
    var_5 = module_0.freeze(var_4)
    assert var_5 is None
    var_6 = 2
    var_7 = 3
    var_8 = [var_0, var_6, var_7]
    var_9 = module_0.freeze(var_8)
    var_10 = [var_0, var_6, var_7]
    var_11 = [var_6, var_7]
    var_12 = [var_0, var_11]
    var_13 = module_0.freeze(var_12)
    var_14 = [var_6, var_7]
    var_15 = 'a'
    var_16 = 'b'
    var_17 = {var_15: var_0, var_16: var_6}
    var_18 = module_0.freeze(var_17)
    var_19 = {var_15: var_0, var_16: var_6}
    var_20 = module_1.pmap(var_19)
    var_21 = [var_0, var_6]
    var_22 = {var_15: var_21}
    var_23 = module_0.freeze(var_22)
    var_24 = [var_0, var_6]
    var_25 = {var_15: var_0, var_16: var_6}
    var_26 = {var_15: var_0, var_16: var_6}
    var_27 = module_1.pmap(var_26)
    var_28 = (var_0, var_6, var_7)
    var_29 = module_0.freeze(var_28)
    var_30 = [var_6, var_7]
    var_31 = (var_0, var_30)
    var_32 = module_0.freeze(var_31)
    var_33 = [var_6, var_7]
    var_34 = {var_0, var_6, var_7}
    var_35 = module_0.freeze(var_34)
    var_36 = {var_0, var_6, var_7}
    var_37 = module_2.pset(var_36)
    var_38 = 'c'
    var_39 = {var_16: var_7}
    var_40 = [var_0, var_6, var_39]
    var_41 = 4
    var_42 = 5
    var_43 = 6
    var_44 = [var_42, var_43]
    var_45 = (var_41, var_44)
    var_46 = {var_15: var_40, var_38: var_45}
    var_47 = {var_16: var_7}
    var_48 = module_1.pmap(var_47)
    var_49 = [var_0, var_6, var_48]
    var_50 = [var_42, var_43]
    var_51 = module_0.freeze(var_46)
    var_52 = {var_15: var_6}
    var_53 = [var_0, var_52]
    var_54 = False
    var_55 = module_0.freeze(var_53, var_54)
    var_56 = {var_15: var_6}
    var_57 = [var_0, var_56]
    var_58 = [var_0, var_6]
    var_59 = {var_15: var_58}
    var_60 = module_0.freeze(var_59, var_54)
    var_61 = [var_0, var_6]
    var_62 = {var_15: var_61}
    var_63 = module_1.pmap(var_62)
    var_64 = [var_0, var_6, var_7]
    var_65 = {var_15: var_0}
    var_66 = module_1.pmap(var_65)
    var_67 = module_0.freeze(var_66)
    var_68 = {var_0, var_6}
    var_69 = module_2.pset(var_68)
    var_70 = module_0.freeze(var_69)



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
    var_7 = [var_0, var_1]
    var_8 = {var_6: var_7}
    var_9 = [var_2, var_4]
    var_10 = [var_2, var_4]
    var_11 = 'key'
    var_12 = 'value'
    var_13 = {var_11: var_12}
    var_14 = 'new_key'
    var_15 = 'new_value'
    var_16 = {var_11: var_12, var_14: var_15}
    var_17 = module_0.pmap(var_16)
    var_18 = [var_0, var_1]
    var_19 = 'original'
    var_20 = {var_11: var_19}
    var_21 = 'kwarg'
    var_22 = {var_21: var_19}
    var_23 = [var_0, var_1, var_0]
    var_24 = 'modified'
    var_25 = {var_11: var_24}
    var_26 = module_0.pmap(var_25)
    var_27 = {var_21: var_24}
    var_28 = module_0.pmap(var_27)
    var_29 = 5
    var_30 = (var_0, var_1)



# Parsed testcases at query #19
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
    var_16 = 'a'
    var_17 = {var_16: var_0}
    var_18 = 'b'
    var_19 = {var_18: var_1}
    var_20 = {var_16: var_0, var_18: var_1}
    var_21 = module_0.pmap(var_20)
    var_22 = 'old_key'
    var_23 = 'old_value'
    var_24 = {var_22: var_23}
    var_25 = 'new_key'
    var_26 = 'new_value'
    var_27 = {var_22: var_23, var_25: var_26}
    var_28 = module_0.pmap(var_27)
    var_29 = [var_0, var_1, var_2]
    var_30 = [var_0, var_1, var_2]



# Parsed testcases at query #20
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
    var_6 = 'nested'
    var_7 = 'other'
    var_8 = 'value'
    var_9 = 'original'
    var_10 = {var_8: var_9}
    var_11 = [var_0, var_1]
    var_12 = {var_6: var_10, var_7: var_11}
    var_13 = 'modified'
    var_14 = {var_8: var_13}
    var_15 = module_0.pmap(var_14)
    var_16 = [var_0, var_1]
    var_17 = 'c'
    var_18 = 'd'
    var_19 = 'old'
    var_20 = [var_0, var_1]
    var_21 = {var_17: var_19, var_18: var_20}
    var_22 = 'a'
    var_23 = 'b'
    var_24 = 'kwargs'
    var_25 = 'new'
    var_26 = [var_0, var_1]
    var_27 = (var_0, var_1, var_2)
    var_28 = {var_0, var_1, var_2}
    var_29 = {var_0, var_1, var_2, var_4}
    var_30 = module_1.pset(var_29)
    var_31 = [var_0, var_1, var_2]
    var_32 = [var_0, var_1, var_2, var_4]



# Parsed testcases at query #21
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
    var_27 = [var_0, var_1, var_2]
    var_28 = [var_0, var_1, var_2]



# Parsed testcases at query #22
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
    var_20 = 'key1'
    var_21 = 'value1'
    var_22 = {var_20: var_21}
    var_23 = 'new_key'
    var_24 = 'new_value'
    var_25 = {var_20: var_21, var_23: var_24}
    var_26 = module_0.pmap(var_25)



# Parsed testcases at query #23
#--------------------------


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 'result'
    var_3 = 'data'
    var_4 = 'extra'
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]
    var_9 = 'key'
    var_10 = 'old_value'
    var_11 = {var_9: var_10}
    var_12 = {var_5, var_6, var_7}
    var_13 = 'result'
    var_14 = 'data'
    var_15 = 'extra'
    var_16 = [var_5, var_6, var_7, var_5]
    var_17 = 'value'
    var_18 = {var_9: var_17}
    var_19 = module_0.pmap(var_18)
    var_20 = 5
    var_21 = {var_5, var_6, var_7, var_20}
    var_22 = module_1.pset(var_21)



# Parsed testcases at query #24
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
    var_12 = 'dict'
    var_13 = [var_0, var_1, var_2]
    var_14 = {var_6: var_0}
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = [var_0, var_1, var_2, var_4]
    var_17 = 'c'
    var_18 = {var_6: var_0, var_17: var_2}
    var_19 = module_0.pmap(var_18)
    var_20 = {var_0, var_1, var_2}
    var_21 = {var_0, var_1, var_2, var_4}
    var_22 = module_1.pset(var_21)



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}



# Parsed testcases at query #26
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
    var_16 = 'x'
    var_17 = 'y'
    var_18 = 5
    var_19 = 20
    var_20 = {var_16: var_18, var_17: var_19}
    var_21 = 10
    var_22 = {var_16: var_21, var_17: var_19}
    var_23 = module_0.pmap(var_22)
    var_24 = [var_0, var_1]
    var_25 = 'a'
    var_26 = {var_25: var_0}
    var_27 = 'z'
    var_28 = {var_27: var_21}
    var_29 = [var_0, var_1, var_0]
    var_30 = 'new_key'
    var_31 = 'new_value'
    var_32 = {var_25: var_0, var_30: var_31}
    var_33 = module_0.pmap(var_32)
    var_34 = 30
    var_35 = {var_27: var_34}
    var_36 = module_0.pmap(var_35)
    var_37 = (var_0, var_1, var_2)



# Parsed testcases at query #27
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
    var_20 = 'x'
    var_21 = 10
    var_22 = {var_6: var_0, var_8: var_1, var_20: var_21}
    var_23 = module_0.pmap(var_22)
    var_24 = (var_0, var_1, var_2)



# Parsed testcases at query #28
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
    var_13 = [var_0, var_1]
    var_14 = {var_2, var_4}
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = 10
    var_17 = [var_0, var_1, var_16]
    var_18 = 20
    var_19 = {var_2, var_4, var_18}
    var_20 = module_1.pset(var_19)
    var_21 = 'd'
    var_22 = 'c'
    var_23 = {var_21: var_2, var_22: var_2}
    var_24 = module_0.pmap(var_23)



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_0, var_1, var_3, var_4]
    var_7 = [var_0, var_1, var_3]
    var_8 = [var_0, var_1, var_3, var_4]
    var_9 = 'a'
    var_10 = 'b'
    var_11 = [var_1, var_3]
    var_12 = {var_9: var_0, var_10: var_11}
    var_13 = 'new_key'
    var_14 = [var_1, var_3]
    var_15 = 'new_value'
    var_16 = [var_0, var_1, var_3]
    var_17 = [var_0, var_1, var_3]



# Parsed testcases at query #30
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
    var_12 = 'dict'
    var_13 = [var_0, var_1, var_2]
    var_14 = {var_6: var_0}
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = [var_0, var_1, var_2, var_4]
    var_17 = 'c'
    var_18 = {var_6: var_0, var_17: var_2}
    var_19 = module_0.pmap(var_18)
    var_20 = 'x'
    var_21 = 10
    var_22 = {var_6: var_0, var_8: var_1, var_20: var_21}
    var_23 = module_0.pmap(var_22)
    var_24 = (var_0, var_1, var_2)
    var_25 = {var_0, var_1, var_2}
    var_26 = {var_0, var_1, var_2, var_4}
    var_27 = module_1.pset(var_26)



# Parsed testcases at query #31
#--------------------------


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'key'
    var_5 = 'old_value'
    var_6 = {var_4: var_5}
    var_7 = 4
    var_8 = {var_2, var_7}
    var_9 = 'result'
    var_10 = 'data'
    var_11 = 'set'
    var_12 = [var_0, var_1, var_2, var_0]
    var_13 = 'value'
    var_14 = {var_4: var_13}
    var_15 = module_0.pmap(var_14)
    var_16 = {var_2, var_7, var_1}
    var_17 = module_1.pset(var_16)
    var_18 = [var_0]
    var_19 = 'a'
    var_20 = {var_19: var_0}
    var_21 = 'nested'
    var_22 = 5
    var_23 = {var_13: var_22}
    var_24 = {var_21: var_23}



# Parsed testcases at query #32
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
    var_16 = 'a'
    var_17 = {var_16: var_0}
    var_18 = 'b'
    var_19 = {var_18: var_1}
    var_20 = {var_16: var_0, var_18: var_1}
    var_21 = module_0.pmap(var_20)
    var_22 = [var_0, var_1]
    var_23 = [var_0, var_1, var_2]
    var_24 = 'list'
    var_25 = 'set'
    var_26 = [var_0, var_1, var_2]
    var_27 = {var_0, var_1, var_2}
    var_28 = {var_7: var_11}
    var_29 = {var_24: var_26, var_25: var_27, var_6: var_28}
    var_30 = [var_0, var_1, var_2, var_4]
    var_31 = {var_0, var_1, var_2, var_8}
    var_32 = module_1.pset(var_31)
    var_33 = 20
    var_34 = {var_7: var_33}
    var_35 = module_0.pmap(var_34)



# Parsed testcases at query #33
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
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = 'c'
    var_10 = {var_6: var_0, var_7: var_1, var_9: var_2}
    var_11 = module_0.pmap(var_10)
    var_12 = 'list'
    var_13 = 'dict'
    var_14 = [var_0, var_1]
    var_15 = 'key'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = {var_12: var_14, var_13: var_17}
    var_19 = 10
    var_20 = [var_0, var_1, var_19]
    var_21 = 'new_key'
    var_22 = 'new_value'
    var_23 = {var_15: var_16, var_21: var_22}
    var_24 = module_0.pmap(var_23)
    var_25 = 20
    var_26 = 'kwargs'
    var_27 = 'x'
    var_28 = 'y'
    var_29 = {var_27: var_19, var_28: var_25, var_21: var_22}
    var_30 = module_0.pmap(var_29)
    var_31 = {var_6: var_0, var_7: var_1, var_26: var_30}
    var_32 = module_0.pmap(var_31)



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    pass



# Parsed testcases at query #35
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
    var_9 = 'new_value'
    var_10 = {var_6: var_0, var_8: var_9}
    var_11 = module_0.pmap(var_10)
    var_12 = 'list'
    var_13 = 'dict'
    var_14 = [var_0, var_1]
    var_15 = {var_6: var_0}
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = 5
    var_18 = [var_0, var_1, var_17]
    var_19 = 'b'
    var_20 = {var_6: var_0, var_19: var_1}
    var_21 = module_0.pmap(var_20)
    var_22 = 'key'
    var_23 = 'value'
    var_24 = {var_22: var_23}
    var_25 = [var_0, var_1]
    var_26 = [var_0, var_1]
    var_27 = 'modified'
    var_28 = True
    var_29 = {var_22: var_23, var_27: var_28}
    var_30 = module_0.pmap(var_29)



# Parsed testcases at query #36
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
    var_20 = 5



# Parsed testcases at query #37
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
    var_15 = 'list'
    var_16 = 'dict'
    var_17 = [var_0, var_1, var_2]
    var_18 = {var_6: var_0, var_8: var_1}
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = [var_0, var_1, var_2, var_4]
    var_21 = 'c'
    var_22 = {var_6: var_0, var_8: var_1, var_21: var_2}
    var_23 = module_0.pmap(var_22)



# Parsed testcases at query #38
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
    var_23 = 5



# Parsed testcases at query #39
#--------------------------


import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 'a'
    var_7 = 'b'
    var_8 = [var_0, var_1, var_2]
    var_9 = {var_6: var_8, var_7: var_1}
    var_10 = [var_0, var_1, var_2, var_4]
    var_11 = 'c'
    var_12 = 'd'
    var_13 = [var_0, var_1, var_2]
    var_14 = {var_11: var_13, var_12: var_1}
    var_15 = [var_0, var_1, var_2, var_4]
    var_16 = [var_0, var_1, var_2]
    var_17 = (var_0, var_1, var_2)
    var_18 = {var_0, var_1, var_2}
    var_19 = {var_0, var_1, var_2, var_4}
    var_20 = module_0.pset(var_19)



# Parsed testcases at query #40
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
    var_7 = 'b'
    var_8 = [var_1, var_2]
    var_9 = {var_6: var_0, var_7: var_8}
    var_10 = 'c'
    var_11 = [var_1, var_2]
    var_12 = 'new_key'
    var_13 = 'new_value'
    var_14 = {var_6: var_0, var_7: var_1, var_12: var_13}
    var_15 = module_0.pmap(var_14)
    var_16 = [var_0, var_1]
    var_17 = 'x'
    var_18 = 10
    var_19 = {var_17: var_18}
    var_20 = [var_0, var_1, var_2]
    var_21 = {var_17: var_18, var_12: var_2}
    var_22 = module_0.pmap(var_21)
    var_23 = [var_0, var_1, var_2]



# Parsed testcases at query #41
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 'key'
    var_7 = 'value'
    var_8 = [var_7]
    var_9 = {var_6: var_8}
    var_10 = 'modified'
    var_11 = [var_10]
    var_12 = 'existing'
    var_13 = {var_12: var_0}
    var_14 = [var_1, var_2]
    var_15 = 'new_key'
    var_16 = {var_12: var_0, var_15: var_1}
    var_17 = module_0.pmap(var_16)
    var_18 = 'x'
    var_19 = 'y'
    var_20 = {var_18: var_0, var_19: var_1}
    var_21 = 'changed'
    var_22 = {var_18: var_21, var_19: var_1}
    var_23 = module_0.pmap(var_22)
    var_24 = 'a'



# Parsed testcases at query #42
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
    var_20 = 'key1'
    var_21 = 'value1'
    var_22 = {var_20: var_21}
    var_23 = 'new_key'
    var_24 = 'new_value'
    var_25 = {var_20: var_21, var_23: var_24}
    var_26 = module_0.pmap(var_25)



# Parsed testcases at query #43
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
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = 'c'
    var_10 = {var_6: var_0, var_7: var_1, var_9: var_2}
    var_11 = module_0.pmap(var_10)
    var_12 = 'list'
    var_13 = 'set'
    var_14 = [var_0, var_1, var_2]
    var_15 = {var_0, var_1, var_2}
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = [var_0, var_1, var_2, var_4]
    var_18 = {var_0, var_1, var_2, var_4}
    var_19 = module_1.pset(var_18)
    var_20 = {var_6: var_1}
    var_21 = {var_2, var_4}
    var_22 = [var_0, var_20, var_21]
    var_23 = {var_6: var_1}
    var_24 = module_0.pmap(var_23)
    var_25 = {var_2, var_4}
    var_26 = module_1.pset(var_25)
    var_27 = [var_0, var_24, var_26]



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    pass



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
    var_11 = 'list'
    var_12 = 'dict'
    var_13 = [var_0, var_1, var_2]
    var_14 = {var_6: var_0, var_8: var_1}
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = [var_0, var_1, var_2, var_4]
    var_17 = 'c'
    var_18 = {var_6: var_0, var_8: var_1, var_17: var_2}
    var_19 = module_0.pmap(var_18)
    var_20 = (var_0, var_1, var_2)
    var_21 = {var_0, var_1, var_2}
    var_22 = {var_0, var_1, var_2, var_4}
    var_23 = module_1.pset(var_22)
    var_24 = [var_0, var_1, var_2]



# Parsed testcases at query #3
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_0, var_1, var_3, var_4]
    var_7 = [var_0, var_1, var_3]
    var_8 = [var_0, var_1, var_3, var_4]
    var_9 = 'a'
    var_10 = {var_9: var_0}
    var_11 = 'b'
    var_12 = {var_9: var_0, var_11: var_1}
    var_13 = module_0.pmap(var_12)
    var_14 = 'nested'
    var_15 = 'value'
    var_16 = 5
    var_17 = {var_15: var_16}
    var_18 = {var_14: var_17}
    var_19 = 10
    var_20 = {var_15: var_19}
    var_21 = module_0.pmap(var_20)
    var_22 = {var_14: var_21}
    var_23 = module_0.pmap(var_22)
    var_24 = {var_9: var_0, var_11: var_1}
    var_25 = 'new_key'
    var_26 = 'new_value'
    var_27 = {var_9: var_0, var_11: var_1, var_25: var_26}
    var_28 = module_0.pmap(var_27)
    var_29 = [var_0, var_1]
    var_30 = 'x'
    var_31 = {var_30: var_19}
    var_32 = 'arg1'
    var_33 = [var_0, var_1, var_3]



# Parsed testcases at query #4
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
    var_19 = 5



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
    var_28 = [var_0, var_1, var_2]



# Parsed testcases at query #6
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
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = 'new_key'
    var_10 = 'new_value'
    var_11 = {var_6: var_0, var_7: var_1, var_9: var_10}
    var_12 = module_0.pmap(var_11)
    var_13 = {var_0, var_1}
    var_14 = {var_0, var_1, var_2}
    var_15 = module_1.pset(var_14)
    var_16 = (var_0, var_1, var_2)
    var_17 = {var_6: var_0}
    var_18 = [var_17, var_1, var_2]
    var_19 = 10
    var_20 = {var_6: var_19}
    var_21 = module_0.pmap(var_20)
    var_22 = [var_21, var_1, var_2]



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
    var_8 = [var_0, var_1]
    var_9 = 'new_key'
    var_10 = 'value'
    var_11 = {var_6: var_0, var_9: var_10}
    var_12 = module_0.pmap(var_11)
    var_13 = 5
    var_14 = [var_0, var_1, var_13]
    var_15 = 'list'
    var_16 = 'dict'
    var_17 = [var_0, var_1]
    var_18 = {var_6: var_0}
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = [var_0, var_1, var_2]
    var_21 = 'new'
    var_22 = {var_6: var_0, var_21: var_10}
    var_23 = module_0.pmap(var_22)
    var_24 = {var_6: var_0}
    var_25 = {var_6: var_0, var_21: var_10}
    var_26 = module_0.pmap(var_25)
    var_27 = [var_0, var_1, var_2]



# Parsed testcases at query #8
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
    var_15 = 'value'
    var_16 = {var_14: var_15}
    var_17 = {var_11: var_13, var_12: var_16}
    var_18 = [var_0, var_1, var_0]
    var_19 = 'new_key'
    var_20 = 'new_value'
    var_21 = {var_14: var_15, var_19: var_20}
    var_22 = module_0.pmap(var_21)
    var_23 = {var_6: var_0}
    var_24 = {var_6: var_0, var_19: var_20}
    var_25 = module_0.pmap(var_24)



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
    var_8 = 'original'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = 'modified'
    var_12 = {var_7: var_11}
    var_13 = module_0.pmap(var_12)
    var_14 = {var_6: var_13}
    var_15 = module_0.pmap(var_14)
    var_16 = 'key'
    var_17 = 'other'
    var_18 = 'old_value'
    var_19 = 'unchanged'
    var_20 = {var_16: var_18, var_17: var_19}
    var_21 = 'new_value'
    var_22 = {var_16: var_21, var_17: var_19}
    var_23 = module_0.pmap(var_22)
    var_24 = [var_0, var_1]
    var_25 = [var_2, var_4]
    var_26 = 'extra'
    var_27 = [var_0, var_1, var_2, var_4, var_26]
    var_28 = 'a'



# Parsed testcases at query #10
#--------------------------


import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 'a'
    var_7 = 'nested'
    var_8 = [var_0, var_1]
    var_9 = {var_6: var_0, var_7: var_8}
    var_10 = 'b'
    var_11 = 10
    var_12 = [var_0, var_1, var_11]
    var_13 = {var_0, var_1, var_2}
    var_14 = {var_0, var_1, var_2, var_4}
    var_15 = module_0.pset(var_14)
    var_16 = [var_0, var_1]
    var_17 = 5
    var_18 = [var_0, var_1, var_17]
    var_19 = [var_0, var_1]
    var_20 = [var_0, var_1, var_2]



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
    var_7 = {var_6: var_0}
    var_8 = 'b'
    var_9 = {var_6: var_0, var_8: var_1}
    var_10 = module_0.pmap(var_9)
    var_11 = 'list'
    var_12 = 'dict'
    var_13 = [var_0, var_1]
    var_14 = 'key'
    var_15 = 'value'
    var_16 = {var_14: var_15}
    var_17 = {var_11: var_13, var_12: var_16}
    var_18 = [var_0, var_1, var_0]
    var_19 = 'new_key'
    var_20 = 'new_value'
    var_21 = {var_14: var_15, var_19: var_20}
    var_22 = module_0.pmap(var_21)
    var_23 = {var_6: var_0, var_8: var_1, var_19: var_20}
    var_24 = module_0.pmap(var_23)
    var_25 = 42
    var_26 = 'string'



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
    var_6 = 'a'
    var_7 = {var_6: var_0}
    var_8 = 'b'
    var_9 = {var_6: var_0, var_8: var_1}
    var_10 = module_0.pmap(var_9)
    var_11 = 'lst'
    var_12 = 'd'
    var_13 = [var_0, var_1, var_2]
    var_14 = 'x'
    var_15 = 5
    var_16 = {var_14: var_15}
    var_17 = {var_11: var_13, var_12: var_16}
    var_18 = [var_0, var_1, var_2, var_4]
    var_19 = 10
    var_20 = {var_14: var_19}
    var_21 = module_0.pmap(var_20)
    var_22 = {var_6: var_1, var_8: var_1}
    var_23 = module_0.pmap(var_22)
    var_24 = [var_0, var_1, var_2]
    var_25 = (var_0, var_1, var_2)



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
    var_8 = 'b'
    var_9 = {var_6: var_0, var_8: var_1}
    var_10 = module_0.pmap(var_9)
    var_11 = 'list'
    var_12 = [var_0, var_1, var_2]
    var_13 = {var_11: var_12}
    var_14 = [var_0, var_1, var_2, var_4]
    var_15 = (var_0, var_1, var_2)



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
    var_7 = [var_0, var_1]
    var_8 = {var_6: var_7}
    var_9 = [var_2, var_4]
    var_10 = [var_2, var_4]
    var_11 = 'old_key'
    var_12 = 'old_value'
    var_13 = {var_11: var_12}
    var_14 = 'new_key'
    var_15 = 'new_value'
    var_16 = {var_11: var_12, var_14: var_15}
    var_17 = module_0.pmap(var_16)
    var_18 = [var_0, var_1]
    var_19 = [var_2, var_4]
    var_20 = [var_0, var_1, var_2, var_4]
    var_21 = 5



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
    var_11 = 'list'
    var_12 = 'dict'
    var_13 = [var_0, var_1, var_2]
    var_14 = {var_6: var_0}
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = [var_0, var_1, var_2, var_4]
    var_17 = 'c'
    var_18 = {var_6: var_0, var_17: var_2}
    var_19 = module_0.pmap(var_18)
    var_20 = {var_6: var_0, var_8: var_1}
    var_21 = 'new_key'
    var_22 = 'new_value'
    var_23 = {var_6: var_0, var_8: var_1, var_21: var_22}
    var_24 = module_0.pmap(var_23)
    var_25 = (var_0, var_1, var_2)
    var_26 = {var_0, var_1, var_2}
    var_27 = {var_0, var_1, var_2, var_4}
    var_28 = module_1.pset(var_27)
    var_29 = 5
    var_30 = 'hello'



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    pass



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
    var_6 = 'nested'
    var_7 = 'value'
    var_8 = 'original'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = 'modified'
    var_12 = {var_7: var_11}
    var_13 = module_0.pmap(var_12)
    var_14 = {var_6: var_13}
    var_15 = module_0.pmap(var_14)
    var_16 = [var_0, var_1]
    var_17 = [var_2, var_4]
    var_18 = 'new'
    var_19 = [var_0, var_1, var_2, var_4, var_18]
    var_20 = 'a'
    var_21 = {var_20: var_0}
    var_22 = 'b'
    var_23 = 'c'
    var_24 = {var_20: var_0, var_22: var_1, var_23: var_2}
    var_25 = module_0.pmap(var_24)
    var_26 = 'key'
    var_27 = (var_0, var_1, var_2)
    var_28 = {var_0, var_1, var_2}
    var_29 = 5
    var_30 = {var_0, var_1, var_2, var_4, var_29}
    var_31 = module_1.pset(var_30)



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
    var_13 = [var_0, var_1, var_2]
    var_14 = {var_6: var_0}
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = [var_0, var_1, var_2, var_4]
    var_17 = 'c'
    var_18 = {var_6: var_0, var_17: var_2}
    var_19 = module_0.pmap(var_18)
    var_20 = [var_0, var_1, var_2]
    var_21 = [var_0, var_1, var_2, var_4]



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    pass



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = 'b'
    var_6 = [var_0, var_1]
    var_7 = 'c'
    var_8 = {var_7: var_2}
    var_9 = {var_4: var_6, var_5: var_8}
    var_10 = 'x'
    var_11 = [var_0, var_1]
    var_12 = {var_10: var_11}
    var_13 = [var_0, var_1, var_2]



# Parsed testcases at query #21
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
    var_15 = 'value'
    var_16 = {var_14: var_15}
    var_17 = {var_11: var_13, var_12: var_16}
    var_18 = [var_0, var_1, var_0]
    var_19 = 'new_key'
    var_20 = 'new_value'
    var_21 = {var_14: var_15, var_19: var_20}
    var_22 = module_0.pmap(var_21)
    var_23 = [var_0, var_1, var_2]
    var_24 = [var_0, var_1, var_2]



# Parsed testcases at query #22
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
    var_7 = 'b'
    var_8 = {var_7: var_0}
    var_9 = {var_6: var_8}
    var_10 = {var_7: var_1}
    var_11 = module_0.pmap(var_10)
    var_12 = {var_6: var_11}
    var_13 = module_0.pmap(var_12)
    var_14 = {var_6: var_0}
    var_15 = {var_7: var_1}
    var_16 = {var_6: var_0, var_7: var_1}
    var_17 = module_0.pmap(var_16)
    var_18 = [var_0, var_1]
    var_19 = [var_0, var_1, var_2]



# Parsed testcases at query #23
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
    var_23 = 10
    var_24 = 20
    var_25 = 11
    var_26 = 21
    var_27 = {var_6: var_25, var_8: var_26}
    var_28 = module_0.pmap(var_27)



# Parsed testcases at query #24
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
    var_20 = 5



# Parsed testcases at query #25
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
    var_14 = 'key'
    var_15 = 'value'
    var_16 = {var_14: var_15}
    var_17 = {var_11: var_13, var_12: var_16}
    var_18 = [var_0, var_1, var_2, var_4]
    var_19 = 'new_key'
    var_20 = 'new_value'
    var_21 = {var_14: var_15, var_19: var_20}
    var_22 = module_0.pmap(var_21)
    var_23 = {var_14: var_15}
    var_24 = {var_14: var_15, var_19: var_20}
    var_25 = module_0.pmap(var_24)



# Parsed testcases at query #26
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
    var_11 = {var_0, var_1, var_2}
    var_12 = {var_0, var_1, var_2, var_4}
    var_13 = module_1.pset(var_12)
    var_14 = (var_0, var_1, var_2)
    var_15 = 'list'
    var_16 = 'dict'
    var_17 = [var_0, var_1, var_2]
    var_18 = 'key'
    var_19 = 'value'
    var_20 = {var_18: var_19}
    var_21 = {var_15: var_17, var_16: var_20}
    var_22 = [var_0, var_1, var_2, var_4]
    var_23 = 'new_key'
    var_24 = 'new_value'
    var_25 = {var_18: var_19, var_23: var_24}
    var_26 = module_0.pmap(var_25)



# Parsed testcases at query #27
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
    var_23 = {var_1: var_0}
    var_24 = {var_1: var_0}
    var_25 = module_1.pmap(var_24)
    var_26 = 4
    var_27 = [var_2, var_26]
    var_28 = {var_1: var_27}
    var_29 = [var_0, var_28]
    var_30 = False
    var_31 = module_0.freeze(var_29, var_30)
    var_32 = [var_2, var_26]
    var_33 = {var_1: var_32}
    var_34 = [var_0, var_33]
    var_35 = [var_0, var_13, var_2]
    var_36 = {var_1: var_0}
    var_37 = module_1.pmap(var_36)
    var_38 = module_0.freeze(var_37)
    var_39 = [var_0, var_13]
    var_40 = module_2.pset(var_39)
    var_41 = module_0.freeze(var_40)
    var_42 = 42
    var_43 = module_0.freeze(var_42)
    assert var_43 == 42
    var_44 = 'hello'
    var_45 = module_0.freeze(var_44)
    assert var_45 == 'hello'



# Parsed testcases at query #28
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
    var_16 = 'x'
    var_17 = 'y'
    var_18 = {var_16: var_0, var_17: var_1}
    var_19 = 999
    var_20 = {var_16: var_19, var_17: var_1}
    var_21 = module_0.pmap(var_20)
    var_22 = [var_0, var_1, var_2]
    var_23 = [var_0, var_1, var_2]
    var_24 = False



# Parsed testcases at query #29
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
    var_11 = (var_0, var_1)
    var_12 = {var_0, var_1}
    var_13 = {var_0, var_1, var_2}
    var_14 = module_1.pset(var_13)
    var_15 = 'list'
    var_16 = 'dict'
    var_17 = [var_0, var_1, var_2]
    var_18 = 'key'
    var_19 = 'value'
    var_20 = {var_18: var_19}
    var_21 = {var_15: var_17, var_16: var_20}
    var_22 = [var_0, var_1, var_2, var_4]
    var_23 = 'new_key'
    var_24 = 'new_value'
    var_25 = {var_18: var_19, var_23: var_24}
    var_26 = module_0.pmap(var_25)



# Parsed testcases at query #30
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
    var_7 = 'nested'
    var_8 = [var_0, var_1]
    var_9 = {var_6: var_0, var_7: var_8}
    var_10 = 'b'
    var_11 = 10
    var_12 = [var_0, var_1, var_11]
    var_13 = 'new_key'
    var_14 = 'new_value'
    var_15 = {var_6: var_0, var_10: var_1, var_13: var_14}
    var_16 = module_0.pmap(var_15)
    var_17 = [var_0, var_1, var_2]
    var_18 = [var_0, var_1, var_2]
    var_19 = [var_0, var_1, var_2]



# Parsed testcases at query #31
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



# Parsed testcases at query #32
#--------------------------


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = [var_3]
    var_5 = [var_0, var_1, var_3]
    var_6 = [var_0, var_1, var_3]
    var_7 = 4
    var_8 = [var_0, var_1, var_3, var_7]
    var_9 = 'a'
    var_10 = {var_9: var_0}
    var_11 = 'b'
    var_12 = {var_9: var_0, var_11: var_1}
    var_13 = module_0.pmap(var_12)
    var_14 = 'list'
    var_15 = 'dict'
    var_16 = [var_0, var_1]
    var_17 = {var_9: var_0}
    var_18 = {var_14: var_16, var_15: var_17}
    var_19 = [var_0, var_1, var_0]
    var_20 = 'new_key'
    var_21 = 'new_value'
    var_22 = {var_9: var_0, var_20: var_21}
    var_23 = module_0.pmap(var_22)
    var_24 = (var_0, var_1, var_3)
    var_25 = {var_0, var_1, var_3}
    var_26 = {var_0, var_1, var_3, var_7}
    var_27 = module_1.pset(var_26)



# Parsed testcases at query #33
#--------------------------


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 3
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = 1
    var_6 = 2
    var_7 = [var_5, var_6]
    var_8 = 'y'
    var_9 = {var_8: var_5}
    var_10 = 4
    var_11 = 5
    var_12 = {var_10, var_11}
    var_13 = 'a'
    var_14 = 'b'
    var_15 = 'c'
    var_16 = [var_5, var_6, var_5]
    var_17 = 'x'
    var_18 = {var_8: var_5, var_17: var_6}
    var_19 = module_0.pmap(var_18)
    var_20 = 3
    var_21 = {var_10, var_11, var_20}
    var_22 = module_1.pset(var_21)
    var_23 = [var_5]
    var_24 = {var_8: var_5}
    var_25 = 'list'
    var_26 = 'dict'
    var_27 = [var_5, var_6]
    var_28 = {var_8: var_5}
    var_29 = {var_25: var_27, var_26: var_28}
    var_30 = [var_5, var_6, var_5]
    var_31 = {var_8: var_5, var_17: var_6}
    var_32 = module_0.pmap(var_31)

import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 3
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = 1
    var_6 = 2
    var_7 = [var_5, var_6]
    var_8 = 'y'
    var_9 = {var_8: var_5}
    var_10 = 4
    var_11 = 5
    var_12 = {var_10, var_11}
    var_13 = 'a'
    var_14 = 'b'
    var_15 = 'c'
    var_16 = [var_5, var_6, var_5]
    var_17 = 'x'
    var_18 = {var_8: var_5, var_17: var_6}
    var_19 = module_0.pmap(var_18)
    var_20 = 3
    var_21 = {var_10, var_11, var_20}
    var_22 = module_1.pset(var_21)
    var_23 = [var_5]
    var_24 = {var_8: var_5}
    var_25 = 'list'
    var_26 = 'dict'
    var_27 = [var_5, var_6]
    var_28 = {var_8: var_5}
    var_29 = {var_25: var_27, var_26: var_28}
    var_30 = [var_5, var_6, var_5]
    var_31 = {var_8: var_5, var_17: var_6}
    var_32 = module_0.pmap(var_31)



# Parsed testcases at query #34
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
    var_14 = {var_6: var_0}
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = [var_0, var_1, var_0]
    var_17 = 'new_key'
    var_18 = 'value'
    var_19 = {var_6: var_0, var_17: var_18}
    var_20 = module_0.pmap(var_19)
    var_21 = [var_0, var_1, var_2]
    var_22 = [var_0, var_1, var_2]



# Parsed testcases at query #35
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
    var_7 = 'b'
    var_8 = [var_1, var_2]
    var_9 = {var_6: var_0, var_7: var_8}
    var_10 = 'c'
    var_11 = [var_1, var_2]
    var_12 = 'key1'
    var_13 = 'value1'
    var_14 = {var_12: var_13}
    var_15 = 'new_key'
    var_16 = 'new_value'
    var_17 = {var_12: var_13, var_15: var_16}
    var_18 = module_0.pmap(var_17)
    var_19 = [var_0, var_1, var_2]
    var_20 = [var_1, var_2]
    var_21 = [var_1, var_2]



# Parsed testcases at query #36
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
    var_15 = 'value'
    var_16 = {var_14: var_15}
    var_17 = {var_11: var_13, var_12: var_16}
    var_18 = [var_0, var_1, var_0]
    var_19 = 'new_key'
    var_20 = 'new_value'
    var_21 = {var_14: var_15, var_19: var_20}
    var_22 = module_0.pmap(var_21)
    var_23 = 5



# Parsed testcases at query #37
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
    var_20 = 'x'
    var_21 = 10
    var_22 = {var_6: var_0, var_8: var_1, var_20: var_21}
    var_23 = module_0.pmap(var_22)



# Parsed testcases at query #38
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
    var_12 = 'nested_dict'
    var_13 = [var_0, var_1, var_2]
    var_14 = {var_6: var_0}
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = [var_0, var_1, var_2, var_4]
    var_17 = 'c'
    var_18 = {var_6: var_0, var_17: var_2}
    var_19 = module_0.pmap(var_18)
    var_20 = 5
    var_21 = 10
    var_22 = 'x'
    var_23 = 'y'
    var_24 = 6
    var_25 = {var_22: var_24, var_23: var_21}
    var_26 = module_0.pmap(var_25)
    var_27 = (var_0, var_1, var_2)
    var_28 = {var_0, var_1, var_2}
    var_29 = {var_0, var_1, var_2, var_4}
    var_30 = module_1.pset(var_29)



# Parsed testcases at query #39
#--------------------------


import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_0, var_1, var_3, var_4]
    var_7 = [var_0, var_1, var_3]
    var_8 = [var_0, var_1, var_3, var_4]
    var_9 = 'values'
    var_10 = [var_0, var_1, var_3]
    var_11 = {var_9: var_10}
    var_12 = [var_0, var_1, var_3, var_4]
    var_13 = (var_0, var_1, var_3)
    var_14 = {var_0, var_1, var_3}
    var_15 = {var_0, var_1, var_3, var_4}
    var_16 = module_0.pset(var_15)
    var_17 = [var_0, var_1, var_3]



# Parsed testcases at query #40
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
    var_9 = 'new_value'
    var_10 = {var_6: var_0, var_8: var_9}
    var_11 = module_0.pmap(var_10)
    var_12 = 'list'
    var_13 = 'nested'
    var_14 = [var_0, var_1]
    var_15 = 'value'
    var_16 = 5
    var_17 = {var_15: var_16}
    var_18 = {var_12: var_14, var_13: var_17}
    var_19 = [var_0, var_1, var_16]
    var_20 = 10
    var_21 = {var_15: var_20}
    var_22 = module_0.pmap(var_21)
    var_23 = 'b'
    var_24 = 'kwargs'
    var_25 = 'd'
    var_26 = 'c'
    var_27 = {var_25: var_4, var_26: var_2}
    var_28 = module_0.pmap(var_27)
    var_29 = {var_6: var_0, var_23: var_1, var_24: var_28}
    var_30 = module_0.pmap(var_29)
    var_31 = 'string'



# Parsed testcases at query #41
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



# Parsed testcases at query #42
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
    var_20 = 'x'
    var_21 = 10
    var_22 = {var_6: var_0, var_8: var_1, var_20: var_21}
    var_23 = module_0.pmap(var_22)
    var_24 = [var_0, var_1, var_2]
    var_25 = (var_0, var_1, var_2)



# Parsed testcases at query #43
#--------------------------


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 4
    var_1 = 4
    var_2 = 'result'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_3, var_8: var_4}
    var_10 = {var_3, var_4, var_5}
    var_11 = 0
    var_12 = 'result'
    var_13 = 4
    var_14 = [var_3, var_4, var_5, var_13]
    var_15 = 'd'
    var_16 = {var_7: var_3, var_8: var_4, var_15: var_13}
    var_17 = module_0.pmap(var_16)
    var_18 = {var_3, var_4, var_5, var_13}
    var_19 = module_1.pset(var_18)

import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 4
    var_1 = 4
    var_2 = 'result'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_3, var_8: var_4}
    var_10 = {var_3, var_4, var_5}
    var_11 = 0
    var_12 = 'result'
    var_13 = 4
    var_14 = [var_3, var_4, var_5, var_13]
    var_15 = 'd'
    var_16 = {var_7: var_3, var_8: var_4, var_15: var_13}
    var_17 = module_0.pmap(var_16)
    var_18 = {var_3, var_4, var_5, var_13}
    var_19 = module_1.pset(var_18)



# Parsed testcases at query #44
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
    var_11 = 'lst'
    var_12 = 'd'
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
    var_27 = (var_0, var_1, var_2)



# Parsed testcases at query #45
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
    var_20 = 5



# Parsed testcases at query #46
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



# Parsed testcases at query #47
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



