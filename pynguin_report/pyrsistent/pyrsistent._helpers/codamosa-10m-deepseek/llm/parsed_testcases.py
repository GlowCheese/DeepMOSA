####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._helpers as module_0


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = list(var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.freeze(var_8)
    var_10 = dict(var_9)
    var_11 = {var_0, var_1, var_2}
    var_12 = module_0.freeze(var_11)
    var_13 = set(var_12)
    var_14 = [var_1, var_2]
    var_15 = 4
    var_16 = {var_6: var_15}
    var_17 = (var_0, var_14, var_16)
    var_18 = module_0.freeze(var_17)
    var_19 = var_18[var_0]
    var_20 = var_18[var_0]
    var_21 = list(var_20)
    var_22 = var_18[var_1]
    var_23 = var_18[var_1]
    var_24 = dict(var_23)
    var_25 = [var_0, var_1, var_2]
    var_26 = False
    var_27 = module_0.freeze(var_25, var_26)
    var_28 = list(var_27)
    var_29 = {var_6: var_0, var_7: var_1}
    var_30 = module_0.freeze(var_29, var_26)
    var_31 = dict(var_30)
    var_32 = [var_0, var_1, var_2]
    var_33 = 'c'
    var_34 = 'd'
    var_35 = 5
    var_36 = 6
    var_37 = [var_35, var_36]
    var_38 = {var_33: var_15, var_34: var_37}
    var_39 = {var_6: var_32, var_7: var_38}
    var_40 = module_0.freeze(var_39)
    var_41 = var_40[var_6]
    var_42 = var_40[var_6]
    var_43 = list(var_42)
    var_44 = var_40[var_7]
    var_45 = var_40[var_7][var_34]
    var_46 = var_40[var_7][var_34]
    var_47 = list(var_46)
    var_48 = 'All tests passed!'
    var_49 = print(var_48)



# Parsed testcases at query #2
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = {var_4, var_5, var_6}
    var_8 = print(var_3)
    var_9 = print(var_7)
    var_10 = (var_0, var_1, var_2)
    var_11 = [var_4, var_5, var_6]
    var_12 = frozenset(var_11)
    var_13 = print(var_10)
    var_14 = print(var_12)
    var_15 = [var_0, var_1, var_2]
    var_16 = [var_4, var_5, var_6]
    var_17 = frozenset(var_16)
    var_18 = print(var_15)
    var_19 = print(var_17)
    var_20 = [var_1, var_2]
    var_21 = [var_0, var_1]
    var_22 = [var_2, var_4]
    var_23 = [var_21, var_22]
    var_24 = print(var_23)
    var_25 = 'b'
    var_26 = {var_25: var_1}
    var_27 = print(var_26)
    var_28 = {var_1, var_2}
    var_29 = print(var_28)
    var_30 = (var_1, var_2)
    var_31 = print(var_30)
    var_32 = range(var_2)
    var_33 = list(var_30)
    var_34 = print(var_33)
    var_35 = None
    var_36 = [var_0, var_1]
    var_37 = {var_2, var_4}
    var_38 = (var_5, var_6)
    var_39 = print(var_36)
    var_40 = print(var_37)
    var_41 = print(var_38)
    var_42 = [var_0, var_1]
    var_43 = {var_2, var_4}
    var_44 = (var_5, var_6)
    var_45 = [var_42, var_43, var_44]
    var_46 = print(var_45)
    var_47 = [var_1, var_2]
    var_48 = [var_0, var_1]
    var_49 = {var_2, var_4}
    var_50 = print(var_48)
    var_51 = print(var_49)
    var_52 = [var_0, var_1]
    var_53 = print(var_52)
    var_54 = print(var_52)
    var_55 = 'a'
    var_56 = {var_0, var_1}
    var_57 = [var_2, var_4]
    var_58 = [var_55, var_56, var_57]
    var_59 = print(var_58)
    var_60 = []
    var_61 = print(var_60)
    var_62 = 1000
    var_63 = range(var_62)
    var_64 = list(var_63)
    var_65 = len(var_64)
    var_66 = print(var_65)
    var_67 = [var_0]
    var_68 = lambda x: x + var_67
    var_69 = module_0.mutant(var_68)
    var_70 = [var_1, var_2]



# Parsed testcases at query #3
#--------------------------


import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.freeze(var_8)
    var_10 = {var_6: var_0, var_7: var_1}
    var_11 = module_1.pmap(var_10)
    var_12 = {var_0, var_1, var_2}
    var_13 = module_0.freeze(var_12)
    var_14 = {var_0, var_1, var_2}
    var_15 = module_2.pset(var_14)
    var_16 = [var_1, var_2]
    var_17 = (var_0, var_16)
    var_18 = module_0.freeze(var_17)
    var_19 = [var_1, var_2]
    var_20 = {var_6: var_0}
    var_21 = {var_1, var_2}
    var_22 = [var_20, var_21]
    var_23 = module_0.freeze(var_22)
    var_24 = {var_6: var_0}
    var_25 = module_1.pmap(var_24)
    var_26 = {var_1, var_2}
    var_27 = module_2.pset(var_26)
    var_28 = [var_25, var_27]
    var_29 = [var_0, var_1, var_2]
    var_30 = False
    var_31 = module_0.freeze(var_29, var_30)
    var_32 = [var_0, var_1, var_2]
    var_33 = [var_0, var_1, var_2]
    var_34 = True
    var_35 = module_0.freeze(var_33, var_34)
    var_36 = [var_34, var_1, var_2]
    var_37 = 'All tests passed!'
    var_38 = print(var_37)



# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.freeze(var_8)
    var_10 = {var_6: var_0, var_7: var_1}
    var_11 = module_1.pmap(var_10)
    var_12 = {var_0, var_1, var_2}
    var_13 = module_0.freeze(var_12)
    var_14 = {var_0, var_1, var_2}
    var_15 = module_2.pset(var_14)
    var_16 = [var_1, var_2]
    var_17 = (var_0, var_16)
    var_18 = module_0.freeze(var_17)
    var_19 = [var_1, var_2]
    var_20 = {var_6: var_0}
    var_21 = {var_1, var_2}
    var_22 = [var_20, var_21]
    var_23 = module_0.freeze(var_22)
    var_24 = {var_6: var_0}
    var_25 = module_1.pmap(var_24)
    var_26 = {var_1, var_2}
    var_27 = module_2.pset(var_26)
    var_28 = [var_25, var_27]
    var_29 = [var_0, var_1, var_2]
    var_30 = False
    var_31 = module_0.freeze(var_29, var_30)
    var_32 = [var_0, var_1, var_2]
    var_33 = {var_6: var_0}
    var_34 = {var_6: var_0}
    var_35 = module_1.pmap(var_34)
    var_36 = 'All tests passed for freeze.'
    var_37 = print(var_36)



# Parsed testcases at query #5
#--------------------------




# Parsed testcases at query #6
#--------------------------



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
    var_16 = {var_11: var_0}
    var_17 = {var_10: var_16}
    var_18 = module_0.freeze(var_17)
    var_19 = {var_11: var_0}
    var_20 = module_1.pmap(var_19)
    var_21 = {var_10: var_20}
    var_22 = module_1.pmap(var_21)
    var_23 = {var_0, var_1, var_2}
    var_24 = module_0.freeze(var_23)
    var_25 = {var_0, var_1, var_2}
    var_26 = module_2.pset(var_25)
    var_27 = [var_1, var_2]
    var_28 = (var_0, var_27)
    var_29 = module_0.freeze(var_28)
    var_30 = [var_1, var_2]
    var_31 = [var_0]
    var_32 = [var_0, var_1, var_2]
    var_33 = False
    var_34 = module_0.freeze(var_32, var_33)
    var_35 = [var_0, var_1, var_2]
    var_36 = {var_10: var_0}
    var_37 = module_0.freeze(var_36, var_33)
    var_38 = {var_10: var_0}
    var_39 = module_1.pmap(var_38)
    var_40 = [var_0, var_1, var_2]
    var_41 = [var_0, var_1, var_2]
    var_42 = {var_10: var_0}
    var_43 = module_1.pmap(var_42)
    var_44 = module_0.freeze(var_43)
    var_45 = {var_10: var_0}
    var_46 = module_1.pmap(var_45)
    var_47 = [var_0, var_1, var_2]
    var_48 = [var_0, var_1, var_2]
    var_49 = {var_10: var_0}
    var_50 = module_1.pmap(var_49)
    var_51 = module_0.freeze(var_50, var_33)
    var_52 = {var_10: var_0}
    var_53 = module_1.pmap(var_52)
    var_54 = 'All tests passed!'
    var_55 = print(var_54)



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
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 'new_key'
    var_10 = 'new_value'
    var_11 = {var_6: var_7, var_9: var_10}
    var_12 = module_0.pmap(var_11)
    var_13 = 'All tests passed!'
    var_14 = print(var_13)



# Parsed testcases at query #8
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2, var_0]
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 'new_key'
    var_9 = 'new_value'
    var_10 = {var_5: var_6, var_8: var_9}
    var_11 = module_0.pmap(var_10)
    var_12 = 'list'
    var_13 = [var_0, var_1, var_2]
    var_14 = {var_12: var_13}
    var_15 = 4
    var_16 = [var_0, var_1, var_2, var_15]
    var_17 = 'All tests passed!'
    var_18 = print(var_17)



# Parsed testcases at query #9
#--------------------------




# Parsed testcases at query #10
#--------------------------




# Parsed testcases at query #11
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2, var_0]
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 'new_key'
    var_9 = 'new_value'
    var_10 = {var_5: var_6, var_8: var_9}
    var_11 = module_0.pmap(var_10)
    var_12 = 'All tests passed!'
    var_13 = print(var_12)



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = 'test_mutant passed'
    var_7 = print(var_6)



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_0, var_1, var_2}
    var_8 = 'All tests passed!'
    var_9 = print(var_8)



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = 'b'
    var_6 = {var_4: var_0, var_5: var_1}
    var_7 = 0
    var_8 = 'All tests passed!'
    var_9 = print(var_8)



# Parsed testcases at query #15
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2, var_0]
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 'new_key'
    var_9 = 'new_value'
    var_10 = {var_5: var_6, var_8: var_9}
    var_11 = module_0.pmap(var_10)
    var_12 = 'All tests passed'
    var_13 = print(var_12)



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test passed: list mutation isolated'
    var_5 = print(var_4)
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 'Test passed: dict mutation isolated'
    var_10 = print(var_9)
    var_11 = 'list'
    var_12 = 'dict'
    var_13 = [var_0, var_1, var_2]
    var_14 = 'existing'
    var_15 = 'old'
    var_16 = {var_14: var_15}
    var_17 = {var_11: var_13, var_12: var_16}
    var_18 = 4
    var_19 = [var_0, var_1, var_2, var_18]
    var_20 = 'new'
    var_21 = {var_14: var_15, var_20: var_7}
    var_22 = {var_11: var_19, var_12: var_21}
    var_23 = 'Test passed: nested mutation isolated'
    var_24 = print(var_23)
    var_25 = 'All tests passed!'
    var_26 = print(var_25)



# Parsed testcases at query #17
#--------------------------


import pyrsistent._helpers as module_0


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.freeze(var_8)
    var_10 = {var_6: var_0, var_7: var_1}
    var_11 = module_1.pmap(var_10)
    var_12 = {var_0, var_1, var_2}
    var_13 = module_0.freeze(var_12)
    var_14 = {var_0, var_1, var_2}
    var_15 = module_2.pset(var_14)
    var_16 = [var_1, var_2]
    var_17 = (var_0, var_16)
    var_18 = module_0.freeze(var_17)
    var_19 = [var_1, var_2]
    var_20 = {var_6: var_0}
    var_21 = {var_1, var_2}
    var_22 = [var_20, var_21]
    var_23 = module_0.freeze(var_22)
    var_24 = {var_6: var_0}
    var_25 = module_1.pmap(var_24)
    var_26 = {var_1, var_2}
    var_27 = module_2.pset(var_26)
    var_28 = [var_25, var_27]
    var_29 = [var_0, var_1, var_2]
    var_30 = False
    var_31 = module_0.freeze(var_29, var_30)
    var_32 = [var_0, var_1, var_2]
    var_33 = [var_0, var_1, var_2]
    var_34 = [var_0, var_1, var_2]
    var_35 = {var_6: var_0}
    var_36 = module_1.pmap(var_35)
    var_37 = module_0.freeze(var_36)
    var_38 = {var_6: var_0}
    var_39 = module_1.pmap(var_38)
    var_40 = 'All tests passed for freeze'
    var_41 = print(var_40)



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = 'Result list should be a PVector'
    var_7 = 'Result dict should be a PMap'
    var_8 = 'All tests passed!'
    var_9 = print(var_8)



# Parsed testcases at query #19
#--------------------------


import pyrsistent._pmap as module_0


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2, var_0]
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 'new_key'
    var_9 = 'new_value'
    var_10 = {var_5: var_6, var_8: var_9}
    var_11 = module_0.pmap(var_10)
    var_12 = 'All tests passed!'
    var_13 = print(var_12)



# Parsed testcases at query #20
#--------------------------


import pyrsistent._pset as module_1


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2, var_0]
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 'new_key'
    var_9 = 'new_value'
    var_10 = {var_5: var_6, var_8: var_9}
    var_11 = module_0.pmap(var_10)
    var_12 = {var_0, var_1, var_2}
    var_13 = 4
    var_14 = {var_0, var_1, var_2, var_13}
    var_15 = module_1.pset(var_14)
    var_16 = 'All tests passed!'
    var_17 = print(var_16)



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = 'Result list should be a pvector'
    var_7 = 'Result dict should be a pmap'
    var_8 = 'All tests passed!'
    var_9 = print(var_8)



# Parsed testcases at query #22
#--------------------------


import pyrsistent._helpers as module_0


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = list(var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.freeze(var_8)
    var_10 = dict(var_9)
    var_11 = {var_0, var_1, var_2}
    var_12 = module_0.freeze(var_11)
    var_13 = set(var_12)
    var_14 = [var_1, var_2]
    var_15 = 4
    var_16 = {var_6: var_15}
    var_17 = (var_0, var_14, var_16)
    var_18 = module_0.freeze(var_17)
    var_19 = var_18[var_0]
    var_20 = var_18[var_0]
    var_21 = list(var_20)
    var_22 = var_18[var_1]
    var_23 = var_18[var_1]
    var_24 = dict(var_23)
    var_25 = [var_0, var_1, var_2]
    var_26 = False
    var_27 = module_0.freeze(var_25, var_26)
    var_28 = list(var_27)
    var_29 = {var_6: var_0, var_7: var_1}
    var_30 = module_0.freeze(var_29, var_26)
    var_31 = dict(var_30)
    var_32 = 'All tests passed!'
    var_33 = print(var_32)



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = 'All tests passed!'
    var_3 = print(var_2)



# Parsed testcases at query #24
#--------------------------


import pyrsistent._pmap as module_0


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2, var_0]
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 'new_key'
    var_9 = 'new_value'
    var_10 = {var_5: var_6, var_8: var_9}
    var_11 = module_0.pmap(var_10)
    var_12 = 'All tests passed!'
    var_13 = print(var_12)



# Parsed testcases at query #25
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = 4
    var_7 = [var_0, var_1, var_2, var_6]
    var_8 = 'b'
    var_9 = {var_4: var_0, var_8: var_1}
    var_10 = module_0.pmap(var_9)
    var_11 = [var_0, var_1, var_2]
    var_12 = {var_4: var_0}
    var_13 = module_0.pmap(var_12)
    var_14 = [var_0, var_1, var_2, var_6]
    var_15 = {var_4: var_0, var_8: var_1}
    var_16 = module_0.pmap(var_15)
    var_17 = 'All tests passed!'
    var_18 = print(var_17)



# Parsed testcases at query #26
#--------------------------




# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = 'All tests passed!'
    var_7 = print(var_6)



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = 'All tests passed'
    var_7 = print(var_6)



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test passed: list mutation isolated'
    var_5 = print(var_4)
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 'Test passed: dict mutation isolated'
    var_10 = print(var_9)
    var_11 = 'list'
    var_12 = 'dict'
    var_13 = [var_0, var_1, var_2]
    var_14 = 'existing'
    var_15 = {var_14: var_7}
    var_16 = {var_11: var_13, var_12: var_15}
    var_17 = 4
    var_18 = [var_0, var_1, var_2, var_17]
    var_19 = 'new'
    var_20 = {var_14: var_7, var_19: var_7}
    var_21 = {var_11: var_18, var_12: var_20}
    var_22 = 'Test passed: nested mutation isolated'
    var_23 = print(var_22)
    var_24 = 'All tests passed!'
    var_25 = print(var_24)



# Parsed testcases at query #30
#--------------------------




####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.freeze(var_8)
    var_10 = {var_6: var_0, var_7: var_1}
    var_11 = module_1.pmap(var_10)
    var_12 = {var_0, var_1, var_2}
    var_13 = module_0.freeze(var_12)
    var_14 = {var_0, var_1, var_2}
    var_15 = module_2.pset(var_14)
    var_16 = [var_1, var_2]
    var_17 = (var_0, var_16)
    var_18 = module_0.freeze(var_17)
    var_19 = [var_1, var_2]
    var_20 = [var_0]
    var_21 = [var_0, var_1, var_2]
    var_22 = False
    var_23 = module_0.freeze(var_21, var_22)
    var_24 = [var_0, var_1, var_2]
    var_25 = {var_6: var_0, var_7: var_1}
    var_26 = module_0.freeze(var_25, var_22)
    var_27 = {var_6: var_0, var_7: var_1}
    var_28 = module_1.pmap(var_27)
    var_29 = [var_0, var_1, var_2]
    var_30 = [var_0, var_1, var_2]
    var_31 = {var_6: var_0, var_7: var_1}
    var_32 = module_1.pmap(var_31)
    var_33 = module_0.freeze(var_32)
    var_34 = {var_6: var_0, var_7: var_1}
    var_35 = module_1.pmap(var_34)
    var_36 = 'All tests passed!'
    var_37 = print(var_36)



# Parsed testcases at query #2
#--------------------------


import pyrsistent._helpers as module_2
import pyrsistent._pmap as module_0
import pyrsistent._pvector as module_1


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.m()
    var_4 = module_1.v()
    var_5 = (var_0, var_4)
    var_6 = module_2.thaw(var_5)
    var_7 = 'All tests passed!'
    var_8 = print(var_7)



# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.m()
    var_4 = module_1.v()
    var_5 = (var_0, var_4)
    var_6 = module_2.thaw(var_5)
    var_7 = 'All tests passed!'
    var_8 = print(var_7)



# Parsed testcases at query #4
#--------------------------




# Parsed testcases at query #5
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.m()
    var_4 = module_1.v()
    var_5 = (var_0, var_4)
    var_6 = module_2.thaw(var_5)
    var_7 = [var_0, var_1, var_2]
    var_8 = module_2.thaw(var_7)
    var_9 = 4
    var_10 = {var_0: var_1, var_2: var_9}
    var_11 = module_2.thaw(var_10)
    var_12 = (var_0, var_1, var_2)
    var_13 = module_2.thaw(var_12)
    var_14 = [var_0, var_1, var_2]
    var_15 = set(var_14)
    var_16 = module_2.thaw(var_15)
    var_17 = module_2.thaw(var_0)
    assert var_17 == 1
    var_18 = 'a'
    var_19 = module_2.thaw(var_18)
    assert var_19 == 'a'
    var_20 = None
    var_21 = module_2.thaw(var_20)
    assert var_21 is None
    var_22 = True
    var_23 = module_2.thaw(var_22)
    assert var_23 is True
    var_24 = False
    var_25 = module_2.thaw(var_24)
    assert var_25 is False
    var_26 = module_2.thaw(var_22)
    var_27 = b'abc'
    var_28 = module_2.thaw(var_27)
    assert var_28 == b'abc'
    var_29 = bytearray(var_27)
    var_30 = module_2.thaw(var_29)
    var_31 = bytearray(var_27)
    var_32 = memoryview(var_27)
    var_33 = module_2.thaw(var_32)
    var_34 = memoryview(var_27)
    var_35 = slice(var_22, var_1, var_2)
    var_36 = module_2.thaw(var_35)
    var_37 = slice(var_22, var_1, var_2)
    var_38 = range(var_22, var_1, var_2)
    var_39 = module_2.thaw(var_38)
    var_40 = range(var_22, var_1, var_2)
    var_41 = Exception()
    var_42 = module_2.thaw(var_41)
    var_43 = Exception()
    var_44 = lambda x: x
    var_45 = module_2.thaw(var_44)
    var_46 = [var_22, var_1, var_2]
    var_47 = module_2.thaw(var_46, var_24)
    var_48 = {var_22: var_1, var_2: var_9}
    var_49 = module_2.thaw(var_48, var_24)
    var_50 = (var_22, var_1, var_2)
    var_51 = module_2.thaw(var_50, var_24)
    var_52 = [var_22, var_1, var_2]
    var_53 = set(var_52)
    var_54 = module_2.thaw(var_53, var_24)
    var_55 = module_2.thaw(var_22, var_24)
    assert var_55 == 1
    var_56 = module_2.thaw(var_18, var_24)
    assert var_56 == 'a'
    var_57 = module_2.thaw(var_20, var_24)
    assert var_57 is None
    var_58 = True
    var_59 = module_2.thaw(var_58, var_24)
    assert var_59 is True
    var_60 = module_2.thaw(var_24, var_24)
    assert var_60 is False
    var_61 = module_2.thaw(var_58, var_24)
    var_62 = module_2.thaw(var_27, var_24)
    assert var_62 == b'abc'
    var_63 = bytearray(var_27)
    var_64 = module_2.thaw(var_63, var_24)
    var_65 = bytearray(var_27)
    var_66 = memoryview(var_27)
    var_67 = module_2.thaw(var_66, var_24)
    var_68 = memoryview(var_27)
    var_69 = slice(var_58, var_1, var_2)
    var_70 = module_2.thaw(var_69, var_24)
    var_71 = slice(var_58, var_1, var_2)
    var_72 = range(var_58, var_1, var_2)
    var_73 = module_2.thaw(var_72, var_24)
    var_74 = range(var_58, var_1, var_2)
    var_75 = Exception()
    var_76 = module_2.thaw(var_75, var_24)
    var_77 = Exception()
    var_78 = lambda x: x
    var_79 = module_2.thaw(var_78, var_24)
    var_80 = [var_58, var_1, var_2]
    var_81 = True
    var_82 = module_2.thaw(var_80, var_81)
    var_83 = {var_81: var_1, var_2: var_9}
    var_84 = True
    var_85 = module_2.thaw(var_83, var_84)
    var_86 = (var_84, var_1, var_2)
    var_87 = True
    var_88 = module_2.thaw(var_86, var_87)
    var_89 = [var_87, var_1, var_2]
    var_90 = set(var_89)
    var_91 = True
    var_92 = module_2.thaw(var_90, var_91)
    var_93 = True
    var_94 = module_2.thaw(var_91, var_93)
    assert var_94 == 1
    var_95 = True
    var_96 = module_2.thaw(var_18, var_95)
    assert var_96 == 'a'
    var_97 = True
    var_98 = module_2.thaw(var_20, var_97)
    assert var_98 is None
    var_99 = True
    var_100 = True
    var_101 = module_2.thaw(var_99, var_100)
    assert var_101 is True
    var_102 = True
    var_103 = module_2.thaw(var_24, var_102)
    assert var_103 is False
    var_104 = True
    var_105 = module_2.thaw(var_102, var_104)
    var_106 = True
    var_107 = True
    var_108 = module_2.thaw(var_27, var_107)
    assert var_108 == b'abc'
    var_109 = bytearray(var_27)
    var_110 = True
    var_111 = module_2.thaw(var_109, var_110)
    var_112 = bytearray(var_27)
    var_113 = memoryview(var_27)
    var_114 = True
    var_115 = module_2.thaw(var_113, var_114)
    var_116 = memoryview(var_27)
    var_117 = slice(var_114, var_1, var_2)
    var_118 = True
    var_119 = module_2.thaw(var_117, var_118)
    var_120 = slice(var_118, var_1, var_2)
    var_121 = range(var_118, var_1, var_2)
    var_122 = True
    var_123 = module_2.thaw(var_121, var_122)
    var_124 = range(var_122, var_1, var_2)
    var_125 = True
    var_126 = True
    var_127 = True
    var_128 = Exception()
    var_129 = True
    var_130 = module_2.thaw(var_128, var_129)
    var_131 = Exception()
    var_132 = lambda x: x
    var_133 = True
    var_134 = module_2.thaw(var_132, var_133)
    var_135 = [var_133, var_1, var_2]
    var_136 = module_2.thaw(var_135, var_24)
    var_137 = {var_133: var_1, var_2: var_9}
    var_138 = module_2.thaw(var_137, var_24)
    var_139 = (var_133, var_1, var_2)
    var_140 = module_2.thaw(var_139, var_24)
    var_141 = [var_133, var_1, var_2]
    var_142 = set(var_141)
    var_143 = module_2.thaw(var_142, var_24)
    var_144 = module_2.thaw(var_133, var_24)
    assert var_144 == 1
    var_145 = module_2.thaw(var_18, var_24)
    assert var_145 == 'a'
    var_146 = module_2.thaw(var_20, var_24)
    assert var_146 is None
    var_147 = True
    var_148 = module_2.thaw(var_147, var_24)
    assert var_148 is True
    var_149 = module_2.thaw(var_24, var_24)
    assert var_149 is False
    var_150 = module_2.thaw(var_147, var_24)
    var_151 = module_2.thaw(var_27, var_24)
    assert var_151 == b'abc'
    var_152 = bytearray(var_27)
    var_153 = module_2.thaw(var_152, var_24)
    var_154 = bytearray(var_27)
    var_155 = memoryview(var_27)
    var_156 = module_2.thaw(var_155, var_24)
    var_157 = memoryview(var_27)
    var_158 = slice(var_147, var_1, var_2)
    var_159 = module_2.thaw(var_158, var_24)
    var_160 = slice(var_147, var_1, var_2)
    var_161 = range(var_147, var_1, var_2)
    var_162 = module_2.thaw(var_161, var_24)
    var_163 = range(var_147, var_1, var_2)
    var_164 = Exception()
    var_165 = module_2.thaw(var_164, var_24)
    var_166 = Exception()
    var_167 = lambda x: x
    var_168 = module_2.thaw(var_167, var_24)
    var_169 = [var_147, var_1, var_2]
    var_170 = True
    var_171 = module_2.thaw(var_169, var_170)
    var_172 = {var_170: var_1, var_2: var_9}
    var_173 = True
    var_174 = module_2.thaw(var_172, var_173)
    var_175 = (var_173, var_1, var_2)
    var_176 = True
    var_177 = module_2.thaw(var_175, var_176)
    var_178 = [var_176, var_1, var_2]
    var_179 = set(var_178)
    var_180 = True
    var_181 = module_2.thaw(var_179, var_180)
    var_182 = True
    var_183 = module_2.thaw(var_180, var_182)
    assert var_183 == 1
    var_184 = True
    var_185 = module_2.thaw(var_18, var_184)
    assert var_185 == 'a'
    var_186 = True
    var_187 = module_2.thaw(var_20, var_186)
    assert var_187 is None
    var_188 = True
    var_189 = True
    var_190 = module_2.thaw(var_188, var_189)
    assert var_190 is True
    var_191 = True
    var_192 = module_2.thaw(var_24, var_191)
    assert var_192 is False
    var_193 = True
    var_194 = module_2.thaw(var_191, var_193)
    var_195 = True
    var_196 = True
    var_197 = module_2.thaw(var_27, var_196)
    assert var_197 == b'abc'
    var_198 = bytearray(var_27)
    var_199 = True
    var_200 = module_2.thaw(var_198, var_199)
    var_201 = bytearray(var_27)
    var_202 = memoryview(var_27)
    var_203 = True
    var_204 = module_2.thaw(var_202, var_203)
    var_205 = memoryview(var_27)
    var_206 = slice(var_203, var_1, var_2)
    var_207 = True
    var_208 = module_2.thaw(var_206, var_207)
    var_209 = slice(var_207, var_1, var_2)
    var_210 = range(var_207, var_1, var_2)
    var_211 = True
    var_212 = module_2.thaw(var_210, var_211)
    var_213 = range(var_211, var_1, var_2)
    var_214 = True
    var_215 = True
    var_216 = True
    var_217 = Exception()
    var_218 = True
    var_219 = module_2.thaw(var_217, var_218)
    var_220 = Exception()
    var_221 = lambda x: x
    var_222 = True
    var_223 = module_2.thaw(var_221, var_222)
    var_224 = [var_222, var_1, var_2]
    var_225 = module_2.thaw(var_224, var_24)
    var_226 = {var_222: var_1, var_2: var_9}
    var_227 = module_2.thaw(var_226, var_24)
    var_228 = (var_222, var_1, var_2)
    var_229 = module_2.thaw(var_228, var_24)
    var_230 = [var_222, var_1, var_2]
    var_231 = set(var_230)
    var_232 = module_2.thaw(var_231, var_24)
    var_233 = module_2.thaw(var_222, var_24)
    assert var_233 == 1
    var_234 = module_2.thaw(var_18, var_24)
    assert var_234 == 'a'
    var_235 = module_2.thaw(var_20, var_24)
    assert var_235 is None
    var_236 = True
    var_237 = module_2.thaw(var_236, var_24)
    assert var_237 is True
    var_238 = module_2.thaw(var_24, var_24)
    assert var_238 is False



# Parsed testcases at query #6
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 'new_key'
    var_10 = 'new_value'
    var_11 = {var_6: var_7, var_9: var_10}
    var_12 = module_0.pmap(var_11)
    var_13 = 'All tests passed!'
    var_14 = print(var_13)



# Parsed testcases at query #7
#--------------------------



def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = 'a'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = [var_3]
    var_6 = 2
    var_7 = {var_2: var_6}
    var_8 = module_0.pmap(var_7)
    var_9 = 'All tests passed!'
    var_10 = print(var_9)



# Parsed testcases at query #8
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 'new_key'
    var_10 = 'new_value'
    var_11 = {var_6: var_7, var_9: var_10}
    var_12 = module_0.pmap(var_11)
    var_13 = 'All tests passed!'
    var_14 = print(var_13)



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test passed: list mutation isolated'
    var_5 = print(var_4)
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 'Test passed: dict mutation isolated'
    var_10 = print(var_9)
    var_11 = 'list'
    var_12 = 'dict'
    var_13 = [var_0, var_1, var_2]
    var_14 = 'existing'
    var_15 = {var_14: var_7}
    var_16 = {var_11: var_13, var_12: var_15}
    var_17 = 4
    var_18 = [var_0, var_1, var_2, var_17]
    var_19 = 'new'
    var_20 = {var_14: var_7, var_19: var_7}
    var_21 = {var_11: var_18, var_12: var_20}
    var_22 = 'Test passed: nested mutation isolated'
    var_23 = print(var_22)



# Parsed testcases at query #10
#--------------------------




# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = 'All tests passed!'
    var_8 = print(var_7)



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'All tests passed.'
    var_5 = print(var_4)



# Parsed testcases at query #13
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
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.freeze(var_8)
    var_10 = {var_6: var_0, var_7: var_1}
    var_11 = module_1.pmap(var_10)
    var_12 = {var_0, var_1, var_2}
    var_13 = module_0.freeze(var_12)
    var_14 = {var_0, var_1, var_2}
    var_15 = module_2.pset(var_14)
    var_16 = [var_1, var_2]
    var_17 = (var_0, var_16)
    var_18 = module_0.freeze(var_17)
    var_19 = [var_1, var_2]
    var_20 = [var_0]
    var_21 = [var_0, var_1, var_2]
    var_22 = False
    var_23 = module_0.freeze(var_21, var_22)
    var_24 = [var_0, var_1, var_2]
    var_25 = {var_6: var_0, var_7: var_1}
    var_26 = module_0.freeze(var_25, var_22)
    var_27 = {var_6: var_0, var_7: var_1}
    var_28 = module_1.pmap(var_27)
    var_29 = [var_0, var_1, var_2]
    var_30 = [var_0, var_1, var_2]
    var_31 = {var_6: var_0, var_7: var_1}
    var_32 = module_1.pmap(var_31)
    var_33 = module_0.freeze(var_32)
    var_34 = {var_6: var_0, var_7: var_1}
    var_35 = module_1.pmap(var_34)
    var_36 = {var_0, var_1, var_2}
    var_37 = module_2.pset(var_36)
    var_38 = module_0.freeze(var_37)
    var_39 = {var_0, var_1, var_2}
    var_40 = module_2.pset(var_39)
    var_41 = [var_1, var_2]
    var_42 = [var_1, var_2]
    var_43 = {var_6: var_0}
    var_44 = module_1.pmap(var_43)
    var_45 = [var_44]
    var_46 = module_0.freeze(var_45)
    var_47 = {var_6: var_0}
    var_48 = module_1.pmap(var_47)
    var_49 = [var_48]
    var_50 = [var_0, var_1]
    var_51 = [var_0, var_1]
    var_52 = [var_0, var_1]
    var_53 = [var_0, var_1]
    var_54 = [var_1, var_2]
    var_55 = (var_0, var_54)
    var_56 = module_0.freeze(var_55)
    var_57 = [var_1, var_2]
    var_58 = (var_0, var_1)
    var_59 = 4
    var_60 = (var_2, var_59)
    var_61 = [var_58, var_60]
    var_62 = module_0.freeze(var_61)
    var_63 = (var_0, var_1)
    var_64 = (var_2, var_59)
    var_65 = [var_63, var_64]
    var_66 = (var_0, var_1)
    var_67 = {var_6: var_66}
    var_68 = module_0.freeze(var_67)
    var_69 = (var_0, var_1)
    var_70 = {var_6: var_69}
    var_71 = module_1.pmap(var_70)
    var_72 = {var_6: var_1}
    var_73 = (var_0, var_72)
    var_74 = module_0.freeze(var_73)
    var_75 = {var_6: var_1}
    var_76 = module_1.pmap(var_75)
    var_77 = (var_0, var_76)
    var_78 = {var_0, var_1}
    var_79 = {var_2, var_59}
    var_80 = [var_78, var_79]
    var_81 = module_0.freeze(var_80)
    var_82 = {var_0, var_1}
    var_83 = module_2.pset(var_82)
    var_84 = {var_2, var_59}
    var_85 = module_2.pset(var_84)
    var_86 = [var_83, var_85]
    var_87 = {var_0, var_1}
    var_88 = {var_6: var_87}
    var_89 = module_0.freeze(var_88)
    var_90 = {var_0, var_1}
    var_91 = module_2.pset(var_90)
    var_92 = {var_6: var_91}
    var_93 = module_1.pmap(var_92)
    var_94 = {var_1, var_2}
    var_95 = (var_0, var_94)
    var_96 = module_0.freeze(var_95)
    var_97 = {var_1, var_2}
    var_98 = module_2.pset(var_97)
    var_99 = (var_0, var_98)
    var_100 = [var_0]
    var_101 = [var_0]
    var_102 = [var_0]
    var_103 = [var_0, var_1]
    var_104 = True
    var_105 = [var_104, var_1]
    var_106 = {var_7: var_104}
    var_107 = module_1.pmap(var_106)
    var_108 = {var_6: var_107}
    var_109 = True
    var_110 = module_0.freeze(var_108, var_109)
    var_111 = {var_7: var_109}
    var_112 = module_1.pmap(var_111)
    var_113 = {var_6: var_112}
    var_114 = module_1.pmap(var_113)
    var_115 = [var_1, var_2]
    var_116 = True
    var_117 = [var_1, var_2]
    var_118 = [var_116, var_1]
    var_119 = [var_116, var_1]
    var_120 = {var_7: var_116}
    var_121 = module_1.pmap(var_120)
    var_122 = {var_6: var_121}
    var_123 = module_0.freeze(var_122, var_22)
    var_124 = {var_7: var_116}
    var_125 = module_1.pmap(var_124)
    var_126 = {var_6: var_125}
    var_127 = module_1.pmap(var_126)
    var_128 = [var_1, var_2]
    var_129 = [var_1, var_2]
    var_130 = [var_116, var_1]
    var_131 = [var_2, var_59]
    var_132 = [var_130, var_131]
    var_133 = True
    var_134 = module_0.freeze(var_132, var_133)
    var_135 = [var_133, var_1]
    var_136 = [var_2, var_59]
    var_137 = {var_7: var_133}
    var_138 = {var_6: var_137}
    var_139 = True
    var_140 = module_0.freeze(var_138, var_139)
    var_141 = {var_7: var_139}
    var_142 = module_1.pmap(var_141)
    var_143 = {var_6: var_142}
    var_144 = module_1.pmap(var_143)
    var_145 = (var_1, var_2)
    var_146 = (var_139, var_145)
    var_147 = True
    var_148 = module_0.freeze(var_146, var_147)
    var_149 = [var_147, var_1]
    var_150 = [var_2, var_59]
    var_151 = [var_149, var_150]
    var_152 = module_0.freeze(var_151, var_22)
    var_153 = [var_147, var_1]
    var_154 = [var_2, var_59]
    var_155 = [var_153, var_154]
    var_156 = {var_7: var_147}
    var_157 = {var_6: var_156}
    var_158 = module_0.freeze(var_157, var_22)
    var_159 = {var_7: var_147}
    var_160 = {var_6: var_159}
    var_161 = module_1.pmap(var_160)
    var_162 = (var_1, var_2)
    var_163 = (var_147, var_162)
    var_164 = module_0.freeze(var_163, var_22)
    var_165 = {var_147, var_1}
    var_166 = {var_2, var_59}
    var_167 = [var_165, var_166]
    var_168 = True
    var_169 = module_0.freeze(var_167, var_168)
    var_170 = {var_168, var_1}
    var_171 = module_2.pset(var_170)
    var_172 = {var_2, var_59}
    var_173 = module_2.pset(var_172)
    var_174 = [var_171, var_173]
    var_175 = {var_168, var_1}
    var_176 = {var_6: var_175}
    var_177 = True
    var_178 = module_0.freeze(var_176, var_177)
    var_179 = {var_177, var_1}
    var_180 = module_2.pset(var_179)
    var_181 = {var_6: var_180}
    var_182 = module_1.pmap(var_181)
    var_183 = {var_1, var_2}
    var_184 = (var_177, var_183)
    var_185 = True
    var_186 = module_0.freeze(var_184, var_185)
    var_187 = {var_1, var_2}
    var_188 = module_2.pset(var_187)
    var_189 = (var_185, var_188)
    var_190 = {var_185, var_1}
    var_191 = {var_2, var_59}
    var_192 = [var_190, var_191]
    var_193 = module_0.freeze(var_192, var_22)
    var_194 = {var_185, var_1}
    var_195 = {var_2, var_59}
    var_196 = [var_194, var_195]
    var_197 = {var_185, var_1}
    var_198 = {var_6: var_197}
    var_199 = module_0.freeze(var_198, var_22)
    var_200 = {var_185, var_1}
    var_201 = {var_6: var_200}
    var_202 = module_1.pmap(var_201)
    var_203 = {var_1, var_2}
    var_204 = (var_185, var_203)
    var_205 = module_0.freeze(var_204, var_22)
    var_206 = True
    var_207 = [var_206]
    var_208 = True
    var_209 = [var_208]
    var_210 = True
    var_211 = [var_210]



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'test_mutant passed'
    var_5 = print(var_4)



# Parsed testcases at query #15
#--------------------------




# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test passed!'
    var_5 = print(var_4)



# Parsed testcases at query #17
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.freeze(var_8)
    var_10 = {var_6: var_0, var_7: var_1}
    var_11 = module_1.pmap(var_10)
    var_12 = {var_0, var_1, var_2}
    var_13 = module_0.freeze(var_12)
    var_14 = {var_0, var_1, var_2}
    var_15 = module_2.pset(var_14)
    var_16 = [var_1, var_2]
    var_17 = (var_0, var_16)
    var_18 = module_0.freeze(var_17)
    var_19 = [var_1, var_2]
    var_20 = {var_6: var_0}
    var_21 = {var_1, var_2}
    var_22 = [var_20, var_21]
    var_23 = module_0.freeze(var_22)
    var_24 = {var_6: var_0}
    var_25 = module_1.pmap(var_24)
    var_26 = {var_1, var_2}
    var_27 = module_2.pset(var_26)
    var_28 = [var_25, var_27]
    var_29 = [var_0, var_1, var_2]
    var_30 = False
    var_31 = module_0.freeze(var_29, var_30)
    var_32 = [var_0, var_1, var_2]
    var_33 = [var_0, var_1, var_2]
    var_34 = [var_0, var_1, var_2]
    var_35 = 'All tests passed!'
    var_36 = print(var_35)



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
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 'new_key'
    var_10 = 'new_value'
    var_11 = {var_6: var_7, var_9: var_10}
    var_12 = module_0.pmap(var_11)
    var_13 = 'All tests passed!'
    var_14 = print(var_13)



# Parsed testcases at query #19
#--------------------------



def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = 'a'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = [var_3]
    var_6 = 2
    var_7 = {var_2: var_6}
    var_8 = module_0.pmap(var_7)
    var_9 = 'All tests passed'
    var_10 = print(var_9)



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_0, var_1}
    var_7 = 'All tests passed!'
    var_8 = print(var_7)



# Parsed testcases at query #21
#--------------------------


import pyrsistent._pset as module_1


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = 'b'
    var_6 = {var_4: var_0, var_5: var_1}
    var_7 = module_0.pmap()
    var_8 = {var_0, var_1, var_2}
    var_9 = module_1.pset()
    var_10 = (var_0, var_1, var_2)
    var_11 = tuple()
    var_12 = 'All tests passed!'
    var_13 = print(var_12)



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = 'Result list should be a PVector'
    var_7 = 'Result dict should be a PMap'
    var_8 = 'All tests passed!'
    var_9 = print(var_8)



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_0, var_1, var_2}
    var_8 = 'All tests passed!'
    var_9 = print(var_8)



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = 'Test passed!'
    var_3 = print(var_2)



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_0, var_1, var_2}
    var_8 = 'All tests passed!'
    var_9 = print(var_8)



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = 'Result list should be a pvector'
    var_7 = 'Result dict should be a pmap'
    var_8 = 'All tests passed!'
    var_9 = print(var_8)



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = 'All tests passed!'
    var_7 = print(var_6)



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = 'test_mutant passed'
    var_7 = print(var_6)



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = 1
    var_3 = [var_0, var_2]



# Parsed testcases at query #30
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 'new_key'
    var_10 = 'new_value'
    var_11 = {var_6: var_7, var_9: var_10}
    var_12 = module_0.pmap(var_11)
    var_13 = 'All tests passed!'
    var_14 = print(var_13)



