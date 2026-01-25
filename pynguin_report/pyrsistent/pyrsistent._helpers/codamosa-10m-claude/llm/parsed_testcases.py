####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
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
    var_5 = list(var_4)
    var_6 = 'a'
    var_7 = {var_6: var_2}
    var_8 = [var_0, var_7]
    var_9 = module_0.freeze(var_8)
    var_10 = var_9[var_0]
    var_11 = 'b'
    var_12 = {var_6: var_0, var_11: var_1}
    var_13 = module_0.freeze(var_12)
    var_14 = [var_0, var_1]
    var_15 = {var_6: var_14}
    var_16 = module_0.freeze(var_15)
    var_17 = var_16[var_6]
    var_18 = var_16[var_6]
    var_19 = list(var_18)
    var_20 = {var_0, var_1, var_2}
    var_21 = module_0.freeze(var_20)
    var_22 = set(var_21)
    var_23 = [var_1, var_2]
    var_24 = (var_0, var_23)
    var_25 = module_0.freeze(var_24)
    var_26 = var_25[var_0]
    var_27 = var_25[var_0]
    var_28 = list(var_27)
    var_29 = {var_6: var_1}
    var_30 = (var_0, var_29)
    var_31 = module_0.freeze(var_30)
    var_32 = var_31[var_0]
    var_33 = 42
    var_34 = module_0.freeze(var_33)
    assert var_34 == 42
    var_35 = 'string'
    var_36 = module_0.freeze(var_35)
    assert var_36 == 'string'
    var_37 = None
    var_38 = module_0.freeze(var_37)
    assert var_38 is None
    var_39 = [var_1, var_2]
    var_40 = {var_11: var_39}
    var_41 = [var_0, var_40]
    var_42 = {var_6: var_41}
    var_43 = module_0.freeze(var_42)
    var_44 = var_43[var_6]
    var_45 = var_43[var_6][var_0]
    var_46 = var_43[var_6][var_0][var_11]
    var_47 = [var_0, var_1]
    var_48 = False
    var_49 = [var_0, var_1]
    var_50 = True
    var_51 = {var_6: var_50}
    var_52 = module_1.pmap(var_51)
    var_53 = module_0.freeze(var_52, var_48)
    var_54 = {var_6: var_50}
    var_55 = module_1.pmap(var_54)
    var_56 = True
    var_57 = module_0.freeze(var_55, var_56)
    var_58 = []
    var_59 = module_0.freeze(var_58)
    var_60 = {}
    var_61 = module_0.freeze(var_60)
    var_62 = set()
    var_63 = module_0.freeze(var_62)
    var_64 = ()
    var_65 = module_0.freeze(var_64)



# Parsed testcases at query #2
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_1.pmap(var_6)
    var_8 = {var_1: var_3}
    var_9 = {var_0: var_8}
    var_10 = module_0.freeze(var_9)
    var_11 = {var_1: var_3}
    var_12 = module_1.pmap(var_11)
    var_13 = {var_0: var_12}
    var_14 = module_1.pmap(var_13)
    var_15 = 3
    var_16 = [var_2, var_3, var_15]
    var_17 = module_0.freeze(var_16)
    var_18 = [var_2, var_3, var_15]
    var_19 = [var_3, var_15]
    var_20 = [var_2, var_19]
    var_21 = module_0.freeze(var_20)
    var_22 = [var_3, var_15]
    var_23 = {var_0: var_15}
    var_24 = [var_2, var_23]
    var_25 = module_0.freeze(var_24)
    var_26 = {var_0: var_15}
    var_27 = module_1.pmap(var_26)
    var_28 = [var_2, var_27]
    var_29 = {var_2, var_3}
    var_30 = module_0.freeze(var_29)
    var_31 = [var_2, var_3]
    var_32 = module_2.pset(var_31)
    var_33 = (var_2, var_3)
    var_34 = module_0.freeze(var_33)
    var_35 = [var_3, var_15]
    var_36 = (var_2, var_35)
    var_37 = module_0.freeze(var_36)
    var_38 = [var_3, var_15]
    var_39 = {var_0: var_15}
    var_40 = (var_2, var_39)
    var_41 = module_0.freeze(var_40)
    var_42 = {var_0: var_15}
    var_43 = module_1.pmap(var_42)
    var_44 = (var_2, var_43)
    var_45 = {var_0: var_2}
    var_46 = module_1.pmap(var_45)
    var_47 = {var_1: var_3}
    var_48 = module_1.pmap(var_47)
    var_49 = {var_0: var_48}
    var_50 = module_1.pmap(var_49)
    var_51 = 42
    var_52 = module_0.freeze(var_51)
    assert var_52 == 42
    var_53 = 'string'
    var_54 = module_0.freeze(var_53)
    assert var_54 == 'string'
    var_55 = 3.14
    var_56 = module_0.freeze(var_55)
    var_57 = None
    var_58 = module_0.freeze(var_57)
    assert var_58 is None
    var_59 = [var_2, var_3, var_15]
    var_60 = False
    var_61 = {var_0: var_2}
    var_62 = module_1.pmap(var_61)
    var_63 = module_0.freeze(var_62, var_60)
    var_64 = [var_2, var_3, var_15]
    var_65 = True
    var_66 = [var_65, var_3, var_15]
    var_67 = {var_0: var_65}
    var_68 = module_1.pmap(var_67)
    var_69 = True
    var_70 = module_0.freeze(var_68, var_69)
    var_71 = {var_0: var_69}
    var_72 = module_1.pmap(var_71)
    var_73 = 'list'
    var_74 = 'dict'
    var_75 = 'tuple'
    var_76 = 'set'
    var_77 = 'nested'
    var_78 = {var_77: var_74}
    var_79 = [var_69, var_3, var_78]
    var_80 = 'key'
    var_81 = [var_69, var_3, var_15]
    var_82 = {var_80: var_81}
    var_83 = [var_3, var_15]
    var_84 = (var_69, var_83)
    var_85 = {var_69, var_3, var_15}
    var_86 = {var_73: var_79, var_74: var_82, var_75: var_84, var_76: var_85}
    var_87 = module_0.freeze(var_86)
    var_88 = {var_77: var_74}
    var_89 = module_1.pmap(var_88)
    var_90 = [var_69, var_3, var_89]
    var_91 = [var_69, var_3, var_15]
    var_92 = [var_3, var_15]
    var_93 = [var_69, var_3, var_15]
    var_94 = module_2.pset(var_93)
    var_95 = []
    var_96 = module_0.freeze(var_95)
    var_97 = []
    var_98 = {}
    var_99 = module_0.freeze(var_98)
    var_100 = {}
    var_101 = module_1.pmap(var_100)
    var_102 = ()
    var_103 = module_0.freeze(var_102)
    var_104 = set()
    var_105 = module_0.freeze(var_104)
    var_106 = []
    var_107 = module_2.pset(var_106)
    var_108 = (var_69, var_3)
    var_109 = 'value'
    var_110 = {var_108: var_109}
    var_111 = module_0.freeze(var_110)
    var_112 = (var_69, var_3)
    var_113 = {var_112: var_109}
    var_114 = module_1.pmap(var_113)



# Parsed testcases at query #3
#--------------------------


import pyrsistent._pmap as module_0
import pyrsistent._pvector as module_1
import pyrsistent._helpers as module_2
import pyrsistent._pset as module_3

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.m()
    var_4 = module_1.v()
    var_5 = (var_0, var_4)
    var_6 = module_2.thaw(var_5)
    var_7 = module_0.m()
    var_8 = module_0.m()
    var_9 = 42
    var_10 = module_2.thaw(var_9)
    assert var_10 == 42
    var_11 = 'string'
    var_12 = module_2.thaw(var_11)
    assert var_12 == 'string'
    var_13 = None
    var_14 = module_2.thaw(var_13)
    assert var_14 is None
    var_15 = module_0.m()
    var_16 = 'a'
    var_17 = {var_16: var_0}
    var_18 = False
    var_19 = module_2.thaw(var_17, var_18)
    var_20 = [var_0, var_1, var_2]
    var_21 = module_2.thaw(var_20, var_18)
    var_22 = [var_0, var_1]
    var_23 = {var_16: var_22}
    var_24 = True
    var_25 = module_2.thaw(var_23, var_24)
    var_26 = True
    var_27 = module_0.m()
    var_28 = module_3.s()
    var_29 = module_2.thaw(var_28)
    var_30 = set()
    var_31 = module_1.v()
    var_32 = module_2.thaw(var_31)
    var_33 = module_0.m()
    var_34 = module_2.thaw(var_33)
    var_35 = ()
    var_36 = module_2.thaw(var_35)
    var_37 = 4
    var_38 = 5
    var_39 = module_0.m()
    var_40 = 'data'
    var_41 = 'metadata'
    var_42 = 'nested'
    var_43 = [var_2, var_37]
    var_44 = {var_42: var_43}
    var_45 = [var_26, var_1, var_44]
    var_46 = 'version'
    var_47 = {var_46: var_26}
    var_48 = {var_40: var_45, var_41: var_47}



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'Test the mutant decorator'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = [var_1, var_2, var_3]
    var_6 = 'a'
    var_7 = {var_6: var_1}
    var_8 = 'list'
    var_9 = [var_1, var_2, var_3]
    var_10 = {var_8: var_9}
    var_11 = [var_1, var_2]
    var_12 = 'key'
    var_13 = {var_12: var_3}
    var_14 = [var_1, var_2]
    var_15 = 5
    var_16 = {var_1, var_2, var_3}
    var_17 = (var_1, var_2, var_3)



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'Test the mutant decorator'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 'a'
    var_6 = {var_5: var_1}
    var_7 = 'list'
    var_8 = 'set'
    var_9 = [var_1, var_2]
    var_10 = 4
    var_11 = {var_3, var_10}
    var_12 = {var_7: var_9, var_8: var_11}
    var_13 = [var_1, var_2]
    var_14 = 'x'
    var_15 = {var_14: var_1}
    var_16 = 'b'
    var_17 = 'key'
    var_18 = [var_1]
    var_19 = {var_5: var_1}
    var_20 = {var_2, var_3}
    var_21 = 'dict'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'Test the mutant decorator for freezing arguments and return values'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_5: var_1, var_6: var_2}
    var_8 = 'list'
    var_9 = 'dict'
    var_10 = [var_1, var_2]
    var_11 = 'x'
    var_12 = 10
    var_13 = {var_11: var_12}
    var_14 = {var_8: var_10, var_9: var_13}
    var_15 = [var_1, var_2]
    var_16 = 'key'
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = 0
    var_20 = [var_1, var_2]
    var_21 = 4
    var_22 = [var_3, var_21]
    var_23 = {var_1, var_2, var_3}
    var_24 = [var_2, var_3]
    var_25 = (var_1, var_24)



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = 'list'
    var_7 = [var_0, var_1, var_2]
    var_8 = {var_6: var_7}
    var_9 = [var_0, var_1]
    var_10 = 4
    var_11 = [var_2, var_10]
    var_12 = 'key'
    var_13 = 'value'
    var_14 = {var_12: var_13}
    var_15 = 'test'
    var_16 = {var_0, var_1, var_2}
    var_17 = (var_0, var_1, var_2)



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'Test the mutant decorator'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = [var_1, var_2, var_3]
    var_6 = 'a'
    var_7 = {var_6: var_1}
    var_8 = [var_1, var_2, var_3]
    var_9 = 'list'
    var_10 = 'dict'
    var_11 = [var_1, var_2]
    var_12 = 'key'
    var_13 = 'value'
    var_14 = {var_12: var_13}
    var_15 = {var_9: var_11, var_10: var_14}
    var_16 = [var_1, var_2]
    var_17 = 'x'
    var_18 = {var_17: var_1}
    var_19 = 'b'
    var_20 = {var_1, var_2, var_3}
    var_21 = [var_2, var_3]
    var_22 = (var_1, var_21)



# Parsed testcases at query #9
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'Unit tests for the mutant decorator.'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 99
    var_6 = [var_5, var_2, var_3]
    var_7 = 'key'
    var_8 = 'original'
    var_9 = {var_7: var_8}
    var_10 = 'modified'
    var_11 = {var_7: var_10}
    var_12 = module_0.pmap(var_11)
    var_13 = [var_1, var_2]
    var_14 = 4
    var_15 = [var_3, var_14]
    var_16 = 'a'
    var_17 = 'b'
    var_18 = {var_17: var_3}
    var_19 = [var_1, var_2, var_18]
    var_20 = {var_16: var_19}
    var_21 = [var_1, var_2]
    var_22 = [var_3, var_14]
    var_23 = {var_1, var_2, var_3}
    var_24 = [var_2, var_3]
    var_25 = (var_1, var_24)



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'Test the mutant decorator for freezing arguments and return values.'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 'a'
    var_6 = {var_5: var_1}
    var_7 = 'list'
    var_8 = 'nested'
    var_9 = [var_1, var_2]
    var_10 = 'value'
    var_11 = 10
    var_12 = {var_10: var_11}
    var_13 = {var_7: var_9, var_8: var_12}
    var_14 = [var_1, var_2]
    var_15 = 'x'
    var_16 = {var_15: var_11}
    var_17 = 5
    var_18 = [var_1, var_2]
    var_19 = 'initial'
    var_20 = {var_19: var_10}
    var_21 = {var_1, var_2, var_3}
    var_22 = (var_1, var_2, var_3)



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'Test the mutant decorator'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = [var_1, var_2, var_3]
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_1, var_7: var_2}
    var_9 = {var_6: var_1}
    var_10 = {var_7: var_2}
    var_11 = [var_9, var_10]
    var_12 = len(var_11)
    assert var_12 == 2
    var_13 = [var_1, var_2]
    var_14 = 'key'
    var_15 = 'value'
    var_16 = {var_14: var_15}
    var_17 = {var_1, var_2, var_3}



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'Test the mutant decorator'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 'list'
    var_9 = 'set'
    var_10 = [var_1, var_2, var_3]
    var_11 = 4
    var_12 = 5
    var_13 = {var_11, var_12}
    var_14 = {var_8: var_10, var_9: var_13}
    var_15 = [var_1, var_2]
    var_16 = 'x'
    var_17 = {var_16: var_1}
    var_18 = 'a'
    var_19 = 'b'
    var_20 = [var_2, var_3]
    var_21 = (var_1, var_20)
    var_22 = [var_1]
    var_23 = {var_16: var_2}
    var_24 = {var_3, var_11}
    var_25 = 0



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'Test the mutant decorator freezes arguments and return values.'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 'a'
    var_6 = {var_5: var_1}
    var_7 = 'items'
    var_8 = 'inner'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = [var_1, var_2, var_10]
    var_12 = {var_7: var_11}
    var_13 = [var_1, var_2]
    var_14 = 'x'
    var_15 = 10
    var_16 = {var_14: var_15}
    var_17 = 'b'
    var_18 = {var_1, var_2, var_3}
    var_19 = [var_2, var_3]
    var_20 = (var_1, var_19)



