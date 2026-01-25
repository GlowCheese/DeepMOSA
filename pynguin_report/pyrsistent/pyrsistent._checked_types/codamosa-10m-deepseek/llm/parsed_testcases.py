# Parsed testcases at query #5
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = True
    var_5 = True
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = 1
    var_9 = 'a'
    var_10 = 3
    var_11 = [var_8, var_9, var_10]
    var_12 = lambda x: (x > 0, 'Negative')
    var_13 = 1
    var_14 = -2
    var_15 = 3
    var_16 = [var_13, var_14, var_15]
    var_17 = lambda x: (x > 0, 'Negative')
    var_18 = [var_5, var_16, var_2]
    var_19 = [var_5, var_16, var_2]
    var_20 = [var_5, var_16, var_2]
    var_21 = 'a'
    var_22 = 'b'
    var_23 = 'c'
    var_24 = [var_21, var_22, var_23]
    var_25 = [var_5, var_24, var_2]
    var_26 = 'a'
    var_27 = 'b'
    var_28 = 'c'
    var_29 = [var_26, var_27, var_28]
    var_30 = [var_5, var_29, var_2]
    var_31 = True
    var_32 = 'a'
    var_33 = 'b'
    var_34 = 'c'
    var_35 = [var_32, var_33, var_34]
    var_36 = True
    var_37 = [var_31, var_35, var_36]
    var_38 = set()
    var_39 = 'a'
    var_40 = 'b'
    var_41 = 'c'
    var_42 = [var_39, var_40, var_41]
    var_43 = set()
    var_44 = [var_31, var_42, var_43]
    var_45 = True
    var_46 = set()
    var_47 = 'a'
    var_48 = 'b'
    var_49 = 'c'
    var_50 = [var_47, var_48, var_49]
    var_51 = True
    var_52 = set()
    var_53 = lambda format, value: value
    var_54 = [var_45, var_50, var_51]
    var_55 = True
    var_56 = set()
    var_57 = lambda format, value: value
    var_58 = 'a'
    var_59 = 'b'
    var_60 = 'c'
    var_61 = [var_58, var_59, var_60]
    var_62 = True
    var_63 = set()
    var_64 = lambda format, value: str(value)
    var_65 = [var_55, var_61, var_62]
    var_66 = True
    var_67 = set()
    var_68 = lambda format, value: str(value)



# Parsed testcases at query #6
#--------------------------




# Parsed testcases at query #7
#--------------------------




# Parsed testcases at query #8
#--------------------------




# Parsed testcases at query #9
#--------------------------




# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = -1
    var_6 = 'All tests passed!'
    var_7 = print(var_6)



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = -1
    var_6 = 'All tests passed.'
    var_7 = print(var_6)



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -5
    var_2 = 15
    var_3 = 'All tests passed.'
    var_4 = print(var_3)



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -5
    var_2 = 4
    var_3 = 3
    var_4 = 'All tests passed'
    var_5 = print(var_4)



# Parsed testcases at query #14
#--------------------------




# Parsed testcases at query #15
#--------------------------


import pyrsistent._checked_types as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.InvariantException()
    var_1 = str(var_0)
    assert var_1 == ', invariant_errors=[], missing_fields=[]'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.InvariantException(var_5)
    var_7 = str(var_6)
    assert var_7 == ', invariant_errors=[1, 2, 3], missing_fields=[]'
    var_8 = 'a'
    var_9 = 'b'
    var_10 = [var_8, var_9]
    var_11 = module_0.InvariantException(missing_fields=var_10)
    var_12 = str(var_11)
    assert var_12 == ', invariant_errors=[], missing_fields=[a, b]'
    var_13 = [var_2, var_3, var_4]
    var_14 = [var_8, var_9]
    var_15 = module_0.InvariantException(var_13, var_14)
    var_16 = str(var_15)
    assert var_16 == ', invariant_errors=[1, 2, 3], missing_fields=[a, b]'
    var_17 = lambda : var_2
    var_18 = lambda : var_3
    var_19 = [var_17, var_18]
    var_20 = module_0.InvariantException(var_19)
    var_21 = str(var_20)
    assert var_21 == ', invariant_errors=[1, 2], missing_fields=[]'
    var_22 = lambda : var_2
    var_23 = [var_22, var_3]
    var_24 = module_0.InvariantException(var_23)
    var_25 = str(var_24)
    assert var_25 == ', invariant_errors=[1, 2], missing_fields=[]'
    var_26 = []
    var_27 = []
    var_28 = module_0.InvariantException(var_26, var_27)
    var_29 = str(var_28)
    assert var_29 == ', invariant_errors=[], missing_fields=[]'
    var_30 = (var_2, var_3, var_4)
    var_31 = (var_8, var_9)
    var_32 = module_0.InvariantException(var_30, var_31)
    var_33 = str(var_32)
    assert var_33 == ', invariant_errors=[1, 2, 3], missing_fields=[a, b]'
    var_34 = {var_2, var_3, var_4}
    var_35 = {var_8, var_9}
    var_36 = module_0.InvariantException(var_34, var_35)
    var_37 = str(var_36)
    assert var_37 == ', invariant_errors=[1, 2, 3], missing_fields=[a, b]'
    var_38 = [var_2, var_3, var_4]
    var_39 = [var_8, var_9]
    var_40 = str(var_36)
    assert var_40 == ', invariant_errors=[1, 2, 3], missing_fields=[a, b]'
    var_41 = []
    var_42 = []
    var_43 = str(var_36)
    assert var_43 == ', invariant_errors=[], missing_fields=[]'
    var_44 = set()
    var_45 = set()
    var_46 = module_0.InvariantException(var_44, var_45)
    var_47 = str(var_46)
    assert var_47 == ', invariant_errors=[], missing_fields=[]'
    var_48 = ()
    var_49 = ()
    var_50 = module_0.InvariantException(var_48, var_49)
    var_51 = str(var_50)
    assert var_51 == ', invariant_errors=[], missing_fields=[]'
    var_52 = []
    var_53 = []
    var_54 = module_0.InvariantException(var_52, var_53)
    var_55 = str(var_54)
    assert var_55 == ', invariant_errors=[], missing_fields=[]'
    var_56 = {}
    var_57 = {}
    var_58 = module_0.InvariantException(var_56, var_57)
    var_59 = str(var_58)
    assert var_59 == ', invariant_errors=[], missing_fields=[]'
    var_60 = ''
    var_61 = module_0.InvariantException(var_60, var_60)
    var_62 = str(var_61)
    assert var_62 == ', invariant_errors=[], missing_fields=[]'
    var_63 = b''
    var_64 = module_0.InvariantException(var_63, var_63)
    var_65 = str(var_64)
    assert var_65 == ', invariant_errors=[], missing_fields=[]'
    var_66 = bytearray()
    var_67 = bytearray()
    var_68 = module_0.InvariantException(var_66, var_67)
    var_69 = str(var_68)
    assert var_69 == ', invariant_errors=[], missing_fields=[]'
    var_70 = memoryview(var_63)
    var_71 = memoryview(var_63)
    var_72 = module_0.InvariantException(var_70, var_71)
    var_73 = str(var_72)
    assert var_73 == ', invariant_errors=[], missing_fields=[]'
    var_74 = 0
    var_75 = range(var_74)
    var_76 = range(var_74)
    var_77 = module_0.InvariantException(var_75, var_76)
    var_78 = str(var_77)
    assert var_78 == ', invariant_errors=[], missing_fields=[]'
    var_79 = zip()
    var_80 = zip()
    var_81 = module_0.InvariantException(var_79, var_80)
    var_82 = str(var_81)
    assert var_82 == ', invariant_errors=[], missing_fields=[]'
    var_83 = lambda x: x
    var_84 = []
    var_85 = map(var_83, var_84)
    var_86 = lambda x: x
    var_87 = []
    var_88 = map(var_86, var_87)
    var_89 = module_0.InvariantException(var_85, var_88)
    var_90 = str(var_89)
    assert var_90 == ', invariant_errors=[], missing_fields=[]'
    var_91 = lambda x: x
    var_92 = []
    var_93 = filter(var_91, var_92)
    var_94 = lambda x: x
    var_95 = []
    var_96 = filter(var_94, var_95)
    var_97 = module_0.InvariantException(var_93, var_96)
    var_98 = str(var_97)
    assert var_98 == ', invariant_errors=[], missing_fields=[]'
    var_99 = []
    var_100 = enumerate(var_99)
    var_101 = []
    var_102 = enumerate(var_101)
    var_103 = module_0.InvariantException(var_100, var_102)
    var_104 = str(var_103)
    assert var_104 == ', invariant_errors=[], missing_fields=[]'
    var_105 = []
    var_106 = reversed(var_105)
    var_107 = []
    var_108 = reversed(var_107)
    var_109 = module_0.InvariantException(var_106, var_108)
    var_110 = str(var_109)
    assert var_110 == ', invariant_errors=[], missing_fields=[]'
    var_111 = slice(var_74)
    var_112 = slice(var_74)
    var_113 = module_0.InvariantException(var_111, var_112)
    var_114 = str(var_113)
    assert var_114 == ', invariant_errors=[], missing_fields=[]'
    var_115 = complex()
    var_116 = complex()
    var_117 = module_0.InvariantException(var_115, var_116)
    var_118 = str(var_117)
    assert var_118 == ', invariant_errors=[], missing_fields=[]'
    var_119 = float()
    var_120 = float()
    var_121 = module_0.InvariantException(var_119, var_120)
    var_122 = str(var_121)
    assert var_122 == ', invariant_errors=[], missing_fields=[]'
    var_123 = int()
    var_124 = int()
    var_125 = module_0.InvariantException(var_123, var_124)
    var_126 = str(var_125)
    assert var_126 == ', invariant_errors=[], missing_fields=[]'
    var_127 = bool()
    var_128 = bool()
    var_129 = module_0.InvariantException(var_127, var_128)
    var_130 = str(var_129)
    assert var_130 == ', invariant_errors=[], missing_fields=[]'
    var_131 = None
    var_132 = module_0.InvariantException(var_131, var_131)
    var_133 = str(var_132)
    assert var_133 == ', invariant_errors=[], missing_fields=[]'
    var_134 = module_1.object()
    var_135 = module_1.object()
    var_136 = module_0.InvariantException(var_134, var_135)
    var_137 = str(var_136)
    assert var_137 == ', invariant_errors=[], missing_fields=[]'
    var_138 = str(var_136)
    assert var_138 == ', invariant_errors=[], missing_fields=[]'
    var_139 = lambda : var_131
    var_140 = lambda : var_131
    var_141 = module_0.InvariantException(var_139, var_140)
    var_142 = str(var_141)
    assert var_142 == ', invariant_errors=[], missing_fields=[]'
    var_143 = str(var_141)
    assert var_143 == ', invariant_errors=[], missing_fields=[]'
    var_144 = 'sys'
    var_145 = __import__(var_144)
    var_146 = __import__(var_144)
    var_147 = module_0.InvariantException(var_145, var_146)
    var_148 = str(var_147)
    assert var_148 == ', invariant_errors=[], missing_fields=[]'
    var_149 = property()
    var_150 = property()
    var_151 = module_0.InvariantException(var_149, var_150)
    var_152 = str(var_151)
    assert var_152 == ', invariant_errors=[], missing_fields=[]'
    var_153 = lambda : var_131
    var_154 = staticmethod(var_153)
    var_155 = lambda : var_131
    var_156 = staticmethod(var_155)
    var_157 = module_0.InvariantException(var_154, var_156)
    var_158 = str(var_157)
    assert var_158 == ', invariant_errors=[], missing_fields=[]'
    var_159 = lambda : var_131
    var_160 = classmethod(var_159)
    var_161 = lambda : var_131
    var_162 = classmethod(var_161)
    var_163 = module_0.InvariantException(var_160, var_162)



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = 'All tests passed.'
    var_6 = print(var_5)



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = -1
    var_6 = -5
    var_7 = 'All tests passed!'
    var_8 = print(var_7)



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -5
    var_2 = -5
    var_3 = 15
    var_4 = 'All tests passed.'
    var_5 = print(var_4)



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = 1
    var_5 = -2
    var_6 = 12
    var_7 = 'All wrap_invariant tests passed!'
    var_8 = print(var_7)



# Parsed testcases at query #20
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = 123
    var_6 = module_0.maybe_parse_user_type(var_5)
    var_7 = 'All tests passed!'
    var_8 = print(var_7)



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -5
    var_2 = 15
    var_3 = 'All tests passed!'
    var_4 = print(var_3)



# Parsed testcases at query #22
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = 'str'
    var_6 = [var_0, var_5]
    var_7 = module_0.maybe_parse_user_type(var_6)
    var_8 = 123
    var_9 = module_0.maybe_parse_user_type(var_8)
    var_10 = 'All tests passed!'
    var_11 = print(var_10)



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = 'All tests passed.'
    var_6 = print(var_5)



# Parsed testcases at query #24
#--------------------------




# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -5
    var_2 = 15
    var_3 = 'All tests passed'
    var_4 = print(var_3)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 1



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 1



# Parsed testcases at query #3
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'str'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = 'float'



# Parsed testcases at query #4
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'unittest.TestCase'
    var_1 = module_0.get_type(var_0)
    var_2 = 'unittest'
    var_3 = __import__(var_2)
    var_4 = var_3.TestCase



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 15
    var_2 = -5
    var_3 = 25
    var_4 = -5



# Parsed testcases at query #6
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}
    var_4 = 'a'
    var_5 = 1.0
    var_6 = {var_4: var_5}
    var_7 = 1
    var_8 = 'a'
    var_9 = {var_7: var_8}
    var_10 = 1
    var_11 = 1.5
    var_12 = {var_10: var_11}
    var_13 = 3
    var_14 = {var_13: var_13}
    var_15 = {var_10: var_10, var_11: var_11}
    var_16 = module_0.pmap(var_15)



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 1.5
    var_4 = 2.25
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'a'
    var_7 = 1.5
    var_8 = {var_6: var_7}
    var_9 = 1
    var_10 = 'a'
    var_11 = {var_9: var_10}
    var_12 = 1
    var_13 = 2.5
    var_14 = {var_12: var_13}



# Parsed testcases at query #8
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'error1'
    var_1 = lambda : var_0
    var_2 = 'error2'
    var_3 = lambda : var_2
    var_4 = [var_1, var_3]
    var_5 = 'field1'
    var_6 = 'field2'
    var_7 = [var_5, var_6]
    var_8 = module_0.InvariantException(var_4, var_7)
    var_9 = str(var_8)
    assert var_9 == "InvariantException(invariant_errors=['error1', 'error2'], missing_fields=['field1', 'field2'])"



# Parsed testcases at query #9
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'msg'
    var_1 = 'args'
    var_2 = 'Invariant failed'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = (var_3, var_4, var_5)
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = 'a'
    var_9 = 'b'
    var_10 = 'c'
    var_11 = (var_8, var_9, var_10)
    var_12 = [var_7]
    var_13 = module_0.InvariantException(var_12, var_11)
    var_14 = str(var_13)
    assert var_14 == ", invariant_errors=[{'msg': 'Invariant failed', 'args': (1, 2, 3)}], missing_fields=[a, b, c]"
    var_15 = module_0.InvariantException()
    var_16 = str(var_15)
    assert var_16 == ', invariant_errors=[], missing_fields=[]'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -3
    var_2 = -3
    var_3 = 15
    var_4 = -5



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant_derived'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 'invariant_base'
    var_6 = var_0[var_1]
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 0
    var_9 = var_0[var_1][var_8]
    var_10 = None
    var_11 = 1
    var_12 = var_0[var_1][var_11]



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 5



# Parsed testcases at query #13
#--------------------------




# Parsed testcases at query #14
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = 1
    var_3 = 'str'
    var_4 = (var_0, var_3)
    var_5 = module_0.maybe_parse_user_type(var_4)
    var_6 = 123
    var_7 = module_0.maybe_parse_user_type(var_6)



# Parsed testcases at query #15
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = module_0.get_type(var_0)
    var_2 = 'os.path'
    var_3 = module_0.get_type(var_2)
    var_4 = []
    var_5 = __import__(var_2, fromlist=var_4)



# Parsed testcases at query #16
#--------------------------




# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 5



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant1'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 'invariant2'
    var_6 = var_0[var_1]
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 'invariant3'
    var_9 = var_0[var_1]
    var_10 = len(var_9)
    assert var_10 == 1



# Parsed testcases at query #19
#--------------------------




# Parsed testcases at query #20
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.get_type(var_0)
    var_2 = 'math.sqrt'
    var_3 = module_0.get_type(var_2)
    var_4 = var_3.__name__
    assert var_4 == 'sqrt'
    var_5 = 'nonexistent.module.Class'
    var_6 = module_0.get_type(var_5)
    var_7 = 'All get_type tests passed'
    var_8 = print(var_7)



# Parsed testcases at query #21
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = module_0.get_type(var_0)
    var_2 = 'builtins.str'
    var_3 = module_0.get_type(var_2)
    var_4 = 'nonexistent.module.NonExistentClass'
    var_5 = module_0.get_type(var_4)



# Parsed testcases at query #22
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'TestTuple'
    var_1 = 'field1 field2'
    var_2 = 'non_existent_module.NonExistentClass'
    var_3 = module_0.get_type(var_2)
    var_4 = 'not_a_valid_type_name'
    var_5 = module_0.get_type(var_4)



# Parsed testcases at query #23
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'str'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = 1
    var_3 = 123
    var_4 = module_0.maybe_parse_user_type(var_3)



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = -1



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant'
    var_3 = 0
    var_4 = var_0[var_1][var_3]
    var_5 = callable(var_4)
    var_6 = var_0[var_1][var_3]
    var_7 = {}
    var_8 = var_7[var_1]
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = var_7[var_1][var_3]
    var_11 = 1
    var_12 = var_7[var_1][var_11]



# Parsed testcases at query #26
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'foo'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = 123
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = 'All tests passed!'
    var_5 = print(var_4)



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 15
    var_1 = 5
    var_2 = 25



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = 1.5
    var_7 = {var_5: var_6}
    var_8 = 1
    var_9 = 'a'
    var_10 = {var_8: var_9}
    var_11 = {}
    var_12 = 0



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = 'All tests passed!'
    var_6 = print(var_5)



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = -2
    var_4 = 3
    var_5 = -3
    var_6 = 'All tests passed!'
    var_7 = print(var_6)



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 15
    var_3 = -1
    var_4 = -1
    var_5 = 'All test cases passed!'
    var_6 = print(var_5)



# Parsed testcases at query #32
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'str'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = 123
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = 123
    var_5 = [var_4]
    var_6 = module_0.maybe_parse_user_type(var_5)



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 1.5
    var_4 = 2.25
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'a'
    var_7 = 1.5
    var_8 = {var_6: var_7}
    var_9 = 1
    var_10 = 'a'
    var_11 = {var_9: var_10}
    var_12 = 1
    var_13 = 2.5
    var_14 = {var_12: var_13}



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -5
    var_2 = 15
    var_3 = -5
    var_4 = -5



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant1'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 'invariant2'
    var_6 = var_0[var_1]
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 'invariants'
    var_9 = 'invalid_invariant'
    var_10 = {}
    var_11 = var_10[var_8]
    var_12 = len(var_11)
    assert var_12 == 1
    var_13 = var_10[var_8]
    var_14 = len(var_13)
    assert var_14 == 2
    var_15 = 'invariant3'
    var_16 = var_10[var_8]
    var_17 = len(var_16)
    assert var_17 == 3
    var_18 = {}
    var_19 = 'invariant4'
    var_20 = var_18[var_8]
    var_21 = len(var_20)
    assert var_21 == 1



# Parsed testcases at query #36
#--------------------------


def test_case_0():
    var_0 = 1



# Parsed testcases at query #37
#--------------------------


def test_case_0():
    var_0 = 1



# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = 1



# Parsed testcases at query #39
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'int'
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = 123
    var_5 = module_0.maybe_parse_user_type(var_4)
    var_6 = ()
    var_7 = module_0.maybe_parse_user_type(var_6)
    var_8 = 'All test cases passed!'
    var_9 = print(var_8)



# Parsed testcases at query #40
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant1'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 'invariant2'
    var_6 = var_0[var_1]
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = {}
    var_9 = 0
    var_10 = var_8[var_1][var_9]
    var_11 = None
    var_12 = {}
    var_13 = var_12[var_1][var_9]
    var_14 = {}
    var_15 = var_14[var_1][var_9]
    var_16 = 1
    var_17 = var_14[var_1][var_16]
    var_18 = {}
    var_19 = 'invariant3'
    var_20 = var_18[var_1]
    var_21 = len(var_20)
    assert var_21 == 3
    var_22 = var_18[var_1][var_9]
    var_23 = var_18[var_1][var_16]
    var_24 = 2
    var_25 = var_18[var_1][var_24]
    var_26 = {}
    var_27 = 'invariant'
    var_28 = var_26[var_1][var_9]
    var_29 = {}
    var_30 = var_29[var_1][var_9]
    var_31 = {}
    var_32 = var_31[var_1][var_9]
    var_33 = {}
    var_34 = var_33[var_1][var_9]
    var_35 = {}
    var_36 = var_35[var_1][var_9]
    var_37 = {}
    var_38 = var_37[var_1][var_9]
    var_39 = {}
    var_40 = var_39[var_1][var_9]
    var_41 = {}
    var_42 = var_41[var_1][var_9]
    var_43 = {}
    var_44 = var_43[var_1][var_9]
    var_45 = {}
    var_46 = var_45[var_1][var_9]
    var_47 = {}
    var_48 = var_47[var_1][var_9]
    var_49 = {}
    var_50 = var_49[var_1][var_9]
    var_51 = {}
    var_52 = var_51[var_1][var_9]
    var_53 = {}



# Parsed testcases at query #41
#--------------------------


def test_case_0():
    var_0 = 5



# Parsed testcases at query #42
#--------------------------




# Parsed testcases at query #43
#--------------------------


def test_case_0():
    var_0 = 6
    var_1 = 4
    var_2 = 7
    var_3 = 11
    var_4 = 3



# Parsed testcases at query #44
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = '_invariants'
    var_2 = 'invariant1'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = {}
    var_6 = 'invariant2'
    var_7 = var_5[var_1]
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = {}
    var_10 = var_9[var_1]
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = {}
    var_13 = var_12[var_1]
    var_14 = len(var_13)
    assert var_14 == 1
    var_15 = 'not_callable'
    var_16 = {}
    var_17 = '_invariants'
    var_18 = 'invariant3'
    var_19 = {}
    var_20 = 'invariant4'
    var_21 = var_19[var_17]
    var_22 = len(var_21)
    assert var_22 == 1
    var_23 = {}
    var_24 = ()
    var_25 = 'invariant'
    var_26 = module_0.store_invariants(var_23, var_24, var_17, var_25)
    var_27 = {}
    var_28 = 'invariant5'
    var_29 = var_27[var_17]
    var_30 = len(var_29)
    assert var_30 == 1
    var_31 = {}
    var_32 = 'invariant6'
    var_33 = var_31[var_17]
    var_34 = len(var_33)
    assert var_34 == 1
    var_35 = {}
    var_36 = 'invariant7'
    var_37 = var_35[var_17]
    var_38 = len(var_37)
    assert var_38 == 1
    var_39 = {}
    var_40 = 'invariant8'
    var_41 = var_39[var_17]
    var_42 = len(var_41)
    assert var_42 == 1
    var_43 = {}
    var_44 = 'invariant9'
    var_45 = var_43[var_17]
    var_46 = len(var_45)
    assert var_46 == 1
    var_47 = {}
    var_48 = 'invariant10'
    var_49 = var_47[var_17]
    var_50 = len(var_49)
    assert var_50 == 1
    var_51 = {}
    var_52 = 'invariant11'
    var_53 = var_51[var_17]
    var_54 = len(var_53)
    assert var_54 == 1
    var_55 = {}
    var_56 = 'invariant12'
    var_57 = var_55[var_17]
    var_58 = len(var_57)
    assert var_58 == 1
    var_59 = {}
    var_60 = 'invariant13'
    var_61 = var_59[var_17]
    var_62 = len(var_61)
    assert var_62 == 1
    var_63 = {}
    var_64 = 'invariant14'
    var_65 = var_63[var_17]
    var_66 = len(var_65)
    assert var_66 == 1
    var_67 = {}
    var_68 = 'invariant15'
    var_69 = var_67[var_17]
    var_70 = len(var_69)
    assert var_70 == 1
    var_71 = {}
    var_72 = 'invariant16'
    var_73 = var_71[var_17]
    var_74 = len(var_73)
    assert var_74 == 1
    var_75 = {}
    var_76 = 'invariant17'
    var_77 = var_75[var_17]
    var_78 = len(var_77)
    assert var_78 == 1
    var_79 = {}
    var_80 = 'invariant18'
    var_81 = var_79[var_17]
    var_82 = len(var_81)
    assert var_82 == 1
    var_83 = {}
    var_84 = 'invariant19'
    var_85 = var_83[var_17]
    var_86 = len(var_85)
    assert var_86 == 1
    var_87 = {}
    var_88 = 'invariant20'
    var_89 = var_87[var_17]
    var_90 = len(var_89)
    assert var_90 == 1
    var_91 = {}
    var_92 = 'invariant21'
    var_93 = var_91[var_17]
    var_94 = len(var_93)
    assert var_94 == 1



# Parsed testcases at query #45
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 15
    var_3 = -5



# Parsed testcases at query #46
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = -1
    var_3 = 15
    var_4 = -1



# Parsed testcases at query #47
#--------------------------


def test_case_0():
    var_0 = 15
    var_1 = 5
    var_2 = 12
    var_3 = 11
    var_4 = 9



# Parsed testcases at query #48
#--------------------------


def test_case_0():
    var_0 = 'test_arg'



# Parsed testcases at query #49
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 15
    var_2 = -5



# Parsed testcases at query #50
#--------------------------




# Parsed testcases at query #51
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'str'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = 1
    var_3 = 'int'
    var_4 = [var_3, var_0]
    var_5 = module_0.maybe_parse_user_type(var_4)
    var_6 = 123
    var_7 = module_0.maybe_parse_user_type(var_6)



