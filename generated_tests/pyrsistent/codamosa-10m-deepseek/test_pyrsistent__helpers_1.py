# Check out: https://github.com/GlowCheese/deepmosa
import collections as module_2

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_3
import pyrsistent._pset as module_1
import pyrsistent._pvector as module_4
import pytest


def test_case_0():
    var_0 = None
    var_1 = module_0.freeze(var_0)

def test_case_1():
    var_0 = None
    var_1 = module_0.freeze(var_0, var_0)

def test_case_2():
    var_0 = 'S5i9Bt'
    var_1 = module_0.thaw(var_0, var_0)
    assert var_1 == 'S5i9Bt'

def test_case_3():
    var_0 = None
    var_1 = module_0.thaw(var_0, var_0)

def test_case_4():
    var_0 = None
    var_1 = module_0.mutant(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = module_1.pset()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.thaw(var_0)
    var_2 = module_0.mutant(var_0)
    var_3 = None
    var_4 = module_0.freeze(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_4) == 0
    var_5 = module_0.mutant(var_3)
    var_1.upper()

def test_case_6():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_4) == 3
    var_4.rstrip()

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1, var_0]
    var_3 = module_0.freeze(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_3) == 3
    var_4 = [var_0, var_1, var_0]
    var_5 = 'b'
    var_6 = {var_5: var_1}
    var_7 = module_0.freeze(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 1
    var_8 = module_0.freeze(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 1
    var_9 = module_2.defaultdict()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'collections.defaultdict'
    assert len(var_9) == 0
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_10 = module_0.freeze(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 0
    var_9.extend(var_4)

def test_case_9():
    var_0 = 1
    var_1 = module_3.m()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_3.KT).__module__}.{type(module_3.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_3.VT_co).__module__}.{type(module_3.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = module_4.v()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_2) == 0
    assert f'{type(module_4.T_co).__module__}.{type(module_4.T_co).__qualname__}' == 'typing.TypeVar'
    assert module_4.BRANCH_FACTOR == 32
    assert module_4.BIT_MASK == 31
    assert module_4.SHIFT == 5
    var_3 = module_0.freeze(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_3) == 0
    var_4 = (var_0, var_2)
    var_5 = module_0.thaw(var_4)
    var_6 = 'All tests passed!'
    var_7 = print(var_6)

def test_case_10():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_4) == 3
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_5: var_0, var_6: var_1}
    var_8 = module_0.freeze(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 2
    var_9 = {var_5: var_0, var_6: var_1}
    var_10 = module_3.pmap(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 2
    assert f'{type(module_3.KT).__module__}.{type(module_3.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_3.VT_co).__module__}.{type(module_3.VT_co).__qualname__}' == 'typing.TypeVar'
    var_11 = {var_0, var_1, var_2}
    var_12 = module_0.freeze(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_12) == 3
    var_13 = {var_0, var_1, var_2}
    var_14 = module_1.pset(var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_14) == 3
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    var_15 = [var_1, var_2]
    var_16 = (var_0, var_15)
    var_17 = module_0.freeze(var_16)
    var_18 = [var_0, var_1, var_2]
    var_19 = False
    var_20 = module_0.freeze(var_18, var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_20) == 3
    var_21 = module_0.freeze(var_4, var_19)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_21) == 3
    var_22 = {var_5: var_0, var_6: var_1}
    var_23 = module_3.pmap(var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_23) == 2
    var_24 = {var_5: var_0, var_6: var_1}
    var_25 = module_3.pmap(var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_25) == 2
    var_26 = module_0.freeze(var_25)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_26) == 2
    var_27 = {var_5: var_0, var_6: var_1}
    var_28 = module_3.pmap(var_27)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_28) == 2
    var_29 = 'All tests passed!'
    var_30 = print(var_29)

def test_case_11():
    var_0 = 2
    var_1 = 'b'
    var_2 = {var_1: var_0}
    var_3 = module_0.freeze(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    var_4 = module_2.defaultdict()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'collections.defaultdict'
    assert len(var_4) == 0
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_5 = module_0.freeze(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 0

def test_case_12():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1, var_0]
    var_3 = module_0.freeze(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_3) == 3
    var_4 = 'a'
    var_5 = 'b'
    var_6 = {var_5: var_1}
    var_7 = module_0.mutant(var_3)
    var_8 = module_0.freeze(var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 1
    var_9 = {var_4: var_0, var_5: var_1}
    var_10 = module_3.pmap(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 2
    assert f'{type(module_3.KT).__module__}.{type(module_3.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_3.VT_co).__module__}.{type(module_3.VT_co).__qualname__}' == 'typing.TypeVar'
    var_11 = module_2.defaultdict()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'collections.defaultdict'
    assert len(var_11) == 0
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_12 = module_0.freeze(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_12) == 0
    var_13 = module_0.thaw(var_2, var_8)
    var_14 = None
    var_15 = module_0.freeze(var_14)
    var_16 = var_12.__str__()
    assert var_16 == 'pmap({})'

def test_case_13():
    var_0 = 1
    var_1 = module_3.m()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_3.KT).__module__}.{type(module_3.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_3.VT_co).__module__}.{type(module_3.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.thaw(var_1)
    var_3 = 'All tests passed!'
    var_4 = print(var_3)

def test_case_14():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_3.m()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    assert f'{type(module_3.KT).__module__}.{type(module_3.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_3.VT_co).__module__}.{type(module_3.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = module_4.v()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_4) == 0
    assert f'{type(module_4.T_co).__module__}.{type(module_4.T_co).__qualname__}' == 'typing.TypeVar'
    assert module_4.BRANCH_FACTOR == 32
    assert module_4.BIT_MASK == 31
    assert module_4.SHIFT == 5
    var_5 = (var_0, var_4)
    var_6 = module_0.thaw(var_5)
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.thaw(var_7)
    var_9 = 4
    var_10 = {var_0: var_1, var_2: var_9}
    var_11 = module_0.thaw(var_10)
    var_12 = (var_0, var_1, var_2)
    var_13 = module_0.thaw(var_12)
    var_14 = [var_0, var_1, var_2]
    var_15 = set(var_14)
    var_16 = module_0.thaw(var_15)
    var_17 = module_0.thaw(var_0)
    assert var_17 == 1
    var_18 = 'a'
    var_19 = module_0.thaw(var_18)
    assert var_19 == 'a'
    var_20 = None
    var_21 = module_0.thaw(var_20)
    assert var_21 is None
    var_22 = True
    var_23 = module_0.thaw(var_22)
    assert var_23 is True
    var_24 = False
    var_25 = module_0.thaw(var_24)
    assert var_25 is False
    var_26 = module_0.thaw(var_22)
    assert var_26 is True
    var_27 = b'abc'
    var_28 = module_0.thaw(var_27)
    assert var_28 == b'abc'
    var_29 = bytearray(var_27)
    var_30 = module_0.thaw(var_29)
    var_31 = bytearray(var_27)
    var_32 = memoryview(var_27)
    var_33 = module_0.thaw(var_32)
    var_34 = memoryview(var_27)
    var_35 = slice(var_22, var_1, var_2)
    var_36 = module_0.thaw(var_35)
    var_37 = slice(var_22, var_1, var_2)
    var_38 = range(var_22, var_1, var_2)
    var_39 = module_0.thaw(var_38)
    var_40 = range(var_22, var_1, var_2)
    var_41 = Exception()
    var_42 = module_0.thaw(var_41)
    var_43 = Exception()
    var_44 = lambda x: x
    var_45 = module_0.thaw(var_44)
    var_46 = [var_22, var_1, var_2]
    var_47 = module_0.thaw(var_46, var_24)
    var_48 = {var_22: var_1, var_2: var_9}
    var_49 = module_0.thaw(var_48, var_24)
    var_50 = (var_22, var_1, var_2)
    var_51 = module_0.thaw(var_50, var_24)
    var_52 = [var_22, var_1, var_2]
    var_53 = set(var_52)
    var_54 = module_0.thaw(var_53, var_24)
    var_55 = module_0.thaw(var_22, var_24)
    assert var_55 == 1
    var_56 = module_0.thaw(var_18, var_24)
    assert var_56 == 'a'
    var_57 = module_0.thaw(var_20, var_24)
    assert var_57 is None
    var_58 = True
    var_59 = module_0.thaw(var_58, var_24)
    assert var_59 is True
    var_60 = module_0.thaw(var_24, var_24)
    assert var_60 is False
    var_61 = module_0.thaw(var_58, var_24)
    assert var_61 is True
    var_62 = module_0.thaw(var_27, var_24)
    assert var_62 == b'abc'
    var_63 = bytearray(var_27)
    var_64 = module_0.thaw(var_63, var_24)
    var_65 = bytearray(var_27)
    var_66 = memoryview(var_27)
    var_67 = module_0.thaw(var_66, var_24)
    var_68 = memoryview(var_27)
    var_69 = slice(var_58, var_1, var_2)
    var_70 = module_0.thaw(var_69, var_24)
    var_71 = slice(var_58, var_1, var_2)
    var_72 = range(var_58, var_1, var_2)
    var_73 = module_0.thaw(var_72, var_24)
    var_74 = range(var_58, var_1, var_2)
    var_75 = Exception()
    var_76 = module_0.thaw(var_75, var_24)
    var_77 = Exception()
    var_78 = lambda x: x
    var_79 = module_0.thaw(var_78, var_24)
    var_80 = [var_58, var_1, var_2]
    var_81 = True
    var_82 = module_0.thaw(var_80, var_81)
    var_83 = {var_81: var_1, var_2: var_9}
    var_84 = True
    var_85 = module_0.thaw(var_83, var_84)
    var_86 = (var_84, var_1, var_2)
    var_87 = True
    var_88 = module_0.thaw(var_86, var_87)
    var_89 = [var_87, var_1, var_2]
    var_90 = set(var_89)
    var_91 = True
    var_92 = module_0.thaw(var_90, var_91)
    var_93 = True
    var_94 = module_0.thaw(var_91, var_93)
    assert var_94 == 1
    var_95 = True
    var_96 = module_0.thaw(var_18, var_95)
    assert var_96 == 'a'
    var_97 = True
    var_98 = module_0.thaw(var_20, var_97)
    assert var_98 is None
    var_99 = True
    var_100 = True
    var_101 = module_0.thaw(var_99, var_100)
    assert var_101 is True
    var_102 = True
    var_103 = module_0.thaw(var_24, var_102)
    assert var_103 is False
    var_104 = True
    var_105 = module_0.thaw(var_102, var_104)
    assert var_105 is True
    var_106 = True
    var_107 = module_0.thaw(var_27, var_106)
    assert var_107 == b'abc'
    var_108 = bytearray(var_27)
    var_109 = True
    var_110 = module_0.thaw(var_108, var_109)
    var_111 = bytearray(var_27)
    var_112 = memoryview(var_27)
    var_113 = True
    var_114 = module_0.thaw(var_112, var_113)
    var_115 = memoryview(var_27)
    var_116 = slice(var_113, var_1, var_2)
    var_117 = True
    var_118 = module_0.thaw(var_116, var_117)
    var_119 = slice(var_117, var_1, var_2)
    var_120 = range(var_117, var_1, var_2)
    var_121 = True
    var_122 = module_0.thaw(var_120, var_121)
    var_123 = range(var_121, var_1, var_2)
    var_124 = Exception()
    var_125 = True
    var_126 = module_0.thaw(var_124, var_125)
    var_127 = Exception()
    var_128 = lambda x: x
    var_129 = True
    var_130 = module_0.thaw(var_128, var_129)
    var_131 = [var_129, var_1, var_2]
    var_132 = module_0.thaw(var_131, var_24)
    var_133 = {var_129: var_1, var_2: var_9}
    var_134 = module_0.thaw(var_133, var_24)
    var_135 = (var_129, var_1, var_2)
    var_136 = module_0.thaw(var_135, var_24)
    var_137 = [var_129, var_1, var_2]
    var_138 = set(var_137)
    var_139 = module_0.thaw(var_138, var_24)
    var_140 = module_0.thaw(var_129, var_24)
    assert var_140 == 1
    var_141 = module_0.thaw(var_18, var_24)
    assert var_141 == 'a'
    var_142 = module_0.thaw(var_20, var_24)
    assert var_142 is None
    var_143 = True
    var_144 = module_0.thaw(var_143, var_24)
    assert var_144 is True
    var_145 = module_0.thaw(var_24, var_24)
    assert var_145 is False
    var_146 = module_0.thaw(var_143, var_24)
    assert var_146 is True
    var_147 = module_0.thaw(var_27, var_24)
    assert var_147 == b'abc'
    var_148 = bytearray(var_27)
    var_149 = module_0.thaw(var_148, var_24)
    var_150 = bytearray(var_27)
    var_151 = memoryview(var_27)
    var_152 = module_0.thaw(var_151, var_24)
    var_153 = memoryview(var_27)
    var_154 = slice(var_143, var_1, var_2)
    var_155 = slice(var_143, var_1, var_2)
    var_156 = range(var_143, var_1, var_2)
    var_157 = module_0.thaw(var_156, var_24)
    var_158 = range(var_143, var_1, var_2)
    var_159 = Exception()
    var_160 = module_0.thaw(var_159, var_24)
    var_161 = Exception()
    var_162 = lambda x: x
    var_163 = module_0.thaw(var_162, var_24)
    var_164 = [var_143, var_1, var_2]
    var_165 = True
    var_166 = module_0.thaw(var_164, var_165)
    var_167 = {var_165: var_1, var_2: var_9}
    var_168 = True
    var_169 = module_0.thaw(var_167, var_168)
    var_170 = True
    var_171 = [var_170, var_1, var_2]
    var_172 = set(var_171)
    var_173 = True
    var_174 = module_0.thaw(var_172, var_173)
    var_175 = True
    var_176 = module_0.thaw(var_173, var_175)
    assert var_176 == 1
    var_177 = True
    var_178 = module_0.thaw(var_18, var_177)
    assert var_178 == 'a'
    var_179 = True
    var_180 = module_0.thaw(var_20, var_179)
    assert var_180 is None
    var_181 = True
    var_182 = True
    var_183 = module_0.thaw(var_181, var_182)
    assert var_183 is True
    var_184 = True
    var_185 = module_0.thaw(var_24, var_184)
    assert var_185 is False
    var_186 = True
    var_187 = module_0.thaw(var_184, var_186)
    assert var_187 is True
    var_188 = True
    var_189 = module_0.thaw(var_27, var_188)
    assert var_189 == b'abc'
    var_190 = bytearray(var_27)
    var_191 = True
    var_192 = module_0.thaw(var_190, var_191)
    var_193 = bytearray(var_27)
    var_194 = memoryview(var_27)
    var_195 = True
    var_196 = module_0.thaw(var_194, var_195)
    var_197 = memoryview(var_27)
    var_198 = slice(var_195, var_1, var_2)
    var_199 = True
    var_200 = module_0.thaw(var_198, var_199)
    var_201 = slice(var_199, var_1, var_2)
    var_202 = range(var_199, var_1, var_2)
    var_203 = True
    var_204 = module_0.thaw(var_202, var_203)
    var_205 = range(var_203, var_1, var_2)
    var_206 = Exception()
    var_207 = True
    var_208 = module_0.thaw(var_206, var_207)
    var_209 = Exception()
    var_210 = lambda x: x
    var_211 = True
    var_212 = module_0.thaw(var_210, var_211)
    var_213 = [var_211, var_1, var_2]
    var_214 = module_0.thaw(var_213, var_24)
    var_215 = {var_211: var_1, var_2: var_9}
    var_216 = module_0.thaw(var_215, var_24)
    var_217 = (var_211, var_1, var_2)
    var_218 = module_0.thaw(var_217, var_24)
    var_219 = [var_211, var_1, var_2]
    var_220 = set(var_219)
    var_221 = module_0.thaw(var_220, var_24)
    var_222 = module_0.thaw(var_211, var_24)
    assert var_222 == 1
    var_223 = module_0.thaw(var_18, var_24)
    assert var_223 == 'a'
    var_224 = module_0.thaw(var_20, var_24)
    assert var_224 is None
    var_225 = True
    var_226 = module_0.thaw(var_225, var_24)
    assert var_226 is True
    var_227 = module_0.thaw(var_24, var_24)
    assert var_227 is False

def test_case_15():
    var_0 = 1
    var_1 = 10
    var_2 = module_4.v()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_2) == 0
    assert f'{type(module_4.T_co).__module__}.{type(module_4.T_co).__qualname__}' == 'typing.TypeVar'
    assert module_4.BRANCH_FACTOR == 32
    assert module_4.BIT_MASK == 31
    assert module_4.SHIFT == 5
    var_3 = (var_0, var_2)
    var_4 = module_0.thaw(var_3)
    var_5 = print(var_1)

def test_case_16():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = {var_1: var_1}
    var_4 = module_0.freeze(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = {var_2: var_0, var_1: var_1}
    var_6 = module_0.thaw(var_4, var_4)
    var_7 = module_3.pmap(var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 2
    assert f'{type(module_3.KT).__module__}.{type(module_3.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_3.VT_co).__module__}.{type(module_3.VT_co).__qualname__}' == 'typing.TypeVar'
    var_8 = module_2.defaultdict()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'collections.defaultdict'
    assert len(var_8) == 0
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_9 = module_0.freeze(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 0

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1, var_0]
    var_3 = module_0.freeze(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_3) == 3
    var_4 = [var_0, var_1, var_0]
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_6: var_1}
    var_8 = module_0.freeze(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 1
    var_9 = {var_5: var_0, var_6: var_1}
    var_10 = module_0.thaw(var_3, var_6)
    var_11 = module_3.pmap(var_9)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_11) == 2
    assert f'{type(module_3.KT).__module__}.{type(module_3.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_3.VT_co).__module__}.{type(module_3.VT_co).__qualname__}' == 'typing.TypeVar'
    var_12 = module_2.defaultdict()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'collections.defaultdict'
    assert len(var_12) == 0
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_13 = module_0.freeze(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_13) == 0
    var_12.extend(var_4)

def test_case_18():
    var_0 = module_2.defaultdict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'collections.defaultdict'
    assert len(var_0) == 0
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0