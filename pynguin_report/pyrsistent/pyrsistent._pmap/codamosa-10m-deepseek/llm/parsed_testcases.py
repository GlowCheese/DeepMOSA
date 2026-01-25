####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._pmap as module_0
import pyrsistent._transformations as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = module_0.PMapValues(var_7)
    var_9 = list(var_8)
    var_10 = {}
    var_11 = module_0.pmap(var_10)
    var_12 = module_0.PMapValues(var_11)
    var_13 = list(var_12)
    var_14 = {var_0: var_3, var_1: var_3, var_2: var_4}
    var_15 = module_0.pmap(var_14)
    var_16 = module_0.PMapValues(var_15)
    var_17 = list(var_16)
    var_18 = {var_0: var_3, var_1: var_4}
    var_19 = module_0.pmap(var_18)
    var_20 = module_0.PMapValues(var_19)
    var_21 = iter(var_20)
    var_22 = next(var_21)
    var_23 = list(var_21)
    var_24 = 1000
    var_25 = range(var_24)
    var_26 = {i: i * var_4 for i in var_25}
    var_27 = module_0.pmap(var_26)
    var_28 = module_0.PMapValues(var_27)
    var_29 = list(var_28)
    var_30 = range(var_24)
    var_31 = [i * var_4 for i in var_30]
    var_32 = 'x'
    var_33 = 'y'
    var_34 = 10
    var_35 = 20
    var_36 = {var_32: var_34, var_33: var_35}
    var_37 = module_0.pmap(var_36)
    var_38 = module_0.PMapValues(var_37)
    var_39 = []
    var_40 = {var_0: var_3}
    var_41 = module_0.pmap(var_40)
    var_42 = module_0.PMapValues(var_41)
    var_43 = iter(var_42)
    var_44 = next(var_43)
    assert var_44 == 1
    var_45 = next(var_43)
    var_46 = {var_45: var_3, var_1: var_4}
    var_47 = module_0.pmap(var_46)
    var_48 = module_0.PMapValues(var_47)
    var_49 = iter(var_48)
    var_50 = iter(var_48)
    var_51 = next(var_49)
    assert var_51 == 1
    var_52 = next(var_50)
    assert var_52 == 1
    var_53 = next(var_49)
    assert var_53 == 2
    var_54 = next(var_50)
    assert var_54 == 2
    var_55 = {var_45: var_3, var_1: var_4}
    var_56 = module_0.pmap(var_55)
    var_57 = lambda x: x * var_4
    var_58 = module_1.transform(var_57, var_56)
    var_59 = module_0.PMapValues(var_58)
    var_60 = list(var_59)
    var_61 = {var_45: var_3, var_1: var_4}
    var_62 = module_0.pmap(var_61)
    var_63 = module_0.PMapValues(var_62)
    var_64 = list(var_63)
    var_65 = (var_45, var_3)
    var_66 = (var_1, var_4)
    var_67 = [var_65, var_66]
    var_68 = module_0.pmap(var_67)
    var_69 = module_0.PMapValues(var_68)
    var_70 = list(var_69)
    var_71 = {var_45: var_3}
    var_72 = module_0.pmap(var_71)
    var_73 = module_0.PMapValues(var_68)
    var_74 = list(var_73)
    var_75 = {var_45: var_3, var_1: var_4, var_2: var_5}
    var_76 = module_0.pmap(var_75)
    var_77 = module_0.PMapValues(var_68)
    var_78 = list(var_77)
    var_79 = {var_45: var_3, var_1: var_4}
    var_80 = module_0.pmap(var_79)
    var_81 = module_0.PMapValues(var_68)
    var_82 = list(var_81)
    var_83 = {var_45: var_3}
    var_84 = module_0.pmap(var_83)
    var_85 = {var_1: var_4}
    var_86 = module_0.pmap(var_85)
    var_87 = module_0.PMapValues(var_68)
    var_88 = list(var_87)
    var_89 = {var_45: var_3}
    var_90 = module_0.pmap(var_89)
    var_91 = {var_1: var_4}
    var_92 = module_0.PMapValues(var_68)
    var_93 = list(var_92)
    var_94 = {var_45: var_3}
    var_95 = module_0.pmap(var_94)
    var_96 = (var_1, var_4)
    var_97 = [var_96]
    var_98 = module_0.PMapValues(var_68)
    var_99 = list(var_98)
    var_100 = {var_45: var_3}
    var_101 = module_0.pmap(var_100)
    var_102 = module_0.PMapValues(var_68)
    var_103 = list(var_102)
    var_104 = {var_45: var_3}
    var_105 = module_0.pmap(var_104)
    var_106 = {var_1: var_4}
    var_107 = module_0.PMapValues(var_68)
    var_108 = list(var_107)
    var_109 = {var_45: var_3}
    var_110 = module_0.pmap(var_109)
    var_111 = {var_1: var_4}
    var_112 = lambda x: var_111
    var_113 = module_0.PMapValues(var_68)
    var_114 = list(var_113)
    var_115 = {var_45: var_3}
    var_116 = module_0.pmap(var_115)
    var_117 = (var_1, var_4)
    var_118 = (var_117,)
    var_119 = module_0.PMapValues(var_68)
    var_120 = list(var_119)
    var_121 = {var_45: var_3}
    var_122 = module_0.pmap(var_121)
    var_123 = {var_1: var_4}
    var_124 = module_0.pmap(var_123)
    var_125 = module_0.PMapValues(var_122)
    var_126 = list(var_125)
    var_127 = {var_45: var_3}
    var_128 = module_0.pmap(var_127)
    var_129 = []
    var_130 = module_0.PMapValues(var_122)
    var_131 = list(var_130)
    var_132 = {var_45: var_3}
    var_133 = module_0.pmap(var_132)
    var_134 = {}
    var_135 = module_0.PMapValues(var_122)
    var_136 = list(var_135)
    var_137 = {var_45: var_3}
    var_138 = module_0.pmap(var_137)
    var_139 = module_0.PMapValues(var_122)
    var_140 = list(var_139)
    var_141 = {var_45: var_3}
    var_142 = module_0.pmap(var_141)
    var_143 = {}
    var_144 = lambda x: var_143
    var_145 = module_0.PMapValues(var_122)
    var_146 = list(var_145)
    var_147 = {var_45: var_3}
    var_148 = module_0.pmap(var_147)
    var_149 = ()
    var_150 = module_0.PMapValues(var_122)
    var_151 = list(var_150)
    var_152 = {var_45: var_3}
    var_153 = module_0.pmap(var_152)
    var_154 = {}
    var_155 = module_0.pmap(var_154)
    var_156 = module_0.PMapValues(var_153)
    var_157 = list(var_156)



# Parsed testcases at query #2
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 3
    var_2 = 2
    var_3 = 4
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = module_0.PMapValues(var_5)
    var_7 = str(var_6)
    assert var_7 == 'pmap_values([2, 4])'



# Parsed testcases at query #3
#--------------------------


import pyrsistent._pmap as module_0
import pyrsistent._transformations as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 'a'
    var_4 = module_1.discard(var_3)
    var_5 = 'c'
    var_6 = module_1.discard(var_5)
    var_7 = 'test_PMap_discard passed'
    var_8 = print(var_7)



# Parsed testcases at query #4
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
    var_8 = {var_0: var_2, var_1: var_3}
    var_9 = module_0.PMapView(var_8)
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = [var_11, var_12, var_13]
    var_15 = module_0.PMapView(var_14)
    var_16 = 'c'
    var_17 = 3
    var_18 = {var_16: var_17}
    var_19 = reversed(var_15)



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
    var_6 = module_0.PMapView(var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = reversed(var_6)



# Parsed testcases at query #6
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()
    var_4 = 3
    var_5 = module_0.m()
    var_6 = module_0.m()
    var_7 = module_0.m()
    var_8 = module_0.m()
    var_9 = hash(var_7)
    var_10 = hash(var_8)
    var_11 = var_9 == var_10
    var_12 = 'All tests passed for PMap.__eq__'
    var_13 = print(var_12)



# Parsed testcases at query #7
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = var_2.c
    var_4 = '1'
    var_5 = {var_4: var_1}
    var_6 = module_0.m(**var_5)
    var_7 = var_6.d
    var_8 = 3
    var_9 = module_0.m()



# Parsed testcases at query #8
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = module_0.PMapItems(var_5)
    var_7 = module_0.PMapItems(var_5)
    var_8 = {var_0: var_2, var_1: var_3}
    var_9 = module_0.pmap(var_8)
    var_10 = 'c'
    var_11 = {var_0: var_2, var_1: var_10}
    var_12 = module_0.pmap(var_11)
    var_13 = module_0.PMapItems(var_9)
    var_14 = module_0.PMapItems(var_12)
    var_15 = {var_0: var_2, var_1: var_3}
    var_16 = module_0.pmap(var_15)
    var_17 = module_0.PMapItems(var_16)
    var_18 = {var_0: var_2, var_1: var_3}
    var_19 = module_0.pmap(var_18)
    var_20 = module_0.PMapItems(var_19)



# Parsed testcases at query #9
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.m()
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 2
    var_5 = module_0.m()
    var_6 = 'c'
    var_7 = 3
    var_8 = {var_1: var_4, var_6: var_7}
    var_9 = module_0.m()
    var_10 = {var_1: var_4}
    var_11 = {var_6: var_7}
    var_12 = module_0.m()
    var_13 = {var_1: var_2}
    var_14 = 'b'
    var_15 = {var_14: var_4}
    var_16 = module_0.m()
    var_17 = {}
    var_18 = module_0.m()
    var_19 = {}
    var_20 = {}
    var_21 = module_0.m()
    var_22 = {}
    var_23 = {}
    var_24 = {}
    var_25 = module_0.m()
    var_26 = {}
    var_27 = {}
    var_28 = {}
    var_29 = {}
    var_30 = module_0.m()
    var_31 = {}
    var_32 = {}
    var_33 = {}
    var_34 = {}
    var_35 = {}
    var_36 = module_0.m()
    var_37 = {}
    var_38 = {}
    var_39 = {}
    var_40 = {}
    var_41 = {}
    var_42 = {}
    var_43 = module_0.m()
    var_44 = {}
    var_45 = {}
    var_46 = {}
    var_47 = {}
    var_48 = {}
    var_49 = {}
    var_50 = {}
    var_51 = module_0.m()
    var_52 = {}
    var_53 = {}
    var_54 = {}
    var_55 = {}
    var_56 = {}
    var_57 = {}
    var_58 = {}
    var_59 = {}
    var_60 = module_0.m()
    var_61 = {}
    var_62 = {}
    var_63 = {}
    var_64 = {}
    var_65 = {}
    var_66 = {}
    var_67 = {}
    var_68 = {}
    var_69 = {}
    var_70 = module_0.m()
    var_71 = {}
    var_72 = {}
    var_73 = {}
    var_74 = {}
    var_75 = {}
    var_76 = {}
    var_77 = {}
    var_78 = {}
    var_79 = {}
    var_80 = {}
    var_81 = module_0.m()
    var_82 = {}
    var_83 = {}
    var_84 = {}
    var_85 = {}
    var_86 = {}
    var_87 = {}
    var_88 = {}
    var_89 = {}
    var_90 = {}
    var_91 = {}
    var_92 = {}
    var_93 = module_0.m()
    var_94 = {}
    var_95 = {}
    var_96 = {}
    var_97 = {}
    var_98 = {}
    var_99 = {}
    var_100 = {}
    var_101 = {}
    var_102 = {}
    var_103 = {}
    var_104 = {}
    var_105 = {}
    var_106 = module_0.m()
    var_107 = {}
    var_108 = {}
    var_109 = {}
    var_110 = {}
    var_111 = {}
    var_112 = {}
    var_113 = {}
    var_114 = {}
    var_115 = {}
    var_116 = {}
    var_117 = {}
    var_118 = {}
    var_119 = {}
    var_120 = module_0.m()
    var_121 = {}
    var_122 = {}
    var_123 = {}
    var_124 = {}
    var_125 = {}
    var_126 = {}
    var_127 = {}
    var_128 = {}
    var_129 = {}
    var_130 = {}
    var_131 = {}
    var_132 = {}
    var_133 = {}
    var_134 = {}
    var_135 = module_0.m()
    var_136 = {}
    var_137 = {}
    var_138 = {}
    var_139 = {}
    var_140 = {}
    var_141 = {}
    var_142 = {}
    var_143 = {}
    var_144 = {}
    var_145 = {}
    var_146 = {}
    var_147 = {}
    var_148 = {}
    var_149 = {}
    var_150 = {}
    var_151 = module_0.m()
    var_152 = {}
    var_153 = {}
    var_154 = {}
    var_155 = {}
    var_156 = {}
    var_157 = {}
    var_158 = {}
    var_159 = {}
    var_160 = {}
    var_161 = {}
    var_162 = {}
    var_163 = {}
    var_164 = {}
    var_165 = {}
    var_166 = {}
    var_167 = {}
    var_168 = module_0.m()
    var_169 = {}
    var_170 = {}
    var_171 = {}
    var_172 = {}
    var_173 = {}
    var_174 = {}
    var_175 = {}
    var_176 = {}
    var_177 = {}
    var_178 = {}
    var_179 = {}
    var_180 = {}
    var_181 = {}
    var_182 = {}
    var_183 = {}
    var_184 = {}
    var_185 = {}
    var_186 = module_0.m()
    var_187 = {}
    var_188 = {}
    var_189 = {}
    var_190 = {}
    var_191 = {}
    var_192 = {}
    var_193 = {}
    var_194 = {}
    var_195 = {}
    var_196 = {}
    var_197 = {}
    var_198 = {}
    var_199 = {}
    var_200 = {}
    var_201 = {}
    var_202 = {}
    var_203 = {}
    var_204 = {}
    var_205 = module_0.m()
    var_206 = {}
    var_207 = {}
    var_208 = {}
    var_209 = {}
    var_210 = {}
    var_211 = {}
    var_212 = {}
    var_213 = {}
    var_214 = {}
    var_215 = {}
    var_216 = {}
    var_217 = {}
    var_218 = {}
    var_219 = {}
    var_220 = {}
    var_221 = {}
    var_222 = {}
    var_223 = {}
    var_224 = {}
    var_225 = module_0.m()
    var_226 = {}
    var_227 = {}
    var_228 = {}
    var_229 = {}
    var_230 = {}
    var_231 = {}
    var_232 = {}
    var_233 = {}
    var_234 = {}
    var_235 = {}
    var_236 = {}
    var_237 = {}
    var_238 = {}
    var_239 = {}
    var_240 = {}
    var_241 = {}
    var_242 = {}
    var_243 = {}
    var_244 = {}
    var_245 = {}
    var_246 = module_0.m()
    var_247 = {}
    var_248 = {}
    var_249 = {}
    var_250 = {}
    var_251 = {}
    var_252 = {}
    var_253 = {}
    var_254 = {}
    var_255 = {}
    var_256 = {}
    var_257 = {}
    var_258 = {}
    var_259 = {}
    var_260 = {}
    var_261 = {}
    var_262 = {}
    var_263 = {}
    var_264 = {}
    var_265 = {}
    var_266 = {}
    var_267 = {}
    var_268 = module_0.m()
    var_269 = {}
    var_270 = {}
    var_271 = {}
    var_272 = {}
    var_273 = {}
    var_274 = {}
    var_275 = {}
    var_276 = {}
    var_277 = {}
    var_278 = {}
    var_279 = {}
    var_280 = {}
    var_281 = {}
    var_282 = {}
    var_283 = {}
    var_284 = {}
    var_285 = {}
    var_286 = {}
    var_287 = {}
    var_288 = {}
    var_289 = {}
    var_290 = {}
    var_291 = module_0.m()
    var_292 = {}
    var_293 = {}
    var_294 = {}
    var_295 = {}
    var_296 = {}
    var_297 = {}
    var_298 = {}
    var_299 = {}
    var_300 = {}
    var_301 = {}
    var_302 = {}
    var_303 = {}
    var_304 = {}
    var_305 = {}
    var_306 = {}
    var_307 = {}
    var_308 = {}
    var_309 = {}
    var_310 = {}
    var_311 = {}
    var_312 = {}
    var_313 = {}
    var_314 = {}



# Parsed testcases at query #10
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 3
    var_2 = 2
    var_3 = 4
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = module_0.PMapValues(var_5)
    var_7 = repr(var_6)
    assert var_7 == 'pmap_values([2, 4])'



# Parsed testcases at query #11
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = module_0.PMapItems(var_7)



# Parsed testcases at query #12
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()
    var_4 = 3
    var_5 = module_0.m()
    var_6 = module_0.m()
    var_7 = module_0.m()
    var_8 = 10
    var_9 = 20
    var_10 = module_0.m()
    var_11 = 100
    var_12 = 200
    var_13 = module_0.m()
    var_14 = 'a'
    var_15 = 'b'
    var_16 = {var_14: var_0, var_15: var_1}
    var_17 = {var_14: var_0}
    var_18 = {var_14: var_0, var_15: var_4}
    var_19 = range(var_11)
    var_20 = {str(i): i for i in var_19}
    var_21 = module_0.m(**var_20)
    var_22 = module_0.m(**var_20)
    var_23 = {var_14: var_0, var_15: var_1}
    var_24 = {var_15: var_1, var_14: var_0}
    var_25 = 'All tests passed for PMap.__eq__'
    var_26 = print(var_25)



# Parsed testcases at query #13
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = module_0.PMapItems(var_5)



# Parsed testcases at query #14
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = var_2.c
    var_4 = '123'
    var_5 = 456
    var_6 = {var_4: var_5}
    var_7 = module_0.m(**var_6)
    var_8 = 'class'
    var_9 = 'test'
    var_10 = {var_8: var_9}
    var_11 = module_0.m(**var_10)
    var_12 = 'def'
    var_13 = 'function'
    var_14 = {var_12: var_13}
    var_15 = module_0.m(**var_14)
    var_16 = 'len'
    var_17 = 5
    var_18 = {var_16: var_17}
    var_19 = module_0.m(**var_18)
    var_20 = 'int'
    var_21 = 10
    var_22 = {var_20: var_21}
    var_23 = module_0.m(**var_22)
    var_24 = 'True'
    var_25 = 'yes'
    var_26 = {var_24: var_25}
    var_27 = module_0.m(**var_26)
    var_28 = 'Exception'
    var_29 = 'error'
    var_30 = {var_28: var_29}
    var_31 = module_0.m(**var_30)
    var_32 = 'sys'
    var_33 = 'module'
    var_34 = {var_32: var_33}
    var_35 = module_0.m(**var_34)
    var_36 = 'print'
    var_37 = {var_36: var_13}
    var_38 = module_0.m(**var_37)
    var_39 = 'list'
    var_40 = 3
    var_41 = [var_3, var_1, var_40]
    var_42 = {var_39: var_41}
    var_43 = module_0.m(**var_42)
    var_44 = 'None'
    var_45 = 'nothing'
    var_46 = {var_44: var_45}
    var_47 = module_0.m(**var_46)
    var_48 = 'KeyError'
    var_49 = 'missing'
    var_50 = {var_48: var_49}
    var_51 = module_0.m(**var_50)
    var_52 = 'os'
    var_53 = 'operating system'
    var_54 = {var_52: var_53}
    var_55 = module_0.m(**var_54)
    var_56 = 'sorted'
    var_57 = {var_56: var_13}
    var_58 = module_0.m(**var_57)
    var_59 = 'dict'
    var_60 = 'mapping'
    var_61 = {var_59: var_60}
    var_62 = module_0.m(**var_61)
    var_63 = 'False'
    var_64 = 'no'
    var_65 = {var_63: var_64}
    var_66 = module_0.m(**var_65)
    var_67 = 'ValueError'
    var_68 = 'invalid'
    var_69 = {var_67: var_68}
    var_70 = module_0.m(**var_69)
    var_71 = 'json'
    var_72 = 'JavaScript Object Notation'
    var_73 = {var_71: var_72}
    var_74 = module_0.m(**var_73)
    var_75 = 'range'
    var_76 = 'sequence'
    var_77 = {var_75: var_76}
    var_78 = module_0.m(**var_77)
    var_79 = 'set'
    var_80 = 'collection'
    var_81 = {var_79: var_80}
    var_82 = module_0.m(**var_81)
    var_83 = 'Ellipsis'
    var_84 = '...'
    var_85 = {var_83: var_84}
    var_86 = module_0.m(**var_85)
    var_87 = 'TypeError'
    var_88 = 'wrong type'
    var_89 = {var_87: var_88}
    var_90 = module_0.m(**var_89)
    var_91 = 'math'
    var_92 = 'mathematics'
    var_93 = {var_91: var_92}
    var_94 = module_0.m(**var_93)
    var_95 = 'max'
    var_96 = 'maximum'
    var_97 = {var_95: var_96}
    var_98 = module_0.m(**var_97)
    var_99 = 'tuple'
    var_100 = 'immutable sequence'
    var_101 = {var_99: var_100}
    var_102 = module_0.m(**var_101)
    var_103 = 'NotImplemented'
    var_104 = 'not implemented'
    var_105 = {var_103: var_104}
    var_106 = module_0.m(**var_105)
    var_107 = 'AttributeError'
    var_108 = 'no attribute'
    var_109 = {var_107: var_108}
    var_110 = module_0.m(**var_109)
    var_111 = 'random'
    var_112 = 'random number generation'
    var_113 = {var_111: var_112}
    var_114 = module_0.m(**var_113)
    var_115 = 'min'
    var_116 = 'minimum'
    var_117 = {var_115: var_116}
    var_118 = module_0.m(**var_117)
    var_119 = 'frozenset'
    var_120 = 'immutable set'
    var_121 = {var_119: var_120}
    var_122 = module_0.m(**var_121)
    var_123 = '__debug__'
    var_124 = 'debug mode'
    var_125 = {var_123: var_124}
    var_126 = module_0.m(**var_125)
    var_127 = 'ImportError'
    var_128 = 'import error'
    var_129 = {var_127: var_128}
    var_130 = module_0.m(**var_129)
    var_131 = 'datetime'
    var_132 = 'date and time'
    var_133 = {var_131: var_132}
    var_134 = module_0.m(**var_133)



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #16
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 3
    var_2 = 2
    var_3 = 4
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = module_0.PMapItems(var_5)



####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()
    var_4 = module_0.m()
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_5: var_0, var_6: var_1}
    var_8 = module_0.m()
    var_9 = 3
    var_10 = {var_5: var_0, var_6: var_9}
    var_11 = module_0.m()
    var_12 = module_0.m()
    var_13 = module_0.m()
    var_14 = module_0.m()
    var_15 = module_0.m()
    var_16 = module_0.m()
    var_17 = module_0.m()
    var_18 = module_0.m()
    var_19 = '2'
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
    var_30 = module_0.m()
    var_31 = module_0.m()
    var_32 = module_0.m()
    var_33 = module_0.m()
    var_34 = module_0.m()
    var_35 = module_0.m()
    var_36 = module_0.m()
    var_37 = module_0.m()
    var_38 = module_0.m()
    var_39 = module_0.m()
    var_40 = module_0.m()
    var_41 = module_0.m()
    var_42 = module_0.m()
    var_43 = module_0.m()
    var_44 = module_0.m()
    var_45 = module_0.m()
    var_46 = module_0.m()
    var_47 = module_0.m()
    var_48 = module_0.m()
    var_49 = module_0.m()
    var_50 = module_0.m()
    var_51 = module_0.m()
    var_52 = module_0.m()
    var_53 = module_0.m()
    var_54 = module_0.m()
    var_55 = module_0.m()
    var_56 = module_0.m()
    var_57 = module_0.m()
    var_58 = module_0.m()
    var_59 = module_0.m()
    var_60 = module_0.m()
    var_61 = module_0.m()
    var_62 = module_0.m()
    var_63 = module_0.m()
    var_64 = module_0.m()
    var_65 = module_0.m()
    var_66 = module_0.m()
    var_67 = module_0.m()
    var_68 = module_0.m()
    var_69 = module_0.m()
    var_70 = module_0.m()
    var_71 = module_0.m()
    var_72 = module_0.m()



# Parsed testcases at query #2
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()
    var_4 = 3
    var_5 = module_0.m()
    var_6 = module_0.m()
    var_7 = 100
    var_8 = range(var_7)
    var_9 = {i: i for i in var_8}
    var_10 = module_0.m()
    var_11 = module_0.m()
    var_12 = hash(var_10)
    var_13 = module_0.m()
    var_14 = module_0.m()
    var_15 = hash(var_13)
    var_16 = hash(var_14)
    var_17 = 0
    var_18 = module_0.m()
    var_19 = 'a'
    var_20 = 'b'
    var_21 = {var_19: var_0, var_20: var_1}
    var_22 = {var_19: var_0}
    var_23 = {var_19: var_0, var_20: var_4}
    var_24 = 'All tests passed for PMap.__eq__'
    var_25 = print(var_24)



# Parsed testcases at query #3
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_0.pmap(var_6)
    var_8 = module_0.PMapItems(var_5)
    var_9 = module_0.PMapItems(var_7)
    var_10 = 'c'
    var_11 = {var_0: var_2, var_1: var_10}
    var_12 = module_0.pmap(var_11)
    var_13 = module_0.PMapItems(var_12)



# Parsed testcases at query #4
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = module_0.PMapItems(var_5)



# Parsed testcases at query #5
#--------------------------


import pyrsistent._pmap as module_0
import builtins as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = module_0.PMapValues(var_3)
    var_5 = module_0.PMapValues(var_3)
    var_6 = {var_0: var_1}
    var_7 = module_0.PMapItems(var_3)
    var_8 = var_4 == var_7
    var_9 = {var_1}
    var_10 = frozenset(var_9)
    var_11 = var_4 == var_10
    var_12 = range(var_0)
    var_13 = var_4 == var_12
    var_14 = lambda x: x
    var_15 = [var_1]
    var_16 = map(var_14, var_15)
    var_17 = var_4 == var_16
    var_18 = [var_0]
    var_19 = [var_1]
    var_20 = zip(var_18, var_19)
    var_21 = var_4 == var_20
    var_22 = lambda x: x
    var_23 = [var_1]
    var_24 = filter(var_22, var_23)
    var_25 = var_4 == var_24
    var_26 = [var_1]
    var_27 = enumerate(var_26)
    var_28 = var_4 == var_27
    var_29 = [var_1]
    var_30 = reversed(var_29)
    var_31 = var_4 == var_30
    var_32 = slice(var_0)
    var_33 = var_4 == var_32
    var_34 = b'2'
    var_35 = memoryview(var_34)
    var_36 = var_4 == var_35
    var_37 = bytearray(var_34)
    var_38 = var_4 == var_37
    var_39 = module_1.object()
    var_40 = var_4 == var_39



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = lambda l, r: r
    var_1 = lambda l, r: r

def test_case_0():
    var_0 = lambda l, r: r
    var_1 = lambda l, r: r
    var_2 = lambda l, r: r
    var_3 = lambda l, r: r
    var_4 = lambda l, r: r
    var_5 = lambda l, r: r
    var_6 = lambda l, r: r

def test_case_0():
    var_0 = lambda l, r: r
    var_1 = lambda l, r: r
    var_2 = lambda l, r: r
    var_3 = lambda l, r: r
    var_4 = lambda l, r: r
    var_5 = lambda l, r: r
    var_6 = lambda l, r: r



# Parsed testcases at query #7
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 3
    var_2 = 2
    var_3 = 4
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 5
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = module_0.pmap(var_7)



# Parsed testcases at query #8
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()
    var_4 = 3
    var_5 = module_0.m()
    var_6 = None
    var_7 = 'a'
    var_8 = (var_7, var_0)
    var_9 = 'b'
    var_10 = (var_9, var_1)
    var_11 = [var_8, var_10]
    var_12 = [var_6, var_11]
    var_13 = (var_7, var_0)
    var_14 = (var_9, var_1)
    var_15 = [var_13, var_14]
    var_16 = [var_6, var_15]
    var_17 = (var_7, var_0)
    var_18 = (var_9, var_4)
    var_19 = [var_17, var_18]
    var_20 = [var_6, var_19]
    var_21 = module_0.m()
    var_22 = module_0.m()
    var_23 = module_0.m()
    var_24 = module_0.m()
    var_25 = (var_7, var_0)
    var_26 = (var_9, var_1)
    var_27 = [var_25, var_26]
    var_28 = (var_7, var_0)
    var_29 = (var_9, var_1)
    var_30 = [var_28, var_29]
    var_31 = {var_7: var_0, var_9: var_1}
    var_32 = module_0.m()
    var_33 = (var_9, var_1)
    var_34 = (var_7, var_0)
    var_35 = [var_33, var_34]
    var_36 = [var_35, var_6]
    var_37 = (var_9, var_1)
    var_38 = (var_7, var_0)
    var_39 = [var_37, var_38]
    var_40 = [var_39, var_6, var_6]
    var_41 = (var_9, var_1)
    var_42 = (var_7, var_0)
    var_43 = [var_41, var_42]
    var_44 = [var_43, var_6, var_6]
    var_45 = (var_9, var_1)
    var_46 = (var_7, var_0)
    var_47 = [var_45, var_46]
    var_48 = [var_47, var_6, var_6]
    var_49 = (var_9, var_1)
    var_50 = (var_7, var_0)
    var_51 = [var_49, var_50]
    var_52 = [var_51, var_6, var_6]
    var_53 = (var_9, var_1)
    var_54 = (var_7, var_0)
    var_55 = [var_53, var_54]
    var_56 = [var_55, var_6, var_6]
    var_57 = (var_9, var_1)
    var_58 = (var_7, var_0)
    var_59 = [var_57, var_58]
    var_60 = [var_59, var_6, var_6]
    var_61 = (var_9, var_1)
    var_62 = (var_7, var_0)
    var_63 = [var_61, var_62]
    var_64 = [var_63, var_6, var_6]
    var_65 = (var_9, var_1)
    var_66 = (var_7, var_0)
    var_67 = [var_65, var_66]
    var_68 = [var_67, var_6, var_6]
    var_69 = (var_9, var_1)
    var_70 = (var_7, var_0)
    var_71 = [var_69, var_70]
    var_72 = [var_71, var_6, var_6]
    var_73 = (var_9, var_1)
    var_74 = (var_7, var_0)
    var_75 = [var_73, var_74]
    var_76 = [var_75, var_6, var_6]
    var_77 = (var_9, var_1)
    var_78 = (var_7, var_0)
    var_79 = [var_77, var_78]
    var_80 = [var_79, var_6, var_6]



# Parsed testcases at query #9
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()
    var_4 = 3
    var_5 = module_0.m()
    var_6 = module_0.m()
    var_7 = module_0.m()
    var_8 = module_0.m()
    var_9 = None
    var_10 = 'a'
    var_11 = (var_10, var_0)
    var_12 = 'b'
    var_13 = (var_12, var_1)
    var_14 = [var_11, var_13]
    var_15 = [var_9, var_14]
    var_16 = (var_10, var_0)
    var_17 = (var_12, var_1)
    var_18 = [var_16, var_17]
    var_19 = [var_18, var_9]
    var_20 = (var_10, var_0)
    var_21 = [var_20]
    var_22 = (var_12, var_1)
    var_23 = [var_22]
    var_24 = [var_21, var_23]
    var_25 = (var_12, var_1)
    var_26 = [var_25]
    var_27 = (var_10, var_0)
    var_28 = [var_27]
    var_29 = [var_26, var_28]
    var_30 = (var_10, var_0)
    var_31 = [var_30]
    var_32 = (var_12, var_1)
    var_33 = [var_32]
    var_34 = [var_31, var_9, var_33]
    var_35 = (var_10, var_0)
    var_36 = [var_35]
    var_37 = (var_12, var_1)
    var_38 = [var_37]
    var_39 = [var_9, var_36, var_9, var_38]
    var_40 = (var_10, var_0)
    var_41 = (var_12, var_1)
    var_42 = [var_40, var_41]
    var_43 = [var_9, var_9, var_42]
    var_44 = (var_10, var_0)
    var_45 = (var_12, var_1)
    var_46 = [var_44, var_45]
    var_47 = [var_9, var_9, var_9, var_46]
    var_48 = (var_10, var_0)
    var_49 = (var_12, var_1)
    var_50 = [var_48, var_49]
    var_51 = [var_9, var_9, var_9, var_9, var_50]
    var_52 = (var_10, var_0)
    var_53 = (var_12, var_1)
    var_54 = [var_52, var_53]
    var_55 = [var_9, var_9, var_9, var_9, var_9, var_54]
    var_56 = (var_10, var_0)
    var_57 = (var_12, var_1)
    var_58 = [var_56, var_57]
    var_59 = [var_9, var_9, var_9, var_9, var_9, var_9, var_58]
    var_60 = (var_10, var_0)
    var_61 = (var_12, var_1)
    var_62 = [var_60, var_61]
    var_63 = [var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_62]
    var_64 = (var_10, var_0)
    var_65 = (var_12, var_1)
    var_66 = [var_64, var_65]
    var_67 = [var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_66]
    var_68 = (var_10, var_0)
    var_69 = (var_12, var_1)
    var_70 = [var_68, var_69]
    var_71 = [var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_70]
    var_72 = (var_10, var_0)
    var_73 = (var_12, var_1)
    var_74 = [var_72, var_73]
    var_75 = [var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_74]
    var_76 = (var_10, var_0)
    var_77 = (var_12, var_1)
    var_78 = [var_76, var_77]
    var_79 = [var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_78]
    var_80 = (var_10, var_0)
    var_81 = (var_12, var_1)
    var_82 = [var_80, var_81]
    var_83 = [var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_82]
    var_84 = (var_10, var_0)
    var_85 = (var_12, var_1)
    var_86 = [var_84, var_85]
    var_87 = [var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_86]
    var_88 = (var_10, var_0)
    var_89 = (var_12, var_1)
    var_90 = [var_88, var_89]
    var_91 = [var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_90]
    var_92 = (var_10, var_0)
    var_93 = (var_12, var_1)
    var_94 = [var_92, var_93]
    var_95 = [var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_94]
    var_96 = (var_10, var_0)
    var_97 = (var_12, var_1)
    var_98 = [var_96, var_97]
    var_99 = [var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_98]
    var_100 = (var_10, var_0)
    var_101 = (var_12, var_1)
    var_102 = [var_100, var_101]
    var_103 = [var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_102]
    var_104 = (var_10, var_0)
    var_105 = (var_12, var_1)
    var_106 = [var_104, var_105]
    var_107 = [var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_106]



# Parsed testcases at query #10
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 3
    var_2 = 2
    var_3 = 4
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_0.pmap(var_6)
    var_8 = 5
    var_9 = 6
    var_10 = {var_0: var_2, var_1: var_3, var_8: var_9}
    var_11 = module_0.pmap(var_10)
    var_12 = {var_0: var_2}
    var_13 = module_0.pmap(var_12)
    var_14 = {}
    var_15 = module_0.pmap(var_14)
    var_16 = {var_0: var_1}
    var_17 = module_0.pmap(var_16)
    var_18 = {var_0: var_2, var_1: var_8}
    var_19 = module_0.pmap(var_18)
    var_20 = {var_0: var_2, var_3: var_3}
    var_21 = module_0.pmap(var_20)
    var_22 = {var_2: var_2, var_1: var_3}
    var_23 = module_0.pmap(var_22)
    var_24 = {var_0: var_2, var_1: var_3, var_8: var_9}
    var_25 = module_0.pmap(var_24)
    var_26 = 7
    var_27 = 8
    var_28 = {var_0: var_2, var_1: var_3, var_8: var_9, var_26: var_27}
    var_29 = module_0.pmap(var_28)
    var_30 = 9
    var_31 = 10
    var_32 = {var_0: var_2, var_1: var_3, var_8: var_9, var_26: var_27, var_30: var_31}
    var_33 = module_0.pmap(var_32)
    var_34 = 11
    var_35 = 12
    var_36 = {var_0: var_2, var_1: var_3, var_8: var_9, var_26: var_27, var_30: var_31, var_34: var_35}
    var_37 = module_0.pmap(var_36)
    var_38 = 13
    var_39 = 14
    var_40 = {var_0: var_2, var_1: var_3, var_8: var_9, var_26: var_27, var_30: var_31, var_34: var_35, var_38: var_39}
    var_41 = module_0.pmap(var_40)
    var_42 = 15
    var_43 = 16
    var_44 = {var_0: var_2, var_1: var_3, var_8: var_9, var_26: var_27, var_30: var_31, var_34: var_35, var_38: var_39, var_42: var_43}
    var_45 = module_0.pmap(var_44)
    var_46 = 17
    var_47 = 18
    var_48 = {var_0: var_2, var_1: var_3, var_8: var_9, var_26: var_27, var_30: var_31, var_34: var_35, var_38: var_39, var_42: var_43, var_46: var_47}
    var_49 = module_0.pmap(var_48)
    var_50 = 19
    var_51 = 20
    var_52 = {var_0: var_2, var_1: var_3, var_8: var_9, var_26: var_27, var_30: var_31, var_34: var_35, var_38: var_39, var_42: var_43, var_46: var_47, var_50: var_51}
    var_53 = module_0.pmap(var_52)
    var_54 = 21
    var_55 = 22
    var_56 = {var_0: var_2, var_1: var_3, var_8: var_9, var_26: var_27, var_30: var_31, var_34: var_35, var_38: var_39, var_42: var_43, var_46: var_47, var_50: var_51, var_54: var_55}
    var_57 = module_0.pmap(var_56)
    var_58 = 23
    var_59 = 24
    var_60 = {var_0: var_2, var_1: var_3, var_8: var_9, var_26: var_27, var_30: var_31, var_34: var_35, var_38: var_39, var_42: var_43, var_46: var_47, var_50: var_51, var_54: var_55, var_58: var_59}
    var_61 = module_0.pmap(var_60)
    var_62 = 25
    var_63 = 26
    var_64 = {var_0: var_2, var_1: var_3, var_8: var_9, var_26: var_27, var_30: var_31, var_34: var_35, var_38: var_39, var_42: var_43, var_46: var_47, var_50: var_51, var_54: var_55, var_58: var_59, var_62: var_63}
    var_65 = module_0.pmap(var_64)
    var_66 = 27
    var_67 = 28
    var_68 = {var_0: var_2, var_1: var_3, var_8: var_9, var_26: var_27, var_30: var_31, var_34: var_35, var_38: var_39, var_42: var_43, var_46: var_47, var_50: var_51, var_54: var_55, var_58: var_59, var_62: var_63, var_66: var_67}
    var_69 = module_0.pmap(var_68)
    var_70 = 29
    var_71 = 30
    var_72 = {var_0: var_2, var_1: var_3, var_8: var_9, var_26: var_27, var_30: var_31, var_34: var_35, var_38: var_39, var_42: var_43, var_46: var_47, var_50: var_51, var_54: var_55, var_58: var_59, var_62: var_63, var_66: var_67, var_70: var_71}
    var_73 = module_0.pmap(var_72)
    var_74 = 31
    var_75 = 32
    var_76 = {var_0: var_2, var_1: var_3, var_8: var_9, var_26: var_27, var_30: var_31, var_34: var_35, var_38: var_39, var_42: var_43, var_46: var_47, var_50: var_51, var_54: var_55, var_58: var_59, var_62: var_63, var_66: var_67, var_70: var_71, var_74: var_75}
    var_77 = module_0.pmap(var_76)
    var_78 = 33
    var_79 = 34
    var_80 = {var_0: var_2, var_1: var_3, var_8: var_9, var_26: var_27, var_30: var_31, var_34: var_35, var_38: var_39, var_42: var_43, var_46: var_47, var_50: var_51, var_54: var_55, var_58: var_59, var_62: var_63, var_66: var_67, var_70: var_71, var_74: var_75, var_78: var_79}
    var_81 = module_0.pmap(var_80)
    var_82 = 35
    var_83 = 36
    var_84 = {var_0: var_2, var_1: var_3, var_8: var_9, var_26: var_27, var_30: var_31, var_34: var_35, var_38: var_39, var_42: var_43, var_46: var_47, var_50: var_51, var_54: var_55, var_58: var_59, var_62: var_63, var_66: var_67, var_70: var_71, var_74: var_75, var_78: var_79, var_82: var_83}
    var_85 = module_0.pmap(var_84)
    var_86 = 37
    var_87 = 38
    var_88 = {var_0: var_2, var_1: var_3, var_8: var_9, var_26: var_27, var_30: var_31, var_34: var_35, var_38: var_39, var_42: var_43, var_46: var_47, var_50: var_51, var_54: var_55, var_58: var_59, var_62: var_63, var_66: var_67, var_70: var_71, var_74: var_75, var_78: var_79, var_82: var_83, var_86: var_87}
    var_89 = module_0.pmap(var_88)
    var_90 = 39
    var_91 = 40
    var_92 = {var_0: var_2, var_1: var_3, var_8: var_9, var_26: var_27, var_30: var_31, var_34: var_35, var_38: var_39, var_42: var_43, var_46: var_47, var_50: var_51, var_54: var_55, var_58: var_59, var_62: var_63, var_66: var_67, var_70: var_71, var_74: var_75, var_78: var_79, var_82: var_83, var_86: var_87, var_90: var_91}
    var_93 = module_0.pmap(var_92)
    var_94 = 41
    var_95 = 42
    var_96 = {var_0: var_2, var_1: var_3, var_8: var_9, var_26: var_27, var_30: var_31, var_34: var_35, var_38: var_39, var_42: var_43, var_46: var_47, var_50: var_51, var_54: var_55, var_58: var_59, var_62: var_63, var_66: var_67, var_70: var_71, var_74: var_75, var_78: var_79, var_82: var_83, var_86: var_87, var_90: var_91, var_94: var_95}
    var_97 = module_0.pmap(var_96)
    var_98 = 43
    var_99 = 44
    var_100 = {var_0: var_2, var_1: var_3, var_8: var_9, var_26: var_27, var_30: var_31, var_34: var_35, var_38: var_39, var_42: var_43, var_46: var_47, var_50: var_51, var_54: var_55, var_58: var_59, var_62: var_63, var_66: var_67, var_70: var_71, var_74: var_75, var_78: var_79, var_82: var_83, var_86: var_87, var_90: var_91, var_94: var_95, var_98: var_99}
    var_101 = module_0.pmap(var_100)



