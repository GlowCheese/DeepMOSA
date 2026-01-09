####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
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
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = list(var_6)
    var_9 = str(var_6)
    assert var_9 == "pmap_items([(1, 'a'), (2, 'b')])"
    var_10 = repr(var_6)
    assert var_10 == "pmap_items([(1, 'a'), (2, 'b')])"
    var_11 = 3
    var_12 = 'c'
    var_13 = {var_0: var_2, var_1: var_3, var_11: var_12}
    var_14 = module_0.pmap(var_13)
    var_15 = module_0.PMapItems(var_14)
    var_16 = {var_0: var_2, var_1: var_12}
    var_17 = module_0.pmap(var_16)
    var_18 = module_0.PMapItems(var_17)
    var_19 = {var_0: var_2}
    var_20 = module_0.pmap(var_19)
    var_21 = module_0.PMapItems(var_20)
    var_22 = {}
    var_23 = module_0.pmap(var_22)
    var_24 = module_0.PMapItems(var_23)
    var_25 = {var_0: var_2, var_1: var_3, var_11: var_12}
    var_26 = module_0.pmap(var_25)
    var_27 = module_0.PMapItems(var_26)
    var_28 = {var_0: var_2, var_1: var_12}
    var_29 = module_0.pmap(var_28)
    var_30 = module_0.PMapItems(var_29)
    var_31 = {var_0: var_2}
    var_32 = module_0.pmap(var_31)
    var_33 = module_0.PMapItems(var_32)
    var_34 = {}
    var_35 = module_0.pmap(var_34)
    var_36 = module_0.PMapItems(var_35)
    var_37 = {var_0: var_2, var_1: var_3, var_11: var_12}
    var_38 = module_0.pmap(var_37)
    var_39 = module_0.PMapItems(var_38)
    var_40 = {var_0: var_2, var_1: var_12}
    var_41 = module_0.pmap(var_40)
    var_42 = module_0.PMapItems(var_41)
    var_43 = {var_0: var_2}
    var_44 = module_0.pmap(var_43)
    var_45 = module_0.PMapItems(var_44)
    var_46 = {}
    var_47 = module_0.pmap(var_46)
    var_48 = module_0.PMapItems(var_47)
    var_49 = {var_0: var_2, var_1: var_3, var_11: var_12}
    var_50 = module_0.pmap(var_49)
    var_51 = module_0.PMapItems(var_50)
    var_52 = {var_0: var_2, var_1: var_12}
    var_53 = module_0.pmap(var_52)
    var_54 = module_0.PMapItems(var_53)
    var_55 = {var_0: var_2}
    var_56 = module_0.pmap(var_55)
    var_57 = module_0.PMapItems(var_56)
    var_58 = {}
    var_59 = module_0.pmap(var_58)
    var_60 = module_0.PMapItems(var_59)
    var_61 = {var_0: var_2, var_1: var_3, var_11: var_12}
    var_62 = module_0.pmap(var_61)
    var_63 = module_0.PMapItems(var_62)
    var_64 = {var_0: var_2, var_1: var_12}
    var_65 = module_0.pmap(var_64)
    var_66 = module_0.PMapItems(var_65)
    var_67 = {var_0: var_2}
    var_68 = module_0.pmap(var_67)
    var_69 = module_0.PMapItems(var_68)
    var_70 = {}
    var_71 = module_0.pmap(var_70)
    var_72 = module_0.PMapItems(var_71)
    var_73 = {var_0: var_2, var_1: var_3, var_11: var_12}
    var_74 = module_0.pmap(var_73)
    var_75 = module_0.PMapItems(var_74)
    var_76 = {var_0: var_2, var_1: var_12}
    var_77 = module_0.pmap(var_76)
    var_78 = module_0.PMapItems(var_77)
    var_79 = {var_0: var_2}
    var_80 = module_0.pmap(var_79)
    var_81 = module_0.PMapItems(var_80)
    var_82 = {}
    var_83 = module_0.pmap(var_82)
    var_84 = module_0.PMapItems(var_83)
    var_85 = {var_0: var_2, var_1: var_3, var_11: var_12}
    var_86 = module_0.pmap(var_85)
    var_87 = module_0.PMapItems(var_86)
    var_88 = {var_0: var_2, var_1: var_12}
    var_89 = module_0.pmap(var_88)
    var_90 = module_0.PMapItems(var_89)
    var_91 = {var_0: var_2}
    var_92 = module_0.pmap(var_91)
    var_93 = module_0.PMapItems(var_92)
    var_94 = {}
    var_95 = module_0.pmap(var_94)
    var_96 = module_0.PMapItems(var_95)
    var_97 = {var_0: var_2, var_1: var_3, var_11: var_12}
    var_98 = module_0.pmap(var_97)
    var_99 = module_0.PMapItems(var_98)
    var_100 = {var_0: var_2, var_1: var_12}
    var_101 = module_0.pmap(var_100)
    var_102 = module_0.PMapItems(var_101)
    var_103 = {var_0: var_2}
    var_104 = module_0.pmap(var_103)
    var_105 = module_0.PMapItems(var_104)
    var_106 = {}
    var_107 = module_0.pmap(var_106)
    var_108 = module_0.PMapItems(var_107)
    var_109 = {var_0: var_2, var_1: var_3, var_11: var_12}
    var_110 = module_0.pmap(var_109)
    var_111 = module_0.PMapItems(var_110)
    var_112 = {var_0: var_2, var_1: var_12}
    var_113 = module_0.pmap(var_112)
    var_114 = module_0.PMapItems(var_113)
    var_115 = {var_0: var_2}
    var_116 = module_0.pmap(var_115)
    var_117 = module_0.PMapItems(var_116)
    var_118 = {}
    var_119 = module_0.pmap(var_118)
    var_120 = module_0.PMapItems(var_119)
    var_121 = {var_0: var_2, var_1: var_3, var_11: var_12}
    var_122 = module_0.pmap(var_121)
    var_123 = module_0.PMapItems(var_122)
    var_124 = {var_0: var_2, var_1: var_12}
    var_125 = module_0.pmap(var_124)
    var_126 = module_0.PMapItems(var_125)
    var_127 = {var_0: var_2}
    var_128 = module_0.pmap(var_127)
    var_129 = module_0.PMapItems(var_128)
    var_130 = {}
    var_131 = module_0.pmap(var_130)
    var_132 = module_0.PMapItems(var_131)
    var_133 = {var_0: var_2, var_1: var_3, var_11: var_12}
    var_134 = module_0.pmap(var_133)
    var_135 = module_0.PMapItems(var_134)
    var_136 = {var_0: var_2, var_1: var_12}
    var_137 = module_0.pmap(var_136)
    var_138 = module_0.PMapItems(var_137)
    var_139 = {var_0: var_2}
    var_140 = module_0.pmap(var_139)
    var_141 = module_0.PMapItems(var_140)
    var_142 = {}
    var_143 = module_0.pmap(var_142)
    var_144 = module_0.PMapItems(var_143)
    var_145 = {var_0: var_2, var_1: var_3, var_11: var_12}
    var_146 = module_0.pmap(var_145)
    var_147 = module_0.PMapItems(var_146)
    var_148 = {var_0: var_2, var_1: var_12}
    var_149 = module_0.pmap(var_148)
    var_150 = module_0.PMapItems(var_149)
    var_151 = {var_0: var_2}
    var_152 = module_0.pmap(var_151)
    var_153 = module_0.PMapItems(var_152)
    var_154 = {}
    var_155 = module_0.pmap(var_154)
    var_156 = module_0.PMapItems(var_155)
    var_157 = {var_0: var_2, var_1: var_3, var_11: var_12}
    var_158 = module_0.pmap(var_157)
    var_159 = module_0.PMapItems(var_158)
    var_160 = {var_0: var_2, var_1: var_12}
    var_161 = module_0.pmap(var_160)
    var_162 = module_0.PMapItems(var_161)
    var_163 = {var_0: var_2}
    var_164 = module_0.pmap(var_163)
    var_165 = module_0.PMapItems(var_164)
    var_166 = {}
    var_167 = module_0.pmap(var_166)
    var_168 = module_0.PMapItems(var_167)
    var_169 = {var_0: var_2, var_1: var_3, var_11: var_12}
    var_170 = module_0.pmap(var_169)
    var_171 = module_0.PMapItems(var_170)
    var_172 = {var_0: var_2, var_1: var_12}
    var_173 = module_0.pmap(var_172)
    var_174 = module_0.PMapItems(var_173)
    var_175 = {var_0: var_2}
    var_176 = module_0.pmap(var_175)
    var_177 = module_0.PMapItems(var_176)
    var_178 = {}
    var_179 = module_0.pmap(var_178)
    var_180 = module_0.PMapItems(var_179)
    var_181 = {var_0: var_2, var_1: var_3, var_11: var_12}
    var_182 = module_0.pmap(var_181)
    var_183 = module_0.PMapItems(var_182)
    var_184 = {var_0: var_2, var_1: var_12}
    var_185 = module_0.pmap(var_184)
    var_186 = module_0.PMapItems(var_185)
    var_187 = {var_0: var_2}
    var_188 = module_0.pmap(var_187)
    var_189 = module_0.PMapItems(var_188)
    var_190 = {}
    var_191 = module_0.pmap(var_190)
    var_192 = module_0.PMapItems(var_191)
    var_193 = {var_0: var_2, var_1: var_3, var_11: var_12}
    var_194 = module_0.pmap(var_193)
    var_195 = module_0.PMapItems(var_194)
    var_196 = {var_0: var_2, var_1: var_12}
    var_197 = module_0.pmap(var_196)
    var_198 = module_0.PMapItems(var_197)
    var_199 = {var_0: var_2}
    var_200 = module_0.pmap(var_199)
    var_201 = module_0.PMapItems(var_200)
    var_202 = {}
    var_203 = module_0.pmap(var_202)
    var_204 = module_0.PMapItems(var_203)
    var_205 = {var_0: var_2, var_1: var_3, var_11: var_12}
    var_206 = module_0.pmap(var_205)
    var_207 = module_0.PMapItems(var_206)
    var_208 = {var_0: var_2, var_1: var_12}
    var_209 = module_0.pmap(var_208)
    var_210 = module_0.PMapItems(var_209)
    var_211 = {var_0: var_2}
    var_212 = module_0.pmap(var_211)
    var_213 = module_0.PMapItems(var_212)
    var_214 = {}
    var_215 = module_0.pmap(var_214)
    var_216 = module_0.PMapItems(var_215)
    var_217 = {var_0: var_2, var_1: var_3, var_11: var_12}
    var_218 = module_0.pmap(var_217)
    var_219 = module_0.PMapItems(var_218)
    var_220 = {var_0: var_2, var_1: var_12}
    var_221 = module_0.pmap(var_220)
    var_222 = module_0.PMapItems(var_221)
    var_223 = {var_0: var_2}
    var_224 = module_0.pmap(var_223)
    var_225 = module_0.PMapItems(var_224)
    var_226 = {}
    var_227 = module_0.pmap(var_226)
    var_228 = module_0.PMapItems(var_227)
    var_229 = {var_0: var_2, var_1: var_3, var_11: var_12}
    var_230 = module_0.pmap(var_229)
    var_231 = module_0.PMapItems(var_230)
    var_232 = {var_0: var_2, var_1: var_12}
    var_233 = module_0.pmap(var_232)
    var_234 = module_0.PMapItems(var_233)
    var_235 = {var_0: var_2}
    var_236 = module_0.pmap(var_235)
    var_237 = module_0.PMapItems(var_236)
    var_238 = {}
    var_239 = module_0.pmap(var_238)
    var_240 = module_0.PMapItems(var_239)
    var_241 = {var_0: var_2, var_1: var_3, var_11: var_12}
    var_242 = module_0.pmap(var_241)
    var_243 = module_0.PMapItems(var_242)
    var_244 = {var_0: var_2, var_1: var_12}
    var_245 = module_0.pmap(var_244)
    var_246 = module_0.PMapItems(var_245)
    var_247 = {var_0: var_2}
    var_248 = module_0.pmap(var_247)
    var_249 = module_0.PMapItems(var_248)
    var_250 = {}
    var_251 = module_0.pmap(var_250)
    var_252 = module_0.PMapItems(var_251)
    var_253 = {var_0: var_2, var_1: var_3, var_11: var_12}
    var_254 = module_0.pmap(var_253)
    var_255 = module_0.PMapItems(var_254)
    var_256 = {var_0: var_2, var_1: var_12}
    var_257 = module_0.pmap(var_256)
    var_258 = module_0.PMapItems(var_257)
    var_259 = {var_0: var_2}
    var_260 = module_0.pmap(var_259)
    var_261 = module_0.PMapItems(var_260)
    var_262 = {}
    var_263 = module_0.pmap(var_262)
    var_264 = module_0.PMapItems(var_263)
    var_265 = {var_0: var_2, var_1: var_3, var_11: var_12}
    var_266 = module_0.pmap(var_265)
    var_267 = module_0.PMapItems(var_266)
    var_268 = {var_0: var_2, var_1: var_12}
    var_269 = module_0.pmap(var_268)
    var_270 = module_0.PMapItems(var_269)
    var_271 = {var_0: var_2}
    var_272 = module_0.pmap(var_271)
    var_273 = module_0.PMapItems(var_272)
    var_274 = {}
    var_275 = module_0.pmap(var_274)
    var_276 = module_0.PMapItems(var_275)
    var_277 = {var_0: var_2, var_1: var_3, var_11: var_12}
    var_278 = module_0.pmap(var_277)
    var_279 = module_0.PMapItems(var_278)
    var_280 = {var_0: var_2, var_1: var_12}
    var_281 = module_0.pmap(var_280)
    var_282 = module_0.PMapItems(var_281)
    var_283 = {var_0: var_2}
    var_284 = module_0.pmap(var_283)
    var_285 = module_0.PMapItems(var_284)
    var_286 = {}
    var_287 = module_0.pmap(var_286)
    var_288 = module_0.PMapItems(var_287)
    var_289 = {var_0: var_2, var_1: var_3, var_11: var_12}
    var_290 = module_0.pmap(var_289)
    var_291 = module_0.PMapItems(var_290)
    var_292 = {var_0: var_2, var_1: var_12}
    var_293 = module_0.pmap(var_292)
    var_294 = module_0.PMapItems(var_293)
    var_295 = {var_0: var_2}
    var_296 = module_0.pmap(var_295)
    var_297 = module_0.PMapItems(var_296)
    var_298 = {}
    var_299 = module_0.pmap(var_298)
    var_300 = module_0.PMapItems(var_299)



# Parsed testcases at query #2
#--------------------------



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
    var_9 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_10 = module_0.pmap(var_9)
    var_11 = module_0.PMapItems(var_10)
    var_12 = {}
    var_13 = module_0.pmap(var_12)
    var_14 = module_0.PMapItems(var_13)
    var_15 = [var_0, var_1, var_2]
    var_16 = {var_4: var_1}
    var_17 = {var_0: var_15, var_3: var_16}
    var_18 = module_0.pmap(var_17)
    var_19 = module_0.PMapItems(var_18)
    var_20 = {var_0: var_3, var_1: var_3}
    var_21 = module_0.pmap(var_20)
    var_22 = module_0.PMapItems(var_21)
    var_23 = 'All tests passed for PMapItems.__contains__'
    var_24 = print(var_23)



# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = module_0.PMapValues(var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = list(var_6)
    var_9 = str(var_6)
    assert var_9 == 'pmap_values([1, 2])'
    var_10 = repr(var_6)
    assert var_10 == 'pmap_values([1, 2])'
    var_11 = {var_0: var_2}
    var_12 = module_0.pmap(var_11)
    var_13 = module_0.PMapValues(var_12)
    var_14 = 'c'
    var_15 = 3
    var_16 = {var_14: var_15}
    var_17 = {var_14: var_16, var_15: var_3}
    var_18 = module_0.PMapValues(var_17)
    var_19 = list(var_18)
    var_20 = 1
    var_21 = 2
    var_22 = 3
    var_23 = [var_20, var_21, var_22]
    var_24 = module_0.PMapValues(var_23)



# Parsed testcases at query #4
#--------------------------



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



# Parsed testcases at query #5
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 3
    var_2 = module_0.m()
    var_3 = var_2.c



# Parsed testcases at query #6
#--------------------------



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
    var_10 = module_0.m()
    var_11 = {var_5: var_7, var_6: var_8}
    var_12 = module_0.m()
    var_13 = {var_5: var_7, var_6: var_8}
    var_14 = module_0.m()
    var_15 = {var_5: var_7, var_6: var_8}
    var_16 = module_0.m()
    var_17 = {var_5: var_7, var_6: var_8}
    var_18 = module_0.m()
    var_19 = {var_5: var_7, var_6: var_8}
    var_20 = module_0.m()
    var_21 = {var_5: var_7, var_6: var_8}
    var_22 = module_0.m()
    var_23 = {var_5: var_7, var_6: var_8}
    var_24 = module_0.m()
    var_25 = {var_5: var_7, var_6: var_8}
    var_26 = module_0.m()
    var_27 = {var_5: var_7, var_6: var_8}
    var_28 = module_0.m()
    var_29 = {var_5: var_7, var_6: var_8}
    var_30 = module_0.m()
    var_31 = {var_5: var_7, var_6: var_8}
    var_32 = module_0.m()
    var_33 = {var_5: var_7, var_6: var_8}
    var_34 = module_0.m()
    var_35 = {var_5: var_7, var_6: var_8}
    var_36 = module_0.m()
    var_37 = {var_5: var_7, var_6: var_8}
    var_38 = module_0.m()
    var_39 = {var_5: var_7, var_6: var_8}
    var_40 = module_0.m()
    var_41 = {var_5: var_7, var_6: var_8}
    var_42 = module_0.m()
    var_43 = {var_5: var_7, var_6: var_8}
    var_44 = module_0.m()
    var_45 = {var_5: var_7, var_6: var_8}
    var_46 = module_0.m()
    var_47 = {var_5: var_7, var_6: var_8}
    var_48 = module_0.m()
    var_49 = {var_5: var_7, var_6: var_8}
    var_50 = module_0.m()
    var_51 = {var_5: var_7, var_6: var_8}
    var_52 = module_0.m()
    var_53 = {var_5: var_7, var_6: var_8}
    var_54 = module_0.m()
    var_55 = {var_5: var_7, var_6: var_8}
    var_56 = module_0.m()
    var_57 = {var_5: var_7, var_6: var_8}
    var_58 = module_0.m()
    var_59 = {var_5: var_7, var_6: var_8}
    var_60 = module_0.m()
    var_61 = {var_5: var_7, var_6: var_8}
    var_62 = module_0.m()
    var_63 = {var_5: var_7, var_6: var_8}
    var_64 = module_0.m()
    var_65 = {var_5: var_7, var_6: var_8}
    var_66 = module_0.m()
    var_67 = {var_5: var_7, var_6: var_8}
    var_68 = module_0.m()
    var_69 = {var_5: var_7, var_6: var_8}
    var_70 = module_0.m()
    var_71 = {var_5: var_7, var_6: var_8}
    var_72 = module_0.m()
    var_73 = {var_5: var_7, var_6: var_8}
    var_74 = module_0.m()
    var_75 = {var_5: var_7, var_6: var_8}
    var_76 = module_0.m()
    var_77 = {var_5: var_7, var_6: var_8}
    var_78 = module_0.m()
    var_79 = {var_5: var_7, var_6: var_8}
    var_80 = module_0.m()
    var_81 = {var_5: var_7, var_6: var_8}
    var_82 = module_0.m()
    var_83 = {var_5: var_7, var_6: var_8}
    var_84 = module_0.m()
    var_85 = {var_5: var_7, var_6: var_8}
    var_86 = module_0.m()
    var_87 = {var_5: var_7, var_6: var_8}
    var_88 = module_0.m()
    var_89 = {var_5: var_7, var_6: var_8}
    var_90 = module_0.m()
    var_91 = {var_5: var_7, var_6: var_8}
    var_92 = module_0.m()
    var_93 = {var_5: var_7, var_6: var_8}
    var_94 = module_0.m()
    var_95 = {var_5: var_7, var_6: var_8}
    var_96 = module_0.m()
    var_97 = {var_5: var_7, var_6: var_8}
    var_98 = module_0.m()
    var_99 = {var_5: var_7, var_6: var_8}
    var_100 = module_0.m()
    var_101 = {var_5: var_7, var_6: var_8}
    var_102 = module_0.m()
    var_103 = {var_5: var_7, var_6: var_8}
    var_104 = module_0.m()
    var_105 = {var_5: var_7, var_6: var_8}
    var_106 = module_0.m()
    var_107 = {var_5: var_7, var_6: var_8}
    var_108 = module_0.m()
    var_109 = {var_5: var_7, var_6: var_8}
    var_110 = module_0.m()
    var_111 = {var_5: var_7, var_6: var_8}
    var_112 = module_0.m()
    var_113 = {var_5: var_7, var_6: var_8}
    var_114 = module_0.m()
    var_115 = {var_5: var_7, var_6: var_8}
    var_116 = module_0.m()
    var_117 = {var_5: var_7, var_6: var_8}
    var_118 = module_0.m()
    var_119 = {var_5: var_7, var_6: var_8}
    var_120 = module_0.m()
    var_121 = {var_5: var_7, var_6: var_8}
    var_122 = module_0.m()
    var_123 = {var_5: var_7, var_6: var_8}
    var_124 = module_0.m()
    var_125 = {var_5: var_7, var_6: var_8}
    var_126 = module_0.m()
    var_127 = {var_5: var_7, var_6: var_8}
    var_128 = module_0.m()
    var_129 = {var_5: var_7, var_6: var_8}
    var_130 = module_0.m()
    var_131 = {var_5: var_7, var_6: var_8}
    var_132 = module_0.m()
    var_133 = {var_5: var_7, var_6: var_8}
    var_134 = module_0.m()
    var_135 = {var_5: var_7, var_6: var_8}
    var_136 = module_0.m()
    var_137 = {var_5: var_7, var_6: var_8}
    var_138 = module_0.m()
    var_139 = {var_5: var_7, var_6: var_8}
    var_140 = module_0.m()
    var_141 = {var_5: var_7, var_6: var_8}
    var_142 = module_0.m()
    var_143 = {var_5: var_7, var_6: var_8}



# Parsed testcases at query #7
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 3
    var_2 = 2
    var_3 = 4
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = module_0.PMapItems(var_5)



# Parsed testcases at query #8
#--------------------------



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
    var_10 = [var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9, var_9]
    var_11 = module_0.m()
    var_12 = 'a'
    var_13 = 'b'
    var_14 = {var_12: var_0, var_13: var_1}
    var_15 = module_0.m()
    var_16 = (var_12, var_0)
    var_17 = (var_13, var_1)
    var_18 = [var_16, var_17]
    var_19 = iter(var_18)
    var_20 = module_0.m()
    var_21 = (var_12, var_0)
    var_22 = (var_13, var_4)
    var_23 = [var_21, var_22]
    var_24 = iter(var_23)
    var_25 = module_0.m()
    var_26 = (var_12, var_0)
    var_27 = (var_13, var_1)
    var_28 = 'c'
    var_29 = (var_28, var_4)
    var_30 = [var_26, var_27, var_29]
    var_31 = iter(var_30)
    var_32 = module_0.m()
    var_33 = (var_12, var_0)
    var_34 = [var_33]
    var_35 = iter(var_34)
    var_36 = module_0.m()
    var_37 = []
    var_38 = iter(var_37)
    var_39 = module_0.m()
    var_40 = (var_12, var_0)
    var_41 = (var_13, var_1)
    var_42 = (var_12, var_0)
    var_43 = [var_40, var_41, var_42]
    var_44 = iter(var_43)
    var_45 = module_0.m()
    var_46 = (var_12, var_0)
    var_47 = (var_13, var_1)
    var_48 = (var_12, var_1)
    var_49 = [var_46, var_47, var_48]
    var_50 = iter(var_49)
    var_51 = module_0.m()
    var_52 = (var_12, var_0)
    var_53 = (var_13, var_1)
    var_54 = (var_28, var_0)
    var_55 = [var_52, var_53, var_54]
    var_56 = iter(var_55)
    var_57 = module_0.m()
    var_58 = (var_12, var_0)
    var_59 = (var_13, var_1)
    var_60 = (var_28, var_1)
    var_61 = [var_58, var_59, var_60]
    var_62 = iter(var_61)
    var_63 = module_0.m()
    var_64 = (var_12, var_0)
    var_65 = (var_13, var_1)
    var_66 = (var_28, var_4)
    var_67 = [var_64, var_65, var_66]
    var_68 = iter(var_67)
    var_69 = module_0.m()
    var_70 = (var_12, var_0)
    var_71 = (var_13, var_1)
    var_72 = 4
    var_73 = (var_28, var_72)
    var_74 = [var_70, var_71, var_73]
    var_75 = iter(var_74)
    var_76 = module_0.m()
    var_77 = (var_12, var_0)
    var_78 = (var_13, var_1)
    var_79 = 5
    var_80 = (var_28, var_79)
    var_81 = [var_77, var_78, var_80]
    var_82 = iter(var_81)
    var_83 = module_0.m()
    var_84 = (var_12, var_0)
    var_85 = (var_13, var_1)
    var_86 = 6
    var_87 = (var_28, var_86)
    var_88 = [var_84, var_85, var_87]
    var_89 = iter(var_88)
    var_90 = module_0.m()
    var_91 = (var_12, var_0)
    var_92 = (var_13, var_1)
    var_93 = 7
    var_94 = (var_28, var_93)
    var_95 = [var_91, var_92, var_94]
    var_96 = iter(var_95)
    var_97 = module_0.m()
    var_98 = (var_12, var_0)
    var_99 = (var_13, var_1)
    var_100 = 8
    var_101 = (var_28, var_100)
    var_102 = [var_98, var_99, var_101]
    var_103 = iter(var_102)
    var_104 = module_0.m()
    var_105 = (var_12, var_0)
    var_106 = (var_13, var_1)
    var_107 = 9
    var_108 = (var_28, var_107)
    var_109 = [var_105, var_106, var_108]
    var_110 = iter(var_109)
    var_111 = module_0.m()
    var_112 = (var_12, var_0)
    var_113 = (var_13, var_1)
    var_114 = 10
    var_115 = (var_28, var_114)
    var_116 = [var_112, var_113, var_115]
    var_117 = iter(var_116)
    var_118 = module_0.m()
    var_119 = (var_12, var_0)
    var_120 = (var_13, var_1)
    var_121 = 11
    var_122 = (var_28, var_121)
    var_123 = [var_119, var_120, var_122]
    var_124 = iter(var_123)
    var_125 = module_0.m()
    var_126 = (var_12, var_0)
    var_127 = (var_13, var_1)
    var_128 = 12
    var_129 = (var_28, var_128)
    var_130 = [var_126, var_127, var_129]
    var_131 = iter(var_130)
    var_132 = module_0.m()
    var_133 = (var_12, var_0)
    var_134 = (var_13, var_1)
    var_135 = 13
    var_136 = (var_28, var_135)
    var_137 = [var_133, var_134, var_136]
    var_138 = iter(var_137)



# Parsed testcases at query #9
#--------------------------




# Parsed testcases at query #10
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()
    var_4 = module_0.m()
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_5: var_0, var_6: var_1}
    var_8 = 3
    var_9 = module_0.m()
    var_10 = module_0.m()
    var_11 = module_0.m()
    var_12 = module_0.m()
    var_13 = module_0.m()
    var_14 = module_0.m()
    var_15 = module_0.m()
    var_16 = module_0.m()
    var_17 = module_0.m()
    var_18 = module_0.m()
    var_19 = module_0.m()
    var_20 = module_0.m()
    var_21 = module_0.m()
    var_22 = module_0.m()
    var_23 = module_0.m()
    var_24 = module_0.m()
    var_25 = module_0.m()
    var_26 = {var_5: var_0, var_6: var_1}
    var_27 = module_0.m()
    var_28 = {var_6: var_1, var_5: var_0}
    var_29 = module_0.m()
    var_30 = 'c'
    var_31 = {var_5: var_0, var_6: var_1, var_30: var_8}
    var_32 = module_0.m()
    var_33 = {var_5: var_0, var_6: var_1}
    var_34 = module_0.m()
    var_35 = {var_6: var_1, var_5: var_0}
    var_36 = module_0.m()
    var_37 = {var_5: var_0, var_6: var_1, var_30: var_8}
    var_38 = module_0.m()
    var_39 = {var_6: var_1, var_5: var_0, var_30: var_8}
    var_40 = module_0.m()
    var_41 = {var_6: var_1, var_5: var_0, var_30: var_8}
    var_42 = module_0.m()
    var_43 = {var_5: var_0, var_6: var_1, var_30: var_8}
    var_44 = module_0.m()
    var_45 = 'd'
    var_46 = 4
    var_47 = {var_5: var_0, var_6: var_1, var_30: var_8, var_45: var_46}
    var_48 = module_0.m()
    var_49 = {var_5: var_0, var_6: var_1, var_30: var_8, var_45: var_46}
    var_50 = module_0.m()
    var_51 = {var_6: var_1, var_5: var_0, var_30: var_8, var_45: var_46}
    var_52 = module_0.m()
    var_53 = 'e'
    var_54 = 5
    var_55 = {var_6: var_1, var_5: var_0, var_30: var_8, var_45: var_46, var_53: var_54}
    var_56 = module_0.m()
    var_57 = {var_6: var_1, var_5: var_0, var_30: var_8, var_45: var_46, var_53: var_54}
    var_58 = module_0.m()
    var_59 = {var_5: var_0, var_6: var_1, var_30: var_8, var_45: var_46, var_53: var_54}



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 'a'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = 2
    var_6 = (var_1, var_2)
    var_7 = 'b'
    var_8 = (var_5, var_7)
    var_9 = [var_6, var_8]
    var_10 = (var_5, var_7)
    var_11 = (var_1, var_2)
    var_12 = [var_10, var_11]
    var_13 = (var_1, var_2)
    var_14 = (var_5, var_7)
    var_15 = [var_13, var_14]
    var_16 = (var_1, var_2)
    var_17 = (var_5, var_7)
    var_18 = [var_16, var_17]
    var_19 = (var_1, var_2)
    var_20 = (var_5, var_7)
    var_21 = [var_19, var_20]
    var_22 = (var_1, var_2)
    var_23 = (var_5, var_7)
    var_24 = [var_22, var_23]
    var_25 = (var_1, var_2)
    var_26 = (var_5, var_7)
    var_27 = [var_25, var_26]
    var_28 = (var_1, var_2)
    var_29 = 'c'
    var_30 = (var_5, var_29)
    var_31 = [var_28, var_30]



# Parsed testcases at query #12
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = module_0.PMapItems(var_5)



####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._transformations as module_1


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()
    var_4 = lambda l, r: l
    var_5 = module_0.m()
    var_6 = 'a'
    var_7 = 3
    var_8 = {var_6: var_7}
    var_9 = module_0.m()
    var_10 = module_0.m()
    var_11 = module_0.m()
    var_12 = None
    var_13 = lambda l, r: var_12
    var_14 = module_0.m()
    var_15 = 1
    var_16 = 0
    var_17 = var_15 / var_16
    var_18 = lambda l, r: var_17
    var_19 = 2
    var_20 = module_0.m()
    var_21 = 42
    var_22 = 2
    var_23 = module_0.m()
    var_24 = 42
    var_25 = 2
    var_26 = module_0.m()
    var_27 = 3
    var_28 = 'c'
    var_29 = 42
    var_30 = [var_20]
    var_31 = lambda x: x * var_26
    var_32 = module_1.transform(var_30, var_31)
    var_33 = 4
    var_34 = module_1.discard(var_20)
    var_35 = module_1.discard(var_28)
    var_36 = 5
    var_37 = {var_20: var_33, var_28: var_36}
    var_38 = {var_20: var_33, var_28: var_36}
    var_39 = {var_20: var_33, var_28: var_36}
    var_40 = 'a'
    var_41 = 'b'
    var_42 = 3
    var_43 = 2
    var_44 = {var_40: var_42, var_41: var_43}
    var_45 = 'a'
    var_46 = 'b'
    var_47 = 3
    var_48 = 2
    var_49 = {var_45: var_47, var_46: var_48}
    var_50 = 'a'
    var_51 = 'b'
    var_52 = 3
    var_53 = 2
    var_54 = {var_50: var_52, var_51: var_53}
    var_55 = 'a'
    var_56 = 'b'
    var_57 = 3
    var_58 = 2
    var_59 = {var_55: var_57, var_56: var_58}
    var_60 = 'a'
    var_61 = 'a'
    var_62 = 'c'
    var_63 = 4
    var_64 = 5
    var_65 = {var_61: var_63, var_62: var_64}
    var_66 = 'a'
    var_67 = 'c'
    var_68 = 4
    var_69 = 5
    var_70 = {var_66: var_68, var_67: var_69}
    var_71 = 'a'
    var_72 = {var_71}



# Parsed testcases at query #2
#--------------------------



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



# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = module_0.PMapItems(var_5)
    var_7 = {}
    var_8 = module_0.pmap(var_7)
    var_9 = module_0.PMapItems(var_8)
    var_10 = {var_0: var_2, var_1: var_3}
    var_11 = module_0.PMapItems(var_10)
    var_12 = 1
    var_13 = 2
    var_14 = 3
    var_15 = [var_12, var_13, var_14]
    var_16 = module_0.PMapItems(var_15)
    var_17 = {var_12: var_14, var_13: var_15}
    var_18 = module_0.pmap(var_17)
    var_19 = 'c'
    var_20 = 3
    var_21 = {var_19: var_20}
    var_22 = lambda x: x + var_14
    var_23 = module_1.transform(var_22, var_18)
    var_24 = module_0.PMapItems(var_23)
    var_25 = {var_12: var_15}
    var_26 = 'd'
    var_27 = 4
    var_28 = module_1.discard(var_12)
    var_29 = module_0.PMapItems(var_28)
    var_30 = 'c'
    var_31 = 'c'
    var_32 = 3
    var_33 = (var_31, var_32)
    var_34 = [var_33]
    var_35 = {var_19: var_20}
    var_36 = {var_19: var_20}
    var_37 = module_0.pmap(var_36)
    var_38 = {var_19: var_20}
    var_39 = (var_19, var_20)
    var_40 = [var_39]
    var_41 = (var_19, var_20)
    var_42 = (var_41,)
    var_43 = 'one'
    var_44 = {var_33: var_43}
    var_45 = []
    var_46 = 'list'
    var_47 = {var_45: var_46}
    var_48 = []
    var_49 = {var_19: var_48}
    var_50 = []
    var_51 = []
    var_52 = {var_50: var_51}
    var_53 = []
    var_54 = 1
    var_55 = {var_53: var_54}
    var_56 = []
    var_57 = {var_19: var_56}
    var_58 = {var_19: var_20}
    var_59 = []
    var_60 = []
    var_61 = {var_59: var_60}
    var_62 = []
    var_63 = 1
    var_64 = {var_62: var_63}
    var_65 = []
    var_66 = {var_19: var_65}
    var_67 = {var_19: var_20}
    var_68 = []
    var_69 = []
    var_70 = {var_68: var_69}
    var_71 = []
    var_72 = 1
    var_73 = {var_71: var_72}
    var_74 = []
    var_75 = {var_19: var_74}
    var_76 = {var_19: var_20}



# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 3
    var_2 = 2
    var_3 = 4
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_0.pmap(var_6)
    var_8 = module_0.PMapItems(var_5)
    var_9 = module_0.PMapItems(var_7)
    var_10 = {var_0: var_2, var_1: var_3}
    var_11 = module_0.pmap(var_10)
    var_12 = 5
    var_13 = {var_0: var_2, var_1: var_12}
    var_14 = module_0.pmap(var_13)
    var_15 = module_0.PMapItems(var_11)
    var_16 = module_0.PMapItems(var_14)



# Parsed testcases at query #5
#--------------------------



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
    var_8 = {}
    var_9 = module_0.pmap(var_8)
    var_10 = module_0.PMapValues(var_9)
    var_11 = repr(var_10)
    assert var_11 == 'pmap_values([])'
    var_12 = {var_0: var_2}
    var_13 = module_0.pmap(var_12)
    var_14 = module_0.PMapValues(var_13)
    var_15 = repr(var_14)
    assert var_15 == 'pmap_values([2])'
    var_16 = 100
    var_17 = range(var_16)
    var_18 = {i: i * var_2 for i in var_17}
    var_19 = module_0.pmap(var_18)
    var_20 = module_0.PMapValues(var_19)
    var_21 = repr(var_20)
    var_22 = 'a'
    var_23 = 'b'
    var_24 = {var_22: var_0, var_23: var_2}
    var_25 = module_0.pmap(var_24)
    var_26 = module_0.PMapValues(var_25)
    var_27 = repr(var_26)
    var_28 = {var_0: var_22, var_2: var_23}
    var_29 = module_0.pmap(var_28)
    var_30 = module_0.PMapValues(var_29)
    var_31 = repr(var_30)
    var_32 = {var_0: var_22, var_23: var_2}
    var_33 = module_0.pmap(var_32)
    var_34 = module_0.PMapValues(var_33)
    var_35 = repr(var_34)
    var_36 = {var_2: var_1}
    var_37 = module_0.pmap(var_36)
    var_38 = {var_0: var_37}
    var_39 = module_0.pmap(var_38)
    var_40 = module_0.PMapValues(var_39)
    var_41 = repr(var_40)
    var_42 = [var_2, var_1]
    var_43 = {var_0: var_42}
    var_44 = module_0.pmap(var_43)
    var_45 = module_0.PMapValues(var_44)
    var_46 = repr(var_45)
    var_47 = (var_2, var_1)
    var_48 = {var_0: var_47}
    var_49 = module_0.pmap(var_48)
    var_50 = module_0.PMapValues(var_49)
    var_51 = repr(var_50)
    var_52 = {var_2, var_1}
    var_53 = {var_0: var_52}
    var_54 = module_0.pmap(var_53)
    var_55 = module_0.PMapValues(var_54)
    var_56 = repr(var_55)
    var_57 = {var_2: var_1}
    var_58 = {var_0: var_57}
    var_59 = module_0.pmap(var_58)
    var_60 = module_0.PMapValues(var_59)
    var_61 = repr(var_60)
    var_62 = {var_2: var_1}
    var_63 = module_0.pmap(var_62)
    var_64 = module_0.PMapValues(var_63)
    var_65 = {var_0: var_64}
    var_66 = module_0.pmap(var_65)
    var_67 = module_0.PMapValues(var_66)
    var_68 = repr(var_67)
    var_69 = {var_2: var_1}
    var_70 = module_0.pmap(var_69)
    var_71 = module_0.PMapItems(var_70)
    var_72 = {var_0: var_71}
    var_73 = module_0.pmap(var_72)
    var_74 = module_0.PMapValues(var_73)
    var_75 = repr(var_74)
    var_76 = {var_2: var_1}
    var_77 = module_0.pmap(var_76)
    var_78 = module_0.PMapValues(var_73)
    var_79 = repr(var_78)
    var_80 = {var_2: var_1}
    var_81 = module_0.pmap(var_80)
    var_82 = {var_0: var_81}
    var_83 = module_0.pmap(var_82)
    var_84 = module_0.PMapValues(var_83)
    var_85 = repr(var_84)
    var_86 = [var_2, var_1]
    var_87 = module_0.PMapValues(var_83)
    var_88 = repr(var_87)
    var_89 = [var_2, var_1]
    var_90 = module_0.PMapValues(var_83)
    var_91 = repr(var_90)
    var_92 = [var_2, var_1]
    var_93 = module_0.PMapValues(var_83)
    var_94 = repr(var_93)
    var_95 = [var_2, var_1]
    var_96 = module_0.PMapValues(var_83)
    var_97 = repr(var_96)
    var_98 = [var_2, var_1]
    var_99 = module_0.PMapValues(var_83)
    var_100 = repr(var_99)
    var_101 = module_0.PMapValues(var_83)
    var_102 = repr(var_101)
    var_103 = module_0.PMapValues(var_83)
    var_104 = repr(var_103)
    var_105 = module_0.PMapValues(var_83)
    var_106 = repr(var_105)
    var_107 = module_0.PMapValues(var_83)
    var_108 = repr(var_107)
    var_109 = module_0.PMapValues(var_83)
    var_110 = repr(var_109)
    var_111 = module_0.PMapValues(var_83)
    var_112 = repr(var_111)
    var_113 = module_0.PMapValues(var_83)
    var_114 = repr(var_113)
    var_115 = module_0.PMapValues(var_83)
    var_116 = repr(var_115)
    var_117 = module_0.PMapValues(var_83)
    var_118 = repr(var_117)



# Parsed testcases at query #6
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = module_0.PMapItems(var_5)
    var_7 = {var_0: var_2, var_1: var_3}
    var_8 = module_0.PMapItems(var_7)
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = [var_9, var_10, var_11]
    var_13 = module_0.PMapItems(var_12)
    var_14 = {var_9: var_11, var_10: var_12}
    var_15 = module_0.pmap(var_14)
    var_16 = 'c'
    var_17 = 3
    var_18 = {var_9: var_11, var_10: var_12}
    var_19 = module_0.pmap(var_18)
    var_20 = {var_9: var_11, var_10: var_12}
    var_21 = module_0.pmap(var_20)
    var_22 = {var_9: var_11, var_10: var_12}
    var_23 = module_0.pmap(var_22)
    var_24 = 'd'
    var_25 = 4
    var_26 = {var_9: var_11, var_10: var_12}
    var_27 = module_0.pmap(var_26)
    var_28 = {var_9: var_11, var_10: var_12}
    var_29 = module_0.pmap(var_28)
    var_30 = 'e'
    var_31 = 5
    var_32 = {var_9: var_11, var_10: var_12}
    var_33 = module_0.pmap(var_32)
    var_34 = {var_9: var_11, var_10: var_12}
    var_35 = module_0.pmap(var_34)
    var_36 = 'f'
    var_37 = 6
    var_38 = {var_9: var_11, var_10: var_12}
    var_39 = module_0.pmap(var_38)
    var_40 = {var_9: var_11, var_10: var_12}
    var_41 = module_0.pmap(var_40)



# Parsed testcases at query #7
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 3
    var_2 = 2
    var_3 = 4
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = {var_0: var_2, var_1: var_3}
    var_8 = {var_0: var_2, var_1: var_3}
    var_9 = {var_0: var_2, var_1: var_3}
    var_10 = {var_0: var_2, var_1: var_3}
    var_11 = {var_0: var_2, var_1: var_3}
    var_12 = {var_0: var_2, var_1: var_3}
    var_13 = {var_0: var_2, var_1: var_3}
    var_14 = {var_0: var_2, var_1: var_3}
    var_15 = {var_0: var_2, var_1: var_3}
    var_16 = {var_0: var_2, var_1: var_3}
    var_17 = {var_0: var_2, var_1: var_3}
    var_18 = {var_0: var_2, var_1: var_3}
    var_19 = {var_0: var_2, var_1: var_3}
    var_20 = {var_0: var_2, var_1: var_3}
    var_21 = {var_0: var_2, var_1: var_3}
    var_22 = {var_0: var_2, var_1: var_3}
    var_23 = {var_0: var_2, var_1: var_3}
    var_24 = {var_0: var_2, var_1: var_3}
    var_25 = {var_0: var_2, var_1: var_3}
    var_26 = {var_0: var_2, var_1: var_3}
    var_27 = {var_0: var_2, var_1: var_3}
    var_28 = {var_0: var_2, var_1: var_3}
    var_29 = {var_0: var_2, var_1: var_3}
    var_30 = {var_0: var_2, var_1: var_3}
    var_31 = {var_0: var_2, var_1: var_3}
    var_32 = {var_0: var_2, var_1: var_3}
    var_33 = {var_0: var_2, var_1: var_3}
    var_34 = {var_0: var_2, var_1: var_3}
    var_35 = {var_0: var_2, var_1: var_3}
    var_36 = {var_0: var_2, var_1: var_3}
    var_37 = {var_0: var_2, var_1: var_3}
    var_38 = {var_0: var_2, var_1: var_3}
    var_39 = {var_0: var_2, var_1: var_3}
    var_40 = {var_0: var_2, var_1: var_3}
    var_41 = {var_0: var_2, var_1: var_3}
    var_42 = {var_0: var_2, var_1: var_3}
    var_43 = {var_0: var_2, var_1: var_3}
    var_44 = {var_0: var_2, var_1: var_3}
    var_45 = {var_0: var_2, var_1: var_3}
    var_46 = {var_0: var_2, var_1: var_3}
    var_47 = {var_0: var_2, var_1: var_3}
    var_48 = {var_0: var_2, var_1: var_3}
    var_49 = {var_0: var_2, var_1: var_3}
    var_50 = {var_0: var_2, var_1: var_3}
    var_51 = {var_0: var_2, var_1: var_3}
    var_52 = {var_0: var_2, var_1: var_3}
    var_53 = {var_0: var_2, var_1: var_3}
    var_54 = {var_0: var_2, var_1: var_3}
    var_55 = {var_0: var_2, var_1: var_3}
    var_56 = {var_0: var_2, var_1: var_3}
    var_57 = {var_0: var_2, var_1: var_3}
    var_58 = {var_0: var_2, var_1: var_3}
    var_59 = {var_0: var_2, var_1: var_3}
    var_60 = {var_0: var_2, var_1: var_3}
    var_61 = {var_0: var_2, var_1: var_3}
    var_62 = {var_0: var_2, var_1: var_3}
    var_63 = {var_0: var_2, var_1: var_3}
    var_64 = {var_0: var_2, var_1: var_3}
    var_65 = {var_0: var_2, var_1: var_3}
    var_66 = {var_0: var_2, var_1: var_3}
    var_67 = {var_0: var_2, var_1: var_3}
    var_68 = {var_0: var_2, var_1: var_3}
    var_69 = {var_0: var_2, var_1: var_3}
    var_70 = {var_0: var_2, var_1: var_3}
    var_71 = {var_0: var_2, var_1: var_3}
    var_72 = {var_0: var_2, var_1: var_3}
    var_73 = {var_0: var_2, var_1: var_3}
    var_74 = {var_0: var_2, var_1: var_3}
    var_75 = {var_0: var_2, var_1: var_3}
    var_76 = {var_0: var_2, var_1: var_3}
    var_77 = {var_0: var_2, var_1: var_3}
    var_78 = {var_0: var_2, var_1: var_3}
    var_79 = {var_0: var_2, var_1: var_3}
    var_80 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #8
#--------------------------



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



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_0.pmap(var_6)
    var_8 = module_0.PMapItems(var_5)
    var_9 = module_0.PMapItems(var_7)
    var_10 = {var_0: var_2, var_1: var_3}
    var_11 = module_0.pmap(var_10)
    var_12 = 3
    var_13 = {var_0: var_2, var_1: var_12}
    var_14 = module_0.pmap(var_13)
    var_15 = module_0.PMapItems(var_11)
    var_16 = module_0.PMapItems(var_14)
    var_17 = module_0.PMapValues(var_5)
    var_18 = {var_0: var_2, var_1: var_3}
    var_19 = module_0.PMapItems(var_18)



# Parsed testcases at query #10
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = module_0.PMapItems(var_5)



# Parsed testcases at query #11
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()
    var_4 = module_0.m()
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_5: var_0, var_6: var_1}
    var_8 = {var_6: var_1, var_5: var_0}



# Parsed testcases at query #12
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 3
    var_2 = 2
    var_3 = 4
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_0.pmap(var_6)
    var_8 = module_0.PMapItems(var_5)
    var_9 = module_0.PMapItems(var_7)
    var_10 = 5
    var_11 = {var_0: var_2, var_1: var_10}
    var_12 = module_0.pmap(var_11)
    var_13 = module_0.PMapItems(var_5)
    var_14 = module_0.PMapItems(var_12)
    var_15 = var_13 == var_14
    var_16 = module_0.PMapItems(var_5)
    var_17 = module_0.PMapItems(var_5)
    var_18 = module_0.PMapItems(var_5)
    var_19 = var_18 == var_5
    var_20 = module_0.PMapItems(var_5)
    var_21 = module_0.PMapValues(var_5)
    var_22 = var_20 == var_21



