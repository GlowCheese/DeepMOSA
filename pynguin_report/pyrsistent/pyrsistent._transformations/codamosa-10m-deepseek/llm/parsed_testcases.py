####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 'b'
    var_4 = 'd'
    var_5 = 5
    var_6 = 'All tests passed for discard function.'
    var_7 = print(var_6)



# Parsed testcases at query #2
#--------------------------


import pyrsistent._transformations as module_0


def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123
    var_5 = ''
    var_6 = '^\\d+$'
    var_7 = module_0.rex(var_6)
    var_8 = '123'
    var_9 = 'abc'
    var_10 = 'All test cases passed!'
    var_11 = print(var_10)



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 'b'
    var_4 = 'd'
    var_5 = 5
    var_6 = 'All tests passed!'
    var_7 = print(var_6)



# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = 'b'
    var_6 = {var_4: var_0, var_5: var_1}
    var_7 = [var_0, var_1, var_2]
    var_8 = 'c'
    var_9 = 4
    var_10 = {var_8: var_9}
    var_11 = {var_4: var_7, var_5: var_10}
    var_12 = {var_4: var_0, var_5: var_1}
    var_13 = 'a1'
    var_14 = 'a2'
    var_15 = 'b1'
    var_16 = {var_13: var_0, var_14: var_1, var_15: var_2}
    var_17 = 'a.*'
    var_18 = module_0.rex(var_17)
    var_19 = {var_4: var_0, var_5: var_1, var_8: var_2}
    var_20 = [var_4, var_5]
    var_21 = lambda k: k in var_20
    var_22 = {var_4: var_0, var_5: var_1, var_8: var_2}
    var_23 = 0
    var_24 = lambda k, v: v % var_1 == var_23
    var_25 = {}
    var_26 = {}
    var_27 = {var_4: var_26}
    var_28 = 'All test cases passed!'
    var_29 = print(var_28)



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 'b'
    var_4 = 'd'
    var_5 = 4
    var_6 = 5
    var_7 = 10
    var_8 = 'All discard tests passed!'
    var_9 = print(var_8)



# Parsed testcases at query #6
#--------------------------



def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123
    var_5 = ''
    var_6 = '^\\d+$'
    var_7 = module_0.rex(var_6)
    var_8 = '123'
    var_9 = '12a'
    var_10 = '^[A-Z]+$'
    var_11 = module_0.rex(var_10)
    var_12 = 'HELLO'
    var_13 = 'hello'
    var_14 = 'All test cases passed!'
    var_15 = print(var_14)



# Parsed testcases at query #7
#--------------------------



def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123
    var_5 = ''
    var_6 = '\\d+'
    var_7 = module_0.rex(var_6)
    var_8 = '123'
    var_9 = 'abc'
    var_10 = 'All test cases passed!'
    var_11 = print(var_10)



# Parsed testcases at query #8
#--------------------------



def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = var_1.__name__
    assert var_2 == '<lambda>'
    var_3 = module_0.rex(var_0)
    var_4 = 'abc'
    var_5 = module_0.rex(var_0)
    var_6 = 'bac'
    var_7 = module_0.rex(var_0)
    var_8 = 1



# Parsed testcases at query #9
#--------------------------



def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123
    var_5 = ''
    var_6 = '^\\d+$'
    var_7 = module_0.rex(var_6)
    var_8 = '123'
    var_9 = 'abc'
    var_10 = 'All test cases passed!'
    var_11 = print(var_10)



# Parsed testcases at query #10
#--------------------------



def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = var_1.__code__.co_argcount
    assert var_2 == 1
    var_3 = module_0.rex(var_0)
    var_4 = 'a'
    var_5 = module_0.rex(var_0)
    var_6 = 'b'



# Parsed testcases at query #11
#--------------------------



def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = var_1.__name__
    assert var_2 == '<lambda>'
    var_3 = module_0.rex(var_0)
    var_4 = 'abc'
    var_5 = module_0.rex(var_0)
    var_6 = 'bac'
    var_7 = module_0.rex(var_0)
    var_8 = 5



# Parsed testcases at query #12
#--------------------------



def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'a'
    var_3 = module_0.rex(var_0)
    var_4 = 'ba'
    var_5 = module_0.rex(var_0)
    var_6 = 1
    var_7 = 'All tests passed'
    var_8 = print(var_7)



# Parsed testcases at query #13
#--------------------------



def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123
    var_5 = ''
    var_6 = '^\\d+$'
    var_7 = module_0.rex(var_6)
    var_8 = '123'
    var_9 = 'abc'
    var_10 = 'All test cases passed!'
    var_11 = print(var_10)



# Parsed testcases at query #14
#--------------------------



def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123
    var_5 = ''
    var_6 = '\\d+'
    var_7 = module_0.rex(var_6)
    var_8 = '123'
    var_9 = 'abc'
    var_10 = 'All test cases passed!'
    var_11 = print(var_10)



# Parsed testcases at query #15
#--------------------------



def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123
    var_5 = ''
    var_6 = '^\\d+$'
    var_7 = module_0.rex(var_6)
    var_8 = '123'
    var_9 = 'abc'
    var_10 = 'All test cases passed!'
    var_11 = print(var_10)



# Parsed testcases at query #16
#--------------------------



def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'ab'
    var_5 = 'ba'
    var_6 = 1
    var_7 = None
    var_8 = []
    var_9 = {}
    var_10 = ()
    var_11 = True
    var_12 = False
    var_13 = ''
    var_14 = 100
    var_15 = var_2 * var_14
    var_16 = var_3 * var_14
    var_17 = var_2 * var_14
    var_18 = var_17 + var_3
    var_19 = var_3 * var_14
    var_20 = var_19 + var_2
    var_21 = var_2 * var_14
    var_22 = var_3 * var_14
    var_23 = var_21 + var_22
    var_24 = var_3 * var_14
    var_25 = var_2 * var_14
    var_26 = var_24 + var_25
    var_27 = var_2 * var_14
    var_28 = var_3 * var_14
    var_29 = var_27 + var_28
    var_30 = var_29 + var_2
    var_31 = var_3 * var_14
    var_32 = var_2 * var_14
    var_33 = var_31 + var_32
    var_34 = var_33 + var_3
    var_35 = var_2 * var_14
    var_36 = var_3 * var_14
    var_37 = var_35 + var_36
    var_38 = var_2 * var_14
    var_39 = var_37 + var_38
    var_40 = var_3 * var_14
    var_41 = var_2 * var_14
    var_42 = var_40 + var_41
    var_43 = var_3 * var_14
    var_44 = var_42 + var_43
    var_45 = var_2 * var_14
    var_46 = var_3 * var_14
    var_47 = var_45 + var_46
    var_48 = var_2 * var_14
    var_49 = var_47 + var_48
    var_50 = var_49 + var_3
    var_51 = var_3 * var_14
    var_52 = var_2 * var_14
    var_53 = var_51 + var_52
    var_54 = var_3 * var_14
    var_55 = var_53 + var_54
    var_56 = var_55 + var_2
    var_57 = var_2 * var_14
    var_58 = var_3 * var_14
    var_59 = var_57 + var_58
    var_60 = var_2 * var_14
    var_61 = var_59 + var_60
    var_62 = var_3 * var_14
    var_63 = var_61 + var_62
    var_64 = var_3 * var_14
    var_65 = var_2 * var_14
    var_66 = var_64 + var_65
    var_67 = var_3 * var_14
    var_68 = var_66 + var_67
    var_69 = var_2 * var_14
    var_70 = var_68 + var_69
    var_71 = var_2 * var_14
    var_72 = var_3 * var_14
    var_73 = var_71 + var_72
    var_74 = var_2 * var_14
    var_75 = var_73 + var_74
    var_76 = var_3 * var_14
    var_77 = var_75 + var_76
    var_78 = var_77 + var_2
    var_79 = var_3 * var_14
    var_80 = var_2 * var_14
    var_81 = var_79 + var_80
    var_82 = var_3 * var_14
    var_83 = var_81 + var_82
    var_84 = var_2 * var_14
    var_85 = var_83 + var_84
    var_86 = var_85 + var_3
    var_87 = var_2 * var_14
    var_88 = var_3 * var_14
    var_89 = var_87 + var_88
    var_90 = var_2 * var_14
    var_91 = var_89 + var_90
    var_92 = var_3 * var_14
    var_93 = var_91 + var_92
    var_94 = var_2 * var_14
    var_95 = var_93 + var_94
    var_96 = var_3 * var_14
    var_97 = var_2 * var_14
    var_98 = var_96 + var_97
    var_99 = var_3 * var_14
    var_100 = var_98 + var_99
    var_101 = var_2 * var_14
    var_102 = var_100 + var_101
    var_103 = var_3 * var_14
    var_104 = var_102 + var_103
    var_105 = var_2 * var_14
    var_106 = var_3 * var_14
    var_107 = var_105 + var_106
    var_108 = var_2 * var_14
    var_109 = var_107 + var_108
    var_110 = var_3 * var_14
    var_111 = var_109 + var_110
    var_112 = var_2 * var_14
    var_113 = var_111 + var_112
    var_114 = var_113 + var_3
    var_115 = var_3 * var_14
    var_116 = var_2 * var_14
    var_117 = var_115 + var_116
    var_118 = var_3 * var_14
    var_119 = var_117 + var_118
    var_120 = var_2 * var_14
    var_121 = var_119 + var_120
    var_122 = var_3 * var_14
    var_123 = var_121 + var_122
    var_124 = var_123 + var_2
    var_125 = var_2 * var_14
    var_126 = var_3 * var_14
    var_127 = var_125 + var_126
    var_128 = var_2 * var_14
    var_129 = var_127 + var_128
    var_130 = var_3 * var_14
    var_131 = var_129 + var_130
    var_132 = var_2 * var_14
    var_133 = var_131 + var_132
    var_134 = var_3 * var_14
    var_135 = var_133 + var_134
    var_136 = var_3 * var_14
    var_137 = var_2 * var_14
    var_138 = var_136 + var_137
    var_139 = var_3 * var_14
    var_140 = var_138 + var_139
    var_141 = var_2 * var_14
    var_142 = var_140 + var_141
    var_143 = var_3 * var_14
    var_144 = var_142 + var_143
    var_145 = var_2 * var_14
    var_146 = var_144 + var_145
    var_147 = var_2 * var_14
    var_148 = var_3 * var_14
    var_149 = var_147 + var_148
    var_150 = var_2 * var_14
    var_151 = var_149 + var_150
    var_152 = var_3 * var_14
    var_153 = var_151 + var_152
    var_154 = var_2 * var_14
    var_155 = var_153 + var_154
    var_156 = var_3 * var_14
    var_157 = var_155 + var_156
    var_158 = var_157 + var_2
    var_159 = var_3 * var_14
    var_160 = var_2 * var_14
    var_161 = var_159 + var_160
    var_162 = var_3 * var_14
    var_163 = var_161 + var_162
    var_164 = var_2 * var_14
    var_165 = var_163 + var_164
    var_166 = var_3 * var_14
    var_167 = var_165 + var_166
    var_168 = var_2 * var_14
    var_169 = var_167 + var_168
    var_170 = var_169 + var_3
    var_171 = var_2 * var_14
    var_172 = var_3 * var_14
    var_173 = var_171 + var_172
    var_174 = var_2 * var_14
    var_175 = var_173 + var_174
    var_176 = var_3 * var_14
    var_177 = var_175 + var_176
    var_178 = var_2 * var_14
    var_179 = var_177 + var_178
    var_180 = var_3 * var_14
    var_181 = var_179 + var_180
    var_182 = var_2 * var_14
    var_183 = var_181 + var_182
    var_184 = var_3 * var_14
    var_185 = var_2 * var_14
    var_186 = var_184 + var_185
    var_187 = var_3 * var_14
    var_188 = var_186 + var_187
    var_189 = var_2 * var_14
    var_190 = var_188 + var_189
    var_191 = var_3 * var_14
    var_192 = var_190 + var_191
    var_193 = var_2 * var_14
    var_194 = var_192 + var_193
    var_195 = var_3 * var_14
    var_196 = var_194 + var_195
    var_197 = var_2 * var_14
    var_198 = var_3 * var_14
    var_199 = var_197 + var_198
    var_200 = var_2 * var_14
    var_201 = var_199 + var_200
    var_202 = var_3 * var_14
    var_203 = var_201 + var_202
    var_204 = var_2 * var_14
    var_205 = var_203 + var_204
    var_206 = var_3 * var_14
    var_207 = var_205 + var_206
    var_208 = var_2 * var_14
    var_209 = var_207 + var_208
    var_210 = var_209 + var_3
    var_211 = var_3 * var_14
    var_212 = var_2 * var_14
    var_213 = var_211 + var_212
    var_214 = var_3 * var_14
    var_215 = var_213 + var_214
    var_216 = var_2 * var_14
    var_217 = var_215 + var_216
    var_218 = var_3 * var_14
    var_219 = var_217 + var_218
    var_220 = var_2 * var_14
    var_221 = var_219 + var_220
    var_222 = var_3 * var_14
    var_223 = var_221 + var_222
    var_224 = var_223 + var_2
    var_225 = var_2 * var_14
    var_226 = var_3 * var_14
    var_227 = var_225 + var_226
    var_228 = var_2 * var_14
    var_229 = var_227 + var_228
    var_230 = var_3 * var_14
    var_231 = var_229 + var_230
    var_232 = var_2 * var_14
    var_233 = var_231 + var_232
    var_234 = var_3 * var_14
    var_235 = var_233 + var_234
    var_236 = var_2 * var_14
    var_237 = var_235 + var_236
    var_238 = var_3 * var_14
    var_239 = var_237 + var_238
    var_240 = var_3 * var_14
    var_241 = var_2 * var_14
    var_242 = var_240 + var_241
    var_243 = var_3 * var_14
    var_244 = var_242 + var_243
    var_245 = var_2 * var_14
    var_246 = var_244 + var_245
    var_247 = var_3 * var_14
    var_248 = var_246 + var_247
    var_249 = var_2 * var_14
    var_250 = var_248 + var_249
    var_251 = var_3 * var_14
    var_252 = var_250 + var_251
    var_253 = var_2 * var_14
    var_254 = var_252 + var_253
    var_255 = var_2 * var_14
    var_256 = var_3 * var_14
    var_257 = var_255 + var_256
    var_258 = var_2 * var_14
    var_259 = var_257 + var_258
    var_260 = var_3 * var_14
    var_261 = var_259 + var_260
    var_262 = var_2 * var_14
    var_263 = var_261 + var_262
    var_264 = var_3 * var_14
    var_265 = var_263 + var_264
    var_266 = var_2 * var_14
    var_267 = var_265 + var_266
    var_268 = var_3 * var_14
    var_269 = var_267 + var_268
    var_270 = var_269 + var_2
    var_271 = var_3 * var_14
    var_272 = var_2 * var_14
    var_273 = var_271 + var_272
    var_274 = var_3 * var_14
    var_275 = var_273 + var_274
    var_276 = var_2 * var_14
    var_277 = var_275 + var_276
    var_278 = var_3 * var_14
    var_279 = var_277 + var_278
    var_280 = var_2 * var_14
    var_281 = var_279 + var_280
    var_282 = var_3 * var_14
    var_283 = var_281 + var_282
    var_284 = var_2 * var_14
    var_285 = var_283 + var_284
    var_286 = var_285 + var_3
    var_287 = var_2 * var_14
    var_288 = var_3 * var_14
    var_289 = var_287 + var_288
    var_290 = var_2 * var_14
    var_291 = var_289 + var_290
    var_292 = var_3 * var_14
    var_293 = var_291 + var_292
    var_294 = var_2 * var_14
    var_295 = var_293 + var_294
    var_296 = var_3 * var_14
    var_297 = var_295 + var_296
    var_298 = var_2 * var_14
    var_299 = var_297 + var_298
    var_300 = var_3 * var_14
    var_301 = var_299 + var_300
    var_302 = var_2 * var_14
    var_303 = var_301 + var_302
    var_304 = var_3 * var_14
    var_305 = var_2 * var_14
    var_306 = var_304 + var_305
    var_307 = var_3 * var_14
    var_308 = var_306 + var_307
    var_309 = var_2 * var_14
    var_310 = var_308 + var_309
    var_311 = var_3 * var_14
    var_312 = var_310 + var_311
    var_313 = var_2 * var_14
    var_314 = var_312 + var_313
    var_315 = var_3 * var_14
    var_316 = var_314 + var_315
    var_317 = var_2 * var_14
    var_318 = var_316 + var_317
    var_319 = var_3 * var_14
    var_320 = var_318 + var_319
    var_321 = var_2 * var_14
    var_322 = var_3 * var_14
    var_323 = var_321 + var_322
    var_324 = var_2 * var_14
    var_325 = var_323 + var_324
    var_326 = var_3 * var_14
    var_327 = var_325 + var_326
    var_328 = var_2 * var_14
    var_329 = var_327 + var_328
    var_330 = var_3 * var_14
    var_331 = var_329 + var_330
    var_332 = var_2 * var_14
    var_333 = var_331 + var_332
    var_334 = var_3 * var_14
    var_335 = var_333 + var_334
    var_336 = var_2 * var_14
    var_337 = var_335 + var_336
    var_338 = var_337 + var_3
    var_339 = var_3 * var_14
    var_340 = var_2 * var_14
    var_341 = var_339 + var_340
    var_342 = var_3 * var_14
    var_343 = var_341 + var_342
    var_344 = var_2 * var_14
    var_345 = var_343 + var_344
    var_346 = var_3 * var_14
    var_347 = var_345 + var_346
    var_348 = var_2 * var_14
    var_349 = var_347 + var_348
    var_350 = var_3 * var_14
    var_351 = var_349 + var_350
    var_352 = var_2 * var_14
    var_353 = var_351 + var_352
    var_354 = var_3 * var_14
    var_355 = var_353 + var_354
    var_356 = var_355 + var_2
    var_357 = var_2 * var_14
    var_358 = var_3 * var_14
    var_359 = var_357 + var_358
    var_360 = var_2 * var_14
    var_361 = var_359 + var_360
    var_362 = var_3 * var_14
    var_363 = var_361 + var_362
    var_364 = var_2 * var_14
    var_365 = var_363 + var_364
    var_366 = var_3 * var_14
    var_367 = var_365 + var_366
    var_368 = var_2 * var_14
    var_369 = var_367 + var_368
    var_370 = var_3 * var_14
    var_371 = var_369 + var_370
    var_372 = var_2 * var_14
    var_373 = var_371 + var_372
    var_374 = var_3 * var_14
    var_375 = var_373 + var_374
    var_376 = var_3 * var_14
    var_377 = var_2 * var_14
    var_378 = var_376 + var_377
    var_379 = var_3 * var_14
    var_380 = var_378 + var_379
    var_381 = var_2 * var_14
    var_382 = var_380 + var_381
    var_383 = var_3 * var_14
    var_384 = var_382 + var_383
    var_385 = var_2 * var_14
    var_386 = var_384 + var_385
    var_387 = var_3 * var_14
    var_388 = var_386 + var_387
    var_389 = var_2 * var_14
    var_390 = var_388 + var_389
    var_391 = var_3 * var_14
    var_392 = var_390 + var_391
    var_393 = var_2 * var_14
    var_394 = var_392 + var_393
    var_395 = var_2 * var_14
    var_396 = var_3 * var_14
    var_397 = var_395 + var_396
    var_398 = var_2 * var_14
    var_399 = var_397 + var_398
    var_400 = var_3 * var_14
    var_401 = var_399 + var_400
    var_402 = var_2 * var_14
    var_403 = var_401 + var_402
    var_404 = var_3 * var_14
    var_405 = var_403 + var_404
    var_406 = var_2 * var_14
    var_407 = var_405 + var_406
    var_408 = var_3 * var_14
    var_409 = var_407 + var_408
    var_410 = var_2 * var_14
    var_411 = var_409 + var_410
    var_412 = var_3 * var_14
    var_413 = var_411 + var_412
    var_414 = var_413 + var_2
    var_415 = var_3 * var_14
    var_416 = var_2 * var_14
    var_417 = var_415 + var_416
    var_418 = var_3 * var_14
    var_419 = var_417 + var_418
    var_420 = var_2 * var_14
    var_421 = var_419 + var_420
    var_422 = var_3 * var_14
    var_423 = var_421 + var_422
    var_424 = var_2 * var_14
    var_425 = var_423 + var_424
    var_426 = var_3 * var_14
    var_427 = var_425 + var_426
    var_428 = var_2 * var_14
    var_429 = var_427 + var_428
    var_430 = var_3 * var_14
    var_431 = var_429 + var_430
    var_432 = var_2 * var_14
    var_433 = var_431 + var_432
    var_434 = var_433 + var_3
    var_435 = var_2 * var_14
    var_436 = var_3 * var_14
    var_437 = var_435 + var_436
    var_438 = var_2 * var_14
    var_439 = var_437 + var_438
    var_440 = var_3 * var_14
    var_441 = var_439 + var_440
    var_442 = var_2 * var_14
    var_443 = var_441 + var_442
    var_444 = var_3 * var_14
    var_445 = var_443 + var_444
    var_446 = var_2 * var_14
    var_447 = var_445 + var_446
    var_448 = var_3 * var_14
    var_449 = var_447 + var_448
    var_450 = var_2 * var_14
    var_451 = var_449 + var_450
    var_452 = var_3 * var_14
    var_453 = var_451 + var_452
    var_454 = var_2 * var_14
    var_455 = var_453 + var_454
    var_456 = var_3 * var_14
    var_457 = var_2 * var_14
    var_458 = var_456 + var_457
    var_459 = var_3 * var_14
    var_460 = var_458 + var_459
    var_461 = var_2 * var_14
    var_462 = var_460 + var_461
    var_463 = var_3 * var_14
    var_464 = var_462 + var_463
    var_465 = var_2 * var_14
    var_466 = var_464 + var_465
    var_467 = var_3 * var_14
    var_468 = var_466 + var_467
    var_469 = var_2 * var_14
    var_470 = var_468 + var_469
    var_471 = var_3 * var_14
    var_472 = var_470 + var_471
    var_473 = var_2 * var_14
    var_474 = var_472 + var_473
    var_475 = var_3 * var_14
    var_476 = var_474 + var_475
    var_477 = var_2 * var_14
    var_478 = var_3 * var_14
    var_479 = var_477 + var_478
    var_480 = var_2 * var_14
    var_481 = var_479 + var_480
    var_482 = var_3 * var_14
    var_483 = var_481 + var_482
    var_484 = var_2 * var_14
    var_485 = var_483 + var_484
    var_486 = var_3 * var_14
    var_487 = var_485 + var_486
    var_488 = var_2 * var_14
    var_489 = var_487 + var_488
    var_490 = var_3 * var_14
    var_491 = var_489 + var_490
    var_492 = var_2 * var_14
    var_493 = var_491 + var_492
    var_494 = var_3 * var_14
    var_495 = var_493 + var_494
    var_496 = var_2 * var_14
    var_497 = var_495 + var_496
    var_498 = var_497 + var_3
    var_499 = var_3 * var_14
    var_500 = var_2 * var_14
    var_501 = var_499 + var_500
    var_502 = var_3 * var_14
    var_503 = var_501 + var_502
    var_504 = var_2 * var_14
    var_505 = var_503 + var_504
    var_506 = var_3 * var_14
    var_507 = var_505 + var_506
    var_508 = var_2 * var_14
    var_509 = var_507 + var_508
    var_510 = var_3 * var_14
    var_511 = var_509 + var_510
    var_512 = var_2 * var_14
    var_513 = var_511 + var_512
    var_514 = var_3 * var_14
    var_515 = var_513 + var_514
    var_516 = var_2 * var_14
    var_517 = var_515 + var_516
    var_518 = var_3 * var_14
    var_519 = var_517 + var_518
    var_520 = var_519 + var_2



# Parsed testcases at query #17
#--------------------------



def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123
    var_5 = ''
    var_6 = '^\\d+$'
    var_7 = module_0.rex(var_6)
    var_8 = '123'
    var_9 = 'abc'
    var_10 = 'All test cases passed!'
    var_11 = print(var_10)



# Parsed testcases at query #18
#--------------------------



def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123
    var_5 = ''
    var_6 = '\\d+'
    var_7 = module_0.rex(var_6)
    var_8 = '123'
    var_9 = 'abc'
    var_10 = 'All test cases passed!'
    var_11 = print(var_10)



# Parsed testcases at query #19
#--------------------------



def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123
    var_5 = ''
    var_6 = '^\\d+$'
    var_7 = module_0.rex(var_6)
    var_8 = '123'
    var_9 = 'abc'
    var_10 = 'All test cases passed!'
    var_11 = print(var_10)



# Parsed testcases at query #20
#--------------------------



def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = var_1.__name__
    assert var_2 == '<lambda>'
    var_3 = module_0.rex(var_0)
    var_4 = 'abc'
    var_5 = module_0.rex(var_0)
    var_6 = 'bac'
    var_7 = module_0.rex(var_0)
    var_8 = 5
    var_9 = 'All tests passed.'
    var_10 = print(var_9)



# Parsed testcases at query #21
#--------------------------



def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'rex should return a callable function'
    var_4 = 'apple'
    var_5 = 'banana'
    var_6 = 123
    var_7 = '^[0-9]+$'
    var_8 = module_0.rex(var_7)
    var_9 = '123'
    var_10 = 'abc'
    var_11 = 'All tests passed!'
    var_12 = print(var_11)



# Parsed testcases at query #22
#--------------------------



def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123
    var_5 = ''
    var_6 = '\\d+'
    var_7 = module_0.rex(var_6)
    var_8 = '123'
    var_9 = 'abc'
    var_10 = 'All test cases passed!'
    var_11 = print(var_10)



# Parsed testcases at query #23
#--------------------------



def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123
    var_5 = ''
    var_6 = '\\d+'
    var_7 = module_0.rex(var_6)
    var_8 = '123'
    var_9 = 'abc'
    var_10 = 'All test cases passed!'
    var_11 = print(var_10)



# Parsed testcases at query #24
#--------------------------



def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123
    var_5 = ''
    var_6 = '^\\d+$'
    var_7 = module_0.rex(var_6)
    var_8 = '123'
    var_9 = '12a'
    var_10 = '^[A-Z]+$'
    var_11 = module_0.rex(var_10)
    var_12 = 'HELLO'
    var_13 = 'hello'
    var_14 = 'All test cases passed!'
    var_15 = print(var_14)



# Parsed testcases at query #25
#--------------------------



def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123
    var_5 = ''
    var_6 = '^\\d+$'
    var_7 = module_0.rex(var_6)
    var_8 = '123'
    var_9 = '12a'
    var_10 = '^hello'
    var_11 = module_0.rex(var_10)
    var_12 = 'Hello'
    var_13 = 'HELLO'
    var_14 = 'hello'
    var_15 = 'hi'
    var_16 = '^(\\w+)\\s(\\w+)$'
    var_17 = module_0.rex(var_16)
    var_18 = 'John Doe'
    var_19 = 'John'
    var_20 = '^a+b*$'
    var_21 = module_0.rex(var_20)
    var_22 = 'aaabbb'
    var_23 = 'aaa'
    var_24 = 'bbb'
    var_25 = '^start.*end$'
    var_26 = module_0.rex(var_25)
    var_27 = 'start middle end'
    var_28 = 'start end'
    var_29 = 'start middle'
    var_30 = '^[A-Z][a-z]+$'
    var_31 = module_0.rex(var_30)
    var_32 = 'H'
    var_33 = '^\\d+(?=px)$'
    var_34 = module_0.rex(var_33)
    var_35 = '100px'
    var_36 = '100em'
    var_37 = 'All test cases passed!'
    var_38 = print(var_37)



# Parsed testcases at query #26
#--------------------------



def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123
    var_5 = ''
    var_6 = '^\\d+$'
    var_7 = module_0.rex(var_6)
    var_8 = '123'
    var_9 = '12a'
    var_10 = 'All test cases passed!'
    var_11 = print(var_10)



# Parsed testcases at query #27
#--------------------------


import builtins as module_1


def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'a'
    var_3 = module_0.rex(var_0)
    var_4 = 'ba'
    var_5 = module_0.rex(var_0)
    var_6 = 1
    var_7 = module_0.rex(var_0)
    var_8 = None
    var_9 = module_0.rex(var_0)
    var_10 = []
    var_11 = module_0.rex(var_0)
    var_12 = {}
    var_13 = module_0.rex(var_0)
    var_14 = ()
    var_15 = module_0.rex(var_0)
    var_16 = set()
    var_17 = module_0.rex(var_0)
    var_18 = frozenset()
    var_19 = module_0.rex(var_0)
    var_20 = module_1.object()
    var_21 = module_0.rex(var_0)
    var_22 = lambda x: x
    var_23 = module_0.rex(var_0)
    var_24 = module_0.rex(var_0)
    var_25 = ''
    var_26 = module_0.rex(var_0)
    var_27 = ()
    var_28 = {}
    var_29 = module_0.rex(var_0)
    var_30 = ()
    var_31 = {}
    var_32 = module_0.rex(var_0)
    var_33 = ()
    var_34 = {}
    var_35 = module_0.rex(var_0)
    var_36 = ()
    var_37 = {}
    var_38 = module_0.rex(var_0)
    var_39 = ()
    var_40 = {}
    var_41 = module_0.rex(var_0)
    var_42 = ()
    var_43 = {}
    var_44 = module_0.rex(var_0)
    var_45 = ()
    var_46 = {}
    var_47 = module_0.rex(var_0)
    var_48 = ()
    var_49 = {}
    var_50 = module_0.rex(var_0)
    var_51 = ()
    var_52 = {}
    var_53 = module_0.rex(var_0)
    var_54 = ()
    var_55 = {}
    var_56 = module_0.rex(var_0)
    var_57 = ()
    var_58 = {}
    var_59 = module_0.rex(var_0)
    var_60 = ()
    var_61 = {}
    var_62 = module_0.rex(var_0)
    var_63 = ()
    var_64 = {}
    var_65 = module_0.rex(var_0)
    var_66 = ()
    var_67 = {}
    var_68 = module_0.rex(var_0)
    var_69 = ()
    var_70 = {}
    var_71 = module_0.rex(var_0)
    var_72 = ()
    var_73 = {}
    var_74 = module_0.rex(var_0)
    var_75 = ()
    var_76 = {}
    var_77 = module_0.rex(var_0)
    var_78 = ()
    var_79 = {}
    var_80 = module_0.rex(var_0)
    var_81 = ()
    var_82 = {}
    var_83 = module_0.rex(var_0)
    var_84 = ()
    var_85 = {}
    var_86 = module_0.rex(var_0)
    var_87 = ()
    var_88 = {}
    var_89 = module_0.rex(var_0)
    var_90 = ()
    var_91 = {}
    var_92 = module_0.rex(var_0)
    var_93 = ()
    var_94 = {}
    var_95 = module_0.rex(var_0)
    var_96 = ()
    var_97 = {}
    var_98 = module_0.rex(var_0)
    var_99 = ()
    var_100 = {}
    var_101 = module_0.rex(var_0)
    var_102 = ()
    var_103 = {}
    var_104 = module_0.rex(var_0)
    var_105 = ()
    var_106 = {}
    var_107 = module_0.rex(var_0)
    var_108 = ()
    var_109 = {}



####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 'b'
    var_4 = 'd'
    var_5 = 4
    var_6 = 5
    var_7 = 10
    var_8 = 'All tests passed!'
    var_9 = print(var_8)



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 'b'
    var_4 = 'd'
    var_5 = 4
    var_6 = 5
    var_7 = 10
    var_8 = 'All discard tests passed.'
    var_9 = print(var_8)



# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123
    var_5 = ''
    var_6 = '\\d+'
    var_7 = module_0.rex(var_6)
    var_8 = '123'
    var_9 = 'abc'
    var_10 = 'All test cases passed!'
    var_11 = print(var_10)



# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123
    var_5 = ''
    var_6 = '^\\d+$'
    var_7 = module_0.rex(var_6)
    var_8 = '123'
    var_9 = 'abc'
    var_10 = 'All test cases passed!'
    var_11 = print(var_10)



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 'b'
    var_4 = 'd'
    var_5 = 5
    var_6 = 'All tests passed for discard function.'
    var_7 = print(var_6)



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 'b'
    var_4 = 'd'
    var_5 = 5
    var_6 = 'All tests passed for discard function.'
    var_7 = print(var_6)



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 'b'
    var_4 = 'd'
    var_5 = 5
    var_6 = 4
    var_7 = 'All tests passed!'
    var_8 = print(var_7)



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 'b'
    var_4 = 'd'
    var_5 = 'All tests passed!'
    var_6 = print(var_5)



# Parsed testcases at query #9
#--------------------------



def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123
    var_5 = ''
    var_6 = '\\d+'
    var_7 = module_0.rex(var_6)
    var_8 = '123'
    var_9 = 'abc'
    var_10 = 'All test cases passed!'
    var_11 = print(var_10)



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 'b'
    var_4 = 'd'
    var_5 = 4
    var_6 = 5
    var_7 = 'a'
    var_8 = 'All tests passed!'
    var_9 = print(var_8)



# Parsed testcases at query #11
#--------------------------



def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123
    var_5 = ''
    var_6 = '^\\d+$'
    var_7 = module_0.rex(var_6)
    var_8 = '123'
    var_9 = 'abc'
    var_10 = 'All test cases passed!'
    var_11 = print(var_10)



# Parsed testcases at query #12
#--------------------------



def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.rex(var_0)
    var_2 = 'abcd'
    var_3 = 'ab'
    var_4 = ''
    var_5 = 123
    var_6 = '^abc'
    var_7 = module_0.rex(var_6)
    var_8 = 'abc$'
    var_9 = module_0.rex(var_8)
    var_10 = 'dabc'
    var_11 = '^[0-9]+$'
    var_12 = module_0.rex(var_11)
    var_13 = '123'
    var_14 = '^[a-zA-Z]+$'
    var_15 = module_0.rex(var_14)
    var_16 = 'ABC'
    var_17 = '^[a-zA-Z0-9]+$'
    var_18 = module_0.rex(var_17)
    var_19 = 'abc123'
    var_20 = 'ABC123'
    var_21 = '^[a-zA-Z0-9_]+$'
    var_22 = module_0.rex(var_21)
    var_23 = 'abc_123'
    var_24 = 'ABC_123'
    var_25 = '^[a-zA-Z0-9_-]+$'
    var_26 = module_0.rex(var_25)
    var_27 = 'abc-123'
    var_28 = '^[a-zA-Z0-9_.-]+$'
    var_29 = module_0.rex(var_28)
    var_30 = 'abc.123'
    var_31 = '^[a-zA-Z0-9_. -]+$'
    var_32 = module_0.rex(var_31)
    var_33 = 'abc 123'
    var_34 = '^[a-zA-Z0-9_. @-]+$'
    var_35 = module_0.rex(var_34)
    var_36 = 'abc@123'
    var_37 = '^[a-zA-Z0-9_. +@-]+$'
    var_38 = module_0.rex(var_37)
    var_39 = 'abc+123'
    var_40 = '^[a-zA-Z0-9_. +@()-]+$'
    var_41 = module_0.rex(var_40)
    var_42 = 'abc(123)'
    var_43 = '^[a-zA-Z0-9_. +@()\\[\\]-]+$'
    var_44 = module_0.rex(var_43)
    var_45 = 'abc[123]'
    var_46 = '^[a-zA-Z0-9_. +@()\\[\\]{}-]+$'
    var_47 = module_0.rex(var_46)
    var_48 = 'abc{123}'
    var_49 = '^[a-zA-Z0-9_. +@()\\[\\]{}<>-]+$'
    var_50 = module_0.rex(var_49)
    var_51 = 'abc<123>'
    var_52 = '^[a-zA-Z0-9_. +@()\\[\\]{}\\\\<>-]+$'
    var_53 = module_0.rex(var_52)
    var_54 = 'abc\\123'
    var_55 = '^[a-zA-Z0-9_. +@()\\[\\]{}\\\\<>/-]+$'
    var_56 = module_0.rex(var_55)
    var_57 = 'abc/123'
    var_58 = '^[a-zA-Z0-9_. +@()\\[\\]{}\\\\<>/:;-]+$'
    var_59 = module_0.rex(var_58)
    var_60 = 'abc:123'



# Parsed testcases at query #13
#--------------------------



def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = var_1.__name__
    assert var_2 == '<lambda>'
    var_3 = module_0.rex(var_0)
    var_4 = 'abc'
    var_5 = module_0.rex(var_0)
    var_6 = 'bac'
    var_7 = module_0.rex(var_0)
    var_8 = 'a'
    var_9 = module_0.rex(var_0)
    var_10 = 'b'
    var_11 = module_0.rex(var_0)
    var_12 = 1
    var_13 = module_0.rex(var_0)
    var_14 = None
    var_15 = module_0.rex(var_0)
    var_16 = []
    var_17 = module_0.rex(var_0)
    var_18 = {}
    var_19 = module_0.rex(var_0)
    var_20 = ()
    var_21 = module_0.rex(var_0)
    var_22 = set()
    var_23 = module_0.rex(var_0)
    var_24 = frozenset()
    var_25 = module_0.rex(var_0)
    var_26 = module_1.object()
    var_27 = module_0.rex(var_0)
    var_28 = module_0.rex(var_0)
    var_29 = module_0.rex(var_0)
    var_30 = True
    var_31 = module_0.rex(var_0)
    var_32 = False
    var_33 = module_0.rex(var_0)
    var_34 = module_0.rex(var_0)
    var_35 = module_0.rex(var_0)
    var_36 = module_0.rex(var_0)
    var_37 = module_0.rex(var_0)
    var_38 = module_0.rex(var_0)
    var_39 = module_0.rex(var_0)
    var_40 = module_0.rex(var_0)
    var_41 = module_0.rex(var_0)
    var_42 = []
    var_43 = module_0.rex(var_0)
    var_44 = {}
    var_45 = module_0.rex(var_0)
    var_46 = ()
    var_47 = module_0.rex(var_0)
    var_48 = set()
    var_49 = module_0.rex(var_0)
    var_50 = frozenset()
    var_51 = module_0.rex(var_0)
    var_52 = module_1.object()
    var_53 = module_0.rex(var_0)
    var_54 = module_0.rex(var_0)
    var_55 = module_0.rex(var_0)
    var_56 = True
    var_57 = module_0.rex(var_0)
    var_58 = module_0.rex(var_0)
    var_59 = module_0.rex(var_0)
    var_60 = module_0.rex(var_0)
    var_61 = module_0.rex(var_0)
    var_62 = module_0.rex(var_0)
    var_63 = module_0.rex(var_0)
    var_64 = module_0.rex(var_0)
    var_65 = module_0.rex(var_0)
    var_66 = module_0.rex(var_0)
    var_67 = []
    var_68 = module_0.rex(var_0)
    var_69 = {}
    var_70 = module_0.rex(var_0)
    var_71 = ()
    var_72 = module_0.rex(var_0)
    var_73 = set()
    var_74 = module_0.rex(var_0)
    var_75 = frozenset()
    var_76 = module_0.rex(var_0)
    var_77 = module_1.object()
    var_78 = module_0.rex(var_0)
    var_79 = module_0.rex(var_0)
    var_80 = module_0.rex(var_0)
    var_81 = True
    var_82 = module_0.rex(var_0)
    var_83 = module_0.rex(var_0)
    var_84 = module_0.rex(var_0)
    var_85 = module_0.rex(var_0)
    var_86 = module_0.rex(var_0)
    var_87 = module_0.rex(var_0)
    var_88 = module_0.rex(var_0)
    var_89 = module_0.rex(var_0)
    var_90 = module_0.rex(var_0)
    var_91 = module_0.rex(var_0)
    var_92 = []
    var_93 = module_0.rex(var_0)
    var_94 = {}
    var_95 = module_0.rex(var_0)
    var_96 = ()
    var_97 = module_0.rex(var_0)
    var_98 = set()
    var_99 = module_0.rex(var_0)
    var_100 = frozenset()
    var_101 = module_0.rex(var_0)
    var_102 = module_1.object()
    var_103 = module_0.rex(var_0)
    var_104 = module_0.rex(var_0)
    var_105 = module_0.rex(var_0)
    var_106 = True
    var_107 = module_0.rex(var_0)
    var_108 = module_0.rex(var_0)
    var_109 = module_0.rex(var_0)
    var_110 = module_0.rex(var_0)
    var_111 = module_0.rex(var_0)
    var_112 = module_0.rex(var_0)
    var_113 = module_0.rex(var_0)
    var_114 = module_0.rex(var_0)
    var_115 = module_0.rex(var_0)
    var_116 = module_0.rex(var_0)
    var_117 = []
    var_118 = module_0.rex(var_0)
    var_119 = {}
    var_120 = module_0.rex(var_0)
    var_121 = ()
    var_122 = module_0.rex(var_0)
    var_123 = set()
    var_124 = module_0.rex(var_0)
    var_125 = frozenset()
    var_126 = module_0.rex(var_0)
    var_127 = module_1.object()
    var_128 = module_0.rex(var_0)
    var_129 = module_0.rex(var_0)
    var_130 = module_0.rex(var_0)
    var_131 = True
    var_132 = module_0.rex(var_0)
    var_133 = module_0.rex(var_0)
    var_134 = module_0.rex(var_0)
    var_135 = module_0.rex(var_0)
    var_136 = module_0.rex(var_0)
    var_137 = module_0.rex(var_0)
    var_138 = module_0.rex(var_0)
    var_139 = module_0.rex(var_0)
    var_140 = module_0.rex(var_0)
    var_141 = module_0.rex(var_0)
    var_142 = []
    var_143 = module_0.rex(var_0)
    var_144 = {}
    var_145 = module_0.rex(var_0)
    var_146 = ()
    var_147 = module_0.rex(var_0)
    var_148 = set()
    var_149 = module_0.rex(var_0)
    var_150 = frozenset()
    var_151 = module_0.rex(var_0)
    var_152 = module_1.object()
    var_153 = module_0.rex(var_0)
    var_154 = module_0.rex(var_0)
    var_155 = module_0.rex(var_0)
    var_156 = True
    var_157 = module_0.rex(var_0)
    var_158 = module_0.rex(var_0)
    var_159 = module_0.rex(var_0)
    var_160 = module_0.rex(var_0)
    var_161 = module_0.rex(var_0)
    var_162 = module_0.rex(var_0)
    var_163 = module_0.rex(var_0)
    var_164 = module_0.rex(var_0)
    var_165 = module_0.rex(var_0)
    var_166 = module_0.rex(var_0)
    var_167 = []
    var_168 = module_0.rex(var_0)
    var_169 = {}
    var_170 = module_0.rex(var_0)
    var_171 = ()
    var_172 = module_0.rex(var_0)
    var_173 = set()
    var_174 = module_0.rex(var_0)
    var_175 = frozenset()
    var_176 = module_0.rex(var_0)
    var_177 = module_1.object()
    var_178 = module_0.rex(var_0)
    var_179 = module_0.rex(var_0)
    var_180 = module_0.rex(var_0)
    var_181 = True
    var_182 = module_0.rex(var_0)
    var_183 = module_0.rex(var_0)
    var_184 = module_0.rex(var_0)
    var_185 = module_0.rex(var_0)
    var_186 = module_0.rex(var_0)
    var_187 = module_0.rex(var_0)
    var_188 = module_0.rex(var_0)
    var_189 = module_0.rex(var_0)
    var_190 = module_0.rex(var_0)
    var_191 = module_0.rex(var_0)
    var_192 = []
    var_193 = module_0.rex(var_0)
    var_194 = {}
    var_195 = module_0.rex(var_0)
    var_196 = ()
    var_197 = module_0.rex(var_0)
    var_198 = set()
    var_199 = module_0.rex(var_0)
    var_200 = frozenset()
    var_201 = module_0.rex(var_0)
    var_202 = module_1.object()
    var_203 = module_0.rex(var_0)
    var_204 = module_0.rex(var_0)
    var_205 = module_0.rex(var_0)
    var_206 = True
    var_207 = module_0.rex(var_0)
    var_208 = module_0.rex(var_0)
    var_209 = module_0.rex(var_0)
    var_210 = module_0.rex(var_0)
    var_211 = module_0.rex(var_0)
    var_212 = module_0.rex(var_0)
    var_213 = module_0.rex(var_0)
    var_214 = module_0.rex(var_0)
    var_215 = module_0.rex(var_0)



# Parsed testcases at query #14
#--------------------------



def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123
    var_5 = ''
    var_6 = '^\\d+$'
    var_7 = module_0.rex(var_6)
    var_8 = '123'
    var_9 = 'abc'
    var_10 = 'All test cases passed!'
    var_11 = print(var_10)



# Parsed testcases at query #15
#--------------------------



def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123
    var_5 = ''
    var_6 = '^\\d+$'
    var_7 = module_0.rex(var_6)
    var_8 = '123'
    var_9 = 'abc'
    var_10 = 'All test cases passed!'
    var_11 = print(var_10)



# Parsed testcases at query #16
#--------------------------



def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'a'
    var_3 = module_0.rex(var_0)
    var_4 = 'ba'
    var_5 = module_0.rex(var_0)
    var_6 = 1
    var_7 = 'All tests passed'
    var_8 = print(var_7)



# Parsed testcases at query #17
#--------------------------



def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = '^A.*'
    var_5 = module_0.rex(var_4)
    var_6 = 'Apple'
    var_7 = '^\\d+$'
    var_8 = module_0.rex(var_7)
    var_9 = '123'
    var_10 = 'abc'
    var_11 = '^$'
    var_12 = module_0.rex(var_11)
    var_13 = ''
    var_14 = 'a'
    var_15 = module_0.rex(var_0)
    var_16 = 'apple\nbanana'
    var_17 = 'banana\napple'
    var_18 = 'All test cases passed!'
    var_19 = print(var_18)



# Parsed testcases at query #18
#--------------------------



def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123
    var_5 = ''
    var_6 = '^\\d+$'
    var_7 = module_0.rex(var_6)
    var_8 = '123'
    var_9 = 'abc'
    var_10 = '^[A-Z]+$'
    var_11 = module_0.rex(var_10)
    var_12 = 'HELLO'
    var_13 = 'hello'
    var_14 = '^(\\w+)\\s(\\w+)$'
    var_15 = module_0.rex(var_14)
    var_16 = 'John Doe'
    var_17 = 'John'
    var_18 = '^a{2,3}$'
    var_19 = module_0.rex(var_18)
    var_20 = 'aa'
    var_21 = 'aaa'
    var_22 = 'a'
    var_23 = '^[aeiou]+$'
    var_24 = module_0.rex(var_23)
    var_25 = 'aeiou'
    var_26 = 'bcdfg'
    var_27 = '^start.*end$'
    var_28 = module_0.rex(var_27)
    var_29 = 'start middle end'
    var_30 = 'start middle'
    var_31 = '^\\d+(?= dollars)'
    var_32 = module_0.rex(var_31)
    var_33 = '100 dollars'
    var_34 = '100 euros'
    var_35 = 'All test cases passed!'
    var_36 = print(var_35)



# Parsed testcases at query #19
#--------------------------



def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'apple'
    var_4 = 'banana'
    var_5 = 123
    var_6 = 'a'
    var_7 = 'b'
    var_8 = [var_6, var_7]
    var_9 = 'All test cases pass'
    var_10 = print(var_9)



# Parsed testcases at query #20
#--------------------------



def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123
    var_5 = 'a'
    var_6 = 'b'
    var_7 = [var_5, var_6]
    var_8 = ''
    var_9 = '^\\d+$'
    var_10 = module_0.rex(var_9)
    var_11 = '123'
    var_12 = '12a'
    var_13 = '^[A-Z]+$'
    var_14 = module_0.rex(var_13)
    var_15 = 'HELLO'
    var_16 = 'hello'
    var_17 = 'All test cases passed!'
    var_18 = print(var_17)



# Parsed testcases at query #21
#--------------------------



def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123
    var_5 = ''
    var_6 = '^\\d+$'
    var_7 = module_0.rex(var_6)
    var_8 = '123'
    var_9 = 'abc'
    var_10 = '^[A-Z]+$'
    var_11 = module_0.rex(var_10)
    var_12 = 'HELLO'
    var_13 = 'hello'
    var_14 = 'All test cases passed!'
    var_15 = print(var_14)



# Parsed testcases at query #22
#--------------------------



def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'ab'
    var_5 = 'ba'
    var_6 = 1
    var_7 = None
    var_8 = []
    var_9 = {}
    var_10 = ()
    var_11 = set()



# Parsed testcases at query #23
#--------------------------



def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'a'
    var_3 = module_0.rex(var_0)
    var_4 = 'ba'
    var_5 = module_0.rex(var_0)
    var_6 = 1



# Parsed testcases at query #24
#--------------------------



def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'a'
    var_3 = module_0.rex(var_0)
    var_4 = 'ba'
    var_5 = module_0.rex(var_0)
    var_6 = 1
    var_7 = module_0.rex(var_0)
    var_8 = []
    var_9 = module_0.rex(var_0)
    var_10 = {}
    var_11 = module_0.rex(var_0)
    var_12 = None
    var_13 = module_0.rex(var_0)
    var_14 = True
    var_15 = module_0.rex(var_0)
    var_16 = False
    var_17 = module_0.rex(var_0)
    var_18 = module_0.rex(var_0)
    var_19 = module_0.rex(var_0)
    var_20 = module_0.rex(var_0)
    var_21 = module_0.rex(var_0)
    var_22 = module_0.rex(var_0)
    var_23 = module_0.rex(var_0)
    var_24 = module_0.rex(var_0)
    var_25 = module_0.rex(var_0)
    var_26 = module_0.rex(var_0)
    var_27 = module_0.rex(var_0)
    var_28 = module_0.rex(var_0)
    var_29 = module_0.rex(var_0)
    var_30 = module_0.rex(var_0)
    var_31 = module_0.rex(var_0)
    var_32 = module_0.rex(var_0)
    var_33 = module_0.rex(var_0)
    var_34 = module_0.rex(var_0)
    var_35 = module_0.rex(var_0)
    var_36 = module_0.rex(var_0)
    var_37 = module_0.rex(var_0)
    var_38 = module_0.rex(var_0)
    var_39 = module_0.rex(var_0)
    var_40 = module_0.rex(var_0)
    var_41 = module_0.rex(var_0)
    var_42 = module_0.rex(var_0)
    var_43 = module_0.rex(var_0)
    var_44 = module_0.rex(var_0)
    var_45 = module_0.rex(var_0)
    var_46 = module_0.rex(var_0)
    var_47 = module_0.rex(var_0)
    var_48 = module_0.rex(var_0)
    var_49 = module_0.rex(var_0)
    var_50 = module_0.rex(var_0)
    var_51 = module_0.rex(var_0)
    var_52 = module_0.rex(var_0)
    var_53 = module_0.rex(var_0)
    var_54 = module_0.rex(var_0)
    var_55 = module_0.rex(var_0)
    var_56 = module_0.rex(var_0)
    var_57 = module_0.rex(var_0)
    var_58 = module_0.rex(var_0)
    var_59 = module_0.rex(var_0)
    var_60 = module_0.rex(var_0)
    var_61 = module_0.rex(var_0)
    var_62 = module_0.rex(var_0)
    var_63 = module_0.rex(var_0)
    var_64 = module_0.rex(var_0)
    var_65 = module_0.rex(var_0)
    var_66 = module_0.rex(var_0)
    var_67 = module_0.rex(var_0)
    var_68 = module_0.rex(var_0)
    var_69 = module_0.rex(var_0)
    var_70 = module_0.rex(var_0)
    var_71 = module_0.rex(var_0)
    var_72 = module_0.rex(var_0)
    var_73 = module_0.rex(var_0)
    var_74 = module_0.rex(var_0)
    var_75 = module_0.rex(var_0)
    var_76 = module_0.rex(var_0)
    var_77 = module_0.rex(var_0)
    var_78 = module_0.rex(var_0)
    var_79 = module_0.rex(var_0)
    var_80 = module_0.rex(var_0)
    var_81 = module_0.rex(var_0)
    var_82 = module_0.rex(var_0)
    var_83 = module_0.rex(var_0)
    var_84 = module_0.rex(var_0)
    var_85 = module_0.rex(var_0)
    var_86 = module_0.rex(var_0)
    var_87 = module_0.rex(var_0)
    var_88 = module_0.rex(var_0)
    var_89 = module_0.rex(var_0)
    var_90 = module_0.rex(var_0)
    var_91 = module_0.rex(var_0)
    var_92 = module_0.rex(var_0)
    var_93 = module_0.rex(var_0)
    var_94 = module_0.rex(var_0)
    var_95 = module_0.rex(var_0)
    var_96 = module_0.rex(var_0)
    var_97 = module_0.rex(var_0)
    var_98 = module_0.rex(var_0)
    var_99 = module_0.rex(var_0)
    var_100 = module_0.rex(var_0)
    var_101 = module_0.rex(var_0)
    var_102 = module_0.rex(var_0)
    var_103 = module_0.rex(var_0)
    var_104 = module_0.rex(var_0)
    var_105 = module_0.rex(var_0)
    var_106 = module_0.rex(var_0)
    var_107 = module_0.rex(var_0)
    var_108 = module_0.rex(var_0)
    var_109 = module_0.rex(var_0)
    var_110 = module_0.rex(var_0)
    var_111 = module_0.rex(var_0)
    var_112 = module_0.rex(var_0)
    var_113 = module_0.rex(var_0)
    var_114 = module_0.rex(var_0)
    var_115 = module_0.rex(var_0)
    var_116 = module_0.rex(var_0)
    var_117 = module_0.rex(var_0)
    var_118 = module_0.rex(var_0)
    var_119 = module_0.rex(var_0)
    var_120 = module_0.rex(var_0)
    var_121 = module_0.rex(var_0)
    var_122 = module_0.rex(var_0)
    var_123 = module_0.rex(var_0)



# Parsed testcases at query #25
#--------------------------



def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123
    var_5 = ''
    var_6 = '^\\d+$'
    var_7 = module_0.rex(var_6)
    var_8 = '123'
    var_9 = '12a'
    var_10 = '^[A-Z]+$'
    var_11 = module_0.rex(var_10)
    var_12 = 'HELLO'
    var_13 = 'hello'
    var_14 = 'All test cases passed!'
    var_15 = print(var_14)



# Parsed testcases at query #26
#--------------------------



def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = var_1.__code__.co_argcount
    assert var_2 == 1
    var_3 = module_0.rex(var_0)
    var_4 = 'a'
    var_5 = module_0.rex(var_0)
    var_6 = 'ba'
    var_7 = module_0.rex(var_0)
    var_8 = 'ab'
    var_9 = module_0.rex(var_0)
    var_10 = 5
    var_11 = module_0.rex(var_0)
    var_12 = b'a'
    var_13 = b'^a'
    var_14 = module_0.rex(var_13)
    var_15 = module_0.rex(var_13)
    var_16 = b'ba'
    var_17 = module_0.rex(var_13)
    var_18 = b'ab'
    var_19 = module_0.rex(var_13)
    var_20 = module_0.rex(var_13)



# Parsed testcases at query #27
#--------------------------



def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123
    var_5 = ''
    var_6 = '^\\d+$'
    var_7 = module_0.rex(var_6)
    var_8 = '123'
    var_9 = 'abc'
    var_10 = '^[A-Z]+$'
    var_11 = module_0.rex(var_10)
    var_12 = 'HELLO'
    var_13 = 'hello'
    var_14 = 'All test cases passed!'
    var_15 = print(var_14)



# Parsed testcases at query #28
#--------------------------



def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = '^b.*'
    var_5 = module_0.rex(var_4)
    var_6 = '^[0-9]+$'
    var_7 = module_0.rex(var_6)
    var_8 = '123'
    var_9 = 'abc'
    var_10 = '^[a-z]+$'
    var_11 = module_0.rex(var_10)
    var_12 = '^[a-z]+[0-9]+$'
    var_13 = module_0.rex(var_12)
    var_14 = 'abc123'
    var_15 = '123abc'
    var_16 = module_0.rex(var_12)
    var_17 = '^a.*b$'
    var_18 = module_0.rex(var_17)
    var_19 = module_0.rex(var_17)
    var_20 = '^a{2,3}$'
    var_21 = module_0.rex(var_20)
    var_22 = 'aa'
    var_23 = 'aaa'
    var_24 = 'aaaa'
    var_25 = module_0.rex(var_20)
    var_26 = 'a'
    var_27 = 'All test cases pass'
    var_28 = print(var_27)



# Parsed testcases at query #29
#--------------------------



def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123
    var_5 = ''
    var_6 = '^\\d+$'
    var_7 = module_0.rex(var_6)
    var_8 = '123'
    var_9 = 'abc'
    var_10 = '^[A-Z]+$'
    var_11 = module_0.rex(var_10)
    var_12 = 'HELLO'
    var_13 = 'hello'
    var_14 = 'All test cases passed!'
    var_15 = print(var_14)



# Parsed testcases at query #30
#--------------------------


import re as module_2


def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'ab'
    var_5 = 'ba'
    var_6 = ''
    var_7 = 1
    var_8 = None
    var_9 = []
    var_10 = {}
    var_11 = ()
    var_12 = True
    var_13 = False
    var_14 = module_1.object()
    var_15 = module_2.compile(var_0)
    var_16 = module_2.compile(var_0)
    var_17 = module_2.match(var_2)
    var_18 = module_2.compile(var_0)
    var_19 = module_2.match(var_3)
    var_20 = module_2.compile(var_0)
    var_21 = module_2.match(var_4)
    var_22 = module_2.compile(var_0)
    var_23 = module_2.match(var_5)
    var_24 = module_2.compile(var_0)
    var_25 = module_2.match(var_6)
    var_26 = module_2.compile(var_0)
    var_27 = module_2.match(var_12)
    var_28 = module_2.compile(var_0)
    var_29 = module_2.match(var_8)
    var_30 = module_2.compile(var_0)
    var_31 = []
    var_32 = module_2.match(var_31)
    var_33 = module_2.compile(var_0)
    var_34 = {}
    var_35 = module_2.match(var_34)
    var_36 = module_2.compile(var_0)
    var_37 = ()
    var_38 = module_2.match(var_37)
    var_39 = module_2.compile(var_0)
    var_40 = module_2.match(var_12)
    var_41 = module_2.compile(var_0)
    var_42 = True
    var_43 = module_2.match(var_42)
    var_44 = module_2.compile(var_0)
    var_45 = module_2.match(var_13)
    var_46 = module_2.compile(var_0)
    var_47 = module_1.object()
    var_48 = module_2.match(var_47)
    var_49 = module_2.compile(var_0)
    var_50 = module_2.compile(var_0)
    var_51 = module_2.match(var_50)
    var_52 = module_2.compile(var_0)
    var_53 = module_2.compile(var_0)
    var_54 = module_2.match(var_2)
    var_55 = module_2.match(var_54)
    var_56 = module_2.compile(var_0)
    var_57 = module_2.compile(var_0)
    var_58 = module_2.match(var_3)
    var_59 = module_2.match(var_58)
    var_60 = module_2.compile(var_0)
    var_61 = module_2.compile(var_0)
    var_62 = module_2.match(var_4)
    var_63 = module_2.match(var_62)
    var_64 = module_2.compile(var_0)
    var_65 = module_2.compile(var_0)
    var_66 = module_2.match(var_5)
    var_67 = module_2.match(var_66)
    var_68 = module_2.compile(var_0)
    var_69 = module_2.compile(var_0)
    var_70 = module_2.match(var_6)
    var_71 = module_2.match(var_70)
    var_72 = module_2.compile(var_0)
    var_73 = module_2.compile(var_0)
    var_74 = module_2.match(var_42)
    var_75 = module_2.match(var_74)
    var_76 = module_2.compile(var_0)
    var_77 = module_2.compile(var_0)
    var_78 = module_2.match(var_8)
    var_79 = module_2.match(var_78)
    var_80 = module_2.compile(var_0)
    var_81 = module_2.compile(var_0)
    var_82 = []
    var_83 = module_2.match(var_82)
    var_84 = module_2.match(var_83)
    var_85 = module_2.compile(var_0)
    var_86 = module_2.compile(var_0)
    var_87 = {}
    var_88 = module_2.match(var_87)
    var_89 = module_2.match(var_88)
    var_90 = module_2.compile(var_0)
    var_91 = module_2.compile(var_0)
    var_92 = ()
    var_93 = module_2.match(var_92)
    var_94 = module_2.match(var_93)
    var_95 = module_2.compile(var_0)
    var_96 = module_2.compile(var_0)
    var_97 = module_2.match(var_42)
    var_98 = module_2.match(var_97)
    var_99 = module_2.compile(var_0)
    var_100 = module_2.compile(var_0)
    var_101 = True
    var_102 = module_2.match(var_101)
    var_103 = module_2.match(var_102)
    var_104 = module_2.compile(var_0)
    var_105 = module_2.compile(var_0)
    var_106 = module_2.match(var_13)
    var_107 = module_2.match(var_106)
    var_108 = module_2.compile(var_0)
    var_109 = module_2.compile(var_0)
    var_110 = module_1.object()
    var_111 = module_2.match(var_110)
    var_112 = module_2.match(var_111)
    var_113 = module_2.compile(var_0)
    var_114 = module_2.compile(var_0)
    var_115 = module_2.compile(var_0)
    var_116 = module_2.match(var_115)
    var_117 = module_2.match(var_116)
    var_118 = module_2.compile(var_0)
    var_119 = module_2.compile(var_0)
    var_120 = module_2.compile(var_0)
    var_121 = module_2.match(var_2)
    var_122 = module_2.match(var_121)
    var_123 = module_2.match(var_122)
    var_124 = module_2.compile(var_0)
    var_125 = module_2.compile(var_0)
    var_126 = module_2.compile(var_0)
    var_127 = module_2.match(var_3)
    var_128 = module_2.match(var_127)
    var_129 = module_2.match(var_128)
    var_130 = module_2.compile(var_0)
    var_131 = module_2.compile(var_0)
    var_132 = module_2.compile(var_0)
    var_133 = module_2.match(var_4)
    var_134 = module_2.match(var_133)
    var_135 = module_2.match(var_134)
    var_136 = module_2.compile(var_0)
    var_137 = module_2.compile(var_0)
    var_138 = module_2.compile(var_0)
    var_139 = module_2.match(var_5)
    var_140 = module_2.match(var_139)
    var_141 = module_2.match(var_140)
    var_142 = module_2.compile(var_0)
    var_143 = module_2.compile(var_0)
    var_144 = module_2.compile(var_0)
    var_145 = module_2.match(var_6)
    var_146 = module_2.match(var_145)
    var_147 = module_2.match(var_146)
    var_148 = module_2.compile(var_0)
    var_149 = module_2.compile(var_0)
    var_150 = module_2.compile(var_0)
    var_151 = module_2.match(var_101)
    var_152 = module_2.match(var_151)
    var_153 = module_2.match(var_152)
    var_154 = module_2.compile(var_0)
    var_155 = module_2.compile(var_0)
    var_156 = module_2.compile(var_0)
    var_157 = module_2.match(var_8)
    var_158 = module_2.match(var_157)
    var_159 = module_2.match(var_158)
    var_160 = module_2.compile(var_0)
    var_161 = module_2.compile(var_0)
    var_162 = module_2.compile(var_0)
    var_163 = []
    var_164 = module_2.match(var_163)
    var_165 = module_2.match(var_164)
    var_166 = module_2.match(var_165)
    var_167 = module_2.compile(var_0)
    var_168 = module_2.compile(var_0)
    var_169 = module_2.compile(var_0)
    var_170 = {}
    var_171 = module_2.match(var_170)
    var_172 = module_2.match(var_171)
    var_173 = module_2.match(var_172)
    var_174 = module_2.compile(var_0)
    var_175 = module_2.compile(var_0)
    var_176 = module_2.compile(var_0)
    var_177 = ()
    var_178 = module_2.match(var_177)
    var_179 = module_2.match(var_178)
    var_180 = module_2.match(var_179)
    var_181 = module_2.compile(var_0)
    var_182 = module_2.compile(var_0)
    var_183 = module_2.compile(var_0)
    var_184 = module_2.match(var_101)
    var_185 = module_2.match(var_184)
    var_186 = module_2.match(var_185)
    var_187 = module_2.compile(var_0)
    var_188 = module_2.compile(var_0)
    var_189 = module_2.compile(var_0)
    var_190 = True
    var_191 = module_2.match(var_190)
    var_192 = module_2.match(var_191)
    var_193 = module_2.match(var_192)
    var_194 = module_2.compile(var_0)
    var_195 = module_2.compile(var_0)
    var_196 = module_2.compile(var_0)
    var_197 = module_2.match(var_13)
    var_198 = module_2.match(var_197)
    var_199 = module_2.match(var_198)
    var_200 = module_2.compile(var_0)
    var_201 = module_2.compile(var_0)
    var_202 = module_2.compile(var_0)
    var_203 = module_1.object()
    var_204 = module_2.match(var_203)
    var_205 = module_2.match(var_204)
    var_206 = module_2.match(var_205)
    var_207 = module_2.compile(var_0)
    var_208 = module_2.compile(var_0)
    var_209 = module_2.compile(var_0)
    var_210 = module_2.compile(var_0)
    var_211 = module_2.match(var_210)
    var_212 = module_2.match(var_211)
    var_213 = module_2.match(var_212)
    var_214 = module_2.compile(var_0)
    var_215 = module_2.compile(var_0)
    var_216 = module_2.compile(var_0)
    var_217 = module_2.compile(var_0)
    var_218 = module_2.match(var_2)
    var_219 = module_2.match(var_218)
    var_220 = module_2.match(var_219)
    var_221 = module_2.match(var_220)
    var_222 = module_2.compile(var_0)
    var_223 = module_2.compile(var_0)
    var_224 = module_2.compile(var_0)
    var_225 = module_2.compile(var_0)
    var_226 = module_2.match(var_3)
    var_227 = module_2.match(var_226)
    var_228 = module_2.match(var_227)
    var_229 = module_2.match(var_228)
    var_230 = module_2.compile(var_0)
    var_231 = module_2.compile(var_0)
    var_232 = module_2.compile(var_0)
    var_233 = module_2.compile(var_0)
    var_234 = module_2.match(var_4)
    var_235 = module_2.match(var_234)
    var_236 = module_2.match(var_235)
    var_237 = module_2.match(var_236)
    var_238 = module_2.compile(var_0)
    var_239 = module_2.compile(var_0)
    var_240 = module_2.compile(var_0)
    var_241 = module_2.compile(var_0)
    var_242 = module_2.match(var_5)
    var_243 = module_2.match(var_242)
    var_244 = module_2.match(var_243)
    var_245 = module_2.match(var_244)
    var_246 = module_2.compile(var_0)
    var_247 = module_2.compile(var_0)
    var_248 = module_2.compile(var_0)
    var_249 = module_2.compile(var_0)
    var_250 = module_2.match(var_6)
    var_251 = module_2.match(var_250)
    var_252 = module_2.match(var_251)
    var_253 = module_2.match(var_252)
    var_254 = module_2.compile(var_0)
    var_255 = module_2.compile(var_0)
    var_256 = module_2.compile(var_0)
    var_257 = module_2.compile(var_0)
    var_258 = module_2.match(var_190)
    var_259 = module_2.match(var_258)
    var_260 = module_2.match(var_259)
    var_261 = module_2.match(var_260)
    var_262 = module_2.compile(var_0)
    var_263 = module_2.compile(var_0)
    var_264 = module_2.compile(var_0)
    var_265 = module_2.compile(var_0)
    var_266 = module_2.match(var_8)
    var_267 = module_2.match(var_266)
    var_268 = module_2.match(var_267)
    var_269 = module_2.match(var_268)
    var_270 = module_2.compile(var_0)
    var_271 = module_2.compile(var_0)
    var_272 = module_2.compile(var_0)
    var_273 = module_2.compile(var_0)
    var_274 = []
    var_275 = module_2.match(var_274)
    var_276 = module_2.match(var_275)
    var_277 = module_2.match(var_276)
    var_278 = module_2.match(var_277)
    var_279 = module_2.compile(var_0)
    var_280 = module_2.compile(var_0)
    var_281 = module_2.compile(var_0)
    var_282 = module_2.compile(var_0)
    var_283 = {}
    var_284 = module_2.match(var_283)
    var_285 = module_2.match(var_284)
    var_286 = module_2.match(var_285)
    var_287 = module_2.match(var_286)
    var_288 = module_2.compile(var_0)
    var_289 = module_2.compile(var_0)
    var_290 = module_2.compile(var_0)
    var_291 = module_2.compile(var_0)
    var_292 = ()
    var_293 = module_2.match(var_292)
    var_294 = module_2.match(var_293)
    var_295 = module_2.match(var_294)
    var_296 = module_2.match(var_295)
    var_297 = module_2.compile(var_0)
    var_298 = module_2.compile(var_0)
    var_299 = module_2.compile(var_0)
    var_300 = module_2.compile(var_0)
    var_301 = module_2.match(var_190)
    var_302 = module_2.match(var_301)
    var_303 = module_2.match(var_302)
    var_304 = module_2.match(var_303)
    var_305 = module_2.compile(var_0)
    var_306 = module_2.compile(var_0)
    var_307 = module_2.compile(var_0)
    var_308 = module_2.compile(var_0)
    var_309 = True
    var_310 = module_2.match(var_309)
    var_311 = module_2.match(var_310)
    var_312 = module_2.match(var_311)
    var_313 = module_2.match(var_312)
    var_314 = module_2.compile(var_0)
    var_315 = module_2.compile(var_0)
    var_316 = module_2.compile(var_0)
    var_317 = module_2.compile(var_0)
    var_318 = module_2.match(var_13)
    var_319 = module_2.match(var_318)
    var_320 = module_2.match(var_319)
    var_321 = module_2.match(var_320)
    var_322 = module_2.compile(var_0)
    var_323 = module_2.compile(var_0)
    var_324 = module_2.compile(var_0)
    var_325 = module_2.compile(var_0)
    var_326 = module_1.object()
    var_327 = module_2.match(var_326)
    var_328 = module_2.match(var_327)
    var_329 = module_2.match(var_328)
    var_330 = module_2.match(var_329)
    var_331 = module_2.compile(var_0)
    var_332 = module_2.compile(var_0)
    var_333 = module_2.compile(var_0)
    var_334 = module_2.compile(var_0)
    var_335 = module_2.compile(var_0)
    var_336 = module_2.match(var_335)
    var_337 = module_2.match(var_336)
    var_338 = module_2.match(var_337)
    var_339 = module_2.match(var_338)
    var_340 = module_2.compile(var_0)
    var_341 = module_2.compile(var_0)
    var_342 = module_2.compile(var_0)
    var_343 = module_2.compile(var_0)
    var_344 = module_2.compile(var_0)
    var_345 = module_2.match(var_2)
    var_346 = module_2.match(var_345)
    var_347 = module_2.match(var_346)
    var_348 = module_2.match(var_347)
    var_349 = module_2.match(var_348)
    var_350 = module_2.compile(var_0)
    var_351 = module_2.compile(var_0)
    var_352 = module_2.compile(var_0)
    var_353 = module_2.compile(var_0)
    var_354 = module_2.compile(var_0)
    var_355 = module_2.match(var_3)
    var_356 = module_2.match(var_355)
    var_357 = module_2.match(var_356)
    var_358 = module_2.match(var_357)
    var_359 = module_2.match(var_358)



