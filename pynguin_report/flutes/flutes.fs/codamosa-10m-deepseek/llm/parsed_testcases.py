####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = '.'
    var_1 = '.'
    var_2 = 'All tests passed!'
    var_3 = print(var_2)



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'test_cache.pkl'
    var_1 = 5
    var_2 = 'Cache file should exist'
    var_3 = 20
    var_4 = 'nonexistent.pkl'
    var_5 = 'All tests passed!'
    var_6 = print(var_5)



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'test_cache.pkl'
    var_1 = 5
    var_2 = 7
    var_3 = 'All cache tests passed.'
    var_4 = print(var_3)



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'dir1'
    var_1 = 'dir2'
    var_2 = 'Hello'
    var_3 = 'World'
    var_4 = 'Foo'
    var_5 = 'Bar'
    var_6 = 'file1.txt'
    var_7 = 'file2.txt'
    var_8 = 'file3.txt'
    var_9 = 'empty_dir'
    var_10 = 'All tests passed!'
    var_11 = print(var_10)



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'src'
    var_1 = 'dst'
    var_2 = 'file.txt'
    var_3 = 'Hello, world!'
    var_4 = 'subdir'
    var_5 = 'file2.txt'
    var_6 = True
    var_7 = False
    var_8 = 'Goodbye, world!'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'All tests passed.'
    var_1 = print(var_0)



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = '.'
    var_1 = '.'
    var_2 = 'All tests passed.'
    var_3 = print(var_2)



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'test_cache.pkl'
    var_1 = 5
    var_2 = 20
    var_3 = 'nonexistent.pkl'
    var_4 = 'All cache tests passed!'
    var_5 = print(var_4)



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'src'
    var_1 = 'Hello, world!'
    var_2 = 'subdir'
    var_3 = 'Another file'
    var_4 = 'dst'

def test_case_0():
    var_0 = 'src'
    var_1 = 'Source file'
    var_2 = 'dst'
    var_3 = 'Destination file'
    var_4 = True
    var_5 = 'Source file'

def test_case_0():
    var_0 = 'src'
    var_1 = 'Source file'
    var_2 = 'dst'
    var_3 = 'Destination file'
    var_4 = False
    var_5 = 'Destination file'
    var_6 = 2

def test_case_0():
    var_0 = 'src'
    var_1 = 'Source file'
    var_2 = 'dst'
    var_3 = 'Destination file'
    var_4 = False
    var_5 = 'Destination file'
    var_6 = 2



# Parsed testcases at query #10
#--------------------------


import flutes.fs as module_0

def test_case_0():
    var_0 = 1024
    var_1 = var_0 * var_0
    var_2 = module_0.readable_size(var_1)
    assert var_2 == '1.00M'
    var_3 = var_0 * var_0
    var_4 = var_3 * var_0
    var_5 = module_0.readable_size(var_4)
    assert var_5 == '1.00G'
    var_6 = var_0 * var_0
    var_7 = var_6 * var_0
    var_8 = var_7 * var_0
    var_9 = module_0.readable_size(var_8)
    assert var_9 == '1.00T'
    var_10 = var_0 * var_0
    var_11 = var_10 * var_0
    var_12 = var_11 * var_0
    var_13 = var_12 * var_0
    var_14 = module_0.readable_size(var_13)
    assert var_14 == '1.00P'
    var_15 = var_0 * var_0
    var_16 = var_15 * var_0
    var_17 = var_16 * var_0
    var_18 = var_17 * var_0
    var_19 = var_18 * var_0
    var_20 = module_0.readable_size(var_19)
    assert var_20 == '1024.00P'
    var_21 = var_0 * var_0
    var_22 = var_21 * var_0
    var_23 = var_22 * var_0
    var_24 = var_23 * var_0
    var_25 = var_24 * var_0
    var_26 = var_25 * var_0
    var_27 = module_0.readable_size(var_26)
    assert var_27 == '1048576.00P'
    var_28 = var_0 * var_0
    var_29 = var_28 * var_0
    var_30 = var_29 * var_0
    var_31 = var_30 * var_0
    var_32 = var_31 * var_0
    var_33 = var_32 * var_0
    var_34 = var_33 * var_0
    var_35 = module_0.readable_size(var_34)
    assert var_35 == '1073741824.00P'
    var_36 = var_0 * var_0
    var_37 = var_36 * var_0
    var_38 = var_37 * var_0
    var_39 = var_38 * var_0
    var_40 = var_39 * var_0
    var_41 = var_40 * var_0
    var_42 = var_41 * var_0
    var_43 = var_42 * var_0
    var_44 = module_0.readable_size(var_43)
    assert var_44 == '1099511627776.00P'
    var_45 = var_0 * var_0
    var_46 = var_45 * var_0
    var_47 = var_46 * var_0
    var_48 = var_47 * var_0
    var_49 = var_48 * var_0
    var_50 = var_49 * var_0
    var_51 = var_50 * var_0
    var_52 = var_51 * var_0
    var_53 = var_52 * var_0
    var_54 = module_0.readable_size(var_53)
    assert var_54 == '1125899906842624.00P'
    var_55 = var_0 * var_0
    var_56 = var_55 * var_0
    var_57 = var_56 * var_0
    var_58 = var_57 * var_0
    var_59 = var_58 * var_0
    var_60 = var_59 * var_0
    var_61 = var_60 * var_0
    var_62 = var_61 * var_0
    var_63 = var_62 * var_0
    var_64 = var_63 * var_0
    var_65 = module_0.readable_size(var_64)
    assert var_65 == '1152921504606846976.00P'
    var_66 = var_0 * var_0
    var_67 = var_66 * var_0
    var_68 = var_67 * var_0
    var_69 = var_68 * var_0
    var_70 = var_69 * var_0
    var_71 = var_70 * var_0
    var_72 = var_71 * var_0
    var_73 = var_72 * var_0
    var_74 = var_73 * var_0
    var_75 = var_74 * var_0
    var_76 = var_75 * var_0
    var_77 = module_0.readable_size(var_76)
    assert var_77 == '1180591620717411303424.00P'
    var_78 = var_0 * var_0
    var_79 = var_78 * var_0
    var_80 = var_79 * var_0
    var_81 = var_80 * var_0
    var_82 = var_81 * var_0
    var_83 = var_82 * var_0
    var_84 = var_83 * var_0
    var_85 = var_84 * var_0
    var_86 = var_85 * var_0
    var_87 = var_86 * var_0
    var_88 = var_87 * var_0
    var_89 = var_88 * var_0
    var_90 = module_0.readable_size(var_89)
    assert var_90 == '1208925819614629174706176.00P'
    var_91 = var_0 * var_0
    var_92 = var_91 * var_0
    var_93 = var_92 * var_0
    var_94 = var_93 * var_0
    var_95 = var_94 * var_0
    var_96 = var_95 * var_0
    var_97 = var_96 * var_0
    var_98 = var_97 * var_0
    var_99 = var_98 * var_0
    var_100 = var_99 * var_0
    var_101 = var_100 * var_0
    var_102 = var_101 * var_0
    var_103 = var_102 * var_0
    var_104 = module_0.readable_size(var_103)
    assert var_104 == '1237940039285380274899124224.00P'
    var_105 = var_0 * var_0
    var_106 = var_105 * var_0
    var_107 = var_106 * var_0
    var_108 = var_107 * var_0
    var_109 = var_108 * var_0
    var_110 = var_109 * var_0
    var_111 = var_110 * var_0
    var_112 = var_111 * var_0
    var_113 = var_112 * var_0
    var_114 = var_113 * var_0
    var_115 = var_114 * var_0
    var_116 = var_115 * var_0
    var_117 = var_116 * var_0
    var_118 = var_117 * var_0
    var_119 = module_0.readable_size(var_118)
    assert var_119 == '1267650600228229401496703205376.00P'
    var_120 = var_0 * var_0
    var_121 = var_120 * var_0
    var_122 = var_121 * var_0
    var_123 = var_122 * var_0
    var_124 = var_123 * var_0
    var_125 = var_124 * var_0
    var_126 = var_125 * var_0
    var_127 = var_126 * var_0
    var_128 = var_127 * var_0
    var_129 = var_128 * var_0
    var_130 = var_129 * var_0
    var_131 = var_130 * var_0
    var_132 = var_131 * var_0
    var_133 = var_132 * var_0
    var_134 = var_133 * var_0
    var_135 = module_0.readable_size(var_134)
    assert var_135 == '1298074214633706907132624082305024.00P'
    var_136 = var_0 * var_0
    var_137 = var_136 * var_0
    var_138 = var_137 * var_0
    var_139 = var_138 * var_0
    var_140 = var_139 * var_0
    var_141 = var_140 * var_0
    var_142 = var_141 * var_0
    var_143 = var_142 * var_0
    var_144 = var_143 * var_0
    var_145 = var_144 * var_0
    var_146 = var_145 * var_0
    var_147 = var_146 * var_0
    var_148 = var_147 * var_0
    var_149 = var_148 * var_0
    var_150 = var_149 * var_0
    var_151 = var_150 * var_0
    var_152 = module_0.readable_size(var_151)
    assert var_152 == '1329227995784915872903807060280344576.00P'
    var_153 = var_0 * var_0
    var_154 = var_153 * var_0
    var_155 = var_154 * var_0
    var_156 = var_155 * var_0
    var_157 = var_156 * var_0
    var_158 = var_157 * var_0
    var_159 = var_158 * var_0
    var_160 = var_159 * var_0
    var_161 = var_160 * var_0
    var_162 = var_161 * var_0
    var_163 = var_162 * var_0
    var_164 = var_163 * var_0
    var_165 = var_164 * var_0
    var_166 = var_165 * var_0
    var_167 = var_166 * var_0
    var_168 = var_167 * var_0
    var_169 = var_168 * var_0
    var_170 = module_0.readable_size(var_169)
    assert var_170 == '1361129467683753853853498429727072845824.00P'
    var_171 = var_0 * var_0
    var_172 = var_171 * var_0
    var_173 = var_172 * var_0
    var_174 = var_173 * var_0
    var_175 = var_174 * var_0
    var_176 = var_175 * var_0
    var_177 = var_176 * var_0
    var_178 = var_177 * var_0
    var_179 = var_178 * var_0
    var_180 = var_179 * var_0
    var_181 = var_180 * var_0
    var_182 = var_181 * var_0
    var_183 = var_182 * var_0
    var_184 = var_183 * var_0
    var_185 = var_184 * var_0
    var_186 = var_185 * var_0
    var_187 = var_186 * var_0
    var_188 = var_187 * var_0
    var_189 = module_0.readable_size(var_188)
    assert var_189 == '1393796574908163946345982392040522594123776.00P'
    var_190 = var_0 * var_0
    var_191 = var_190 * var_0
    var_192 = var_191 * var_0
    var_193 = var_192 * var_0
    var_194 = var_193 * var_0
    var_195 = var_194 * var_0
    var_196 = var_195 * var_0
    var_197 = var_196 * var_0
    var_198 = var_197 * var_0
    var_199 = var_198 * var_0
    var_200 = var_199 * var_0
    var_201 = var_200 * var_0
    var_202 = var_201 * var_0
    var_203 = var_202 * var_0
    var_204 = var_203 * var_0
    var_205 = var_204 * var_0
    var_206 = var_205 * var_0
    var_207 = var_206 * var_0
    var_208 = var_207 * var_0
    var_209 = module_0.readable_size(var_208)
    assert var_209 == '1427247692705959881058285969449495136382746624.00P'
    var_210 = var_0 * var_0
    var_211 = var_210 * var_0
    var_212 = var_211 * var_0
    var_213 = var_212 * var_0
    var_214 = var_213 * var_0
    var_215 = var_214 * var_0
    var_216 = var_215 * var_0
    var_217 = var_216 * var_0
    var_218 = var_217 * var_0
    var_219 = var_218 * var_0
    var_220 = var_219 * var_0
    var_221 = var_220 * var_0
    var_222 = var_221 * var_0
    var_223 = var_222 * var_0
    var_224 = var_223 * var_0
    var_225 = var_224 * var_0
    var_226 = var_225 * var_0
    var_227 = var_226 * var_0
    var_228 = var_227 * var_0
    var_229 = var_228 * var_0
    var_230 = module_0.readable_size(var_229)
    assert var_230 == '1461501637330902918203684832716283019655932542976.00P'
    var_231 = var_0 * var_0
    var_232 = var_231 * var_0
    var_233 = var_232 * var_0
    var_234 = var_233 * var_0
    var_235 = var_234 * var_0
    var_236 = var_235 * var_0
    var_237 = var_236 * var_0
    var_238 = var_237 * var_0
    var_239 = var_238 * var_0
    var_240 = var_239 * var_0
    var_241 = var_240 * var_0
    var_242 = var_241 * var_0
    var_243 = var_242 * var_0
    var_244 = var_243 * var_0
    var_245 = var_244 * var_0
    var_246 = var_245 * var_0
    var_247 = var_246 * var_0
    var_248 = var_247 * var_0
    var_249 = var_248 * var_0
    var_250 = var_249 * var_0
    var_251 = var_250 * var_0
    var_252 = module_0.readable_size(var_251)
    assert var_252 == '1496577676626844588240573268701473812127674924007424.00P'
    var_253 = var_0 * var_0
    var_254 = var_253 * var_0
    var_255 = var_254 * var_0
    var_256 = var_255 * var_0
    var_257 = var_256 * var_0
    var_258 = var_257 * var_0
    var_259 = var_258 * var_0
    var_260 = var_259 * var_0
    var_261 = var_260 * var_0
    var_262 = var_261 * var_0
    var_263 = var_262 * var_0
    var_264 = var_263 * var_0
    var_265 = var_264 * var_0
    var_266 = var_265 * var_0
    var_267 = var_266 * var_0
    var_268 = var_267 * var_0
    var_269 = var_268 * var_0
    var_270 = var_269 * var_0
    var_271 = var_270 * var_0
    var_272 = var_271 * var_0
    var_273 = var_272 * var_0
    var_274 = var_273 * var_0
    var_275 = module_0.readable_size(var_274)
    assert var_275 == '1532495540865888858358347027150309183618739122183602176.00P'
    var_276 = var_0 * var_0
    var_277 = var_276 * var_0
    var_278 = var_277 * var_0
    var_279 = var_278 * var_0
    var_280 = var_279 * var_0
    var_281 = var_280 * var_0
    var_282 = var_281 * var_0
    var_283 = var_282 * var_0
    var_284 = var_283 * var_0
    var_285 = var_284 * var_0
    var_286 = var_285 * var_0
    var_287 = var_286 * var_0
    var_288 = var_287 * var_0
    var_289 = var_288 * var_0
    var_290 = var_289 * var_0
    var_291 = var_290 * var_0
    var_292 = var_291 * var_0
    var_293 = var_292 * var_0
    var_294 = var_293 * var_0
    var_295 = var_294 * var_0
    var_296 = var_295 * var_0
    var_297 = var_296 * var_0
    var_298 = var_297 * var_0
    var_299 = module_0.readable_size(var_298)
    assert var_299 == '1569275433846670190958947355801916604025588861116008628224.00P'
    var_300 = var_0 * var_0
    var_301 = var_300 * var_0
    var_302 = var_301 * var_0
    var_303 = var_302 * var_0
    var_304 = var_303 * var_0
    var_305 = var_304 * var_0
    var_306 = var_305 * var_0
    var_307 = var_306 * var_0
    var_308 = var_307 * var_0
    var_309 = var_308 * var_0
    var_310 = var_309 * var_0
    var_311 = var_310 * var_0
    var_312 = var_311 * var_0
    var_313 = var_312 * var_0
    var_314 = var_313 * var_0
    var_315 = var_314 * var_0
    var_316 = var_315 * var_0
    var_317 = var_316 * var_0
    var_318 = var_317 * var_0
    var_319 = var_318 * var_0
    var_320 = var_319 * var_0
    var_321 = var_320 * var_0
    var_322 = var_321 * var_0
    var_323 = var_322 * var_0
    var_324 = module_0.readable_size(var_323)
    assert var_324 == '1606938044258990275541962092341162602522202993782792835301376.00P'
    var_325 = var_0 * var_0
    var_326 = var_325 * var_0
    var_327 = var_326 * var_0
    var_328 = var_327 * var_0
    var_329 = var_328 * var_0
    var_330 = var_329 * var_0
    var_331 = var_330 * var_0
    var_332 = var_331 * var_0
    var_333 = var_332 * var_0
    var_334 = var_333 * var_0
    var_335 = var_334 * var_0
    var_336 = var_335 * var_0
    var_337 = var_336 * var_0
    var_338 = var_337 * var_0
    var_339 = var_338 * var_0
    var_340 = var_339 * var_0
    var_341 = var_340 * var_0
    var_342 = var_341 * var_0
    var_343 = var_342 * var_0
    var_344 = var_343 * var_0
    var_345 = var_344 * var_0
    var_346 = var_345 * var_0
    var_347 = var_346 * var_0
    var_348 = var_347 * var_0
    var_349 = var_348 * var_0
    var_350 = module_0.readable_size(var_349)
    assert var_350 == '1645504557321206042154969182557350504982735865633579863348609024.00P'
    var_351 = var_0 * var_0
    var_352 = var_351 * var_0
    var_353 = var_352 * var_0
    var_354 = var_353 * var_0
    var_355 = var_354 * var_0
    var_356 = var_355 * var_0
    var_357 = var_356 * var_0
    var_358 = var_357 * var_0
    var_359 = var_358 * var_0
    var_360 = var_359 * var_0
    var_361 = var_360 * var_0
    var_362 = var_361 * var_0
    var_363 = var_362 * var_0
    var_364 = var_363 * var_0
    var_365 = var_364 * var_0
    var_366 = var_365 * var_0
    var_367 = var_366 * var_0
    var_368 = var_367 * var_0
    var_369 = var_368 * var_0
    var_370 = var_369 * var_0
    var_371 = var_370 * var_0
    var_372 = var_371 * var_0
    var_373 = var_372 * var_0
    var_374 = var_373 * var_0
    var_375 = var_374 * var_0
    var_376 = var_375 * var_0
    var_377 = module_0.readable_size(var_376)
    assert var_377 == '1684996666696914987166688442938726917102321526403515786552695619584.00P'



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'test_cache.pkl'
    var_1 = 5
    var_2 = 'Cache file should exist after first call'
    var_3 = 20
    var_4 = 10
    var_5 = 'nonexistent.pkl'
    var_6 = 'All tests passed!'
    var_7 = print(var_6)



# Parsed testcases at query #12
#--------------------------


import flutes.log as module_0

def test_case_0():
    var_0 = 'test_cache.pkl'
    var_1 = 5
    var_2 = 'Cache file should exist'
    var_3 = 'All cache tests passed!'
    var_4 = module_0.log(var_3)



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = '.'
    var_1 = '.'
    var_2 = 'All tests passed!'
    var_3 = print(var_2)



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'test_cache.pkl'
    var_1 = 5
    var_2 = 'Cache file was not created'
    var_3 = 20
    var_4 = 'nonexistent.pkl'
    var_5 = 'All tests passed!'
    var_6 = print(var_5)



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = '.'
    var_1 = '.'
    var_2 = 'All tests passed!'
    var_3 = print(var_2)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'subdir'
    var_1 = 'file1.txt'
    var_2 = 'file2.txt'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'cache.pkl'



# Parsed testcases at query #3
#--------------------------




# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'dir1'
    var_1 = 'dir2'
    var_2 = 'test'
    var_3 = 'test'
    var_4 = 'file1.txt'
    var_5 = 'file2.txt'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'src'
    var_1 = 'dst'
    var_2 = 'file1.txt'
    var_3 = 'file1'
    var_4 = 'file2.txt'
    var_5 = 'file2'
    var_6 = 'subdir'
    var_7 = 'file3.txt'
    var_8 = 'file3'
    var_9 = True
    var_10 = 'modified'
    var_11 = False



# Parsed testcases at query #6
#--------------------------


import flutes.fs as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'test_cache.pkl'
    var_4 = False
    var_5 = None
    var_6 = module_0.cache(var_5)
    var_7 = 'none.pkl'

import flutes.fs as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'test_cache.pkl'
    var_4 = False
    var_5 = None
    var_6 = module_0.cache(var_5)
    var_7 = 'none.pkl'



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'test_cache.pkl'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'wrong_key'
    var_5 = 'wrong_value'
    var_6 = {var_4: var_5}



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'src'
    var_1 = 'dst'
    var_2 = 'subdir'
    var_3 = 'file1'
    var_4 = 'file2'
    var_5 = 'file1.txt'
    var_6 = 'file2.txt'
    var_7 = 'new content'
    var_8 = True
    var_9 = 'another content'
    var_10 = False



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'file1.txt'
    var_1 = 'file2.txt'
    var_2 = [var_0, var_1]
    var_3 = 'dir1'
    var_4 = 'dir2'
    var_5 = [var_3, var_4]
    var_6 = 'test'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'subdir1'
    var_1 = 'subdir2'
    var_2 = 'test'
    var_3 = 'test'
    var_4 = 'file1.txt'
    var_5 = 'file2.txt'



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'test content'
    var_1 = 'subdir'
    var_2 = 'test content'
    var_3 = True
    var_4 = 'test_file1.txt'
    var_5 = 'test_file2.txt'



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'test_value'
    var_1 = 'modified_value'

def test_case_0():
    var_0 = 'test_value'

def test_case_0():
    var_0 = 'test_value'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'temp_dir'
    var_1 = True
    var_2 = 'file1.txt'
    var_3 = 'file2.txt'
    var_4 = 'sub_dir'
    var_5 = 'file3.txt'
    var_6 = 'temp_dir/file1.txt'
    var_7 = 'temp_dir/file2.txt'
    var_8 = 'temp_dir/sub_dir'



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'cache.pkl'
    var_1 = 1



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'file1'
    var_1 = 'file2'
    var_2 = 'subdir'
    var_3 = 'file3'



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'dir1'
    var_1 = 'dir2'
    var_2 = 'test'
    var_3 = 'test'
    var_4 = 'file1.txt'
    var_5 = 'file2.txt'
    var_6 = 'test_scandir passed'
    var_7 = print(var_6)



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'test_src'
    var_1 = 'test_dst'
    var_2 = 'file1.txt'
    var_3 = 'file1'
    var_4 = 'file2.txt'
    var_5 = 'file2'
    var_6 = 'dir1'
    var_7 = 'file3.txt'
    var_8 = 'file3'
    var_9 = 'file4.txt'
    var_10 = 'file4'
    var_11 = 'dir2'
    var_12 = 'file5.txt'
    var_13 = 'file5'
    var_14 = 'file6.txt'
    var_15 = 'file6'



# Parsed testcases at query #18
#--------------------------


import flutes.fs as module_0

def test_case_0():
    var_0 = 1024
    var_1 = module_0.readable_size(var_0)
    assert var_1 == '1.00K'
    var_2 = var_0 * var_0
    var_3 = module_0.readable_size(var_2)
    assert var_3 == '1.00M'
    var_4 = var_0 * var_0
    var_5 = var_4 * var_0
    var_6 = module_0.readable_size(var_5)
    assert var_6 == '1.00G'
    var_7 = var_0 * var_0
    var_8 = var_7 * var_0
    var_9 = var_8 * var_0
    var_10 = module_0.readable_size(var_9)
    assert var_10 == '1.00T'
    var_11 = var_0 * var_0
    var_12 = var_11 * var_0
    var_13 = var_12 * var_0
    var_14 = var_13 * var_0
    var_15 = module_0.readable_size(var_14)
    assert var_15 == '1.00P'
    var_16 = 0
    var_17 = module_0.readable_size(var_16)
    assert var_17 == '0.00'
    var_18 = 1
    var_19 = module_0.readable_size(var_18)
    assert var_19 == '1.00'
    var_20 = 999
    var_21 = module_0.readable_size(var_20)
    assert var_21 == '999.00'
    var_22 = 1000
    var_23 = module_0.readable_size(var_22)
    assert var_23 == '1000.00'
    var_24 = 1023
    var_25 = module_0.readable_size(var_24)
    assert var_25 == '1023.00'
    var_26 = 1025
    var_27 = module_0.readable_size(var_26)
    assert var_27 == '1.00K'
    var_28 = 2048
    var_29 = module_0.readable_size(var_28)
    assert var_29 == '2.00K'
    var_30 = module_0.readable_size(var_28, var_16)
    assert var_30 == '2K'
    var_31 = module_0.readable_size(var_28, var_18)
    assert var_31 == '2.0K'
    var_32 = 3
    var_33 = module_0.readable_size(var_28, var_32)
    assert var_33 == '2.000K'



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'cache.pkl'
    var_1 = 2
    var_2 = 3
    var_3 = 10
    var_4 = 5
    var_5 = 'None.pkl'



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'test_src'
    var_1 = 'test_dst'
    var_2 = True
    var_3 = 'file1.txt'
    var_4 = 'file1'
    var_5 = 'file2.txt'
    var_6 = 'file2'
    var_7 = 'subdir'
    var_8 = 'file3.txt'
    var_9 = 'file3'



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'subdir1'
    var_1 = 'subdir2'
    var_2 = 'test'
    var_3 = 'test'
    var_4 = 'file1.txt'
    var_5 = 'file2.txt'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'none_cache.pkl'

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'none_cache.pkl'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'cache.pkl'
    assert var_0 == 3
    assert var_0 == 5
    assert var_0 == 6
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = 5



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'file1.txt'
    var_1 = 'file2.txt'
    var_2 = 'subdir'
    var_3 = 'subdir/file3.txt'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'src'
    var_1 = 'dst'
    var_2 = 'subdir'
    var_3 = 'file1 content'
    var_4 = 'file2 content'
    var_5 = False
    var_6 = 'file1.txt'
    var_7 = 'file2.txt'
    var_8 = 'modified content'
    assert var_8 == 'file1 content'
    assert var_8 == 'modified content'
    var_9 = True



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'subdir1'
    var_1 = 'subdir2'
    var_2 = 'test'
    var_3 = 'test'
    var_4 = 'file1.txt'
    var_5 = 'file2.txt'



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'test_cache.pkl'



# Parsed testcases at query #8
#--------------------------


import flutes.fs as module_0

def test_case_0():
    var_0 = 2
    var_1 = 8
    var_2 = None
    var_3 = module_0.cache(var_2)
    var_4 = 2



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'file1'
    var_1 = 'file2'
    var_2 = 'dir1'
    var_3 = 'file3'
    var_4 = 'dir2'
    var_5 = 'nonexistent'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'subdir1'
    var_1 = 'subdir2'
    var_2 = 'test'
    var_3 = 'test'
    var_4 = 'file1.txt'
    var_5 = 'file2.txt'



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'subdir1'
    var_1 = 'subdir2'
    var_2 = 'test content'
    var_3 = 'test content'
    var_4 = 'file1.txt'
    var_5 = 'file2.txt'



# Parsed testcases at query #12
#--------------------------


import flutes.fs as module_0

def test_case_0():
    var_0 = 'test_cache.pkl'
    var_1 = module_0.cache(var_0)
    var_2 = 5
    var_3 = 6



# Parsed testcases at query #13
#--------------------------


import flutes.fs as module_0

def test_case_0():
    var_0 = 1024
    var_1 = module_0.readable_size(var_0)
    assert var_1 == '1.00K'
    var_2 = var_0 * var_0
    var_3 = module_0.readable_size(var_2)
    assert var_3 == '1.00M'
    var_4 = var_0 * var_0
    var_5 = var_4 * var_0
    var_6 = module_0.readable_size(var_5)
    assert var_6 == '1.00G'
    var_7 = var_0 * var_0
    var_8 = var_7 * var_0
    var_9 = var_8 * var_0
    var_10 = module_0.readable_size(var_9)
    assert var_10 == '1.00T'
    var_11 = var_0 * var_0
    var_12 = var_11 * var_0
    var_13 = var_12 * var_0
    var_14 = var_13 * var_0
    var_15 = module_0.readable_size(var_14)
    assert var_15 == '1.00P'
    var_16 = 0
    var_17 = module_0.readable_size(var_16)
    assert var_17 == '0.00'
    var_18 = 1
    var_19 = module_0.readable_size(var_18)
    assert var_19 == '1.00'
    var_20 = 1023
    var_21 = module_0.readable_size(var_20)
    assert var_21 == '1023.00'
    var_22 = 1025
    var_23 = module_0.readable_size(var_22)
    assert var_23 == '1.00K'
    var_24 = var_0 * var_0
    var_25 = var_24 - var_18
    var_26 = module_0.readable_size(var_25)
    assert var_26 == '1024.00K'
    var_27 = var_0 * var_0
    var_28 = var_27 + var_18
    var_29 = module_0.readable_size(var_28)
    assert var_29 == '1.00M'
    var_30 = var_0 * var_0
    var_31 = var_30 * var_0
    var_32 = var_31 - var_18
    var_33 = module_0.readable_size(var_32)
    assert var_33 == '1024.00M'
    var_34 = var_0 * var_0
    var_35 = var_34 * var_0
    var_36 = var_35 + var_18
    var_37 = module_0.readable_size(var_36)
    assert var_37 == '1.00G'
    var_38 = var_0 * var_0
    var_39 = var_38 * var_0
    var_40 = var_39 * var_0
    var_41 = var_40 - var_18
    var_42 = module_0.readable_size(var_41)
    assert var_42 == '1024.00G'
    var_43 = var_0 * var_0
    var_44 = var_43 * var_0
    var_45 = var_44 * var_0
    var_46 = var_45 + var_18
    var_47 = module_0.readable_size(var_46)
    assert var_47 == '1.00T'
    var_48 = var_0 * var_0
    var_49 = var_48 * var_0
    var_50 = var_49 * var_0
    var_51 = var_50 * var_0
    var_52 = var_51 - var_18
    var_53 = module_0.readable_size(var_52)
    assert var_53 == '1024.00T'
    var_54 = var_0 * var_0
    var_55 = var_54 * var_0
    var_56 = var_55 * var_0
    var_57 = var_56 * var_0
    var_58 = var_57 + var_18
    var_59 = module_0.readable_size(var_58)
    assert var_59 == '1.00P'



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'test'
    var_2 = 'subdir'
    var_3 = 'subtest.txt'
    var_4 = 'subtest'
    assert var_4 == 'test'
    assert var_4 == 'subtest'
    var_5 = 'test2'
    assert var_5 == 'test2'
    var_6 = True
    var_7 = 'test3'
    assert var_7 == 'test2'
    var_8 = False
    var_9 = 'test_copy_tree passed'
    var_10 = print(var_9)



# Parsed testcases at query #15
#--------------------------


import flutes.fs as module_0

def test_case_0():
    var_0 = 1024
    var_1 = module_0.readable_size(var_0)
    assert var_1 == '1.00K'
    var_2 = var_0 * var_0
    var_3 = module_0.readable_size(var_2)
    assert var_3 == '1.00M'
    var_4 = var_0 * var_0
    var_5 = var_4 * var_0
    var_6 = module_0.readable_size(var_5)
    assert var_6 == '1.00G'
    var_7 = var_0 * var_0
    var_8 = var_7 * var_0
    var_9 = var_8 * var_0
    var_10 = module_0.readable_size(var_9)
    assert var_10 == '1.00T'
    var_11 = var_0 * var_0
    var_12 = var_11 * var_0
    var_13 = var_12 * var_0
    var_14 = var_13 * var_0
    var_15 = module_0.readable_size(var_14)
    assert var_15 == '1.00P'
    var_16 = 0
    var_17 = module_0.readable_size(var_0, var_16)
    assert var_17 == '1K'
    var_18 = 1
    var_19 = module_0.readable_size(var_0, var_18)
    assert var_19 == '1.0K'
    var_20 = 3
    var_21 = module_0.readable_size(var_0, var_20)
    assert var_21 == '1.000K'



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'file1'
    var_1 = 'dir1'
    var_2 = 'file2'
    var_3 = 'file1.txt'
    var_4 = 'file2.txt'



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'subdir1'
    var_1 = 'subdir2'
    var_2 = 'test'
    var_3 = 'test'
    var_4 = 'file1.txt'
    var_5 = 'file2.txt'



# Parsed testcases at query #18
#--------------------------


import flutes.fs as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = True
    var_2 = 'file1.txt'
    var_3 = 'file2.txt'
    var_4 = 'sub_dir'
    var_5 = 'test'
    var_6 = 'test'
    var_7 = module_0.scandir(var_0)
    var_8 = list(var_7)
    var_9 = len(var_8)
    assert var_9 == 3
    var_10 = 0
    var_11 = var_8[var_10]
    var_12 = str(var_11)
    var_13 = var_8[var_6]
    var_14 = str(var_13)
    var_15 = 2
    var_16 = var_8[var_15]
    var_17 = str(var_16)



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'file1.txt'
    var_1 = 'file2.txt'
    var_2 = 'dir1'
    var_3 = 'test'
    var_4 = 'test'
    var_5 = 'file1.txt'
    var_6 = 'file2.txt'
    var_7 = 'dir1'
    var_8 = 'test'
    var_9 = 'test'



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'src'
    var_1 = 'dst'
    var_2 = 'file1'
    var_3 = 'file2'
    var_4 = 'subdir'
    var_5 = 'file3'
    assert var_5 == 'file1'
    assert var_5 == 'file2'
    assert var_5 == 'file3'
    var_6 = 'file1.txt'
    var_7 = 'file2.txt'
    var_8 = 'file3.txt'
    var_9 = 'new_file1'
    assert var_9 == 'new_file1'
    var_10 = True
    var_11 = 'another_new_file1'
    assert var_11 == 'new_file1'
    var_12 = False



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'dir1'
    var_1 = 'dir2'
    var_2 = 'test'
    var_3 = 'test'
    var_4 = 'file1.txt'
    var_5 = 'file2.txt'



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'test_cache.pkl'
    var_1 = 'None'
    var_2 = 'test_cache_quiet.pkl'
    var_3 = 'test_cache_named.pkl'



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'test_cache.pkl'

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'test_cache.pkl'



# Parsed testcases at query #24
#--------------------------


import flutes.fs as module_0

def test_case_0():
    var_0 = 2
    var_1 = 'cache.pkl'
    var_2 = 10
    var_3 = 5
    var_4 = False
    var_5 = None
    var_6 = module_0.cache(var_5)
    var_7 = 'test'

import flutes.fs as module_0

def test_case_0():
    var_0 = 2
    var_1 = 'cache.pkl'
    var_2 = 10
    var_3 = 5
    var_4 = False
    var_5 = None
    var_6 = module_0.cache(var_5)
    var_7 = 'test'



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'subdir'
    var_1 = 'test content'
    var_2 = 'subdir'
    var_3 = 'file.txt'
    var_4 = 'modified content'
    assert var_4 == 'modified content'
    var_5 = True



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'test content'
    var_2 = 'test_subdir'
    var_3 = 'subdir_test.txt'
    var_4 = 'subdir test content'
    assert var_4 == 'test content'
    assert var_4 == 'subdir test content'
    var_5 = 'overwritten content'
    assert var_5 == 'overwritten content'
    assert var_5 == 'test content'
    var_6 = False
    var_7 = True



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'test_src'
    var_1 = True
    var_2 = 'file1.txt'
    var_3 = 'Hello, World!'
    var_4 = 'subdir'
    var_5 = 'file2.txt'
    var_6 = 'Another file'
    var_7 = 'test_dst'



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'file1'
    var_1 = 'file2'
    var_2 = 'dir1'
    var_3 = 'dir2'



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 'file1.txt'
    var_1 = 'w'
    var_2 = 'file2.txt'
    var_3 = 'dir1'
    var_4 = 'file3.txt'



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'file1.txt'
    var_1 = 'file2.txt'
    var_2 = 'subdir'



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 'file1.txt'
    var_1 = 'file2.txt'
    var_2 = 'subdir'
    var_3 = 'file3.txt'



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = 'src'
    var_1 = 'dst'
    var_2 = 'subdir'
    var_3 = 'file1'
    var_4 = 'file2'
    var_5 = 'file1.txt'
    var_6 = 'file2.txt'
    var_7 = 'modified'
    assert var_7 == 'file1'
    assert var_7 == 'modified'
    var_8 = True



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = 'src'
    var_1 = 'dst'
    var_2 = 'subdir'
    var_3 = 'test file 1'
    var_4 = 'test file 2'
    var_5 = 'file1.txt'
    var_6 = 'file2.txt'
    var_7 = 'modified file 1'
    assert var_7 == 'test file 1'
    assert var_7 == 'modified file 1'
    var_8 = False
    var_9 = True



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = 'subdir1'
    var_1 = 'subdir2'
    var_2 = 'test'
    var_3 = 'test'
    var_4 = 'file1.txt'
    var_5 = 'file2.txt'



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = 'test1.txt'
    var_1 = 'test2.txt'
    var_2 = 'test3.txt'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'dir1'
    var_5 = 'dir2'
    var_6 = 'dir3'
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3 + var_7
    var_9 = len(var_8)
    var_10 = var_3 + var_7
    var_11 = len(var_10)



# Parsed testcases at query #36
#--------------------------


def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'test content'
    var_2 = 'subdir'
    var_3 = 'subfile.txt'
    var_4 = 'sub content'
    assert var_4 == 'test content'
    assert var_4 == 'sub content'
    var_5 = 'conflict content'
    assert var_5 == 'conflict content'
    assert var_5 == 'test content'
    var_6 = False
    var_7 = True



# Parsed testcases at query #37
#--------------------------


def test_case_0():
    var_0 = 'test_dir'
    var_1 = True
    var_2 = 'file1.txt'
    var_3 = 'file2.txt'
    var_4 = 'subdir'
    var_5 = 'file3.txt'
    var_6 = list(var_0)
    var_7 = len(var_6)
    assert var_7 == 3
    var_8 = 'file1.txt'
    var_9 = any(var_2)
    var_10 = 'file1.txt not found'
    var_11 = 'file2.txt'
    var_12 = 'file2.txt not found'
    var_13 = 'subdir'
    var_14 = 'subdir not found'
    var_15 = 0



# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'test_cache.pkl'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'test_cache.pkl'



# Parsed testcases at query #39
#--------------------------


import flutes.fs as module_0

def test_case_0():
    var_0 = 'cache.pkl'
    assert var_0 == 4
    var_1 = 2
    var_2 = 3
    var_3 = None
    var_4 = module_0.cache(var_3)



# Parsed testcases at query #40
#--------------------------


def test_case_0():
    var_0 = 2
    var_1 = 2

def test_case_0():
    var_0 = 3
    var_1 = 3

def test_case_0():
    var_0 = 4
    var_1 = 4
    var_2 = 'nonexistent.pkl'

def test_case_0():
    var_0 = 5
    var_1 = 5
    var_2 = 'All cache tests passed.'
    var_3 = print(var_2)

def test_case_0():
    var_0 = 5
    var_1 = 5
    var_2 = 'All cache tests passed.'
    var_3 = print(var_2)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'subdir'
    var_1 = 'file1.txt'
    var_2 = 'file2.txt'
    var_3 = 'test'
    var_4 = 'test'
    var_5 = 'empty'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'file1.txt'
    var_1 = 'file2.txt'
    var_2 = 'subdir1'
    var_3 = 'file3.txt'
    var_4 = 'subdir2'
    var_5 = 'test_scandir passed'
    var_6 = print(var_5)



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'test_cache.pkl'
    var_1 = 'None'
    var_2 = 'test_cache_quiet.pkl'
    var_3 = 'test_cache_named.pkl'
    var_4 = 'temp_cache.pkl'



# Parsed testcases at query #4
#--------------------------


import flutes.fs as module_0

def test_case_0():
    var_0 = 'cache.pkl'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = None
    var_6 = module_0.cache(var_5)
    var_7 = 'test'



# Parsed testcases at query #5
#--------------------------


import flutes.fs as module_0

def test_case_0():
    var_0 = 1024
    var_1 = module_0.readable_size(var_0)
    assert var_1 == '1.00K'
    var_2 = var_0 * var_0
    var_3 = module_0.readable_size(var_2)
    assert var_3 == '1.00M'
    var_4 = var_0 * var_0
    var_5 = var_4 * var_0
    var_6 = module_0.readable_size(var_5)
    assert var_6 == '1.00G'
    var_7 = var_0 * var_0
    var_8 = var_7 * var_0
    var_9 = var_8 * var_0
    var_10 = module_0.readable_size(var_9)
    assert var_10 == '1.00T'
    var_11 = var_0 * var_0
    var_12 = var_11 * var_0
    var_13 = var_12 * var_0
    var_14 = var_13 * var_0
    var_15 = module_0.readable_size(var_14)
    assert var_15 == '1.00P'
    var_16 = 512
    var_17 = module_0.readable_size(var_16)
    assert var_17 == '512.00'
    var_18 = 0
    var_19 = module_0.readable_size(var_16, var_18)
    assert var_19 == '512'
    var_20 = 4
    var_21 = module_0.readable_size(var_16, var_20)
    assert var_21 == '512.0000'



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'dir1'
    var_1 = 'dir2'
    var_2 = 'test'
    var_3 = 'test'
    var_4 = 'file1.txt'
    var_5 = 'file2.txt'
    var_6 = 'dir1'
    var_7 = 'dir2'
    var_8 = 'file1.txt'
    var_9 = 'test'
    var_10 = 'file2.txt'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'dir1'
    var_1 = 'dir2'
    var_2 = 'file1'
    var_3 = 'file2'
    var_4 = 'file3'
    var_5 = 'file1.txt'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 42
    assert var_0 == 42
    var_1 = 'test_cache.pkl'

def test_case_0():
    var_0 = 42
    assert var_0 == 42
    var_1 = 'test_cache.pkl'



# Parsed testcases at query #4
#--------------------------




# Parsed testcases at query #5
#--------------------------


import flutes.fs as module_0

def test_case_0():
    var_0 = 1024
    var_1 = module_0.readable_size(var_0)
    assert var_1 == '1.00K'
    var_2 = var_0 * var_0
    var_3 = module_0.readable_size(var_2)
    assert var_3 == '1.00M'
    var_4 = var_0 * var_0
    var_5 = var_4 * var_0
    var_6 = module_0.readable_size(var_5)
    assert var_6 == '1.00G'
    var_7 = var_0 * var_0
    var_8 = var_7 * var_0
    var_9 = var_8 * var_0
    var_10 = module_0.readable_size(var_9)
    assert var_10 == '1.00T'
    var_11 = var_0 * var_0
    var_12 = var_11 * var_0
    var_13 = var_12 * var_0
    var_14 = var_13 * var_0
    var_15 = module_0.readable_size(var_14)
    assert var_15 == '1.00P'
    var_16 = 512
    var_17 = module_0.readable_size(var_16)
    assert var_17 == '512.00'
    var_18 = 1536
    var_19 = module_0.readable_size(var_18)
    assert var_19 == '1.50K'
    var_20 = 0
    var_21 = module_0.readable_size(var_18, var_20)
    assert var_21 == '2K'
    var_22 = 1
    var_23 = module_0.readable_size(var_18, var_22)
    assert var_23 == '1.5K'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'test'
    assert var_0 == 'test'
    var_1 = 'test_cache.pkl'

def test_case_0():
    var_0 = 'test'
    assert var_0 == 'test'
    var_1 = 'test_cache.pkl'



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'src'
    var_1 = 'dst'
    var_2 = 'file1'
    var_3 = 'file2'
    var_4 = 'subdir'
    var_5 = 'file3'
    assert var_5 == 'file1'
    assert var_5 == 'file2'
    assert var_5 == 'file3'
    var_6 = 'file1.txt'
    var_7 = 'file2.txt'
    var_8 = 'file3.txt'
    var_9 = 'file1_updated'
    assert var_9 == 'file1_updated'
    var_10 = True
    var_11 = 'file2_updated'
    assert var_11 == 'file2'
    var_12 = False



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'src'
    var_1 = 'file1.txt'
    var_2 = 'Hello'
    var_3 = 'file2.txt'
    var_4 = 'World'
    var_5 = 'subdir'
    var_6 = 'file3.txt'
    var_7 = 'File 3'
    var_8 = 'dst'
    var_9 = 'Updated'
    var_10 = True
    var_11 = 'New Update'
    var_12 = False



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'subdir'
    var_1 = 'test content'
    var_2 = 'test content 2'
    var_3 = 'file1.txt'
    var_4 = 'subdir'
    var_5 = 'file2.txt'
    var_6 = 'modified content'
    assert var_6 == 'modified content'
    var_7 = True
    var_8 = 'new content'
    assert var_8 == 'modified content'
    var_9 = False
    var_10 = 'test_copy_tree passed'
    var_11 = print(var_10)



# Parsed testcases at query #10
#--------------------------




# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'cache.pkl'
    var_1 = 1
    var_2 = 2
    var_3 = 'cache.pkl'
    var_4 = 1
    var_5 = 2
    var_6 = 'cache.pkl'
    var_7 = 1
    var_8 = 2
    assert var_8 == 3
    var_9 = 1
    var_10 = 2
    var_11 = 'cache.pkl'
    var_12 = 1
    var_13 = 2
    var_14 = 10
    var_15 = 'cache.pkl'
    var_16 = 1
    var_17 = 2



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'tmp_dir'
    var_1 = True
    var_2 = 'file1.txt'
    var_3 = 'file2.txt'
    var_4 = 'subdir'
    var_5 = 'file3.txt'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'test_cache.pkl'
    var_1 = 1
    var_2 = 2



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'subdir1'
    var_1 = 'subdir2'
    var_2 = 'test'
    var_3 = 'test'
    var_4 = 'file1.txt'
    var_5 = 'file2.txt'



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'src'
    var_1 = 'dst'
    var_2 = 'test'
    assert var_2 == 'test'
    var_3 = 'test.txt'
    var_4 = 'test2'
    assert var_4 == 'test2'
    assert var_4 == 'test'
    var_5 = False
    var_6 = True



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'test_cache_file.pkl'



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'src'
    var_1 = 'dst'
    var_2 = 'file1'
    var_3 = 'subdir'
    var_4 = 'file2'



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'subdir'
    var_1 = 'file1.txt'
    var_2 = 'file2.txt'
    var_3 = 'file1'
    var_4 = 'file2'
    var_5 = 'file1_modified'
    var_6 = True



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'subdir'
    var_1 = 'test1'
    var_2 = 'test2'
    var_3 = False
    var_4 = 'file1.txt'
    var_5 = 'file2.txt'
    var_6 = 'modified'
    assert var_6 == 'modified'
    var_7 = True



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'subdir1'
    var_1 = 'subdir2'
    var_2 = 'test'
    var_3 = 'test'
    var_4 = 'file1.txt'
    var_5 = 'file2.txt'



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'test'
    var_1 = sorted(var_0)



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'test_cache.pkl'
    var_1 = 'test_cache.pkl'
    var_2 = 'test_cache_args.pkl'
    var_3 = 1
    var_4 = 2
    var_5 = 'test_cache_args.pkl'
    var_6 = 10
    var_7 = 20
    var_8 = 'None'
    var_9 = 'test_cache_verbose.pkl'
    var_10 = 'test_cache_verbose.pkl'
    var_11 = 'test_cache_name.pkl'
    var_12 = 'test_cache_name.pkl'



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = True



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'dir1'
    var_1 = 'dir2'
    var_2 = 'test'
    var_3 = 'test'
    var_4 = 'file1.txt'
    var_5 = 'file2.txt'



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'file1.txt'
    var_1 = 'file2.txt'
    var_2 = 'dir1'
    var_3 = 'dir2'
    var_4 = 'test'
    var_5 = 'test'



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 'file1.txt'
    var_1 = 'file2.txt'
    var_2 = 'subdir'
    var_3 = 'subdir/file3.txt'
    var_4 = 'empty'
    var_5 = 'All scandir tests passed!'
    var_6 = print(var_5)



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'subdir'
    var_1 = 'file1.txt'
    var_2 = 'file2.txt'
    var_3 = 'test'
    var_4 = 'test'
    var_5 = 'test_scandir passed'
    var_6 = print(var_5)



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'subdir'
    var_1 = 'test content'
    var_2 = 'test content 2'
    var_3 = 'file1.txt'
    var_4 = 'subdir'
    var_5 = 'file2.txt'
    var_6 = 'modified content'
    assert var_6 == 'modified content'
    var_7 = True
    var_8 = 'new content'
    assert var_8 == 'modified content'



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'Cache file should be created'



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'test'



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 'subdir1'
    var_1 = 'subdir2'
    var_2 = 'test'
    var_3 = 'test'
    var_4 = 'file1.txt'
    var_5 = 'file2.txt'



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = 'src'
    var_1 = 'dst'
    var_2 = 'file1'
    var_3 = 'subdir'
    var_4 = 'file2'
    var_5 = 'file1.txt'
    var_6 = 'file2.txt'
    var_7 = 'file1_modified'
    var_8 = 'file2_modified'
    assert var_8 == 'file1'
    assert var_8 == 'file2'
    assert var_8 == 'file1_modified'
    assert var_8 == 'file2_modified'
    var_9 = True



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = 'test'
    var_1 = 'subdir'



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = 'src'
    var_1 = 'dst'
    var_2 = 'file1'
    var_3 = 'subdir'
    var_4 = 'file2'



# Parsed testcases at query #35
#--------------------------




