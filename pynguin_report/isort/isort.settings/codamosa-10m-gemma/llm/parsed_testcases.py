####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'py'
    var_2 = 'pyi'
    var_3 = 'txt'
    var_4 = 'test.py'
    var_5 = var_0.is_supported_filetype(var_4)
    assert var_5 is True
    var_6 = 'test.pyi'
    var_7 = var_0.is_supported_filetype(var_6)
    assert var_7 is True
    var_8 = 'test.txt'
    var_9 = var_0.is_supported_filetype(var_8)
    assert var_9 is False
    var_10 = 'script.sh'
    var_11 = var_0.is_supported_filetype(var_10)
    assert var_11 is True
    var_12 = 'test.py~'
    var_13 = var_0.is_supported_filetype(var_12)
    assert var_13 is False
    var_14 = 'pipe.py'
    var_15 = var_0.is_supported_filetype(var_14)
    assert var_15 is False
    var_16 = 'nonexistent.py'
    var_17 = var_0.is_supported_filetype(var_16)
    assert var_17 is False
    var_18 = 'unknown.abc'
    var_19 = var_0.is_supported_filetype(var_18)
    assert var_19 is False



# Parsed testcases at query #2
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'line_length'
    var_2 = 'indent'
    var_3 = 'known_third_party'
    var_4 = 100
    var_5 = 4
    var_6 = 'requests'
    var_7 = 'numpy'
    var_8 = [var_6, var_7]
    var_9 = {var_1: var_4, var_2: var_5, var_3: var_8}
    var_10 = module_0.Config(**var_9)
    var_11 = 80
    var_12 = module_0.Config()
    var_13 = 120
    var_14 = module_0.Config(config=var_12)



# Parsed testcases at query #3
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = '\n    Tests find_all_configs by creating a temporary directory structure \n    with various config files and verifying the Trie output.\n    '
    var_1 = 'project'
    var_2 = 'subdir'
    var_3 = 'empty_dir'
    var_4 = 'other_dir'
    var_5 = '.isort.cfg'
    var_6 = '[settings]\nline_length = 88'
    var_7 = 'pyproject.toml'
    var_8 = "[tool.isort]\nprofile = 'black'"
    var_9 = '[settings]\nindent = 4'
    var_10 = 'line_length'
    var_11 = 88
    var_12 = {var_10: var_11}
    var_13 = 'profile'
    var_14 = 'black'
    var_15 = {var_13: var_14}
    var_16 = 'indent'
    var_17 = 4
    var_18 = {var_16: var_17}
    var_19 = module_0.find_all_configs(var_0)
    var_20 = False
    var_21 = False
    var_22 = False
    var_23 = module_0.find_all_configs(var_0)
    var_24 = 0
    var_25 = 'empty_dir'
    var_26 = 'no_config_here'
    var_27 = any(var_5)

import isort.settings as module_0

def test_case_0():
    var_0 = 'Tests find_all_configs when no configuration files exist.'
    var_1 = 'empty'
    var_2 = module_0.find_all_configs(var_0)



# Parsed testcases at query #4
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = '\n    Tests find_all_configs by creating a mock directory structure with \n    various config files and verifying they are correctly inserted into the Trie.\n    '
    var_1 = 'root'
    var_2 = 'sub'
    var_3 = 'empty_dir'
    var_4 = '.isort.cfg'
    var_5 = 'pyproject.toml'
    var_6 = 'section_name = value1'
    var_7 = 'section_name = value2'
    var_8 = [var_4, var_5]
    var_9 = module_0.find_all_configs(var_0)
    var_10 = False
    var_11 = False
    var_12 = module_0.find_all_configs(var_0)
    var_13 = [call.args for call in var_2]
    var_14 = 'key1'
    var_15 = 1
    var_16 = 'Config 1 was not inserted into Trie'
    var_17 = 'key2'
    var_18 = any(var_6)
    var_19 = 'Config 2 was not inserted into Trie'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'py_version'
    var_1 = 'line_length'
    var_2 = 'indent'
    var_3 = 'sections'
    var_4 = 'known_standard_library'
    var_5 = 'known_third_party'
    var_6 = 'known_first_party'
    var_7 = 'known_local_folder'
    var_8 = 'import_headings'
    var_9 = 'import_footers'
    var_10 = 'skip'
    var_11 = 'skip_glob'
    var_12 = 'extend_skip'
    var_13 = 'extend_skip_glob'
    var_14 = 'sort_order'
    var_15 = 'directory'
    var_16 = 'src_paths'
    var_17 = 'profiles'
    var_18 = 'import_foot_er'
    var_19 = 'pyversion'
    var_20 = 'py39'
    var_21 = 88
    var_22 = 4
    var_23 = 'FUTURE'
    var_24 = 'STDLIB'
    var_25 = (var_23, var_24)
    var_26 = 'os'
    var_27 = 'sys'
    var_28 = (var_26, var_27)
    var_29 = 'requests'
    var_30 = (var_29,)
    var_31 = 'my_app'
    var_32 = (var_31,)
    var_33 = 'utils'
    var_34 = (var_33,)
    var_35 = 'custom'
    var_36 = 'Custom'
    var_37 = {var_35: var_36}
    var_38 = 'end'
    var_39 = 'End'
    var_40 = {var_38: var_39}
    var_41 = 'test'
    var_42 = [var_41]
    var_43 = frozenset(var_42)
    var_44 = '*.tmp'
    var_45 = [var_44]
    var_46 = frozenset(var_45)
    var_47 = '*.log'
    var_48 = [var_47]
    var_49 = frozenset(var_48)
    var_50 = '*.bak'
    var_51 = [var_50]
    var_52 = frozenset(var_51)
    var_53 = 'natural'
    var_54 = '/tmp'
    var_55 = 'src'
    var_56 = (var_55,)
    var_57 = {}
    var_58 = (var_24,)
    var_59 = (var_26,)
    var_60 = (var_29,)
    var_61 = (var_31,)
    var_62 = (var_33,)
    var_63 = {}
    var_64 = {}
    var_65 = frozenset()
    var_66 = frozenset()
    var_67 = frozenset()
    var_68 = frozenset()
    var_69 = (var_55,)
    var_70 = {}
    var_71 = (var_24,)
    var_72 = (var_26,)
    var_73 = (var_29,)
    var_74 = (var_31,)
    var_75 = (var_33,)
    var_76 = {}
    var_77 = {}
    var_78 = frozenset()
    var_79 = frozenset()
    var_80 = frozenset()
    var_81 = frozenset()
    var_82 = (var_55,)
    var_83 = {}
    var_84 = (var_24,)
    var_85 = (var_26,)
    var_86 = (var_29,)
    var_87 = (var_31,)
    var_88 = (var_33,)
    var_89 = {}
    var_90 = {}
    var_91 = frozenset()
    var_92 = frozenset()
    var_93 = frozenset()
    var_94 = frozenset()
    var_95 = (var_55,)
    var_96 = {}
    var_97 = (var_24,)
    var_98 = (var_26,)
    var_99 = (var_29,)
    var_100 = (var_31,)
    var_101 = (var_33,)
    var_102 = {}
    var_103 = {}
    var_104 = frozenset()
    var_105 = frozenset()
    var_106 = frozenset()
    var_107 = frozenset()
    var_108 = (var_55,)
    var_109 = {}
    var_110 = (var_24,)
    var_111 = (var_26,)
    var_112 = (var_29,)
    var_113 = (var_31,)
    var_114 = 'supposed_to_be_here'
    var_115 = (var_114,)
    var_116 = {}
    var_117 = {}
    var_118 = frozenset()
    var_119 = frozenset()
    var_120 = frozenset()
    var_121 = frozenset()
    var_122 = (var_55,)
    var_123 = {}
    var_124 = (var_24,)
    var_125 = (var_26,)
    var_126 = (var_29,)
    var_127 = (var_31,)
    var_128 = (var_33,)
    var_129 = {}
    var_130 = {}
    var_131 = frozenset()
    var_132 = frozenset()
    var_133 = frozenset()
    var_134 = frozenset()
    var_135 = (var_55,)
    var_136 = {}
    var_137 = (var_24,)
    var_138 = (var_26,)
    var_139 = (var_29,)
    var_140 = (var_31,)
    var_141 = (var_33,)
    var_142 = {}
    var_143 = {}
    var_144 = frozenset()
    var_145 = frozenset()
    var_146 = frozenset()
    var_147 = frozenset()
    var_148 = '/src'
    var_149 = (var_55,)
    var_150 = {}
    var_151 = (var_24,)
    var_152 = (var_26,)
    var_153 = (var_29,)
    var_154 = (var_31,)
    var_155 = (var_33,)
    var_156 = {}
    var_157 = {}
    var_158 = frozenset()
    var_159 = frozenset()
    var_160 = frozenset()
    var_161 = frozenset()
    var_162 = (var_55,)
    var_163 = {}
    var_164 = (var_24,)
    var_165 = (var_26,)
    var_166 = (var_29,)
    var_167 = (var_31,)
    var_168 = (var_33,)
    var_169 = {}
    var_170 = {}
    var_171 = frozenset()
    var_172 = frozenset()
    var_173 = frozenset()
    var_174 = frozenset()
    var_175 = (var_55,)
    var_176 = {}
    var_177 = (var_24,)
    var_178 = (var_26,)
    var_179 = (var_29,)
    var_180 = (var_31,)
    var_181 = (var_33,)
    var_182 = {}
    var_183 = {}
    var_184 = frozenset()
    var_185 = frozenset()
    var_186 = frozenset()
    var_187 = frozenset()
    var_188 = (var_55,)
    var_189 = {}
    var_190 = (var_24,)
    var_191 = (var_26,)
    var_192 = (var_29,)
    var_193 = (var_31,)
    var_194 = (var_33,)
    var_195 = {}
    var_196 = {}
    var_197 = frozenset()
    var_198 = frozenset()
    var_199 = frozenset()
    var_200 = frozenset()
    var_201 = (var_55,)
    var_202 = {}
    var_203 = (var_24,)
    var_204 = (var_26,)
    var_205 = (var_29,)
    var_206 = (var_31,)
    var_207 = (var_33,)
    var_208 = {}
    var_209 = {}
    var_210 = frozenset()
    var_211 = frozenset()
    var_212 = frozenset()
    var_213 = frozenset()
    var_214 = (var_55,)
    var_215 = {}
    var_216 = (var_24,)
    var_217 = (var_55,)
    var_218 = (var_29,)
    var_219 = (var_31,)
    var_220 = (var_33,)
    var_221 = {}
    var_222 = {}
    var_223 = frozenset()
    var_224 = frozenset()
    var_225 = frozenset()
    var_226 = frozenset()
    var_227 = (var_55,)
    var_228 = {}
    var_229 = (var_24,)
    var_230 = (var_26,)
    var_231 = (var_29,)
    var_232 = (var_31,)
    var_233 = (var_33,)
    var_234 = {}
    var_235 = {}
    var_236 = frozenset()
    var_237 = frozenset()
    var_238 = frozenset()
    var_239 = frozenset()
    var_240 = (var_55,)
    var_241 = {}
    var_242 = (var_24,)
    var_243 = (var_55,)
    var_244 = (var_29,)
    var_245 = (var_31,)
    var_246 = (var_33,)
    var_247 = {}
    var_248 = {}
    var_249 = frozenset()
    var_250 = frozenset()
    var_251 = frozenset()
    var_252 = frozenset()
    var_253 = (var_55,)
    var_254 = {}
    var_255 = (var_24,)
    var_256 = (var_26,)
    var_257 = (var_29,)
    var_258 = (var_31,)
    var_259 = (var_33,)
    var_260 = {}
    var_261 = {}
    var_262 = frozenset()
    var_263 = frozenset()
    var_264 = frozenset()
    var_265 = frozenset()
    var_266 = (var_55,)
    var_267 = {}
    var_268 = (var_24,)
    var_269 = (var_26,)
    var_270 = (var_29,)
    var_271 = (var_31,)
    var_272 = (var_33,)
    var_273 = {}
    var_274 = {}
    var_275 = frozenset()
    var_276 = frozenset()
    var_277 = frozenset()
    var_278 = frozenset()
    var_279 = (var_55,)
    var_280 = {}
    var_281 = (var_24,)
    var_282 = (var_26,)
    var_283 = (var_29,)
    var_284 = (var_31,)
    var_285 = (var_33,)
    var_286 = {}
    var_287 = {}
    var_288 = frozenset()
    var_289 = frozenset()
    var_290 = frozenset()
    var_291 = frozenset()
    var_292 = (var_55,)
    var_293 = {}
    var_294 = (var_24,)
    var_295 = (var_26,)
    var_296 = (var_29,)
    var_297 = (var_31,)
    var_298 = (var_33,)
    var_299 = {}
    var_300 = {}
    var_301 = frozenset()
    var_302 = frozenset()
    var_303 = frozenset()
    var_304 = frozenset()
    var_305 = (var_55,)
    var_306 = {}
    var_307 = (var_24,)
    var_308 = (var_26,)
    var_309 = (var_29,)
    var_310 = (var_31,)
    var_311 = (var_33,)
    var_312 = {}
    var_313 = {}
    var_314 = frozenset()
    var_315 = frozenset()
    var_316 = frozenset()
    var_317 = frozenset()
    var_318 = (var_55,)
    var_319 = {}
    var_320 = (var_24,)
    var_321 = (var_26,)
    var_322 = (var_29,)
    var_323 = (var_31,)
    var_324 = (var_33,)
    var_325 = {}
    var_326 = {}
    var_327 = frozenset()
    var_328 = frozenset()
    var_329 = frozenset()
    var_330 = frozenset()
    var_331 = (var_55,)
    var_332 = {}
    var_333 = (var_24,)
    var_334 = (var_26,)
    var_335 = (var_29,)
    var_336 = (var_31,)
    var_337 = (var_33,)
    var_338 = {}
    var_339 = {}
    var_340 = frozenset()
    var_341 = frozenset()
    var_342 = frozenset()
    var_343 = frozenset()
    var_344 = (var_55,)
    var_345 = {}
    var_346 = (var_24,)
    var_347 = (var_55,)
    var_348 = (var_29,)
    var_349 = (var_31,)
    var_350 = (var_33,)
    var_351 = {}
    var_352 = {}
    var_353 = frozenset()
    var_354 = frozenset()
    var_355 = frozenset()
    var_356 = frozenset()
    var_357 = (var_55,)
    var_358 = {}
    var_359 = (var_24,)
    var_360 = (var_26,)
    var_361 = (var_29,)
    var_362 = (var_31,)
    var_363 = (var_33,)
    var_364 = {}
    var_365 = {}
    var_366 = frozenset()
    var_367 = frozenset()
    var_368 = frozenset()
    var_369 = frozenset()
    var_370 = (var_55,)
    var_371 = {}
    var_372 = (var_24,)
    var_373 = (var_26,)
    var_374 = (var_29,)
    var_375 = (var_31,)
    var_376 = (var_33,)
    var_377 = {}
    var_378 = {}
    var_379 = frozenset()
    var_380 = frozenset()
    var_381 = frozenset()
    var_382 = frozenset()
    var_383 = (var_55,)
    var_384 = {}
    var_385 = (var_24,)
    var_386 = (var_26,)
    var_387 = (var_29,)
    var_388 = (var_31,)
    var_389 = (var_33,)
    var_390 = {}
    var_391 = {}
    var_392 = frozenset()
    var_393 = frozenset()
    var_394 = frozenset()
    var_395 = frozenset()
    var_396 = (var_55,)
    var_397 = {}
    var_398 = (var_24,)
    var_399 = (var_26,)
    var_400 = (var_29,)
    var_401 = (var_31,)
    var_402 = (var_33,)
    var_403 = {}
    var_404 = {}
    var_405 = frozenset()
    var_406 = frozenset()
    var_407 = frozenset()
    var_408 = frozenset()
    var_409 = (var_55,)
    var_410 = {}
    var_411 = (var_24,)
    var_412 = (var_26,)
    var_413 = (var_29,)
    var_414 = (var_31,)
    var_415 = (var_33,)
    var_416 = {}
    var_417 = {}
    var_418 = frozenset()
    var_419 = frozenset()
    var_420 = frozenset()
    var_421 = frozenset()
    var_422 = (var_55,)
    var_423 = {}
    var_424 = (var_24,)
    var_425 = (var_26,)
    var_426 = (var_29,)
    var_427 = (var_31,)
    var_428 = (var_33,)
    var_429 = {}
    var_430 = {}
    var_431 = frozenset()
    var_432 = frozenset()
    var_433 = frozenset()
    var_434 = frozenset()
    var_435 = (var_55,)
    var_436 = {}
    var_437 = (var_24,)
    var_438 = (var_26,)
    var_439 = (var_29,)
    var_440 = (var_31,)
    var_441 = (var_33,)
    var_442 = {}
    var_443 = {}
    var_444 = frozenset()
    var_445 = frozenset()
    var_446 = frozenset()
    var_447 = frozenset()
    var_448 = (var_55,)
    var_449 = {}
    var_450 = (var_24,)
    var_451 = (var_26,)
    var_452 = (var_29,)
    var_453 = (var_31,)
    var_454 = (var_33,)
    var_455 = {}
    var_456 = {}
    var_457 = frozenset()
    var_458 = frozenset()
    var_459 = frozenset()
    var_460 = frozenset()
    var_461 = (var_55,)
    var_462 = {}
    var_463 = (var_24,)
    var_464 = (var_26,)
    var_465 = (var_29,)
    var_466 = (var_31,)
    var_467 = (var_33,)
    var_468 = {}
    var_469 = {}
    var_470 = frozenset()
    var_471 = frozenset()
    var_472 = frozenset()
    var_473 = frozenset()
    var_474 = '3.10'
    var_475 = {var_0: var_20, var_1: var_21, var_2: var_22, var_3: var_25, var_4: var_28, var_5: var_30, var_6: var_32, var_7: var_34, var_8: var_37, var_9: var_40, var_10: var_43, var_11: var_46, var_12: var_49, var_13: var_52, var_14: var_53, var_15: var_54, var_16: var_56, var_17: var_57, var_3: var_58, var_4: var_59, var_5: var_60, var_6: var_61, var_7: var_62, var_8: var_63, var_9: var_64, var_10: var_65, var_11: var_66, var_12: var_67, var_13: var_68, var_14: var_53, var_15: var_54, var_16: var_69, var_17: var_70, var_3: var_71, var_4: var_72, var_5: var_73, var_6: var_74, var_7: var_75, var_8: var_76, var_9: var_77, var_10: var_78, var_11: var_79, var_12: var_80, var_13: var_81, var_14: var_53, var_15: var_54, var_16: var_82, var_17: var_83, var_3: var_84, var_4: var_85, var_5: var_86, var_6: var_87, var_7: var_88, var_8: var_89, var_9: var_90, var_10: var_91, var_11: var_92, var_12: var_93, var_13: var_94, var_14: var_53, var_15: var_54, var_16: var_95, var_17: var_96, var_3: var_97, var_4: var_98, var_5: var_99, var_6: var_100, var_7: var_101, var_8: var_102, var_9: var_103, var_10: var_104, var_11: var_105, var_12: var_106, var_13: var_107, var_14: var_53, var_15: var_54, var_16: var_108, var_17: var_109, var_3: var_110, var_4: var_111, var_5: var_112, var_6: var_113, var_7: var_115, var_8: var_116, var_9: var_117, var_10: var_118, var_11: var_119, var_12: var_120, var_13: var_121, var_14: var_53, var_15: var_54, var_16: var_122, var_17: var_123, var_3: var_124, var_4: var_125, var_5: var_126, var_6: var_127, var_7: var_128, var_8: var_129, var_9: var_130, var_10: var_131, var_11: var_132, var_12: var_133, var_13: var_134, var_14: var_53, var_15: var_54, var_16: var_135, var_17: var_136, var_3: var_137, var_4: var_138, var_5: var_139, var_6: var_140, var_7: var_141, var_8: var_142, var_9: var_143, var_10: var_144, var_11: var_145, var_12: var_146, var_13: var_147, var_14: var_53, var_15: var_148, var_16: var_149, var_17: var_150, var_3: var_151, var_4: var_152, var_5: var_153, var_6: var_154, var_7: var_155, var_8: var_156, var_9: var_157, var_10: var_158, var_11: var_159, var_12: var_160, var_13: var_161, var_14: var_53, var_15: var_148, var_16: var_162, var_17: var_163, var_3: var_164, var_4: var_165, var_5: var_166, var_6: var_167, var_7: var_168, var_8: var_169, var_9: var_170, var_10: var_171, var_11: var_172, var_12: var_173, var_13: var_174, var_14: var_53, var_15: var_148, var_16: var_175, var_17: var_176, var_3: var_177, var_4: var_178, var_5: var_179, var_6: var_180, var_7: var_181, var_8: var_182, var_9: var_183, var_10: var_184, var_11: var_185, var_12: var_186, var_13: var_187, var_14: var_53, var_15: var_148, var_16: var_188, var_17: var_189, var_3: var_190, var_4: var_191, var_5: var_192, var_6: var_193, var_7: var_194, var_8: var_195, var_9: var_196, var_10: var_197, var_11: var_198, var_12: var_199, var_13: var_200, var_14: var_53, var_15: var_148, var_16: var_201, var_17: var_202, var_3: var_203, var_4: var_204, var_5: var_205, var_6: var_206, var_7: var_207, var_8: var_208, var_9: var_209, var_10: var_210, var_11: var_211, var_12: var_212, var_13: var_213, var_14: var_53, var_15: var_148, var_16: var_214, var_17: var_215, var_3: var_216, var_4: var_217, var_5: var_218, var_6: var_219, var_7: var_220, var_8: var_221, var_9: var_222, var_10: var_223, var_11: var_224, var_12: var_225, var_13: var_226, var_14: var_53, var_15: var_148, var_16: var_227, var_17: var_228, var_3: var_229, var_4: var_230, var_5: var_231, var_6: var_232, var_7: var_233, var_8: var_234, var_9: var_235, var_10: var_236, var_11: var_237, var_12: var_238, var_13: var_239, var_14: var_53, var_15: var_148, var_16: var_240, var_17: var_241, var_3: var_242, var_4: var_243, var_5: var_244, var_6: var_245, var_7: var_246, var_8: var_247, var_9: var_248, var_10: var_249, var_11: var_250, var_12: var_251, var_13: var_252, var_14: var_53, var_15: var_148, var_16: var_253, var_17: var_254, var_3: var_255, var_4: var_256, var_5: var_257, var_6: var_258, var_7: var_259, var_8: var_260, var_9: var_261, var_10: var_262, var_11: var_263, var_12: var_264, var_13: var_265, var_14: var_53, var_15: var_148, var_16: var_266, var_17: var_267, var_3: var_268, var_4: var_269, var_5: var_270, var_6: var_271, var_7: var_272, var_8: var_273, var_9: var_274, var_10: var_275, var_11: var_276, var_12: var_277, var_13: var_278, var_14: var_53, var_15: var_148, var_16: var_279, var_17: var_280, var_3: var_281, var_4: var_282, var_5: var_283, var_6: var_284, var_7: var_285, var_8: var_286, var_9: var_287, var_10: var_288, var_11: var_289, var_12: var_290, var_13: var_291, var_14: var_53, var_15: var_148, var_16: var_292, var_17: var_293, var_3: var_294, var_4: var_295, var_5: var_296, var_6: var_297, var_7: var_298, var_8: var_299, var_9: var_300, var_10: var_301, var_11: var_302, var_12: var_303, var_13: var_304, var_14: var_53, var_15: var_148, var_16: var_305, var_17: var_306, var_3: var_307, var_4: var_308, var_5: var_309, var_6: var_310, var_7: var_311, var_8: var_312, var_9: var_313, var_10: var_314, var_11: var_315, var_12: var_316, var_13: var_317, var_14: var_53, var_15: var_148, var_16: var_318, var_17: var_319, var_3: var_320, var_4: var_321, var_5: var_322, var_6: var_323, var_7: var_324, var_8: var_325, var_9: var_326, var_10: var_327, var_11: var_328, var_12: var_329, var_13: var_330, var_14: var_53, var_15: var_148, var_16: var_331, var_17: var_332, var_3: var_333, var_4: var_334, var_5: var_335, var_6: var_336, var_7: var_337, var_8: var_338, var_9: var_339, var_10: var_340, var_11: var_341, var_12: var_342, var_13: var_343, var_14: var_53, var_15: var_148, var_16: var_344, var_17: var_345, var_3: var_346, var_4: var_347, var_5: var_348, var_6: var_349, var_7: var_350, var_8: var_351, var_9: var_352, var_10: var_353, var_11: var_354, var_12: var_355, var_13: var_356, var_14: var_53, var_15: var_148, var_16: var_357, var_17: var_358, var_3: var_359, var_4: var_360, var_5: var_361, var_6: var_362, var_7: var_363, var_8: var_364, var_9: var_365, var_10: var_366, var_11: var_367, var_12: var_368, var_13: var_369, var_14: var_53, var_15: var_148, var_16: var_370, var_17: var_371, var_3: var_372, var_4: var_373, var_5: var_374, var_6: var_375, var_7: var_376, var_8: var_377, var_18: var_378, var_10: var_379, var_11: var_380, var_12: var_381, var_13: var_382, var_14: var_53, var_15: var_148, var_16: var_383, var_17: var_384, var_3: var_385, var_4: var_386, var_5: var_387, var_6: var_388, var_7: var_389, var_8: var_390, var_9: var_391, var_10: var_392, var_11: var_393, var_12: var_394, var_13: var_395, var_14: var_53, var_15: var_148, var_16: var_396, var_17: var_397, var_3: var_398, var_4: var_399, var_5: var_400, var_6: var_401, var_7: var_402, var_8: var_403, var_9: var_404, var_10: var_405, var_11: var_406, var_12: var_407, var_13: var_408, var_14: var_53, var_15: var_148, var_16: var_409, var_17: var_410, var_3: var_411, var_4: var_412, var_5: var_413, var_6: var_414, var_7: var_415, var_8: var_416, var_9: var_417, var_10: var_418, var_11: var_419, var_12: var_420, var_13: var_421, var_14: var_53, var_15: var_148, var_16: var_422, var_17: var_423, var_3: var_424, var_4: var_425, var_5: var_426, var_6: var_427, var_7: var_428, var_8: var_429, var_9: var_430, var_10: var_431, var_11: var_432, var_12: var_433, var_13: var_434, var_14: var_53, var_15: var_148, var_16: var_435, var_17: var_436, var_3: var_437, var_4: var_438, var_5: var_439, var_6: var_440, var_7: var_441, var_8: var_442, var_9: var_443, var_10: var_444, var_11: var_445, var_12: var_446, var_13: var_447, var_14: var_53, var_15: var_148, var_16: var_448, var_17: var_449, var_3: var_450, var_4: var_451, var_5: var_452, var_6: var_453, var_7: var_454, var_8: var_455, var_9: var_456, var_10: var_457, var_11: var_458, var_12: var_459, var_13: var_460, var_14: var_53, var_15: var_148, var_16: var_461, var_17: var_462, var_3: var_463, var_4: var_464, var_5: var_465, var_6: var_466, var_7: var_467, var_8: var_468, var_9: var_469, var_10: var_470, var_11: var_471, var_12: var_472, var_13: var_473, var_19: var_474}



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = '/tmp/project/skip_me'
    var_1 = '/tmp/project/forbidden_dir/file.py'
    var_2 = '/tmp/project/temp_file.tmp'
    var_3 = '/tmp/project/test_logic.py'
    var_4 = '/tmp/project/src/main.py'
    var_5 = '/tmp/project/ghost.py'
    var_6 = '/tmp/project/subdir/test_data.tmp'
    var_7 = '/tmp/project/tracked.py'
    var_8 = '/tmp/project/untracked.py'
    var_9 = '/tmp/project/.git'



# Parsed testcases at query #2
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = '\n    Tests the find_all_configs function by creating a temporary directory structure\n    with various configuration files and verifying they are correctly inserted into the Trie.\n    '
    var_1 = 'project_root'
    var_2 = 'sub_dir'
    var_3 = 'empty_dir'
    var_4 = 'pyproject.toml'
    var_5 = '[tool.isort]\nline_length = 88'
    var_6 = '.isort.cfg'
    var_7 = '[settings]\nprofile = black'
    var_8 = 'line_length'
    var_9 = 88
    var_10 = {var_8: var_9}
    var_11 = 'profile'
    var_12 = 'black'
    var_13 = {var_11: var_12}
    var_14 = module_0.find_all_configs(var_0)
    var_15 = []



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = '/tmp/project/skip_me'

def test_case_0():
    var_0 = '/tmp/project/ignored_dir/file.py'

def test_case_0():
    var_0 = '/tmp/project/test_file.tmp'
    var_1 = '/tmp/project/temp_data.txt'

def test_case_0():
    var_0 = '/tmp/project/src/main.py'
    assert var_0 is False

def test_case_0():
    var_0 = '/tmp/project/ghost.py'
    assert var_0 is True

def test_case_0():
    var_0 = '/tmp/project'
    assert var_0 is True
    assert var_0 is False
    var_1 = '/tmp/project/src/main.py'
    var_2 = {var_1}
    var_3 = '/tmp/project/src/untracked.py'

def test_case_0():
    var_0 = '/tmp/project'
    assert var_0 is True
    assert var_0 is False
    var_1 = '/tmp/project/src/main.py'
    var_2 = {var_1}
    var_3 = '/tmp/project/src/untracked.py'



# Parsed testcases at query #4
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'py_version'
    var_1 = 'line_length'
    var_2 = 'other_setting'
    var_3 = 'py310'
    var_4 = 88
    var_5 = 'value'
    var_6 = 'line_length'
    var_7 = 'new_setting'
    var_8 = 100
    var_9 = 'extra'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'line_length'
    var_12 = 'source'
    var_13 = 79
    var_14 = 'test_file'
    var_15 = '/tmp/pyproject.toml'
    var_16 = module_0.Config(var_15)
    var_17 = '/abs/path/to/config'
    var_18 = module_0.Config(settings_path=var_17)
    var_19 = 'black'
    var_20 = module_0.Config()
    var_21 = 'mock_get_data'
    var_22 = locals()
    var_23 = var_21 in var_22
    var_24 = []
    var_25 = 'non_existent_profile'
    var_26 = module_0.Config()
    var_27 = '/invalid/path'
    var_28 = module_0.Config(settings_path=var_27)
    var_29 = '4'
    var_30 = module_0.Config()
    var_31 = 'tab'
    var_32 = module_0.Config()
    var_33 = '\t'
    var_34 = module_0.Config()
    var_35 = 'my_module'
    var_36 = [var_35]
    var_37 = module_0.Config()



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = '\n    Tests find_all_configs by creating a temporary directory structure with \n    various config files and verifying the resulting Trie contains the expected data.\n    '
    var_1 = 'root'
    var_2 = 'sub_dir'
    var_3 = 'empty_dir'
    var_4 = '.isort.cfg'
    var_5 = 'pyproject.toml'
    var_6 = 'line_length'
    var_7 = 88
    var_8 = {var_6: var_7}
    var_9 = 'profile'
    var_10 = 'black'
    var_11 = {var_9: var_10}
    var_12 = 'setup.cfg'
    var_13 = [var_4, var_5, var_12]
    var_14 = [var_1, var_2]
    var_15 = '.isort.cfg'
    var_16 = [var_15]
    var_17 = (var_0, var_14, var_16)
    var_18 = []
    var_19 = 'pyproject.toml'
    var_20 = [var_19]
    var_21 = (var_4, var_18, var_20)
    var_22 = []
    var_23 = []
    var_24 = (var_10, var_22, var_23)



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the is_skipped method of the Config class covering various skip scenarios:\n    - Explicit path skips\n    - Parent folder skips\n    - Glob pattern matches\n    - File existence/type checks\n    - Gitignore-based skips (when enabled)\n    '
    var_1 = '/tmp/project/skip_me'
    var_2 = 'ignored_dir'
    assert var_2 is True
    assert var_2 is False
    var_3 = [var_1, var_2]
    var_4 = '*.tmp'
    var_5 = 'temp_*'
    assert var_5 is True
    assert var_5 is False
    var_6 = [var_4, var_5]
    var_7 = '/tmp/project/skip_me'
    var_8 = '/tmp/project/skip_me'
    var_9 = '/tmp/project/ignored_dir/file.py'
    var_10 = '/tmp/project/file.tmp'
    var_11 = '/tmp/project/valid_file.py'
    var_12 = '/tmp/project/non_existent.py'
    var_13 = '/tmp/project'
    var_14 = '/tmp/project/tracked.py'
    var_15 = {var_14}
    var_16 = '/tmp/project/untracked.py'
    var_17 = '/tmp/project/tracked.py'
    var_18 = '/tmp/project/.git/config'



# Parsed testcases at query #7
#--------------------------




# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = '.py'
    var_1 = '.pyi'
    assert var_1 is False
    var_2 = '.txt'
    var_3 = '.md'
    var_4 = 'script.py'
    var_5 = 'type_hint.pyi'
    var_6 = 'readme.txt'
    var_7 = 'notes.md'
    var_8 = 'script.py~'
    var_9 = 'pipe_file.py'
    var_10 = 'unreadable.py'
    var_11 = 'non_existent.py'



