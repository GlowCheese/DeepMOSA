# Parsed testcases at query #4
#--------------------------


import isort.main as module_0

def test_case_0():
    var_0 = '--line-length'
    var_1 = '88'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = '--length-sort'
    var_5 = [var_4]
    var_6 = module_0.parse_args(var_5)
    var_7 = '-m'
    var_8 = '3'
    var_9 = [var_7, var_8]
    var_10 = module_0.parse_args(var_9)
    var_11 = 'vertical'
    var_12 = [var_7, var_11]
    var_13 = module_0.parse_args(var_12)
    var_14 = '--dont-order-by-type'
    var_15 = [var_14]
    var_16 = module_0.parse_args(var_15)
    var_17 = '--known-first-party'
    var_18 = 'module1'
    var_19 = 'module2'
    var_20 = [var_17, var_18, var_17, var_19]
    var_21 = module_0.parse_args(var_20)
    var_22 = '-k'
    var_23 = [var_22]
    var_24 = module_0.parse_args(var_23)
    var_25 = '--float-to-top'
    var_26 = '--dont-float-to-top'
    var_27 = [var_25, var_26]
    var_28 = module_0.parse_args(var_27)
    var_29 = []
    var_30 = module_0.parse_args(var_29)
    var_31 = module_0.parse_args()



# Parsed testcases at query #5
#--------------------------


import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)
    var_2 = '--length-sort'
    var_3 = [var_2]
    var_4 = module_0.parse_args(var_3)
    var_5 = '--indent'
    var_6 = '  '
    var_7 = [var_5, var_6]
    var_8 = module_0.parse_args(var_7)
    var_9 = [var_2, var_5, var_6]
    var_10 = module_0.parse_args(var_9)
    var_11 = '-rc'
    var_12 = [var_11]
    var_13 = module_0.parse_args(var_12)
    var_14 = '-m'
    var_15 = '3'
    var_16 = [var_14, var_15]
    var_17 = module_0.parse_args(var_16)
    var_18 = 'multi_line_output'
    var_19 = 3
    var_20 = 'VERTICAL_HANGING'
    var_21 = [var_14, var_20]
    var_22 = module_0.parse_args(var_21)
    var_23 = '--dont-order-by-type'
    var_24 = [var_23]
    var_25 = module_0.parse_args(var_24)
    var_26 = '--float-to-top'
    var_27 = '--dont-float-to-top'
    var_28 = [var_26, var_27]
    var_29 = module_0.parse_args(var_28)
    var_30 = '-l'
    var_31 = '88'
    var_32 = [var_30, var_31]
    var_33 = module_0.parse_args(var_32)
    var_34 = '--wrap-length'
    var_35 = '79'
    var_36 = [var_34, var_35]
    var_37 = module_0.parse_args(var_36)
    var_38 = '-b'
    var_39 = 'os'
    var_40 = [var_38, var_39]
    var_41 = module_0.parse_args(var_40)
    var_42 = '-o'
    var_43 = 'django'
    var_44 = [var_42, var_43]
    var_45 = module_0.parse_args(var_44)
    var_46 = '-p'
    var_47 = 'myproject'
    var_48 = [var_46, var_47]
    var_49 = module_0.parse_args(var_48)
    var_50 = '--py'
    var_51 = '38'
    var_52 = [var_50, var_51]
    var_53 = module_0.parse_args(var_52)
    var_54 = '--force-grid-wrap'
    var_55 = [var_54, var_15]
    var_56 = module_0.parse_args(var_55)
    var_57 = '--no-sections'
    var_58 = [var_57]
    var_59 = module_0.parse_args(var_58)
    var_60 = '--only-sections'
    var_61 = [var_60]
    var_62 = module_0.parse_args(var_61)
    var_63 = '--force-alphabetical-sort'
    var_64 = [var_63]
    var_65 = module_0.parse_args(var_64)
    var_66 = '--force-sort-within-sections'
    var_67 = [var_66]
    var_68 = module_0.parse_args(var_67)
    var_69 = '--force-alphabetical-sort-within-sections'
    var_70 = [var_69]
    var_71 = module_0.parse_args(var_70)
    var_72 = '--combine-straight-imports'
    var_73 = [var_72]
    var_74 = module_0.parse_args(var_73)
    var_75 = '--no-lines-before'
    var_76 = 'STDLIB'
    var_77 = [var_75, var_76]
    var_78 = module_0.parse_args(var_77)
    var_79 = '--src-path'
    var_80 = 'src'
    var_81 = [var_79, var_80]
    var_82 = module_0.parse_args(var_81)
    var_83 = '-f'
    var_84 = 'future_module'
    var_85 = [var_83, var_84]
    var_86 = module_0.parse_args(var_85)
    var_87 = '--known-local-folder'
    var_88 = 'local_folder'
    var_89 = [var_87, var_88]
    var_90 = module_0.parse_args(var_89)
    var_91 = '--virtual-env'
    var_92 = 'env'
    var_93 = [var_91, var_92]
    var_94 = module_0.parse_args(var_93)
    var_95 = '--conda-env'
    var_96 = 'conda_env'
    var_97 = [var_95, var_96]
    var_98 = module_0.parse_args(var_97)
    var_99 = '--color'
    var_100 = [var_99]
    var_101 = module_0.parse_args(var_100)
    var_102 = '--formatter'
    var_103 = 'black'
    var_104 = [var_102, var_103]
    var_105 = module_0.parse_args(var_104)
    var_106 = '--treat-comment-as-code'
    var_107 = '# noqa'
    var_108 = [var_106, var_107]
    var_109 = module_0.parse_args(var_108)
    var_110 = '--treat-all-comment-as-code'
    var_111 = [var_110]
    var_112 = module_0.parse_args(var_111)
    var_113 = '--honor-noqa'
    var_114 = [var_113]
    var_115 = module_0.parse_args(var_114)
    var_116 = '--remove-redundant-aliases'
    var_117 = [var_116]
    var_118 = module_0.parse_args(var_117)
    var_119 = '--case-sensitive'
    var_120 = [var_119]
    var_121 = module_0.parse_args(var_120)
    var_122 = '--use-parentheses'
    var_123 = [var_122]
    var_124 = module_0.parse_args(var_123)
    var_125 = '--trailing-comma'
    var_126 = [var_125]
    var_127 = module_0.parse_args(var_126)
    var_128 = '--force-single-line-imports'
    var_129 = [var_128]
    var_130 = module_0.parse_args(var_129)
    var_131 = '--single-line-exclusions'
    var_132 = [var_131, var_39]
    var_133 = module_0.parse_args(var_132)
    var_134 = '--reverse-sort'
    var_135 = [var_134]
    var_136 = module_0.parse_args(var_135)
    var_137 = '--reverse-relative'
    var_138 = [var_137]
    var_139 = module_0.parse_args(var_138)
    var_140 = '--star-first'
    var_141 = [var_140]
    var_142 = module_0.parse_args(var_141)
    var_143 = '--split-on-trailing-comma'
    var_144 = [var_143]
    var_145 = module_0.parse_args(var_144)
    var_146 = '--section-default'
    var_147 = 'THIRDPARTY'
    var_148 = [var_146, var_147]
    var_149 = module_0.parse_args(var_148)
    var_150 = '-t'
    var_151 = [var_150, var_39]
    var_152 = module_0.parse_args(var_151)
    var_153 = '--lines-before-imports'
    var_154 = '2'
    var_155 = [var_153, var_154]
    var_156 = module_0.parse_args(var_155)
    var_157 = '--lines-after-imports'
    var_158 = [var_157, var_154]
    var_159 = module_0.parse_args(var_158)
    var_160 = '--lines-between-types'
    var_161 = '1'
    var_162 = [var_160, var_161]
    var_163 = module_0.parse_args(var_162)
    var_164 = '--line-ending'
    var_165 = 'LF'
    var_166 = [var_164, var_165]
    var_167 = module_0.parse_args(var_166)
    var_168 = '--length-sort-straight'
    var_169 = [var_168]
    var_170 = module_0.parse_args(var_169)
    var_171 = '-n'
    var_172 = [var_171]
    var_173 = module_0.parse_args(var_172)
    var_174 = '--no-inline-sort'
    var_175 = [var_174]
    var_176 = module_0.parse_args(var_175)
    var_177 = '--order-by-type'
    var_178 = [var_177]
    var_179 = module_0.parse_args(var_178)
    var_180 = '--sort-order'
    var_181 = 'natural'
    var_182 = [var_180, var_181]
    var_183 = module_0.parse_args(var_182)
    var_184 = '--ext-format'
    var_185 = 'py'
    var_186 = [var_184, var_185]
    var_187 = module_0.parse_args(var_186)
    var_188 = '--extra-builtin'
    var_189 = 'extra_module'
    var_190 = [var_188, var_189]
    var_191 = module_0.parse_args(var_190)
    var_192 = '--honor-case-in-force-sorted-sections'
    var_193 = [var_192]
    var_194 = module_0.parse_args(var_193)
    var_195 = '--sort-relative-in-force-sorted-sections'
    var_196 = [var_195]
    var_197 = module_0.parse_args(var_196)



# Parsed testcases at query #6
#--------------------------


import isort.main as module_0

def test_case_0():
    var_0 = '-'
    var_1 = [var_0]
    var_2 = module_0.identify_imports_main(var_1)
    var_3 = 'import os'
    var_4 = 'import sys'
    var_5 = 'file1.py'
    var_6 = 'file2.py'
    var_7 = [var_5, var_6]
    var_8 = module_0.identify_imports_main(var_7)
    var_9 = 'import os'
    var_10 = 'import sys'
    var_11 = 'file1.py'
    var_12 = '--top-only'
    var_13 = [var_11, var_12]
    var_14 = module_0.identify_imports_main(var_13)
    var_15 = 'import os'
    var_16 = 'file1.py'
    var_17 = '--unique'
    var_18 = [var_16, var_17]
    var_19 = module_0.identify_imports_main(var_18)
    var_20 = 'import os.path'
    var_21 = 'import sys.path'
    var_22 = 'file1.py'
    var_23 = '--packages'
    var_24 = [var_22, var_23]
    var_25 = module_0.identify_imports_main(var_24)
    var_26 = 'import os.path'
    var_27 = 'import sys.path'
    var_28 = 'file1.py'
    var_29 = '--modules'
    var_30 = [var_28, var_29]
    var_31 = module_0.identify_imports_main(var_30)
    var_32 = 'from os import path'
    var_33 = 'from sys import path'
    var_34 = 'file1.py'
    var_35 = '--attributes'
    var_36 = [var_34, var_35]
    var_37 = module_0.identify_imports_main(var_36)



# Parsed testcases at query #7
#--------------------------


import isort.main as module_0

def test_case_0():
    var_0 = '-'
    var_1 = [var_0]
    var_2 = module_0.identify_imports_main(var_1)
    var_3 = 'import os\nimport sys\nfrom collections import defaultdict'
    var_4 = [var_3]
    var_5 = module_0.identify_imports_main(var_4)
    var_6 = 'import os\nimport os\nimport sys'
    var_7 = '--unique'
    var_8 = [var_6, var_7]
    var_9 = module_0.identify_imports_main(var_8)
    var_10 = 'import os.path\nimport sys.platform\nfrom collections import defaultdict'
    var_11 = '--packages'
    var_12 = [var_10, var_11]
    var_13 = module_0.identify_imports_main(var_12)
    var_14 = 'import os.path\nimport sys.platform\nfrom collections import defaultdict'
    var_15 = '--modules'
    var_16 = [var_14, var_15]
    var_17 = module_0.identify_imports_main(var_16)
    var_18 = 'import os.path\nimport sys.platform\nfrom collections import defaultdict'
    var_19 = '--attributes'
    var_20 = [var_18, var_19]
    var_21 = module_0.identify_imports_main(var_20)
    var_22 = 'import os\n\ndef func():\n    import sys'
    var_23 = '--top-only'
    var_24 = [var_22, var_23]
    var_25 = module_0.identify_imports_main(var_24)



# Parsed testcases at query #8
#--------------------------


import isort.main as module_0

def test_case_0():
    var_0 = '--line-length'
    var_1 = '88'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = '--length-sort'
    var_5 = [var_4]
    var_6 = module_0.parse_args(var_5)
    var_7 = '-m'
    var_8 = '3'
    var_9 = [var_7, var_8]
    var_10 = module_0.parse_args(var_9)
    var_11 = '--multi-line'
    var_12 = 'vertical-hanging'
    var_13 = [var_11, var_12]
    var_14 = module_0.parse_args(var_13)
    var_15 = '-k'
    var_16 = [var_15]
    var_17 = module_0.parse_args(var_16)
    var_18 = '--dont-order-by-type'
    var_19 = [var_18]
    var_20 = module_0.parse_args(var_19)
    var_21 = '--dont-float-to-top'
    var_22 = [var_21]
    var_23 = module_0.parse_args(var_22)
    var_24 = '--float-to-top'
    var_25 = '--dont-float-to-top'
    var_26 = [var_24, var_25]
    var_27 = module_0.parse_args(var_26)
    var_28 = '--known-first-party'
    var_29 = 'my_module'
    var_30 = 'another_module'
    var_31 = [var_28, var_29, var_28, var_30]
    var_32 = module_0.parse_args(var_31)
    var_33 = []
    var_34 = module_0.parse_args(var_33)
    var_35 = '120'
    var_36 = 'vertical'
    var_37 = '--order-by-type'
    var_38 = '--known-third-party'
    var_39 = 'django'
    var_40 = 'my_project'
    var_41 = [var_24, var_35, var_11, var_36, var_37, var_38, var_39, var_28, var_40]
    var_42 = module_0.parse_args(var_41)



# Parsed testcases at query #9
#--------------------------


import isort.main as module_0

def test_case_0():
    var_0 = '--line-length'
    var_1 = '88'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = '--length-sort'
    var_5 = [var_4]
    var_6 = module_0.parse_args(var_5)
    var_7 = '-m'
    var_8 = '3'
    var_9 = [var_7, var_8]
    var_10 = module_0.parse_args(var_9)
    var_11 = 'vertical-hanging'
    var_12 = [var_7, var_11]
    var_13 = module_0.parse_args(var_12)
    var_14 = '-rc'
    var_15 = [var_14]
    var_16 = module_0.parse_args(var_15)
    var_17 = '--float-to-top'
    var_18 = '--dont-float-to-top'
    var_19 = [var_17, var_18]
    var_20 = module_0.parse_args(var_19)
    var_21 = '--dont-order-by-type'
    var_22 = [var_21]
    var_23 = module_0.parse_args(var_22)
    var_24 = '--dont-follow-links'
    var_25 = [var_24]
    var_26 = module_0.parse_args(var_25)
    var_27 = '--known-first-party'
    var_28 = 'module1'
    var_29 = 'module2'
    var_30 = [var_27, var_28, var_27, var_29]
    var_31 = module_0.parse_args(var_30)
    var_32 = []
    var_33 = module_0.parse_args(var_32)
    var_34 = '--invalid-arg'
    var_35 = [var_34]
    var_36 = module_0.parse_args(var_35)



# Parsed testcases at query #10
#--------------------------


import isort.main as module_0

def test_case_0():
    var_0 = 'sys.stdin'
    var_1 = 'import os'
    var_2 = 'import sys'
    var_3 = [var_1, var_2]
    var_4 = '-'
    var_5 = [var_4]
    var_6 = module_0.identify_imports_main(var_5)
    var_7 = 'import os\nimport sys'
    var_8 = [var_3]
    var_9 = module_0.identify_imports_main(var_8)
    var_10 = 'import os\ndef func():\n    import sys'
    var_11 = '--top-only'
    var_12 = [var_3, var_11]
    var_13 = module_0.identify_imports_main(var_12)
    var_14 = 'import os\nimport os'
    var_15 = '--unique'
    var_16 = [var_3, var_15]
    var_17 = module_0.identify_imports_main(var_16)
    var_18 = 'import os.path\nimport sys'
    var_19 = '--packages'
    var_20 = [var_3, var_19]
    var_21 = module_0.identify_imports_main(var_20)
    var_22 = 'from os import path\nimport sys'
    var_23 = '--modules'
    var_24 = [var_3, var_23]
    var_25 = module_0.identify_imports_main(var_24)
    var_26 = 'from os import path\nimport sys'
    var_27 = '--attributes'
    var_28 = [var_3, var_27]
    var_29 = module_0.identify_imports_main(var_28)



# Parsed testcases at query #11
#--------------------------


import isort.main as module_0

def test_case_0():
    var_0 = '--line-length'
    var_1 = '88'
    var_2 = '--indent'
    var_3 = '  '
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.parse_args(var_4)
    var_6 = '--length-sort'
    var_7 = '--use-parentheses'
    var_8 = [var_6, var_7]
    var_9 = module_0.parse_args(var_8)
    var_10 = '-m'
    var_11 = '3'
    var_12 = [var_10, var_11]
    var_13 = module_0.parse_args(var_12)
    var_14 = 3
    var_15 = 'VERTICAL_HANGING'
    var_16 = [var_10, var_15]
    var_17 = module_0.parse_args(var_16)
    var_18 = '--recursive'
    var_19 = [var_18]
    var_20 = module_0.parse_args(var_19)
    var_21 = '--dont-order-by-type'
    var_22 = [var_21]
    var_23 = module_0.parse_args(var_22)
    var_24 = '--dont-float-to-top'
    var_25 = [var_24]
    var_26 = module_0.parse_args(var_25)
    var_27 = '--float-to-top'
    var_28 = '--dont-float-to-top'
    var_29 = [var_27, var_28]
    var_30 = module_0.parse_args(var_29)
    var_31 = '--single-line-exclusions'
    var_32 = 'os'
    var_33 = 'sys'
    var_34 = [var_31, var_32, var_31, var_33]
    var_35 = module_0.parse_args(var_34)
    var_36 = []
    var_37 = module_0.parse_args(var_36)
    var_38 = module_0.parse_args()



# Parsed testcases at query #12
#--------------------------


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)
    var_4 = 'test.py'
    var_5 = True
    var_6 = module_1.sort_imports(var_4, var_0, var_5)
    var_7 = 'test.py'
    var_8 = module_1.sort_imports(var_7, var_0)
    var_9 = 'test.py'
    var_10 = module_1.sort_imports(var_9, var_0)
    assert var_10 is None
    var_11 = 'test.py'
    var_12 = module_1.sort_imports(var_11, var_0)
    var_13 = 'test.py'
    var_14 = module_1.sort_imports(var_13, var_0)
    var_15 = 1
    var_16 = 'test.py'
    var_17 = module_1.sort_imports(var_16, var_0)



# Parsed testcases at query #13
#--------------------------


import isort.main as module_0

def test_case_0():
    var_0 = '--line-length'
    var_1 = '88'
    var_2 = '--indent'
    var_3 = '    '
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.parse_args(var_4)
    var_6 = '--order-by-type'
    var_7 = '--reverse-sort'
    var_8 = [var_6, var_7]
    var_9 = module_0.parse_args(var_8)
    var_10 = '-m'
    var_11 = '3'
    var_12 = [var_10, var_11]
    var_13 = module_0.parse_args(var_12)
    var_14 = 3
    var_15 = 'VERTICAL_HANGING'
    var_16 = [var_10, var_15]
    var_17 = module_0.parse_args(var_16)
    var_18 = '--dont-order-by-type'
    var_19 = [var_18]
    var_20 = module_0.parse_args(var_19)
    var_21 = '--single-line-exclusions'
    var_22 = 'os'
    var_23 = 'sys'
    var_24 = [var_21, var_22, var_21, var_23]
    var_25 = module_0.parse_args(var_24)
    var_26 = '79'
    var_27 = '  '
    var_28 = '--multi-line'
    var_29 = '2'
    var_30 = [var_0, var_26, var_2, var_27, var_6, var_7, var_21, var_22, var_28, var_29]
    var_31 = module_0.parse_args(var_30)
    var_32 = 2
    var_33 = []
    var_34 = module_0.parse_args(var_33)
    var_35 = module_0.parse_args()
    var_36 = module_0.parse_args()



# Parsed testcases at query #14
#--------------------------


import isort.settings as module_0
import isort.main as module_1
import locale as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)
    var_4 = 'test.py'
    var_5 = True
    var_6 = module_1.sort_imports(var_4, var_0, var_5)
    var_7 = 'test.py'
    var_8 = True
    var_9 = module_1.sort_imports(var_7, var_0, var_8)
    var_10 = 'test.py'
    var_11 = module_1.sort_imports(var_10, var_0)
    var_12 = False
    var_13 = 'test.py'
    var_14 = module_1.sort_imports(var_13, var_0)
    var_15 = False
    var_16 = 'test.py'
    var_17 = module_1.sort_imports(var_16, var_0)
    var_18 = False
    var_19 = 'test.py'
    var_20 = module_1.sort_imports(var_19, var_0)
    var_21 = 'Encoding not supported for test.py'
    var_22 = 2
    var_23 = 'test.py'
    var_24 = module_1.sort_imports(var_23, var_0)
    assert var_24 is None
    var_25 = 'Unable to parse file test.py due to Test error'
    var_26 = 2
    var_27 = 'test.py'
    var_28 = module_1.sort_imports(var_27, var_0)
    assert var_28 is None
    var_29 = 'Unable to parse file test.py due to Test error'
    var_30 = 2
    var_31 = 'test.py'
    var_32 = module_1.sort_imports(var_31, var_0)
    var_33 = 'Test error'
    var_34 = 'test.py'
    var_35 = module_1.sort_imports(var_34, var_0)
    var_36 = module_2.str(var_34)
    assert var_36 == 'Unexpected error'
    var_37 = 'test.py'



# Parsed testcases at query #15
#--------------------------


import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)
    var_2 = '--force-grid-wrap'
    var_3 = [var_2]
    var_4 = module_0.parse_args(var_3)
    var_5 = '3'
    var_6 = [var_2, var_5]
    var_7 = module_0.parse_args(var_6)
    var_8 = '--indent'
    var_9 = '  '
    var_10 = '--lines-before-imports'
    var_11 = '2'
    var_12 = [var_2, var_5, var_8, var_9, var_10, var_11]
    var_13 = module_0.parse_args(var_12)
    var_14 = '--length-sort'
    var_15 = [var_14]
    var_16 = module_0.parse_args(var_15)
    var_17 = '--reverse-sort'
    var_18 = [var_17]
    var_19 = module_0.parse_args(var_18)
    var_20 = '-m'
    var_21 = [var_20, var_5]
    var_22 = module_0.parse_args(var_21)
    var_23 = 'multi_line_output'
    var_24 = 3
    var_25 = '--multi-line'
    var_26 = 'vertical'
    var_27 = [var_25, var_26]
    var_28 = module_0.parse_args(var_27)
    var_29 = '-rc'
    var_30 = [var_29]
    var_31 = module_0.parse_args(var_30)
    var_32 = '--recursive'
    var_33 = [var_32]
    var_34 = module_0.parse_args(var_33)
    var_35 = '--order-by-type'
    var_36 = [var_35]
    var_37 = module_0.parse_args(var_36)
    var_38 = '--dont-order-by-type'
    var_39 = [var_38]
    var_40 = module_0.parse_args(var_39)
    var_41 = '--float-to-top'
    var_42 = [var_41]
    var_43 = module_0.parse_args(var_42)
    var_44 = '--dont-float-to-top'
    var_45 = [var_44]
    var_46 = module_0.parse_args(var_45)
    var_47 = '--follow-links'
    var_48 = [var_47]
    var_49 = module_0.parse_args(var_48)
    var_50 = '--dont-follow-links'
    var_51 = [var_50]
    var_52 = module_0.parse_args(var_51)
    var_53 = '--single-line-exclusions'
    var_54 = 'module1'
    var_55 = [var_53, var_54]
    var_56 = module_0.parse_args(var_55)
    var_57 = 'module2'
    var_58 = [var_53, var_54, var_53, var_57]
    var_59 = module_0.parse_args(var_58)
    var_60 = '4'
    var_61 = '    '
    var_62 = 'os'
    var_63 = 'sys'
    var_64 = '--line-length'
    var_65 = '88'
    var_66 = [var_2, var_60, var_8, var_61, var_14, var_25, var_26, var_53, var_62, var_53, var_63, var_35, var_64, var_65]
    var_67 = module_0.parse_args(var_66)



# Parsed testcases at query #16
#--------------------------


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = module_0.Config()
    var_2 = True
    var_3 = module_1.sort_imports(var_0, var_1, var_2)
    var_4 = 'test.py'
    var_5 = module_0.Config()
    var_6 = True
    var_7 = module_1.sort_imports(var_4, var_5, var_6)
    var_8 = 'test.py'
    var_9 = module_0.Config()
    var_10 = True
    var_11 = module_1.sort_imports(var_8, var_9, var_10)
    var_12 = 'test.py'
    var_13 = module_0.Config()
    var_14 = module_1.sort_imports(var_12, var_13)
    var_15 = 'test.py'
    var_16 = module_0.Config()
    var_17 = module_1.sort_imports(var_15, var_16)
    assert var_17 is None
    var_18 = 'test.py'
    var_19 = module_0.Config()
    var_20 = module_1.sort_imports(var_18, var_19)
    assert var_20 is None
    var_21 = 'test.py'
    var_22 = module_0.Config()
    var_23 = module_1.sort_imports(var_21, var_22)
    var_24 = 'test.py'
    var_25 = module_0.Config()
    var_26 = module_1.sort_imports(var_24, var_25)



# Parsed testcases at query #17
#--------------------------


import isort.main as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom typing import List\n'
    var_1 = '-'
    var_2 = [var_1]
    var_3 = module_0.identify_imports_main(var_2)
    var_4 = 'import os\nimport sys\nfrom typing import List\n'
    var_5 = [var_4]
    var_6 = module_0.identify_imports_main(var_5)
    var_7 = 'import os\n\ndef foo():\n    import sys\n'
    var_8 = '-'
    var_9 = '--top-only'
    var_10 = [var_8, var_9]
    var_11 = module_0.identify_imports_main(var_10)
    var_12 = 'import os\nimport os\nimport sys\n'
    var_13 = '-'
    var_14 = '--unique'
    var_15 = [var_13, var_14]
    var_16 = module_0.identify_imports_main(var_15)
    var_17 = 'import os.path\nimport sys\nfrom typing import List\n'
    var_18 = '-'
    var_19 = '--packages'
    var_20 = [var_18, var_19]
    var_21 = module_0.identify_imports_main(var_20)
    var_22 = 'import os.path\nimport sys\nfrom typing import List\n'
    var_23 = '-'
    var_24 = '--modules'
    var_25 = [var_23, var_24]
    var_26 = module_0.identify_imports_main(var_25)
    var_27 = 'import os.path\nimport sys\nfrom typing import List\n'
    var_28 = '-'
    var_29 = '--attributes'
    var_30 = [var_28, var_29]
    var_31 = module_0.identify_imports_main(var_30)



# Parsed testcases at query #18
#--------------------------


import isort.main as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\nimport sys\nfrom collections import defaultdict\n'
    var_2 = [var_0]
    var_3 = module_0.identify_imports_main(var_2)
    var_4 = 'import os\n\ndef foo():\n    import sys\n'
    var_5 = '--top-only'
    var_6 = [var_0, var_5]
    var_7 = module_0.identify_imports_main(var_6)
    var_8 = 'import os\nimport os\nimport sys\n'
    var_9 = '--unique'
    var_10 = [var_0, var_9]
    var_11 = module_0.identify_imports_main(var_10)
    var_12 = 'import os'
    var_13 = 'import os.path\nimport sys\nfrom collections import defaultdict\n'
    var_14 = '--packages'
    var_15 = [var_0, var_14]
    var_16 = module_0.identify_imports_main(var_15)
    var_17 = '--modules'
    var_18 = [var_0, var_17]
    var_19 = module_0.identify_imports_main(var_18)
    var_20 = '--attributes'
    var_21 = [var_0, var_20]
    var_22 = module_0.identify_imports_main(var_21)
    var_23 = 'import os\nimport sys\n'
    var_24 = '-'
    var_25 = [var_24]



