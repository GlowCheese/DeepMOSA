####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.main as module_0

def test_case_0():
    var_0 = 'import sys\nfrom os import path\n'
    var_1 = '-'
    var_2 = [var_1]
    var_3 = module_0.identify_imports_main(var_2)
    var_4 = 'import sys\nfrom os import path\n'
    var_5 = [var_4]
    var_6 = module_0.identify_imports_main(var_5)
    var_7 = 'import sys\nfrom os import path\n\ndef foo():\n    import json\n'
    var_8 = '--top-only'
    var_9 = [var_7, var_8]
    var_10 = module_0.identify_imports_main(var_9)
    var_11 = 'import sys\nimport sys\nfrom os import path\nfrom os import path\n'
    var_12 = '--unique'
    var_13 = [var_11, var_12]
    var_14 = module_0.identify_imports_main(var_13)
    var_15 = 'import sys\nfrom os.path import join\n'
    var_16 = '--packages'
    var_17 = [var_15, var_16]
    var_18 = module_0.identify_imports_main(var_17)
    var_19 = 'import sys\nfrom os.path import join\n'
    var_20 = '--modules'
    var_21 = [var_19, var_20]
    var_22 = module_0.identify_imports_main(var_21)
    var_23 = 'import sys\nfrom os.path import join\n'
    var_24 = '--attributes'
    var_25 = [var_23, var_24]
    var_26 = module_0.identify_imports_main(var_25)
    var_27 = 'import sys\n'
    var_28 = 'from os import path\n'
    var_29 = [var_28, var_24]
    var_30 = module_0.identify_imports_main(var_29)



# Parsed testcases at query #2
#--------------------------


import isort.main as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\nimport sys\nfrom typing import List\n'
    var_2 = [var_0]
    var_3 = module_0.identify_imports_main(var_2)
    var_4 = 'import os\n\ndef foo():\n    import sys\n'
    var_5 = '--top-only'
    var_6 = 'import os\nimport os\nimport sys\n'
    var_7 = '--unique'
    var_8 = 'import os'
    var_9 = 'import os.path\nfrom typing import List, Dict\n'
    var_10 = '--packages'
    var_11 = '--modules'
    var_12 = 'from typing import List, Dict\n'
    var_13 = '--attributes'
    var_14 = 'import os\nimport sys\n'
    var_15 = '-'
    var_16 = [var_15]
    var_17 = 'test2.py'
    var_18 = 'import json\n'



# Parsed testcases at query #3
#--------------------------


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)
    var_4 = module_1.sort_imports(var_1, var_0)
    var_5 = False
    var_6 = 'test.py'
    var_7 = True
    var_8 = module_1.sort_imports(var_6, var_0, var_7)
    var_9 = 'test.py'
    var_10 = module_1.sort_imports(var_9, var_0)
    var_11 = 'test.py'
    var_12 = module_1.sort_imports(var_11, var_0)
    var_13 = 'test.py'
    var_14 = module_1.sort_imports(var_13, var_0)
    assert var_14 is None
    var_15 = 'test.py'
    var_16 = module_1.sort_imports(var_15, var_0)
    assert var_16 is None
    var_17 = 'test.py'
    var_18 = module_1.sort_imports(var_17, var_0)
    var_19 = 1
    var_20 = 'test.py'
    var_21 = module_1.sort_imports(var_20, var_0)
    var_22 = 'test.py'



# Parsed testcases at query #4
#--------------------------




# Parsed testcases at query #5
#--------------------------


import isort.main as module_0

def test_case_0():
    var_0 = module_0.identify_imports_main()
    var_1 = 'os'
    var_2 = 'sys'
    var_3 = module_0.identify_imports_main()
    var_4 = 'os'
    var_5 = 'sys'
    var_6 = module_0.identify_imports_main()
    var_7 = 'os'
    var_8 = module_0.identify_imports_main()
    var_9 = 'os'
    var_10 = module_0.identify_imports_main()
    var_11 = 'os'
    var_12 = 'sys'
    var_13 = module_0.identify_imports_main()
    var_14 = 'os.path'
    var_15 = 'sys.platform'
    var_16 = module_0.identify_imports_main()
    var_17 = 'os.path'
    var_18 = 'sys.platform'



# Parsed testcases at query #6
#--------------------------


import isort.main as module_0

def test_case_0():
    var_0 = '-'
    var_1 = [var_0]
    var_2 = module_0.identify_imports_main(var_1)
    var_3 = 'import os\nimport sys'
    var_4 = [var_3]
    var_5 = module_0.identify_imports_main(var_4)
    var_6 = 'import os\n\ndef foo():\n    import sys'
    var_7 = '--top-only'
    var_8 = [var_6, var_7]
    var_9 = module_0.identify_imports_main(var_8)
    var_10 = 'import os\nimport os\nimport sys'
    var_11 = '--unique'
    var_12 = [var_10, var_11]
    var_13 = module_0.identify_imports_main(var_12)
    var_14 = 'import os.path\nimport sys.platform'
    var_15 = '--packages'
    var_16 = [var_14, var_15]
    var_17 = module_0.identify_imports_main(var_16)
    var_18 = 'import os.path\nimport sys.platform'
    var_19 = '--modules'
    var_20 = [var_18, var_19]
    var_21 = module_0.identify_imports_main(var_20)
    var_22 = 'from os import path\nfrom sys import platform'
    var_23 = '--attributes'
    var_24 = [var_22, var_23]
    var_25 = module_0.identify_imports_main(var_24)



# Parsed testcases at query #7
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
    var_8 = True
    var_9 = module_0.Config()
    var_10 = 'test.py'
    var_11 = module_1.sort_imports(var_10, var_9)
    var_12 = 'test.py'
    var_13 = module_0.Config()
    var_14 = module_1.sort_imports(var_12, var_13)
    assert var_14 is None
    var_15 = 'test.py'
    var_16 = module_0.Config()
    var_17 = module_1.sort_imports(var_15, var_16)
    var_18 = 1
    var_19 = 'test.py'
    var_20 = module_0.Config()
    var_21 = module_1.sort_imports(var_19, var_20)



# Parsed testcases at query #8
#--------------------------


import isort.main as module_0

def test_case_0():
    var_0 = '-'
    var_1 = [var_0]
    var_2 = module_0.identify_imports_main(var_1)
    var_3 = 'import os\nimport sys'
    var_4 = [var_3]
    var_5 = module_0.identify_imports_main(var_4)
    var_6 = 'import os\ndef foo():\n    import sys'
    var_7 = '--top-only'
    var_8 = [var_6, var_7]
    var_9 = module_0.identify_imports_main(var_8)
    var_10 = 'import os\nimport os'
    var_11 = '--unique'
    var_12 = [var_10, var_11]
    var_13 = module_0.identify_imports_main(var_12)
    var_14 = 'import os.path\nimport sys'
    var_15 = '--packages'
    var_16 = [var_14, var_15]
    var_17 = module_0.identify_imports_main(var_16)
    var_18 = 'import os.path\nimport sys'
    var_19 = '--modules'
    var_20 = [var_18, var_19]
    var_21 = module_0.identify_imports_main(var_20)
    var_22 = 'from os import path\nfrom sys import argv'
    var_23 = '--attributes'
    var_24 = [var_22, var_23]
    var_25 = module_0.identify_imports_main(var_24)



# Parsed testcases at query #9
#--------------------------


import isort.main as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import sys\nimport os\nfrom pathlib import Path\n'
    var_2 = module_0.identify_imports_main()
    var_3 = module_0.identify_imports_main()
    var_4 = 'import sys\n\ndef foo():\n    import os\n'
    var_5 = module_0.identify_imports_main()
    var_6 = 'import sys\nimport sys\nfrom os import path\nfrom os import path\n'
    var_7 = module_0.identify_imports_main()
    var_8 = 'import sys'
    var_9 = 'from os import path'
    var_10 = 'import sys\nfrom os.path import join\nfrom collections import defaultdict\n'
    var_11 = module_0.identify_imports_main()
    var_12 = module_0.identify_imports_main()
    var_13 = module_0.identify_imports_main()
    var_14 = 'link.py'
    var_15 = module_0.identify_imports_main()



# Parsed testcases at query #10
#--------------------------


import isort.main as module_0

def test_case_0():
    var_0 = module_0.identify_imports_main()
    var_1 = 'import os'
    var_2 = 'import sys'
    var_3 = module_0.identify_imports_main()
    var_4 = 'import os'
    var_5 = 'import sys'
    var_6 = module_0.identify_imports_main()
    var_7 = 'import os'
    var_8 = 'import sys'
    var_9 = module_0.identify_imports_main()
    var_10 = 'import os'
    var_11 = module_0.identify_imports_main()
    var_12 = 'os'
    var_13 = 'sys'
    var_14 = module_0.identify_imports_main()
    var_15 = 'os.path'
    var_16 = 'sys'
    var_17 = module_0.identify_imports_main()
    var_18 = 'os.path'
    var_19 = 'sys.argv'



# Parsed testcases at query #11
#--------------------------


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = module_0.Config()
    var_2 = module_1.sort_imports(var_0, var_1)
    var_3 = 'test.py'
    var_4 = module_0.Config()
    var_5 = module_1.sort_imports(var_3, var_4)
    var_6 = 'test.py'
    var_7 = module_0.Config()
    var_8 = module_1.sort_imports(var_6, var_7)
    var_9 = 'test.py'
    var_10 = module_0.Config()
    var_11 = True
    var_12 = module_1.sort_imports(var_9, var_10, var_11)
    var_13 = 'test.py'
    var_14 = module_0.Config()
    var_15 = True
    var_16 = module_1.sort_imports(var_13, var_14, var_15)
    var_17 = 'test.py'
    var_18 = module_0.Config()
    var_19 = True
    var_20 = module_1.sort_imports(var_17, var_18, var_19)
    var_21 = 'test.py'
    var_22 = module_0.Config()
    var_23 = module_1.sort_imports(var_21, var_22)
    assert var_23 is None
    var_24 = 'Unable to parse file test.py due to test error'
    var_25 = 2
    var_26 = True
    var_27 = module_0.Config()
    var_28 = 'test.py'
    var_29 = module_1.sort_imports(var_28, var_27)
    var_30 = 'Encoding not supported for test.py'
    var_31 = 2
    var_32 = 'test.py'
    var_33 = module_0.Config()
    var_34 = module_1.sort_imports(var_32, var_33)
    var_35 = 1
    var_36 = 'test.py'
    var_37 = module_0.Config()
    var_38 = module_1.sort_imports(var_36, var_37)



# Parsed testcases at query #12
#--------------------------


import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)
    var_2 = '--line-length'
    var_3 = '88'
    var_4 = [var_2, var_3]
    var_5 = module_0.parse_args(var_4)
    var_6 = '--indent'
    var_7 = '    '
    var_8 = [var_2, var_3, var_6, var_7]
    var_9 = module_0.parse_args(var_8)
    var_10 = '--recursive'
    var_11 = [var_10]
    var_12 = module_0.parse_args(var_11)
    var_13 = '-k'
    var_14 = [var_13]
    var_15 = module_0.parse_args(var_14)
    var_16 = '--order-by-type'
    var_17 = [var_16]
    var_18 = module_0.parse_args(var_17)
    var_19 = '--dont-order-by-type'
    var_20 = [var_19]
    var_21 = module_0.parse_args(var_20)
    var_22 = '-m'
    var_23 = '3'
    var_24 = [var_22, var_23]
    var_25 = module_0.parse_args(var_24)
    var_26 = 'multi_line_output'
    var_27 = 3
    var_28 = 'VERT_HANGING'
    var_29 = [var_22, var_28]
    var_30 = module_0.parse_args(var_29)
    var_31 = '--float-to-top'
    var_32 = [var_31]
    var_33 = module_0.parse_args(var_32)
    var_34 = '--dont-float-to-top'
    var_35 = [var_34]
    var_36 = module_0.parse_args(var_35)
    var_37 = '--float-to-top'
    var_38 = '--dont-float-to-top'
    var_39 = [var_37, var_38]
    var_40 = module_0.parse_args(var_39)
    var_41 = '--follow-links'
    var_42 = [var_41]
    var_43 = module_0.parse_args(var_42)
    var_44 = '--dont-follow-links'
    var_45 = [var_44]
    var_46 = module_0.parse_args(var_45)
    var_47 = '--wrap-length'
    var_48 = '79'
    var_49 = [var_38, var_39, var_47, var_48]
    var_50 = module_0.parse_args(var_49)
    var_51 = '--src-path'
    var_52 = 'src'
    var_53 = [var_51, var_52]
    var_54 = module_0.parse_args(var_53)
    var_55 = '--builtin'
    var_56 = 'os'
    var_57 = [var_55, var_56]
    var_58 = module_0.parse_args(var_57)
    var_59 = '--thirdparty'
    var_60 = 'django'
    var_61 = [var_59, var_60]
    var_62 = module_0.parse_args(var_61)
    var_63 = '--project'
    var_64 = 'myproject'
    var_65 = [var_63, var_64]
    var_66 = module_0.parse_args(var_65)
    var_67 = '--known-local-folder'
    var_68 = 'local'
    var_69 = [var_67, var_68]
    var_70 = module_0.parse_args(var_69)
    var_71 = '--virtual-env'
    var_72 = 'env'
    var_73 = [var_71, var_72]
    var_74 = module_0.parse_args(var_73)
    var_75 = '--conda-env'
    var_76 = [var_75, var_72]
    var_77 = module_0.parse_args(var_76)
    var_78 = '--python-version'
    var_79 = '38'
    var_80 = [var_78, var_79]
    var_81 = module_0.parse_args(var_80)
    var_82 = '--section-default'
    var_83 = 'THIRDPARTY'
    var_84 = [var_82, var_83]
    var_85 = module_0.parse_args(var_84)
    var_86 = '--only-sections'
    var_87 = [var_86]
    var_88 = module_0.parse_args(var_87)
    var_89 = '--no-sections'
    var_90 = [var_89]
    var_91 = module_0.parse_args(var_90)
    var_92 = '--force-alphabetical-sort'
    var_93 = [var_92]
    var_94 = module_0.parse_args(var_93)
    var_95 = '--force-sort-within-sections'
    var_96 = [var_95]
    var_97 = module_0.parse_args(var_96)
    var_98 = '--honor-case-in-force-sorted-sections'
    var_99 = [var_98]
    var_100 = module_0.parse_args(var_99)
    var_101 = '--sort-relative-in-force-sorted-sections'
    var_102 = [var_101]
    var_103 = module_0.parse_args(var_102)
    var_104 = '--force-alphabetical-sort-within-sections'
    var_105 = [var_104]
    var_106 = module_0.parse_args(var_105)
    var_107 = '--top'
    var_108 = [var_107, var_56]
    var_109 = module_0.parse_args(var_108)
    var_110 = '--combine-straight-imports'
    var_111 = [var_110]
    var_112 = module_0.parse_args(var_111)
    var_113 = '--no-lines-before'
    var_114 = 'STDLIB'
    var_115 = [var_113, var_114]
    var_116 = module_0.parse_args(var_115)
    var_117 = '--force-grid-wrap'
    var_118 = '2'
    var_119 = [var_117, var_118]
    var_120 = module_0.parse_args(var_119)
    var_121 = '  '
    var_122 = [var_6, var_121]
    var_123 = module_0.parse_args(var_122)
    var_124 = '--lines-before-imports'
    var_125 = [var_124, var_118]
    var_126 = module_0.parse_args(var_125)
    var_127 = '--lines-after-imports'
    var_128 = [var_127, var_118]
    var_129 = module_0.parse_args(var_128)
    var_130 = '--lines-between-types'
    var_131 = [var_130, var_118]
    var_132 = module_0.parse_args(var_131)
    var_133 = '--line-ending'
    var_134 = 'LF'
    var_135 = [var_133, var_134]
    var_136 = module_0.parse_args(var_135)
    var_137 = '--length-sort'
    var_138 = [var_137]
    var_139 = module_0.parse_args(var_138)
    var_140 = '--length-sort-straight'
    var_141 = [var_140]
    var_142 = module_0.parse_args(var_141)
    var_143 = '--ensure-newline-before-comments'
    var_144 = [var_143]
    var_145 = module_0.parse_args(var_144)
    var_146 = '--no-inline-sort'
    var_147 = [var_146]
    var_148 = module_0.parse_args(var_147)
    var_149 = '--reverse-relative'
    var_150 = [var_149]
    var_151 = module_0.parse_args(var_150)
    var_152 = '--reverse-sort'
    var_153 = [var_152]
    var_154 = module_0.parse_args(var_153)
    var_155 = '--sort-order'
    var_156 = 'natural'
    var_157 = [var_155, var_156]
    var_158 = module_0.parse_args(var_157)
    var_159 = '--force-single-line-imports'
    var_160 = [var_159]
    var_161 = module_0.parse_args(var_160)
    var_162 = '--single-line-exclusions'
    var_163 = [var_162, var_56]
    var_164 = module_0.parse_args(var_163)
    var_165 = '--trailing-comma'
    var_166 = [var_165]
    var_167 = module_0.parse_args(var_166)
    var_168 = '--use-parentheses'
    var_169 = [var_168]
    var_170 = module_0.parse_args(var_169)
    var_171 = '--case-sensitive'
    var_172 = [var_171]
    var_173 = module_0.parse_args(var_172)
    var_174 = '--remove-redundant-aliases'
    var_175 = [var_174]
    var_176 = module_0.parse_args(var_175)
    var_177 = '--honor-noqa'
    var_178 = [var_177]
    var_179 = module_0.parse_args(var_178)
    var_180 = '--treat-comment-as-code'
    var_181 = '# noqa'
    var_182 = [var_180, var_181]
    var_183 = module_0.parse_args(var_182)
    var_184 = '--treat-all-comment-as-code'
    var_185 = [var_184]
    var_186 = module_0.parse_args(var_185)
    var_187 = '--formatter'
    var_188 = 'black'
    var_189 = [var_187, var_188]
    var_190 = module_0.parse_args(var_189)



# Parsed testcases at query #13
#--------------------------


import isort.main as module_0

def test_case_0():
    var_0 = '-'
    var_1 = [var_0]
    var_2 = module_0.identify_imports_main(var_1)
    assert var_2 == 'import os\nimport sys\n'
    var_3 = 'import os\nimport sys\n'
    var_4 = module_0.identify_imports_main(var_3)
    var_5 = 'import os\n\ndef foo():\n    import sys\n'
    var_6 = '--top-only'
    var_7 = module_0.identify_imports_main(var_4)
    var_8 = 'import os\nimport os\nimport sys\n'
    var_9 = '--unique'
    var_10 = module_0.identify_imports_main(var_4)
    var_11 = 'import os.path\nimport sys.platform\n'
    var_12 = '--packages'
    var_13 = module_0.identify_imports_main(var_4)
    var_14 = 'import os.path\nimport sys.platform\n'
    var_15 = '--modules'
    var_16 = module_0.identify_imports_main(var_4)
    var_17 = 'from os import path\nfrom sys import platform\n'
    var_18 = '--attributes'
    var_19 = module_0.identify_imports_main(var_4)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.main as module_0

def test_case_0():
    var_0 = '-'
    var_1 = [var_0]
    var_2 = module_0.identify_imports_main(var_1)
    var_3 = 'import os\nimport sys'
    var_4 = [var_3]
    var_5 = module_0.identify_imports_main(var_4)
    var_6 = 'import os\n\ndef func():\n    import sys'
    var_7 = '--top-only'
    var_8 = [var_6, var_7]
    var_9 = module_0.identify_imports_main(var_8)
    var_10 = 'import os\nimport os'
    var_11 = '--unique'
    var_12 = [var_10, var_11]
    var_13 = module_0.identify_imports_main(var_12)
    var_14 = 'import os.path\nimport sys.platform'
    var_15 = '--packages'
    var_16 = [var_14, var_15]
    var_17 = module_0.identify_imports_main(var_16)
    var_18 = 'import os.path\nimport sys.platform'
    var_19 = '--modules'
    var_20 = [var_18, var_19]
    var_21 = module_0.identify_imports_main(var_20)
    var_22 = 'from os import path\nfrom sys import platform'
    var_23 = '--attributes'
    var_24 = [var_22, var_23]
    var_25 = module_0.identify_imports_main(var_24)



# Parsed testcases at query #2
#--------------------------


import isort.main as module_0
import locale as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import sys\nfrom os import path\nimport numpy as np\n'
    var_2 = [var_0]
    var_3 = module_0.identify_imports_main(var_2)
    var_4 = 'import sys\n\ndef foo():\n    import os\n'
    var_5 = '--top-only'
    var_6 = 'import sys\nimport sys\nfrom os import path\nfrom os import path\n'
    var_7 = '--unique'
    var_8 = 'sys'
    var_9 = 'os.path'
    var_10 = 'import sys\nfrom os.path import join\nimport numpy as np\n'
    var_11 = '--packages'
    var_12 = '--modules'
    var_13 = '--attributes'
    var_14 = 'import sys\nfrom os import path\n'
    var_15 = '-'
    var_16 = [var_15]
    var_17 = 'nonexistent.py'
    var_18 = module_1.str(var_2)
    var_19 = [var_18]
    var_20 = module_0.identify_imports_main(var_19)



# Parsed testcases at query #3
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
    var_10 = module_1.sort_imports(var_8, var_9)
    var_11 = 'test.py'
    var_12 = module_0.Config()
    var_13 = module_1.sort_imports(var_11, var_12)
    var_14 = 'test.py'
    var_15 = module_0.Config()
    var_16 = module_1.sort_imports(var_14, var_15)
    assert var_16 is None
    var_17 = 'test.py'
    var_18 = module_0.Config()
    var_19 = module_1.sort_imports(var_17, var_18)
    var_20 = 'test.py'
    var_21 = module_0.Config()
    var_22 = module_1.sort_imports(var_20, var_21)
    var_23 = 1
    var_24 = 'test.py'
    var_25 = module_0.Config()
    var_26 = module_1.sort_imports(var_24, var_25)
    var_27 = 1



# Parsed testcases at query #4
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



# Parsed testcases at query #5
#--------------------------


import isort.main as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\nimport sys\nfrom typing import List\n'
    var_2 = '-'
    var_3 = [var_2]
    var_4 = module_0.identify_imports_main(var_3)
    var_5 = 'import os\n\ndef foo():\n    import sys\n'
    var_6 = '--top-only'
    var_7 = 'import os\nimport os\nimport sys\n'
    var_8 = '--unique'
    var_9 = 'os'
    var_10 = 'import os.path\nimport sys.platform\n'
    var_11 = '--packages'
    var_12 = '--modules'
    var_13 = 'from os import path\nfrom sys import platform\n'
    var_14 = '--attributes'



# Parsed testcases at query #6
#--------------------------


import isort.main as module_0

def test_case_0():
    var_0 = '-'
    var_1 = [var_0]
    var_2 = module_0.identify_imports_main(var_1)
    var_3 = 'import os\nimport sys'
    var_4 = [var_3]
    var_5 = module_0.identify_imports_main(var_4)
    var_6 = 'import os\n\ndef func():\n    import sys'
    var_7 = '--top-only'
    var_8 = [var_6, var_7]
    var_9 = module_0.identify_imports_main(var_8)
    var_10 = 'import os\nimport os'
    var_11 = '--unique'
    var_12 = [var_10, var_11]
    var_13 = module_0.identify_imports_main(var_12)
    var_14 = 'from os.path import join\nimport sys'
    var_15 = '--packages'
    var_16 = [var_14, var_15]
    var_17 = module_0.identify_imports_main(var_16)
    var_18 = 'from os.path import join\nimport sys'
    var_19 = '--modules'
    var_20 = [var_18, var_19]
    var_21 = module_0.identify_imports_main(var_20)
    var_22 = 'from os.path import join\nimport sys'
    var_23 = '--attributes'
    var_24 = [var_22, var_23]
    var_25 = module_0.identify_imports_main(var_24)



# Parsed testcases at query #7
#--------------------------


import isort.main as module_0

def test_case_0():
    var_0 = '-'
    var_1 = [var_0]
    var_2 = module_0.identify_imports_main(var_1)
    var_3 = 'import os\nimport sys'
    var_4 = module_0.identify_imports_main(var_3)
    var_5 = 'import os\n\ndef func():\n    import sys'
    var_6 = '--top-only'
    var_7 = module_0.identify_imports_main(var_4)
    var_8 = 'import os\nimport os'
    var_9 = '--unique'
    var_10 = module_0.identify_imports_main(var_4)
    var_11 = 'import os.path\nimport sys.platform'
    var_12 = '--packages'
    var_13 = module_0.identify_imports_main(var_4)
    var_14 = 'import os.path\nimport sys.platform'
    var_15 = '--modules'
    var_16 = module_0.identify_imports_main(var_4)
    var_17 = 'from os import path\nfrom sys import platform'
    var_18 = '--attributes'
    var_19 = module_0.identify_imports_main(var_4)



# Parsed testcases at query #8
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
    var_11 = module_1.sort_imports(var_8, var_9, write_to_stdout=var_10)
    var_12 = 'test.py'
    var_13 = module_0.Config()
    var_14 = module_1.sort_imports(var_12, var_13)
    assert var_14 is None
    var_15 = module_0.Config()
    var_16 = 'test.py'
    var_17 = module_1.sort_imports(var_16, var_15)
    var_18 = 'test.py'
    var_19 = module_0.Config()
    var_20 = module_1.sort_imports(var_18, var_19)
    var_21 = 1
    var_22 = 'test.py'
    var_23 = module_0.Config()
    var_24 = module_1.sort_imports(var_22, var_23)
    var_25 = 1



# Parsed testcases at query #9
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
    var_12 = 'os'
    var_13 = 'import os.path\nimport sys.platform\n'
    var_14 = '--packages'
    var_15 = [var_0, var_14]
    var_16 = module_0.identify_imports_main(var_15)
    var_17 = '--modules'
    var_18 = [var_0, var_17]
    var_19 = module_0.identify_imports_main(var_18)
    var_20 = 'from os import path\nfrom sys import platform\n'
    var_21 = '--attributes'
    var_22 = [var_0, var_21]
    var_23 = module_0.identify_imports_main(var_22)
    var_24 = 'import json\nimport ast\n'
    var_25 = '-'
    var_26 = [var_25]



# Parsed testcases at query #10
#--------------------------


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'api.check_file'
    var_1 = True
    var_2 = 'api.sort_file'
    var_3 = 'test.py'
    var_4 = module_0.Config()
    var_5 = module_1.sort_imports(var_3, var_4)
    var_6 = False
    var_7 = module_0.Config()
    var_8 = module_1.sort_imports(var_3, var_7, var_1)
    var_9 = module_0.Config()
    var_10 = module_1.sort_imports(var_3, var_9, var_1)
    var_11 = module_0.Config()
    var_12 = module_1.sort_imports(var_3, var_11)
    var_13 = module_0.Config()
    var_14 = module_1.sort_imports(var_3, var_13)
    var_15 = 'Test error'
    var_16 = module_0.Config()
    var_17 = module_1.sort_imports(var_3, var_16)
    assert var_17 is None
    var_18 = module_0.Config()
    var_19 = module_1.sort_imports(var_3, var_18)
    assert var_19 is None
    var_20 = 'sys.exit'
    var_21 = 'test.py'
    var_22 = module_0.Config()
    var_23 = module_1.sort_imports(var_21, var_22)
    var_24 = 'test.py'
    var_25 = module_0.Config()
    var_26 = module_1.sort_imports(var_24, var_25)



# Parsed testcases at query #11
#--------------------------


import isort.main as module_0

def test_case_0():
    var_0 = 'import sys\nimport os\nfrom collections import defaultdict\n'
    var_1 = module_0.identify_imports_main()
    var_2 = 'sys'
    var_3 = 'os'
    var_4 = 'collections.defaultdict'
    var_5 = 'import sys\nimport os\nfrom collections import defaultdict\n'
    var_6 = module_0.identify_imports_main()
    var_7 = 'sys'
    var_8 = 'os'
    var_9 = 'collections.defaultdict'
    var_10 = 'import sys\nimport sys\nfrom collections import defaultdict\nfrom collections import defaultdict\n'
    var_11 = module_0.identify_imports_main()
    var_12 = 'sys'
    var_13 = 'collections.defaultdict'
    var_14 = 'import sys\nimport os\nfrom collections import defaultdict\n'
    var_15 = module_0.identify_imports_main()
    var_16 = 'sys'
    var_17 = 'os'
    var_18 = 'collections'
    var_19 = 'import sys\nimport os\nfrom collections import defaultdict\n'
    var_20 = module_0.identify_imports_main()
    var_21 = 'sys'
    var_22 = 'os'
    var_23 = 'collections'
    var_24 = 'import sys\nimport os\nfrom collections import defaultdict\n'
    var_25 = module_0.identify_imports_main()
    var_26 = 'collections.defaultdict'
    var_27 = 'import sys\nimport os\n\ndef foo():\n    from collections import defaultdict\n'
    var_28 = module_0.identify_imports_main()
    var_29 = 'sys'
    var_30 = 'os'



# Parsed testcases at query #12
#--------------------------


import isort.main as module_0

def test_case_0():
    var_0 = '-'
    var_1 = [var_0]
    var_2 = module_0.identify_imports_main(var_1)
    var_3 = 'import os\nimport sys\n'
    var_4 = 0
    var_5 = [var_3]
    var_6 = module_0.identify_imports_main(var_5)
    var_7 = 'import os\n\ndef func():\n    import sys\n'
    var_8 = 0
    var_9 = '--top-only'
    var_10 = [var_7, var_9]
    var_11 = module_0.identify_imports_main(var_10)
    var_12 = 'import os\nimport os\n'
    var_13 = 0
    var_14 = '--unique'
    var_15 = [var_12, var_14]
    var_16 = module_0.identify_imports_main(var_15)
    var_17 = 'import os.path\nimport sys.platform\n'
    var_18 = 0
    var_19 = '--packages'
    var_20 = [var_17, var_19]
    var_21 = module_0.identify_imports_main(var_20)
    var_22 = 'import os.path\nimport sys.platform\n'
    var_23 = 0
    var_24 = '--modules'
    var_25 = [var_22, var_24]
    var_26 = module_0.identify_imports_main(var_25)
    var_27 = 'from os import path\nfrom sys import platform\n'
    var_28 = 0
    var_29 = '--attributes'
    var_30 = [var_27, var_29]
    var_31 = module_0.identify_imports_main(var_30)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.main as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_2 = [var_0]
    var_3 = module_0.identify_imports_main(var_2)
    var_4 = 'import json\nfrom collections import defaultdict\n'
    var_5 = '-'
    var_6 = [var_5]
    var_7 = 'import os\n\ndef foo():\n    import sys\n'
    var_8 = '--top-only'
    var_9 = 'import os\nimport os\nimport sys\n'
    var_10 = '--unique'
    var_11 = 'os'
    var_12 = 'import os.path\nfrom collections import defaultdict\n'
    var_13 = '--packages'
    var_14 = '--modules'
    var_15 = '--attributes'



# Parsed testcases at query #2
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
    var_11 = 'test.py'
    var_12 = module_1.sort_imports(var_11, var_0)
    assert var_12 is None
    var_13 = 'test.py'
    var_14 = module_1.sort_imports(var_13, var_0)
    var_15 = 'test.py'
    var_16 = module_1.sort_imports(var_15, var_0)



# Parsed testcases at query #3
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
    var_12 = 'vertical'
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
    var_28 = '--dont-follow-links'
    var_29 = [var_28]
    var_30 = module_0.parse_args(var_29)
    var_31 = '--known-first-party'
    var_32 = 'module1'
    var_33 = 'module2'
    var_34 = [var_31, var_32, var_31, var_33]
    var_35 = module_0.parse_args(var_34)
    var_36 = []
    var_37 = module_0.parse_args(var_36)
    var_38 = module_0.parse_args()



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import sys\nfrom os import path\nimport numpy as np\n'
    var_2 = 'import sys\n\ndef foo():\n    import os\n'
    var_3 = '--top-only'
    var_4 = 'import sys\nimport sys\nfrom os import path\nfrom os import path\n'
    var_5 = '--unique'
    var_6 = 'import sys'
    var_7 = 'from os import path'
    var_8 = 'import sys\nfrom os.path import join\nimport numpy as np\n'
    var_9 = '--packages'
    var_10 = '--modules'
    var_11 = '--attributes'
    var_12 = 'import sys\nfrom os import path\n'
    var_13 = '-'
    var_14 = [var_13]



# Parsed testcases at query #5
#--------------------------


import isort.main as module_0

def test_case_0():
    var_0 = '--line-length'
    var_1 = '88'
    var_2 = '--indent'
    var_3 = '    '
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
    var_24 = '--dont-follow-links'
    var_25 = [var_24]
    var_26 = module_0.parse_args(var_25)
    var_27 = '--dont-float-to-top'
    var_28 = [var_27]
    var_29 = module_0.parse_args(var_28)
    var_30 = '--float-to-top'
    var_31 = '--dont-float-to-top'
    var_32 = [var_30, var_31]
    var_33 = module_0.parse_args(var_32)
    var_34 = '--single-line-exclusions'
    var_35 = 'module1'
    var_36 = 'module2'
    var_37 = [var_34, var_35, var_34, var_36]
    var_38 = module_0.parse_args(var_37)
    var_39 = []
    var_40 = module_0.parse_args(var_39)
    var_41 = module_0.parse_args()



# Parsed testcases at query #6
#--------------------------


import isort.main as module_0

def test_case_0():
    var_0 = '-'
    var_1 = [var_0]
    var_2 = module_0.identify_imports_main(var_1)
    var_3 = 'import os\nfrom sys import argv'
    var_4 = module_0.identify_imports_main(var_3)
    var_5 = 'import os\nfrom sys import argv\ndef foo():\n    import json'
    var_6 = '--top-only'
    var_7 = module_0.identify_imports_main(var_4)
    var_8 = 'import os\nimport os\nfrom sys import argv\nfrom sys import argv'
    var_9 = '--unique'
    var_10 = module_0.identify_imports_main(var_4)
    var_11 = 'import os.path\nfrom sys import argv'
    var_12 = '--packages'
    var_13 = module_0.identify_imports_main(var_4)
    var_14 = 'import os.path\nfrom sys import argv'
    var_15 = '--modules'
    var_16 = module_0.identify_imports_main(var_4)
    var_17 = 'import os.path\nfrom sys import argv'
    var_18 = '--attributes'
    var_19 = module_0.identify_imports_main(var_4)



# Parsed testcases at query #7
#--------------------------


import isort.main as module_0

def test_case_0():
    var_0 = module_0.identify_imports_main()
    var_1 = 'import os'
    var_2 = 'from sys import path'
    var_3 = 'os'
    var_4 = None
    var_5 = 'sys'
    var_6 = 'path'
    var_7 = module_0.identify_imports_main()
    var_8 = 'import os'
    var_9 = 'from sys import path'
    var_10 = 'os'
    var_11 = None
    var_12 = 'sys'
    var_13 = 'path'
    var_14 = module_0.identify_imports_main()
    var_15 = 'import os'
    var_16 = 'from sys import path'
    var_17 = 'os'
    var_18 = None
    var_19 = module_0.identify_imports_main()
    var_20 = 'import os'
    var_21 = 'os.path'
    var_22 = None
    var_23 = 'sys.path'
    var_24 = module_0.identify_imports_main()
    var_25 = 'os'
    var_26 = 'sys'
    var_27 = 'os.path'
    var_28 = None
    var_29 = 'sys.path'
    var_30 = module_0.identify_imports_main()
    var_31 = 'os.path'
    var_32 = 'sys.path'
    var_33 = 'os'
    var_34 = 'path'
    var_35 = 'sys'
    var_36 = module_0.identify_imports_main()
    var_37 = 'os.path'
    var_38 = 'sys.path'



# Parsed testcases at query #8
#--------------------------


import isort.main as module_0
import locale as module_1

def test_case_0():
    var_0 = module_0.identify_imports_main()
    var_1 = 'os'
    var_2 = 'sys'
    var_3 = module_0.identify_imports_main()
    var_4 = 'os'
    var_5 = 'sys'
    var_6 = module_0.identify_imports_main()
    var_7 = 0
    var_8 = module_1.str(var_4)
    assert var_8 == 'os'
    var_9 = module_0.identify_imports_main()
    var_10 = 0
    var_11 = module_1.str(var_4)
    assert var_11 == 'os'
    var_12 = module_0.identify_imports_main()
    var_13 = 'os'
    var_14 = 'sys'
    var_15 = module_0.identify_imports_main()
    var_16 = 'os'
    var_17 = 'sys'
    var_18 = module_0.identify_imports_main()
    var_19 = 0
    var_20 = module_1.str(var_16)
    assert var_20 == 'os.path'



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import sys\nfrom os import path\nimport argparse'
    var_2 = 'import sys\n\ndef foo():\n    import os'
    var_3 = '--top-only'
    var_4 = 'import sys\nimport sys\nfrom os import path\nfrom os import path'
    var_5 = '--unique'
    var_6 = 'import sys'
    var_7 = 'from os import path'
    var_8 = 'import sys\nfrom os.path import join\nimport argparse'
    var_9 = '--packages'
    var_10 = '--modules'
    var_11 = '--attributes'
    var_12 = 'import sys\nfrom os import path'
    var_13 = '-'
    var_14 = [var_13]
    var_15 = 'test2.py'
    var_16 = 'import json\nfrom collections import defaultdict'



# Parsed testcases at query #10
#--------------------------


import isort.main as module_0

def test_case_0():
    var_0 = module_0.identify_imports_main()
    var_1 = 'os'
    var_2 = 'sys'
    var_3 = module_0.identify_imports_main()
    var_4 = 'os'
    var_5 = 'sys'
    var_6 = module_0.identify_imports_main()
    var_7 = 'os'
    var_8 = 'sys'
    var_9 = module_0.identify_imports_main()
    var_10 = 'os'
    var_11 = module_0.identify_imports_main()
    var_12 = 'os'
    var_13 = 'sys'
    var_14 = module_0.identify_imports_main()
    var_15 = 'os.path'
    var_16 = 'sys'
    var_17 = module_0.identify_imports_main()
    var_18 = 'os.path.join'



# Parsed testcases at query #11
#--------------------------


import isort.main as module_0

def test_case_0():
    var_0 = module_0.identify_imports_main()
    var_1 = 'import os'
    var_2 = 'import os'
    var_3 = 'os'
    var_4 = None
    var_5 = 1
    var_6 = 'import sys'
    var_7 = 'sys'
    var_8 = 2
    var_9 = module_0.identify_imports_main()
    var_10 = 'import os'
    var_11 = 'import sys'
    var_12 = 'import os'
    var_13 = 'os'
    var_14 = None
    var_15 = 1
    var_16 = 'import sys'
    var_17 = 'sys'
    var_18 = 2
    var_19 = module_0.identify_imports_main()
    var_20 = 'import os'
    var_21 = 'os'
    var_22 = None
    var_23 = 1
    var_24 = 2
    var_25 = module_0.identify_imports_main()
    var_26 = 'import os'
    var_27 = 'import os.path'
    var_28 = 'os.path'
    var_29 = None
    var_30 = 1
    var_31 = 'import sys'
    var_32 = 'sys'
    var_33 = 2
    var_34 = module_0.identify_imports_main()
    var_35 = 'os'
    var_36 = 'sys'
    var_37 = 'import os.path'
    var_38 = 'os.path'
    var_39 = None
    var_40 = 1
    var_41 = 'import sys'
    var_42 = 'sys'
    var_43 = 2
    var_44 = module_0.identify_imports_main()
    var_45 = 'os.path'
    var_46 = 'sys'
    var_47 = 'from os import path'
    var_48 = 'os'
    var_49 = 'path'
    var_50 = 1
    var_51 = 'from sys import argv'
    var_52 = 'sys'
    var_53 = 'argv'
    var_54 = 2
    var_55 = module_0.identify_imports_main()
    var_56 = 'os.path'
    var_57 = 'sys.argv'



# Parsed testcases at query #12
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
    var_10 = module_1.sort_imports(var_8, var_9)
    var_11 = 'test.py'
    var_12 = module_0.Config()
    var_13 = module_1.sort_imports(var_11, var_12)
    var_14 = 'test.py'
    var_15 = module_0.Config()
    var_16 = module_1.sort_imports(var_14, var_15)
    assert var_16 is None
    var_17 = module_0.Config()
    var_18 = 'test.py'
    var_19 = module_1.sort_imports(var_18, var_17)
    var_20 = 'test.py'
    var_21 = module_0.Config()
    var_22 = module_1.sort_imports(var_20, var_21)
    var_23 = 1
    var_24 = 'test.py'
    var_25 = module_0.Config()
    var_26 = module_1.sort_imports(var_24, var_25)
    var_27 = 1



# Parsed testcases at query #13
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
    var_21 = '-k'
    var_22 = [var_21]
    var_23 = module_0.parse_args(var_22)
    var_24 = '--force-alphabetical-sort'
    var_25 = '--no-sections'
    var_26 = [var_24, var_25]
    var_27 = module_0.parse_args(var_26)
    var_28 = '--known-first-party'
    var_29 = 'package1'
    var_30 = 'package2'
    var_31 = [var_28, var_29, var_28, var_30]
    var_32 = module_0.parse_args(var_31)
    var_33 = []
    var_34 = module_0.parse_args(var_33)
    var_35 = module_0.parse_args()
    var_36 = '--float-to-top'
    var_37 = '--dont-float-to-top'
    var_38 = [var_36, var_37]
    var_39 = module_0.parse_args(var_38)



# Parsed testcases at query #14
#--------------------------


import isort.main as module_0

def test_case_0():
    var_0 = '-'
    var_1 = [var_0]
    var_2 = module_0.identify_imports_main(var_1)
    var_3 = 'import os\nfrom sys import argv\n'
    var_4 = [var_3]
    var_5 = module_0.identify_imports_main(var_4)
    var_6 = 'import os\n\ndef foo():\n    from sys import argv\n'
    var_7 = '--top-only'
    var_8 = [var_6, var_7]
    var_9 = module_0.identify_imports_main(var_8)
    var_10 = 'import os\nimport sys\nimport os\n'
    var_11 = '--unique'
    var_12 = [var_10, var_11]
    var_13 = module_0.identify_imports_main(var_12)
    var_14 = 'import os.path\nfrom sys import argv\n'
    var_15 = '--packages'
    var_16 = [var_14, var_15]
    var_17 = module_0.identify_imports_main(var_16)
    var_18 = 'import os.path\nfrom sys import argv\n'
    var_19 = '--modules'
    var_20 = [var_18, var_19]
    var_21 = module_0.identify_imports_main(var_20)
    var_22 = 'import os.path\nfrom sys import argv\n'
    var_23 = '--attributes'
    var_24 = [var_22, var_23]
    var_25 = module_0.identify_imports_main(var_24)



# Parsed testcases at query #15
#--------------------------


import isort.main as module_0

def test_case_0():
    var_0 = 'sys.stdin'
    var_1 = 'import os'
    var_2 = 'from sys import path'
    var_3 = [var_1, var_2]
    var_4 = '-'
    var_5 = [var_4]
    var_6 = module_0.identify_imports_main(var_5)
    var_7 = 'api.find_imports_in_paths'
    var_8 = 'os'
    var_9 = 'sys.path'
    var_10 = 'test.py'
    var_11 = [var_10]
    var_12 = module_0.identify_imports_main(var_11)
    var_13 = '--top-only'
    var_14 = [var_10, var_13]
    var_15 = module_0.identify_imports_main(var_14)
    var_16 = '--unique'
    var_17 = [var_10, var_16]
    var_18 = module_0.identify_imports_main(var_17)
    var_19 = 'os.path'
    var_20 = '--packages'
    var_21 = [var_10, var_20]
    var_22 = module_0.identify_imports_main(var_21)
    var_23 = '--modules'
    var_24 = [var_10, var_23]
    var_25 = module_0.identify_imports_main(var_24)
    var_26 = 'join'
    var_27 = 'append'
    var_28 = '--attributes'
    var_29 = [var_10, var_28]
    var_30 = module_0.identify_imports_main(var_29)



