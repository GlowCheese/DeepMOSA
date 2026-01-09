####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import builtins as module_1

import isort.main as module_0


def test_case_0():
    var_0 = module_0.identify_imports_main()
    var_1 = 'file1.py'
    var_2 = 'file2.py'
    var_3 = [var_1, var_2]
    var_4 = module_0.identify_imports_main(var_3)
    var_5 = [var_1, var_2]
    var_6 = '--top-only'
    var_7 = [var_6, var_1]
    var_8 = module_0.identify_imports_main(var_7)
    var_9 = '--follow-links'
    var_10 = [var_9, var_1]
    var_11 = module_0.identify_imports_main(var_10)
    var_12 = '--unique'
    var_13 = [var_12, var_1]
    var_14 = module_0.identify_imports_main(var_13)
    var_15 = '--packages'
    var_16 = [var_15, var_1]
    var_17 = module_0.identify_imports_main(var_16)
    var_18 = '--modules'
    var_19 = [var_18, var_1]
    var_20 = module_0.identify_imports_main(var_19)
    var_21 = '--attributes'
    var_22 = [var_21, var_1]
    var_23 = module_0.identify_imports_main(var_22)
    var_24 = [var_6, var_9, var_12, var_1]
    var_25 = module_0.identify_imports_main(var_24)
    var_26 = '--invalid-flag'
    var_27 = [var_26, var_1]
    var_28 = module_0.identify_imports_main(var_27)
    var_29 = []
    var_30 = module_0.identify_imports_main(var_29)
    var_31 = ''
    var_32 = [var_31]
    var_33 = module_0.identify_imports_main(var_32)
    var_34 = 'file@name.py'
    var_35 = [var_34]
    var_36 = module_0.identify_imports_main(var_35)
    var_37 = 'file name.py'
    var_38 = [var_37]
    var_39 = module_0.identify_imports_main(var_38)
    var_40 = 'file.txt'
    var_41 = [var_40]
    var_42 = module_0.identify_imports_main(var_41)
    var_43 = 'file'
    var_44 = [var_43]
    var_45 = module_0.identify_imports_main(var_44)
    var_46 = 'file.py.txt'
    var_47 = [var_46]
    var_48 = module_0.identify_imports_main(var_47)
    var_49 = '  file.py  '
    var_50 = [var_49]
    var_51 = module_0.identify_imports_main(var_50)
    var_52 = '/path/to/file.py'
    var_53 = [var_52]
    var_54 = module_0.identify_imports_main(var_53)
    var_55 = './file.py'
    var_56 = [var_55]
    var_57 = module_0.identify_imports_main(var_56)
    var_58 = '../file.py'
    var_59 = [var_58]
    var_60 = module_0.identify_imports_main(var_59)
    var_61 = 'file*.py'
    var_62 = [var_61]
    var_63 = module_0.identify_imports_main(var_62)
    var_64 = '$HOME/file.py'
    var_65 = [var_64]
    var_66 = module_0.identify_imports_main(var_65)
    var_67 = '~/file.py'
    var_68 = [var_67]
    var_69 = module_0.identify_imports_main(var_68)
    var_70 = 'C:\\path\\to\\file.py'
    var_71 = [var_70]
    var_72 = module_0.identify_imports_main(var_71)
    var_73 = [var_52]
    var_74 = module_0.identify_imports_main(var_73)
    var_75 = 'C:/path/to/file.py'
    var_76 = [var_75]
    var_77 = module_0.identify_imports_main(var_76)
    var_78 = 'file_αβγ.py'
    var_79 = [var_78]
    var_80 = module_0.identify_imports_main(var_79)
    var_81 = 'file😀.py'
    var_82 = [var_81]
    var_83 = module_0.identify_imports_main(var_82)
    var_84 = 'file\n.py'
    var_85 = [var_84]
    var_86 = module_0.identify_imports_main(var_85)
    var_87 = 'file\t.py'
    var_88 = [var_87]
    var_89 = module_0.identify_imports_main(var_88)
    var_90 = 'file\r.py'
    var_91 = [var_90]
    var_92 = module_0.identify_imports_main(var_91)
    var_93 = 'file\x00.py'
    var_94 = [var_93]
    var_95 = module_0.identify_imports_main(var_94)
    var_96 = 'file\x08.py'
    var_97 = [var_96]
    var_98 = module_0.identify_imports_main(var_97)
    var_99 = 'file\x0c.py'
    var_100 = [var_99]
    var_101 = module_0.identify_imports_main(var_100)
    var_102 = 'file\x0b.py'
    var_103 = [var_102]
    var_104 = module_0.identify_imports_main(var_103)
    var_105 = 'file\\n.py'
    var_106 = [var_105]
    var_107 = module_0.identify_imports_main(var_106)
    var_108 = 'file\\u03B1.py'
    var_109 = [var_108]
    var_110 = module_0.identify_imports_main(var_109)
    var_111 = 'file\\x41.py'
    var_112 = [var_111]
    var_113 = module_0.identify_imports_main(var_112)
    var_114 = 'file\\101.py'
    var_115 = [var_114]
    var_116 = module_0.identify_imports_main(var_115)
    var_117 = [var_105]
    var_118 = module_0.identify_imports_main(var_117)
    var_119 = b'file.py'
    var_120 = [var_119]
    var_121 = module_0.identify_imports_main(var_120)
    var_122 = 123
    var_123 = [var_122]
    var_124 = module_0.identify_imports_main(var_123)
    var_125 = 3.14
    var_126 = [var_125]
    var_127 = module_0.identify_imports_main(var_126)
    var_128 = True
    var_129 = [var_128]
    var_130 = module_0.identify_imports_main(var_129)
    var_131 = None
    var_132 = [var_131]
    var_133 = module_0.identify_imports_main(var_132)
    var_134 = [var_1, var_2]
    var_135 = [var_134]
    var_136 = module_0.identify_imports_main(var_135)
    var_137 = (var_1, var_2)
    var_138 = [var_137]
    var_139 = module_0.identify_imports_main(var_138)
    var_140 = 'file.py'
    var_141 = {var_43: var_140}
    var_142 = [var_141]
    var_143 = module_0.identify_imports_main(var_142)
    var_144 = {var_140}
    var_145 = [var_144]
    var_146 = module_0.identify_imports_main(var_145)
    var_147 = [var_140]
    var_148 = frozenset(var_147)
    var_149 = [var_148]
    var_150 = module_0.identify_imports_main(var_149)
    var_151 = 5
    var_152 = range(var_151)
    var_153 = [var_152]
    var_154 = module_0.identify_imports_main(var_153)
    var_155 = 2
    var_156 = complex(var_128, var_155)
    var_157 = [var_156]
    var_158 = module_0.identify_imports_main(var_157)
    var_159 = [var_119]
    var_160 = module_0.identify_imports_main(var_159)
    var_161 = bytearray(var_119)
    var_162 = [var_161]
    var_163 = module_0.identify_imports_main(var_162)
    var_164 = memoryview(var_119)
    var_165 = [var_164]
    var_166 = module_0.identify_imports_main(var_165)
    var_167 = 0
    var_168 = 10
    var_169 = module_0.identify_imports_main(var_165)
    var_170 = module_0.identify_imports_main(var_165)
    var_171 = module_0.identify_imports_main(var_165)
    var_172 = module_1.object()
    var_173 = [var_172]
    var_174 = module_0.identify_imports_main(var_173)
    var_175 = lambda x: x
    var_176 = [var_175]



# Parsed testcases at query #2
#--------------------------



def test_case_0():
    var_0 = []
    var_1 = module_0.identify_imports_main(var_0)
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'test.py'
    var_5 = [var_4]
    var_6 = module_0.identify_imports_main(var_5)
    var_7 = 'sys'
    var_8 = 'stdin'
    var_9 = '-'
    var_10 = [var_9]
    var_11 = module_0.identify_imports_main(var_10)
    var_12 = 'test.py'
    var_13 = '--unique'
    var_14 = [var_12, var_13]
    var_15 = module_0.identify_imports_main(var_14)
    var_16 = 'os.path'
    var_17 = 'join'
    var_18 = 'test.py'
    var_19 = '--packages'
    var_20 = [var_18, var_19]
    var_21 = module_0.identify_imports_main(var_20)
    var_22 = 'split'
    var_23 = 'test.py'
    var_24 = '--modules'
    var_25 = [var_23, var_24]
    var_26 = module_0.identify_imports_main(var_25)
    var_27 = 'test.py'
    var_28 = '--attributes'
    var_29 = [var_27, var_28]
    var_30 = module_0.identify_imports_main(var_29)
    var_31 = 'test.py'
    var_32 = '--top-only'
    var_33 = [var_31, var_32]
    var_34 = module_0.identify_imports_main(var_33)
    var_35 = 'test.py'
    var_36 = '--follow-links'
    var_37 = [var_35, var_36]
    var_38 = module_0.identify_imports_main(var_37)
    var_39 = 'test1.py'
    var_40 = 'test2.py'
    var_41 = [var_39, var_40]
    var_42 = module_0.identify_imports_main(var_41)
    var_43 = '--invalid-arg'
    var_44 = [var_43]
    var_45 = module_0.identify_imports_main(var_44)
    var_46 = 'All tests passed!'
    var_47 = print(var_46)



# Parsed testcases at query #3
#--------------------------




# Parsed testcases at query #4
#--------------------------




# Parsed testcases at query #5
#--------------------------


import _io as module_0
import re as module_2

import isort.main as module_1


def test_case_0():
    var_0 = 'import os\nimport sys\nfrom collections import defaultdict\n'
    var_1 = module_0.StringIO()
    var_2 = module_1.identify_imports_main(var_0)
    var_3 = '\n'
    var_4 = module_2.split(var_3)
    var_5 = 'os'
    var_6 = 'sys'
    var_7 = 'collections.defaultdict'
    var_8 = [var_5, var_6, var_7]
    var_9 = set(var_4)
    var_10 = set(var_8)
    var_11 = 'import math\nimport json\n'
    var_12 = module_0.StringIO()
    var_13 = '-'
    var_14 = [var_13]
    var_15 = '\n'
    var_16 = module_2.split(var_15)
    var_17 = 'math'
    var_18 = 'json'
    var_19 = [var_17, var_18]
    var_20 = set(var_16)
    var_21 = set(var_19)
    var_22 = 'import os\nimport sys\nimport os\nimport sys\n'
    var_23 = module_0.StringIO()
    var_24 = '--unique'
    var_25 = module_1.identify_imports_main(var_14)
    var_26 = '\n'
    var_27 = module_2.split(var_26)
    var_28 = 'os'
    var_29 = 'sys'
    var_30 = [var_28, var_29]
    var_31 = set(var_27)
    var_32 = set(var_30)
    var_33 = len(var_27)
    assert var_33 == 2
    var_34 = 'import os.path\nimport os\nfrom os import makedirs\n'
    var_35 = module_0.StringIO()
    var_36 = '--packages'
    var_37 = module_1.identify_imports_main(var_14)
    var_38 = '\n'
    var_39 = module_2.split(var_38)
    var_40 = 'os'
    var_41 = [var_40]
    var_42 = set(var_39)
    var_43 = set(var_41)
    var_44 = 'All tests passed!'
    var_45 = print(var_44)



# Parsed testcases at query #6
#--------------------------




# Parsed testcases at query #7
#--------------------------



def test_case_0():
    var_0 = 'import os\nimport sys\nfrom collections import defaultdict\n'
    var_1 = module_0.StringIO()
    var_2 = module_1.identify_imports_main(var_0)
    var_3 = '\n'
    var_4 = module_2.split(var_3)
    var_5 = 'import json\nimport math\n'
    var_6 = module_0.StringIO()
    var_7 = '-'
    var_8 = [var_7]
    var_9 = '\n'
    var_10 = module_2.split(var_9)
    var_11 = 'import os\nimport sys\nimport os\n'
    var_12 = var_6.name
    var_13 = module_0.StringIO()
    var_14 = '--unique'
    var_15 = [var_12, var_14]
    var_16 = module_1.identify_imports_main(var_15)
    var_17 = '\n'
    var_18 = module_2.split(var_17)
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = 'import os'
    var_21 = 'All tests passed!'
    var_22 = print(var_21)



# Parsed testcases at query #8
#--------------------------


import isort.main as module_0


def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)
    var_2 = '--check-only'
    var_3 = [var_2]
    var_4 = module_0.parse_args(var_3)
    var_5 = '--diff'
    var_6 = [var_2, var_5]
    var_7 = module_0.parse_args(var_6)
    var_8 = '--line-length'
    var_9 = '100'
    var_10 = [var_8, var_9]
    var_11 = module_0.parse_args(var_10)
    var_12 = '--recursive'
    var_13 = [var_12]
    var_14 = module_0.parse_args(var_13)
    var_15 = '-rc'
    var_16 = [var_15]
    var_17 = module_0.parse_args(var_16)
    var_18 = '--known-thirdparty'
    var_19 = 'module1'
    var_20 = 'module2'
    var_21 = [var_18, var_19, var_20]
    var_22 = module_0.parse_args(var_21)
    var_23 = '--py'
    var_24 = 'auto'
    var_25 = [var_23, var_24]
    var_26 = module_0.parse_args(var_25)
    var_27 = '--py'
    var_28 = 'invalid'
    var_29 = [var_27, var_28]
    var_30 = module_0.parse_args(var_29)
    var_31 = '--float-to-top'
    var_32 = '--dont-float-to-top'
    var_33 = [var_31, var_32]
    var_34 = module_0.parse_args(var_33)
    var_35 = '--section-default'
    var_36 = 'STDLIB'
    var_37 = [var_35, var_36]
    var_38 = module_0.parse_args(var_37)
    var_39 = '--multi-line'
    var_40 = '5'
    var_41 = [var_39, var_40]
    var_42 = module_0.parse_args(var_41)
    var_43 = 'VERTICAL_GRID_GROUPED'
    var_44 = [var_39, var_43]
    var_45 = module_0.parse_args(var_44)
    var_46 = '--multi-line'
    var_47 = 'invalid'
    var_48 = [var_46, var_47]
    var_49 = module_0.parse_args(var_48)
    var_50 = '--sort-order'
    var_51 = 'natural'
    var_52 = [var_50, var_51]
    var_53 = module_0.parse_args(var_52)
    var_54 = '--formatter'
    var_55 = 'custom'
    var_56 = [var_54, var_55]
    var_57 = module_0.parse_args(var_56)
    var_58 = '--line-ending'
    var_59 = 'lf'
    var_60 = [var_58, var_59]
    var_61 = module_0.parse_args(var_60)
    var_62 = '--indent'
    var_63 = '  '
    var_64 = [var_62, var_63]
    var_65 = module_0.parse_args(var_64)
    var_66 = '--lines-before-imports'
    var_67 = '2'
    var_68 = [var_66, var_67]
    var_69 = module_0.parse_args(var_68)
    var_70 = '--lines-after-imports'
    var_71 = [var_70, var_67]
    var_72 = module_0.parse_args(var_71)
    var_73 = '--lines-between-types'
    var_74 = [var_73, var_67]
    var_75 = module_0.parse_args(var_74)
    var_76 = '--force-grid-wrap'
    var_77 = [var_76, var_67]
    var_78 = module_0.parse_args(var_77)
    var_79 = '--force-sort-within-sections'
    var_80 = [var_79]
    var_81 = module_0.parse_args(var_80)
    var_82 = '--force-alphabetical-sort'
    var_83 = [var_82]
    var_84 = module_0.parse_args(var_83)
    var_85 = '--force-alphabetical-sort-within-sections'
    var_86 = [var_85]
    var_87 = module_0.parse_args(var_86)
    var_88 = '--honor-case-in-force-sorted-sections'
    var_89 = [var_88]
    var_90 = module_0.parse_args(var_89)
    var_91 = '--sort-relative-in-force-sorted-sections'
    var_92 = [var_91]
    var_93 = module_0.parse_args(var_92)
    var_94 = '--combine-straight-imports'
    var_95 = [var_94]
    var_96 = module_0.parse_args(var_95)
    var_97 = '--no-lines-before'
    var_98 = [var_97, var_36]
    var_99 = module_0.parse_args(var_98)
    var_100 = '--src-path'
    var_101 = '/path/to/src'
    var_102 = [var_100, var_101]
    var_103 = module_0.parse_args(var_102)
    var_104 = '--builtin'
    var_105 = 'module'
    var_106 = [var_104, var_105]
    var_107 = module_0.parse_args(var_106)
    var_108 = '--extra-builtin'
    var_109 = [var_108, var_105]
    var_110 = module_0.parse_args(var_109)
    var_111 = '--future'
    var_112 = [var_111, var_105]
    var_113 = module_0.parse_args(var_112)
    var_114 = '--thirdparty'
    var_115 = [var_114, var_105]
    var_116 = module_0.parse_args(var_115)
    var_117 = '--project'
    var_118 = [var_117, var_105]
    var_119 = module_0.parse_args(var_118)
    var_120 = '--known-local-folder'
    var_121 = [var_120, var_105]
    var_122 = module_0.parse_args(var_121)
    var_123 = '--virtual-env'
    var_124 = '/path/to/venv'
    var_125 = [var_123, var_124]
    var_126 = module_0.parse_args(var_125)
    var_127 = '--conda-env'
    var_128 = '/path/to/conda'
    var_129 = [var_127, var_128]
    var_130 = module_0.parse_args(var_129)
    var_131 = '--python-version'
    var_132 = '3.8'
    var_133 = [var_131, var_132]
    var_134 = module_0.parse_args(var_133)
    var_135 = [var_131, var_24]
    var_136 = module_0.parse_args(var_135)
    var_137 = '--python-version'
    var_138 = 'invalid'
    var_139 = [var_137, var_138]
    var_140 = module_0.parse_args(var_139)
    var_141 = '--python-version'
    var_142 = ''
    var_143 = [var_141, var_142]
    var_144 = module_0.parse_args(var_143)
    var_145 = '--python-version'
    var_146 = [var_145]
    var_147 = module_0.parse_args(var_146)
    var_148 = '--python-version'
    var_149 = '3.8'
    var_150 = 'extra'
    var_151 = [var_148, var_149, var_150]
    var_152 = module_0.parse_args(var_151)
    var_153 = '--python-version'
    var_154 = '3.8'
    var_155 = '--extra'
    var_156 = [var_153, var_154, var_155]
    var_157 = module_0.parse_args(var_156)
    var_158 = '--python-version'
    var_159 = '3.8'
    var_160 = '--extra'
    var_161 = 'value'
    var_162 = [var_158, var_159, var_160, var_161]
    var_163 = module_0.parse_args(var_162)
    var_164 = '--python-version'
    var_165 = '3.8'
    var_166 = '--extra'
    var_167 = 'value'
    var_168 = 'extra'
    var_169 = [var_164, var_165, var_166, var_167, var_168]
    var_170 = module_0.parse_args(var_169)



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    pass



# Parsed testcases at query #10
#--------------------------



def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)
    var_2 = '--check-only'
    var_3 = [var_2]
    var_4 = module_0.parse_args(var_3)
    var_5 = '--diff'
    var_6 = [var_2, var_5]
    var_7 = module_0.parse_args(var_6)
    var_8 = '--line-length'
    var_9 = '100'
    var_10 = [var_8, var_9]
    var_11 = module_0.parse_args(var_10)
    var_12 = '--recursive'
    var_13 = [var_12]
    var_14 = module_0.parse_args(var_13)
    var_15 = '-rc'
    var_16 = [var_15]
    var_17 = module_0.parse_args(var_16)
    var_18 = [var_15, var_2]
    var_19 = module_0.parse_args(var_18)
    var_20 = [var_15, var_8, var_9]
    var_21 = module_0.parse_args(var_20)
    var_22 = '--dont-skip'
    var_23 = [var_15, var_22]
    var_24 = module_0.parse_args(var_23)
    var_25 = '-ns'
    var_26 = [var_15, var_25]
    var_27 = module_0.parse_args(var_26)
    var_28 = [var_15, var_25, var_8, var_9]
    var_29 = module_0.parse_args(var_28)
    var_30 = [var_15, var_25, var_2]
    var_31 = module_0.parse_args(var_30)
    var_32 = [var_15, var_25, var_2, var_8, var_9]
    var_33 = module_0.parse_args(var_32)
    var_34 = [var_15, var_25, var_2, var_8, var_9, var_5]
    var_35 = module_0.parse_args(var_34)
    var_36 = '--force-sort-within-sections'
    var_37 = [var_15, var_25, var_2, var_8, var_9, var_5, var_36]
    var_38 = module_0.parse_args(var_37)
    var_39 = '--honor-case-in-force-sorted-sections'
    var_40 = [var_15, var_25, var_2, var_8, var_9, var_5, var_36, var_39]
    var_41 = module_0.parse_args(var_40)
    var_42 = '--sort-relative-in-force-sorted-sections'
    var_43 = [var_15, var_25, var_2, var_8, var_9, var_5, var_36, var_39, var_42]
    var_44 = module_0.parse_args(var_43)
    var_45 = '--force-alphabetical-sort-within-sections'
    var_46 = [var_15, var_25, var_2, var_8, var_9, var_5, var_36, var_39, var_42, var_45]
    var_47 = module_0.parse_args(var_46)
    var_48 = '--top'
    var_49 = 'module1'
    var_50 = 'module2'
    var_51 = [var_15, var_25, var_2, var_8, var_9, var_5, var_36, var_39, var_42, var_45, var_48, var_49, var_50]
    var_52 = module_0.parse_args(var_51)
    var_53 = '--combine-straight-imports'
    var_54 = [var_15, var_25, var_2, var_8, var_9, var_5, var_36, var_39, var_42, var_45, var_48, var_49, var_50, var_53]
    var_55 = module_0.parse_args(var_54)
    var_56 = '--no-lines-before'
    var_57 = 'section1'
    var_58 = 'section2'
    var_59 = [var_15, var_25, var_2, var_8, var_9, var_5, var_36, var_39, var_42, var_45, var_48, var_49, var_50, var_53, var_56, var_57, var_58]
    var_60 = module_0.parse_args(var_59)



# Parsed testcases at query #11
#--------------------------



def test_case_0():
    var_0 = []
    var_1 = module_0.identify_imports_main(var_0)
    var_2 = 'test_file.py'
    var_3 = [var_2]
    var_4 = module_0.identify_imports_main(var_3)
    var_5 = '-'
    var_6 = [var_5]
    var_7 = 'import os\nimport sys'
    var_8 = '--top-only'
    var_9 = [var_2, var_8]
    var_10 = module_0.identify_imports_main(var_9)
    var_11 = '--unique'
    var_12 = [var_2, var_11]
    var_13 = module_0.identify_imports_main(var_12)
    var_14 = '--packages'
    var_15 = [var_2, var_14]
    var_16 = module_0.identify_imports_main(var_15)
    var_17 = '--modules'
    var_18 = [var_2, var_17]
    var_19 = module_0.identify_imports_main(var_18)
    var_20 = '--attributes'
    var_21 = [var_2, var_20]
    var_22 = module_0.identify_imports_main(var_21)
    var_23 = '--follow-links'
    var_24 = [var_2, var_23]
    var_25 = module_0.identify_imports_main(var_24)
    var_26 = 'file1.py'
    var_27 = 'file2.py'
    var_28 = [var_26, var_27]
    var_29 = module_0.identify_imports_main(var_28)
    var_30 = '--invalid-flag'
    var_31 = [var_2, var_30]
    var_32 = module_0.identify_imports_main(var_31)
    var_33 = []
    var_34 = module_0.identify_imports_main(var_33)
    var_35 = 'empty_file.py'
    var_36 = [var_35]
    var_37 = module_0.identify_imports_main(var_36)
    var_38 = 'multi_import_file.py'
    var_39 = [var_38]
    var_40 = module_0.identify_imports_main(var_39)
    var_41 = 'duplicate_import_file.py'
    var_42 = [var_41]
    var_43 = module_0.identify_imports_main(var_42)
    var_44 = 'relative_import_file.py'
    var_45 = [var_44]
    var_46 = module_0.identify_imports_main(var_45)
    var_47 = 'wildcard_import_file.py'
    var_48 = [var_47]
    var_49 = module_0.identify_imports_main(var_48)
    var_50 = 'conditional_import_file.py'
    var_51 = [var_50]
    var_52 = module_0.identify_imports_main(var_51)
    var_53 = 'try_except_import_file.py'
    var_54 = [var_53]
    var_55 = module_0.identify_imports_main(var_54)
    var_56 = 'function_level_import_file.py'
    var_57 = [var_56]
    var_58 = module_0.identify_imports_main(var_57)



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = []
    var_1 = 'test.py'
    var_2 = [var_1]
    var_3 = '-'
    var_4 = [var_3]
    var_5 = '--unique'
    var_6 = [var_1, var_5]
    var_7 = '--packages'
    var_8 = [var_1, var_7]
    var_9 = '--modules'
    var_10 = [var_1, var_9]
    var_11 = '--attributes'
    var_12 = [var_1, var_11]
    var_13 = '--top-only'
    var_14 = [var_1, var_13]
    var_15 = '--follow-links'
    var_16 = [var_1, var_15]
    var_17 = 'test1.py'
    var_18 = 'test2.py'
    var_19 = [var_17, var_18]
    var_20 = '--invalid'
    var_21 = [var_1, var_20]
    var_22 = []
    var_23 = []
    var_24 = 'nonexistent.py'
    var_25 = [var_24]
    var_26 = 'test.txt'
    var_27 = [var_26]
    var_28 = '.'
    var_29 = [var_28]
    var_30 = 'symlink.py'
    var_31 = [var_30]
    var_32 = 'broken_symlink.py'
    var_33 = [var_32]
    var_34 = '.hidden.py'
    var_35 = [var_34]
    var_36 = '.hidden'
    var_37 = [var_36]
    var_38 = '.hidden_symlink.py'
    var_39 = [var_38]
    var_40 = '.hidden_broken_symlink.py'
    var_41 = [var_40]
    var_42 = '.hidden_dir_symlink'
    var_43 = [var_42]
    var_44 = '.hidden_dir_broken_symlink'
    var_45 = [var_44]
    var_46 = '.hidden_file_symlink.py'
    var_47 = [var_46]
    var_48 = '.hidden_file_broken_symlink.py'
    var_49 = [var_48]
    var_50 = '.hidden_file_dir_symlink'
    var_51 = [var_50]
    var_52 = '.hidden_file_dir_broken_symlink'
    var_53 = [var_52]
    var_54 = '.hidden_file_file_symlink.py'
    var_55 = [var_54]
    var_56 = '.hidden_file_file_broken_symlink.py'
    var_57 = [var_56]
    var_58 = '.hidden_file_file_dir_symlink'
    var_59 = [var_58]
    var_60 = '.hidden_file_file_dir_broken_symlink'
    var_61 = [var_60]
    var_62 = '.hidden_file_file_file_symlink.py'
    var_63 = [var_62]
    var_64 = '.hidden_file_file_file_broken_symlink.py'
    var_65 = [var_64]
    var_66 = '.hidden_file_file_file_dir_symlink'
    var_67 = [var_66]
    var_68 = '.hidden_file_file_file_dir_broken_symlink'
    var_69 = [var_68]
    var_70 = '.hidden_file_file_file_file_symlink.py'
    var_71 = [var_70]
    var_72 = '.hidden_file_file_file_file_broken_symlink.py'
    var_73 = [var_72]
    var_74 = '.hidden_file_file_file_file_dir_symlink'
    var_75 = [var_74]
    var_76 = '.hidden_file_file_file_file_dir_broken_symlink'
    var_77 = [var_76]
    var_78 = '.hidden_file_file_file_file_file_symlink.py'
    var_79 = [var_78]
    var_80 = '.hidden_file_file_file_file_file_broken_symlink.py'
    var_81 = [var_80]
    var_82 = '.hidden_file_file_file_file_file_dir_symlink'
    var_83 = [var_82]
    var_84 = '.hidden_file_file_file_file_file_dir_broken_symlink'
    var_85 = [var_84]
    var_86 = '.hidden_file_file_file_file_file_file_symlink.py'
    var_87 = [var_86]
    var_88 = '.hidden_file_file_file_file_file_file_broken_symlink.py'
    var_89 = [var_88]
    var_90 = '.hidden_file_file_file_file_file_file_dir_symlink'
    var_91 = [var_90]
    var_92 = '.hidden_file_file_file_file_file_file_dir_broken_symlink'
    var_93 = [var_92]
    var_94 = '.hidden_file_file_file_file_file_file_file_symlink.py'
    var_95 = [var_94]
    var_96 = '.hidden_file_file_file_file_file_file_file_broken_symlink.py'
    var_97 = [var_96]
    var_98 = '.hidden_file_file_file_file_file_file_file_dir_symlink'
    var_99 = [var_98]
    var_100 = '.hidden_file_file_file_file_file_file_file_dir_broken_symlink'
    var_101 = [var_100]
    var_102 = '.hidden_file_file_file_file_file_file_file_file_symlink.py'
    var_103 = [var_102]
    var_104 = '.hidden_file_file_file_file_file_file_file_file_broken_symlink.py'
    var_105 = [var_104]
    var_106 = '.hidden_file_file_file_file_file_file_file_file_dir_symlink'
    var_107 = [var_106]
    var_108 = '.hidden_file_file_file_file_file_file_file_file_dir_broken_symlink'
    var_109 = [var_108]
    var_110 = '.hidden_file_file_file_file_file_file_file_file_file_symlink.py'
    var_111 = [var_110]
    var_112 = '.hidden_file_file_file_file_file_file_file_file_file_broken_symlink.py'
    var_113 = [var_112]
    var_114 = '.hidden_file_file_file_file_file_file_file_file_file_dir_symlink'
    var_115 = [var_114]



# Parsed testcases at query #13
#--------------------------



def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)
    var_2 = '--check-only'
    var_3 = [var_2]
    var_4 = module_0.parse_args(var_3)
    var_5 = '--diff'
    var_6 = [var_2, var_5]
    var_7 = module_0.parse_args(var_6)
    var_8 = '--line-length'
    var_9 = '100'
    var_10 = [var_8, var_9]
    var_11 = module_0.parse_args(var_10)
    var_12 = '--recursive'
    var_13 = [var_12]
    var_14 = module_0.parse_args(var_13)
    var_15 = '-rc'
    var_16 = [var_15]
    var_17 = module_0.parse_args(var_16)
    var_18 = '--known-thirdparty'
    var_19 = 'requests'
    var_20 = 'numpy'
    var_21 = [var_18, var_19, var_18, var_20]
    var_22 = module_0.parse_args(var_21)
    var_23 = '--py'
    var_24 = 'auto'
    var_25 = [var_23, var_24]
    var_26 = module_0.parse_args(var_25)
    var_27 = '--py'
    var_28 = 'invalid'
    var_29 = [var_27, var_28]
    var_30 = module_0.parse_args(var_29)
    var_31 = '--float-to-top'
    var_32 = '--dont-float-to-top'
    var_33 = [var_31, var_32]
    var_34 = module_0.parse_args(var_33)
    var_35 = '-k'
    var_36 = [var_35]
    var_37 = module_0.parse_args(var_36)
    var_38 = [var_35, var_32]
    var_39 = module_0.parse_args(var_38)
    var_40 = [var_35, var_8, var_9]
    var_41 = module_0.parse_args(var_40)
    var_42 = [var_35, var_18, var_19, var_18, var_20]
    var_43 = module_0.parse_args(var_42)
    var_44 = [var_35, var_23, var_24]
    var_45 = module_0.parse_args(var_44)
    var_46 = '-k'
    var_47 = '--py'
    var_48 = 'invalid'
    var_49 = [var_46, var_47, var_48]
    var_50 = module_0.parse_args(var_49)
    var_51 = '-k'
    var_52 = '--float-to-top'
    var_53 = '--dont-float-to-top'
    var_54 = [var_51, var_52, var_53]
    var_55 = module_0.parse_args(var_54)
    var_56 = [var_35, var_12]
    var_57 = module_0.parse_args(var_56)
    var_58 = [var_35, var_15]
    var_59 = module_0.parse_args(var_58)
    var_60 = [var_35, var_15, var_52]
    var_61 = module_0.parse_args(var_60)
    var_62 = [var_35, var_15, var_8, var_9]
    var_63 = module_0.parse_args(var_62)
    var_64 = [var_35, var_15, var_18, var_19, var_18, var_20]
    var_65 = module_0.parse_args(var_64)
    var_66 = [var_35, var_15, var_23, var_24]
    var_67 = module_0.parse_args(var_66)
    var_68 = '-k'
    var_69 = '-rc'
    var_70 = '--py'
    var_71 = 'invalid'
    var_72 = [var_68, var_69, var_70, var_71]
    var_73 = module_0.parse_args(var_72)
    var_74 = '-k'
    var_75 = '-rc'
    var_76 = '--float-to-top'
    var_77 = '--dont-float-to-top'
    var_78 = [var_74, var_75, var_76, var_77]
    var_79 = module_0.parse_args(var_78)
    var_80 = [var_35, var_15, var_12]
    var_81 = module_0.parse_args(var_80)
    var_82 = '-ns'
    var_83 = [var_35, var_15, var_82]
    var_84 = module_0.parse_args(var_83)
    var_85 = [var_35, var_15, var_82, var_75]
    var_86 = module_0.parse_args(var_85)
    var_87 = [var_35, var_15, var_82, var_8, var_9]
    var_88 = module_0.parse_args(var_87)
    var_89 = [var_35, var_15, var_82, var_18, var_19, var_18, var_20]
    var_90 = module_0.parse_args(var_89)
    var_91 = [var_35, var_15, var_82, var_23, var_24]
    var_92 = module_0.parse_args(var_91)
    var_93 = '-k'
    var_94 = '-rc'
    var_95 = '-ns'
    var_96 = '--py'
    var_97 = 'invalid'
    var_98 = [var_93, var_94, var_95, var_96, var_97]
    var_99 = module_0.parse_args(var_98)
    var_100 = '-k'
    var_101 = '-rc'
    var_102 = '-ns'
    var_103 = '--float-to-top'
    var_104 = '--dont-float-to-top'
    var_105 = [var_100, var_101, var_102, var_103, var_104]
    var_106 = module_0.parse_args(var_105)
    var_107 = [var_35, var_15, var_82, var_12]
    var_108 = module_0.parse_args(var_107)



####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------




# Parsed testcases at query #2
#--------------------------




# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)
    var_2 = '--check-only'
    var_3 = [var_2]
    var_4 = module_0.parse_args(var_3)
    var_5 = '--diff'
    var_6 = [var_2, var_5]
    var_7 = module_0.parse_args(var_6)
    var_8 = '--line-length'
    var_9 = '100'
    var_10 = [var_8, var_9]
    var_11 = module_0.parse_args(var_10)
    var_12 = '--recursive'
    var_13 = [var_12]
    var_14 = module_0.parse_args(var_13)
    var_15 = '-rc'
    var_16 = [var_15]
    var_17 = module_0.parse_args(var_16)
    var_18 = [var_8, var_9, var_12]
    var_19 = module_0.parse_args(var_18)
    var_20 = [var_8, var_9, var_15]
    var_21 = module_0.parse_args(var_20)
    var_22 = [var_8, var_9, var_12, var_15]
    var_23 = module_0.parse_args(var_22)
    var_24 = '-ns'
    var_25 = [var_8, var_9, var_15, var_24]
    var_26 = module_0.parse_args(var_25)
    var_27 = [var_8, var_9, var_12, var_15]
    var_28 = module_0.parse_args(var_27)
    var_29 = [var_8, var_9, var_12, var_15, var_2]
    var_30 = module_0.parse_args(var_29)
    var_31 = '--dont-skip'
    var_32 = [var_8, var_9, var_12, var_15, var_2, var_31]
    var_33 = module_0.parse_args(var_32)
    var_34 = '--dont-order-by-type'
    var_35 = [var_8, var_9, var_12, var_15, var_2, var_31, var_34]
    var_36 = module_0.parse_args(var_35)
    var_37 = '--dont-follow-links'
    var_38 = [var_8, var_9, var_12, var_15, var_2, var_31, var_37]
    var_39 = module_0.parse_args(var_38)
    var_40 = '--dont-float-to-top'
    var_41 = [var_8, var_9, var_12, var_15, var_2, var_31, var_40]
    var_42 = module_0.parse_args(var_41)
    var_43 = '--line-length'
    var_44 = '100'
    var_45 = '--recursive'
    var_46 = '-rc'
    var_47 = '--check-only'
    var_48 = '--dont-skip'
    var_49 = '--dont-float-to-top'
    var_50 = '--float-to-top'
    var_51 = [var_43, var_44, var_45, var_46, var_47, var_48, var_49, var_50]
    var_52 = module_0.parse_args(var_51)
    var_53 = '--multi-line'
    var_54 = '0'
    var_55 = [var_48, var_49, var_51, var_15, var_44, var_31, var_53, var_54]
    var_56 = module_0.parse_args(var_55)
    var_57 = 'line_length'
    var_58 = 'deprecated_flags'
    var_59 = 'check_only'
    var_60 = 'multi_line_output'
    var_61 = 100
    var_62 = [var_51, var_15, var_31]
    var_63 = True
    var_64 = 0
    var_65 = 'GRID'
    var_66 = [var_48, var_49, var_51, var_15, var_44, var_31, var_53, var_65]
    var_67 = module_0.parse_args(var_66)
    var_68 = 'VERTICAL_HANGING_INDENT'
    var_69 = [var_48, var_49, var_51, var_15, var_44, var_31, var_53, var_68]
    var_70 = module_0.parse_args(var_69)
    var_71 = 'VERTICAL_HANGING_INDENT_BRACKET'
    var_72 = [var_48, var_49, var_51, var_15, var_44, var_31, var_53, var_71]
    var_73 = module_0.parse_args(var_72)
    var_74 = [var_48, var_49, var_51, var_15, var_44, var_31, var_53, var_71]
    var_75 = module_0.parse_args(var_74)
    var_76 = [var_48, var_49, var_51, var_15, var_44, var_31, var_53, var_71]
    var_77 = module_0.parse_args(var_76)



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    pass



# Parsed testcases at query #5
#--------------------------



def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)
    var_2 = '--check-only'
    var_3 = [var_2]
    var_4 = module_0.parse_args(var_3)
    var_5 = '--diff'
    var_6 = [var_2, var_5]
    var_7 = module_0.parse_args(var_6)
    var_8 = '--line-length'
    var_9 = '80'
    var_10 = [var_8, var_9]
    var_11 = module_0.parse_args(var_10)
    var_12 = '--recursive'
    var_13 = [var_12]
    var_14 = module_0.parse_args(var_13)
    var_15 = '-rc'
    var_16 = [var_15]
    var_17 = module_0.parse_args(var_16)
    var_18 = '--known-thirdparty'
    var_19 = 'requests'
    var_20 = 'numpy'
    var_21 = [var_18, var_19, var_18, var_20]
    var_22 = module_0.parse_args(var_21)
    var_23 = '--py'
    var_24 = 'auto'
    var_25 = [var_23, var_24]
    var_26 = module_0.parse_args(var_25)
    var_27 = '--py'
    var_28 = 'invalid'
    var_29 = [var_27, var_28]
    var_30 = module_0.parse_args(var_29)
    var_31 = '--float-to-top'
    var_32 = '--dont-float-to-top'
    var_33 = [var_31, var_32]
    var_34 = module_0.parse_args(var_33)
    var_35 = 'All test cases passed!'
    var_36 = print(var_35)



####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------



def test_case_0():
    var_0 = []
    var_1 = module_0.identify_imports_main(var_0)
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'test.py'
    var_5 = [var_4]
    var_6 = module_0.identify_imports_main(var_5)
    var_7 = 'sys'
    var_8 = 'stdout'
    var_9 = '-'
    var_10 = [var_9]
    var_11 = module_0.identify_imports_main(var_10)
    var_12 = 'os.path'
    var_13 = 'join'
    var_14 = 'test.py'
    var_15 = '--packages'
    var_16 = [var_14, var_15]
    var_17 = module_0.identify_imports_main(var_16)
    var_18 = 'os.path'
    var_19 = 'join'
    var_20 = 'test.py'
    var_21 = '--modules'
    var_22 = [var_20, var_21]
    var_23 = module_0.identify_imports_main(var_22)
    var_24 = 'os.path'
    var_25 = 'join'
    var_26 = 'test.py'
    var_27 = '--attributes'
    var_28 = [var_26, var_27]
    var_29 = module_0.identify_imports_main(var_28)
    var_30 = 'sys'
    var_31 = 'stdout'
    var_32 = 'test.py'
    var_33 = '--top-only'
    var_34 = [var_32, var_33]
    var_35 = module_0.identify_imports_main(var_34)
    var_36 = 'os'
    var_37 = 'path'
    var_38 = 'test.py'
    var_39 = '--follow-links'
    var_40 = [var_38, var_39]
    var_41 = module_0.identify_imports_main(var_40)
    var_42 = 'All tests passed!'
    var_43 = print(var_42)



# Parsed testcases at query #2
#--------------------------




# Parsed testcases at query #3
#--------------------------


import builtins as module_1


def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)
    var_2 = '--check-only'
    var_3 = [var_2]
    var_4 = module_0.parse_args(var_3)
    var_5 = '--diff'
    var_6 = [var_2, var_5]
    var_7 = module_0.parse_args(var_6)
    var_8 = '--line-length'
    var_9 = '100'
    var_10 = [var_8, var_9]
    var_11 = module_0.parse_args(var_10)
    var_12 = '--recursive'
    var_13 = [var_12]
    var_14 = module_0.parse_args(var_13)
    var_15 = '-rc'
    var_16 = [var_15]
    var_17 = module_0.parse_args(var_16)
    var_18 = '--known-thirdparty'
    var_19 = 'module1'
    var_20 = 'module2'
    var_21 = [var_18, var_19, var_20]
    var_22 = module_0.parse_args(var_21)
    var_23 = '--py'
    var_24 = 'auto'
    var_25 = [var_23, var_24]
    var_26 = module_0.parse_args(var_25)
    var_27 = '--py'
    var_28 = 'invalid'
    var_29 = [var_27, var_28]
    var_30 = module_0.parse_args(var_29)
    var_31 = '--float-to-top'
    var_32 = '--dont-float-to-top'
    var_33 = [var_31, var_32]
    var_34 = module_0.parse_args(var_33)
    var_35 = '--section-default'
    var_36 = 'STDLIB'
    var_37 = [var_35, var_36]
    var_38 = module_0.parse_args(var_37)
    var_39 = '--sort-order'
    var_40 = 'natural'
    var_41 = [var_39, var_40]
    var_42 = module_0.parse_args(var_41)
    var_43 = '--formatter'
    var_44 = 'custom'
    var_45 = [var_43, var_44]
    var_46 = module_0.parse_args(var_45)
    var_47 = '--line-ending'
    var_48 = 'lf'
    var_49 = [var_47, var_48]
    var_50 = module_0.parse_args(var_49)
    var_51 = '--indent'
    var_52 = '  '
    var_53 = [var_51, var_52]
    var_54 = module_0.parse_args(var_53)
    var_55 = '--multi-line'
    var_56 = '3'
    var_57 = [var_55, var_56]
    var_58 = module_0.parse_args(var_57)
    var_59 = 'multi_line_output'
    var_60 = 3
    var_61 = 'VERTICAL_HANGING_INDENT'
    var_62 = [var_55, var_61]
    var_63 = module_0.parse_args(var_62)
    var_64 = '--multi-line'
    var_65 = 'invalid'
    var_66 = [var_64, var_65]
    var_67 = module_0.parse_args(var_66)
    var_68 = '--multi-line'
    var_69 = '99'
    var_70 = [var_68, var_69]
    var_71 = module_0.parse_args(var_70)
    var_72 = '--multi-line'
    var_73 = 'INVALID'
    var_74 = [var_72, var_73]
    var_75 = module_0.parse_args(var_74)
    var_76 = '--multi-line'
    var_77 = '3.14'
    var_78 = [var_76, var_77]
    var_79 = module_0.parse_args(var_78)
    var_80 = '--multi-line'
    var_81 = 'three'
    var_82 = [var_80, var_81]
    var_83 = module_0.parse_args(var_82)
    var_84 = '--multi-line'
    var_85 = ''
    var_86 = [var_84, var_85]
    var_87 = module_0.parse_args(var_86)
    var_88 = '--multi-line'
    var_89 = ' '
    var_90 = [var_88, var_89]
    var_91 = module_0.parse_args(var_90)
    var_92 = '--multi-line'
    var_93 = '!'
    var_94 = [var_92, var_93]
    var_95 = module_0.parse_args(var_94)
    var_96 = '--multi-line'
    var_97 = '😀'
    var_98 = [var_96, var_97]
    var_99 = module_0.parse_args(var_98)
    var_100 = '--multi-line'
    var_101 = '\n'
    var_102 = [var_100, var_101]
    var_103 = module_0.parse_args(var_102)
    var_104 = '--multi-line'
    var_105 = '\x00'
    var_106 = [var_104, var_105]
    var_107 = module_0.parse_args(var_106)
    var_108 = '--multi-line'
    var_109 = b'\x00'
    var_110 = [var_108, var_109]
    var_111 = module_0.parse_args(var_110)
    var_112 = '--multi-line'
    var_113 = module_1.object()
    var_114 = [var_112, var_113]
    var_115 = module_0.parse_args(var_114)
    var_116 = '--multi-line'
    var_117 = None
    var_118 = [var_116, var_117]
    var_119 = module_0.parse_args(var_118)
    var_120 = '--multi-line'
    var_121 = True
    var_122 = [var_120, var_121]
    var_123 = module_0.parse_args(var_122)
    var_124 = '--multi-line'
    var_125 = False
    var_126 = [var_124, var_125]
    var_127 = module_0.parse_args(var_126)
    var_128 = '--multi-line'
    var_129 = []
    var_130 = [var_128, var_129]
    var_131 = module_0.parse_args(var_130)
    var_132 = '--multi-line'
    var_133 = ()
    var_134 = [var_132, var_133]
    var_135 = module_0.parse_args(var_134)
    var_136 = '--multi-line'
    var_137 = {}
    var_138 = [var_136, var_137]
    var_139 = module_0.parse_args(var_138)
    var_140 = '--multi-line'
    var_141 = set()
    var_142 = [var_140, var_141]
    var_143 = module_0.parse_args(var_142)
    var_144 = '--multi-line'
    var_145 = frozenset()
    var_146 = [var_144, var_145]
    var_147 = module_0.parse_args(var_146)
    var_148 = '--multi-line'
    var_149 = b''
    var_150 = [var_148, var_149]
    var_151 = module_0.parse_args(var_150)
    var_152 = '--multi-line'
    var_153 = bytearray()
    var_154 = [var_152, var_153]
    var_155 = module_0.parse_args(var_154)
    var_156 = '--multi-line'
    var_157 = b''
    var_158 = memoryview(var_157)
    var_159 = [var_156, var_158]
    var_160 = module_0.parse_args(var_159)
    var_161 = '--multi-line'
    var_162 = 0
    var_163 = range(var_162)
    var_164 = [var_161, var_163]
    var_165 = module_0.parse_args(var_164)
    var_166 = '--multi-line'
    var_167 = 0
    var_168 = [var_166, var_163]
    var_169 = module_0.parse_args(var_168)
    var_170 = '--multi-line'
    var_171 = [var_170, var_167]
    var_172 = module_0.parse_args(var_171)
    var_173 = '--multi-line'
    var_174 = module_0.parse_args(var_167)
    var_175 = '--multi-line'
    var_176 = 0
    var_177 = complex(var_176, var_176)
    var_178 = [var_175, var_177]
    var_179 = module_0.parse_args(var_178)
    var_180 = '--multi-line'
    var_181 = 3.14
    var_182 = [var_180, var_181]
    var_183 = module_0.parse_args(var_182)
    var_184 = '--multi-line'
    var_185 = 42
    var_186 = [var_184, var_185]
    var_187 = module_0.parse_args(var_186)



# Parsed testcases at query #4
#--------------------------




# Parsed testcases at query #5
#--------------------------



def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)
    var_2 = '--help'
    var_3 = [var_2]
    var_4 = module_0.parse_args(var_3)
    var_5 = '--version'
    var_6 = [var_5]
    var_7 = module_0.parse_args(var_6)
    var_8 = '--check'
    var_9 = [var_8]
    var_10 = module_0.parse_args(var_9)
    var_11 = '--diff'
    var_12 = [var_11]
    var_13 = module_0.parse_args(var_12)
    var_14 = '--quiet'
    var_15 = [var_14]
    var_16 = module_0.parse_args(var_15)
    var_17 = '--verbose'
    var_18 = [var_17]
    var_19 = module_0.parse_args(var_18)
    var_20 = '--fss'
    var_21 = [var_20]
    var_22 = module_0.parse_args(var_21)
    var_23 = '--ot'
    var_24 = [var_23]
    var_25 = module_0.parse_args(var_24)
    var_26 = '--dt'
    var_27 = [var_26]
    var_28 = module_0.parse_args(var_27)
    var_29 = '--reverse-sort'
    var_30 = [var_29]
    var_31 = module_0.parse_args(var_30)
    var_32 = '--fas'
    var_33 = [var_32]
    var_34 = module_0.parse_args(var_33)
    var_35 = '--fass'
    var_36 = [var_35]
    var_37 = module_0.parse_args(var_36)
    var_38 = '--only-sections'
    var_39 = [var_38]
    var_40 = module_0.parse_args(var_39)
    var_41 = '--ds'
    var_42 = [var_41]
    var_43 = module_0.parse_args(var_42)
    var_44 = '--csi'
    var_45 = [var_44]
    var_46 = module_0.parse_args(var_45)
    var_47 = '--float-to-top'
    var_48 = [var_47]
    var_49 = module_0.parse_args(var_48)
    var_50 = '--dont-float-to-top'
    var_51 = [var_50]
    var_52 = module_0.parse_args(var_51)
    var_53 = '--ca'
    var_54 = [var_53]
    var_55 = module_0.parse_args(var_54)
    var_56 = '--remove-redundant-aliases'
    var_57 = [var_56]
    var_58 = module_0.parse_args(var_57)
    var_59 = '--sl'
    var_60 = [var_59]
    var_61 = module_0.parse_args(var_60)
    var_62 = '--nsl'
    var_63 = 'module1'
    var_64 = 'module2'
    var_65 = [var_62, var_63, var_64]
    var_66 = module_0.parse_args(var_65)
    var_67 = '--sd'
    var_68 = 'THIRDPARTY'
    var_69 = [var_67, var_68]
    var_70 = module_0.parse_args(var_69)
    var_71 = '--src'
    var_72 = '/path/to/src'
    var_73 = [var_71, var_72]
    var_74 = module_0.parse_args(var_73)
    var_75 = '--builtin'
    var_76 = [var_75, var_63, var_64]
    var_77 = module_0.parse_args(var_76)
    var_78 = '--extra-builtin'
    var_79 = [var_78, var_63, var_64]
    var_80 = module_0.parse_args(var_79)
    var_81 = '--future'
    var_82 = [var_81, var_63, var_64]
    var_83 = module_0.parse_args(var_82)
    var_84 = '--thirdparty'
    var_85 = [var_84, var_63, var_64]
    var_86 = module_0.parse_args(var_85)
    var_87 = '--project'
    var_88 = [var_87, var_63, var_64]
    var_89 = module_0.parse_args(var_88)
    var_90 = '--known-local-folder'
    var_91 = 'folder1'
    var_92 = 'folder2'
    var_93 = [var_90, var_91, var_92]
    var_94 = module_0.parse_args(var_93)
    var_95 = '--virtual-env'
    var_96 = '/path/to/venv'
    var_97 = [var_95, var_96]
    var_98 = module_0.parse_args(var_97)
    var_99 = '--conda-env'
    var_100 = '/path/to/conda/env'
    var_101 = [var_99, var_100]
    var_102 = module_0.parse_args(var_101)
    var_103 = '--py'
    var_104 = '3.8'
    var_105 = [var_103, var_104]
    var_106 = module_0.parse_args(var_105)
    var_107 = '--line-length'
    var_108 = '100'
    var_109 = [var_107, var_108]
    var_110 = module_0.parse_args(var_109)
    var_111 = '--wrap-length'
    var_112 = '80'
    var_113 = [var_111, var_112]
    var_114 = module_0.parse_args(var_113)
    var_115 = '--indent'
    var_116 = '    '
    var_117 = [var_115, var_116]
    var_118 = module_0.parse_args(var_117)
    var_119 = '--tab-width'
    var_120 = '4'
    var_121 = [var_119, var_120]
    var_122 = module_0.parse_args(var_121)
    var_123 = '--lbi'
    var_124 = '2'
    var_125 = [var_123, var_124]
    var_126 = module_0.parse_args(var_125)
    var_127 = '--lai'
    var_128 = [var_127, var_124]
    var_129 = module_0.parse_args(var_128)
    var_130 = '--lbt'
    var_131 = [var_130, var_124]
    var_132 = module_0.parse_args(var_131)
    var_133 = '--tc'
    var_134 = [var_133]
    var_135 = module_0.parse_args(var_134)
    var_136 = '--up'
    var_137 = [var_136]
    var_138 = module_0.parse_args(var_137)
    var_139 = '--fgw'
    var_140 = [var_139, var_124]
    var_141 = module_0.parse_args(var_140)
    var_142 = '--multi-line'
    var_143 = '3'
    var_144 = [var_142, var_143]
    var_145 = module_0.parse_args(var_144)
    var_146 = '--ensure-newline-before-comments'
    var_147 = [var_146]
    var_148 = module_0.parse_args(var_147)
    var_149 = '--case-sensitive'
    var_150 = [var_149]
    var_151 = module_0.parse_args(var_150)
    var_152 = '--honor-noqa'
    var_153 = [var_152]
    var_154 = module_0.parse_args(var_153)
    var_155 = '--treat-comment-as-code'
    var_156 = '# noqa'
    var_157 = '# isort: skip'
    var_158 = [var_155, var_156, var_157]
    var_159 = module_0.parse_args(var_158)
    var_160 = '--treat-all-comment-as-code'
    var_161 = [var_160]
    var_162 = module_0.parse_args(var_161)
    var_163 = '--formatter'
    var_164 = 'my_formatter'
    var_165 = [var_163, var_164]
    var_166 = module_0.parse_args(var_165)
    var_167 = '--color'
    var_168 = [var_167]
    var_169 = module_0.parse_args(var_168)
    var_170 = '--ext-format'
    var_171 = '.py'
    var_172 = [var_170, var_171]
    var_173 = module_0.parse_args(var_172)
    var_174 = '--star-first'
    var_175 = [var_174]
    var_176 = module_0.parse_args(var_175)
    var_177 = '--split-on-trailing-comma'
    var_178 = [var_177]
    var_179 = module_0.parse_args(var_178)
    var_180 = '--nis'
    var_181 = [var_180]
    var_182 = module_0.parse_args(var_181)
    var_183 = '--ls'
    var_184 = [var_183]
    var_185 = module_0.parse_args(var_184)
    var_186 = '--lss'
    var_187 = [var_186]
    var_188 = module_0.parse_args(var_187)
    var_189 = '--rr'
    var_190 = [var_189]
    var_191 = module_0.parse_args(var_190)
    var_192 = '--top'
    var_193 = [var_192, var_63, var_64]
    var_194 = module_0.parse_args(var_193)



# Parsed testcases at query #6
#--------------------------




# Parsed testcases at query #7
#--------------------------


def test_case_0():
    pass



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    pass



# Parsed testcases at query #9
#--------------------------



def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)
    var_2 = '--check-only'
    var_3 = [var_2]
    var_4 = module_0.parse_args(var_3)
    var_5 = '--diff'
    var_6 = [var_2, var_5]
    var_7 = module_0.parse_args(var_6)
    var_8 = '--line-length'
    var_9 = '80'
    var_10 = [var_8, var_9]
    var_11 = module_0.parse_args(var_10)
    var_12 = '--recursive'
    var_13 = [var_12]
    var_14 = module_0.parse_args(var_13)
    var_15 = '-rc'
    var_16 = [var_15]
    var_17 = module_0.parse_args(var_16)
    var_18 = '--known-thirdparty'
    var_19 = 'requests'
    var_20 = 'numpy'
    var_21 = [var_18, var_19, var_18, var_20]
    var_22 = module_0.parse_args(var_21)
    var_23 = '--py'
    var_24 = 'invalid'
    var_25 = [var_23, var_24]
    var_26 = module_0.parse_args(var_25)
    var_27 = '--py'
    var_28 = '3.8'
    var_29 = [var_27, var_28]
    var_30 = module_0.parse_args(var_29)
    var_31 = 'auto'
    var_32 = [var_27, var_31]
    var_33 = module_0.parse_args(var_32)
    var_34 = '--multi-line'
    var_35 = '5'
    var_36 = [var_34, var_35]
    var_37 = module_0.parse_args(var_36)
    var_38 = 'VERTICAL_GRID_GROUPED'
    var_39 = [var_34, var_38]
    var_40 = module_0.parse_args(var_39)
    var_41 = '--float-to-top'
    var_42 = '--dont-float-to-top'
    var_43 = [var_41, var_42]
    var_44 = module_0.parse_args(var_43)
    var_45 = '--dont-float-to-top'
    var_46 = [var_45]
    var_47 = module_0.parse_args(var_46)
    var_48 = '--dont-order-by-type'
    var_49 = [var_48]
    var_50 = module_0.parse_args(var_49)
    var_51 = '--dont-follow-links'
    var_52 = [var_51]
    var_53 = module_0.parse_args(var_52)
    var_54 = [var_42, var_5]
    var_55 = module_0.parse_args(var_54)
    var_56 = [var_42, var_5, var_8, var_9]
    var_57 = module_0.parse_args(var_56)
    var_58 = [var_42, var_5, var_8, var_9, var_18, var_19]
    var_59 = module_0.parse_args(var_58)
    var_60 = [var_42, var_5, var_8, var_9, var_18, var_19, var_27, var_28]
    var_61 = module_0.parse_args(var_60)
    var_62 = [var_42, var_5, var_8, var_9, var_18, var_19, var_27, var_28, var_34, var_35]
    var_63 = module_0.parse_args(var_62)
    var_64 = [var_42, var_5, var_8, var_9, var_18, var_19, var_27, var_28, var_34, var_35, var_45]
    var_65 = module_0.parse_args(var_64)
    var_66 = [var_42, var_5, var_8, var_9, var_18, var_19, var_27, var_28, var_34, var_35, var_45, var_48]
    var_67 = module_0.parse_args(var_66)
    var_68 = [var_42, var_5, var_8, var_9, var_18, var_19, var_27, var_28, var_34, var_35, var_45, var_48, var_51]
    var_69 = module_0.parse_args(var_68)
    var_70 = [var_42, var_5, var_8, var_9, var_18, var_19, var_27, var_28, var_34, var_35, var_45, var_48, var_51, var_15]
    var_71 = module_0.parse_args(var_70)



# Parsed testcases at query #10
#--------------------------




