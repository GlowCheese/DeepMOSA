####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = 'x = 1'
    var_2 = [var_0, var_1]
    var_3 = 'THIRDPARTY'
    var_4 = 'straight'
    var_5 = 'from'
    var_6 = 'os'
    var_7 = 'sys'
    var_8 = []
    var_9 = []
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'collections'
    var_12 = 'itertools'
    var_13 = 'defaultdict'
    var_14 = [var_13]
    var_15 = 'chain'
    var_16 = [var_15]
    var_17 = {var_11: var_14, var_12: var_16}
    var_18 = {var_4: var_10, var_5: var_17}
    var_19 = {var_3: var_18}
    var_20 = 0
    var_21 = 2
    var_22 = '\n'
    var_23 = module_0.ParsedContent()
    var_24 = module_1.Config()
    var_25 = module_2.sorted_imports(var_23, var_24)
    assert var_25 == 'import os\nimport sys\n\nfrom collections import defaultdict\nfrom itertools import chain\n\nx = 1\n'
    var_26 = [var_1]
    var_27 = {}
    var_28 = -1
    var_29 = 1
    var_30 = module_0.ParsedContent()
    var_31 = module_1.Config()
    var_32 = module_2.sorted_imports(var_30, var_31)
    assert var_32 == 'x = 1\n'
    var_33 = [var_0, var_1]
    var_34 = 'FUTURE'
    var_35 = 'STDLIB'
    var_36 = '__future__'
    var_37 = 'annotations'
    var_38 = [var_37]
    var_39 = {var_36: var_38}
    var_40 = {}
    var_41 = {var_4: var_39, var_5: var_40}
    var_42 = []
    var_43 = {var_6: var_42}
    var_44 = 'argv'
    var_45 = [var_44]
    var_46 = {var_7: var_45}
    var_47 = {var_4: var_43, var_5: var_46}
    var_48 = {var_34: var_41, var_35: var_47}
    var_49 = module_0.ParsedContent()
    var_50 = 'future'
    var_51 = 'stdlib'
    var_52 = 'Future'
    var_53 = 'Standard Library'
    var_54 = {var_50: var_52, var_51: var_53}
    var_55 = module_1.Config()
    var_56 = module_2.sorted_imports(var_49, var_55)
    assert var_56 == '# Future\nfrom __future__ import annotations\n\n# Standard Library\nimport os\n\nfrom sys import argv\n\nx = 1\n'
    var_57 = [var_0, var_1]
    var_58 = 'FIRSTPARTY'
    var_59 = 'django'
    var_60 = []
    var_61 = {var_59: var_60}
    var_62 = {}
    var_63 = {var_4: var_61, var_5: var_62}
    var_64 = 'myapp'
    var_65 = []
    var_66 = {var_64: var_65}
    var_67 = {}
    var_68 = {var_4: var_66, var_5: var_67}
    var_69 = {var_3: var_63, var_58: var_68}
    var_70 = module_0.ParsedContent()
    var_71 = 'LOCALFOLDER'
    var_72 = [var_71]
    var_73 = module_1.Config()
    var_74 = '.utils'
    var_75 = []
    var_76 = {var_74: var_75}
    var_77 = {}
    var_78 = module_2.sorted_imports(var_70, var_73)
    var_79 = [var_0, var_1]
    var_80 = 'unused'
    var_81 = []
    var_82 = []
    var_83 = []
    var_84 = {var_6: var_81, var_7: var_82, var_80: var_83}
    var_85 = [var_13]
    var_86 = {var_11: var_85}
    var_87 = {var_4: var_84, var_5: var_86}
    var_88 = {var_3: var_87}
    var_89 = module_0.ParsedContent()
    var_90 = [var_80]
    var_91 = module_1.Config()
    var_92 = module_2.sorted_imports(var_89, var_91)



# Parsed testcases at query #2
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'x = 1'
    var_1 = [var_0]
    var_2 = 'THIRDPARTY'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'zlib'
    var_6 = 'os'
    var_7 = 'import zlib'
    var_8 = [var_7]
    var_9 = 'import os'
    var_10 = [var_9]
    var_11 = {var_5: var_8, var_6: var_10}
    var_12 = 'sys'
    var_13 = 'from sys import path'
    var_14 = [var_13]
    var_15 = {var_12: var_14}
    var_16 = {var_3: var_11, var_4: var_15}
    var_17 = {var_2: var_16}
    var_18 = 0
    var_19 = 1
    var_20 = '\n'
    var_21 = module_0.ParsedContent()
    var_22 = module_1.Config()
    var_23 = module_2.sorted_imports(var_21, var_22)
    assert var_23 == 'import os\nimport zlib\n\nfrom sys import path\n\nx = 1\n'
    var_24 = [var_0]
    var_25 = 'FUTURE'
    var_26 = '__future__'
    var_27 = 'from __future__ import annotations'
    var_28 = [var_27]
    var_29 = {var_26: var_28}
    var_30 = {}
    var_31 = {var_3: var_29, var_4: var_30}
    var_32 = [var_7]
    var_33 = [var_9]
    var_34 = {var_5: var_32, var_6: var_33}
    var_35 = [var_13]
    var_36 = {var_12: var_35}
    var_37 = {var_3: var_34, var_4: var_36}
    var_38 = {var_25: var_31, var_2: var_37}
    var_39 = module_0.ParsedContent()
    var_40 = True
    var_41 = module_1.Config()
    var_42 = module_2.sorted_imports(var_39, var_41)
    assert var_42 == 'from __future__ import annotations\nimport os\nimport zlib\n\nfrom sys import path\n\nx = 1\n'
    var_43 = [var_0]
    var_44 = [var_7]
    var_45 = [var_9]
    var_46 = {var_5: var_44, var_6: var_45}
    var_47 = [var_13]
    var_48 = {var_12: var_47}
    var_49 = {var_3: var_46, var_4: var_48}
    var_50 = {var_2: var_49}
    var_51 = module_0.ParsedContent()
    var_52 = True
    var_53 = module_1.Config()
    var_54 = module_2.sorted_imports(var_51, var_53)
    assert var_54 == 'import os\nimport zlib\n\nfrom sys import path\n\nx = 1\n'
    var_55 = [var_0]
    var_56 = [var_7]
    var_57 = [var_9]
    var_58 = {var_5: var_56, var_6: var_57}
    var_59 = [var_13]
    var_60 = {var_12: var_59}
    var_61 = {var_3: var_58, var_4: var_60}
    var_62 = {var_2: var_61}
    var_63 = module_0.ParsedContent()
    var_64 = True
    var_65 = module_1.Config()
    var_66 = module_2.sorted_imports(var_63, var_65)
    assert var_66 == 'from sys import path\n\nimport os\nimport zlib\n\nx = 1\n'
    var_67 = [var_0]
    var_68 = [var_7]
    var_69 = [var_9]
    var_70 = {var_5: var_68, var_6: var_69}
    var_71 = 'from sys import *'
    var_72 = [var_71]
    var_73 = 'from os import path'
    var_74 = [var_73]
    var_75 = {var_12: var_72, var_6: var_74}
    var_76 = {var_3: var_70, var_4: var_75}
    var_77 = {var_2: var_76}
    var_78 = module_0.ParsedContent()
    var_79 = True
    var_80 = module_1.Config()
    var_81 = module_2.sorted_imports(var_78, var_80)
    assert var_81 == 'import os\nimport zlib\n\nfrom sys import *\nfrom os import path\n\nx = 1\n'



# Parsed testcases at query #3
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = 0
    var_5 = '\n'
    var_6 = {}
    var_7 = {}
    var_8 = module_0.ParsedContent()
    var_9 = "print('hello')"
    var_10 = [var_9]
    var_11 = {}
    var_12 = -1
    var_13 = 1
    var_14 = {}
    var_15 = {}
    var_16 = module_0.ParsedContent()
    var_17 = [var_0]
    var_18 = 'THIRDPARTY'
    var_19 = 'straight'
    var_20 = 'from'
    var_21 = 'zlib'
    var_22 = 'os'
    var_23 = []
    var_24 = []
    var_25 = {var_21: var_23, var_22: var_24}
    var_26 = 'sys'
    var_27 = 'json'
    var_28 = 'path'
    var_29 = [var_28]
    var_30 = 'load'
    var_31 = [var_30]
    var_32 = {var_26: var_29, var_27: var_31}
    var_33 = {var_19: var_25, var_20: var_32}
    var_34 = {var_18: var_33}
    var_35 = {}
    var_36 = {}
    var_37 = module_0.ParsedContent()
    var_38 = 'import os\nimport zlib\n\nfrom json import load\nfrom sys import path\n\n'
    var_39 = [var_0]
    var_40 = []
    var_41 = []
    var_42 = {var_22: var_40, var_21: var_41}
    var_43 = [var_30]
    var_44 = [var_28]
    var_45 = {var_27: var_43, var_26: var_44}
    var_46 = {var_19: var_42, var_20: var_45}
    var_47 = {var_18: var_46}
    var_48 = {}
    var_49 = {}
    var_50 = module_0.ParsedContent()
    var_51 = True
    var_52 = module_1.Config()
    var_53 = 'import zlib\nimport os\n\nfrom sys import path\nfrom json import load\n\n'
    var_54 = module_2.sorted_imports(var_50, var_52)
    var_55 = [var_0]
    var_56 = 'FIRSTPARTY'
    var_57 = []
    var_58 = {var_22: var_57}
    var_59 = [var_28]
    var_60 = {var_26: var_59}
    var_61 = {var_19: var_58, var_20: var_60}
    var_62 = 'my_module'
    var_63 = []
    var_64 = {var_62: var_63}
    var_65 = 'my_package'
    var_66 = 'utils'
    var_67 = [var_66]
    var_68 = {var_65: var_67}
    var_69 = {var_19: var_64, var_20: var_68}
    var_70 = {var_18: var_61, var_56: var_69}
    var_71 = {}
    var_72 = {}
    var_73 = module_0.ParsedContent()
    var_74 = 'LOCALFOLDER'
    var_75 = [var_74]
    var_76 = module_1.Config()
    var_77 = 'import os\n\nfrom sys import path\n\nimport my_module\n\nfrom my_package import utils\n\n'
    var_78 = module_2.sorted_imports(var_73, var_76)
    var_79 = [var_0]
    var_80 = 'FUTURE'
    var_81 = '__future__'
    var_82 = 'annotations'
    var_83 = [var_82]
    var_84 = {var_81: var_83}
    var_85 = {}
    var_86 = {var_19: var_84, var_20: var_85}
    var_87 = []
    var_88 = []
    var_89 = {var_21: var_87, var_22: var_88}
    var_90 = [var_28]
    var_91 = {var_26: var_90}
    var_92 = {var_19: var_89, var_20: var_91}
    var_93 = {var_80: var_86, var_18: var_92}
    var_94 = {}
    var_95 = {}
    var_96 = module_0.ParsedContent()
    var_97 = True
    var_98 = module_1.Config()
    var_99 = 'from __future__ import annotations\nimport os\nimport zlib\n\nfrom sys import path\n\n'
    var_100 = module_2.sorted_imports(var_96, var_98)
    var_101 = [var_0]
    var_102 = {}
    var_103 = '*'
    var_104 = [var_103]
    var_105 = [var_28]
    var_106 = [var_30]
    var_107 = {var_22: var_104, var_26: var_105, var_27: var_106}
    var_108 = {var_19: var_102, var_20: var_107}
    var_109 = {var_18: var_108}
    var_110 = {}
    var_111 = {}
    var_112 = module_0.ParsedContent()
    var_113 = True
    var_114 = module_1.Config()
    var_115 = 'from os import *\nfrom json import load\nfrom sys import path\n\n'
    var_116 = module_2.sorted_imports(var_112, var_114)
    var_117 = [var_0]
    var_118 = []
    var_119 = []
    var_120 = {var_22: var_118, var_21: var_119}
    var_121 = [var_28]
    var_122 = [var_30]
    var_123 = {var_26: var_121, var_27: var_122}
    var_124 = {var_19: var_120, var_20: var_123}
    var_125 = {var_18: var_124}
    var_126 = {}
    var_127 = {}
    var_128 = module_0.ParsedContent()
    var_129 = True
    var_130 = module_1.Config()
    var_131 = 'from json import load\nfrom sys import path\n\nimport os\nimport zlib\n\n'
    var_132 = module_2.sorted_imports(var_128, var_130)
    var_133 = [var_0]
    var_134 = []
    var_135 = {var_22: var_134}
    var_136 = [var_28]
    var_137 = {var_26: var_136}
    var_138 = {var_19: var_135, var_20: var_137}
    var_139 = {var_18: var_138}
    var_140 = {}
    var_141 = {}
    var_142 = module_0.ParsedContent()
    var_143 = 'thirdparty'
    var_144 = 'Third Party Imports'
    var_145 = {var_143: var_144}
    var_146 = module_1.Config()
    var_147 = '# Third Party Imports\nimport os\n\nfrom sys import path\n\n'
    var_148 = module_2.sorted_imports(var_142, var_146)
    var_149 = [var_0]
    var_150 = []
    var_151 = {var_22: var_150}
    var_152 = {}
    var_153 = {var_19: var_151, var_20: var_152}
    var_154 = []
    var_155 = {var_62: var_154}
    var_156 = {}
    var_157 = {var_19: var_155, var_20: var_156}
    var_158 = {var_18: var_153, var_56: var_157}
    var_159 = {}
    var_160 = {}
    var_161 = module_0.ParsedContent()
    var_162 = 2
    var_163 = module_1.Config()
    var_164 = 'import os\n\n\nimport my_module\n\n'
    var_165 = module_2.sorted_imports(var_161, var_163)



# Parsed testcases at query #4
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = 0
    var_5 = '\n'
    var_6 = {}
    var_7 = {}
    var_8 = []
    var_9 = module_0.ParsedContent()
    var_10 = module_1.Config()
    var_11 = module_2.sorted_imports(var_9, var_10)
    assert var_11 == '\n'
    var_12 = "print('hello')"
    var_13 = [var_12]
    var_14 = {}
    var_15 = -1
    var_16 = 1
    var_17 = {}
    var_18 = {}
    var_19 = []
    var_20 = module_0.ParsedContent()
    var_21 = module_1.Config()
    var_22 = module_2.sorted_imports(var_20, var_21)
    assert var_22 == "print('hello')\n"
    var_23 = [var_0]
    var_24 = 'THIRDPARTY'
    var_25 = 'straight'
    var_26 = 'from'
    var_27 = 'zlib'
    var_28 = 'os'
    var_29 = [var_27]
    var_30 = [var_28]
    var_31 = {var_27: var_29, var_28: var_30}
    var_32 = 'sys'
    var_33 = 'json'
    var_34 = [var_32]
    var_35 = [var_33]
    var_36 = {var_32: var_34, var_33: var_35}
    var_37 = {var_25: var_31, var_26: var_36}
    var_38 = {var_24: var_37}
    var_39 = {}
    var_40 = {}
    var_41 = [var_24]
    var_42 = module_0.ParsedContent()
    var_43 = module_1.Config()
    var_44 = module_2.sorted_imports(var_42, var_43)
    var_45 = '\nimport os\nimport zlib\n\nfrom json import json\nfrom sys import sys\n'
    var_46 = [var_0]
    var_47 = [var_27]
    var_48 = [var_28]
    var_49 = {var_27: var_47, var_28: var_48}
    var_50 = [var_32]
    var_51 = [var_33]
    var_52 = {var_32: var_50, var_33: var_51}
    var_53 = {var_25: var_49, var_26: var_52}
    var_54 = {var_24: var_53}
    var_55 = {}
    var_56 = {}
    var_57 = [var_24]
    var_58 = module_0.ParsedContent()
    var_59 = True
    var_60 = module_1.Config()
    var_61 = module_2.sorted_imports(var_58, var_60)
    var_62 = '\nfrom json import json\nfrom sys import sys\n\nimport os\nimport zlib\n'
    var_63 = [var_0]
    var_64 = [var_27]
    var_65 = [var_28]
    var_66 = {var_27: var_64, var_28: var_65}
    var_67 = [var_32]
    var_68 = [var_33]
    var_69 = {var_32: var_67, var_33: var_68}
    var_70 = {var_25: var_66, var_26: var_69}
    var_71 = {var_24: var_70}
    var_72 = {}
    var_73 = {}
    var_74 = [var_24]
    var_75 = module_0.ParsedContent()
    var_76 = [var_27, var_33]
    var_77 = module_1.Config()
    var_78 = module_2.sorted_imports(var_75, var_77)
    var_79 = '\nimport os\n\nfrom sys import sys\n'
    var_80 = [var_0]
    var_81 = [var_27]
    var_82 = [var_28]
    var_83 = {var_27: var_81, var_28: var_82}
    var_84 = [var_32]
    var_85 = [var_33]
    var_86 = {var_32: var_84, var_33: var_85}
    var_87 = {var_25: var_83, var_26: var_86}
    var_88 = {var_24: var_87}
    var_89 = {}
    var_90 = {}
    var_91 = [var_24]
    var_92 = module_0.ParsedContent()
    var_93 = 'thirdparty'
    var_94 = 'Third Party Imports'
    var_95 = {var_93: var_94}
    var_96 = module_1.Config()
    var_97 = module_2.sorted_imports(var_92, var_96)
    var_98 = '\n# Third Party Imports\nimport os\nimport zlib\n\nfrom json import json\nfrom sys import sys\n'
    var_99 = 'def func():'
    var_100 = '    pass'
    var_101 = [var_0, var_99, var_100]
    var_102 = [var_27]
    var_103 = [var_28]
    var_104 = {var_27: var_102, var_28: var_103}
    var_105 = [var_32]
    var_106 = [var_33]
    var_107 = {var_32: var_105, var_33: var_106}
    var_108 = {var_25: var_104, var_26: var_107}
    var_109 = {var_24: var_108}
    var_110 = 3
    var_111 = 'import os'
    var_112 = 'import zlib'
    var_113 = [var_111, var_112]
    var_114 = {var_24: var_113}
    var_115 = {var_99}
    var_116 = [var_24]
    var_117 = module_0.ParsedContent()
    var_118 = module_1.Config()
    var_119 = module_2.sorted_imports(var_117, var_118)
    var_120 = '\n\ndef func():\nimport os\nimport zlib\n    pass\n'



# Parsed testcases at query #5
#--------------------------


import isort.parse as module_0
import isort.output as module_1
import isort.settings as module_2

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = '\n'
    var_5 = module_0.ParsedContent()
    var_6 = module_1.sorted_imports(var_5)
    assert var_6 == "print('hello')"
    var_7 = [var_0]
    var_8 = 'STDLIB'
    var_9 = 'straight'
    var_10 = 'from'
    var_11 = {}
    var_12 = {}
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = {var_8: var_13}
    var_15 = 0
    var_16 = module_0.ParsedContent()
    var_17 = module_1.sorted_imports(var_16)
    assert var_17 == "print('hello')"
    var_18 = [var_0]
    var_19 = 'os'
    var_20 = 'sys'
    var_21 = 'import os'
    var_22 = [var_21]
    var_23 = 'import sys'
    var_24 = [var_23]
    var_25 = {var_19: var_22, var_20: var_24}
    var_26 = 'os.path'
    var_27 = 'from os.path import join'
    var_28 = [var_27]
    var_29 = {var_26: var_28}
    var_30 = {var_9: var_25, var_10: var_29}
    var_31 = {var_8: var_30}
    var_32 = module_0.ParsedContent()
    var_33 = "\nimport os\nimport sys\n\nfrom os.path import join\n\nprint('hello')"
    var_34 = module_1.sorted_imports(var_32)
    var_35 = True
    var_36 = module_2.Config()
    var_37 = [var_0]
    var_38 = [var_23]
    var_39 = [var_21]
    var_40 = {var_20: var_38, var_19: var_39}
    var_41 = [var_27]
    var_42 = {var_26: var_41}
    var_43 = {var_9: var_40, var_10: var_42}
    var_44 = {var_8: var_43}
    var_45 = module_0.ParsedContent()
    var_46 = "\nimport os\nimport sys\n\nfrom os.path import join\n\nprint('hello')"
    var_47 = module_1.sorted_imports(var_45, var_36)
    var_48 = 'stdlib'
    var_49 = 'Standard Library'
    var_50 = {var_48: var_49}
    var_51 = module_2.Config()
    var_52 = [var_0]
    var_53 = [var_21]
    var_54 = {var_19: var_53}
    var_55 = {}
    var_56 = {var_9: var_54, var_10: var_55}
    var_57 = {var_8: var_56}
    var_58 = module_0.ParsedContent()
    var_59 = "\n# Standard Library\nimport os\n\nprint('hello')"
    var_60 = module_1.sorted_imports(var_58, var_51)
    var_61 = 'THIRDPARTY'
    var_62 = [var_61]
    var_63 = module_2.Config()
    var_64 = [var_0]
    var_65 = [var_21]
    var_66 = {var_19: var_65}
    var_67 = {}
    var_68 = {var_9: var_66, var_10: var_67}
    var_69 = 'numpy'
    var_70 = 'import numpy'
    var_71 = [var_70]
    var_72 = {var_69: var_71}
    var_73 = {}
    var_74 = {var_9: var_72, var_10: var_73}
    var_75 = {var_8: var_68, var_61: var_74}
    var_76 = module_0.ParsedContent()
    var_77 = "\nimport os\n\nimport numpy\n\nprint('hello')"
    var_78 = module_1.sorted_imports(var_76, var_63)
    var_79 = [var_21]
    var_80 = module_2.Config()
    var_81 = [var_0]
    var_82 = [var_21]
    var_83 = [var_23]
    var_84 = {var_19: var_82, var_20: var_83}
    var_85 = {}
    var_86 = {var_9: var_84, var_10: var_85}
    var_87 = {var_8: var_86}
    var_88 = module_0.ParsedContent()
    var_89 = "\nimport sys\n\nprint('hello')"
    var_90 = module_1.sorted_imports(var_88, var_80)
    var_91 = 2
    var_92 = module_2.Config()
    var_93 = [var_0]
    var_94 = [var_21]
    var_95 = {var_19: var_94}
    var_96 = {}
    var_97 = {var_9: var_95, var_10: var_96}
    var_98 = [var_70]
    var_99 = {var_69: var_98}
    var_100 = {}
    var_101 = {var_9: var_99, var_10: var_100}
    var_102 = {var_8: var_97, var_61: var_101}
    var_103 = module_0.ParsedContent()
    var_104 = "\nimport os\n\n\n\nimport numpy\n\nprint('hello')"
    var_105 = module_1.sorted_imports(var_103, var_92)
    var_106 = module_2.Config()
    var_107 = [var_0]
    var_108 = [var_21]
    var_109 = {var_19: var_108}
    var_110 = {}
    var_111 = {var_9: var_109, var_10: var_110}
    var_112 = {var_8: var_111}
    var_113 = module_0.ParsedContent()
    var_114 = "\nimport os\n\n\nprint('hello')"
    var_115 = module_1.sorted_imports(var_113, var_106)
    var_116 = [var_0]
    var_117 = [var_21]
    var_118 = {var_19: var_117}
    var_119 = {}
    var_120 = {var_9: var_118, var_10: var_119}
    var_121 = {var_8: var_120}
    var_122 = module_0.ParsedContent()
    var_123 = "\r\nimport os\r\n\r\nprint('hello')"
    var_124 = module_1.sorted_imports(var_122, var_106)



# Parsed testcases at query #6
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = -1
    var_3 = {}
    var_4 = {}
    var_5 = {}
    var_6 = 1
    var_7 = '\n'
    var_8 = module_0.ParsedContent()
    var_9 = [var_0]
    var_10 = 0
    var_11 = 'THIRDPARTY'
    var_12 = 'straight'
    var_13 = 'from'
    var_14 = 'os'
    var_15 = 'sys'
    var_16 = 'import os'
    var_17 = [var_16]
    var_18 = 'import sys'
    var_19 = [var_18]
    var_20 = {var_14: var_17, var_15: var_19}
    var_21 = 'datetime'
    var_22 = 'from datetime import datetime'
    var_23 = [var_22]
    var_24 = {var_21: var_23}
    var_25 = {var_12: var_20, var_13: var_24}
    var_26 = {var_11: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = module_0.ParsedContent()
    var_30 = "\nimport os\nimport sys\n\nfrom datetime import datetime\n\nprint('hello')"
    var_31 = [var_0]
    var_32 = 'FUTURE'
    var_33 = 'STDLIB'
    var_34 = '__future__'
    var_35 = 'from __future__ import annotations'
    var_36 = [var_35]
    var_37 = {var_34: var_36}
    var_38 = {}
    var_39 = {var_12: var_37, var_13: var_38}
    var_40 = [var_16]
    var_41 = [var_18]
    var_42 = {var_14: var_40, var_15: var_41}
    var_43 = [var_22]
    var_44 = {var_21: var_43}
    var_45 = {var_12: var_42, var_13: var_44}
    var_46 = 'numpy'
    var_47 = 'pandas'
    var_48 = 'import numpy'
    var_49 = [var_48]
    var_50 = 'import pandas'
    var_51 = [var_50]
    var_52 = {var_46: var_49, var_47: var_51}
    var_53 = 'django'
    var_54 = 'from django.conf import settings'
    var_55 = [var_54]
    var_56 = {var_53: var_55}
    var_57 = {var_12: var_52, var_13: var_56}
    var_58 = {var_32: var_39, var_33: var_45, var_11: var_57}
    var_59 = {}
    var_60 = {}
    var_61 = module_0.ParsedContent()
    var_62 = 'future'
    var_63 = 'stdlib'
    var_64 = 'thirdparty'
    var_65 = 'Future'
    var_66 = 'Standard Library'
    var_67 = 'Third Party'
    var_68 = {var_62: var_65, var_63: var_66, var_64: var_67}
    var_69 = True
    var_70 = module_1.Config()
    var_71 = module_2.sorted_imports(var_61, var_70)
    var_72 = "\n# Future\nfrom __future__ import annotations\n\n# Standard Library\nimport os\nimport sys\n\nfrom datetime import datetime\n\n# Third Party\nimport numpy\nimport pandas\n\nfrom django.conf import settings\n\nprint('hello')"
    var_73 = [var_0]
    var_74 = [var_16]
    var_75 = [var_18]
    var_76 = {var_14: var_74, var_15: var_75}
    var_77 = [var_22]
    var_78 = {var_21: var_77}
    var_79 = {var_12: var_76, var_13: var_78}
    var_80 = {var_11: var_79}
    var_81 = {}
    var_82 = {}
    var_83 = module_0.ParsedContent()
    var_84 = [var_16]
    var_85 = module_1.Config()
    var_86 = module_2.sorted_imports(var_83, var_85)
    var_87 = "\nimport sys\n\nfrom datetime import datetime\n\nprint('hello')"
    var_88 = [var_0]
    var_89 = [var_35]
    var_90 = {var_34: var_89}
    var_91 = {}
    var_92 = {var_12: var_90, var_13: var_91}
    var_93 = [var_16]
    var_94 = [var_18]
    var_95 = {var_14: var_93, var_15: var_94}
    var_96 = [var_22]
    var_97 = {var_21: var_96}
    var_98 = {var_12: var_95, var_13: var_97}
    var_99 = [var_48]
    var_100 = [var_50]
    var_101 = {var_46: var_99, var_47: var_100}
    var_102 = [var_54]
    var_103 = {var_53: var_102}
    var_104 = {var_12: var_101, var_13: var_103}
    var_105 = {var_32: var_92, var_33: var_98, var_11: var_104}
    var_106 = {}
    var_107 = {}
    var_108 = module_0.ParsedContent()
    var_109 = True
    var_110 = module_1.Config()
    var_111 = module_2.sorted_imports(var_108, var_110)
    var_112 = "\nfrom __future__ import annotations\nimport numpy\nimport os\nimport pandas\nimport sys\n\nfrom datetime import datetime\nfrom django.conf import settings\n\nprint('hello')"



# Parsed testcases at query #7
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = '\n'
    var_5 = 1
    var_6 = module_0.ParsedContent()
    var_7 = module_1.Config()
    var_8 = module_2.sorted_imports(var_6, var_7)
    assert var_8 == "print('hello')\n"
    var_9 = [var_0]
    var_10 = 'THIRDPARTY'
    var_11 = 'straight'
    var_12 = 'from'
    var_13 = 'os'
    var_14 = 'sys'
    var_15 = set()
    var_16 = set()
    var_17 = {var_13: var_15, var_14: var_16}
    var_18 = 'collections'
    var_19 = 'defaultdict'
    var_20 = set()
    var_21 = {var_19: var_20}
    var_22 = {var_18: var_21}
    var_23 = {var_11: var_17, var_12: var_22}
    var_24 = {var_10: var_23}
    var_25 = 0
    var_26 = 2
    var_27 = module_0.ParsedContent()
    var_28 = module_1.Config()
    var_29 = module_2.sorted_imports(var_27, var_28)
    assert var_29 == "import os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')\n"
    var_30 = [var_0]
    var_31 = 'FUTURE'
    var_32 = 'STDLIB'
    var_33 = '__future__'
    var_34 = set()
    var_35 = {var_33: var_34}
    var_36 = {}
    var_37 = {var_11: var_35, var_12: var_36}
    var_38 = set()
    var_39 = {var_13: var_38}
    var_40 = {}
    var_41 = {var_11: var_39, var_12: var_40}
    var_42 = 'django'
    var_43 = set()
    var_44 = {var_42: var_43}
    var_45 = {}
    var_46 = {var_11: var_44, var_12: var_45}
    var_47 = {var_31: var_37, var_32: var_41, var_10: var_46}
    var_48 = module_0.ParsedContent()
    var_49 = module_1.Config()
    var_50 = module_2.sorted_imports(var_48, var_49)
    assert var_50 == "from __future__ import absolute_import\n\nimport os\n\nimport django\n\nprint('hello')\n"
    var_51 = [var_0]
    var_52 = set()
    var_53 = set()
    var_54 = {var_13: var_52, var_14: var_53}
    var_55 = set()
    var_56 = {var_19: var_55}
    var_57 = {var_18: var_56}
    var_58 = {var_11: var_54, var_12: var_57}
    var_59 = {var_10: var_58}
    var_60 = module_0.ParsedContent()
    var_61 = True
    var_62 = module_1.Config()
    var_63 = module_2.sorted_imports(var_60, var_62)
    assert var_63 == "from collections import defaultdict\n\nimport os\nimport sys\n\nprint('hello')\n"
    var_64 = [var_0]
    var_65 = set()
    var_66 = set()
    var_67 = {var_13: var_65, var_14: var_66}
    var_68 = set()
    var_69 = {var_19: var_68}
    var_70 = {var_18: var_69}
    var_71 = {var_11: var_67, var_12: var_70}
    var_72 = {var_10: var_71}
    var_73 = module_0.ParsedContent()
    var_74 = [var_13]
    var_75 = module_1.Config()
    var_76 = module_2.sorted_imports(var_73, var_75)
    assert var_76 == "import sys\n\nfrom collections import defaultdict\n\nprint('hello')\n"



# Parsed testcases at query #8
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'x = 1'
    var_1 = [var_0]
    var_2 = 'THIRDPARTY'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = 'sys'
    var_7 = []
    var_8 = []
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'collections'
    var_11 = 'defaultdict'
    var_12 = [var_11]
    var_13 = {var_10: var_12}
    var_14 = {var_3: var_9, var_4: var_13}
    var_15 = {var_2: var_14}
    var_16 = 0
    var_17 = 1
    var_18 = '\n'
    var_19 = module_0.ParsedContent()
    var_20 = module_1.Config()
    var_21 = module_2.sorted_imports(var_19, var_20)
    assert var_21 == 'import os\nimport sys\n\nfrom collections import defaultdict\n\nx = 1\n'
    var_22 = [var_0]
    var_23 = 'FUTURE'
    var_24 = '__future__'
    var_25 = []
    var_26 = {var_24: var_25}
    var_27 = {}
    var_28 = {var_3: var_26, var_4: var_27}
    var_29 = []
    var_30 = []
    var_31 = {var_5: var_29, var_6: var_30}
    var_32 = {}
    var_33 = {var_3: var_31, var_4: var_32}
    var_34 = {var_23: var_28, var_2: var_33}
    var_35 = module_0.ParsedContent()
    var_36 = True
    var_37 = module_1.Config()
    var_38 = module_2.sorted_imports(var_35, var_37)
    assert var_38 == 'from __future__ import absolute_import\nimport os\nimport sys\n\nx = 1\n'
    var_39 = [var_0]
    var_40 = 'FIRSTPARTY'
    var_41 = []
    var_42 = {var_5: var_41}
    var_43 = {}
    var_44 = {var_3: var_42, var_4: var_43}
    var_45 = []
    var_46 = {var_6: var_45}
    var_47 = {}
    var_48 = {var_3: var_46, var_4: var_47}
    var_49 = {var_2: var_44, var_40: var_48}
    var_50 = module_0.ParsedContent()
    var_51 = [var_40]
    var_52 = module_1.Config()
    var_53 = module_2.sorted_imports(var_50, var_52)
    assert var_53 == 'import os\n\nimport sys\n\nx = 1\n'
    var_54 = [var_0]
    var_55 = []
    var_56 = []
    var_57 = {var_5: var_55, var_6: var_56}
    var_58 = [var_11]
    var_59 = {var_10: var_58}
    var_60 = {var_3: var_57, var_4: var_59}
    var_61 = {var_2: var_60}
    var_62 = module_0.ParsedContent()
    var_63 = [var_5]
    var_64 = module_1.Config()
    var_65 = module_2.sorted_imports(var_62, var_64)
    assert var_65 == 'import sys\n\nfrom collections import defaultdict\n\nx = 1\n'
    var_66 = [var_0]
    var_67 = {}
    var_68 = -1
    var_69 = module_0.ParsedContent()
    var_70 = module_1.Config()
    var_71 = module_2.sorted_imports(var_69, var_70)
    assert var_71 == 'x = 1\n'
    var_72 = [var_0]
    var_73 = []
    var_74 = {var_5: var_73}
    var_75 = {}
    var_76 = {var_3: var_74, var_4: var_75}
    var_77 = {var_2: var_76}
    var_78 = module_0.ParsedContent()
    var_79 = 'thirdparty'
    var_80 = 'Third Party Imports'
    var_81 = {var_79: var_80}
    var_82 = module_1.Config()
    var_83 = module_2.sorted_imports(var_78, var_82)
    assert var_83 == '# Third Party Imports\nimport os\n\nx = 1\n'
    var_84 = [var_0]
    var_85 = []
    var_86 = {var_5: var_85}
    var_87 = {}
    var_88 = {var_3: var_86, var_4: var_87}
    var_89 = []
    var_90 = {var_6: var_89}
    var_91 = {}
    var_92 = {var_3: var_90, var_4: var_91}
    var_93 = {var_2: var_88, var_40: var_92}
    var_94 = module_0.ParsedContent()
    var_95 = 2
    var_96 = module_1.Config()
    var_97 = module_2.sorted_imports(var_94, var_96)
    assert var_97 == 'import os\n\n\nimport sys\n\nx = 1\n'
    var_98 = [var_0]
    var_99 = []
    var_100 = {var_5: var_99}
    var_101 = {}
    var_102 = {var_3: var_100, var_4: var_101}
    var_103 = {var_2: var_102}
    var_104 = module_0.ParsedContent()
    var_105 = module_1.Config()
    var_106 = module_2.sorted_imports(var_104, var_105)
    assert var_106 == 'import os\n\n\nx = 1\n'
    var_107 = [var_0]
    var_108 = []
    var_109 = {var_5: var_108}
    var_110 = [var_11]
    var_111 = {var_10: var_110}
    var_112 = {var_3: var_109, var_4: var_111}
    var_113 = {var_2: var_112}
    var_114 = module_0.ParsedContent()
    var_115 = True
    var_116 = module_1.Config()
    var_117 = module_2.sorted_imports(var_114, var_116)
    assert var_117 == 'from collections import defaultdict\n\nimport os\n\nx = 1\n'
    var_118 = [var_0]
    var_119 = {}
    var_120 = [var_11]
    var_121 = '*'
    var_122 = [var_121]
    var_123 = {var_10: var_120, var_5: var_122}
    var_124 = {var_3: var_119, var_4: var_123}
    var_125 = {var_2: var_124}
    var_126 = module_0.ParsedContent()
    var_127 = True
    var_128 = module_1.Config()
    var_129 = module_2.sorted_imports(var_126, var_128)
    assert var_129 == 'from os import *\nfrom collections import defaultdict\n\nx = 1\n'



# Parsed testcases at query #9
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = 0
    var_5 = '\n'
    var_6 = {}
    var_7 = {}
    var_8 = []
    var_9 = module_0.ParsedContent()
    var_10 = module_1.Config()
    var_11 = module_2.sorted_imports(var_9, var_10)
    assert var_11 == '\n'
    var_12 = "print('hello')"
    var_13 = [var_12]
    var_14 = {}
    var_15 = -1
    var_16 = 1
    var_17 = {}
    var_18 = {}
    var_19 = []
    var_20 = module_0.ParsedContent()
    var_21 = module_1.Config()
    var_22 = module_2.sorted_imports(var_20, var_21)
    assert var_22 == "print('hello')\n"
    var_23 = [var_0]
    var_24 = 'THIRDPARTY'
    var_25 = 'straight'
    var_26 = 'from'
    var_27 = 'os'
    var_28 = 'sys'
    var_29 = [var_27]
    var_30 = [var_28]
    var_31 = {var_27: var_29, var_28: var_30}
    var_32 = 'collections'
    var_33 = 'OrderedDict'
    var_34 = 'defaultdict'
    var_35 = [var_33, var_34]
    var_36 = {var_32: var_35}
    var_37 = {var_25: var_31, var_26: var_36}
    var_38 = {var_24: var_37}
    var_39 = {}
    var_40 = {}
    var_41 = [var_24]
    var_42 = module_0.ParsedContent()
    var_43 = module_1.Config()
    var_44 = module_2.sorted_imports(var_42, var_43)
    var_45 = 'from collections import OrderedDict, defaultdict\nimport os\nimport sys\n'
    var_46 = [var_0]
    var_47 = 'FUTURE'
    var_48 = '__future__'
    var_49 = [var_48]
    var_50 = {var_48: var_49}
    var_51 = {}
    var_52 = {var_25: var_50, var_26: var_51}
    var_53 = [var_27]
    var_54 = {var_27: var_53}
    var_55 = [var_34]
    var_56 = {var_32: var_55}
    var_57 = {var_25: var_54, var_26: var_56}
    var_58 = {var_47: var_52, var_24: var_57}
    var_59 = {}
    var_60 = {}
    var_61 = [var_47, var_24]
    var_62 = module_0.ParsedContent()
    var_63 = 2
    var_64 = module_1.Config()
    var_65 = module_2.sorted_imports(var_62, var_64)
    var_66 = 'from __future__ import __future__\n\n\nfrom collections import defaultdict\nimport os\n'
    var_67 = [var_0]
    var_68 = [var_27]
    var_69 = {var_27: var_68}
    var_70 = [var_34]
    var_71 = {var_32: var_70}
    var_72 = {var_25: var_69, var_26: var_71}
    var_73 = {var_24: var_72}
    var_74 = {}
    var_75 = {}
    var_76 = [var_24]
    var_77 = module_0.ParsedContent()
    var_78 = 'thirdparty'
    var_79 = 'Third Party Imports'
    var_80 = {var_78: var_79}
    var_81 = module_1.Config()
    var_82 = module_2.sorted_imports(var_77, var_81)
    var_83 = '# Third Party Imports\nfrom collections import defaultdict\nimport os\n'
    var_84 = [var_0]
    var_85 = 'FIRSTPARTY'
    var_86 = [var_27]
    var_87 = {var_27: var_86}
    var_88 = [var_34]
    var_89 = {var_32: var_88}
    var_90 = {var_25: var_87, var_26: var_89}
    var_91 = 'my_module'
    var_92 = [var_91]
    var_93 = {var_91: var_92}
    var_94 = {}
    var_95 = {var_25: var_93, var_26: var_94}
    var_96 = {var_24: var_90, var_85: var_95}
    var_97 = {}
    var_98 = {}
    var_99 = [var_24, var_85]
    var_100 = module_0.ParsedContent()
    var_101 = 'LOCALFOLDER'
    var_102 = [var_101]
    var_103 = module_1.Config()
    var_104 = module_2.sorted_imports(var_100, var_103)
    var_105 = 'from collections import defaultdict\nimport os\n\nimport my_module\n'
    var_106 = [var_0]
    var_107 = [var_48]
    var_108 = {var_48: var_107}
    var_109 = {}
    var_110 = {var_25: var_108, var_26: var_109}
    var_111 = [var_27]
    var_112 = {var_27: var_111}
    var_113 = [var_34]
    var_114 = {var_32: var_113}
    var_115 = {var_25: var_112, var_26: var_114}
    var_116 = {var_47: var_110, var_24: var_115}
    var_117 = {}
    var_118 = {}
    var_119 = [var_47, var_24]
    var_120 = module_0.ParsedContent()
    var_121 = True
    var_122 = module_1.Config()
    var_123 = module_2.sorted_imports(var_120, var_122)
    var_124 = 'from __future__ import __future__\nfrom collections import defaultdict\nimport os\n'
    var_125 = [var_0]
    var_126 = {}
    var_127 = 'module1'
    var_128 = 'module2'
    var_129 = 'module3'
    var_130 = '*'
    var_131 = [var_130]
    var_132 = 'func1'
    var_133 = 'func2'
    var_134 = [var_132, var_133]
    var_135 = [var_130]
    var_136 = {var_127: var_131, var_128: var_134, var_129: var_135}
    var_137 = {var_25: var_126, var_26: var_136}
    var_138 = {var_24: var_137}
    var_139 = {}
    var_140 = {}
    var_141 = [var_24]
    var_142 = module_0.ParsedContent()
    var_143 = True
    var_144 = module_1.Config()
    var_145 = module_2.sorted_imports(var_142, var_144)
    var_146 = 'from module1 import *\nfrom module3 import *\nfrom module2 import func1, func2\n'
    var_147 = [var_0]
    var_148 = [var_27]
    var_149 = {var_27: var_148}
    var_150 = [var_34]
    var_151 = {var_32: var_150}
    var_152 = {var_25: var_149, var_26: var_151}
    var_153 = {var_24: var_152}
    var_154 = {}
    var_155 = {}
    var_156 = [var_24]
    var_157 = module_0.ParsedContent()
    var_158 = True
    var_159 = module_1.Config()
    var_160 = module_2.sorted_imports(var_157, var_159)
    var_161 = 'from collections import defaultdict\n\nimport os\n'
    var_162 = 'def main():'
    var_163 = '    pass'
    var_164 = [var_162, var_163]
    var_165 = [var_27]
    var_166 = {var_27: var_165}
    var_167 = {}
    var_168 = {var_25: var_166, var_26: var_167}
    var_169 = {var_24: var_168}
    var_170 = {}
    var_171 = {}
    var_172 = [var_24]
    var_173 = module_0.ParsedContent()
    var_174 = module_1.Config()
    var_175 = module_2.sorted_imports(var_173, var_174)
    var_176 = 'import os\n\n\ndef main():\n    pass\n'



# Parsed testcases at query #10
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'x = 1'
    var_1 = [var_0]
    var_2 = 'THIRDPARTY'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'zlib'
    var_6 = 'os'
    var_7 = []
    var_8 = []
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'sys'
    var_11 = 'json'
    var_12 = 'path'
    var_13 = [var_12]
    var_14 = 'load'
    var_15 = [var_14]
    var_16 = {var_10: var_13, var_11: var_15}
    var_17 = {var_3: var_9, var_4: var_16}
    var_18 = {var_2: var_17}
    var_19 = 0
    var_20 = '\n'
    var_21 = module_0.ParsedContent()
    var_22 = module_1.Config()
    var_23 = module_2.sorted_imports(var_21, var_22)
    var_24 = 'import os\nimport zlib\n\nfrom json import load\nfrom sys import path\n\nx = 1'
    var_25 = [var_0]
    var_26 = {}
    var_27 = -1
    var_28 = module_0.ParsedContent()
    var_29 = module_2.sorted_imports(var_28, var_22)
    assert var_29 == 'x = 1'
    var_30 = 'numpy'
    var_31 = [var_30]
    var_32 = module_1.Config()
    var_33 = [var_0]
    var_34 = []
    var_35 = []
    var_36 = {var_5: var_34, var_6: var_35}
    var_37 = [var_12]
    var_38 = [var_14]
    var_39 = {var_10: var_37, var_11: var_38}
    var_40 = {var_3: var_36, var_4: var_39}
    var_41 = []
    var_42 = {var_30: var_41}
    var_43 = {}
    var_44 = {var_3: var_42, var_4: var_43}
    var_45 = {var_2: var_40, var_30: var_44}
    var_46 = module_0.ParsedContent()
    var_47 = module_2.sorted_imports(var_46, var_32)
    var_48 = True
    var_49 = module_1.Config()
    var_50 = [var_0]
    var_51 = 'FUTURE'
    var_52 = '__future__'
    var_53 = []
    var_54 = {var_52: var_53}
    var_55 = {}
    var_56 = {var_3: var_54, var_4: var_55}
    var_57 = []
    var_58 = []
    var_59 = {var_5: var_57, var_6: var_58}
    var_60 = [var_12]
    var_61 = [var_14]
    var_62 = {var_10: var_60, var_11: var_61}
    var_63 = {var_3: var_59, var_4: var_62}
    var_64 = {var_51: var_56, var_2: var_63}
    var_65 = module_0.ParsedContent()
    var_66 = module_2.sorted_imports(var_65, var_49)
    var_67 = [var_5]
    var_68 = module_1.Config()
    var_69 = [var_0]
    var_70 = []
    var_71 = []
    var_72 = {var_5: var_70, var_6: var_71}
    var_73 = {}
    var_74 = {var_3: var_72, var_4: var_73}
    var_75 = {var_2: var_74}
    var_76 = module_0.ParsedContent()
    var_77 = module_2.sorted_imports(var_76, var_68)
    var_78 = module_1.Config()
    var_79 = [var_0]
    var_80 = {}
    var_81 = '*'
    var_82 = [var_81]
    var_83 = [var_12]
    var_84 = {var_6: var_82, var_10: var_83}
    var_85 = {var_3: var_80, var_4: var_84}
    var_86 = {var_2: var_85}
    var_87 = module_0.ParsedContent()
    var_88 = module_2.sorted_imports(var_87, var_78)
    var_89 = 'from os import *'
    var_90 = 'from sys import path'
    var_91 = module_1.Config()
    var_92 = [var_0]
    var_93 = []
    var_94 = {var_6: var_93}
    var_95 = [var_12]
    var_96 = {var_10: var_95}
    var_97 = {var_3: var_94, var_4: var_96}
    var_98 = {var_2: var_97}
    var_99 = module_0.ParsedContent()
    var_100 = module_2.sorted_imports(var_99, var_91)
    var_101 = 'import os'
    var_102 = 'thirdparty'
    var_103 = 'Third Party'
    var_104 = {var_102: var_103}
    var_105 = module_1.Config()
    var_106 = [var_0]
    var_107 = []
    var_108 = {var_6: var_107}
    var_109 = {}
    var_110 = {var_3: var_108, var_4: var_109}
    var_111 = {var_2: var_110}
    var_112 = module_0.ParsedContent()
    var_113 = module_2.sorted_imports(var_112, var_105)
    var_114 = 2
    var_115 = module_1.Config()
    var_116 = [var_0]
    var_117 = []
    var_118 = {var_52: var_117}
    var_119 = {}
    var_120 = {var_3: var_118, var_4: var_119}
    var_121 = []
    var_122 = {var_6: var_121}
    var_123 = {}
    var_124 = {var_3: var_122, var_4: var_123}
    var_125 = {var_51: var_120, var_2: var_124}
    var_126 = module_0.ParsedContent()
    var_127 = module_2.sorted_imports(var_126, var_115)
    var_128 = '\n\n'
    var_129 = module_1.Config()
    var_130 = [var_0]
    var_131 = []
    var_132 = {var_6: var_131}
    var_133 = {}
    var_134 = {var_3: var_132, var_4: var_133}
    var_135 = {var_2: var_134}
    var_136 = module_0.ParsedContent()
    var_137 = module_2.sorted_imports(var_136, var_129)
    var_138 = '\nimport os\n\n\n\nx = 1'
    var_139 = module_1.Config()
    var_140 = '# comment'
    var_141 = [var_140, var_0]
    var_142 = []
    var_143 = {var_6: var_142}
    var_144 = {}
    var_145 = {var_3: var_143, var_4: var_144}
    var_146 = {var_2: var_145}
    var_147 = module_0.ParsedContent()
    var_148 = module_2.sorted_imports(var_147, var_139)



# Parsed testcases at query #11
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'x = 1'
    var_1 = [var_0]
    var_2 = 'THIRDPARTY'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = 'sys'
    var_7 = []
    var_8 = []
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'collections'
    var_11 = 'itertools'
    var_12 = 'defaultdict'
    var_13 = [var_12]
    var_14 = 'chain'
    var_15 = [var_14]
    var_16 = {var_10: var_13, var_11: var_15}
    var_17 = {var_3: var_9, var_4: var_16}
    var_18 = {var_2: var_17}
    var_19 = 0
    var_20 = 1
    var_21 = '\n'
    var_22 = module_0.ParsedContent()
    var_23 = module_1.Config()
    var_24 = module_2.sorted_imports(var_22, var_23)
    assert var_24 == 'from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n\nx = 1\n'
    var_25 = [var_0]
    var_26 = 'FUTURE'
    var_27 = '__future__'
    var_28 = []
    var_29 = {var_27: var_28}
    var_30 = {}
    var_31 = {var_3: var_29, var_4: var_30}
    var_32 = []
    var_33 = []
    var_34 = {var_5: var_32, var_6: var_33}
    var_35 = [var_12]
    var_36 = {var_10: var_35}
    var_37 = {var_3: var_34, var_4: var_36}
    var_38 = {var_26: var_31, var_2: var_37}
    var_39 = module_0.ParsedContent()
    var_40 = True
    var_41 = module_1.Config()
    var_42 = module_2.sorted_imports(var_39, var_41)
    assert var_42 == 'from __future__ import absolute_import\nfrom collections import defaultdict\nimport os\nimport sys\n\nx = 1\n'
    var_43 = True
    var_44 = module_1.Config()
    var_45 = module_2.sorted_imports(var_22, var_44)
    assert var_45 == 'from collections import defaultdict\nfrom itertools import chain\n\nimport os\nimport sys\n\nx = 1\n'
    var_46 = [var_0]
    var_47 = {}
    var_48 = 'module1'
    var_49 = 'module2'
    var_50 = '*'
    var_51 = [var_50]
    var_52 = 'func'
    var_53 = [var_52]
    var_54 = {var_48: var_51, var_49: var_53}
    var_55 = {var_3: var_47, var_4: var_54}
    var_56 = {var_2: var_55}
    var_57 = module_0.ParsedContent()
    var_58 = True
    var_59 = module_1.Config()
    var_60 = module_2.sorted_imports(var_57, var_59)
    assert var_60 == 'from module1 import *\nfrom module2 import func\n\nx = 1\n'
    var_61 = 2
    var_62 = module_1.Config()
    var_63 = module_2.sorted_imports(var_22, var_62)
    assert var_63 == 'from collections import defaultdict\nfrom itertools import chain\n\n\nimport os\nimport sys\n\nx = 1\n'
    var_64 = [var_0]
    var_65 = []
    var_66 = {var_27: var_65}
    var_67 = {}
    var_68 = {var_3: var_66, var_4: var_67}
    var_69 = []
    var_70 = {var_5: var_69}
    var_71 = {}
    var_72 = {var_3: var_70, var_4: var_71}
    var_73 = {var_26: var_68, var_2: var_72}
    var_74 = module_0.ParsedContent()
    var_75 = module_1.Config()
    var_76 = module_2.sorted_imports(var_74, var_75)
    assert var_76 == 'from __future__ import absolute_import\n\n\nimport os\n\nx = 1\n'
    var_77 = 'thirdparty'
    var_78 = 'Third Party'
    var_79 = {var_77: var_78}
    var_80 = module_1.Config()
    var_81 = module_2.sorted_imports(var_22, var_80)
    assert var_81 == '# Third Party\nfrom collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n\nx = 1\n'
    var_82 = {var_77: var_78}
    var_83 = True
    var_84 = module_1.Config()
    var_85 = '# Third Party'
    var_86 = [var_85, var_0]
    var_87 = []
    var_88 = {var_5: var_87}
    var_89 = [var_12]
    var_90 = {var_10: var_89}
    var_91 = {var_3: var_88, var_4: var_90}
    var_92 = {var_2: var_91}
    var_93 = module_0.ParsedContent()
    var_94 = module_2.sorted_imports(var_93, var_84)
    assert var_94 == '# Third Party\nfrom collections import defaultdict\nimport os\n\nx = 1\n'
    var_95 = module_1.Config()
    var_96 = module_2.sorted_imports(var_22, var_95)
    assert var_96 == 'from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n\n\nx = 1\n'
    var_97 = module_1.Config()
    var_98 = module_2.sorted_imports(var_22, var_97)
    assert var_98 == '\n\nfrom collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n\nx = 1\n'
    var_99 = [var_0]
    var_100 = []
    var_101 = []
    var_102 = {var_5: var_100, var_6: var_101}
    var_103 = [var_12]
    var_104 = [var_14]
    var_105 = {var_10: var_103, var_11: var_104}
    var_106 = {var_3: var_102, var_4: var_105}
    var_107 = {var_2: var_106}
    var_108 = module_0.ParsedContent()
    var_109 = True
    var_110 = module_1.Config()
    var_111 = module_2.sorted_imports(var_108, var_110)
    assert var_111 == 'from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n\nx = 1\n'
    var_112 = [var_5]
    var_113 = module_1.Config()
    var_114 = module_2.sorted_imports(var_22, var_113)
    assert var_114 == 'from collections import defaultdict\nfrom itertools import chain\nimport sys\n\nx = 1\n'
    var_115 = True
    var_116 = module_1.Config()
    var_117 = module_2.sorted_imports(var_22, var_116)
    assert var_117 == 'from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n\nx = 1\n'
    var_118 = True
    var_119 = module_1.Config()
    var_120 = module_2.sorted_imports(var_22, var_119)
    assert var_120 == 'from itertools import chain\nfrom collections import defaultdict\nimport sys\nimport os\n\nx = 1\n'
    var_121 = [var_0]
    var_122 = {}
    var_123 = -1
    var_124 = module_0.ParsedContent()
    var_125 = module_2.sorted_imports(var_124, var_23)
    assert var_125 == 'x = 1\n'
    var_126 = 'y = 2'
    var_127 = [var_0, var_126]
    var_128 = []
    var_129 = {var_5: var_128}
    var_130 = {}
    var_131 = {var_3: var_129, var_4: var_130}
    var_132 = {var_2: var_131}
    var_133 = 'import os'
    var_134 = [var_133]
    var_135 = {var_2: var_134}
    var_136 = {var_126: var_2}
    var_137 = module_0.ParsedContent()
    var_138 = module_2.sorted_imports(var_137, var_23)
    assert var_138 == 'x = 1\ny = 2\nimport os\n'



# Parsed testcases at query #12
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = 0
    var_5 = '\n'
    var_6 = {}
    var_7 = {}
    var_8 = module_0.ParsedContent()
    var_9 = module_1.Config()
    var_10 = module_2.sorted_imports(var_8, var_9)
    assert var_10 == '\n'
    var_11 = "print('hello')"
    var_12 = [var_11]
    var_13 = {}
    var_14 = -1
    var_15 = 1
    var_16 = {}
    var_17 = {}
    var_18 = module_0.ParsedContent()
    var_19 = module_1.Config()
    var_20 = module_2.sorted_imports(var_18, var_19)
    assert var_20 == "print('hello')\n"
    var_21 = [var_0]
    var_22 = 'THIRDPARTY'
    var_23 = 'straight'
    var_24 = 'from'
    var_25 = 'os'
    var_26 = 'sys'
    var_27 = []
    var_28 = []
    var_29 = {var_25: var_27, var_26: var_28}
    var_30 = 'collections'
    var_31 = 'itertools'
    var_32 = 'defaultdict'
    var_33 = [var_32]
    var_34 = 'chain'
    var_35 = [var_34]
    var_36 = {var_30: var_33, var_31: var_35}
    var_37 = {var_23: var_29, var_24: var_36}
    var_38 = {var_22: var_37}
    var_39 = {}
    var_40 = {}
    var_41 = module_0.ParsedContent()
    var_42 = module_1.Config()
    var_43 = module_2.sorted_imports(var_41, var_42)
    var_44 = 'from collections import defaultdict\nfrom itertools import chain\n\nimport os\nimport sys\n\n'
    var_45 = [var_0]
    var_46 = 'FUTURE'
    var_47 = 'STDLIB'
    var_48 = '__future__'
    var_49 = 'annotations'
    var_50 = [var_49]
    var_51 = {var_48: var_50}
    var_52 = {}
    var_53 = {var_23: var_51, var_24: var_52}
    var_54 = []
    var_55 = []
    var_56 = {var_25: var_54, var_26: var_55}
    var_57 = {}
    var_58 = {var_23: var_56, var_24: var_57}
    var_59 = {var_46: var_53, var_47: var_58}
    var_60 = {}
    var_61 = {}
    var_62 = module_0.ParsedContent()
    var_63 = True
    var_64 = 2
    var_65 = module_1.Config()
    var_66 = module_2.sorted_imports(var_62, var_65)
    var_67 = 'from __future__ import annotations\n\n\nimport os\nimport sys\n\n'
    var_68 = [var_0]
    var_69 = 'unused'
    var_70 = []
    var_71 = []
    var_72 = []
    var_73 = {var_25: var_70, var_26: var_71, var_69: var_72}
    var_74 = [var_32]
    var_75 = {var_30: var_74}
    var_76 = {var_23: var_73, var_24: var_75}
    var_77 = {var_22: var_76}
    var_78 = {}
    var_79 = {}
    var_80 = module_0.ParsedContent()
    var_81 = [var_69]
    var_82 = module_1.Config()
    var_83 = module_2.sorted_imports(var_80, var_82)
    var_84 = 'from collections import defaultdict\n\nimport os\nimport sys\n\n'
    var_85 = [var_0]
    var_86 = 'django'
    var_87 = []
    var_88 = {var_86: var_87}
    var_89 = 'flask'
    var_90 = 'Flask'
    var_91 = [var_90]
    var_92 = {var_89: var_91}
    var_93 = {var_23: var_88, var_24: var_92}
    var_94 = {var_22: var_93}
    var_95 = {}
    var_96 = {}
    var_97 = module_0.ParsedContent()
    var_98 = 'thirdparty'
    var_99 = 'Third Party Imports'
    var_100 = {var_98: var_99}
    var_101 = module_1.Config()
    var_102 = module_2.sorted_imports(var_97, var_101)
    var_103 = '# Third Party Imports\nfrom flask import Flask\n\nimport django\n\n'
    var_104 = 'def main():'
    var_105 = '    pass'
    var_106 = [var_104, var_105]
    var_107 = []
    var_108 = {var_25: var_107}
    var_109 = {}
    var_110 = {var_23: var_108, var_24: var_109}
    var_111 = {var_47: var_110}
    var_112 = {}
    var_113 = {}
    var_114 = module_0.ParsedContent()
    var_115 = module_1.Config()
    var_116 = module_2.sorted_imports(var_114, var_115)
    var_117 = 'import os\n\n\ndef main():\n    pass\n'
    var_118 = '# Placeholder'
    var_119 = [var_118, var_104, var_105]
    var_120 = []
    var_121 = {var_25: var_120}
    var_122 = {}
    var_123 = {var_23: var_121, var_24: var_122}
    var_124 = {var_47: var_123}
    var_125 = 3
    var_126 = 'import sys'
    var_127 = [var_126]
    var_128 = {var_47: var_127}
    var_129 = {var_118: var_47}
    var_130 = module_0.ParsedContent()
    var_131 = module_1.Config()
    var_132 = module_2.sorted_imports(var_130, var_131)
    var_133 = 'import os\n\n# Placeholder\nimport sys\n\ndef main():\n    pass\n'



# Parsed testcases at query #13
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = 0
    var_5 = '\n'
    var_6 = {}
    var_7 = {}
    var_8 = module_0.ParsedContent()
    var_9 = "print('hello')"
    var_10 = [var_9]
    var_11 = {}
    var_12 = -1
    var_13 = 1
    var_14 = {}
    var_15 = {}
    var_16 = module_0.ParsedContent()
    var_17 = [var_0]
    var_18 = 'THIRDPARTY'
    var_19 = 'straight'
    var_20 = 'from'
    var_21 = 'zlib'
    var_22 = 'os'
    var_23 = []
    var_24 = []
    var_25 = {var_21: var_23, var_22: var_24}
    var_26 = 'sys'
    var_27 = 'json'
    var_28 = 'argv'
    var_29 = [var_28]
    var_30 = 'load'
    var_31 = [var_30]
    var_32 = {var_26: var_29, var_27: var_31}
    var_33 = {var_19: var_25, var_20: var_32}
    var_34 = {var_18: var_33}
    var_35 = {}
    var_36 = {}
    var_37 = module_0.ParsedContent()
    var_38 = 'import os\nimport zlib\n\nfrom json import load\nfrom sys import argv\n\n'
    var_39 = [var_0]
    var_40 = 'FUTURE'
    var_41 = 'STDLIB'
    var_42 = '__future__'
    var_43 = 'annotations'
    var_44 = [var_43]
    var_45 = {var_42: var_44}
    var_46 = {}
    var_47 = {var_19: var_45, var_20: var_46}
    var_48 = []
    var_49 = {var_22: var_48}
    var_50 = {}
    var_51 = {var_19: var_49, var_20: var_50}
    var_52 = 'numpy'
    var_53 = []
    var_54 = {var_52: var_53}
    var_55 = {}
    var_56 = {var_19: var_54, var_20: var_55}
    var_57 = {var_40: var_47, var_41: var_51, var_18: var_56}
    var_58 = {}
    var_59 = {}
    var_60 = module_0.ParsedContent()
    var_61 = 'future'
    var_62 = 'stdlib'
    var_63 = 'thirdparty'
    var_64 = 'Future Imports'
    var_65 = 'Standard Library'
    var_66 = 'Third Party'
    var_67 = {var_61: var_64, var_62: var_65, var_63: var_66}
    var_68 = module_1.Config()
    var_69 = module_2.sorted_imports(var_60, var_68)
    var_70 = '# Future Imports\nfrom __future__ import annotations\n\n# Standard Library\nimport os\n\n# Third Party\nimport numpy\n\n'
    var_71 = [var_0]
    var_72 = []
    var_73 = []
    var_74 = {var_21: var_72, var_22: var_73}
    var_75 = [var_28]
    var_76 = [var_30]
    var_77 = {var_26: var_75, var_27: var_76}
    var_78 = {var_19: var_74, var_20: var_77}
    var_79 = {var_18: var_78}
    var_80 = {}
    var_81 = {}
    var_82 = module_0.ParsedContent()
    var_83 = [var_22, var_27]
    var_84 = module_1.Config()
    var_85 = module_2.sorted_imports(var_82, var_84)
    var_86 = 'import zlib\n\nfrom sys import argv\n\n'
    var_87 = [var_0]
    var_88 = [var_43]
    var_89 = {var_42: var_88}
    var_90 = {}
    var_91 = {var_19: var_89, var_20: var_90}
    var_92 = []
    var_93 = {var_22: var_92}
    var_94 = {}
    var_95 = {var_19: var_93, var_20: var_94}
    var_96 = []
    var_97 = {var_52: var_96}
    var_98 = {}
    var_99 = {var_19: var_97, var_20: var_98}
    var_100 = {var_40: var_91, var_41: var_95, var_18: var_99}
    var_101 = {}
    var_102 = {}
    var_103 = module_0.ParsedContent()
    var_104 = True
    var_105 = module_1.Config()
    var_106 = module_2.sorted_imports(var_103, var_105)
    var_107 = 'from __future__ import annotations\nimport os\nimport numpy\n\n'
    var_108 = [var_0]
    var_109 = []
    var_110 = []
    var_111 = {var_21: var_109, var_22: var_110}
    var_112 = [var_28]
    var_113 = [var_30]
    var_114 = {var_26: var_112, var_27: var_113}
    var_115 = {var_19: var_111, var_20: var_114}
    var_116 = {var_18: var_115}
    var_117 = {}
    var_118 = {}
    var_119 = module_0.ParsedContent()
    var_120 = True
    var_121 = module_1.Config()
    var_122 = module_2.sorted_imports(var_119, var_121)
    var_123 = 'import os\nimport zlib\n\nfrom json import load\nfrom sys import argv\n\n'
    var_124 = [var_0]
    var_125 = [var_43]
    var_126 = {var_42: var_125}
    var_127 = {}
    var_128 = {var_19: var_126, var_20: var_127}
    var_129 = []
    var_130 = {var_22: var_129}
    var_131 = {}
    var_132 = {var_19: var_130, var_20: var_131}
    var_133 = {var_40: var_128, var_41: var_132}
    var_134 = {}
    var_135 = {}
    var_136 = module_0.ParsedContent()
    var_137 = 2
    var_138 = module_1.Config()
    var_139 = module_2.sorted_imports(var_136, var_138)
    var_140 = 'from __future__ import annotations\n\n\nimport os\n\n'
    var_141 = 'def main():\n    pass'
    var_142 = [var_141]
    var_143 = []
    var_144 = {var_52: var_143}
    var_145 = {}
    var_146 = {var_19: var_144, var_20: var_145}
    var_147 = {var_18: var_146}
    var_148 = {}
    var_149 = {}
    var_150 = module_0.ParsedContent()
    var_151 = module_1.Config()
    var_152 = module_2.sorted_imports(var_150, var_151)
    var_153 = 'import numpy\n\n\ndef main():\n    pass\n'
    var_154 = [var_0]
    var_155 = []
    var_156 = {var_52: var_155}
    var_157 = {}
    var_158 = {var_19: var_156, var_20: var_157}
    var_159 = {var_18: var_158}
    var_160 = {}
    var_161 = {}
    var_162 = module_0.ParsedContent()
    var_163 = module_2.sorted_imports(var_162, var_151)
    var_164 = 'IMPORT NUMPY\n'



# Parsed testcases at query #14
#--------------------------


import isort.parse as module_0
import isort.output as module_1
import isort.settings as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = 0
    var_5 = '\n'
    var_6 = {}
    var_7 = {}
    var_8 = module_0.ParsedContent()
    var_9 = module_1.sorted_imports(var_8)
    assert var_9 == '\n'
    var_10 = "print('hello')"
    var_11 = [var_10]
    var_12 = {}
    var_13 = -1
    var_14 = 1
    var_15 = {}
    var_16 = {}
    var_17 = module_0.ParsedContent()
    var_18 = module_1.sorted_imports(var_17)
    assert var_18 == "print('hello')\n"
    var_19 = [var_0]
    var_20 = 'THIRDPARTY'
    var_21 = 'straight'
    var_22 = 'from'
    var_23 = 'os'
    var_24 = 'sys'
    var_25 = []
    var_26 = []
    var_27 = {var_23: var_25, var_24: var_26}
    var_28 = 'collections'
    var_29 = 'itertools'
    var_30 = 'defaultdict'
    var_31 = [var_30]
    var_32 = 'chain'
    var_33 = [var_32]
    var_34 = {var_28: var_31, var_29: var_33}
    var_35 = {var_21: var_27, var_22: var_34}
    var_36 = {var_20: var_35}
    var_37 = {}
    var_38 = {}
    var_39 = module_0.ParsedContent()
    var_40 = module_2.Config()
    var_41 = module_1.sorted_imports(var_39, var_40)
    var_42 = [var_0]
    var_43 = []
    var_44 = []
    var_45 = {var_23: var_43, var_24: var_44}
    var_46 = [var_30]
    var_47 = [var_32]
    var_48 = {var_28: var_46, var_29: var_47}
    var_49 = {var_21: var_45, var_22: var_48}
    var_50 = {var_20: var_49}
    var_51 = {}
    var_52 = {}
    var_53 = module_0.ParsedContent()
    var_54 = 2
    var_55 = True
    var_56 = module_2.Config()
    var_57 = module_1.sorted_imports(var_53, var_56)
    var_58 = [var_0]
    var_59 = []
    var_60 = []
    var_61 = {var_23: var_59, var_24: var_60}
    var_62 = [var_30]
    var_63 = [var_32]
    var_64 = {var_28: var_62, var_29: var_63}
    var_65 = {var_21: var_61, var_22: var_64}
    var_66 = {var_20: var_65}
    var_67 = {}
    var_68 = {}
    var_69 = module_0.ParsedContent()
    var_70 = [var_23, var_24]
    var_71 = module_2.Config()
    var_72 = module_1.sorted_imports(var_69, var_71)
    var_73 = [var_0]
    var_74 = {}
    var_75 = '*'
    var_76 = [var_75]
    var_77 = [var_32]
    var_78 = {var_28: var_76, var_29: var_77}
    var_79 = {var_21: var_74, var_22: var_78}
    var_80 = {var_20: var_79}
    var_81 = {}
    var_82 = {}
    var_83 = module_0.ParsedContent()
    var_84 = True
    var_85 = module_2.Config()
    var_86 = module_1.sorted_imports(var_83, var_85)
    var_87 = [var_0]
    var_88 = []
    var_89 = []
    var_90 = {var_23: var_88, var_24: var_89}
    var_91 = [var_30]
    var_92 = [var_32]
    var_93 = {var_28: var_91, var_29: var_92}
    var_94 = {var_21: var_90, var_22: var_93}
    var_95 = {var_20: var_94}
    var_96 = {}
    var_97 = {}
    var_98 = module_0.ParsedContent()
    var_99 = 'thirdparty'
    var_100 = 'Third Party Imports'
    var_101 = {var_99: var_100}
    var_102 = module_2.Config()
    var_103 = module_1.sorted_imports(var_98, var_102)
    var_104 = [var_0]
    var_105 = []
    var_106 = []
    var_107 = {var_23: var_105, var_24: var_106}
    var_108 = [var_30]
    var_109 = [var_32]
    var_110 = {var_28: var_108, var_29: var_109}
    var_111 = {var_21: var_107, var_22: var_110}
    var_112 = {var_20: var_111}
    var_113 = {}
    var_114 = {}
    var_115 = module_0.ParsedContent()
    var_116 = 'End of Third Party Imports'
    var_117 = {var_99: var_116}
    var_118 = module_2.Config()
    var_119 = module_1.sorted_imports(var_115, var_118)
    var_120 = [var_0]
    var_121 = []
    var_122 = []
    var_123 = {var_23: var_121, var_24: var_122}
    var_124 = [var_30]
    var_125 = [var_32]
    var_126 = {var_28: var_124, var_29: var_125}
    var_127 = {var_21: var_123, var_22: var_126}
    var_128 = {var_20: var_127}
    var_129 = {}
    var_130 = {}
    var_131 = module_0.ParsedContent()
    var_132 = lambda code, extension, config: code.upper()
    var_133 = module_2.Config()
    var_134 = 'py'
    var_135 = module_1.sorted_imports(var_131, var_133, var_134)
    var_136 = 'def main():'
    var_137 = [var_0, var_136]
    var_138 = []
    var_139 = []
    var_140 = {var_23: var_138, var_24: var_139}
    var_141 = [var_30]
    var_142 = [var_32]
    var_143 = {var_28: var_141, var_29: var_142}
    var_144 = {var_21: var_140, var_22: var_143}
    var_145 = {var_20: var_144}
    var_146 = 'import os'
    var_147 = 'import sys'
    var_148 = [var_146, var_147]
    var_149 = {var_20: var_148}
    var_150 = 'def main()'
    var_151 = {var_150: var_20}
    var_152 = module_0.ParsedContent()
    var_153 = module_2.Config()
    var_154 = module_1.sorted_imports(var_152, var_153)



# Parsed testcases at query #15
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = 'x = 1'
    var_2 = [var_0, var_1]
    var_3 = 'THIRDPARTY'
    var_4 = 'straight'
    var_5 = 'from'
    var_6 = 'os'
    var_7 = 'sys'
    var_8 = set()
    var_9 = set()
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'collections'
    var_12 = 'defaultdict'
    var_13 = {var_12}
    var_14 = {var_11: var_13}
    var_15 = {var_4: var_10, var_5: var_14}
    var_16 = {var_3: var_15}
    var_17 = 0
    var_18 = '\n'
    var_19 = 2
    var_20 = {}
    var_21 = {}
    var_22 = module_0.ParsedContent()
    var_23 = module_1.Config()
    var_24 = module_2.sorted_imports(var_22, var_23)
    assert var_24 == 'import os\nimport sys\n\nfrom collections import defaultdict\n\nx = 1\n'
    var_25 = [var_0, var_1]
    var_26 = 'FUTURE'
    var_27 = '__future__'
    var_28 = 'print_function'
    var_29 = {var_28}
    var_30 = {var_27: var_29}
    var_31 = {}
    var_32 = {var_4: var_30, var_5: var_31}
    var_33 = set()
    var_34 = set()
    var_35 = {var_6: var_33, var_7: var_34}
    var_36 = {var_12}
    var_37 = {var_11: var_36}
    var_38 = {var_4: var_35, var_5: var_37}
    var_39 = {var_26: var_32, var_3: var_38}
    var_40 = {}
    var_41 = {}
    var_42 = module_0.ParsedContent()
    var_43 = True
    var_44 = module_1.Config()
    var_45 = module_2.sorted_imports(var_42, var_44)
    assert var_45 == 'from __future__ import print_function\n\nimport os\nimport sys\n\nfrom collections import defaultdict\n\nx = 1\n'
    var_46 = [var_0, var_1]
    var_47 = set()
    var_48 = {var_6: var_47}
    var_49 = {var_12}
    var_50 = {var_11: var_49}
    var_51 = {var_4: var_48, var_5: var_50}
    var_52 = {var_3: var_51}
    var_53 = {}
    var_54 = {}
    var_55 = module_0.ParsedContent()
    var_56 = module_1.Config()
    var_57 = module_2.sorted_imports(var_55, var_56)
    assert var_57 == 'from collections import defaultdict\n\nimport os\n\nx = 1\n'
    var_58 = [var_0, var_1]
    var_59 = {}
    var_60 = '*'
    var_61 = {var_60}
    var_62 = {var_12}
    var_63 = {var_6: var_61, var_11: var_62}
    var_64 = {var_4: var_59, var_5: var_63}
    var_65 = {var_3: var_64}
    var_66 = {}
    var_67 = {}
    var_68 = module_0.ParsedContent()
    var_69 = module_1.Config()
    var_70 = module_2.sorted_imports(var_68, var_69)
    assert var_70 == 'from os import *\nfrom collections import defaultdict\n\nx = 1\n'
    var_71 = [var_0, var_1]
    var_72 = set()
    var_73 = {var_6: var_72}
    var_74 = {var_12}
    var_75 = {var_11: var_74}
    var_76 = {var_4: var_73, var_5: var_75}
    var_77 = {var_3: var_76}
    var_78 = {}
    var_79 = {}
    var_80 = module_0.ParsedContent()
    var_81 = 'thirdparty'
    var_82 = 'Third Party Imports'
    var_83 = {var_81: var_82}
    var_84 = module_1.Config()
    var_85 = module_2.sorted_imports(var_80, var_84)
    assert var_85 == '# Third Party Imports\nimport os\n\nfrom collections import defaultdict\n\nx = 1\n'
    var_86 = [var_0, var_1]
    var_87 = {var_28}
    var_88 = {var_27: var_87}
    var_89 = {}
    var_90 = {var_4: var_88, var_5: var_89}
    var_91 = set()
    var_92 = {var_6: var_91}
    var_93 = {var_12}
    var_94 = {var_11: var_93}
    var_95 = {var_4: var_92, var_5: var_94}
    var_96 = {var_26: var_90, var_3: var_95}
    var_97 = {}
    var_98 = {}
    var_99 = module_0.ParsedContent()
    var_100 = module_1.Config()
    var_101 = module_2.sorted_imports(var_99, var_100)
    assert var_101 == 'from __future__ import print_function\n\n\n\nimport os\n\nfrom collections import defaultdict\n\nx = 1\n'
    var_102 = [var_0, var_1]
    var_103 = set()
    var_104 = {var_6: var_103}
    var_105 = {var_12}
    var_106 = {var_11: var_105}
    var_107 = {var_4: var_104, var_5: var_106}
    var_108 = {var_3: var_107}
    var_109 = {}
    var_110 = {}
    var_111 = module_0.ParsedContent()
    var_112 = module_1.Config()
    var_113 = module_2.sorted_imports(var_111, var_112)
    assert var_113 == 'import os\n\nfrom collections import defaultdict\n\n\n\nx = 1\n'
    var_114 = [var_0, var_1]
    var_115 = set()
    var_116 = set()
    var_117 = {var_6: var_115, var_7: var_116}
    var_118 = {var_12}
    var_119 = {var_11: var_118}
    var_120 = {var_4: var_117, var_5: var_119}
    var_121 = {var_3: var_120}
    var_122 = {}
    var_123 = {}
    var_124 = module_0.ParsedContent()
    var_125 = [var_7]
    var_126 = module_1.Config()
    var_127 = module_2.sorted_imports(var_124, var_126)
    assert var_127 == 'import os\n\nfrom collections import defaultdict\n\nx = 1\n'
    var_128 = [var_1]
    var_129 = {}
    var_130 = -1
    var_131 = {}
    var_132 = {}
    var_133 = module_0.ParsedContent()
    var_134 = module_1.Config()
    var_135 = module_2.sorted_imports(var_133, var_134)
    assert var_135 == 'x = 1\n'



# Parsed testcases at query #16
#--------------------------


import isort.parse as module_0
import isort.output as module_1
import isort.settings as module_2

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = 1
    var_5 = '\n'
    var_6 = {}
    var_7 = {}
    var_8 = module_0.ParsedContent()
    var_9 = module_1.sorted_imports(var_8)
    assert var_9 == "print('hello')\n"
    var_10 = [var_0]
    var_11 = 'THIRDPARTY'
    var_12 = 'straight'
    var_13 = 'from'
    var_14 = 'os'
    var_15 = 'sys'
    var_16 = []
    var_17 = []
    var_18 = {var_14: var_16, var_15: var_17}
    var_19 = 'collections'
    var_20 = 'itertools'
    var_21 = 'defaultdict'
    var_22 = [var_21]
    var_23 = 'chain'
    var_24 = [var_23]
    var_25 = {var_19: var_22, var_20: var_24}
    var_26 = {var_12: var_18, var_13: var_25}
    var_27 = {var_11: var_26}
    var_28 = 0
    var_29 = 2
    var_30 = {}
    var_31 = {}
    var_32 = module_0.ParsedContent()
    var_33 = module_1.sorted_imports(var_32)
    var_34 = True
    var_35 = True
    var_36 = module_2.Config()
    var_37 = [var_0]
    var_38 = []
    var_39 = []
    var_40 = {var_14: var_38, var_15: var_39}
    var_41 = [var_21]
    var_42 = [var_23]
    var_43 = {var_19: var_41, var_20: var_42}
    var_44 = {var_12: var_40, var_13: var_43}
    var_45 = {var_11: var_44}
    var_46 = {}
    var_47 = {}
    var_48 = module_0.ParsedContent()
    var_49 = module_1.sorted_imports(var_48, var_36)
    var_50 = '\n\nfrom itertools import chain\nfrom collections import defaultdict\n\nimport sys\nimport os'



# Parsed testcases at query #17
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'x = 1'
    var_1 = [var_0]
    var_2 = 'THIRDPARTY'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = 'sys'
    var_7 = []
    var_8 = []
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'collections'
    var_11 = 'defaultdict'
    var_12 = [var_11]
    var_13 = {var_10: var_12}
    var_14 = {var_3: var_9, var_4: var_13}
    var_15 = {var_2: var_14}
    var_16 = 0
    var_17 = 1
    var_18 = '\n'
    var_19 = {}
    var_20 = {}
    var_21 = module_0.ParsedContent()
    var_22 = module_1.Config()
    var_23 = module_2.sorted_imports(var_21, var_22)
    assert var_23 == 'import os\nimport sys\n\nfrom collections import defaultdict\n\nx = 1'
    var_24 = [var_0]
    var_25 = 'FIRSTPARTY'
    var_26 = []
    var_27 = []
    var_28 = {var_5: var_26, var_6: var_27}
    var_29 = [var_11]
    var_30 = {var_10: var_29}
    var_31 = {var_3: var_28, var_4: var_30}
    var_32 = 'myapp'
    var_33 = []
    var_34 = {var_32: var_33}
    var_35 = 'myapp.utils'
    var_36 = 'helper'
    var_37 = [var_36]
    var_38 = {var_35: var_37}
    var_39 = {var_3: var_34, var_4: var_38}
    var_40 = {var_2: var_31, var_25: var_39}
    var_41 = {}
    var_42 = {}
    var_43 = module_0.ParsedContent()
    var_44 = True
    var_45 = module_1.Config()
    var_46 = module_2.sorted_imports(var_43, var_45)
    var_47 = [var_0]
    var_48 = []
    var_49 = {var_5: var_48}
    var_50 = [var_11]
    var_51 = {var_10: var_50}
    var_52 = {var_3: var_49, var_4: var_51}
    var_53 = {var_2: var_52}
    var_54 = {}
    var_55 = {}
    var_56 = module_0.ParsedContent()
    var_57 = True
    var_58 = module_1.Config()
    var_59 = module_2.sorted_imports(var_56, var_58)
    var_60 = 'from collections import defaultdict\n\nimport os'
    var_61 = [var_0]
    var_62 = {}
    var_63 = '*'
    var_64 = [var_63]
    var_65 = 'path'
    var_66 = [var_65]
    var_67 = {var_10: var_64, var_5: var_66}
    var_68 = {var_3: var_62, var_4: var_67}
    var_69 = {var_2: var_68}
    var_70 = {}
    var_71 = {}
    var_72 = module_0.ParsedContent()
    var_73 = True
    var_74 = module_1.Config()
    var_75 = module_2.sorted_imports(var_72, var_74)
    var_76 = 'from collections import *'
    var_77 = 'from os import path'
    var_78 = [var_0]
    var_79 = []
    var_80 = {var_5: var_79}
    var_81 = {}
    var_82 = {var_3: var_80, var_4: var_81}
    var_83 = {var_2: var_82}
    var_84 = {}
    var_85 = {}
    var_86 = module_0.ParsedContent()
    var_87 = 'thirdparty'
    var_88 = 'Third Party Imports'
    var_89 = {var_87: var_88}
    var_90 = module_1.Config()
    var_91 = module_2.sorted_imports(var_86, var_90)
    var_92 = [var_0]
    var_93 = []
    var_94 = []
    var_95 = {var_5: var_93, var_6: var_94}
    var_96 = [var_11]
    var_97 = {var_10: var_96}
    var_98 = {var_3: var_95, var_4: var_97}
    var_99 = {var_2: var_98}
    var_100 = {}
    var_101 = {}
    var_102 = module_0.ParsedContent()
    var_103 = [var_5]
    var_104 = module_1.Config()
    var_105 = module_2.sorted_imports(var_102, var_104)
    var_106 = [var_0]
    var_107 = {}
    var_108 = -1
    var_109 = {}
    var_110 = {}
    var_111 = module_0.ParsedContent()
    var_112 = module_1.Config()
    var_113 = module_2.sorted_imports(var_111, var_112)
    assert var_113 == 'x = 1'
    var_114 = 'y = 2'
    var_115 = [var_0, var_114]
    var_116 = []
    var_117 = {var_5: var_116}
    var_118 = {}
    var_119 = {var_3: var_117, var_4: var_118}
    var_120 = {var_2: var_119}
    var_121 = 2
    var_122 = 'import os'
    var_123 = [var_122]
    var_124 = {var_2: var_123}
    var_125 = {var_114: var_2}
    var_126 = module_0.ParsedContent()
    var_127 = module_1.Config()
    var_128 = module_2.sorted_imports(var_126, var_127)
    assert var_128 == 'x = 1\ny = 2\nimport os\n'



# Parsed testcases at query #18
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = '\n'
    var_5 = 1
    var_6 = {}
    var_7 = {}
    var_8 = module_0.ParsedContent()
    var_9 = module_1.Config()
    var_10 = module_2.sorted_imports(var_8, var_9)
    assert var_10 == "print('hello')"
    var_11 = [var_0]
    var_12 = 'THIRDPARTY'
    var_13 = 'straight'
    var_14 = 'from'
    var_15 = 'os'
    var_16 = 'sys'
    var_17 = [var_15]
    var_18 = [var_16]
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = 'collections'
    var_21 = 'defaultdict'
    var_22 = [var_21]
    var_23 = {var_20: var_22}
    var_24 = {var_13: var_19, var_14: var_23}
    var_25 = {var_12: var_24}
    var_26 = 0
    var_27 = 2
    var_28 = {}
    var_29 = {}
    var_30 = module_0.ParsedContent()
    var_31 = module_1.Config()
    var_32 = module_2.sorted_imports(var_30, var_31)
    var_33 = 'import os'
    var_34 = 'import sys'
    var_35 = ''
    var_36 = 'from collections import defaultdict'
    var_37 = [var_33, var_34, var_35, var_36, var_35, var_0]
    var_38 = [var_0]
    var_39 = 'FUTURE'
    var_40 = 'STDLIB'
    var_41 = '__future__'
    var_42 = 'annotations'
    var_43 = [var_42]
    var_44 = {var_41: var_43}
    var_45 = {}
    var_46 = {var_13: var_44, var_14: var_45}
    var_47 = [var_15]
    var_48 = {var_15: var_47}
    var_49 = 'argv'
    var_50 = [var_49]
    var_51 = {var_16: var_50}
    var_52 = {var_13: var_48, var_14: var_51}
    var_53 = 'django'
    var_54 = [var_53]
    var_55 = {var_53: var_54}
    var_56 = {}
    var_57 = {var_13: var_55, var_14: var_56}
    var_58 = {var_39: var_46, var_40: var_52, var_12: var_57}
    var_59 = {}
    var_60 = {}
    var_61 = module_0.ParsedContent()
    var_62 = 'future'
    var_63 = 'stdlib'
    var_64 = 'thirdparty'
    var_65 = 'Future imports'
    var_66 = 'Standard library imports'
    var_67 = 'Third party imports'
    var_68 = {var_62: var_65, var_63: var_66, var_64: var_67}
    var_69 = module_1.Config()
    var_70 = module_2.sorted_imports(var_61, var_69)
    var_71 = '# Future imports'
    var_72 = 'from __future__ import annotations'
    var_73 = '# Standard library imports'
    var_74 = 'from sys import argv'
    var_75 = '# Third party imports'
    var_76 = 'import django'
    var_77 = [var_71, var_72, var_35, var_73, var_33, var_35, var_74, var_35, var_75, var_76, var_35, var_0]
    var_78 = [var_0]
    var_79 = [var_15]
    var_80 = [var_16]
    var_81 = {var_15: var_79, var_16: var_80}
    var_82 = [var_21]
    var_83 = {var_20: var_82}
    var_84 = {var_13: var_81, var_14: var_83}
    var_85 = {var_12: var_84}
    var_86 = {}
    var_87 = {}
    var_88 = module_0.ParsedContent()
    var_89 = [var_15]
    var_90 = module_1.Config()
    var_91 = module_2.sorted_imports(var_88, var_90)
    var_92 = [var_34, var_35, var_36, var_35, var_0]
    var_93 = [var_0]
    var_94 = 'flask'
    var_95 = [var_53]
    var_96 = [var_94]
    var_97 = {var_53: var_95, var_94: var_96}
    var_98 = 'numpy'
    var_99 = 'pandas'
    var_100 = 'array'
    var_101 = [var_100]
    var_102 = 'DataFrame'
    var_103 = [var_102]
    var_104 = {var_98: var_101, var_99: var_103}
    var_105 = {var_13: var_97, var_14: var_104}
    var_106 = {var_12: var_105}
    var_107 = {}
    var_108 = {}
    var_109 = module_0.ParsedContent()
    var_110 = True
    var_111 = module_1.Config()
    var_112 = module_2.sorted_imports(var_109, var_111)
    var_113 = 'import flask'
    var_114 = 'from numpy import array'
    var_115 = 'from pandas import DataFrame'
    var_116 = [var_76, var_113, var_35, var_114, var_115, var_35, var_0]



# Parsed testcases at query #19
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 'THIRDPARTY'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = 'sys'
    var_7 = []
    var_8 = []
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'collections'
    var_11 = 'itertools'
    var_12 = 'defaultdict'
    var_13 = [var_12]
    var_14 = 'chain'
    var_15 = [var_14]
    var_16 = {var_10: var_13, var_11: var_15}
    var_17 = {var_3: var_9, var_4: var_16}
    var_18 = {var_2: var_17}
    var_19 = 0
    var_20 = 1
    var_21 = '\n'
    var_22 = module_0.ParsedContent()
    var_23 = module_1.Config()
    var_24 = module_2.sorted_imports(var_22, var_23)
    assert var_24 == 'from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n'
    var_25 = [var_0]
    var_26 = 'FUTURE'
    var_27 = {}
    var_28 = {}
    var_29 = {var_3: var_27, var_4: var_28}
    var_30 = []
    var_31 = []
    var_32 = {var_5: var_30, var_6: var_31}
    var_33 = [var_12]
    var_34 = [var_14]
    var_35 = {var_10: var_33, var_11: var_34}
    var_36 = {var_3: var_32, var_4: var_35}
    var_37 = {var_26: var_29, var_2: var_36}
    var_38 = module_0.ParsedContent()
    var_39 = True
    var_40 = module_1.Config()
    var_41 = module_2.sorted_imports(var_38, var_40)
    assert var_41 == 'from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n'
    var_42 = [var_0]
    var_43 = []
    var_44 = []
    var_45 = {var_5: var_43, var_6: var_44}
    var_46 = [var_12]
    var_47 = [var_14]
    var_48 = {var_10: var_46, var_11: var_47}
    var_49 = {var_3: var_45, var_4: var_48}
    var_50 = {var_2: var_49}
    var_51 = module_0.ParsedContent()
    var_52 = 'LOCALFOLDER'
    var_53 = [var_52]
    var_54 = module_1.Config()
    var_55 = module_2.sorted_imports(var_51, var_54)
    assert var_55 == 'from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n'
    var_56 = [var_0]
    var_57 = []
    var_58 = []
    var_59 = {var_5: var_57, var_6: var_58}
    var_60 = [var_12]
    var_61 = [var_14]
    var_62 = {var_10: var_60, var_11: var_61}
    var_63 = {var_3: var_59, var_4: var_62}
    var_64 = {var_2: var_63}
    var_65 = module_0.ParsedContent()
    var_66 = [var_5, var_6]
    var_67 = module_1.Config()
    var_68 = module_2.sorted_imports(var_65, var_67)
    assert var_68 == 'from collections import defaultdict\nfrom itertools import chain\n'
    var_69 = [var_0]
    var_70 = []
    var_71 = []
    var_72 = {var_5: var_70, var_6: var_71}
    var_73 = [var_12]
    var_74 = [var_14]
    var_75 = {var_10: var_73, var_11: var_74}
    var_76 = {var_3: var_72, var_4: var_75}
    var_77 = {var_2: var_76}
    var_78 = module_0.ParsedContent()
    var_79 = module_1.Config()
    var_80 = module_2.sorted_imports(var_78, var_79)
    assert var_80 == 'from collections import defaultdict\nfrom itertools import chain\n\nimport os\nimport sys\n'
    var_81 = [var_0]
    var_82 = []
    var_83 = []
    var_84 = {var_5: var_82, var_6: var_83}
    var_85 = [var_12]
    var_86 = [var_14]
    var_87 = {var_10: var_85, var_11: var_86}
    var_88 = {var_3: var_84, var_4: var_87}
    var_89 = {var_2: var_88}
    var_90 = module_0.ParsedContent()
    var_91 = True
    var_92 = module_1.Config()
    var_93 = module_2.sorted_imports(var_90, var_92)
    assert var_93 == 'from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n'
    var_94 = [var_0]
    var_95 = {}
    var_96 = '*'
    var_97 = [var_96]
    var_98 = [var_14]
    var_99 = {var_10: var_97, var_11: var_98}
    var_100 = {var_3: var_95, var_4: var_99}
    var_101 = {var_2: var_100}
    var_102 = module_0.ParsedContent()
    var_103 = True
    var_104 = module_1.Config()
    var_105 = module_2.sorted_imports(var_102, var_104)
    assert var_105 == 'from collections import *\nfrom itertools import chain\n'
    var_106 = [var_0]
    var_107 = []
    var_108 = []
    var_109 = {var_5: var_107, var_6: var_108}
    var_110 = [var_12]
    var_111 = [var_14]
    var_112 = {var_10: var_110, var_11: var_111}
    var_113 = {var_3: var_109, var_4: var_112}
    var_114 = {var_2: var_113}
    var_115 = module_0.ParsedContent()
    var_116 = 'thirdparty'
    var_117 = 'Third Party Imports'
    var_118 = {var_116: var_117}
    var_119 = module_1.Config()
    var_120 = module_2.sorted_imports(var_115, var_119)
    assert var_120 == '# Third Party Imports\nfrom collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n'
    var_121 = [var_0]
    var_122 = []
    var_123 = []
    var_124 = {var_5: var_122, var_6: var_123}
    var_125 = [var_12]
    var_126 = [var_14]
    var_127 = {var_10: var_125, var_11: var_126}
    var_128 = {var_3: var_124, var_4: var_127}
    var_129 = {var_2: var_128}
    var_130 = module_0.ParsedContent()
    var_131 = 'End of Third Party Imports'
    var_132 = {var_116: var_131}
    var_133 = module_1.Config()
    var_134 = module_2.sorted_imports(var_130, var_133)
    assert var_134 == 'from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n\n# End of Third Party Imports\n'
    var_135 = [var_0]
    var_136 = {}
    var_137 = {}
    var_138 = {var_3: var_136, var_4: var_137}
    var_139 = []
    var_140 = []
    var_141 = {var_5: var_139, var_6: var_140}
    var_142 = [var_12]
    var_143 = [var_14]
    var_144 = {var_10: var_142, var_11: var_143}
    var_145 = {var_3: var_141, var_4: var_144}
    var_146 = {var_26: var_138, var_2: var_145}
    var_147 = module_0.ParsedContent()
    var_148 = module_1.Config()
    var_149 = module_2.sorted_imports(var_147, var_148)
    assert var_149 == 'from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n'
    var_150 = [var_0]
    var_151 = []
    var_152 = []
    var_153 = {var_5: var_151, var_6: var_152}
    var_154 = [var_12]
    var_155 = [var_14]
    var_156 = {var_10: var_154, var_11: var_155}
    var_157 = {var_3: var_153, var_4: var_156}
    var_158 = {var_2: var_157}
    var_159 = module_0.ParsedContent()
    var_160 = True
    var_161 = module_1.Config()
    var_162 = module_2.sorted_imports(var_159, var_161)
    assert var_162 == 'from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n'
    var_163 = [var_0]
    var_164 = []
    var_165 = []
    var_166 = {var_5: var_164, var_6: var_165}
    var_167 = [var_12]
    var_168 = [var_14]
    var_169 = {var_10: var_167, var_11: var_168}
    var_170 = {var_3: var_166, var_4: var_169}
    var_171 = {var_2: var_170}
    var_172 = module_0.ParsedContent()
    var_173 = lambda x, y, z: x
    var_174 = module_1.Config()
    var_175 = module_2.sorted_imports(var_172, var_174)
    assert var_175 == 'from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n'
    var_176 = [var_0]
    var_177 = []
    var_178 = []
    var_179 = {var_5: var_177, var_6: var_178}
    var_180 = [var_12]
    var_181 = [var_14]
    var_182 = {var_10: var_180, var_11: var_181}
    var_183 = {var_3: var_179, var_4: var_182}
    var_184 = {var_2: var_183}
    var_185 = module_0.ParsedContent()
    var_186 = module_1.Config()
    var_187 = module_2.sorted_imports(var_185, var_186)
    assert var_187 == '\nfrom collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n'
    var_188 = [var_0]
    var_189 = []
    var_190 = []
    var_191 = {var_5: var_189, var_6: var_190}
    var_192 = [var_12]
    var_193 = [var_14]
    var_194 = {var_10: var_192, var_11: var_193}
    var_195 = {var_3: var_191, var_4: var_194}
    var_196 = {var_2: var_195}
    var_197 = module_0.ParsedContent()
    var_198 = module_1.Config()
    var_199 = module_2.sorted_imports(var_197, var_198)
    assert var_199 == 'from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n\n'



# Parsed testcases at query #20
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = 0
    var_5 = '\n'
    var_6 = {}
    var_7 = {}
    var_8 = module_0.ParsedContent()
    var_9 = "print('hello')"
    var_10 = [var_9]
    var_11 = {}
    var_12 = -1
    var_13 = 1
    var_14 = {}
    var_15 = {}
    var_16 = module_0.ParsedContent()
    var_17 = [var_0]
    var_18 = 'THIRDPARTY'
    var_19 = 'straight'
    var_20 = 'from'
    var_21 = 'zlib'
    var_22 = 'os'
    var_23 = []
    var_24 = []
    var_25 = {var_21: var_23, var_22: var_24}
    var_26 = 'sys'
    var_27 = 'json'
    var_28 = 'path'
    var_29 = [var_28]
    var_30 = 'load'
    var_31 = [var_30]
    var_32 = {var_26: var_29, var_27: var_31}
    var_33 = {var_19: var_25, var_20: var_32}
    var_34 = {var_18: var_33}
    var_35 = {}
    var_36 = {}
    var_37 = module_0.ParsedContent()
    var_38 = 'import os\nimport zlib\n\nfrom json import load\nfrom sys import path'
    var_39 = [var_0]
    var_40 = 'FUTURE'
    var_41 = '__future__'
    var_42 = 'annotations'
    var_43 = [var_42]
    var_44 = {var_41: var_43}
    var_45 = {}
    var_46 = {var_19: var_44, var_20: var_45}
    var_47 = []
    var_48 = []
    var_49 = {var_21: var_47, var_22: var_48}
    var_50 = {}
    var_51 = {var_19: var_49, var_20: var_50}
    var_52 = {var_40: var_46, var_18: var_51}
    var_53 = {}
    var_54 = {}
    var_55 = module_0.ParsedContent()
    var_56 = 'LOCALFOLDER'
    var_57 = [var_56]
    var_58 = module_1.Config()
    var_59 = module_2.sorted_imports(var_55, var_58)
    var_60 = 'from __future__ import annotations\n\nimport os\nimport zlib'
    var_61 = [var_0]
    var_62 = [var_42]
    var_63 = {var_41: var_62}
    var_64 = {}
    var_65 = {var_19: var_63, var_20: var_64}
    var_66 = []
    var_67 = []
    var_68 = {var_21: var_66, var_22: var_67}
    var_69 = {}
    var_70 = {var_19: var_68, var_20: var_69}
    var_71 = {var_40: var_65, var_18: var_70}
    var_72 = {}
    var_73 = {}
    var_74 = module_0.ParsedContent()
    var_75 = True
    var_76 = module_1.Config()
    var_77 = module_2.sorted_imports(var_74, var_76)
    var_78 = 'from __future__ import annotations\n\nimport os\nimport zlib'
    var_79 = [var_0]
    var_80 = []
    var_81 = []
    var_82 = {var_21: var_80, var_22: var_81}
    var_83 = {}
    var_84 = {var_19: var_82, var_20: var_83}
    var_85 = {var_18: var_84}
    var_86 = {}
    var_87 = {}
    var_88 = module_0.ParsedContent()
    var_89 = 'thirdparty'
    var_90 = 'Third Party Imports'
    var_91 = {var_89: var_90}
    var_92 = module_1.Config()
    var_93 = module_2.sorted_imports(var_88, var_92)
    var_94 = '# Third Party Imports\nimport os\nimport zlib'
    var_95 = [var_0]
    var_96 = [var_42]
    var_97 = {var_41: var_96}
    var_98 = {}
    var_99 = {var_19: var_97, var_20: var_98}
    var_100 = []
    var_101 = []
    var_102 = {var_21: var_100, var_22: var_101}
    var_103 = {}
    var_104 = {var_19: var_102, var_20: var_103}
    var_105 = {var_40: var_99, var_18: var_104}
    var_106 = {}
    var_107 = {}
    var_108 = module_0.ParsedContent()
    var_109 = 2
    var_110 = module_1.Config()
    var_111 = module_2.sorted_imports(var_108, var_110)
    var_112 = 'from __future__ import annotations\n\n\n\nimport os\nimport zlib'
    var_113 = [var_0]
    var_114 = {}
    var_115 = '*'
    var_116 = [var_115]
    var_117 = [var_30]
    var_118 = {var_26: var_116, var_27: var_117}
    var_119 = {var_19: var_114, var_20: var_118}
    var_120 = {var_18: var_119}
    var_121 = {}
    var_122 = {}
    var_123 = module_0.ParsedContent()
    var_124 = True
    var_125 = module_1.Config()
    var_126 = module_2.sorted_imports(var_123, var_125)
    var_127 = 'from sys import *\nfrom json import load'
    var_128 = [var_0]
    var_129 = []
    var_130 = []
    var_131 = {var_21: var_129, var_22: var_130}
    var_132 = [var_28]
    var_133 = [var_30]
    var_134 = {var_26: var_132, var_27: var_133}
    var_135 = {var_19: var_131, var_20: var_134}
    var_136 = {var_18: var_135}
    var_137 = {}
    var_138 = {}
    var_139 = module_0.ParsedContent()
    var_140 = True
    var_141 = module_1.Config()
    var_142 = module_2.sorted_imports(var_139, var_141)
    var_143 = 'from json import load\nfrom sys import path\n\nimport os\nimport zlib'
    var_144 = [var_0]
    var_145 = []
    var_146 = []
    var_147 = {var_21: var_145, var_22: var_146}
    var_148 = [var_28]
    var_149 = [var_30]
    var_150 = {var_26: var_148, var_27: var_149}
    var_151 = {var_19: var_147, var_20: var_150}
    var_152 = {var_18: var_151}
    var_153 = {}
    var_154 = {}
    var_155 = module_0.ParsedContent()
    var_156 = True
    var_157 = module_1.Config()
    var_158 = module_2.sorted_imports(var_155, var_157)
    var_159 = 'import os\nimport zlib\n\nfrom json import load\nfrom sys import path'



# Parsed testcases at query #21
#--------------------------


import isort.parse as module_0
import isort.output as module_1
import isort.settings as module_2

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = '\n'
    var_5 = 1
    var_6 = {}
    var_7 = {}
    var_8 = module_0.ParsedContent()
    var_9 = module_1.sorted_imports(var_8)
    assert var_9 == "print('hello')\n"
    var_10 = [var_0]
    var_11 = 'THIRDPARTY'
    var_12 = 'straight'
    var_13 = 'from'
    var_14 = 'os'
    var_15 = 'sys'
    var_16 = 'os.path'
    var_17 = [var_16]
    var_18 = 'sys.argv'
    var_19 = [var_18]
    var_20 = {var_14: var_17, var_15: var_19}
    var_21 = 'collections'
    var_22 = 'defaultdict'
    var_23 = [var_22]
    var_24 = {var_21: var_23}
    var_25 = {var_12: var_20, var_13: var_24}
    var_26 = {var_11: var_25}
    var_27 = 0
    var_28 = 2
    var_29 = {}
    var_30 = {}
    var_31 = module_0.ParsedContent()
    var_32 = False
    var_33 = module_2.Config()
    var_34 = module_1.sorted_imports(var_31, var_33)
    var_35 = "import os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')\n"
    var_36 = [var_0]
    var_37 = 'FUTURE'
    var_38 = '__future__'
    var_39 = 'print_function'
    var_40 = [var_39]
    var_41 = {var_38: var_40}
    var_42 = {}
    var_43 = {var_12: var_41, var_13: var_42}
    var_44 = [var_16]
    var_45 = [var_18]
    var_46 = {var_14: var_44, var_15: var_45}
    var_47 = [var_22]
    var_48 = {var_21: var_47}
    var_49 = {var_12: var_46, var_13: var_48}
    var_50 = {var_37: var_43, var_11: var_49}
    var_51 = {}
    var_52 = {}
    var_53 = module_0.ParsedContent()
    var_54 = 'future'
    var_55 = 'thirdparty'
    var_56 = 'Future imports'
    var_57 = 'Third party imports'
    var_58 = {var_54: var_56, var_55: var_57}
    var_59 = module_2.Config()
    var_60 = module_1.sorted_imports(var_53, var_59)
    var_61 = "# Future imports\nfrom __future__ import print_function\n\n# Third party imports\nimport os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')\n"
    var_62 = [var_0]
    var_63 = [var_16]
    var_64 = [var_18]
    var_65 = {var_14: var_63, var_15: var_64}
    var_66 = [var_22]
    var_67 = {var_21: var_66}
    var_68 = {var_12: var_65, var_13: var_67}
    var_69 = {var_11: var_68}
    var_70 = {}
    var_71 = {}
    var_72 = module_0.ParsedContent()
    var_73 = [var_14]
    var_74 = module_2.Config()
    var_75 = module_1.sorted_imports(var_72, var_74)
    var_76 = "import sys\n\nfrom collections import defaultdict\n\nprint('hello')\n"
    var_77 = [var_0]
    var_78 = [var_39]
    var_79 = {var_38: var_78}
    var_80 = {}
    var_81 = {var_12: var_79, var_13: var_80}
    var_82 = [var_16]
    var_83 = [var_18]
    var_84 = {var_14: var_82, var_15: var_83}
    var_85 = [var_22]
    var_86 = {var_21: var_85}
    var_87 = {var_12: var_84, var_13: var_86}
    var_88 = {var_37: var_81, var_11: var_87}
    var_89 = {}
    var_90 = {}
    var_91 = module_0.ParsedContent()
    var_92 = True
    var_93 = module_2.Config()
    var_94 = module_1.sorted_imports(var_91, var_93)
    var_95 = "from __future__ import print_function\nimport os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')\n"



# Parsed testcases at query #22
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = -1
    var_3 = '\n'
    var_4 = {}
    var_5 = {}
    var_6 = {}
    var_7 = []
    var_8 = 1
    var_9 = module_0.ParsedContent()
    var_10 = module_1.Config()
    var_11 = module_2.sorted_imports(var_9, var_10)
    assert var_11 == "print('hello')"
    var_12 = [var_0]
    var_13 = 0
    var_14 = 'THIRDPARTY'
    var_15 = 'straight'
    var_16 = 'from'
    var_17 = 'os'
    var_18 = 'sys'
    var_19 = [var_17]
    var_20 = [var_18]
    var_21 = {var_17: var_19, var_18: var_20}
    var_22 = 'collections'
    var_23 = 'defaultdict'
    var_24 = 'Counter'
    var_25 = [var_23, var_24]
    var_26 = {var_22: var_25}
    var_27 = {var_15: var_21, var_16: var_26}
    var_28 = {var_14: var_27}
    var_29 = {}
    var_30 = {}
    var_31 = [var_14]
    var_32 = module_0.ParsedContent()
    var_33 = module_1.Config()
    var_34 = module_2.sorted_imports(var_32, var_33)
    assert var_34 == "import os\nimport sys\n\nfrom collections import Counter, defaultdict\n\nprint('hello')"
    var_35 = [var_0]
    var_36 = [var_17]
    var_37 = [var_18]
    var_38 = {var_17: var_36, var_18: var_37}
    var_39 = [var_23, var_24]
    var_40 = {var_22: var_39}
    var_41 = {var_15: var_38, var_16: var_40}
    var_42 = {var_14: var_41}
    var_43 = {}
    var_44 = {}
    var_45 = [var_14]
    var_46 = module_0.ParsedContent()
    var_47 = [var_17]
    var_48 = module_1.Config()
    var_49 = module_2.sorted_imports(var_46, var_48)
    assert var_49 == "import sys\n\nfrom collections import Counter, defaultdict\n\nprint('hello')"
    var_50 = [var_0]
    var_51 = 'FUTURE'
    var_52 = '__future__'
    var_53 = 'annotations'
    var_54 = [var_53]
    var_55 = {var_52: var_54}
    var_56 = {}
    var_57 = {var_15: var_55, var_16: var_56}
    var_58 = [var_17]
    var_59 = [var_18]
    var_60 = {var_17: var_58, var_18: var_59}
    var_61 = [var_23, var_24]
    var_62 = {var_22: var_61}
    var_63 = {var_15: var_60, var_16: var_62}
    var_64 = {var_51: var_57, var_14: var_63}
    var_65 = {}
    var_66 = {}
    var_67 = [var_51, var_14]
    var_68 = module_0.ParsedContent()
    var_69 = True
    var_70 = module_1.Config()
    var_71 = module_2.sorted_imports(var_68, var_70)
    assert var_71 == "from __future__ import annotations\nimport os\nimport sys\n\nfrom collections import Counter, defaultdict\n\nprint('hello')"
    var_72 = [var_0]
    var_73 = {}
    var_74 = '*'
    var_75 = [var_74]
    var_76 = 'path'
    var_77 = [var_76]
    var_78 = {var_22: var_75, var_17: var_77}
    var_79 = {var_15: var_73, var_16: var_78}
    var_80 = {var_14: var_79}
    var_81 = {}
    var_82 = {}
    var_83 = [var_14]
    var_84 = module_0.ParsedContent()
    var_85 = True
    var_86 = module_1.Config()
    var_87 = module_2.sorted_imports(var_84, var_86)
    assert var_87 == "from collections import *\nfrom os import path\n\nprint('hello')"
    var_88 = [var_0]
    var_89 = [var_17]
    var_90 = [var_18]
    var_91 = {var_17: var_89, var_18: var_90}
    var_92 = [var_23, var_24]
    var_93 = {var_22: var_92}
    var_94 = {var_15: var_91, var_16: var_93}
    var_95 = {var_14: var_94}
    var_96 = {}
    var_97 = {}
    var_98 = [var_14]
    var_99 = module_0.ParsedContent()
    var_100 = True
    var_101 = module_1.Config()
    var_102 = module_2.sorted_imports(var_99, var_101)
    assert var_102 == "from collections import Counter, defaultdict\n\nimport os\nimport sys\n\nprint('hello')"
    var_103 = [var_0]
    var_104 = [var_17]
    var_105 = [var_18]
    var_106 = {var_17: var_104, var_18: var_105}
    var_107 = [var_23, var_24]
    var_108 = {var_22: var_107}
    var_109 = {var_15: var_106, var_16: var_108}
    var_110 = {var_14: var_109}
    var_111 = {}
    var_112 = {}
    var_113 = [var_14]
    var_114 = module_0.ParsedContent()
    var_115 = 'thirdparty'
    var_116 = 'Third Party Imports'
    var_117 = {var_115: var_116}
    var_118 = module_1.Config()
    var_119 = module_2.sorted_imports(var_114, var_118)
    assert var_119 == "# Third Party Imports\nimport os\nimport sys\n\nfrom collections import Counter, defaultdict\n\nprint('hello')"
    var_120 = [var_0]
    var_121 = [var_17]
    var_122 = [var_18]
    var_123 = {var_17: var_121, var_18: var_122}
    var_124 = [var_23, var_24]
    var_125 = {var_22: var_124}
    var_126 = {var_15: var_123, var_16: var_125}
    var_127 = {var_14: var_126}
    var_128 = {}
    var_129 = {}
    var_130 = [var_14]
    var_131 = module_0.ParsedContent()
    var_132 = 'End of Third Party Imports'
    var_133 = {var_115: var_132}
    var_134 = module_1.Config()
    var_135 = module_2.sorted_imports(var_131, var_134)
    assert var_135 == "import os\nimport sys\n\nfrom collections import Counter, defaultdict\n\n# End of Third Party Imports\nprint('hello')"
    var_136 = [var_0]
    var_137 = [var_53]
    var_138 = {var_52: var_137}
    var_139 = {}
    var_140 = {var_15: var_138, var_16: var_139}
    var_141 = [var_17]
    var_142 = [var_18]
    var_143 = {var_17: var_141, var_18: var_142}
    var_144 = [var_23, var_24]
    var_145 = {var_22: var_144}
    var_146 = {var_15: var_143, var_16: var_145}
    var_147 = {var_51: var_140, var_14: var_146}
    var_148 = {}
    var_149 = {}
    var_150 = [var_51, var_14]
    var_151 = module_0.ParsedContent()
    var_152 = 2
    var_153 = module_1.Config()
    var_154 = module_2.sorted_imports(var_151, var_153)
    assert var_154 == "from __future__ import annotations\n\n\n\nimport os\nimport sys\n\nfrom collections import Counter, defaultdict\n\nprint('hello')"
    var_155 = [var_0]
    var_156 = [var_17]
    var_157 = [var_18]
    var_158 = {var_17: var_156, var_18: var_157}
    var_159 = [var_23, var_24]
    var_160 = {var_22: var_159}
    var_161 = {var_15: var_158, var_16: var_160}
    var_162 = {var_14: var_161}
    var_163 = {}
    var_164 = {}
    var_165 = [var_14]
    var_166 = module_0.ParsedContent()
    var_167 = module_1.Config()
    var_168 = module_2.sorted_imports(var_166, var_167)
    assert var_168 == "import os\nimport sys\n\nfrom collections import Counter, defaultdict\n\n\nprint('hello')"



# Parsed testcases at query #23
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = '\n'
    var_5 = 1
    var_6 = {}
    var_7 = {}
    var_8 = module_0.ParsedContent()
    var_9 = [var_0]
    var_10 = 'THIRDPARTY'
    var_11 = 'straight'
    var_12 = 'from'
    var_13 = 'os'
    var_14 = 'import os'
    var_15 = [var_14]
    var_16 = {var_13: var_15}
    var_17 = {}
    var_18 = {var_11: var_16, var_12: var_17}
    var_19 = {var_10: var_18}
    var_20 = 0
    var_21 = 2
    var_22 = {}
    var_23 = {}
    var_24 = module_0.ParsedContent()
    var_25 = [var_0]
    var_26 = 'FUTURE'
    var_27 = 'STDLIB'
    var_28 = '__future__'
    var_29 = 'from __future__ import annotations'
    var_30 = [var_29]
    var_31 = {var_28: var_30}
    var_32 = {}
    var_33 = {var_11: var_31, var_12: var_32}
    var_34 = [var_14]
    var_35 = {var_13: var_34}
    var_36 = {}
    var_37 = {var_11: var_35, var_12: var_36}
    var_38 = 'numpy'
    var_39 = 'import numpy'
    var_40 = [var_39]
    var_41 = {var_38: var_40}
    var_42 = {}
    var_43 = {var_11: var_41, var_12: var_42}
    var_44 = {var_26: var_33, var_27: var_37, var_10: var_43}
    var_45 = 4
    var_46 = {}
    var_47 = {}
    var_48 = module_0.ParsedContent()
    var_49 = "from __future__ import annotations\n\nimport os\n\nimport numpy\n\nprint('hello')\n"
    var_50 = [var_0]
    var_51 = 'pandas'
    var_52 = [var_39]
    var_53 = 'import pandas'
    var_54 = [var_53]
    var_55 = {var_38: var_52, var_51: var_54}
    var_56 = 'from numpy import array'
    var_57 = [var_56]
    var_58 = {var_38: var_57}
    var_59 = {var_11: var_55, var_12: var_58}
    var_60 = {var_10: var_59}
    var_61 = {}
    var_62 = {}
    var_63 = module_0.ParsedContent()
    var_64 = True
    var_65 = True
    var_66 = module_1.Config()
    var_67 = "from numpy import array\n\nimport numpy\nimport pandas\n\n\nprint('hello')\n"
    var_68 = module_2.sorted_imports(var_63, var_66)
    var_69 = [var_0]
    var_70 = [var_39]
    var_71 = {var_38: var_70}
    var_72 = [var_56]
    var_73 = {var_38: var_72}
    var_74 = {var_11: var_71, var_12: var_73}
    var_75 = {var_10: var_74}
    var_76 = 3
    var_77 = {}
    var_78 = {}
    var_79 = module_0.ParsedContent()
    var_80 = [var_10]
    var_81 = module_1.Config()
    var_82 = "from numpy import array\nimport numpy\n\nprint('hello')\n"
    var_83 = module_2.sorted_imports(var_79, var_81)



# Parsed testcases at query #24
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2
import re as module_3

def test_case_0():
    var_0 = '# Test file'
    var_1 = ''
    var_2 = [var_0, var_1]
    var_3 = 'THIRDPARTY'
    var_4 = 'straight'
    var_5 = 'from'
    var_6 = 'os'
    var_7 = 'sys'
    var_8 = None
    var_9 = {var_6: var_8, var_7: var_8}
    var_10 = 'collections'
    var_11 = 'defaultdict'
    var_12 = 'Counter'
    var_13 = [var_11, var_12]
    var_14 = {var_10: var_13}
    var_15 = {var_4: var_9, var_5: var_14}
    var_16 = {var_3: var_15}
    var_17 = 1
    var_18 = '\n'
    var_19 = 2
    var_20 = {}
    var_21 = {}
    var_22 = module_0.ParsedContent()
    var_23 = [var_0, var_1]
    var_24 = 'FUTURE'
    var_25 = '__future__'
    var_26 = {var_25: var_8}
    var_27 = {}
    var_28 = {var_4: var_26, var_5: var_27}
    var_29 = {var_6: var_8}
    var_30 = [var_12]
    var_31 = {var_10: var_30}
    var_32 = {var_4: var_29, var_5: var_31}
    var_33 = {var_24: var_28, var_3: var_32}
    var_34 = {}
    var_35 = {}
    var_36 = module_0.ParsedContent()
    var_37 = True
    var_38 = module_1.Config()
    var_39 = module_2.sorted_imports(var_36, var_38)
    var_40 = [var_0, var_1]
    var_41 = 'django'
    var_42 = 'flask'
    var_43 = {var_41: var_8, var_42: var_8}
    var_44 = 'path'
    var_45 = [var_44]
    var_46 = 'argv'
    var_47 = [var_46]
    var_48 = {var_6: var_45, var_7: var_47}
    var_49 = {var_4: var_43, var_5: var_48}
    var_50 = {var_3: var_49}
    var_51 = {}
    var_52 = {}
    var_53 = module_0.ParsedContent()
    var_54 = True
    var_55 = module_1.Config()
    var_56 = module_2.sorted_imports(var_53, var_55)
    var_57 = [var_0, var_1]
    var_58 = {var_6: var_8, var_7: var_8}
    var_59 = [var_12]
    var_60 = {var_10: var_59}
    var_61 = {var_4: var_58, var_5: var_60}
    var_62 = {var_3: var_61}
    var_63 = {}
    var_64 = {}
    var_65 = module_0.ParsedContent()
    var_66 = [var_6]
    var_67 = module_1.Config()
    var_68 = module_2.sorted_imports(var_65, var_67)
    var_69 = [var_0, var_1]
    var_70 = 'numpy'
    var_71 = {var_70: var_8}
    var_72 = 'pandas'
    var_73 = 'DataFrame'
    var_74 = [var_73]
    var_75 = {var_72: var_74}
    var_76 = {var_4: var_71, var_5: var_75}
    var_77 = {var_3: var_76}
    var_78 = {}
    var_79 = {}
    var_80 = module_0.ParsedContent()
    var_81 = 'thirdparty'
    var_82 = 'Third Party Imports'
    var_83 = {var_81: var_82}
    var_84 = module_1.Config()
    var_85 = module_2.sorted_imports(var_80, var_84)
    var_86 = [var_0, var_1]
    var_87 = 'STDLIB'
    var_88 = {var_25: var_8}
    var_89 = {}
    var_90 = {var_4: var_88, var_5: var_89}
    var_91 = {var_6: var_8}
    var_92 = {}
    var_93 = {var_4: var_91, var_5: var_92}
    var_94 = {var_41: var_8}
    var_95 = {}
    var_96 = {var_4: var_94, var_5: var_95}
    var_97 = {var_24: var_90, var_87: var_93, var_3: var_96}
    var_98 = {}
    var_99 = {}
    var_100 = module_0.ParsedContent()
    var_101 = module_1.Config()
    var_102 = module_2.sorted_imports(var_100, var_101)
    var_103 = module_3.split(var_18)
    var_104 = 'import __future__'
    var_105 = 'import os'
    var_106 = 'import django'
    var_107 = [var_0, var_1]
    var_108 = {var_6: var_8}
    var_109 = [var_46]
    var_110 = {var_7: var_109}
    var_111 = {var_4: var_108, var_5: var_110}
    var_112 = {var_3: var_111}
    var_113 = {}
    var_114 = {}
    var_115 = module_0.ParsedContent()
    var_116 = True
    var_117 = module_1.Config()
    var_118 = module_2.sorted_imports(var_115, var_117)
    var_119 = module_3.split(var_18)
    var_120 = 'from sys import argv'
    var_121 = [var_0, var_1]
    var_122 = {}
    var_123 = '*'
    var_124 = [var_123]
    var_125 = [var_73]
    var_126 = {var_70: var_124, var_72: var_125}
    var_127 = {var_4: var_122, var_5: var_126}
    var_128 = {var_3: var_127}
    var_129 = {}
    var_130 = {}
    var_131 = module_0.ParsedContent()
    var_132 = True
    var_133 = module_1.Config()
    var_134 = module_2.sorted_imports(var_131, var_133)
    var_135 = module_3.split(var_18)
    var_136 = 'from numpy import *'
    var_137 = 'from pandas import DataFrame'
    var_138 = 'x = 1  # comment'
    var_139 = [var_0, var_138]
    var_140 = {var_6: var_8}
    var_141 = {}
    var_142 = {var_4: var_140, var_5: var_141}
    var_143 = {var_3: var_142}
    var_144 = {}
    var_145 = {}
    var_146 = module_0.ParsedContent()
    var_147 = True
    var_148 = module_1.Config()
    var_149 = module_2.sorted_imports(var_146, var_148)
    var_150 = module_3.split(var_18)
    var_151 = 'x = 1'
    var_152 = '# PLACE_HOLDER'
    var_153 = [var_0, var_151, var_152]
    var_154 = {var_6: var_8}
    var_155 = {}
    var_156 = {var_4: var_154, var_5: var_155}
    var_157 = {var_3: var_156}
    var_158 = 3
    var_159 = 'PLACE_HOLDER'
    var_160 = 'import sys'
    var_161 = [var_160]
    var_162 = {var_159: var_161}
    var_163 = {var_152: var_159}
    var_164 = module_0.ParsedContent()
    var_165 = module_2.sorted_imports(var_164, var_148)
    var_166 = module_3.split(var_18)



# Parsed testcases at query #25
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = 0
    var_5 = '\n'
    var_6 = {}
    var_7 = {}
    var_8 = module_0.ParsedContent()
    var_9 = "print('hello')"
    var_10 = [var_9]
    var_11 = {}
    var_12 = -1
    var_13 = 1
    var_14 = {}
    var_15 = {}
    var_16 = module_0.ParsedContent()
    var_17 = [var_0]
    var_18 = 'THIRDPARTY'
    var_19 = 'straight'
    var_20 = 'from'
    var_21 = 'zlib'
    var_22 = 'os'
    var_23 = [var_21]
    var_24 = [var_22]
    var_25 = {var_21: var_23, var_22: var_24}
    var_26 = 'sys'
    var_27 = 'import sys'
    var_28 = [var_27]
    var_29 = {var_26: var_28}
    var_30 = {var_19: var_25, var_20: var_29}
    var_31 = {var_18: var_30}
    var_32 = {}
    var_33 = {}
    var_34 = module_0.ParsedContent()
    var_35 = 'import os\nimport zlib\n\nfrom sys import sys'
    var_36 = [var_0]
    var_37 = [var_21]
    var_38 = [var_22]
    var_39 = {var_21: var_37, var_22: var_38}
    var_40 = [var_27]
    var_41 = {var_26: var_40}
    var_42 = {var_19: var_39, var_20: var_41}
    var_43 = {var_18: var_42}
    var_44 = {}
    var_45 = {}
    var_46 = module_0.ParsedContent()
    var_47 = 2
    var_48 = True
    var_49 = module_1.Config()
    var_50 = 'from sys import sys\n\nimport os\nimport zlib'
    var_51 = module_2.sorted_imports(var_46, var_49)
    var_52 = [var_0]
    var_53 = 'FIRSTPARTY'
    var_54 = [var_21]
    var_55 = {var_21: var_54}
    var_56 = {}
    var_57 = {var_19: var_55, var_20: var_56}
    var_58 = [var_22]
    var_59 = {var_22: var_58}
    var_60 = {}
    var_61 = {var_19: var_59, var_20: var_60}
    var_62 = {var_18: var_57, var_53: var_61}
    var_63 = {}
    var_64 = {}
    var_65 = module_0.ParsedContent()
    var_66 = [var_53]
    var_67 = module_1.Config()
    var_68 = 'import os\n\nimport zlib'
    var_69 = module_2.sorted_imports(var_65, var_67)
    var_70 = [var_0]
    var_71 = [var_21]
    var_72 = {var_21: var_71}
    var_73 = {}
    var_74 = {var_19: var_72, var_20: var_73}
    var_75 = [var_22]
    var_76 = {var_22: var_75}
    var_77 = {}
    var_78 = {var_19: var_76, var_20: var_77}
    var_79 = {var_18: var_74, var_53: var_78}
    var_80 = {}
    var_81 = {}
    var_82 = module_0.ParsedContent()
    var_83 = True
    var_84 = module_1.Config()
    var_85 = 'import os\nimport zlib'
    var_86 = module_2.sorted_imports(var_82, var_84)
    var_87 = [var_0]
    var_88 = {}
    var_89 = 'import *'
    var_90 = [var_89]
    var_91 = 'import path'
    var_92 = [var_91]
    var_93 = {var_26: var_90, var_22: var_92}
    var_94 = {var_19: var_88, var_20: var_93}
    var_95 = {var_18: var_94}
    var_96 = {}
    var_97 = {}
    var_98 = module_0.ParsedContent()
    var_99 = True
    var_100 = module_1.Config()
    var_101 = 'from sys import *\nfrom os import path'
    var_102 = module_2.sorted_imports(var_98, var_100)
    var_103 = [var_0]
    var_104 = [var_21]
    var_105 = {var_21: var_104}
    var_106 = {}
    var_107 = {var_19: var_105, var_20: var_106}
    var_108 = {var_18: var_107}
    var_109 = {}
    var_110 = {}
    var_111 = module_0.ParsedContent()
    var_112 = 'thirdparty'
    var_113 = 'Third Party Imports'
    var_114 = {var_112: var_113}
    var_115 = module_1.Config()
    var_116 = '# Third Party Imports\nimport zlib'
    var_117 = module_2.sorted_imports(var_111, var_115)
    var_118 = [var_0]
    var_119 = [var_21]
    var_120 = [var_22]
    var_121 = {var_21: var_119, var_22: var_120}
    var_122 = {}
    var_123 = {var_19: var_121, var_20: var_122}
    var_124 = {var_18: var_123}
    var_125 = {}
    var_126 = {}
    var_127 = module_0.ParsedContent()
    var_128 = [var_22]
    var_129 = module_1.Config()
    var_130 = 'import zlib'
    var_131 = module_2.sorted_imports(var_127, var_129)
    var_132 = "print('world')"
    var_133 = [var_9, var_132]
    var_134 = [var_21]
    var_135 = {var_21: var_134}
    var_136 = {}
    var_137 = {var_19: var_135, var_20: var_136}
    var_138 = {var_18: var_137}
    var_139 = 'import zlib'
    var_140 = [var_139]
    var_141 = {var_18: var_140}
    var_142 = {var_9: var_18}
    var_143 = module_0.ParsedContent()
    var_144 = "print('hello')\nimport zlib\n\nprint('world')"
    var_145 = module_2.sorted_imports(var_143, var_129)



# Parsed testcases at query #26
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = 'x = 1'
    var_2 = [var_0, var_1]
    var_3 = 'THIRDPARTY'
    var_4 = 'straight'
    var_5 = 'from'
    var_6 = 'os'
    var_7 = 'sys'
    var_8 = []
    var_9 = []
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'collections'
    var_12 = 'defaultdict'
    var_13 = [var_12]
    var_14 = {var_11: var_13}
    var_15 = {var_4: var_10, var_5: var_14}
    var_16 = {var_3: var_15}
    var_17 = 0
    var_18 = '\n'
    var_19 = 2
    var_20 = {}
    var_21 = {}
    var_22 = module_0.ParsedContent()
    var_23 = module_1.Config()
    var_24 = module_2.sorted_imports(var_22, var_23)
    assert var_24 == '\n\nfrom collections import defaultdict\n\nimport os\nimport sys\n\n\nx = 1\n'
    var_25 = [var_1]
    var_26 = {}
    var_27 = -1
    var_28 = 1
    var_29 = {}
    var_30 = {}
    var_31 = module_0.ParsedContent()
    var_32 = module_2.sorted_imports(var_31, var_23)
    assert var_32 == 'x = 1\n'
    var_33 = True
    var_34 = 'Third Party Imports'
    var_35 = {var_3: var_34}
    var_36 = module_1.Config()
    var_37 = [var_0, var_1]
    var_38 = []
    var_39 = []
    var_40 = {var_6: var_38, var_7: var_39}
    var_41 = [var_12]
    var_42 = {var_11: var_41}
    var_43 = {var_4: var_40, var_5: var_42}
    var_44 = {var_3: var_43}
    var_45 = {}
    var_46 = {}
    var_47 = module_0.ParsedContent()
    var_48 = module_2.sorted_imports(var_47, var_36)
    assert var_48 == '\n\n# Third Party Imports\nfrom collections import defaultdict\n\nimport os\nimport sys\n\n\nx = 1\n'
    var_49 = 'LOCALFOLDER'
    var_50 = [var_49]
    var_51 = module_1.Config()
    var_52 = [var_0, var_1]
    var_53 = []
    var_54 = {var_6: var_53}
    var_55 = {}
    var_56 = {var_4: var_54, var_5: var_55}
    var_57 = []
    var_58 = {var_7: var_57}
    var_59 = {}
    var_60 = {var_4: var_58, var_5: var_59}
    var_61 = {var_3: var_56, var_49: var_60}
    var_62 = {}
    var_63 = {}
    var_64 = module_0.ParsedContent()
    var_65 = module_2.sorted_imports(var_64, var_51)
    assert var_65 == '\n\nimport os\n\nimport sys\n\n\nx = 1\n'
    var_66 = True
    var_67 = module_1.Config()
    var_68 = [var_0, var_1]
    var_69 = []
    var_70 = []
    var_71 = {var_6: var_69, var_7: var_70}
    var_72 = [var_12]
    var_73 = {var_11: var_72}
    var_74 = {var_4: var_71, var_5: var_73}
    var_75 = {var_3: var_74}
    var_76 = {}
    var_77 = {}
    var_78 = module_0.ParsedContent()
    var_79 = module_2.sorted_imports(var_78, var_67)
    assert var_79 == '\n\nfrom collections import defaultdict\n\nimport os\nimport sys\n\n\nx = 1\n'



# Parsed testcases at query #27
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = '\n'
    var_5 = 1
    var_6 = module_0.ParsedContent()
    var_7 = [var_0]
    var_8 = 'THIRDPARTY'
    var_9 = 'straight'
    var_10 = 'from'
    var_11 = 'os'
    var_12 = 'sys'
    var_13 = 'import os'
    var_14 = [var_13]
    var_15 = 'import sys'
    var_16 = [var_15]
    var_17 = {var_11: var_14, var_12: var_16}
    var_18 = 'collections'
    var_19 = 'from collections import defaultdict'
    var_20 = [var_19]
    var_21 = {var_18: var_20}
    var_22 = {var_9: var_17, var_10: var_21}
    var_23 = {var_8: var_22}
    var_24 = 0
    var_25 = 2
    var_26 = module_0.ParsedContent()
    var_27 = "\nimport os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')"
    var_28 = [var_0]
    var_29 = 'FUTURE'
    var_30 = '__future__'
    var_31 = 'from __future__ import annotations'
    var_32 = [var_31]
    var_33 = {var_30: var_32}
    var_34 = {}
    var_35 = {var_9: var_33, var_10: var_34}
    var_36 = [var_13]
    var_37 = [var_15]
    var_38 = {var_11: var_36, var_12: var_37}
    var_39 = [var_19]
    var_40 = {var_18: var_39}
    var_41 = {var_9: var_38, var_10: var_40}
    var_42 = {var_29: var_35, var_8: var_41}
    var_43 = module_0.ParsedContent()
    var_44 = 'future'
    var_45 = 'thirdparty'
    var_46 = 'Future'
    var_47 = 'Third Party'
    var_48 = {var_44: var_46, var_45: var_47}
    var_49 = module_1.Config()
    var_50 = module_2.sorted_imports(var_43, var_49)
    var_51 = "\n# Future\nfrom __future__ import annotations\n\n# Third Party\nimport os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')"
    var_52 = [var_0]
    var_53 = [var_13]
    var_54 = [var_15]
    var_55 = {var_11: var_53, var_12: var_54}
    var_56 = [var_19]
    var_57 = {var_18: var_56}
    var_58 = {var_9: var_55, var_10: var_57}
    var_59 = {var_8: var_58}
    var_60 = module_0.ParsedContent()
    var_61 = [var_11]
    var_62 = module_1.Config()
    var_63 = module_2.sorted_imports(var_60, var_62)
    var_64 = "\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')"
    var_65 = [var_0]
    var_66 = [var_31]
    var_67 = {var_30: var_66}
    var_68 = {}
    var_69 = {var_9: var_67, var_10: var_68}
    var_70 = [var_13]
    var_71 = [var_15]
    var_72 = {var_11: var_70, var_12: var_71}
    var_73 = [var_19]
    var_74 = {var_18: var_73}
    var_75 = {var_9: var_72, var_10: var_74}
    var_76 = {var_29: var_69, var_8: var_75}
    var_77 = module_0.ParsedContent()
    var_78 = True
    var_79 = module_1.Config()
    var_80 = module_2.sorted_imports(var_77, var_79)
    var_81 = "\nfrom __future__ import annotations\nimport os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')"
    var_82 = [var_0]
    var_83 = {}
    var_84 = 'from collections import *'
    var_85 = [var_84]
    var_86 = 'from os import path'
    var_87 = [var_86]
    var_88 = {var_18: var_85, var_11: var_87}
    var_89 = {var_9: var_83, var_10: var_88}
    var_90 = {var_8: var_89}
    var_91 = module_0.ParsedContent()
    var_92 = True
    var_93 = module_1.Config()
    var_94 = module_2.sorted_imports(var_91, var_93)
    var_95 = "\nfrom collections import *\nfrom os import path\n\nprint('hello')"
    var_96 = [var_0]
    var_97 = [var_13]
    var_98 = [var_15]
    var_99 = {var_11: var_97, var_12: var_98}
    var_100 = [var_19]
    var_101 = {var_18: var_100}
    var_102 = {var_9: var_99, var_10: var_101}
    var_103 = {var_8: var_102}
    var_104 = module_0.ParsedContent()
    var_105 = True
    var_106 = module_1.Config()
    var_107 = module_2.sorted_imports(var_104, var_106)
    var_108 = "\nfrom collections import defaultdict\n\nimport os\nimport sys\n\nprint('hello')"
    var_109 = [var_0]
    var_110 = [var_31]
    var_111 = {var_30: var_110}
    var_112 = {}
    var_113 = {var_9: var_111, var_10: var_112}
    var_114 = [var_13]
    var_115 = [var_15]
    var_116 = {var_11: var_114, var_12: var_115}
    var_117 = [var_19]
    var_118 = {var_18: var_117}
    var_119 = {var_9: var_116, var_10: var_118}
    var_120 = {var_29: var_113, var_8: var_119}
    var_121 = module_0.ParsedContent()
    var_122 = module_1.Config()
    var_123 = module_2.sorted_imports(var_121, var_122)
    var_124 = "\nfrom __future__ import annotations\n\n\n\nimport os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')"
    var_125 = [var_0]
    var_126 = [var_13]
    var_127 = [var_15]
    var_128 = {var_11: var_126, var_12: var_127}
    var_129 = [var_19]
    var_130 = {var_18: var_129}
    var_131 = {var_9: var_128, var_10: var_130}
    var_132 = {var_8: var_131}
    var_133 = module_0.ParsedContent()
    var_134 = module_1.Config()
    var_135 = module_2.sorted_imports(var_133, var_134)
    var_136 = "\nimport os\nimport sys\n\nfrom collections import defaultdict\n\n\n\nprint('hello')"
    var_137 = [var_0]
    var_138 = [var_13]
    var_139 = [var_15]
    var_140 = {var_11: var_138, var_12: var_139}
    var_141 = [var_19]
    var_142 = {var_18: var_141}
    var_143 = {var_9: var_140, var_10: var_142}
    var_144 = {var_8: var_143}
    var_145 = module_0.ParsedContent()
    var_146 = True
    var_147 = module_1.Config()
    var_148 = module_2.sorted_imports(var_145, var_147)
    var_149 = "\nimport os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')"



# Parsed testcases at query #28
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = '\n'
    var_5 = 1
    var_6 = module_0.ParsedContent()
    var_7 = module_1.Config()
    var_8 = module_2.sorted_imports(var_6, var_7)
    assert var_8 == "print('hello')"
    var_9 = [var_0]
    var_10 = 'THIRDPARTY'
    var_11 = 'straight'
    var_12 = 'from'
    var_13 = 'os'
    var_14 = 'sys'
    var_15 = [var_13]
    var_16 = [var_14]
    var_17 = {var_13: var_15, var_14: var_16}
    var_18 = 'collections'
    var_19 = 'collections.OrderedDict'
    var_20 = [var_19]
    var_21 = {var_18: var_20}
    var_22 = {var_11: var_17, var_12: var_21}
    var_23 = {var_10: var_22}
    var_24 = 0
    var_25 = 2
    var_26 = module_0.ParsedContent()
    var_27 = module_1.Config()
    var_28 = module_2.sorted_imports(var_26, var_27)
    assert var_28 == "import os\nimport sys\n\nfrom collections import OrderedDict\n\nprint('hello')"
    var_29 = [var_0]
    var_30 = 'FUTURE'
    var_31 = 'STDLIB'
    var_32 = '__future__'
    var_33 = [var_32]
    var_34 = {var_32: var_33}
    var_35 = {}
    var_36 = {var_11: var_34, var_12: var_35}
    var_37 = [var_13]
    var_38 = {var_13: var_37}
    var_39 = {}
    var_40 = {var_11: var_38, var_12: var_39}
    var_41 = {}
    var_42 = 'django'
    var_43 = 'django.conf'
    var_44 = [var_43]
    var_45 = {var_42: var_44}
    var_46 = {var_11: var_41, var_12: var_45}
    var_47 = {var_30: var_36, var_31: var_40, var_10: var_46}
    var_48 = module_0.ParsedContent()
    var_49 = module_1.Config()
    var_50 = module_2.sorted_imports(var_48, var_49)
    assert var_50 == "from __future__ import absolute_import\n\nimport os\n\nfrom django import conf\n\nprint('hello')"
    var_51 = [var_0]
    var_52 = [var_13]
    var_53 = [var_14]
    var_54 = {var_13: var_52, var_14: var_53}
    var_55 = [var_19]
    var_56 = {var_18: var_55}
    var_57 = {var_11: var_54, var_12: var_56}
    var_58 = {var_10: var_57}
    var_59 = module_0.ParsedContent()
    var_60 = True
    var_61 = module_1.Config()
    var_62 = module_2.sorted_imports(var_59, var_61)
    assert var_62 == "from collections import OrderedDict\n\nimport os\nimport sys\n\nprint('hello')"
    var_63 = [var_0]
    var_64 = [var_13]
    var_65 = [var_14]
    var_66 = {var_13: var_64, var_14: var_65}
    var_67 = [var_19]
    var_68 = {var_18: var_67}
    var_69 = {var_11: var_66, var_12: var_68}
    var_70 = {var_10: var_69}
    var_71 = module_0.ParsedContent()
    var_72 = [var_13]
    var_73 = module_1.Config()
    var_74 = module_2.sorted_imports(var_71, var_73)
    assert var_74 == "import sys\n\nfrom collections import OrderedDict\n\nprint('hello')"



# Parsed testcases at query #29
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'x = 1'
    var_1 = [var_0]
    var_2 = 'STDLIB'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = 'sys'
    var_7 = [var_5]
    var_8 = [var_6]
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {}
    var_11 = {var_3: var_9, var_4: var_10}
    var_12 = {var_2: var_11}
    var_13 = 0
    var_14 = 1
    var_15 = '\n'
    var_16 = module_0.ParsedContent()
    var_17 = module_1.Config()
    var_18 = module_2.sorted_imports(var_16, var_17)
    assert var_18 == 'import os\nimport sys\n\nx = 1'
    var_19 = [var_0]
    var_20 = 'FUTURE'
    var_21 = '__future__'
    var_22 = [var_21]
    var_23 = {var_21: var_22}
    var_24 = {}
    var_25 = {var_3: var_23, var_4: var_24}
    var_26 = [var_5]
    var_27 = {var_5: var_26}
    var_28 = {}
    var_29 = {var_3: var_27, var_4: var_28}
    var_30 = {var_20: var_25, var_2: var_29}
    var_31 = module_0.ParsedContent()
    var_32 = True
    var_33 = module_1.Config()
    var_34 = module_2.sorted_imports(var_31, var_33)
    assert var_34 == 'from __future__ import __future__\nimport os\n\nx = 1'
    var_35 = [var_0]
    var_36 = [var_5]
    var_37 = {var_5: var_36}
    var_38 = {}
    var_39 = {var_3: var_37, var_4: var_38}
    var_40 = {var_2: var_39}
    var_41 = module_0.ParsedContent()
    var_42 = 'THIRDPARTY'
    var_43 = [var_42]
    var_44 = module_1.Config()
    var_45 = module_2.sorted_imports(var_41, var_44)
    assert var_45 == 'import os\n\nx = 1'
    var_46 = [var_0]
    var_47 = [var_5]
    var_48 = [var_6]
    var_49 = {var_5: var_47, var_6: var_48}
    var_50 = {}
    var_51 = {var_3: var_49, var_4: var_50}
    var_52 = {var_2: var_51}
    var_53 = module_0.ParsedContent()
    var_54 = [var_6]
    var_55 = module_1.Config()
    var_56 = module_2.sorted_imports(var_53, var_55)
    assert var_56 == 'import os\n\nx = 1'
    var_57 = [var_0]
    var_58 = {}
    var_59 = -1
    var_60 = module_0.ParsedContent()
    var_61 = module_1.Config()
    var_62 = module_2.sorted_imports(var_60, var_61)
    assert var_62 == 'x = 1'
    var_63 = [var_0]
    var_64 = [var_5]
    var_65 = {var_5: var_64}
    var_66 = [var_6]
    var_67 = {var_6: var_66}
    var_68 = {var_3: var_65, var_4: var_67}
    var_69 = {var_2: var_68}
    var_70 = module_0.ParsedContent()
    var_71 = True
    var_72 = module_1.Config()
    var_73 = module_2.sorted_imports(var_70, var_72)
    assert var_73 == 'from sys import sys\n\nimport os\n\nx = 1'
    var_74 = [var_0]
    var_75 = {}
    var_76 = '*'
    var_77 = [var_76]
    var_78 = 'path'
    var_79 = [var_78]
    var_80 = {var_6: var_77, var_5: var_79}
    var_81 = {var_3: var_75, var_4: var_80}
    var_82 = {var_2: var_81}
    var_83 = module_0.ParsedContent()
    var_84 = True
    var_85 = module_1.Config()
    var_86 = module_2.sorted_imports(var_83, var_85)
    assert var_86 == 'from sys import *\nfrom os import path\n\nx = 1'
    var_87 = [var_0]
    var_88 = [var_5]
    var_89 = {var_5: var_88}
    var_90 = {}
    var_91 = {var_3: var_89, var_4: var_90}
    var_92 = {var_2: var_91}
    var_93 = module_0.ParsedContent()
    var_94 = 'stdlib'
    var_95 = 'Standard Library'
    var_96 = {var_94: var_95}
    var_97 = module_1.Config()
    var_98 = module_2.sorted_imports(var_93, var_97)
    assert var_98 == '# Standard Library\nimport os\n\nx = 1'
    var_99 = [var_0]
    var_100 = [var_5]
    var_101 = {var_5: var_100}
    var_102 = {}
    var_103 = {var_3: var_101, var_4: var_102}
    var_104 = {var_2: var_103}
    var_105 = module_0.ParsedContent()
    var_106 = 'End Standard Library'
    var_107 = {var_94: var_106}
    var_108 = module_1.Config()
    var_109 = module_2.sorted_imports(var_105, var_108)
    assert var_109 == 'import os\n\n# End Standard Library\n\nx = 1'
    var_110 = [var_0]
    var_111 = [var_21]
    var_112 = {var_21: var_111}
    var_113 = {}
    var_114 = {var_3: var_112, var_4: var_113}
    var_115 = [var_5]
    var_116 = {var_5: var_115}
    var_117 = {}
    var_118 = {var_3: var_116, var_4: var_117}
    var_119 = {var_20: var_114, var_2: var_118}
    var_120 = module_0.ParsedContent()
    var_121 = 2
    var_122 = module_1.Config()
    var_123 = module_2.sorted_imports(var_120, var_122)
    assert var_123 == 'from __future__ import __future__\n\n\nimport os\n\nx = 1'
    var_124 = [var_0]
    var_125 = [var_5]
    var_126 = {var_5: var_125}
    var_127 = {}
    var_128 = {var_3: var_126, var_4: var_127}
    var_129 = {var_2: var_128}
    var_130 = module_0.ParsedContent()
    var_131 = module_1.Config()
    var_132 = module_2.sorted_imports(var_130, var_131)
    assert var_132 == 'import os\n\n\nx = 1'
    var_133 = [var_0]
    var_134 = [var_5]
    var_135 = {var_5: var_134}
    var_136 = {}
    var_137 = {var_3: var_135, var_4: var_136}
    var_138 = {var_2: var_137}
    var_139 = module_0.ParsedContent()
    var_140 = module_1.Config()
    var_141 = module_2.sorted_imports(var_139, var_140)
    assert var_141 == '\n\nimport os\n\nx = 1'
    var_142 = [var_0]
    var_143 = [var_5]
    var_144 = {var_5: var_143}
    var_145 = {}
    var_146 = {var_3: var_144, var_4: var_145}
    var_147 = {var_2: var_146}
    var_148 = module_0.ParsedContent()
    var_149 = True
    var_150 = module_1.Config()
    var_151 = module_2.sorted_imports(var_148, var_150)
    assert var_151 == 'import os\n\nx = 1'
    var_152 = [var_0]
    var_153 = [var_5]
    var_154 = {var_5: var_153}
    var_155 = {}
    var_156 = {var_3: var_154, var_4: var_155}
    var_157 = {var_2: var_156}
    var_158 = module_0.ParsedContent()
    var_159 = module_2.sorted_imports(var_158, var_150)
    assert var_159 == 'import os\r\n\r\nx = 1'
    var_160 = 'y = 2'
    var_161 = [var_0, var_160]
    var_162 = [var_5]
    var_163 = {var_5: var_162}
    var_164 = {}
    var_165 = {var_3: var_163, var_4: var_164}
    var_166 = {var_2: var_165}
    var_167 = 'import sys'
    var_168 = [var_167]
    var_169 = {var_2: var_168}
    var_170 = {var_0: var_2}
    var_171 = module_0.ParsedContent()
    var_172 = module_1.Config()
    var_173 = module_2.sorted_imports(var_171, var_172)
    assert var_173 == 'import os\n\nx = 1\nimport sys\n\ny = 2'



# Parsed testcases at query #30
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = 'THIRDPARTY'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = 'sys'
    var_7 = set()
    var_8 = set()
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'collections'
    var_11 = 'defaultdict'
    var_12 = {var_11}
    var_13 = {var_10: var_12}
    var_14 = {var_3: var_9, var_4: var_13}
    var_15 = {var_2: var_14}
    var_16 = 0
    var_17 = 1
    var_18 = '\n'
    var_19 = module_0.ParsedContent()
    var_20 = module_1.Config()
    var_21 = module_2.sorted_imports(var_19, var_20)
    assert var_21 == "import os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')"
    var_22 = [var_0]
    var_23 = 'FIRSTPARTY'
    var_24 = set()
    var_25 = set()
    var_26 = {var_5: var_24, var_6: var_25}
    var_27 = {var_11}
    var_28 = {var_10: var_27}
    var_29 = {var_3: var_26, var_4: var_28}
    var_30 = 'my_module'
    var_31 = set()
    var_32 = {var_30: var_31}
    var_33 = {}
    var_34 = {var_3: var_32, var_4: var_33}
    var_35 = {var_2: var_29, var_23: var_34}
    var_36 = module_0.ParsedContent()
    var_37 = True
    var_38 = module_1.Config()
    var_39 = module_2.sorted_imports(var_36, var_38)
    assert var_39 == "import my_module\nimport os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')"
    var_40 = [var_0]
    var_41 = set()
    var_42 = {var_5: var_41}
    var_43 = {var_11}
    var_44 = {var_10: var_43}
    var_45 = {var_3: var_42, var_4: var_44}
    var_46 = {var_2: var_45}
    var_47 = module_0.ParsedContent()
    var_48 = True
    var_49 = module_1.Config()
    var_50 = module_2.sorted_imports(var_47, var_49)
    assert var_50 == "from collections import defaultdict\n\nimport os\n\nprint('hello')"
    var_51 = [var_0]
    var_52 = {}
    var_53 = 'module1'
    var_54 = 'module2'
    var_55 = '*'
    var_56 = {var_55}
    var_57 = 'function1'
    var_58 = {var_57}
    var_59 = {var_53: var_56, var_54: var_58}
    var_60 = {var_3: var_52, var_4: var_59}
    var_61 = {var_2: var_60}
    var_62 = module_0.ParsedContent()
    var_63 = True
    var_64 = module_1.Config()
    var_65 = module_2.sorted_imports(var_62, var_64)
    assert var_65 == "from module1 import *\nfrom module2 import function1\n\nprint('hello')"
    var_66 = [var_0]
    var_67 = set()
    var_68 = {var_5: var_67}
    var_69 = {}
    var_70 = {var_3: var_68, var_4: var_69}
    var_71 = {var_2: var_70}
    var_72 = module_0.ParsedContent()
    var_73 = 'thirdparty'
    var_74 = 'Third Party Imports'
    var_75 = {var_73: var_74}
    var_76 = module_1.Config()
    var_77 = module_2.sorted_imports(var_72, var_76)
    assert var_77 == "# Third Party Imports\nimport os\n\nprint('hello')"
    var_78 = [var_0]
    var_79 = 'FUTURE'
    var_80 = '__future__'
    var_81 = 'print_function'
    var_82 = {var_81}
    var_83 = {var_80: var_82}
    var_84 = {}
    var_85 = {var_3: var_83, var_4: var_84}
    var_86 = set()
    var_87 = {var_5: var_86}
    var_88 = {}
    var_89 = {var_3: var_87, var_4: var_88}
    var_90 = {var_79: var_85, var_2: var_89}
    var_91 = module_0.ParsedContent()
    var_92 = 2
    var_93 = module_1.Config()
    var_94 = module_2.sorted_imports(var_91, var_93)
    assert var_94 == "from __future__ import print_function\n\n\nimport os\n\nprint('hello')"
    var_95 = [var_0]
    var_96 = set()
    var_97 = {var_5: var_96}
    var_98 = {}
    var_99 = {var_3: var_97, var_4: var_98}
    var_100 = {var_2: var_99}
    var_101 = module_0.ParsedContent()
    var_102 = module_1.Config()
    var_103 = module_2.sorted_imports(var_101, var_102)
    assert var_103 == "import os\n\n\nprint('hello')"



# Parsed testcases at query #31
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = 0
    var_5 = '\n'
    var_6 = {}
    var_7 = {}
    var_8 = module_0.ParsedContent()
    var_9 = "print('hello')"
    var_10 = [var_9]
    var_11 = {}
    var_12 = -1
    var_13 = 1
    var_14 = {}
    var_15 = {}
    var_16 = module_0.ParsedContent()
    var_17 = [var_0]
    var_18 = 'THIRDPARTY'
    var_19 = 'straight'
    var_20 = 'from'
    var_21 = 'os'
    var_22 = 'sys'
    var_23 = [var_21]
    var_24 = [var_22]
    var_25 = {var_21: var_23, var_22: var_24}
    var_26 = 'collections'
    var_27 = 'OrderedDict'
    var_28 = [var_27]
    var_29 = {var_26: var_28}
    var_30 = {var_19: var_25, var_20: var_29}
    var_31 = {var_18: var_30}
    var_32 = {}
    var_33 = {}
    var_34 = module_0.ParsedContent()
    var_35 = '\nimport os\nimport sys\n\nfrom collections import OrderedDict\n'
    var_36 = [var_0]
    var_37 = [var_21]
    var_38 = [var_22]
    var_39 = {var_21: var_37, var_22: var_38}
    var_40 = [var_27]
    var_41 = {var_26: var_40}
    var_42 = {var_19: var_39, var_20: var_41}
    var_43 = {var_18: var_42}
    var_44 = {}
    var_45 = {}
    var_46 = module_0.ParsedContent()
    var_47 = 2
    var_48 = True
    var_49 = module_1.Config()
    var_50 = module_2.sorted_imports(var_46, var_49)
    var_51 = '\nfrom collections import OrderedDict\n\nimport os\nimport sys\n'
    var_52 = [var_0]
    var_53 = 'FUTURE'
    var_54 = '__future__'
    var_55 = 'print_function'
    var_56 = [var_55]
    var_57 = {var_54: var_56}
    var_58 = {var_19: var_57}
    var_59 = [var_21]
    var_60 = [var_22]
    var_61 = {var_21: var_59, var_22: var_60}
    var_62 = [var_27]
    var_63 = {var_26: var_62}
    var_64 = {var_19: var_61, var_20: var_63}
    var_65 = {var_53: var_58, var_18: var_64}
    var_66 = {}
    var_67 = {}
    var_68 = module_0.ParsedContent()
    var_69 = 'LOCALFOLDER'
    var_70 = [var_69]
    var_71 = module_1.Config()
    var_72 = module_2.sorted_imports(var_68, var_71)
    var_73 = '\nfrom __future__ import print_function\n\nimport os\nimport sys\n\nfrom collections import OrderedDict\n'
    var_74 = [var_0]
    var_75 = [var_55]
    var_76 = {var_54: var_75}
    var_77 = {var_19: var_76}
    var_78 = [var_21]
    var_79 = [var_22]
    var_80 = {var_21: var_78, var_22: var_79}
    var_81 = [var_27]
    var_82 = {var_26: var_81}
    var_83 = {var_19: var_80, var_20: var_82}
    var_84 = {var_53: var_77, var_18: var_83}
    var_85 = {}
    var_86 = {}
    var_87 = module_0.ParsedContent()
    var_88 = True
    var_89 = module_1.Config()
    var_90 = module_2.sorted_imports(var_87, var_89)
    var_91 = '\nfrom __future__ import print_function\nimport os\nimport sys\nfrom collections import OrderedDict\n'
    var_92 = [var_0]
    var_93 = [var_21]
    var_94 = [var_22]
    var_95 = {var_21: var_93, var_22: var_94}
    var_96 = [var_27]
    var_97 = {var_26: var_96}
    var_98 = {var_19: var_95, var_20: var_97}
    var_99 = {var_18: var_98}
    var_100 = {}
    var_101 = {}
    var_102 = module_0.ParsedContent()
    var_103 = 'thirdparty'
    var_104 = 'Third Party Imports'
    var_105 = {var_103: var_104}
    var_106 = True
    var_107 = module_1.Config()
    var_108 = module_2.sorted_imports(var_102, var_107)
    var_109 = '\n# Third Party Imports\nimport os\nimport sys\n\nfrom collections import OrderedDict\n'
    var_110 = '# Placeholder'
    var_111 = [var_110, var_9]
    var_112 = [var_21]
    var_113 = [var_22]
    var_114 = {var_21: var_112, var_22: var_113}
    var_115 = [var_27]
    var_116 = {var_26: var_115}
    var_117 = {var_19: var_114, var_20: var_116}
    var_118 = {var_18: var_117}
    var_119 = 'import os'
    var_120 = 'import sys'
    var_121 = [var_119, var_120]
    var_122 = {var_18: var_121}
    var_123 = {var_110: var_18}
    var_124 = module_0.ParsedContent()
    var_125 = module_2.sorted_imports(var_124, var_107)
    var_126 = "# Placeholder\nimport os\nimport sys\n\nprint('hello')\n"
    var_127 = [var_0]
    var_128 = [var_21]
    var_129 = [var_22]
    var_130 = {var_21: var_128, var_22: var_129}
    var_131 = [var_27]
    var_132 = {var_26: var_131}
    var_133 = {var_19: var_130, var_20: var_132}
    var_134 = {var_18: var_133}
    var_135 = {}
    var_136 = {}
    var_137 = module_0.ParsedContent()
    var_138 = module_2.sorted_imports(var_137, var_107)
    var_139 = '\nfrom os\nfrom sys\n\nimport collections import OrderedDict\n'
    var_140 = [var_9]
    var_141 = [var_21]
    var_142 = [var_22]
    var_143 = {var_21: var_141, var_22: var_142}
    var_144 = [var_27]
    var_145 = {var_26: var_144}
    var_146 = {var_19: var_143, var_20: var_145}
    var_147 = {var_18: var_146}
    var_148 = {}
    var_149 = {}
    var_150 = module_0.ParsedContent()
    var_151 = module_1.Config()
    var_152 = module_2.sorted_imports(var_150, var_151)
    var_153 = "\n\nimport os\nimport sys\n\nfrom collections import OrderedDict\n\n\nprint('hello')\n"



# Parsed testcases at query #32
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = "print('hello')"
    var_2 = [var_0, var_1]
    var_3 = 'THIRDPARTY'
    var_4 = 'straight'
    var_5 = 'from'
    var_6 = 'os'
    var_7 = 'sys'
    var_8 = []
    var_9 = []
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'collections'
    var_12 = 'defaultdict'
    var_13 = [var_12]
    var_14 = {var_11: var_13}
    var_15 = {var_4: var_10, var_5: var_14}
    var_16 = {var_3: var_15}
    var_17 = 0
    var_18 = 2
    var_19 = '\n'
    var_20 = {}
    var_21 = {}
    var_22 = module_0.ParsedContent()
    var_23 = module_1.Config()
    var_24 = module_2.sorted_imports(var_22, var_23)
    assert var_24 == "import os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')"
    var_25 = [var_0, var_1]
    var_26 = 'FIRSTPARTY'
    var_27 = []
    var_28 = []
    var_29 = {var_6: var_27, var_7: var_28}
    var_30 = [var_12]
    var_31 = {var_11: var_30}
    var_32 = {var_4: var_29, var_5: var_31}
    var_33 = 'my_module'
    var_34 = []
    var_35 = {var_33: var_34}
    var_36 = {}
    var_37 = {var_4: var_35, var_5: var_36}
    var_38 = {var_3: var_32, var_26: var_37}
    var_39 = {}
    var_40 = {}
    var_41 = module_0.ParsedContent()
    var_42 = True
    var_43 = module_1.Config()
    var_44 = module_2.sorted_imports(var_41, var_43)
    assert var_44 == "import os\nimport sys\n\nfrom collections import defaultdict\n\nimport my_module\n\nprint('hello')"
    var_45 = module_1.Config()
    var_46 = module_2.sorted_imports(var_22, var_45)
    assert var_46 == "from collections import defaultdict\n\nimport os\nimport sys\n\nprint('hello')"
    var_47 = [var_0, var_1]
    var_48 = {}
    var_49 = 'module1'
    var_50 = 'module2'
    var_51 = '*'
    var_52 = [var_51]
    var_53 = 'function'
    var_54 = [var_53]
    var_55 = {var_49: var_52, var_50: var_54}
    var_56 = {var_4: var_48, var_5: var_55}
    var_57 = {var_3: var_56}
    var_58 = {}
    var_59 = {}
    var_60 = module_0.ParsedContent()
    var_61 = module_1.Config()
    var_62 = module_2.sorted_imports(var_60, var_61)
    assert var_62 == "from module1 import *\nfrom module2 import function\n\nprint('hello')"
    var_63 = 'thirdparty'
    var_64 = 'Third Party Imports'
    var_65 = {var_63: var_64}
    var_66 = module_1.Config()
    var_67 = module_2.sorted_imports(var_22, var_66)
    assert var_67 == "# Third Party Imports\nimport os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')"
    var_68 = [var_1]
    var_69 = {}
    var_70 = -1
    var_71 = {}
    var_72 = {}
    var_73 = module_0.ParsedContent()
    var_74 = module_2.sorted_imports(var_73, var_23)
    assert var_74 == "print('hello')"



# Parsed testcases at query #33
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = -1
    var_3 = '\n'
    var_4 = []
    var_5 = {}
    var_6 = {}
    var_7 = {}
    var_8 = 1
    var_9 = module_0.ParsedContent()
    var_10 = [var_0]
    var_11 = 0
    var_12 = 'THIRDPARTY'
    var_13 = [var_12]
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = 'os'
    var_17 = 'sys'
    var_18 = 'os.path'
    var_19 = [var_18]
    var_20 = 'sys.argv'
    var_21 = [var_20]
    var_22 = {var_16: var_19, var_17: var_21}
    var_23 = 'collections'
    var_24 = 'defaultdict'
    var_25 = [var_24]
    var_26 = {var_23: var_25}
    var_27 = {var_14: var_22, var_15: var_26}
    var_28 = {var_12: var_27}
    var_29 = {}
    var_30 = {}
    var_31 = module_0.ParsedContent()
    var_32 = "import os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')\n"
    var_33 = [var_0]
    var_34 = [var_12]
    var_35 = [var_18]
    var_36 = [var_20]
    var_37 = {var_16: var_35, var_17: var_36}
    var_38 = [var_24]
    var_39 = {var_23: var_38}
    var_40 = {var_14: var_37, var_15: var_39}
    var_41 = {var_12: var_40}
    var_42 = {}
    var_43 = {}
    var_44 = module_0.ParsedContent()
    var_45 = 2
    var_46 = True
    var_47 = True
    var_48 = module_1.Config()
    var_49 = module_2.sorted_imports(var_44, var_48)
    var_50 = "from collections import defaultdict\n\nimport os\nimport sys\n\n\nprint('hello')\n"
    var_51 = [var_0]
    var_52 = [var_12]
    var_53 = [var_18]
    var_54 = [var_20]
    var_55 = {var_16: var_53, var_17: var_54}
    var_56 = [var_24]
    var_57 = {var_23: var_56}
    var_58 = {var_14: var_55, var_15: var_57}
    var_59 = {var_12: var_58}
    var_60 = {}
    var_61 = {}
    var_62 = module_0.ParsedContent()
    var_63 = 'thirdparty'
    var_64 = 'Third Party Imports'
    var_65 = {var_63: var_64}
    var_66 = True
    var_67 = module_1.Config()
    var_68 = module_2.sorted_imports(var_62, var_67)
    var_69 = "# Third Party Imports\nimport os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')\n"
    var_70 = [var_0]
    var_71 = [var_12]
    var_72 = [var_18]
    var_73 = [var_20]
    var_74 = {var_16: var_72, var_17: var_73}
    var_75 = [var_24]
    var_76 = {var_23: var_75}
    var_77 = {var_14: var_74, var_15: var_76}
    var_78 = {var_12: var_77}
    var_79 = {}
    var_80 = {}
    var_81 = module_0.ParsedContent()
    var_82 = 'LOCALFOLDER'
    var_83 = [var_82]
    var_84 = module_1.Config()
    var_85 = module_2.sorted_imports(var_81, var_84)
    var_86 = "import os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')\n"
    var_87 = [var_0]
    var_88 = [var_12]
    var_89 = [var_18]
    var_90 = [var_20]
    var_91 = {var_16: var_89, var_17: var_90}
    var_92 = [var_24]
    var_93 = {var_23: var_92}
    var_94 = {var_14: var_91, var_15: var_93}
    var_95 = {var_12: var_94}
    var_96 = {}
    var_97 = {}
    var_98 = module_0.ParsedContent()
    var_99 = 'from collections import defaultdict'
    var_100 = [var_99]
    var_101 = module_1.Config()
    var_102 = module_2.sorted_imports(var_98, var_101)
    var_103 = "import os\nimport sys\n\nprint('hello')\n"
    var_104 = [var_0]
    var_105 = 'FUTURE'
    var_106 = [var_12, var_105]
    var_107 = [var_18]
    var_108 = [var_20]
    var_109 = {var_16: var_107, var_17: var_108}
    var_110 = [var_24]
    var_111 = {var_23: var_110}
    var_112 = {var_14: var_109, var_15: var_111}
    var_113 = '__future__'
    var_114 = 'annotations'
    var_115 = [var_114]
    var_116 = {var_113: var_115}
    var_117 = {}
    var_118 = {var_14: var_116, var_15: var_117}
    var_119 = {var_12: var_112, var_105: var_118}
    var_120 = {}
    var_121 = {}
    var_122 = module_0.ParsedContent()
    var_123 = True
    var_124 = module_1.Config()
    var_125 = module_2.sorted_imports(var_122, var_124)
    var_126 = "from __future__ import annotations\n\nimport os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')\n"
    var_127 = [var_0]
    var_128 = [var_12]
    var_129 = [var_18]
    var_130 = [var_20]
    var_131 = {var_16: var_129, var_17: var_130}
    var_132 = 'typing'
    var_133 = [var_24]
    var_134 = '*'
    var_135 = [var_134]
    var_136 = {var_23: var_133, var_132: var_135}
    var_137 = {var_14: var_131, var_15: var_136}
    var_138 = {var_12: var_137}
    var_139 = {}
    var_140 = {}
    var_141 = module_0.ParsedContent()
    var_142 = True
    var_143 = module_1.Config()
    var_144 = module_2.sorted_imports(var_141, var_143)
    var_145 = "import os\nimport sys\n\nfrom typing import *\nfrom collections import defaultdict\n\nprint('hello')\n"



# Parsed testcases at query #34
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 'THIRDPARTY'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = 'sys'
    var_7 = []
    var_8 = []
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'collections'
    var_11 = 'defaultdict'
    var_12 = [var_11]
    var_13 = {var_10: var_12}
    var_14 = {var_3: var_9, var_4: var_13}
    var_15 = {var_2: var_14}
    var_16 = 0
    var_17 = 1
    var_18 = '\n'
    var_19 = module_0.ParsedContent()
    var_20 = module_1.Config()
    var_21 = module_2.sorted_imports(var_19, var_20)
    assert var_21 == 'from collections import defaultdict\nimport os\nimport sys\n'
    var_22 = [var_0]
    var_23 = 'FUTURE'
    var_24 = '__future__'
    var_25 = []
    var_26 = {var_24: var_25}
    var_27 = {}
    var_28 = {var_3: var_26, var_4: var_27}
    var_29 = []
    var_30 = []
    var_31 = {var_5: var_29, var_6: var_30}
    var_32 = [var_11]
    var_33 = {var_10: var_32}
    var_34 = {var_3: var_31, var_4: var_33}
    var_35 = {var_23: var_28, var_2: var_34}
    var_36 = module_0.ParsedContent()
    var_37 = True
    var_38 = module_1.Config()
    var_39 = module_2.sorted_imports(var_36, var_38)
    assert var_39 == 'from __future__ import absolute_import\nfrom collections import defaultdict\nimport os\nimport sys\n'
    var_40 = [var_0]
    var_41 = []
    var_42 = []
    var_43 = {var_5: var_41, var_6: var_42}
    var_44 = [var_11]
    var_45 = {var_10: var_44}
    var_46 = {var_3: var_43, var_4: var_45}
    var_47 = {var_2: var_46}
    var_48 = module_0.ParsedContent()
    var_49 = True
    var_50 = module_1.Config()
    var_51 = module_2.sorted_imports(var_48, var_50)
    assert var_51 == 'from collections import defaultdict\nimport os\nimport sys\n'
    var_52 = [var_0]
    var_53 = {}
    var_54 = '*'
    var_55 = [var_54]
    var_56 = 'path'
    var_57 = [var_56]
    var_58 = {var_10: var_55, var_5: var_57}
    var_59 = {var_3: var_53, var_4: var_58}
    var_60 = {var_2: var_59}
    var_61 = module_0.ParsedContent()
    var_62 = True
    var_63 = module_1.Config()
    var_64 = module_2.sorted_imports(var_61, var_63)
    assert var_64 == 'from collections import *\nfrom os import path\n'
    var_65 = [var_0]
    var_66 = []
    var_67 = []
    var_68 = {var_5: var_66, var_6: var_67}
    var_69 = [var_11]
    var_70 = {var_10: var_69}
    var_71 = {var_3: var_68, var_4: var_70}
    var_72 = {var_2: var_71}
    var_73 = module_0.ParsedContent()
    var_74 = True
    var_75 = module_1.Config()
    var_76 = module_2.sorted_imports(var_73, var_75)
    assert var_76 == 'from collections import defaultdict\n\nimport os\nimport sys\n'
    var_77 = [var_0]
    var_78 = []
    var_79 = {var_24: var_78}
    var_80 = {}
    var_81 = {var_3: var_79, var_4: var_80}
    var_82 = []
    var_83 = []
    var_84 = {var_5: var_82, var_6: var_83}
    var_85 = [var_11]
    var_86 = {var_10: var_85}
    var_87 = {var_3: var_84, var_4: var_86}
    var_88 = {var_23: var_81, var_2: var_87}
    var_89 = module_0.ParsedContent()
    var_90 = 2
    var_91 = module_1.Config()
    var_92 = module_2.sorted_imports(var_89, var_91)
    assert var_92 == 'from __future__ import absolute_import\n\n\nfrom collections import defaultdict\nimport os\nimport sys\n'
    var_93 = [var_0]
    var_94 = []
    var_95 = []
    var_96 = {var_5: var_94, var_6: var_95}
    var_97 = [var_11]
    var_98 = {var_10: var_97}
    var_99 = {var_3: var_96, var_4: var_98}
    var_100 = {var_2: var_99}
    var_101 = module_0.ParsedContent()
    var_102 = 'thirdparty'
    var_103 = 'Third Party Imports'
    var_104 = {var_102: var_103}
    var_105 = module_1.Config()
    var_106 = module_2.sorted_imports(var_101, var_105)
    assert var_106 == '# Third Party Imports\nfrom collections import defaultdict\nimport os\nimport sys\n'
    var_107 = [var_0]
    var_108 = []
    var_109 = []
    var_110 = {var_5: var_108, var_6: var_109}
    var_111 = [var_11]
    var_112 = {var_10: var_111}
    var_113 = {var_3: var_110, var_4: var_112}
    var_114 = {var_2: var_113}
    var_115 = module_0.ParsedContent()
    var_116 = 'End of Third Party Imports'
    var_117 = {var_102: var_116}
    var_118 = module_1.Config()
    var_119 = module_2.sorted_imports(var_115, var_118)
    assert var_119 == 'from collections import defaultdict\nimport os\nimport sys\n\n# End of Third Party Imports\n'
    var_120 = [var_0]
    var_121 = []
    var_122 = []
    var_123 = {var_5: var_121, var_6: var_122}
    var_124 = [var_11]
    var_125 = {var_10: var_124}
    var_126 = {var_3: var_123, var_4: var_125}
    var_127 = {var_2: var_126}
    var_128 = module_0.ParsedContent()
    var_129 = module_1.Config()
    var_130 = module_2.sorted_imports(var_128, var_129)
    assert var_130 == '\n\nfrom collections import defaultdict\nimport os\nimport sys\n'
    var_131 = [var_0]
    var_132 = []
    var_133 = []
    var_134 = {var_5: var_132, var_6: var_133}
    var_135 = [var_11]
    var_136 = {var_10: var_135}
    var_137 = {var_3: var_134, var_4: var_136}
    var_138 = {var_2: var_137}
    var_139 = module_0.ParsedContent()
    var_140 = module_1.Config()
    var_141 = module_2.sorted_imports(var_139, var_140)
    assert var_141 == 'from collections import defaultdict\nimport os\nimport sys\n\n\n'



# Parsed testcases at query #35
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = 'x = 1'
    var_2 = [var_0, var_1]
    var_3 = 'THIRDPARTY'
    var_4 = 'straight'
    var_5 = 'from'
    var_6 = 'os'
    var_7 = 'sys'
    var_8 = []
    var_9 = []
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'collections'
    var_12 = 'defaultdict'
    var_13 = [var_12]
    var_14 = {var_11: var_13}
    var_15 = {var_4: var_10, var_5: var_14}
    var_16 = {var_3: var_15}
    var_17 = 0
    var_18 = 2
    var_19 = '\n'
    var_20 = module_0.ParsedContent()
    var_21 = module_1.Config()
    var_22 = module_2.sorted_imports(var_20, var_21)
    assert var_22 == 'import os\nimport sys\n\nfrom collections import defaultdict\n\nx = 1\n'
    var_23 = [var_1]
    var_24 = {}
    var_25 = -1
    var_26 = 1
    var_27 = module_0.ParsedContent()
    var_28 = module_2.sorted_imports(var_27, var_21)
    assert var_28 == 'x = 1\n'
    var_29 = 'thirdparty'
    var_30 = 'Third Party Imports'
    var_31 = {var_29: var_30}
    var_32 = module_1.Config()
    var_33 = [var_0, var_1]
    var_34 = []
    var_35 = []
    var_36 = {var_6: var_34, var_7: var_35}
    var_37 = {}
    var_38 = {var_4: var_36, var_5: var_37}
    var_39 = {var_3: var_38}
    var_40 = module_0.ParsedContent()
    var_41 = module_2.sorted_imports(var_40, var_32)
    assert var_41 == '# Third Party Imports\nimport os\nimport sys\n\nx = 1\n'
    var_42 = 'LOCALFOLDER'
    var_43 = [var_42]
    var_44 = module_1.Config()
    var_45 = [var_0, var_1]
    var_46 = []
    var_47 = {var_6: var_46}
    var_48 = {}
    var_49 = {var_4: var_47, var_5: var_48}
    var_50 = []
    var_51 = {var_7: var_50}
    var_52 = {}
    var_53 = {var_4: var_51, var_5: var_52}
    var_54 = {var_3: var_49, var_42: var_53}
    var_55 = module_0.ParsedContent()
    var_56 = module_2.sorted_imports(var_55, var_44)
    assert var_56 == 'import os\n\nimport sys\n\nx = 1\n'
    var_57 = [var_6]
    var_58 = module_1.Config()
    var_59 = [var_0, var_1]
    var_60 = []
    var_61 = []
    var_62 = {var_6: var_60, var_7: var_61}
    var_63 = {}
    var_64 = {var_4: var_62, var_5: var_63}
    var_65 = {var_3: var_64}
    var_66 = module_0.ParsedContent()
    var_67 = module_2.sorted_imports(var_66, var_58)
    assert var_67 == 'import sys\n\nx = 1\n'
    var_68 = module_1.Config()
    var_69 = [var_0, var_1]
    var_70 = 'FIRSTPARTY'
    var_71 = []
    var_72 = {var_6: var_71}
    var_73 = {}
    var_74 = {var_4: var_72, var_5: var_73}
    var_75 = []
    var_76 = {var_7: var_75}
    var_77 = {}
    var_78 = {var_4: var_76, var_5: var_77}
    var_79 = {var_3: var_74, var_70: var_78}
    var_80 = module_0.ParsedContent()
    var_81 = module_2.sorted_imports(var_80, var_68)
    assert var_81 == 'import os\n\n\n\nimport sys\n\nx = 1\n'
    var_82 = module_1.Config()
    var_83 = [var_0, var_1]
    var_84 = []
    var_85 = {var_6: var_84}
    var_86 = {}
    var_87 = {var_4: var_85, var_5: var_86}
    var_88 = {var_3: var_87}
    var_89 = module_0.ParsedContent()
    var_90 = module_2.sorted_imports(var_89, var_82)
    assert var_90 == 'import os\n\n\nx = 1\n'
    var_91 = True
    var_92 = module_1.Config()
    var_93 = [var_0, var_1]
    var_94 = []
    var_95 = []
    var_96 = {var_6: var_94, var_7: var_95}
    var_97 = {}
    var_98 = {var_4: var_96, var_5: var_97}
    var_99 = 'json'
    var_100 = []
    var_101 = {var_99: var_100}
    var_102 = {}
    var_103 = {var_4: var_101, var_5: var_102}
    var_104 = {var_3: var_98, var_70: var_103}
    var_105 = module_0.ParsedContent()
    var_106 = module_2.sorted_imports(var_105, var_92)
    assert var_106 == 'import json\nimport os\nimport sys\n\nx = 1\n'
    var_107 = True
    var_108 = module_1.Config()
    var_109 = [var_0, var_1]
    var_110 = []
    var_111 = {var_6: var_110}
    var_112 = 'path'
    var_113 = [var_112]
    var_114 = {var_7: var_113}
    var_115 = {var_4: var_111, var_5: var_114}
    var_116 = {var_3: var_115}
    var_117 = module_0.ParsedContent()
    var_118 = module_2.sorted_imports(var_117, var_108)
    assert var_118 == 'from sys import path\n\nimport os\n\nx = 1\n'
    var_119 = True
    var_120 = module_1.Config()
    var_121 = [var_0, var_1]
    var_122 = {}
    var_123 = '*'
    var_124 = [var_123]
    var_125 = [var_112]
    var_126 = {var_6: var_124, var_7: var_125}
    var_127 = {var_4: var_122, var_5: var_126}
    var_128 = {var_3: var_127}
    var_129 = module_0.ParsedContent()
    var_130 = module_2.sorted_imports(var_129, var_120)
    assert var_130 == 'from os import *\nfrom sys import path\n\nx = 1\n'



# Parsed testcases at query #36
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = -1
    var_3 = '\n'
    var_4 = 1
    var_5 = module_0.ParsedContent()
    var_6 = [var_0]
    var_7 = 0
    var_8 = 'THIRDPARTY'
    var_9 = 'straight'
    var_10 = 'from'
    var_11 = 'os'
    var_12 = 'sys'
    var_13 = []
    var_14 = []
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = 'collections'
    var_17 = 'defaultdict'
    var_18 = [var_17]
    var_19 = {var_16: var_18}
    var_20 = {var_9: var_15, var_10: var_19}
    var_21 = {var_8: var_20}
    var_22 = 2
    var_23 = module_0.ParsedContent()
    var_24 = [var_0]
    var_25 = []
    var_26 = []
    var_27 = {var_11: var_25, var_12: var_26}
    var_28 = [var_17]
    var_29 = {var_16: var_28}
    var_30 = {var_9: var_27, var_10: var_29}
    var_31 = {var_8: var_30}
    var_32 = module_0.ParsedContent()
    var_33 = True
    var_34 = module_1.Config()
    var_35 = module_2.sorted_imports(var_32, var_34)
    assert var_35 == "import os\nimport sys\n\n\nfrom collections import defaultdict\n\n\nprint('hello')\n"
    var_36 = [var_0]
    var_37 = []
    var_38 = []
    var_39 = {var_11: var_37, var_12: var_38}
    var_40 = [var_17]
    var_41 = {var_16: var_40}
    var_42 = {var_9: var_39, var_10: var_41}
    var_43 = {var_8: var_42}
    var_44 = module_0.ParsedContent()
    var_45 = [var_11]
    var_46 = module_1.Config()
    var_47 = module_2.sorted_imports(var_44, var_46)
    assert var_47 == "from collections import defaultdict\n\nimport sys\n\nprint('hello')\n"
    var_48 = [var_0]
    var_49 = '*'
    var_50 = [var_49]
    var_51 = [var_17]
    var_52 = {var_11: var_50, var_16: var_51}
    var_53 = {var_10: var_52}
    var_54 = {var_8: var_53}
    var_55 = module_0.ParsedContent()
    var_56 = True
    var_57 = module_1.Config()
    var_58 = module_2.sorted_imports(var_55, var_57)
    assert var_58 == "from os import *\nfrom collections import defaultdict\n\nprint('hello')\n"
    var_59 = [var_0]
    var_60 = 'FUTURE'
    var_61 = '__future__'
    var_62 = 'print_function'
    var_63 = [var_62]
    var_64 = {var_61: var_63}
    var_65 = {var_9: var_64}
    var_66 = []
    var_67 = []
    var_68 = {var_11: var_66, var_12: var_67}
    var_69 = [var_17]
    var_70 = {var_16: var_69}
    var_71 = {var_9: var_68, var_10: var_70}
    var_72 = {var_60: var_65, var_8: var_71}
    var_73 = module_0.ParsedContent()
    var_74 = True
    var_75 = module_1.Config()
    var_76 = module_2.sorted_imports(var_73, var_75)
    assert var_76 == "from __future__ import print_function\n\nfrom collections import defaultdict\nimport os\nimport sys\n\nprint('hello')\n"
    var_77 = [var_0]
    var_78 = []
    var_79 = []
    var_80 = {var_11: var_78, var_12: var_79}
    var_81 = [var_17]
    var_82 = {var_16: var_81}
    var_83 = {var_9: var_80, var_10: var_82}
    var_84 = {var_8: var_83}
    var_85 = module_0.ParsedContent()
    var_86 = 'thirdparty'
    var_87 = 'Third Party Imports'
    var_88 = {var_86: var_87}
    var_89 = module_1.Config()
    var_90 = module_2.sorted_imports(var_85, var_89)
    assert var_90 == "# Third Party Imports\nfrom collections import defaultdict\n\nimport os\nimport sys\n\nprint('hello')\n"
    var_91 = [var_0]
    var_92 = []
    var_93 = {var_11: var_92}
    var_94 = {var_9: var_93}
    var_95 = {var_8: var_94}
    var_96 = module_0.ParsedContent()
    var_97 = module_1.Config()
    var_98 = module_2.sorted_imports(var_96, var_97)
    assert var_98 == "import os\n\n\nprint('hello')\n"
    var_99 = [var_0]
    var_100 = []
    var_101 = {var_11: var_100}
    var_102 = {var_9: var_101}
    var_103 = {var_8: var_102}
    var_104 = module_0.ParsedContent()
    var_105 = lambda code, ext, cfg: code.upper()
    var_106 = module_1.Config()
    var_107 = module_2.sorted_imports(var_104, var_106)
    assert var_107 == "IMPORT OS\n\nPRINT('HELLO')\n"
    var_108 = '# Placeholder'
    var_109 = [var_108, var_0]
    var_110 = []
    var_111 = {var_11: var_110}
    var_112 = {var_9: var_111}
    var_113 = {var_8: var_112}
    var_114 = 'import sys'
    var_115 = [var_114]
    var_116 = {var_8: var_115}
    var_117 = {var_108: var_8}
    var_118 = module_0.ParsedContent()
    var_119 = module_2.sorted_imports(var_118, var_106)
    assert var_119 == "import os\n\n# Placeholder\nimport sys\n\nprint('hello')\n"



# Parsed testcases at query #37
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = -1
    var_3 = '\n'
    var_4 = []
    var_5 = {}
    var_6 = {}
    var_7 = {}
    var_8 = 1
    var_9 = module_0.ParsedContent()
    var_10 = module_1.Config()
    var_11 = module_2.sorted_imports(var_9, var_10)
    assert var_11 == "print('hello')\n"
    var_12 = [var_0]
    var_13 = 0
    var_14 = 'THIRDPARTY'
    var_15 = [var_14]
    var_16 = 'straight'
    var_17 = 'from'
    var_18 = 'os'
    var_19 = 'sys'
    var_20 = [var_18]
    var_21 = [var_19]
    var_22 = {var_18: var_20, var_19: var_21}
    var_23 = 'collections'
    var_24 = 'OrderedDict'
    var_25 = [var_24]
    var_26 = {var_23: var_25}
    var_27 = {var_16: var_22, var_17: var_26}
    var_28 = {var_14: var_27}
    var_29 = {}
    var_30 = {}
    var_31 = 2
    var_32 = module_0.ParsedContent()
    var_33 = module_1.Config()
    var_34 = module_2.sorted_imports(var_32, var_33)
    var_35 = "import os\nimport sys\nfrom collections import OrderedDict\n\nprint('hello')\n"
    var_36 = [var_0]
    var_37 = [var_14]
    var_38 = [var_18]
    var_39 = [var_19]
    var_40 = {var_18: var_38, var_19: var_39}
    var_41 = [var_24]
    var_42 = {var_23: var_41}
    var_43 = {var_16: var_40, var_17: var_42}
    var_44 = {var_14: var_43}
    var_45 = {}
    var_46 = {}
    var_47 = module_0.ParsedContent()
    var_48 = 'thirdparty'
    var_49 = 'Third Party Imports'
    var_50 = {var_48: var_49}
    var_51 = module_1.Config()
    var_52 = module_2.sorted_imports(var_47, var_51)
    var_53 = "# Third Party Imports\nimport os\nimport sys\n\nfrom collections import OrderedDict\n\n\nprint('hello')\n"
    var_54 = [var_0]
    var_55 = [var_14]
    var_56 = [var_18]
    var_57 = [var_19]
    var_58 = {var_18: var_56, var_19: var_57}
    var_59 = [var_24]
    var_60 = {var_23: var_59}
    var_61 = {var_16: var_58, var_17: var_60}
    var_62 = {var_14: var_61}
    var_63 = {}
    var_64 = {}
    var_65 = module_0.ParsedContent()
    var_66 = [var_18]
    var_67 = module_1.Config()
    var_68 = module_2.sorted_imports(var_65, var_67)
    var_69 = "import sys\nfrom collections import OrderedDict\n\nprint('hello')\n"
    var_70 = [var_0]
    var_71 = 'FUTURE'
    var_72 = [var_71, var_14]
    var_73 = '__future__'
    var_74 = [var_73]
    var_75 = {var_73: var_74}
    var_76 = {}
    var_77 = {var_16: var_75, var_17: var_76}
    var_78 = [var_18]
    var_79 = [var_19]
    var_80 = {var_18: var_78, var_19: var_79}
    var_81 = [var_24]
    var_82 = {var_23: var_81}
    var_83 = {var_16: var_80, var_17: var_82}
    var_84 = {var_71: var_77, var_14: var_83}
    var_85 = {}
    var_86 = {}
    var_87 = module_0.ParsedContent()
    var_88 = True
    var_89 = module_1.Config()
    var_90 = module_2.sorted_imports(var_87, var_89)
    var_91 = "from __future__ import __future__\nimport os\nimport sys\nfrom collections import OrderedDict\n\nprint('hello')\n"



# Parsed testcases at query #38
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = 0
    var_5 = '\n'
    var_6 = {}
    var_7 = {}
    var_8 = module_0.ParsedContent()
    var_9 = "print('hello')"
    var_10 = [var_9]
    var_11 = {}
    var_12 = -1
    var_13 = 1
    var_14 = {}
    var_15 = {}
    var_16 = module_0.ParsedContent()
    var_17 = [var_0]
    var_18 = 'THIRDPARTY'
    var_19 = 'straight'
    var_20 = 'from'
    var_21 = 'os'
    var_22 = 'sys'
    var_23 = [var_21]
    var_24 = [var_22]
    var_25 = {var_21: var_23, var_22: var_24}
    var_26 = 'collections'
    var_27 = 'defaultdict'
    var_28 = 'OrderedDict'
    var_29 = [var_27, var_28]
    var_30 = {var_26: var_29}
    var_31 = {var_19: var_25, var_20: var_30}
    var_32 = {var_18: var_31}
    var_33 = {}
    var_34 = {}
    var_35 = module_0.ParsedContent()
    var_36 = 'import os\nimport sys\n\nfrom collections import defaultdict, OrderedDict\n'
    var_37 = [var_0]
    var_38 = [var_21]
    var_39 = [var_22]
    var_40 = {var_21: var_38, var_22: var_39}
    var_41 = [var_27, var_28]
    var_42 = {var_26: var_41}
    var_43 = {var_19: var_40, var_20: var_42}
    var_44 = {var_18: var_43}
    var_45 = {}
    var_46 = {}
    var_47 = module_0.ParsedContent()
    var_48 = 2
    var_49 = True
    var_50 = module_1.Config()
    var_51 = module_2.sorted_imports(var_47, var_50)
    var_52 = 'from collections import defaultdict, OrderedDict\n\nimport os\nimport sys\n'
    var_53 = [var_0]
    var_54 = [var_21]
    var_55 = [var_22]
    var_56 = {var_21: var_54, var_22: var_55}
    var_57 = [var_27, var_28]
    var_58 = {var_26: var_57}
    var_59 = {var_19: var_56, var_20: var_58}
    var_60 = {var_18: var_59}
    var_61 = {}
    var_62 = {}
    var_63 = module_0.ParsedContent()
    var_64 = 'LOCALFOLDER'
    var_65 = [var_64]
    var_66 = 'local'
    var_67 = [var_66]
    var_68 = {var_66: var_67}
    var_69 = {var_19: var_68}
    var_70 = {var_64: var_69}
    var_71 = module_1.Config()
    var_72 = module_2.sorted_imports(var_63, var_71)
    var_73 = 'import os\nimport sys\n\nfrom collections import defaultdict, OrderedDict\n\nimport local\n'



# Parsed testcases at query #39
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = 'x = 1'
    var_2 = [var_0, var_1]
    var_3 = 'THIRDPARTY'
    var_4 = 'straight'
    var_5 = 'from'
    var_6 = 'os'
    var_7 = 'sys'
    var_8 = []
    var_9 = []
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'collections'
    var_12 = 'defaultdict'
    var_13 = [var_12]
    var_14 = {var_11: var_13}
    var_15 = {var_4: var_10, var_5: var_14}
    var_16 = {var_3: var_15}
    var_17 = 0
    var_18 = 2
    var_19 = '\n'
    var_20 = {}
    var_21 = {}
    var_22 = module_0.ParsedContent()
    var_23 = module_1.Config()
    var_24 = module_2.sorted_imports(var_22, var_23)
    assert var_24 == 'import os\nimport sys\n\nfrom collections import defaultdict\n\nx = 1\n'
    var_25 = [var_1]
    var_26 = {}
    var_27 = -1
    var_28 = 1
    var_29 = {}
    var_30 = {}
    var_31 = module_0.ParsedContent()
    var_32 = module_2.sorted_imports(var_31, var_23)
    assert var_32 == 'x = 1\n'
    var_33 = [var_0, var_1]
    var_34 = 'FUTURE'
    var_35 = 'STDLIB'
    var_36 = '__future__'
    var_37 = []
    var_38 = {var_36: var_37}
    var_39 = {}
    var_40 = {var_4: var_38, var_5: var_39}
    var_41 = []
    var_42 = {var_6: var_41}
    var_43 = {}
    var_44 = {var_4: var_42, var_5: var_43}
    var_45 = 'django'
    var_46 = []
    var_47 = {var_45: var_46}
    var_48 = 'numpy'
    var_49 = 'array'
    var_50 = [var_49]
    var_51 = {var_48: var_50}
    var_52 = {var_4: var_47, var_5: var_51}
    var_53 = {var_34: var_40, var_35: var_44, var_3: var_52}
    var_54 = {}
    var_55 = {}
    var_56 = module_0.ParsedContent()
    var_57 = 'future'
    var_58 = 'stdlib'
    var_59 = 'thirdparty'
    var_60 = 'Future'
    var_61 = 'Standard Library'
    var_62 = 'Third Party'
    var_63 = {var_57: var_60, var_58: var_61, var_59: var_62}
    var_64 = module_1.Config()
    var_65 = module_2.sorted_imports(var_56, var_64)
    var_66 = [var_0, var_1]
    var_67 = 'unused'
    var_68 = []
    var_69 = []
    var_70 = []
    var_71 = {var_6: var_68, var_7: var_69, var_67: var_70}
    var_72 = [var_12]
    var_73 = {var_11: var_72}
    var_74 = {var_4: var_71, var_5: var_73}
    var_75 = {var_3: var_74}
    var_76 = {}
    var_77 = {}
    var_78 = module_0.ParsedContent()
    var_79 = [var_67]
    var_80 = module_1.Config()
    var_81 = module_2.sorted_imports(var_78, var_80)
    var_82 = [var_0, var_1]
    var_83 = []
    var_84 = {var_36: var_83}
    var_85 = {}
    var_86 = {var_4: var_84, var_5: var_85}
    var_87 = []
    var_88 = {var_6: var_87}
    var_89 = {}
    var_90 = {var_4: var_88, var_5: var_89}
    var_91 = []
    var_92 = {var_45: var_91}
    var_93 = {}
    var_94 = {var_4: var_92, var_5: var_93}
    var_95 = {var_34: var_86, var_35: var_90, var_3: var_94}
    var_96 = {}
    var_97 = {}
    var_98 = module_0.ParsedContent()
    var_99 = module_1.Config()
    var_100 = module_2.sorted_imports(var_98, var_99)
    var_101 = '\n\n'
    var_102 = [var_0, var_1]
    var_103 = []
    var_104 = {var_36: var_103}
    var_105 = {}
    var_106 = {var_4: var_104, var_5: var_105}
    var_107 = []
    var_108 = {var_6: var_107}
    var_109 = {}
    var_110 = {var_4: var_108, var_5: var_109}
    var_111 = []
    var_112 = {var_45: var_111}
    var_113 = {}
    var_114 = {var_4: var_112, var_5: var_113}
    var_115 = {var_34: var_106, var_35: var_110, var_3: var_114}
    var_116 = {}
    var_117 = {}
    var_118 = module_0.ParsedContent()
    var_119 = True
    var_120 = module_1.Config()
    var_121 = module_2.sorted_imports(var_118, var_120)
    var_122 = [var_0, var_1]
    var_123 = []
    var_124 = []
    var_125 = {var_6: var_123, var_7: var_124}
    var_126 = [var_12]
    var_127 = {var_11: var_126}
    var_128 = {var_4: var_125, var_5: var_127}
    var_129 = {var_3: var_128}
    var_130 = {}
    var_131 = {}
    var_132 = module_0.ParsedContent()
    var_133 = True
    var_134 = module_1.Config()
    var_135 = module_2.sorted_imports(var_132, var_134)
    var_136 = 'import sys'
    var_137 = [var_0, var_1]
    var_138 = {}
    var_139 = '*'
    var_140 = [var_139, var_49]
    var_141 = [var_12]
    var_142 = {var_48: var_140, var_11: var_141}
    var_143 = {var_4: var_138, var_5: var_142}
    var_144 = {var_3: var_143}
    var_145 = {}
    var_146 = {}
    var_147 = module_0.ParsedContent()
    var_148 = True
    var_149 = module_1.Config()
    var_150 = module_2.sorted_imports(var_147, var_149)
    var_151 = 'from numpy import *'
    var_152 = 'from numpy import array'
    var_153 = [var_0, var_1]
    var_154 = []
    var_155 = {var_6: var_154}
    var_156 = [var_12]
    var_157 = {var_11: var_156}
    var_158 = {var_4: var_155, var_5: var_157}
    var_159 = {var_3: var_158}
    var_160 = {}
    var_161 = {}
    var_162 = module_0.ParsedContent()
    var_163 = True
    var_164 = module_1.Config()
    var_165 = module_2.sorted_imports(var_162, var_164)
    var_166 = 'from collections'
    var_167 = 'import os'
    var_168 = [var_0, var_1]
    var_169 = []
    var_170 = {var_6: var_169}
    var_171 = [var_12]
    var_172 = {var_11: var_171}
    var_173 = {var_4: var_170, var_5: var_172}
    var_174 = {var_3: var_173}
    var_175 = {}
    var_176 = {}
    var_177 = module_0.ParsedContent()
    var_178 = 'py'
    var_179 = module_2.sorted_imports(var_177, var_164, var_178)



# Parsed testcases at query #40
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 'STDLIB'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = 'sys'
    var_7 = []
    var_8 = []
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {}
    var_11 = {var_3: var_9, var_4: var_10}
    var_12 = {var_2: var_11}
    var_13 = 0
    var_14 = 1
    var_15 = '\n'
    var_16 = {}
    var_17 = {}
    var_18 = module_0.ParsedContent()
    var_19 = module_1.Config()
    var_20 = module_2.sorted_imports(var_18, var_19)
    assert var_20 == 'import os\nimport sys\n'
    var_21 = [var_0]
    var_22 = 'FUTURE'
    var_23 = '__future__'
    var_24 = []
    var_25 = {var_23: var_24}
    var_26 = {}
    var_27 = {var_3: var_25, var_4: var_26}
    var_28 = []
    var_29 = {var_5: var_28}
    var_30 = {}
    var_31 = {var_3: var_29, var_4: var_30}
    var_32 = {var_22: var_27, var_2: var_31}
    var_33 = {}
    var_34 = {}
    var_35 = module_0.ParsedContent()
    var_36 = True
    var_37 = module_1.Config()
    var_38 = module_2.sorted_imports(var_35, var_37)
    assert var_38 == 'from __future__ import absolute_import\nimport os\n'
    var_39 = [var_0]
    var_40 = []
    var_41 = []
    var_42 = {var_5: var_40, var_6: var_41}
    var_43 = 'os.path'
    var_44 = 'join'
    var_45 = [var_44]
    var_46 = 'argv'
    var_47 = [var_46]
    var_48 = {var_43: var_45, var_6: var_47}
    var_49 = {var_3: var_42, var_4: var_48}
    var_50 = {var_2: var_49}
    var_51 = {}
    var_52 = {}
    var_53 = module_0.ParsedContent()
    var_54 = True
    var_55 = module_1.Config()
    var_56 = module_2.sorted_imports(var_53, var_55)
    assert var_56 == 'import os\nimport sys\n\nfrom os.path import join\nfrom sys import argv\n'
    var_57 = [var_0]
    var_58 = 'CUSTOM'
    var_59 = 'custom'
    var_60 = []
    var_61 = {var_59: var_60}
    var_62 = {}
    var_63 = {var_3: var_61, var_4: var_62}
    var_64 = []
    var_65 = {var_5: var_64}
    var_66 = {}
    var_67 = {var_3: var_65, var_4: var_66}
    var_68 = {var_58: var_63, var_2: var_67}
    var_69 = {}
    var_70 = {}
    var_71 = module_0.ParsedContent()
    var_72 = [var_58]
    var_73 = module_1.Config()
    var_74 = module_2.sorted_imports(var_71, var_73)
    assert var_74 == 'import custom\n\nimport os\n'
    var_75 = [var_0]
    var_76 = {}
    var_77 = -1
    var_78 = {}
    var_79 = {}
    var_80 = module_0.ParsedContent()
    var_81 = module_1.Config()
    var_82 = module_2.sorted_imports(var_80, var_81)
    assert var_82 == ''



# Parsed testcases at query #41
#--------------------------


import isort.parse as module_0
import isort.output as module_1
import isort.settings as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = 0
    var_5 = '\n'
    var_6 = {}
    var_7 = {}
    var_8 = module_0.ParsedContent()
    var_9 = module_1.sorted_imports(var_8)
    assert var_9 == ''
    var_10 = "print('hello')"
    var_11 = [var_10]
    var_12 = {}
    var_13 = -1
    var_14 = 1
    var_15 = {}
    var_16 = {}
    var_17 = module_0.ParsedContent()
    var_18 = module_1.sorted_imports(var_17)
    assert var_18 == "print('hello')"
    var_19 = [var_0]
    var_20 = 'THIRDPARTY'
    var_21 = 'straight'
    var_22 = 'from'
    var_23 = 'os'
    var_24 = 'sys'
    var_25 = []
    var_26 = []
    var_27 = {var_23: var_25, var_24: var_26}
    var_28 = 'collections'
    var_29 = 'defaultdict'
    var_30 = [var_29]
    var_31 = {var_28: var_30}
    var_32 = {var_21: var_27, var_22: var_31}
    var_33 = {var_20: var_32}
    var_34 = {}
    var_35 = {}
    var_36 = module_0.ParsedContent()
    var_37 = 'import os\nimport sys\n\nfrom collections import defaultdict\n'
    var_38 = module_1.sorted_imports(var_36)
    var_39 = 2
    var_40 = True
    var_41 = module_2.Config()
    var_42 = [var_0]
    var_43 = []
    var_44 = []
    var_45 = {var_23: var_43, var_24: var_44}
    var_46 = [var_29]
    var_47 = {var_28: var_46}
    var_48 = {var_21: var_45, var_22: var_47}
    var_49 = {var_20: var_48}
    var_50 = {}
    var_51 = {}
    var_52 = module_0.ParsedContent()
    var_53 = 'from collections import defaultdict\n\nimport os\nimport sys\n'
    var_54 = module_1.sorted_imports(var_52, var_41)
    var_55 = [var_23]
    var_56 = module_2.Config()
    var_57 = [var_0]
    var_58 = []
    var_59 = []
    var_60 = {var_23: var_58, var_24: var_59}
    var_61 = [var_29]
    var_62 = {var_28: var_61}
    var_63 = {var_21: var_60, var_22: var_62}
    var_64 = {var_20: var_63}
    var_65 = {}
    var_66 = {}
    var_67 = module_0.ParsedContent()
    var_68 = 'import sys\n\nfrom collections import defaultdict\n'
    var_69 = module_1.sorted_imports(var_67, var_56)
    var_70 = 'thirdparty'
    var_71 = 'Third Party Imports'
    var_72 = {var_70: var_71}
    var_73 = module_2.Config()
    var_74 = [var_0]
    var_75 = []
    var_76 = []
    var_77 = {var_23: var_75, var_24: var_76}
    var_78 = [var_29]
    var_79 = {var_28: var_78}
    var_80 = {var_21: var_77, var_22: var_79}
    var_81 = {var_20: var_80}
    var_82 = {}
    var_83 = {}
    var_84 = module_0.ParsedContent()
    var_85 = '# Third Party Imports\nimport os\nimport sys\n\nfrom collections import defaultdict\n'
    var_86 = module_1.sorted_imports(var_84, var_73)
    var_87 = True
    var_88 = module_2.Config()
    var_89 = [var_0]
    var_90 = {}
    var_91 = '*'
    var_92 = [var_91]
    var_93 = 'path'
    var_94 = [var_93]
    var_95 = {var_28: var_92, var_23: var_94}
    var_96 = {var_21: var_90, var_22: var_95}
    var_97 = {var_20: var_96}
    var_98 = {}
    var_99 = {}
    var_100 = module_0.ParsedContent()
    var_101 = 'from collections import *\nfrom os import path\n'
    var_102 = module_1.sorted_imports(var_100, var_88)
    var_103 = True
    var_104 = module_2.Config()
    var_105 = [var_0]
    var_106 = 'FUTURE'
    var_107 = '__future__'
    var_108 = 'annotations'
    var_109 = [var_108]
    var_110 = {var_107: var_109}
    var_111 = {}
    var_112 = {var_21: var_110, var_22: var_111}
    var_113 = []
    var_114 = []
    var_115 = {var_23: var_113, var_24: var_114}
    var_116 = [var_29]
    var_117 = {var_28: var_116}
    var_118 = {var_21: var_115, var_22: var_117}
    var_119 = {var_106: var_112, var_20: var_118}
    var_120 = {}
    var_121 = {}
    var_122 = module_0.ParsedContent()
    var_123 = 'from __future__ import annotations\nimport os\nimport sys\n\nfrom collections import defaultdict\n'
    var_124 = module_1.sorted_imports(var_122, var_104)
    var_125 = [var_10]
    var_126 = []
    var_127 = {var_23: var_126}
    var_128 = {}
    var_129 = {var_21: var_127, var_22: var_128}
    var_130 = {var_20: var_129}
    var_131 = 'import sys'
    var_132 = [var_131]
    var_133 = {var_20: var_132}
    var_134 = {var_10: var_20}
    var_135 = module_0.ParsedContent()
    var_136 = "import os\nprint('hello')\nimport sys\n"
    var_137 = module_1.sorted_imports(var_135)
    var_138 = [var_0]
    var_139 = []
    var_140 = {var_23: var_139}
    var_141 = {}
    var_142 = {var_21: var_140, var_22: var_141}
    var_143 = {var_20: var_142}
    var_144 = {}
    var_145 = {}
    var_146 = module_0.ParsedContent()
    var_147 = 'import os\r\n'
    var_148 = module_1.sorted_imports(var_146, var_104)



# Parsed testcases at query #42
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = 'x = 1'
    var_2 = [var_0, var_1]
    var_3 = 'THIRDPARTY'
    var_4 = 'straight'
    var_5 = 'from'
    var_6 = 'os'
    var_7 = 'sys'
    var_8 = set()
    var_9 = set()
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'collections'
    var_12 = 'defaultdict'
    var_13 = 'OrderedDict'
    var_14 = {var_12, var_13}
    var_15 = {var_11: var_14}
    var_16 = {var_4: var_10, var_5: var_15}
    var_17 = {var_3: var_16}
    var_18 = 0
    var_19 = 2
    var_20 = '\n'
    var_21 = {}
    var_22 = {}
    var_23 = module_0.ParsedContent()
    var_24 = module_1.Config()
    var_25 = module_2.sorted_imports(var_23, var_24)
    assert var_25 == 'from collections import OrderedDict, defaultdict\n\nimport os\nimport sys\n\nx = 1\n'
    var_26 = [var_0, var_1]
    var_27 = 'FUTURE'
    var_28 = '__future__'
    var_29 = 'print_function'
    var_30 = {var_29}
    var_31 = {var_28: var_30}
    var_32 = {}
    var_33 = {var_4: var_31, var_5: var_32}
    var_34 = set()
    var_35 = set()
    var_36 = {var_6: var_34, var_7: var_35}
    var_37 = {var_12, var_13}
    var_38 = {var_11: var_37}
    var_39 = {var_4: var_36, var_5: var_38}
    var_40 = {var_27: var_33, var_3: var_39}
    var_41 = {}
    var_42 = {}
    var_43 = module_0.ParsedContent()
    var_44 = True
    var_45 = module_1.Config()
    var_46 = module_2.sorted_imports(var_43, var_45)
    assert var_46 == 'from __future__ import print_function\n\nfrom collections import OrderedDict, defaultdict\n\nimport os\nimport sys\n\nx = 1\n'
    var_47 = [var_0, var_1]
    var_48 = set()
    var_49 = set()
    var_50 = {var_6: var_48, var_7: var_49}
    var_51 = {var_12, var_13}
    var_52 = {var_11: var_51}
    var_53 = {var_4: var_50, var_5: var_52}
    var_54 = {var_3: var_53}
    var_55 = {}
    var_56 = {}
    var_57 = module_0.ParsedContent()
    var_58 = module_1.Config()
    var_59 = module_2.sorted_imports(var_57, var_58)
    assert var_59 == 'from collections import OrderedDict, defaultdict\n\nimport os\nimport sys\n\nx = 1\n'
    var_60 = [var_0, var_1]
    var_61 = set()
    var_62 = {var_6: var_61}
    var_63 = 'numpy'
    var_64 = {var_12, var_13}
    var_65 = '*'
    var_66 = {var_65}
    var_67 = {var_11: var_64, var_63: var_66}
    var_68 = {var_4: var_62, var_5: var_67}
    var_69 = {var_3: var_68}
    var_70 = {}
    var_71 = {}
    var_72 = module_0.ParsedContent()
    var_73 = module_1.Config()
    var_74 = module_2.sorted_imports(var_72, var_73)
    assert var_74 == 'from numpy import *\nfrom collections import OrderedDict, defaultdict\n\nimport os\n\nx = 1\n'
    var_75 = [var_0, var_1]
    var_76 = set()
    var_77 = {var_6: var_76}
    var_78 = {var_12}
    var_79 = {var_11: var_78}
    var_80 = {var_4: var_77, var_5: var_79}
    var_81 = {var_3: var_80}
    var_82 = {}
    var_83 = {}
    var_84 = module_0.ParsedContent()
    var_85 = 'thirdparty'
    var_86 = 'Third Party Imports'
    var_87 = {var_85: var_86}
    var_88 = module_1.Config()
    var_89 = module_2.sorted_imports(var_84, var_88)
    assert var_89 == '# Third Party Imports\nfrom collections import defaultdict\n\nimport os\n\nx = 1\n'
    var_90 = [var_1]
    var_91 = {}
    var_92 = -1
    var_93 = {}
    var_94 = {}
    var_95 = module_0.ParsedContent()
    var_96 = module_1.Config()
    var_97 = module_2.sorted_imports(var_95, var_96)
    assert var_97 == 'x = 1\n'



# Parsed testcases at query #43
#--------------------------


import isort.parse as module_0
import isort.output as module_1
import isort.settings as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = 0
    var_5 = '\n'
    var_6 = {}
    var_7 = {}
    var_8 = module_0.ParsedContent()
    var_9 = module_1.sorted_imports(var_8)
    assert var_9 == ''
    var_10 = "print('hello')"
    var_11 = [var_10]
    var_12 = {}
    var_13 = -1
    var_14 = 1
    var_15 = {}
    var_16 = {}
    var_17 = module_0.ParsedContent()
    var_18 = module_1.sorted_imports(var_17)
    assert var_18 == "print('hello')"
    var_19 = [var_0]
    var_20 = 'THIRDPARTY'
    var_21 = 'straight'
    var_22 = 'from'
    var_23 = 'os'
    var_24 = 'sys'
    var_25 = [var_23]
    var_26 = [var_24]
    var_27 = {var_23: var_25, var_24: var_26}
    var_28 = 'collections'
    var_29 = 'OrderedDict'
    var_30 = [var_29]
    var_31 = {var_28: var_30}
    var_32 = {var_21: var_27, var_22: var_31}
    var_33 = {var_20: var_32}
    var_34 = {}
    var_35 = {}
    var_36 = module_0.ParsedContent()
    var_37 = 'import os\nimport sys\n\nfrom collections import OrderedDict'
    var_38 = module_1.sorted_imports(var_36)
    var_39 = 2
    var_40 = True
    var_41 = module_2.Config()
    var_42 = [var_0]
    var_43 = [var_23]
    var_44 = [var_24]
    var_45 = {var_23: var_43, var_24: var_44}
    var_46 = [var_29]
    var_47 = {var_28: var_46}
    var_48 = {var_21: var_45, var_22: var_47}
    var_49 = {var_20: var_48}
    var_50 = {}
    var_51 = {}
    var_52 = module_0.ParsedContent()
    var_53 = 'from collections import OrderedDict\n\nimport os\nimport sys'
    var_54 = module_1.sorted_imports(var_52, var_41)
    var_55 = 'thirdparty'
    var_56 = 'Third Party Imports'
    var_57 = {var_55: var_56}
    var_58 = True
    var_59 = module_2.Config()
    var_60 = [var_0]
    var_61 = [var_23]
    var_62 = {var_23: var_61}
    var_63 = {}
    var_64 = {var_21: var_62, var_22: var_63}
    var_65 = {var_20: var_64}
    var_66 = {}
    var_67 = {}
    var_68 = module_0.ParsedContent()
    var_69 = '# Third Party Imports\nimport os'
    var_70 = module_1.sorted_imports(var_68, var_59)
    var_71 = 'FUTURE'
    var_72 = 'STDLIB'
    var_73 = [var_71, var_72]
    var_74 = module_2.Config()
    var_75 = [var_0]
    var_76 = '__future__'
    var_77 = [var_76]
    var_78 = {var_76: var_77}
    var_79 = {}
    var_80 = {var_21: var_78, var_22: var_79}
    var_81 = [var_23]
    var_82 = {var_23: var_81}
    var_83 = {}
    var_84 = {var_21: var_82, var_22: var_83}
    var_85 = {var_71: var_80, var_72: var_84}
    var_86 = {}
    var_87 = {}
    var_88 = module_0.ParsedContent()
    var_89 = 'import __future__\n\nimport os'
    var_90 = module_1.sorted_imports(var_88, var_74)
    var_91 = True
    var_92 = module_2.Config()
    var_93 = [var_0]
    var_94 = {}
    var_95 = 'module1'
    var_96 = 'module2'
    var_97 = '*'
    var_98 = [var_97]
    var_99 = 'function1'
    var_100 = [var_99]
    var_101 = {var_95: var_98, var_96: var_100}
    var_102 = {var_21: var_94, var_22: var_101}
    var_103 = {var_20: var_102}
    var_104 = {}
    var_105 = {}
    var_106 = module_0.ParsedContent()
    var_107 = 'from module1 import *\nfrom module2 import function1'
    var_108 = module_1.sorted_imports(var_106, var_92)
    var_109 = module_2.Config()
    var_110 = 'def func():'
    var_111 = '    pass'
    var_112 = [var_0, var_110, var_111]
    var_113 = [var_23]
    var_114 = {var_23: var_113}
    var_115 = {}
    var_116 = {var_21: var_114, var_22: var_115}
    var_117 = {var_20: var_116}
    var_118 = 3
    var_119 = {}
    var_120 = {}
    var_121 = module_0.ParsedContent()
    var_122 = 'import os\n\ndef func():'
    var_123 = module_1.sorted_imports(var_121, var_109)
    var_124 = 'import os'
    var_125 = [var_124]
    var_126 = module_2.Config()
    var_127 = [var_0]
    var_128 = [var_23]
    var_129 = [var_24]
    var_130 = {var_23: var_128, var_24: var_129}
    var_131 = {}
    var_132 = {var_21: var_130, var_22: var_131}
    var_133 = {var_20: var_132}
    var_134 = {}
    var_135 = {}
    var_136 = module_0.ParsedContent()
    var_137 = 'import sys'
    var_138 = module_1.sorted_imports(var_136, var_126)
    var_139 = True
    var_140 = module_2.Config()
    var_141 = [var_0]
    var_142 = [var_76]
    var_143 = {var_76: var_142}
    var_144 = {}
    var_145 = {var_21: var_143, var_22: var_144}
    var_146 = [var_23]
    var_147 = {var_23: var_146}
    var_148 = {}
    var_149 = {var_21: var_147, var_22: var_148}
    var_150 = {var_71: var_145, var_72: var_149}
    var_151 = {}
    var_152 = {}
    var_153 = module_0.ParsedContent()
    var_154 = 'import __future__\nimport os'
    var_155 = module_1.sorted_imports(var_153, var_140)



# Parsed testcases at query #44
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = 0
    var_5 = '\n'
    var_6 = {}
    var_7 = {}
    var_8 = module_0.ParsedContent()
    var_9 = "print('hello')"
    var_10 = [var_9]
    var_11 = {}
    var_12 = -1
    var_13 = 1
    var_14 = {}
    var_15 = {}
    var_16 = module_0.ParsedContent()
    var_17 = [var_0]
    var_18 = 'THIRDPARTY'
    var_19 = 'straight'
    var_20 = 'from'
    var_21 = 'numpy'
    var_22 = 'pandas'
    var_23 = []
    var_24 = []
    var_25 = {var_21: var_23, var_22: var_24}
    var_26 = 'collections'
    var_27 = 'defaultdict'
    var_28 = [var_27]
    var_29 = {var_26: var_28}
    var_30 = {var_19: var_25, var_20: var_29}
    var_31 = {var_18: var_30}
    var_32 = {}
    var_33 = {}
    var_34 = module_0.ParsedContent()
    var_35 = '\nimport numpy\nimport pandas\n\nfrom collections import defaultdict\n'
    var_36 = [var_0]
    var_37 = []
    var_38 = []
    var_39 = {var_21: var_37, var_22: var_38}
    var_40 = [var_27]
    var_41 = {var_26: var_40}
    var_42 = {var_19: var_39, var_20: var_41}
    var_43 = {var_18: var_42}
    var_44 = {}
    var_45 = {}
    var_46 = module_0.ParsedContent()
    var_47 = 2
    var_48 = True
    var_49 = module_1.Config()
    var_50 = module_2.sorted_imports(var_46, var_49)
    var_51 = '\nfrom collections import defaultdict\n\nimport numpy\nimport pandas\n'
    var_52 = [var_0]
    var_53 = 'FIRSTPARTY'
    var_54 = []
    var_55 = {var_21: var_54}
    var_56 = {}
    var_57 = {var_19: var_55, var_20: var_56}
    var_58 = 'my_module'
    var_59 = []
    var_60 = {var_58: var_59}
    var_61 = {}
    var_62 = {var_19: var_60, var_20: var_61}
    var_63 = {var_18: var_57, var_53: var_62}
    var_64 = {}
    var_65 = {}
    var_66 = module_0.ParsedContent()
    var_67 = 'LOCALFOLDER'
    var_68 = [var_67]
    var_69 = module_1.Config()
    var_70 = module_2.sorted_imports(var_66, var_69)
    var_71 = '\nimport numpy\n\nimport my_module\n'



# Parsed testcases at query #45
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = -1
    var_3 = 0
    var_4 = '\n'
    var_5 = {}
    var_6 = {}
    var_7 = []
    var_8 = module_0.ParsedContent()
    var_9 = module_1.Config()
    var_10 = module_2.sorted_imports(var_8, var_9)
    assert var_10 == ''
    var_11 = "print('hello')"
    var_12 = [var_11]
    var_13 = {}
    var_14 = -1
    var_15 = 1
    var_16 = {}
    var_17 = {}
    var_18 = []
    var_19 = module_0.ParsedContent()
    var_20 = module_1.Config()
    var_21 = module_2.sorted_imports(var_19, var_20)
    assert var_21 == "print('hello')"
    var_22 = ''
    var_23 = [var_22]
    var_24 = 'THIRDPARTY'
    var_25 = 'straight'
    var_26 = 'from'
    var_27 = 'os'
    var_28 = 'sys'
    var_29 = [var_27]
    var_30 = [var_28]
    var_31 = {var_27: var_29, var_28: var_30}
    var_32 = 'collections'
    var_33 = 'itertools'
    var_34 = 'defaultdict'
    var_35 = [var_34]
    var_36 = 'chain'
    var_37 = [var_36]
    var_38 = {var_32: var_35, var_33: var_37}
    var_39 = {var_25: var_31, var_26: var_38}
    var_40 = {var_24: var_39}
    var_41 = {}
    var_42 = {}
    var_43 = [var_24]
    var_44 = module_0.ParsedContent()
    var_45 = module_1.Config()
    var_46 = module_2.sorted_imports(var_44, var_45)
    var_47 = 'import os'
    var_48 = 'import sys'
    var_49 = 'from collections import defaultdict'
    var_50 = 'from itertools import chain'
    var_51 = [var_47, var_48, var_22, var_49, var_50]
    var_52 = [var_22]
    var_53 = [var_27]
    var_54 = [var_28]
    var_55 = {var_27: var_53, var_28: var_54}
    var_56 = [var_34]
    var_57 = [var_36]
    var_58 = {var_32: var_56, var_33: var_57}
    var_59 = {var_25: var_55, var_26: var_58}
    var_60 = {var_24: var_59}
    var_61 = {}
    var_62 = {}
    var_63 = [var_24]
    var_64 = module_0.ParsedContent()
    var_65 = True
    var_66 = module_1.Config()
    var_67 = module_2.sorted_imports(var_64, var_66)
    var_68 = [var_48, var_47, var_22, var_50, var_49]
    var_69 = [var_22]
    var_70 = '*'
    var_71 = [var_70]
    var_72 = 'path'
    var_73 = [var_72]
    var_74 = [var_34]
    var_75 = {var_27: var_71, var_28: var_73, var_32: var_74}
    var_76 = {var_26: var_75}
    var_77 = {var_24: var_76}
    var_78 = {}
    var_79 = {}
    var_80 = [var_24]
    var_81 = module_0.ParsedContent()
    var_82 = True
    var_83 = module_1.Config()
    var_84 = module_2.sorted_imports(var_81, var_83)
    var_85 = 'from os import *'
    var_86 = 'from sys import path'
    var_87 = [var_85, var_49, var_86]
    var_88 = [var_22]
    var_89 = 'FUTURE'
    var_90 = 'FIRSTPARTY'
    var_91 = '__future__'
    var_92 = 'annotations'
    var_93 = [var_92]
    var_94 = {var_91: var_93}
    var_95 = {var_25: var_94}
    var_96 = [var_27]
    var_97 = [var_28]
    var_98 = {var_27: var_96, var_28: var_97}
    var_99 = [var_34]
    var_100 = {var_32: var_99}
    var_101 = {var_25: var_98, var_26: var_100}
    var_102 = 'my_module'
    var_103 = [var_102]
    var_104 = {var_102: var_103}
    var_105 = {var_25: var_104}
    var_106 = {var_89: var_95, var_24: var_101, var_90: var_105}
    var_107 = {}
    var_108 = {}
    var_109 = [var_89, var_24, var_90]
    var_110 = module_0.ParsedContent()
    var_111 = True
    var_112 = module_1.Config()
    var_113 = module_2.sorted_imports(var_110, var_112)
    var_114 = 'from __future__ import annotations'
    var_115 = 'import my_module'
    var_116 = [var_114, var_22, var_49, var_115, var_47, var_48]
    var_117 = [var_22]
    var_118 = [var_27]
    var_119 = {var_27: var_118}
    var_120 = {var_25: var_119}
    var_121 = {var_24: var_120}
    var_122 = {}
    var_123 = {}
    var_124 = [var_24]
    var_125 = module_0.ParsedContent()
    var_126 = 'thirdparty'
    var_127 = 'Third Party Imports'
    var_128 = {var_126: var_127}
    var_129 = module_1.Config()
    var_130 = module_2.sorted_imports(var_125, var_129)
    var_131 = '# Third Party Imports'
    var_132 = [var_131, var_47]
    var_133 = [var_22]
    var_134 = [var_27]
    var_135 = {var_27: var_134}
    var_136 = {var_25: var_135}
    var_137 = [var_102]
    var_138 = {var_102: var_137}
    var_139 = {var_25: var_138}
    var_140 = {var_24: var_136, var_90: var_139}
    var_141 = {}
    var_142 = {}
    var_143 = [var_24, var_90]
    var_144 = module_0.ParsedContent()
    var_145 = 2
    var_146 = module_1.Config()
    var_147 = module_2.sorted_imports(var_144, var_146)
    var_148 = [var_47, var_22, var_22, var_115]
    var_149 = 'def main():'
    var_150 = '    pass'
    var_151 = [var_149, var_150]
    var_152 = [var_27]
    var_153 = {var_27: var_152}
    var_154 = {var_25: var_153}
    var_155 = {var_24: var_154}
    var_156 = {}
    var_157 = {}
    var_158 = [var_24]
    var_159 = module_0.ParsedContent()
    var_160 = module_1.Config()
    var_161 = module_2.sorted_imports(var_159, var_160)
    var_162 = [var_47, var_22, var_22, var_149, var_150]
    var_163 = [var_22]
    var_164 = [var_27]
    var_165 = [var_28]
    var_166 = {var_27: var_164, var_28: var_165}
    var_167 = {var_25: var_166}
    var_168 = {var_24: var_167}
    var_169 = {}
    var_170 = {}
    var_171 = [var_24]
    var_172 = module_0.ParsedContent()
    var_173 = [var_27]
    var_174 = module_1.Config()
    var_175 = module_2.sorted_imports(var_172, var_174)
    var_176 = 'import sys'



# Parsed testcases at query #46
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = 'x = 1'
    var_2 = [var_0, var_1]
    var_3 = 'THIRDPARTY'
    var_4 = 'straight'
    var_5 = 'from'
    var_6 = 'os'
    var_7 = 'sys'
    var_8 = [var_6]
    var_9 = [var_7]
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'collections'
    var_12 = 'defaultdict'
    var_13 = 'Counter'
    var_14 = [var_12, var_13]
    var_15 = {var_11: var_14}
    var_16 = {var_4: var_10, var_5: var_15}
    var_17 = {var_3: var_16}
    var_18 = 0
    var_19 = 2
    var_20 = '\n'
    var_21 = module_0.ParsedContent()
    var_22 = module_1.Config()
    var_23 = module_2.sorted_imports(var_21, var_22)
    assert var_23 == '\nos\nsys\n\nfrom collections import Counter, defaultdict\n\nx = 1\n'
    var_24 = [var_0, var_1]
    var_25 = 'FIRSTPARTY'
    var_26 = [var_6]
    var_27 = [var_7]
    var_28 = {var_6: var_26, var_7: var_27}
    var_29 = [var_12, var_13]
    var_30 = {var_11: var_29}
    var_31 = {var_4: var_28, var_5: var_30}
    var_32 = 'myapp'
    var_33 = [var_32]
    var_34 = {var_32: var_33}
    var_35 = 'myapp.utils'
    var_36 = 'helper'
    var_37 = [var_36]
    var_38 = {var_35: var_37}
    var_39 = {var_4: var_34, var_5: var_38}
    var_40 = {var_3: var_31, var_25: var_39}
    var_41 = module_0.ParsedContent()
    var_42 = True
    var_43 = module_1.Config()
    var_44 = module_2.sorted_imports(var_41, var_43)
    assert var_44 == '\nmyapp\nos\nsys\n\nfrom collections import Counter, defaultdict\nfrom myapp.utils import helper\n\nx = 1\n'
    var_45 = [var_0, var_1]
    var_46 = [var_6]
    var_47 = {var_6: var_46}
    var_48 = 'numpy'
    var_49 = [var_12]
    var_50 = '*'
    var_51 = [var_50]
    var_52 = {var_11: var_49, var_48: var_51}
    var_53 = {var_4: var_47, var_5: var_52}
    var_54 = {var_3: var_53}
    var_55 = module_0.ParsedContent()
    var_56 = module_1.Config()
    var_57 = module_2.sorted_imports(var_55, var_56)
    assert var_57 == '\nos\n\nfrom numpy import *\nfrom collections import defaultdict\n\nx = 1\n'
    var_58 = [var_0, var_1]
    var_59 = [var_6]
    var_60 = {var_6: var_59}
    var_61 = [var_12]
    var_62 = {var_11: var_61}
    var_63 = {var_4: var_60, var_5: var_62}
    var_64 = {var_3: var_63}
    var_65 = module_0.ParsedContent()
    var_66 = 'thirdparty'
    var_67 = 'Third-party imports'
    var_68 = {var_66: var_67}
    var_69 = module_1.Config()
    var_70 = module_2.sorted_imports(var_65, var_69)
    assert var_70 == '\n# Third-party imports\nos\n\nfrom collections import defaultdict\n\nx = 1\n'
    var_71 = [var_1]
    var_72 = {}
    var_73 = -1
    var_74 = module_0.ParsedContent()
    var_75 = module_2.sorted_imports(var_74)
    assert var_75 == 'x = 1\n'



# Parsed testcases at query #47
#--------------------------


import isort.parse as module_0
import isort.output as module_1
import isort.settings as module_2

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = -1
    var_3 = 0
    var_4 = '\n'
    var_5 = {}
    var_6 = {}
    var_7 = module_0.ParsedContent()
    var_8 = module_1.sorted_imports(var_7)
    assert var_8 == ''
    var_9 = "print('hello')"
    var_10 = [var_9]
    var_11 = {}
    var_12 = -1
    var_13 = 1
    var_14 = {}
    var_15 = {}
    var_16 = module_0.ParsedContent()
    var_17 = module_1.sorted_imports(var_16)
    assert var_17 == "print('hello')"
    var_18 = [var_9]
    var_19 = 'THIRDPARTY'
    var_20 = 'straight'
    var_21 = 'from'
    var_22 = 'os'
    var_23 = 'sys'
    var_24 = [var_22]
    var_25 = [var_23]
    var_26 = {var_22: var_24, var_23: var_25}
    var_27 = 'collections'
    var_28 = 'OrderedDict'
    var_29 = [var_28]
    var_30 = {var_27: var_29}
    var_31 = {var_20: var_26, var_21: var_30}
    var_32 = {var_19: var_31}
    var_33 = {}
    var_34 = {}
    var_35 = module_0.ParsedContent()
    var_36 = False
    var_37 = module_2.Config()
    var_38 = module_1.sorted_imports(var_35, var_37)
    var_39 = [var_9]
    var_40 = 'FUTURE'
    var_41 = 'STDLIB'
    var_42 = '__future__'
    var_43 = 'annotations'
    var_44 = [var_43]
    var_45 = {var_42: var_44}
    var_46 = {}
    var_47 = {var_20: var_45, var_21: var_46}
    var_48 = [var_22]
    var_49 = {var_22: var_48}
    var_50 = {}
    var_51 = {var_20: var_49, var_21: var_50}
    var_52 = {var_40: var_47, var_41: var_51}
    var_53 = {}
    var_54 = {}
    var_55 = module_0.ParsedContent()
    var_56 = 'future'
    var_57 = 'stdlib'
    var_58 = 'Future'
    var_59 = 'Standard Library'
    var_60 = {var_56: var_58, var_57: var_59}
    var_61 = True
    var_62 = module_2.Config()
    var_63 = module_1.sorted_imports(var_55, var_62)
    var_64 = [var_9]
    var_65 = 'FIRSTPARTY'
    var_66 = 'django'
    var_67 = [var_66]
    var_68 = {var_66: var_67}
    var_69 = {}
    var_70 = {var_20: var_68, var_21: var_69}
    var_71 = 'myapp'
    var_72 = [var_71]
    var_73 = {var_71: var_72}
    var_74 = {}
    var_75 = {var_20: var_73, var_21: var_74}
    var_76 = {var_19: var_70, var_65: var_75}
    var_77 = {}
    var_78 = {}
    var_79 = module_0.ParsedContent()
    var_80 = 'LOCALFOLDER'
    var_81 = [var_80]
    var_82 = module_2.Config()
    var_83 = module_1.sorted_imports(var_79, var_82)
    var_84 = [var_9]
    var_85 = [var_22]
    var_86 = {var_22: var_85}
    var_87 = {}
    var_88 = {var_20: var_86, var_21: var_87}
    var_89 = [var_23]
    var_90 = {var_23: var_89}
    var_91 = {}
    var_92 = {var_20: var_90, var_21: var_91}
    var_93 = {var_19: var_88, var_65: var_92}
    var_94 = {}
    var_95 = {}
    var_96 = module_0.ParsedContent()
    var_97 = True
    var_98 = module_2.Config()
    var_99 = module_1.sorted_imports(var_96, var_98)
    var_100 = [var_9]
    var_101 = {}
    var_102 = 'module1'
    var_103 = 'module2'
    var_104 = '*'
    var_105 = [var_104]
    var_106 = 'function'
    var_107 = [var_106]
    var_108 = {var_102: var_105, var_103: var_107}
    var_109 = {var_20: var_101, var_21: var_108}
    var_110 = {var_19: var_109}
    var_111 = {}
    var_112 = {}
    var_113 = module_0.ParsedContent()
    var_114 = True
    var_115 = module_2.Config()
    var_116 = module_1.sorted_imports(var_113, var_115)
    var_117 = 'from module1 import *'
    var_118 = 'from module2 import function'
    var_119 = '# IMPORTS'
    var_120 = [var_9, var_119]
    var_121 = [var_22]
    var_122 = {var_22: var_121}
    var_123 = {}
    var_124 = {var_20: var_122, var_21: var_123}
    var_125 = {var_19: var_124}
    var_126 = 2
    var_127 = 'import os'
    var_128 = [var_127]
    var_129 = {var_19: var_128}
    var_130 = {var_119: var_19}
    var_131 = module_0.ParsedContent()
    var_132 = module_2.Config()
    var_133 = module_1.sorted_imports(var_131, var_132)
    var_134 = [var_9]
    var_135 = [var_22]
    var_136 = {var_22: var_135}
    var_137 = {}
    var_138 = {var_20: var_136, var_21: var_137}
    var_139 = {var_19: var_138}
    var_140 = {}
    var_141 = {}
    var_142 = module_0.ParsedContent()
    var_143 = lambda x, y, z: x.upper()
    var_144 = module_2.Config()
    var_145 = module_1.sorted_imports(var_142, var_144)
    var_146 = '# comment'
    var_147 = [var_9, var_146]
    var_148 = [var_22]
    var_149 = {var_22: var_148}
    var_150 = {}
    var_151 = {var_20: var_149, var_21: var_150}
    var_152 = {var_19: var_151}
    var_153 = {}
    var_154 = {}
    var_155 = module_0.ParsedContent()
    var_156 = True
    var_157 = module_2.Config()
    var_158 = module_1.sorted_imports(var_155, var_157)



# Parsed testcases at query #48
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = '\n'
    var_5 = 1
    var_6 = {}
    var_7 = {}
    var_8 = module_0.ParsedContent()
    var_9 = [var_0]
    var_10 = 'THIRDPARTY'
    var_11 = 'straight'
    var_12 = 'from'
    var_13 = 'os'
    var_14 = 'sys'
    var_15 = [var_13]
    var_16 = [var_14]
    var_17 = {var_13: var_15, var_14: var_16}
    var_18 = 'collections'
    var_19 = 'defaultdict'
    var_20 = [var_19]
    var_21 = {var_18: var_20}
    var_22 = {var_11: var_17, var_12: var_21}
    var_23 = {var_10: var_22}
    var_24 = 0
    var_25 = 2
    var_26 = {}
    var_27 = {}
    var_28 = module_0.ParsedContent()
    var_29 = "from collections import defaultdict\n\nimport os\nimport sys\n\nprint('hello')"
    var_30 = 'thirdparty'
    var_31 = 'Third Party Imports'
    var_32 = {var_30: var_31}
    var_33 = module_1.Config()
    var_34 = [var_0]
    var_35 = [var_13]
    var_36 = [var_14]
    var_37 = {var_13: var_35, var_14: var_36}
    var_38 = [var_19]
    var_39 = {var_18: var_38}
    var_40 = {var_11: var_37, var_12: var_39}
    var_41 = {var_10: var_40}
    var_42 = {}
    var_43 = {}
    var_44 = module_0.ParsedContent()
    var_45 = module_2.sorted_imports(var_44, var_33)
    var_46 = "# Third Party Imports\nfrom collections import defaultdict\n\nimport os\nimport sys\n\nprint('hello')"
    var_47 = [var_13]
    var_48 = module_1.Config()
    var_49 = [var_0]
    var_50 = [var_13]
    var_51 = [var_14]
    var_52 = {var_13: var_50, var_14: var_51}
    var_53 = [var_19]
    var_54 = {var_18: var_53}
    var_55 = {var_11: var_52, var_12: var_54}
    var_56 = {var_10: var_55}
    var_57 = {}
    var_58 = {}
    var_59 = module_0.ParsedContent()
    var_60 = module_2.sorted_imports(var_59, var_48)
    var_61 = "from collections import defaultdict\n\nimport sys\n\nprint('hello')"
    var_62 = True
    var_63 = module_1.Config()
    var_64 = [var_0]
    var_65 = 'FUTURE'
    var_66 = '__future__'
    var_67 = [var_66]
    var_68 = {var_66: var_67}
    var_69 = {var_11: var_68}
    var_70 = [var_13]
    var_71 = [var_14]
    var_72 = {var_13: var_70, var_14: var_71}
    var_73 = [var_19]
    var_74 = {var_18: var_73}
    var_75 = {var_11: var_72, var_12: var_74}
    var_76 = {var_65: var_69, var_10: var_75}
    var_77 = {}
    var_78 = {}
    var_79 = module_0.ParsedContent()
    var_80 = module_2.sorted_imports(var_79, var_63)
    var_81 = "from __future__ import __future__\n\nfrom collections import defaultdict\n\nimport os\nimport sys\n\nprint('hello')"



# Parsed testcases at query #49
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'x = 1'
    var_1 = [var_0]
    var_2 = 'THIRDPARTY'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'numpy'
    var_6 = 'pandas'
    var_7 = []
    var_8 = []
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'collections'
    var_11 = 'itertools'
    var_12 = 'defaultdict'
    var_13 = [var_12]
    var_14 = 'chain'
    var_15 = [var_14]
    var_16 = {var_10: var_13, var_11: var_15}
    var_17 = {var_3: var_9, var_4: var_16}
    var_18 = {var_2: var_17}
    var_19 = 0
    var_20 = 1
    var_21 = '\n'
    var_22 = module_0.ParsedContent()
    var_23 = module_1.Config()
    var_24 = module_2.sorted_imports(var_22, var_23)
    var_25 = [var_0]
    var_26 = {}
    var_27 = -1
    var_28 = module_0.ParsedContent()
    var_29 = module_2.sorted_imports(var_28, var_23)
    assert var_29 == 'x = 1\n'
    var_30 = 'TEST'
    var_31 = [var_30]
    var_32 = module_1.Config()
    var_33 = [var_0]
    var_34 = 'pytest'
    var_35 = []
    var_36 = {var_34: var_35}
    var_37 = 'fixture'
    var_38 = [var_37]
    var_39 = {var_34: var_38}
    var_40 = {var_3: var_36, var_4: var_39}
    var_41 = []
    var_42 = {var_5: var_41}
    var_43 = [var_12]
    var_44 = {var_10: var_43}
    var_45 = {var_3: var_42, var_4: var_44}
    var_46 = {var_30: var_40, var_2: var_45}
    var_47 = module_0.ParsedContent()
    var_48 = module_2.sorted_imports(var_47, var_32)
    var_49 = True
    var_50 = module_1.Config()
    var_51 = [var_0]
    var_52 = 'FUTURE'
    var_53 = '__future__'
    var_54 = 'annotations'
    var_55 = [var_54]
    var_56 = {var_53: var_55}
    var_57 = {}
    var_58 = {var_3: var_56, var_4: var_57}
    var_59 = []
    var_60 = {var_5: var_59}
    var_61 = [var_12]
    var_62 = {var_10: var_61}
    var_63 = {var_3: var_60, var_4: var_62}
    var_64 = {var_52: var_58, var_2: var_63}
    var_65 = module_0.ParsedContent()
    var_66 = module_2.sorted_imports(var_65, var_50)
    var_67 = True
    var_68 = module_1.Config()
    var_69 = [var_0]
    var_70 = []
    var_71 = []
    var_72 = {var_5: var_70, var_6: var_71}
    var_73 = [var_12]
    var_74 = [var_14]
    var_75 = {var_10: var_73, var_11: var_74}
    var_76 = {var_3: var_72, var_4: var_75}
    var_77 = {var_2: var_76}
    var_78 = module_0.ParsedContent()
    var_79 = module_2.sorted_imports(var_78, var_68)
    var_80 = True
    var_81 = module_1.Config()
    var_82 = [var_0]
    var_83 = {}
    var_84 = 'module'
    var_85 = 'other'
    var_86 = '*'
    var_87 = [var_86]
    var_88 = 'function'
    var_89 = [var_88]
    var_90 = {var_84: var_87, var_85: var_89}
    var_91 = {var_3: var_83, var_4: var_90}
    var_92 = {var_2: var_91}
    var_93 = module_0.ParsedContent()
    var_94 = module_2.sorted_imports(var_93, var_81)
    var_95 = 'from module import *'
    var_96 = 'from other import function'
    var_97 = True
    var_98 = module_1.Config()
    var_99 = [var_0]
    var_100 = []
    var_101 = {var_5: var_100}
    var_102 = [var_12]
    var_103 = {var_10: var_102}
    var_104 = {var_3: var_101, var_4: var_103}
    var_105 = {var_2: var_104}
    var_106 = module_0.ParsedContent()
    var_107 = module_2.sorted_imports(var_106, var_98)
    var_108 = 'from collections import defaultdict'
    var_109 = 'import numpy'
    var_110 = 'thirdparty'
    var_111 = 'Third Party Imports'
    var_112 = {var_110: var_111}
    var_113 = module_1.Config()
    var_114 = [var_0]
    var_115 = []
    var_116 = {var_5: var_115}
    var_117 = [var_12]
    var_118 = {var_10: var_117}
    var_119 = {var_3: var_116, var_4: var_118}
    var_120 = {var_2: var_119}
    var_121 = module_0.ParsedContent()
    var_122 = module_2.sorted_imports(var_121, var_113)
    var_123 = 'End of Third Party Imports'
    var_124 = {var_110: var_123}
    var_125 = module_1.Config()
    var_126 = [var_0]
    var_127 = []
    var_128 = {var_5: var_127}
    var_129 = [var_12]
    var_130 = {var_10: var_129}
    var_131 = {var_3: var_128, var_4: var_130}
    var_132 = {var_2: var_131}
    var_133 = module_0.ParsedContent()
    var_134 = module_2.sorted_imports(var_133, var_125)
    var_135 = 2
    var_136 = module_1.Config()
    var_137 = [var_0]
    var_138 = [var_54]
    var_139 = {var_53: var_138}
    var_140 = {}
    var_141 = {var_3: var_139, var_4: var_140}
    var_142 = []
    var_143 = {var_5: var_142}
    var_144 = [var_12]
    var_145 = {var_10: var_144}
    var_146 = {var_3: var_143, var_4: var_145}
    var_147 = {var_52: var_141, var_2: var_146}
    var_148 = module_0.ParsedContent()
    var_149 = module_2.sorted_imports(var_148, var_136)
    var_150 = '\n\n'
    var_151 = module_1.Config()
    var_152 = [var_0]
    var_153 = []
    var_154 = {var_5: var_153}
    var_155 = [var_12]
    var_156 = {var_10: var_155}
    var_157 = {var_3: var_154, var_4: var_156}
    var_158 = {var_2: var_157}
    var_159 = module_0.ParsedContent()
    var_160 = module_2.sorted_imports(var_159, var_151)
    var_161 = ''
    var_162 = True
    var_163 = module_1.Config()
    var_164 = [var_0]
    var_165 = []
    var_166 = []
    var_167 = {var_5: var_165, var_6: var_166}
    var_168 = [var_12]
    var_169 = [var_14]
    var_170 = {var_10: var_168, var_11: var_169}
    var_171 = {var_3: var_167, var_4: var_170}
    var_172 = {var_2: var_171}
    var_173 = module_0.ParsedContent()
    var_174 = module_2.sorted_imports(var_173, var_163)
    var_175 = 'import pandas'
    var_176 = 'from itertools import chain'
    var_177 = [var_5]
    var_178 = module_1.Config()



# Parsed testcases at query #50
#--------------------------


import isort.parse as module_0
import isort.output as module_1
import isort.settings as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = 0
    var_5 = '\n'
    var_6 = {}
    var_7 = {}
    var_8 = module_0.ParsedContent()
    var_9 = module_1.sorted_imports(var_8)
    assert var_9 == ''
    var_10 = "print('hello')"
    var_11 = [var_10]
    var_12 = {}
    var_13 = -1
    var_14 = 1
    var_15 = {}
    var_16 = {}
    var_17 = module_0.ParsedContent()
    var_18 = module_1.sorted_imports(var_17)
    assert var_18 == "print('hello')"
    var_19 = [var_0]
    var_20 = 'THIRDPARTY'
    var_21 = 'straight'
    var_22 = 'from'
    var_23 = 'django'
    var_24 = 'flask'
    var_25 = []
    var_26 = []
    var_27 = {var_23: var_25, var_24: var_26}
    var_28 = 'os'
    var_29 = 'sys'
    var_30 = 'path'
    var_31 = [var_30]
    var_32 = 'argv'
    var_33 = [var_32]
    var_34 = {var_28: var_31, var_29: var_33}
    var_35 = {var_21: var_27, var_22: var_34}
    var_36 = {var_20: var_35}
    var_37 = {}
    var_38 = {}
    var_39 = module_0.ParsedContent()
    var_40 = 'import django\nimport flask\n\nfrom os import path\nfrom sys import argv\n'
    var_41 = module_1.sorted_imports(var_39)
    var_42 = True
    var_43 = 2
    var_44 = module_2.Config()
    var_45 = [var_0]
    var_46 = []
    var_47 = []
    var_48 = {var_23: var_46, var_24: var_47}
    var_49 = [var_30]
    var_50 = [var_32]
    var_51 = {var_28: var_49, var_29: var_50}
    var_52 = {var_21: var_48, var_22: var_51}
    var_53 = {var_20: var_52}
    var_54 = {}
    var_55 = {}
    var_56 = module_0.ParsedContent()
    var_57 = 'import django\nimport flask\n\nfrom os import path\n\nfrom sys import argv\n'
    var_58 = module_1.sorted_imports(var_56, var_44)
    var_59 = 'LOCALFOLDER'
    var_60 = [var_59]
    var_61 = module_2.Config()
    var_62 = [var_0]
    var_63 = []
    var_64 = {var_23: var_63}
    var_65 = {}
    var_66 = {var_21: var_64, var_22: var_65}
    var_67 = 'my_module'
    var_68 = []
    var_69 = {var_67: var_68}
    var_70 = {}
    var_71 = {var_21: var_69, var_22: var_70}
    var_72 = {var_20: var_66, var_59: var_71}
    var_73 = {}
    var_74 = {}
    var_75 = module_0.ParsedContent()
    var_76 = 'import django\n\nimport my_module\n'
    var_77 = module_1.sorted_imports(var_75, var_61)
    var_78 = 'thirdparty'
    var_79 = 'Third Party Imports'
    var_80 = {var_78: var_79}
    var_81 = True
    var_82 = module_2.Config()
    var_83 = [var_0]
    var_84 = []
    var_85 = {var_23: var_84}
    var_86 = [var_30]
    var_87 = {var_28: var_86}
    var_88 = {var_21: var_85, var_22: var_87}
    var_89 = {var_20: var_88}
    var_90 = {}
    var_91 = {}
    var_92 = module_0.ParsedContent()
    var_93 = '# Third Party Imports\nimport django\n\nfrom os import path\n'
    var_94 = module_1.sorted_imports(var_92, var_82)
    var_95 = True
    var_96 = module_2.Config()
    var_97 = [var_0]
    var_98 = {}
    var_99 = '*'
    var_100 = [var_99]
    var_101 = [var_32]
    var_102 = {var_28: var_100, var_29: var_101}
    var_103 = {var_21: var_98, var_22: var_102}
    var_104 = {var_20: var_103}
    var_105 = {}
    var_106 = {}
    var_107 = module_0.ParsedContent()
    var_108 = 'from os import *\nfrom sys import argv\n'
    var_109 = module_1.sorted_imports(var_107, var_96)
    var_110 = 'import os'
    var_111 = [var_110]
    var_112 = module_2.Config()
    var_113 = [var_0]
    var_114 = []
    var_115 = {var_28: var_114}
    var_116 = {}
    var_117 = {var_21: var_115, var_22: var_116}
    var_118 = {var_20: var_117}
    var_119 = {}
    var_120 = {}
    var_121 = module_0.ParsedContent()
    var_122 = ''
    var_123 = module_1.sorted_imports(var_121, var_112)
    var_124 = module_2.Config()
    var_125 = [var_10]
    var_126 = []
    var_127 = {var_28: var_126}
    var_128 = {}
    var_129 = {var_21: var_127, var_22: var_128}
    var_130 = {var_20: var_129}
    var_131 = {}
    var_132 = {}
    var_133 = module_0.ParsedContent()
    var_134 = "\n\nimport os\n\n\nprint('hello')\n"
    var_135 = module_1.sorted_imports(var_133, var_124)
    var_136 = '# Place imports here'
    var_137 = [var_136, var_10]
    var_138 = []
    var_139 = {var_28: var_138}
    var_140 = {}
    var_141 = {var_21: var_139, var_22: var_140}
    var_142 = {var_20: var_141}
    var_143 = [var_110]
    var_144 = {var_136: var_143}
    var_145 = {var_136: var_136}
    var_146 = module_0.ParsedContent()
    var_147 = "# Place imports here\nimport os\n\nprint('hello')\n"
    var_148 = module_1.sorted_imports(var_146)



# Parsed testcases at query #51
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = "print('hello')"
    var_2 = [var_0, var_1]
    var_3 = 'THIRDPARTY'
    var_4 = 'straight'
    var_5 = 'from'
    var_6 = 'os'
    var_7 = 'sys'
    var_8 = []
    var_9 = []
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'collections'
    var_12 = 'defaultdict'
    var_13 = [var_12]
    var_14 = {var_11: var_13}
    var_15 = {var_4: var_10, var_5: var_14}
    var_16 = {var_3: var_15}
    var_17 = 0
    var_18 = 2
    var_19 = '\n'
    var_20 = module_0.ParsedContent()
    var_21 = module_1.Config()
    var_22 = module_2.sorted_imports(var_20, var_21)
    assert var_22 == "\nos\nsys\n\nfrom collections import defaultdict\n\nprint('hello')"
    var_23 = [var_1]
    var_24 = {}
    var_25 = -1
    var_26 = 1
    var_27 = module_0.ParsedContent()
    var_28 = module_2.sorted_imports(var_27, var_21)
    assert var_28 == "print('hello')"
    var_29 = 'numpy'
    var_30 = [var_29]
    var_31 = module_1.Config()
    var_32 = [var_0, var_1]
    var_33 = []
    var_34 = []
    var_35 = {var_6: var_33, var_7: var_34}
    var_36 = [var_12]
    var_37 = {var_11: var_36}
    var_38 = {var_4: var_35, var_5: var_37}
    var_39 = []
    var_40 = {var_29: var_39}
    var_41 = {}
    var_42 = {var_4: var_40, var_5: var_41}
    var_43 = {var_3: var_38, var_29: var_42}
    var_44 = module_0.ParsedContent()
    var_45 = module_2.sorted_imports(var_44, var_31)
    var_46 = [var_7]
    var_47 = module_1.Config()
    var_48 = [var_0, var_1]
    var_49 = []
    var_50 = []
    var_51 = {var_6: var_49, var_7: var_50}
    var_52 = [var_12]
    var_53 = {var_11: var_52}
    var_54 = {var_4: var_51, var_5: var_53}
    var_55 = {var_3: var_54}
    var_56 = module_0.ParsedContent()
    var_57 = module_2.sorted_imports(var_56, var_47)
    var_58 = module_1.Config()
    var_59 = [var_0, var_1]
    var_60 = 'FUTURE'
    var_61 = '__future__'
    var_62 = 'annotations'
    var_63 = [var_62]
    var_64 = {var_61: var_63}
    var_65 = {}
    var_66 = {var_4: var_64, var_5: var_65}
    var_67 = []
    var_68 = {var_6: var_67}
    var_69 = {}
    var_70 = {var_4: var_68, var_5: var_69}
    var_71 = {var_60: var_66, var_3: var_70}
    var_72 = module_0.ParsedContent()
    var_73 = module_2.sorted_imports(var_72, var_58)
    var_74 = '\n\n'
    var_75 = 'thirdparty'
    var_76 = 'Third Party Imports'
    var_77 = {var_75: var_76}
    var_78 = module_1.Config()
    var_79 = [var_0, var_1]
    var_80 = []
    var_81 = {var_6: var_80}
    var_82 = {}
    var_83 = {var_4: var_81, var_5: var_82}
    var_84 = {var_3: var_83}
    var_85 = module_0.ParsedContent()
    var_86 = module_2.sorted_imports(var_85, var_78)



# Parsed testcases at query #52
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = '\n'
    var_5 = module_0.ParsedContent()
    var_6 = [var_0]
    var_7 = 'THIRDPARTY'
    var_8 = 'straight'
    var_9 = 'from'
    var_10 = 'os'
    var_11 = 'sys'
    var_12 = []
    var_13 = []
    var_14 = {var_10: var_12, var_11: var_13}
    var_15 = 'collections'
    var_16 = 'defaultdict'
    var_17 = [var_16]
    var_18 = {var_15: var_17}
    var_19 = {var_8: var_14, var_9: var_18}
    var_20 = {var_7: var_19}
    var_21 = 0
    var_22 = module_0.ParsedContent()
    var_23 = "\nfrom collections import defaultdict\n\nimport os\nimport sys\n\nprint('hello')\n"
    var_24 = [var_0]
    var_25 = []
    var_26 = []
    var_27 = {var_10: var_25, var_11: var_26}
    var_28 = [var_16]
    var_29 = {var_15: var_28}
    var_30 = {var_8: var_27, var_9: var_29}
    var_31 = {var_7: var_30}
    var_32 = module_0.ParsedContent()
    var_33 = True
    var_34 = module_1.Config()
    var_35 = module_2.sorted_imports(var_32, var_34)
    var_36 = "\nfrom collections import defaultdict\n\nimport sys\nimport os\n\nprint('hello')\n"
    var_37 = [var_0]
    var_38 = []
    var_39 = []
    var_40 = {var_10: var_38, var_11: var_39}
    var_41 = [var_16]
    var_42 = {var_15: var_41}
    var_43 = {var_8: var_40, var_9: var_42}
    var_44 = {var_7: var_43}
    var_45 = module_0.ParsedContent()
    var_46 = [var_10]
    var_47 = module_1.Config()
    var_48 = module_2.sorted_imports(var_45, var_47)
    var_49 = "\nimport os\n\nfrom collections import defaultdict\n\nimport sys\n\nprint('hello')\n"
    var_50 = [var_0]
    var_51 = 'FUTURE'
    var_52 = '__future__'
    var_53 = 'annotations'
    var_54 = [var_53]
    var_55 = {var_52: var_54}
    var_56 = {}
    var_57 = {var_8: var_55, var_9: var_56}
    var_58 = []
    var_59 = []
    var_60 = {var_10: var_58, var_11: var_59}
    var_61 = [var_16]
    var_62 = {var_15: var_61}
    var_63 = {var_8: var_60, var_9: var_62}
    var_64 = {var_51: var_57, var_7: var_63}
    var_65 = module_0.ParsedContent()
    var_66 = module_1.Config()
    var_67 = module_2.sorted_imports(var_65, var_66)
    var_68 = "\nfrom __future__ import annotations\n\nfrom collections import defaultdict\n\nimport os\nimport sys\n\nprint('hello')\n"
    var_69 = [var_0]
    var_70 = {}
    var_71 = 'module1'
    var_72 = 'module2'
    var_73 = 'module3'
    var_74 = '*'
    var_75 = [var_74]
    var_76 = 'func1'
    var_77 = [var_76]
    var_78 = [var_74]
    var_79 = {var_71: var_75, var_72: var_77, var_73: var_78}
    var_80 = {var_8: var_70, var_9: var_79}
    var_81 = {var_7: var_80}
    var_82 = module_0.ParsedContent()
    var_83 = module_1.Config()
    var_84 = module_2.sorted_imports(var_82, var_83)
    var_85 = "\nfrom module1 import *\nfrom module3 import *\nfrom module2 import func1\n\nprint('hello')\n"
    var_86 = [var_0]
    var_87 = []
    var_88 = []
    var_89 = {var_10: var_87, var_11: var_88}
    var_90 = [var_16]
    var_91 = {var_15: var_90}
    var_92 = {var_8: var_89, var_9: var_91}
    var_93 = {var_7: var_92}
    var_94 = module_0.ParsedContent()
    var_95 = 'thirdparty'
    var_96 = 'Third Party Imports'
    var_97 = {var_95: var_96}
    var_98 = module_1.Config()
    var_99 = module_2.sorted_imports(var_94, var_98)
    var_100 = "\n# Third Party Imports\nfrom collections import defaultdict\n\nimport os\nimport sys\n\nprint('hello')\n"
    var_101 = [var_0]
    var_102 = [var_53]
    var_103 = {var_52: var_102}
    var_104 = {}
    var_105 = {var_8: var_103, var_9: var_104}
    var_106 = []
    var_107 = []
    var_108 = {var_10: var_106, var_11: var_107}
    var_109 = [var_16]
    var_110 = {var_15: var_109}
    var_111 = {var_8: var_108, var_9: var_110}
    var_112 = {var_51: var_105, var_7: var_111}
    var_113 = module_0.ParsedContent()
    var_114 = 2
    var_115 = module_1.Config()
    var_116 = module_2.sorted_imports(var_113, var_115)
    var_117 = "\nfrom __future__ import annotations\n\n\n\nfrom collections import defaultdict\n\nimport os\nimport sys\n\nprint('hello')\n"
    var_118 = [var_0]
    var_119 = []
    var_120 = []
    var_121 = {var_10: var_119, var_11: var_120}
    var_122 = [var_16]
    var_123 = {var_15: var_122}
    var_124 = {var_8: var_121, var_9: var_123}
    var_125 = {var_7: var_124}
    var_126 = module_0.ParsedContent()
    var_127 = module_1.Config()
    var_128 = module_2.sorted_imports(var_126, var_127)
    var_129 = "\nfrom collections import defaultdict\n\nimport os\nimport sys\n\n\n\nprint('hello')\n"
    var_130 = [var_0]
    var_131 = []
    var_132 = []
    var_133 = {var_10: var_131, var_11: var_132}
    var_134 = [var_16]
    var_135 = {var_15: var_134}
    var_136 = {var_8: var_133, var_9: var_135}
    var_137 = {var_7: var_136}
    var_138 = module_0.ParsedContent()
    var_139 = [var_10]
    var_140 = module_1.Config()
    var_141 = module_2.sorted_imports(var_138, var_140)
    var_142 = "\nfrom collections import defaultdict\n\nimport sys\n\nprint('hello')\n"



# Parsed testcases at query #53
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'x = 1'
    var_1 = [var_0]
    var_2 = 'THIRDPARTY'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'zlib'
    var_6 = 'sys'
    var_7 = []
    var_8 = []
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'os'
    var_11 = 'path'
    var_12 = [var_11]
    var_13 = {var_10: var_12}
    var_14 = {var_3: var_9, var_4: var_13}
    var_15 = {var_2: var_14}
    var_16 = 0
    var_17 = 1
    var_18 = '\n'
    var_19 = module_0.ParsedContent()
    var_20 = module_1.Config()
    var_21 = module_2.sorted_imports(var_19, var_20)
    assert var_21 == 'sys\nzlib\n\nfrom os import path\n\nx = 1'
    var_22 = [var_0]
    var_23 = 'FUTURE'
    var_24 = '__future__'
    var_25 = []
    var_26 = {var_24: var_25}
    var_27 = {}
    var_28 = {var_3: var_26, var_4: var_27}
    var_29 = []
    var_30 = []
    var_31 = {var_5: var_29, var_6: var_30}
    var_32 = [var_11]
    var_33 = {var_10: var_32}
    var_34 = {var_3: var_31, var_4: var_33}
    var_35 = {var_23: var_28, var_2: var_34}
    var_36 = module_0.ParsedContent()
    var_37 = True
    var_38 = module_1.Config()
    var_39 = module_2.sorted_imports(var_36, var_38)
    assert var_39 == '__future__\nsys\nzlib\n\nfrom os import path\n\nx = 1'
    var_40 = [var_0]
    var_41 = []
    var_42 = []
    var_43 = {var_5: var_41, var_6: var_42}
    var_44 = [var_11]
    var_45 = {var_10: var_44}
    var_46 = {var_3: var_43, var_4: var_45}
    var_47 = {var_2: var_46}
    var_48 = module_0.ParsedContent()
    var_49 = 'FIRSTPARTY'
    var_50 = [var_49]
    var_51 = module_1.Config()
    var_52 = module_2.sorted_imports(var_48, var_51)
    assert var_52 == 'sys\nzlib\n\nfrom os import path\n\nx = 1'
    var_53 = [var_0]
    var_54 = []
    var_55 = []
    var_56 = {var_5: var_54, var_6: var_55}
    var_57 = [var_11]
    var_58 = {var_10: var_57}
    var_59 = {var_3: var_56, var_4: var_58}
    var_60 = {var_2: var_59}
    var_61 = module_0.ParsedContent()
    var_62 = [var_5]
    var_63 = module_1.Config()
    var_64 = module_2.sorted_imports(var_61, var_63)
    assert var_64 == 'sys\n\nfrom os import path\n\nx = 1'
    var_65 = [var_0]
    var_66 = {}
    var_67 = '*'
    var_68 = [var_67]
    var_69 = [var_11]
    var_70 = {var_10: var_68, var_6: var_69}
    var_71 = {var_3: var_66, var_4: var_70}
    var_72 = {var_2: var_71}
    var_73 = module_0.ParsedContent()
    var_74 = True
    var_75 = module_1.Config()
    var_76 = module_2.sorted_imports(var_73, var_75)
    assert var_76 == 'from os import *\nfrom sys import path\n\nx = 1'
    var_77 = [var_0]
    var_78 = []
    var_79 = {var_6: var_78}
    var_80 = [var_11]
    var_81 = {var_10: var_80}
    var_82 = {var_3: var_79, var_4: var_81}
    var_83 = {var_2: var_82}
    var_84 = module_0.ParsedContent()
    var_85 = True
    var_86 = module_1.Config()
    var_87 = module_2.sorted_imports(var_84, var_86)
    assert var_87 == 'from os import path\n\nsys\n\nx = 1'
    var_88 = [var_0]
    var_89 = []
    var_90 = {var_6: var_89}
    var_91 = [var_11]
    var_92 = {var_10: var_91}
    var_93 = {var_3: var_90, var_4: var_92}
    var_94 = {var_2: var_93}
    var_95 = module_0.ParsedContent()
    var_96 = 'thirdparty'
    var_97 = 'Third Party Imports'
    var_98 = {var_96: var_97}
    var_99 = module_1.Config()
    var_100 = module_2.sorted_imports(var_95, var_99)
    assert var_100 == '# Third Party Imports\nsys\n\nfrom os import path\n\nx = 1'
    var_101 = [var_0]
    var_102 = []
    var_103 = {var_24: var_102}
    var_104 = {}
    var_105 = {var_3: var_103, var_4: var_104}
    var_106 = []
    var_107 = {var_6: var_106}
    var_108 = [var_11]
    var_109 = {var_10: var_108}
    var_110 = {var_3: var_107, var_4: var_109}
    var_111 = {var_23: var_105, var_2: var_110}
    var_112 = module_0.ParsedContent()
    var_113 = 2
    var_114 = module_1.Config()
    var_115 = module_2.sorted_imports(var_112, var_114)
    assert var_115 == '__future__\n\n\nsys\n\nfrom os import path\n\nx = 1'
    var_116 = [var_0]
    var_117 = []
    var_118 = {var_6: var_117}
    var_119 = [var_11]
    var_120 = {var_10: var_119}
    var_121 = {var_3: var_118, var_4: var_120}
    var_122 = {var_2: var_121}
    var_123 = module_0.ParsedContent()
    var_124 = module_1.Config()
    var_125 = module_2.sorted_imports(var_123, var_124)
    assert var_125 == 'sys\n\nfrom os import path\n\n\n\nx = 1'
    var_126 = [var_0]
    var_127 = []
    var_128 = {var_6: var_127}
    var_129 = [var_11]
    var_130 = {var_10: var_129}
    var_131 = {var_3: var_128, var_4: var_130}
    var_132 = {var_2: var_131}
    var_133 = module_0.ParsedContent()
    var_134 = lambda x, y, z: x
    var_135 = module_1.Config()
    var_136 = module_2.sorted_imports(var_133, var_135)
    assert var_136 == 'sys\n\nfrom os import path\n\nx = 1'



# Parsed testcases at query #54
#--------------------------


import isort.parse as module_0
import isort.output as module_1
import isort.settings as module_2

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = '\n'
    var_5 = 1
    var_6 = {}
    var_7 = {}
    var_8 = module_0.ParsedContent()
    var_9 = module_1.sorted_imports(var_8)
    assert var_9 == "print('hello')\n"
    var_10 = [var_0]
    var_11 = 'THIRDPARTY'
    var_12 = 'straight'
    var_13 = 'from'
    var_14 = 'os'
    var_15 = 'sys'
    var_16 = 'os.path'
    var_17 = [var_16]
    var_18 = 'sys.argv'
    var_19 = [var_18]
    var_20 = {var_14: var_17, var_15: var_19}
    var_21 = 'collections'
    var_22 = 'defaultdict'
    var_23 = [var_22]
    var_24 = {var_21: var_23}
    var_25 = {var_12: var_20, var_13: var_24}
    var_26 = {var_11: var_25}
    var_27 = 0
    var_28 = 2
    var_29 = {}
    var_30 = {}
    var_31 = module_0.ParsedContent()
    var_32 = module_1.sorted_imports(var_31)
    var_33 = "import os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')\n"
    var_34 = True
    var_35 = module_2.Config()
    var_36 = [var_0]
    var_37 = [var_16]
    var_38 = [var_18]
    var_39 = {var_14: var_37, var_15: var_38}
    var_40 = [var_22]
    var_41 = {var_21: var_40}
    var_42 = {var_12: var_39, var_13: var_41}
    var_43 = {var_11: var_42}
    var_44 = {}
    var_45 = {}
    var_46 = module_0.ParsedContent()
    var_47 = module_1.sorted_imports(var_46, var_35)
    var_48 = "import sys\nimport os\n\nfrom collections import defaultdict\n\nprint('hello')\n"
    var_49 = 'LOCALFOLDER'
    var_50 = [var_49]
    var_51 = module_2.Config()
    var_52 = [var_0]
    var_53 = [var_16]
    var_54 = {var_14: var_53}
    var_55 = {}
    var_56 = {var_12: var_54, var_13: var_55}
    var_57 = 'my_module'
    var_58 = 'my_func'
    var_59 = [var_58]
    var_60 = {var_57: var_59}
    var_61 = {}
    var_62 = {var_12: var_60, var_13: var_61}
    var_63 = {var_11: var_56, var_49: var_62}
    var_64 = {}
    var_65 = {}
    var_66 = module_0.ParsedContent()
    var_67 = module_1.sorted_imports(var_66, var_51)
    var_68 = "import os\n\nimport my_module\n\nprint('hello')\n"
    var_69 = True
    var_70 = module_2.Config()
    var_71 = [var_0]
    var_72 = 'FUTURE'
    var_73 = '__future__'
    var_74 = 'annotations'
    var_75 = [var_74]
    var_76 = {var_73: var_75}
    var_77 = {}
    var_78 = {var_12: var_76, var_13: var_77}
    var_79 = [var_16]
    var_80 = {var_14: var_79}
    var_81 = [var_22]
    var_82 = {var_21: var_81}
    var_83 = {var_12: var_80, var_13: var_82}
    var_84 = {var_72: var_78, var_11: var_83}
    var_85 = {}
    var_86 = {}
    var_87 = module_0.ParsedContent()
    var_88 = module_1.sorted_imports(var_87, var_70)
    var_89 = "from __future__ import annotations\n\nimport os\nfrom collections import defaultdict\n\nprint('hello')\n"



# Parsed testcases at query #55
#--------------------------


import isort.parse as module_0
import isort.output as module_1
import isort.settings as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = 0
    var_5 = '\n'
    var_6 = {}
    var_7 = {}
    var_8 = module_0.ParsedContent()
    var_9 = module_1.sorted_imports(var_8)
    assert var_9 == '\n'
    var_10 = "print('hello')"
    var_11 = [var_10]
    var_12 = {}
    var_13 = -1
    var_14 = 1
    var_15 = {}
    var_16 = {}
    var_17 = module_0.ParsedContent()
    var_18 = module_1.sorted_imports(var_17)
    assert var_18 == "print('hello')\n"
    var_19 = [var_0]
    var_20 = 'THIRDPARTY'
    var_21 = 'straight'
    var_22 = 'from'
    var_23 = 'os'
    var_24 = 'sys'
    var_25 = [var_23]
    var_26 = [var_24]
    var_27 = {var_23: var_25, var_24: var_26}
    var_28 = 'collections'
    var_29 = 'OrderedDict'
    var_30 = [var_29]
    var_31 = {var_28: var_30}
    var_32 = {var_21: var_27, var_22: var_31}
    var_33 = {var_20: var_32}
    var_34 = {}
    var_35 = {}
    var_36 = module_0.ParsedContent()
    var_37 = 'import os\nimport sys\nfrom collections import OrderedDict\n\n'
    var_38 = module_1.sorted_imports(var_36)
    var_39 = 2
    var_40 = True
    var_41 = True
    var_42 = module_2.Config()
    var_43 = [var_0]
    var_44 = [var_23]
    var_45 = [var_24]
    var_46 = {var_23: var_44, var_24: var_45}
    var_47 = [var_29]
    var_48 = {var_28: var_47}
    var_49 = {var_21: var_46, var_22: var_48}
    var_50 = {var_20: var_49}
    var_51 = {}
    var_52 = {}
    var_53 = module_0.ParsedContent()
    var_54 = 'from collections import OrderedDict\n\nimport sys\nimport os\n\n'
    var_55 = module_1.sorted_imports(var_53, var_42)
    var_56 = [var_23]
    var_57 = True
    var_58 = module_2.Config()
    var_59 = [var_0]
    var_60 = [var_23]
    var_61 = [var_24]
    var_62 = {var_23: var_60, var_24: var_61}
    var_63 = [var_29]
    var_64 = {var_28: var_63}
    var_65 = {var_21: var_62, var_22: var_64}
    var_66 = {var_20: var_65}
    var_67 = {}
    var_68 = {}
    var_69 = module_0.ParsedContent()
    var_70 = 'import os\nimport sys\nfrom collections import OrderedDict\n\n'
    var_71 = module_1.sorted_imports(var_69, var_58)
    var_72 = 'thirdparty'
    var_73 = 'Third Party Imports'
    var_74 = {var_72: var_73}
    var_75 = 'End of Third Party Imports'
    var_76 = {var_72: var_75}
    var_77 = module_2.Config()
    var_78 = [var_0]
    var_79 = [var_23]
    var_80 = [var_24]
    var_81 = {var_23: var_79, var_24: var_80}
    var_82 = [var_29]
    var_83 = {var_28: var_82}
    var_84 = {var_21: var_81, var_22: var_83}
    var_85 = {var_20: var_84}
    var_86 = {}
    var_87 = {}
    var_88 = module_0.ParsedContent()
    var_89 = '# Third Party Imports\nimport os\nimport sys\nfrom collections import OrderedDict\n\n# End of Third Party Imports\n\n'
    var_90 = module_1.sorted_imports(var_88, var_77)
    var_91 = True
    var_92 = module_2.Config()
    var_93 = [var_0]
    var_94 = {}
    var_95 = '*'
    var_96 = [var_95]
    var_97 = 'path'
    var_98 = [var_97]
    var_99 = [var_29]
    var_100 = {var_23: var_96, var_24: var_98, var_28: var_99}
    var_101 = {var_21: var_94, var_22: var_100}
    var_102 = {var_20: var_101}
    var_103 = {}
    var_104 = {}
    var_105 = module_0.ParsedContent()
    var_106 = 'from os import *\nfrom collections import OrderedDict\nfrom sys import path\n\n'
    var_107 = module_1.sorted_imports(var_105, var_92)
    var_108 = [var_23]
    var_109 = module_2.Config()
    var_110 = [var_0]
    var_111 = [var_23]
    var_112 = [var_24]
    var_113 = {var_23: var_111, var_24: var_112}
    var_114 = [var_29]
    var_115 = {var_28: var_114}
    var_116 = {var_21: var_113, var_22: var_115}
    var_117 = {var_20: var_116}
    var_118 = {}
    var_119 = {}
    var_120 = module_0.ParsedContent()
    var_121 = 'import sys\nfrom collections import OrderedDict\n\n'
    var_122 = module_1.sorted_imports(var_120, var_109)
    var_123 = [var_0]
    var_124 = [var_23]
    var_125 = [var_24]
    var_126 = {var_23: var_124, var_24: var_125}
    var_127 = [var_29]
    var_128 = {var_28: var_127}
    var_129 = {var_21: var_126, var_22: var_128}
    var_130 = {var_20: var_129}
    var_131 = '\r\n'
    var_132 = {}
    var_133 = {}
    var_134 = module_0.ParsedContent()
    var_135 = 'import os\r\nimport sys\r\nfrom collections import OrderedDict\r\n\r\n'
    var_136 = module_1.sorted_imports(var_134, var_109)
    var_137 = [var_10]
    var_138 = [var_23]
    var_139 = [var_24]
    var_140 = {var_23: var_138, var_24: var_139}
    var_141 = [var_29]
    var_142 = {var_28: var_141}
    var_143 = {var_21: var_140, var_22: var_142}
    var_144 = {var_20: var_143}
    var_145 = 'import os'
    var_146 = 'import sys'
    var_147 = [var_145, var_146]
    var_148 = {var_20: var_147}
    var_149 = {var_10: var_20}
    var_150 = module_0.ParsedContent()
    var_151 = "import os\nimport sys\nprint('hello')\n"
    var_152 = module_1.sorted_imports(var_150)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = -1
    var_3 = '\n'
    var_4 = {}
    var_5 = {}
    var_6 = {}
    var_7 = 1
    var_8 = module_0.ParsedContent()
    var_9 = module_1.Config()
    var_10 = module_2.sorted_imports(var_8, var_9)
    assert var_10 == "print('hello')\n"
    var_11 = [var_0]
    var_12 = 0
    var_13 = 'THIRDPARTY'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = 'os'
    var_17 = 'sys'
    var_18 = []
    var_19 = []
    var_20 = {var_16: var_18, var_17: var_19}
    var_21 = 'collections'
    var_22 = 'defaultdict'
    var_23 = [var_22]
    var_24 = {var_21: var_23}
    var_25 = {var_14: var_20, var_15: var_24}
    var_26 = {var_13: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = 2
    var_30 = module_0.ParsedContent()
    var_31 = module_1.Config()
    var_32 = module_2.sorted_imports(var_30, var_31)
    var_33 = "import os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')\n"
    var_34 = [var_0]
    var_35 = 'FUTURE'
    var_36 = 'STDLIB'
    var_37 = '__future__'
    var_38 = 'annotations'
    var_39 = [var_38]
    var_40 = {var_37: var_39}
    var_41 = {}
    var_42 = {var_14: var_40, var_15: var_41}
    var_43 = []
    var_44 = {var_16: var_43}
    var_45 = 'argv'
    var_46 = [var_45]
    var_47 = {var_17: var_46}
    var_48 = {var_14: var_44, var_15: var_47}
    var_49 = {var_35: var_42, var_36: var_48}
    var_50 = {}
    var_51 = {}
    var_52 = module_0.ParsedContent()
    var_53 = 'future'
    var_54 = 'stdlib'
    var_55 = 'Future imports'
    var_56 = 'Standard library imports'
    var_57 = {var_53: var_55, var_54: var_56}
    var_58 = module_1.Config()
    var_59 = module_2.sorted_imports(var_52, var_58)
    var_60 = "# Future imports\nfrom __future__ import annotations\n\n# Standard library imports\nimport os\nfrom sys import argv\n\nprint('hello')\n"
    var_61 = [var_0]
    var_62 = 'pytest'
    var_63 = []
    var_64 = []
    var_65 = {var_16: var_63, var_62: var_64}
    var_66 = [var_22]
    var_67 = {var_21: var_66}
    var_68 = {var_14: var_65, var_15: var_67}
    var_69 = {var_13: var_68}
    var_70 = {}
    var_71 = {}
    var_72 = module_0.ParsedContent()
    var_73 = [var_62]
    var_74 = module_1.Config()
    var_75 = module_2.sorted_imports(var_72, var_74)
    var_76 = "import os\n\nfrom collections import defaultdict\n\nprint('hello')\n"
    var_77 = [var_0]
    var_78 = [var_38]
    var_79 = {var_37: var_78}
    var_80 = {}
    var_81 = {var_14: var_79, var_15: var_80}
    var_82 = []
    var_83 = {var_16: var_82}
    var_84 = [var_45]
    var_85 = {var_17: var_84}
    var_86 = {var_14: var_83, var_15: var_85}
    var_87 = {var_35: var_81, var_36: var_86}
    var_88 = {}
    var_89 = {}
    var_90 = module_0.ParsedContent()
    var_91 = True
    var_92 = module_1.Config()
    var_93 = module_2.sorted_imports(var_90, var_92)
    var_94 = "from __future__ import annotations\nimport os\nfrom sys import argv\n\nprint('hello')\n"



# Parsed testcases at query #2
#--------------------------


import isort.parse as module_0
import isort.output as module_1
import isort.settings as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = 0
    var_5 = '\n'
    var_6 = {}
    var_7 = {}
    var_8 = module_0.ParsedContent()
    var_9 = module_1.sorted_imports(var_8)
    assert var_9 == ''
    var_10 = "print('hello')"
    var_11 = [var_10]
    var_12 = {}
    var_13 = -1
    var_14 = 1
    var_15 = {}
    var_16 = {}
    var_17 = module_0.ParsedContent()
    var_18 = module_1.sorted_imports(var_17)
    assert var_18 == "print('hello')"
    var_19 = [var_0]
    var_20 = 'THIRDPARTY'
    var_21 = 'straight'
    var_22 = 'from'
    var_23 = 'os'
    var_24 = 'sys'
    var_25 = [var_23]
    var_26 = [var_24]
    var_27 = {var_23: var_25, var_24: var_26}
    var_28 = 'collections'
    var_29 = 'defaultdict'
    var_30 = [var_29]
    var_31 = {var_28: var_30}
    var_32 = {var_21: var_27, var_22: var_31}
    var_33 = {var_20: var_32}
    var_34 = {}
    var_35 = {}
    var_36 = module_0.ParsedContent()
    var_37 = False
    var_38 = module_2.Config()
    var_39 = 'import os\nimport sys\n\nfrom collections import defaultdict\n'
    var_40 = module_1.sorted_imports(var_36, var_38)
    var_41 = '# Main code'
    var_42 = [var_41]
    var_43 = 'FUTURE'
    var_44 = 'STDLIB'
    var_45 = '__future__'
    var_46 = 'annotations'
    var_47 = [var_46]
    var_48 = {var_45: var_47}
    var_49 = {var_21: var_48}
    var_50 = [var_23]
    var_51 = [var_24]
    var_52 = {var_23: var_50, var_24: var_51}
    var_53 = {var_21: var_52}
    var_54 = {var_43: var_49, var_44: var_53}
    var_55 = {}
    var_56 = {}
    var_57 = module_0.ParsedContent()
    var_58 = 'future'
    var_59 = 'stdlib'
    var_60 = 'Future'
    var_61 = 'Standard Library'
    var_62 = {var_58: var_60, var_59: var_61}
    var_63 = module_2.Config()
    var_64 = 'from __future__ import annotations\n\n# Future\n\nimport os\nimport sys\n\n# Standard Library\n\n# Main code\n'
    var_65 = module_1.sorted_imports(var_57, var_63)
    var_66 = [var_0]
    var_67 = [var_23]
    var_68 = [var_24]
    var_69 = {var_23: var_67, var_24: var_68}
    var_70 = [var_29]
    var_71 = {var_28: var_70}
    var_72 = {var_21: var_69, var_22: var_71}
    var_73 = {var_20: var_72}
    var_74 = {}
    var_75 = {}
    var_76 = module_0.ParsedContent()
    var_77 = [var_23]
    var_78 = module_2.Config()
    var_79 = 'import sys\n\nfrom collections import defaultdict\n'
    var_80 = module_1.sorted_imports(var_76, var_78)
    var_81 = [var_0]
    var_82 = [var_46]
    var_83 = {var_45: var_82}
    var_84 = {var_21: var_83}
    var_85 = [var_23]
    var_86 = {var_23: var_85}
    var_87 = {var_21: var_86}
    var_88 = 'django'
    var_89 = [var_88]
    var_90 = {var_88: var_89}
    var_91 = {var_21: var_90}
    var_92 = {var_43: var_84, var_44: var_87, var_20: var_91}
    var_93 = {}
    var_94 = {}
    var_95 = module_0.ParsedContent()
    var_96 = True
    var_97 = module_2.Config()
    var_98 = 'from __future__ import annotations\nimport django\nimport os\n'
    var_99 = module_1.sorted_imports(var_95, var_97)
    var_100 = [var_0]
    var_101 = 'numpy'
    var_102 = 'pandas'
    var_103 = '*'
    var_104 = [var_103]
    var_105 = 'DataFrame'
    var_106 = [var_105]
    var_107 = {var_101: var_104, var_102: var_106}
    var_108 = {var_22: var_107}
    var_109 = {var_20: var_108}
    var_110 = {}
    var_111 = {}
    var_112 = module_0.ParsedContent()
    var_113 = True
    var_114 = module_2.Config()
    var_115 = 'from numpy import *\nfrom pandas import DataFrame\n'
    var_116 = module_1.sorted_imports(var_112, var_114)
    var_117 = [var_0]
    var_118 = [var_23]
    var_119 = {var_23: var_118}
    var_120 = {var_21: var_119}
    var_121 = {var_20: var_120}
    var_122 = {}
    var_123 = {}
    var_124 = module_0.ParsedContent()
    var_125 = lambda x, y, z: x.upper()
    var_126 = module_2.Config()
    var_127 = 'IMPORT OS\n'
    var_128 = module_1.sorted_imports(var_124, var_126)



# Parsed testcases at query #3
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = 'x = 1'
    var_2 = [var_0, var_1]
    var_3 = 'THIRDPARTY'
    var_4 = 'straight'
    var_5 = 'from'
    var_6 = 'os'
    var_7 = 'sys'
    var_8 = []
    var_9 = []
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'collections'
    var_12 = 'defaultdict'
    var_13 = [var_12]
    var_14 = {var_11: var_13}
    var_15 = {var_4: var_10, var_5: var_14}
    var_16 = {var_3: var_15}
    var_17 = 0
    var_18 = 2
    var_19 = '\n'
    var_20 = {}
    var_21 = {}
    var_22 = module_0.ParsedContent()
    var_23 = module_1.Config()
    var_24 = module_2.sorted_imports(var_22, var_23)
    assert var_24 == 'from collections import defaultdict\n\nimport os\nimport sys\n\nx = 1\n'
    var_25 = [var_1]
    var_26 = {}
    var_27 = -1
    var_28 = 1
    var_29 = {}
    var_30 = {}
    var_31 = module_0.ParsedContent()
    var_32 = module_1.Config()
    var_33 = module_2.sorted_imports(var_31, var_32)
    assert var_33 == 'x = 1\n'
    var_34 = [var_0, var_1]
    var_35 = []
    var_36 = []
    var_37 = {var_6: var_35, var_7: var_36}
    var_38 = [var_12]
    var_39 = {var_11: var_38}
    var_40 = {var_4: var_37, var_5: var_39}
    var_41 = {var_3: var_40}
    var_42 = {}
    var_43 = {}
    var_44 = module_0.ParsedContent()
    var_45 = True
    var_46 = module_1.Config()
    var_47 = module_2.sorted_imports(var_44, var_46)
    assert var_47 == 'from collections import defaultdict\n\nimport os\nimport sys\n\nx = 1\n'
    var_48 = [var_0, var_1]
    var_49 = []
    var_50 = []
    var_51 = {var_6: var_49, var_7: var_50}
    var_52 = [var_12]
    var_53 = {var_11: var_52}
    var_54 = {var_4: var_51, var_5: var_53}
    var_55 = {var_3: var_54}
    var_56 = {}
    var_57 = {}
    var_58 = module_0.ParsedContent()
    var_59 = [var_6]
    var_60 = module_1.Config()
    var_61 = module_2.sorted_imports(var_58, var_60)
    assert var_61 == 'from collections import defaultdict\n\nimport os\nimport sys\n\nx = 1\n'
    var_62 = [var_0, var_1]
    var_63 = []
    var_64 = []
    var_65 = {var_6: var_63, var_7: var_64}
    var_66 = [var_12]
    var_67 = {var_11: var_66}
    var_68 = {var_4: var_65, var_5: var_67}
    var_69 = {var_3: var_68}
    var_70 = {}
    var_71 = {}
    var_72 = module_0.ParsedContent()
    var_73 = 'thirdparty'
    var_74 = 'Third Party Imports'
    var_75 = {var_73: var_74}
    var_76 = module_1.Config()
    var_77 = module_2.sorted_imports(var_72, var_76)
    assert var_77 == '# Third Party Imports\nfrom collections import defaultdict\n\nimport os\nimport sys\n\nx = 1\n'



# Parsed testcases at query #4
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'x = 1'
    var_1 = [var_0]
    var_2 = 'THIRDPARTY'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = 'sys'
    var_7 = set()
    var_8 = set()
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'collections'
    var_11 = 'defaultdict'
    var_12 = {var_11}
    var_13 = {var_10: var_12}
    var_14 = {var_3: var_9, var_4: var_13}
    var_15 = {var_2: var_14}
    var_16 = 0
    var_17 = 1
    var_18 = '\n'
    var_19 = module_0.ParsedContent()
    var_20 = module_1.Config()
    var_21 = module_2.sorted_imports(var_19, var_20)
    assert var_21 == 'import os\nimport sys\n\nfrom collections import defaultdict\n\nx = 1\n'
    var_22 = [var_0]
    var_23 = {}
    var_24 = -1
    var_25 = module_0.ParsedContent()
    var_26 = module_2.sorted_imports(var_25, var_20)
    assert var_26 == 'x = 1\n'
    var_27 = 'LOCALFOLDER'
    var_28 = [var_27]
    var_29 = module_1.Config()
    var_30 = [var_0]
    var_31 = 'local'
    var_32 = set()
    var_33 = {var_31: var_32}
    var_34 = {}
    var_35 = {var_3: var_33, var_4: var_34}
    var_36 = set()
    var_37 = {var_5: var_36}
    var_38 = {}
    var_39 = {var_3: var_37, var_4: var_38}
    var_40 = {var_27: var_35, var_2: var_39}
    var_41 = module_0.ParsedContent()
    var_42 = module_2.sorted_imports(var_41, var_29)
    var_43 = True
    var_44 = module_1.Config()
    var_45 = [var_0]
    var_46 = 'FUTURE'
    var_47 = '__future__'
    var_48 = 'print_function'
    var_49 = {var_48}
    var_50 = {var_47: var_49}
    var_51 = {}
    var_52 = {var_3: var_50, var_4: var_51}
    var_53 = set()
    var_54 = {var_5: var_53}
    var_55 = 'exit'
    var_56 = {var_55}
    var_57 = {var_6: var_56}
    var_58 = {var_3: var_54, var_4: var_57}
    var_59 = {var_46: var_52, var_2: var_58}
    var_60 = module_0.ParsedContent()
    var_61 = module_2.sorted_imports(var_60, var_44)
    var_62 = 'from sys import *'
    var_63 = [var_62]
    var_64 = module_1.Config()
    var_65 = [var_0]
    var_66 = set()
    var_67 = {var_5: var_66}
    var_68 = '*'
    var_69 = {var_68}
    var_70 = {var_6: var_69}
    var_71 = {var_3: var_67, var_4: var_70}
    var_72 = {var_2: var_71}
    var_73 = module_0.ParsedContent()
    var_74 = module_2.sorted_imports(var_73, var_64)
    var_75 = True
    var_76 = module_1.Config()
    var_77 = [var_0]
    var_78 = {}
    var_79 = {var_55}
    var_80 = {var_68}
    var_81 = {var_6: var_79, var_5: var_80}
    var_82 = {var_3: var_78, var_4: var_81}
    var_83 = {var_2: var_82}
    var_84 = module_0.ParsedContent()
    var_85 = module_2.sorted_imports(var_84, var_76)
    var_86 = 'from os import *'
    var_87 = 'from sys import exit'
    var_88 = True
    var_89 = module_1.Config()
    var_90 = [var_0]
    var_91 = set()
    var_92 = {var_5: var_91}
    var_93 = {var_55}
    var_94 = {var_6: var_93}
    var_95 = {var_3: var_92, var_4: var_94}
    var_96 = {var_2: var_95}
    var_97 = module_0.ParsedContent()
    var_98 = module_2.sorted_imports(var_97, var_89)
    var_99 = 'import os'
    var_100 = 'thirdparty'
    var_101 = 'Third Party Imports'
    var_102 = {var_100: var_101}
    var_103 = module_1.Config()
    var_104 = [var_0]
    var_105 = set()
    var_106 = {var_5: var_105}
    var_107 = {}
    var_108 = {var_3: var_106, var_4: var_107}
    var_109 = {var_2: var_108}
    var_110 = module_0.ParsedContent()
    var_111 = module_2.sorted_imports(var_110, var_103)
    var_112 = '# Third Party Imports'
    var_113 = 2
    var_114 = module_1.Config()
    var_115 = [var_0]
    var_116 = {var_48}
    var_117 = {var_47: var_116}
    var_118 = {}
    var_119 = {var_3: var_117, var_4: var_118}
    var_120 = set()
    var_121 = {var_5: var_120}
    var_122 = {}
    var_123 = {var_3: var_121, var_4: var_122}
    var_124 = {var_46: var_119, var_2: var_123}
    var_125 = module_0.ParsedContent()
    var_126 = module_2.sorted_imports(var_125, var_114)
    var_127 = 'from __future__ import print_function'
    var_128 = module_1.Config()
    var_129 = [var_0]
    var_130 = set()
    var_131 = {var_5: var_130}
    var_132 = {}
    var_133 = {var_3: var_131, var_4: var_132}
    var_134 = {var_2: var_133}
    var_135 = module_0.ParsedContent()
    var_136 = module_2.sorted_imports(var_135, var_128)



# Parsed testcases at query #5
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = -1
    var_3 = 0
    var_4 = '\n'
    var_5 = {}
    var_6 = {}
    var_7 = []
    var_8 = module_0.ParsedContent()
    var_9 = "print('hello')"
    var_10 = [var_9]
    var_11 = {}
    var_12 = -1
    var_13 = 1
    var_14 = {}
    var_15 = {}
    var_16 = []
    var_17 = module_0.ParsedContent()
    var_18 = [var_9]
    var_19 = 'THIRDPARTY'
    var_20 = 'straight'
    var_21 = 'from'
    var_22 = 'os'
    var_23 = 'sys'
    var_24 = [var_22]
    var_25 = [var_23]
    var_26 = {var_22: var_24, var_23: var_25}
    var_27 = 'collections'
    var_28 = 'OrderedDict'
    var_29 = 'defaultdict'
    var_30 = [var_28, var_29]
    var_31 = {var_27: var_30}
    var_32 = {var_20: var_26, var_21: var_31}
    var_33 = {var_19: var_32}
    var_34 = {}
    var_35 = {}
    var_36 = [var_19]
    var_37 = module_0.ParsedContent()
    var_38 = "from collections import OrderedDict, defaultdict\n\nimport os, sys\n\nprint('hello')"
    var_39 = [var_9]
    var_40 = [var_22]
    var_41 = [var_23]
    var_42 = {var_22: var_40, var_23: var_41}
    var_43 = [var_28, var_29]
    var_44 = {var_27: var_43}
    var_45 = {var_20: var_42, var_21: var_44}
    var_46 = {var_19: var_45}
    var_47 = {}
    var_48 = {}
    var_49 = [var_19]
    var_50 = module_0.ParsedContent()
    var_51 = 2
    var_52 = True
    var_53 = True
    var_54 = module_1.Config()
    var_55 = module_2.sorted_imports(var_50, var_54)
    var_56 = "from collections import OrderedDict, defaultdict\n\n\nimport sys, os\n\n\nprint('hello')"
    var_57 = [var_9]
    var_58 = 'FIRSTPARTY'
    var_59 = [var_22]
    var_60 = {var_22: var_59}
    var_61 = [var_28]
    var_62 = {var_27: var_61}
    var_63 = {var_20: var_60, var_21: var_62}
    var_64 = [var_23]
    var_65 = {var_23: var_64}
    var_66 = {}
    var_67 = {var_20: var_65, var_21: var_66}
    var_68 = {var_19: var_63, var_58: var_67}
    var_69 = {}
    var_70 = {}
    var_71 = [var_19, var_58]
    var_72 = module_0.ParsedContent()
    var_73 = 'LOCALFOLDER'
    var_74 = [var_73]
    var_75 = module_1.Config()
    var_76 = module_2.sorted_imports(var_72, var_75)
    var_77 = "from collections import OrderedDict\n\nimport os\n\n\nimport sys\n\nprint('hello')"



# Parsed testcases at query #6
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = '\n'
    var_5 = 1
    var_6 = {}
    var_7 = {}
    var_8 = module_0.ParsedContent()
    var_9 = module_1.Config()
    var_10 = module_2.sorted_imports(var_8, var_9)
    assert var_10 == "print('hello')\n"
    var_11 = [var_0]
    var_12 = 'THIRDPARTY'
    var_13 = 'straight'
    var_14 = 'from'
    var_15 = 'os'
    var_16 = 'sys'
    var_17 = 'os.path'
    var_18 = [var_17]
    var_19 = [var_16]
    var_20 = {var_15: var_18, var_16: var_19}
    var_21 = 'collections'
    var_22 = 'defaultdict'
    var_23 = [var_22]
    var_24 = {var_21: var_23}
    var_25 = {var_13: var_20, var_14: var_24}
    var_26 = {var_12: var_25}
    var_27 = 0
    var_28 = 2
    var_29 = {}
    var_30 = {}
    var_31 = module_0.ParsedContent()
    var_32 = module_1.Config()
    var_33 = module_2.sorted_imports(var_31, var_32)
    var_34 = "import os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')\n"
    var_35 = [var_0]
    var_36 = [var_17]
    var_37 = [var_16]
    var_38 = {var_15: var_36, var_16: var_37}
    var_39 = [var_22]
    var_40 = {var_21: var_39}
    var_41 = {var_13: var_38, var_14: var_40}
    var_42 = {var_12: var_41}
    var_43 = {}
    var_44 = {}
    var_45 = module_0.ParsedContent()
    var_46 = True
    var_47 = module_1.Config()
    var_48 = module_2.sorted_imports(var_45, var_47)
    var_49 = "from collections import defaultdict\n\n\nimport os\nimport sys\n\nprint('hello')\n"
    var_50 = [var_0]
    var_51 = [var_17]
    var_52 = [var_16]
    var_53 = {var_15: var_51, var_16: var_52}
    var_54 = [var_22]
    var_55 = {var_21: var_54}
    var_56 = {var_13: var_53, var_14: var_55}
    var_57 = {var_12: var_56}
    var_58 = {}
    var_59 = {}
    var_60 = module_0.ParsedContent()
    var_61 = 'thirdparty'
    var_62 = 'Third Party Imports'
    var_63 = {var_61: var_62}
    var_64 = module_1.Config()
    var_65 = module_2.sorted_imports(var_60, var_64)
    var_66 = "# Third Party Imports\nimport os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')\n"
    var_67 = [var_0]
    var_68 = [var_17]
    var_69 = [var_16]
    var_70 = {var_15: var_68, var_16: var_69}
    var_71 = [var_22]
    var_72 = {var_21: var_71}
    var_73 = {var_13: var_70, var_14: var_72}
    var_74 = {var_12: var_73}
    var_75 = {}
    var_76 = {}
    var_77 = module_0.ParsedContent()
    var_78 = [var_15]
    var_79 = module_1.Config()
    var_80 = module_2.sorted_imports(var_77, var_79)
    var_81 = "import sys\n\nfrom collections import defaultdict\n\nprint('hello')\n"



# Parsed testcases at query #7
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = -1
    var_3 = 0
    var_4 = '\n'
    var_5 = {}
    var_6 = {}
    var_7 = []
    var_8 = module_0.ParsedContent()
    var_9 = module_1.Config()
    var_10 = module_2.sorted_imports(var_8, var_9)
    assert var_10 == ''
    var_11 = "print('hello')"
    var_12 = [var_11]
    var_13 = {}
    var_14 = -1
    var_15 = 1
    var_16 = {}
    var_17 = {}
    var_18 = []
    var_19 = module_0.ParsedContent()
    var_20 = module_1.Config()
    var_21 = module_2.sorted_imports(var_19, var_20)
    assert var_21 == "print('hello')\n"
    var_22 = [var_11]
    var_23 = 'THIRDPARTY'
    var_24 = 'straight'
    var_25 = 'from'
    var_26 = 'os'
    var_27 = 'sys'
    var_28 = [var_26]
    var_29 = [var_27]
    var_30 = {var_26: var_28, var_27: var_29}
    var_31 = 'collections'
    var_32 = 'defaultdict'
    var_33 = 'OrderedDict'
    var_34 = [var_32, var_33]
    var_35 = {var_31: var_34}
    var_36 = {var_24: var_30, var_25: var_35}
    var_37 = {var_23: var_36}
    var_38 = {}
    var_39 = {}
    var_40 = [var_23]
    var_41 = module_0.ParsedContent()
    var_42 = module_1.Config()
    var_43 = module_2.sorted_imports(var_41, var_42)
    var_44 = "import os\nimport sys\n\nfrom collections import defaultdict, OrderedDict\n\nprint('hello')\n"
    var_45 = [var_11]
    var_46 = [var_26]
    var_47 = [var_27]
    var_48 = {var_26: var_46, var_27: var_47}
    var_49 = [var_32, var_33]
    var_50 = {var_31: var_49}
    var_51 = {var_24: var_48, var_25: var_50}
    var_52 = {var_23: var_51}
    var_53 = {}
    var_54 = {}
    var_55 = [var_23]
    var_56 = module_0.ParsedContent()
    var_57 = True
    var_58 = 2
    var_59 = module_1.Config()
    var_60 = module_2.sorted_imports(var_56, var_59)
    var_61 = "from collections import defaultdict, OrderedDict\n\nimport os\nimport sys\n\n\nprint('hello')\n"
    var_62 = [var_11]
    var_63 = 'FIRSTPARTY'
    var_64 = [var_26]
    var_65 = {var_26: var_64}
    var_66 = {}
    var_67 = {var_24: var_65, var_25: var_66}
    var_68 = [var_27]
    var_69 = {var_27: var_68}
    var_70 = {}
    var_71 = {var_24: var_69, var_25: var_70}
    var_72 = {var_23: var_67, var_63: var_71}
    var_73 = {}
    var_74 = {}
    var_75 = [var_23, var_63]
    var_76 = module_0.ParsedContent()
    var_77 = [var_63]
    var_78 = module_1.Config()
    var_79 = module_2.sorted_imports(var_76, var_78)
    var_80 = "import os\n\nimport sys\n\nprint('hello')\n"



# Parsed testcases at query #8
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = -1
    var_3 = 0
    var_4 = '\n'
    var_5 = {}
    var_6 = {}
    var_7 = []
    var_8 = module_0.ParsedContent()
    var_9 = module_1.Config()
    var_10 = module_2.sorted_imports(var_8, var_9)
    assert var_10 == ''
    var_11 = "print('hello')"
    var_12 = [var_11]
    var_13 = {}
    var_14 = -1
    var_15 = 1
    var_16 = {}
    var_17 = {}
    var_18 = []
    var_19 = module_0.ParsedContent()
    var_20 = module_1.Config()
    var_21 = module_2.sorted_imports(var_19, var_20)
    assert var_21 == "print('hello')"
    var_22 = []
    var_23 = 'THIRDPARTY'
    var_24 = 'straight'
    var_25 = 'from'
    var_26 = 'zlib'
    var_27 = 'os'
    var_28 = []
    var_29 = []
    var_30 = {var_26: var_28, var_27: var_29}
    var_31 = 'sys'
    var_32 = 'path'
    var_33 = [var_32]
    var_34 = {var_31: var_33}
    var_35 = {var_24: var_30, var_25: var_34}
    var_36 = {var_23: var_35}
    var_37 = {}
    var_38 = {}
    var_39 = [var_23]
    var_40 = module_0.ParsedContent()
    var_41 = module_1.Config()
    var_42 = 'import os\nimport zlib\nfrom sys import path\n'
    var_43 = module_2.sorted_imports(var_40, var_41)
    var_44 = []
    var_45 = []
    var_46 = []
    var_47 = {var_26: var_45, var_27: var_46}
    var_48 = [var_32]
    var_49 = {var_31: var_48}
    var_50 = {var_24: var_47, var_25: var_49}
    var_51 = {var_23: var_50}
    var_52 = {}
    var_53 = {}
    var_54 = [var_23]
    var_55 = module_0.ParsedContent()
    var_56 = True
    var_57 = module_1.Config()
    var_58 = 'import zlib\nimport os\nfrom sys import path\n'
    var_59 = module_2.sorted_imports(var_55, var_57)
    var_60 = []
    var_61 = 'FIRSTPARTY'
    var_62 = []
    var_63 = {var_26: var_62}
    var_64 = {}
    var_65 = {var_24: var_63, var_25: var_64}
    var_66 = []
    var_67 = {var_27: var_66}
    var_68 = {}
    var_69 = {var_24: var_67, var_25: var_68}
    var_70 = {var_23: var_65, var_61: var_69}
    var_71 = {}
    var_72 = {}
    var_73 = [var_23, var_61]
    var_74 = module_0.ParsedContent()
    var_75 = [var_61]
    var_76 = module_1.Config()
    var_77 = 'import zlib\n\nimport os\n'
    var_78 = module_2.sorted_imports(var_74, var_76)
    var_79 = []
    var_80 = []
    var_81 = {var_26: var_80}
    var_82 = {}
    var_83 = {var_24: var_81, var_25: var_82}
    var_84 = {var_23: var_83}
    var_85 = {}
    var_86 = {}
    var_87 = [var_23]
    var_88 = module_0.ParsedContent()
    var_89 = 'thirdparty'
    var_90 = 'Third Party Imports'
    var_91 = {var_89: var_90}
    var_92 = module_1.Config()
    var_93 = '# Third Party Imports\nimport zlib\n'
    var_94 = module_2.sorted_imports(var_88, var_92)
    var_95 = []
    var_96 = []
    var_97 = {var_26: var_96}
    var_98 = {}
    var_99 = {var_24: var_97, var_25: var_98}
    var_100 = []
    var_101 = {var_27: var_100}
    var_102 = {}
    var_103 = {var_24: var_101, var_25: var_102}
    var_104 = {var_23: var_99, var_61: var_103}
    var_105 = {}
    var_106 = {}
    var_107 = [var_23, var_61]
    var_108 = module_0.ParsedContent()
    var_109 = 2
    var_110 = module_1.Config()
    var_111 = 'import zlib\n\n\nimport os\n'
    var_112 = module_2.sorted_imports(var_108, var_110)
    var_113 = []
    var_114 = {}
    var_115 = '*'
    var_116 = [var_115]
    var_117 = [var_32]
    var_118 = {var_31: var_116, var_27: var_117}
    var_119 = {var_24: var_114, var_25: var_118}
    var_120 = {var_23: var_119}
    var_121 = {}
    var_122 = {}
    var_123 = [var_23]
    var_124 = module_0.ParsedContent()
    var_125 = True
    var_126 = module_1.Config()
    var_127 = 'from sys import *\nfrom os import path\n'
    var_128 = module_2.sorted_imports(var_124, var_126)
    var_129 = []
    var_130 = []
    var_131 = {var_26: var_130}
    var_132 = {}
    var_133 = {var_24: var_131, var_25: var_132}
    var_134 = []
    var_135 = {var_27: var_134}
    var_136 = {}
    var_137 = {var_24: var_135, var_25: var_136}
    var_138 = {var_23: var_133, var_61: var_137}
    var_139 = {}
    var_140 = {}
    var_141 = [var_23, var_61]
    var_142 = module_0.ParsedContent()
    var_143 = True
    var_144 = module_1.Config()
    var_145 = 'import os\nimport zlib\n'
    var_146 = module_2.sorted_imports(var_142, var_144)
    var_147 = []
    var_148 = []
    var_149 = {var_26: var_148}
    var_150 = {}
    var_151 = {var_24: var_149, var_25: var_150}
    var_152 = {var_23: var_151}
    var_153 = {}
    var_154 = {}
    var_155 = [var_23]
    var_156 = module_0.ParsedContent()
    var_157 = lambda code, ext, cfg: code.upper()
    var_158 = module_1.Config()
    var_159 = 'IMPORT ZLIB\n'
    var_160 = module_2.sorted_imports(var_156, var_158)



# Parsed testcases at query #9
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 'STDLIB'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = 'sys'
    var_7 = []
    var_8 = []
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {}
    var_11 = {var_3: var_9, var_4: var_10}
    var_12 = {var_2: var_11}
    var_13 = 0
    var_14 = 1
    var_15 = '\n'
    var_16 = {}
    var_17 = {}
    var_18 = module_0.ParsedContent()
    var_19 = module_1.Config()
    var_20 = module_2.sorted_imports(var_18, var_19)
    assert var_20 == 'import os\nimport sys\n'
    var_21 = [var_0]
    var_22 = 'FUTURE'
    var_23 = '__future__'
    var_24 = []
    var_25 = {var_23: var_24}
    var_26 = {}
    var_27 = {var_3: var_25, var_4: var_26}
    var_28 = []
    var_29 = []
    var_30 = {var_5: var_28, var_6: var_29}
    var_31 = {}
    var_32 = {var_3: var_30, var_4: var_31}
    var_33 = {var_22: var_27, var_2: var_32}
    var_34 = {}
    var_35 = {}
    var_36 = module_0.ParsedContent()
    var_37 = module_1.Config()
    var_38 = module_2.sorted_imports(var_36, var_37)
    assert var_38 == 'from __future__ import absolute_import\n\nimport os\nimport sys\n'
    var_39 = [var_0]
    var_40 = []
    var_41 = {var_23: var_40}
    var_42 = {}
    var_43 = {var_3: var_41, var_4: var_42}
    var_44 = []
    var_45 = []
    var_46 = {var_5: var_44, var_6: var_45}
    var_47 = {}
    var_48 = {var_3: var_46, var_4: var_47}
    var_49 = {var_22: var_43, var_2: var_48}
    var_50 = {}
    var_51 = {}
    var_52 = module_0.ParsedContent()
    var_53 = True
    var_54 = module_1.Config()
    var_55 = module_2.sorted_imports(var_52, var_54)
    assert var_55 == 'from __future__ import absolute_import\nimport os\nimport sys\n'
    var_56 = [var_0]
    var_57 = []
    var_58 = {var_5: var_57}
    var_59 = 'path'
    var_60 = 'sys.path'
    var_61 = (var_59, var_60)
    var_62 = [var_61]
    var_63 = {var_6: var_62}
    var_64 = {var_3: var_58, var_4: var_63}
    var_65 = {var_2: var_64}
    var_66 = {}
    var_67 = {}
    var_68 = module_0.ParsedContent()
    var_69 = True
    var_70 = module_1.Config()
    var_71 = module_2.sorted_imports(var_68, var_70)
    assert var_71 == 'from sys import path\n\nimport os\n'
    var_72 = [var_0]
    var_73 = {}
    var_74 = '*'
    var_75 = None
    var_76 = (var_74, var_75)
    var_77 = [var_76]
    var_78 = (var_59, var_60)
    var_79 = [var_78]
    var_80 = {var_5: var_77, var_6: var_79}
    var_81 = {var_3: var_73, var_4: var_80}
    var_82 = {var_2: var_81}
    var_83 = {}
    var_84 = {}
    var_85 = module_0.ParsedContent()
    var_86 = True
    var_87 = module_1.Config()
    var_88 = module_2.sorted_imports(var_85, var_87)
    assert var_88 == 'from os import *\nfrom sys import path\n'
    var_89 = [var_0]
    var_90 = []
    var_91 = []
    var_92 = {var_5: var_90, var_6: var_91}
    var_93 = {}
    var_94 = {var_3: var_92, var_4: var_93}
    var_95 = {var_2: var_94}
    var_96 = {}
    var_97 = {}
    var_98 = module_0.ParsedContent()
    var_99 = [var_5]
    var_100 = module_1.Config()
    var_101 = module_2.sorted_imports(var_98, var_100)
    assert var_101 == 'import sys\n'
    var_102 = "print('hello')"
    var_103 = [var_102]
    var_104 = {}
    var_105 = -1
    var_106 = {}
    var_107 = {}
    var_108 = module_0.ParsedContent()
    var_109 = module_1.Config()
    var_110 = module_2.sorted_imports(var_108, var_109)
    assert var_110 == "print('hello')\n"



# Parsed testcases at query #10
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = '\n'
    var_5 = 1
    var_6 = {}
    var_7 = {}
    var_8 = module_0.ParsedContent()
    var_9 = module_1.Config()
    var_10 = module_2.sorted_imports(var_8, var_9)
    assert var_10 == "print('hello')"
    var_11 = [var_0]
    var_12 = 'THIRDPARTY'
    var_13 = 'straight'
    var_14 = 'from'
    var_15 = 'os'
    var_16 = 'os.path'
    var_17 = [var_16]
    var_18 = {var_15: var_17}
    var_19 = 'sys'
    var_20 = 'sys.argv'
    var_21 = [var_20]
    var_22 = {var_19: var_21}
    var_23 = {var_13: var_18, var_14: var_22}
    var_24 = {var_12: var_23}
    var_25 = 0
    var_26 = 2
    var_27 = {}
    var_28 = {}
    var_29 = module_0.ParsedContent()
    var_30 = module_1.Config()
    var_31 = module_2.sorted_imports(var_29, var_30)
    var_32 = "\nimport os\n\nfrom sys import sys.argv\n\nprint('hello')"
    var_33 = [var_0]
    var_34 = [var_16]
    var_35 = 'sys.platform'
    var_36 = [var_35]
    var_37 = {var_15: var_34, var_19: var_36}
    var_38 = [var_20]
    var_39 = 'os.getcwd'
    var_40 = [var_39]
    var_41 = {var_19: var_38, var_15: var_40}
    var_42 = {var_13: var_37, var_14: var_41}
    var_43 = {var_12: var_42}
    var_44 = {}
    var_45 = {}
    var_46 = module_0.ParsedContent()
    var_47 = True
    var_48 = module_1.Config()
    var_49 = module_2.sorted_imports(var_46, var_48)
    var_50 = "\nfrom os import os.getcwd\nfrom sys import sys.argv\nimport os, sys\n\nprint('hello')"
    var_51 = [var_0]
    var_52 = [var_16]
    var_53 = [var_35]
    var_54 = {var_15: var_52, var_19: var_53}
    var_55 = [var_20]
    var_56 = [var_39]
    var_57 = {var_19: var_55, var_15: var_56}
    var_58 = {var_13: var_54, var_14: var_57}
    var_59 = {var_12: var_58}
    var_60 = {}
    var_61 = {}
    var_62 = module_0.ParsedContent()
    var_63 = [var_15]
    var_64 = module_1.Config()
    var_65 = module_2.sorted_imports(var_62, var_64)
    var_66 = "\nimport sys\n\nfrom sys import sys.argv\n\nprint('hello')"
    var_67 = [var_0]
    var_68 = [var_16]
    var_69 = {var_15: var_68}
    var_70 = [var_20]
    var_71 = {var_19: var_70}
    var_72 = {var_13: var_69, var_14: var_71}
    var_73 = {var_12: var_72}
    var_74 = {}
    var_75 = {}
    var_76 = module_0.ParsedContent()
    var_77 = 'thirdparty'
    var_78 = 'Third Party Imports'
    var_79 = {var_77: var_78}
    var_80 = module_1.Config()
    var_81 = module_2.sorted_imports(var_76, var_80)
    var_82 = "\n# Third Party Imports\nimport os\n\nfrom sys import sys.argv\n\nprint('hello')"
    var_83 = [var_0]
    var_84 = 'FUTURE'
    var_85 = 'from __future__'
    var_86 = 'print_function'
    var_87 = [var_86]
    var_88 = {var_85: var_87}
    var_89 = {}
    var_90 = {var_13: var_88, var_14: var_89}
    var_91 = [var_16]
    var_92 = {var_15: var_91}
    var_93 = [var_20]
    var_94 = {var_19: var_93}
    var_95 = {var_13: var_92, var_14: var_94}
    var_96 = {var_84: var_90, var_12: var_95}
    var_97 = {}
    var_98 = {}
    var_99 = module_0.ParsedContent()
    var_100 = module_1.Config()
    var_101 = module_2.sorted_imports(var_99, var_100)
    var_102 = "\nfrom __future__ import print_function\n\n\n\nimport os\n\nfrom sys import sys.argv\n\nprint('hello')"
    var_103 = '# PLACE_IMPORTS_HERE'
    var_104 = [var_0, var_103]
    var_105 = [var_16]
    var_106 = {var_15: var_105}
    var_107 = [var_20]
    var_108 = {var_19: var_107}
    var_109 = {var_13: var_106, var_14: var_108}
    var_110 = {var_12: var_109}
    var_111 = 'PLACE_IMPORTS_HERE'
    var_112 = 'import os'
    var_113 = 'from sys import sys.argv'
    var_114 = [var_112, var_113]
    var_115 = {var_111: var_114}
    var_116 = {var_103: var_111}
    var_117 = module_0.ParsedContent()
    var_118 = module_1.Config()
    var_119 = module_2.sorted_imports(var_117, var_118)
    var_120 = "print('hello')\n# PLACE_IMPORTS_HERE\nimport os\nfrom sys import sys.argv\n"
    var_121 = [var_0]
    var_122 = 'FIRSTPARTY'
    var_123 = [var_86]
    var_124 = {var_85: var_123}
    var_125 = {}
    var_126 = {var_13: var_124, var_14: var_125}
    var_127 = [var_16]
    var_128 = {var_15: var_127}
    var_129 = [var_20]
    var_130 = {var_19: var_129}
    var_131 = {var_13: var_128, var_14: var_130}
    var_132 = 'my_module'
    var_133 = 'my_function'
    var_134 = [var_133]
    var_135 = {var_132: var_134}
    var_136 = {}
    var_137 = {var_13: var_135, var_14: var_136}
    var_138 = {var_84: var_126, var_12: var_131, var_122: var_137}
    var_139 = {}
    var_140 = {}
    var_141 = module_0.ParsedContent()
    var_142 = True
    var_143 = module_1.Config()
    var_144 = module_2.sorted_imports(var_141, var_143)
    var_145 = "\nfrom __future__ import print_function\nimport my_module, os\n\nfrom sys import sys.argv\n\nprint('hello')"
    var_146 = [var_0]
    var_147 = {}
    var_148 = '*'
    var_149 = [var_148]
    var_150 = [var_20]
    var_151 = {var_15: var_149, var_19: var_150}
    var_152 = {var_13: var_147, var_14: var_151}
    var_153 = {var_12: var_152}
    var_154 = {}
    var_155 = {}
    var_156 = module_0.ParsedContent()
    var_157 = True
    var_158 = module_1.Config()
    var_159 = module_2.sorted_imports(var_156, var_158)
    var_160 = "\nfrom os import *\nfrom sys import sys.argv\n\nprint('hello')"
    var_161 = [var_0]
    var_162 = [var_16]
    var_163 = {var_15: var_162}
    var_164 = [var_20]
    var_165 = {var_19: var_164}
    var_166 = {var_13: var_163, var_14: var_165}
    var_167 = {var_12: var_166}
    var_168 = {}
    var_169 = {}
    var_170 = module_0.ParsedContent()
    var_171 = module_2.sorted_imports(var_170, var_158)
    var_172 = "\r\nimport os\r\n\r\nfrom sys import sys.argv\r\n\r\nprint('hello')"



# Parsed testcases at query #11
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = {}
    var_3 = []
    var_4 = -1
    var_5 = 0
    var_6 = '\n'
    var_7 = {}
    var_8 = {}
    var_9 = module_0.ParsedContent()
    var_10 = "print('hello')"
    var_11 = [var_10]
    var_12 = {}
    var_13 = []
    var_14 = -1
    var_15 = 1
    var_16 = {}
    var_17 = {}
    var_18 = module_0.ParsedContent()
    var_19 = [var_0]
    var_20 = 'THIRDPARTY'
    var_21 = 'straight'
    var_22 = 'from'
    var_23 = 'numpy'
    var_24 = 'pandas'
    var_25 = [var_23]
    var_26 = [var_24]
    var_27 = {var_23: var_25, var_24: var_26}
    var_28 = {}
    var_29 = {var_21: var_27, var_22: var_28}
    var_30 = {var_20: var_29}
    var_31 = [var_20]
    var_32 = {}
    var_33 = {}
    var_34 = module_0.ParsedContent()
    var_35 = [var_0]
    var_36 = 'FUTURE'
    var_37 = 'STDLIB'
    var_38 = '__future__'
    var_39 = [var_38]
    var_40 = {var_38: var_39}
    var_41 = {}
    var_42 = {var_21: var_40, var_22: var_41}
    var_43 = 'os'
    var_44 = 'sys'
    var_45 = [var_43]
    var_46 = [var_44]
    var_47 = {var_43: var_45, var_44: var_46}
    var_48 = {}
    var_49 = {var_21: var_47, var_22: var_48}
    var_50 = [var_23]
    var_51 = {var_23: var_50}
    var_52 = 'DataFrame'
    var_53 = [var_52]
    var_54 = {var_24: var_53}
    var_55 = {var_21: var_51, var_22: var_54}
    var_56 = {var_36: var_42, var_37: var_49, var_20: var_55}
    var_57 = [var_36, var_37, var_20]
    var_58 = {}
    var_59 = {}
    var_60 = module_0.ParsedContent()
    var_61 = [var_0]
    var_62 = [var_23]
    var_63 = [var_24]
    var_64 = {var_23: var_62, var_24: var_63}
    var_65 = {}
    var_66 = {var_21: var_64, var_22: var_65}
    var_67 = {var_20: var_66}
    var_68 = [var_20]
    var_69 = {}
    var_70 = {}
    var_71 = module_0.ParsedContent()
    var_72 = True
    var_73 = module_1.Config()
    var_74 = module_2.sorted_imports(var_71, var_73)
    assert var_74 == 'import pandas\nimport numpy\n\n'
    var_75 = [var_0]
    var_76 = 'FIRSTPARTY'
    var_77 = [var_23]
    var_78 = {var_23: var_77}
    var_79 = {}
    var_80 = {var_21: var_78, var_22: var_79}
    var_81 = 'my_module'
    var_82 = [var_81]
    var_83 = {var_81: var_82}
    var_84 = {}
    var_85 = {var_21: var_83, var_22: var_84}
    var_86 = {var_20: var_80, var_76: var_85}
    var_87 = [var_20]
    var_88 = {}
    var_89 = {}
    var_90 = module_0.ParsedContent()
    var_91 = [var_76]
    var_92 = module_1.Config()
    var_93 = module_2.sorted_imports(var_90, var_92)
    assert var_93 == 'import numpy\n\nimport my_module\n\n'
    var_94 = [var_0]
    var_95 = [var_38]
    var_96 = {var_38: var_95}
    var_97 = {}
    var_98 = {var_21: var_96, var_22: var_97}
    var_99 = [var_23]
    var_100 = {var_23: var_99}
    var_101 = [var_52]
    var_102 = {var_24: var_101}
    var_103 = {var_21: var_100, var_22: var_102}
    var_104 = {var_36: var_98, var_20: var_103}
    var_105 = [var_36, var_20]
    var_106 = {}
    var_107 = {}
    var_108 = module_0.ParsedContent()
    var_109 = True
    var_110 = module_1.Config()
    var_111 = module_2.sorted_imports(var_108, var_110)
    assert var_111 == 'from __future__ import absolute_import\n\nimport numpy\n\nfrom pandas import DataFrame\n\n'
    var_112 = [var_0]
    var_113 = [var_23]
    var_114 = {var_23: var_113}
    var_115 = {}
    var_116 = {var_21: var_114, var_22: var_115}
    var_117 = {var_20: var_116}
    var_118 = [var_20]
    var_119 = {}
    var_120 = {}
    var_121 = module_0.ParsedContent()
    var_122 = 'thirdparty'
    var_123 = 'Third Party Imports'
    var_124 = {var_122: var_123}
    var_125 = module_1.Config()
    var_126 = module_2.sorted_imports(var_121, var_125)
    assert var_126 == '# Third Party Imports\nimport numpy\n\n'
    var_127 = [var_0]
    var_128 = [var_43]
    var_129 = {var_43: var_128}
    var_130 = {}
    var_131 = {var_21: var_129, var_22: var_130}
    var_132 = [var_23]
    var_133 = {var_23: var_132}
    var_134 = {}
    var_135 = {var_21: var_133, var_22: var_134}
    var_136 = {var_37: var_131, var_20: var_135}
    var_137 = [var_37, var_20]
    var_138 = {}
    var_139 = {}
    var_140 = module_0.ParsedContent()
    var_141 = 2
    var_142 = module_1.Config()
    var_143 = module_2.sorted_imports(var_140, var_142)
    assert var_143 == 'import os\n\n\n\nimport numpy\n\n'
    var_144 = [var_0]
    var_145 = [var_23]
    var_146 = [var_24]
    var_147 = {var_23: var_145, var_24: var_146}
    var_148 = {}
    var_149 = {var_21: var_147, var_22: var_148}
    var_150 = {var_20: var_149}
    var_151 = [var_20]
    var_152 = {}
    var_153 = {}
    var_154 = module_0.ParsedContent()
    var_155 = [var_24]
    var_156 = module_1.Config()
    var_157 = module_2.sorted_imports(var_154, var_156)
    assert var_157 == 'import numpy\n\n'



# Parsed testcases at query #12
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = 0
    var_5 = '\n'
    var_6 = {}
    var_7 = {}
    var_8 = module_0.ParsedContent()
    var_9 = "print('hello')"
    var_10 = [var_9]
    var_11 = {}
    var_12 = -1
    var_13 = 1
    var_14 = {}
    var_15 = {}
    var_16 = module_0.ParsedContent()
    var_17 = [var_0]
    var_18 = 'THIRDPARTY'
    var_19 = 'straight'
    var_20 = 'from'
    var_21 = 'zlib'
    var_22 = 'os'
    var_23 = []
    var_24 = []
    var_25 = {var_21: var_23, var_22: var_24}
    var_26 = 'sys'
    var_27 = 'json'
    var_28 = 'path'
    var_29 = [var_28]
    var_30 = 'load'
    var_31 = [var_30]
    var_32 = {var_26: var_29, var_27: var_31}
    var_33 = {var_19: var_25, var_20: var_32}
    var_34 = {var_18: var_33}
    var_35 = {}
    var_36 = {}
    var_37 = module_0.ParsedContent()
    var_38 = 'import os\nimport zlib\n\nfrom json import load\nfrom sys import path\n\n'
    var_39 = [var_0]
    var_40 = []
    var_41 = []
    var_42 = {var_21: var_40, var_22: var_41}
    var_43 = [var_28]
    var_44 = [var_30]
    var_45 = {var_26: var_43, var_27: var_44}
    var_46 = {var_19: var_42, var_20: var_45}
    var_47 = {var_18: var_46}
    var_48 = {}
    var_49 = {}
    var_50 = module_0.ParsedContent()
    var_51 = 2
    var_52 = True
    var_53 = module_1.Config()
    var_54 = module_2.sorted_imports(var_50, var_53)
    var_55 = 'from json import load\nfrom sys import path\n\nimport os\nimport zlib\n\n\n'
    var_56 = [var_0]
    var_57 = 'FIRSTPARTY'
    var_58 = []
    var_59 = {var_21: var_58}
    var_60 = [var_28]
    var_61 = {var_26: var_60}
    var_62 = {var_19: var_59, var_20: var_61}
    var_63 = []
    var_64 = {var_22: var_63}
    var_65 = [var_30]
    var_66 = {var_27: var_65}
    var_67 = {var_19: var_64, var_20: var_66}
    var_68 = {var_18: var_62, var_57: var_67}
    var_69 = {}
    var_70 = {}
    var_71 = module_0.ParsedContent()
    var_72 = [var_57]
    var_73 = module_1.Config()
    var_74 = module_2.sorted_imports(var_71, var_73)
    var_75 = 'import zlib\n\nfrom sys import path\n\n\nimport os\n\nfrom json import load\n\n'
    var_76 = [var_0]
    var_77 = {}
    var_78 = '*'
    var_79 = [var_78]
    var_80 = [var_30]
    var_81 = {var_26: var_79, var_27: var_80}
    var_82 = {var_19: var_77, var_20: var_81}
    var_83 = {var_18: var_82}
    var_84 = {}
    var_85 = {}
    var_86 = module_0.ParsedContent()
    var_87 = True
    var_88 = module_1.Config()
    var_89 = module_2.sorted_imports(var_86, var_88)
    var_90 = 'from sys import *\nfrom json import load\n\n'
    var_91 = [var_0]
    var_92 = []
    var_93 = {var_21: var_92}
    var_94 = [var_28]
    var_95 = {var_26: var_94}
    var_96 = {var_19: var_93, var_20: var_95}
    var_97 = {var_18: var_96}
    var_98 = {}
    var_99 = {}
    var_100 = module_0.ParsedContent()
    var_101 = 'thirdparty'
    var_102 = 'Third Party Imports'
    var_103 = {var_101: var_102}
    var_104 = module_1.Config()
    var_105 = module_2.sorted_imports(var_100, var_104)
    var_106 = '# Third Party Imports\nimport zlib\n\nfrom sys import path\n\n'
    var_107 = [var_0]
    var_108 = []
    var_109 = {var_21: var_108}
    var_110 = [var_28]
    var_111 = {var_26: var_110}
    var_112 = {var_19: var_109, var_20: var_111}
    var_113 = []
    var_114 = {var_22: var_113}
    var_115 = [var_30]
    var_116 = {var_27: var_115}
    var_117 = {var_19: var_114, var_20: var_116}
    var_118 = {var_18: var_112, var_57: var_117}
    var_119 = {}
    var_120 = {}
    var_121 = module_0.ParsedContent()
    var_122 = True
    var_123 = module_1.Config()
    var_124 = module_2.sorted_imports(var_121, var_123)
    var_125 = 'import os\nimport zlib\n\nfrom json import load\nfrom sys import path\n\n'



# Parsed testcases at query #13
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = '\n'
    var_5 = 1
    var_6 = module_0.ParsedContent()
    var_7 = [var_0]
    var_8 = 'THIRDPARTY'
    var_9 = 'straight'
    var_10 = 'from'
    var_11 = 'os'
    var_12 = 'import os'
    var_13 = [var_12]
    var_14 = {var_11: var_13}
    var_15 = {}
    var_16 = {var_9: var_14, var_10: var_15}
    var_17 = {var_8: var_16}
    var_18 = 0
    var_19 = 2
    var_20 = module_0.ParsedContent()
    var_21 = [var_0]
    var_22 = 'sys'
    var_23 = 'import sys'
    var_24 = [var_23]
    var_25 = [var_12]
    var_26 = {var_22: var_24, var_11: var_25}
    var_27 = {}
    var_28 = {var_9: var_26, var_10: var_27}
    var_29 = {var_8: var_28}
    var_30 = 3
    var_31 = module_0.ParsedContent()
    var_32 = [var_0]
    var_33 = 'FUTURE'
    var_34 = '__future__'
    var_35 = 'from __future__ import annotations'
    var_36 = [var_35]
    var_37 = {var_34: var_36}
    var_38 = {}
    var_39 = {var_9: var_37, var_10: var_38}
    var_40 = [var_12]
    var_41 = {var_11: var_40}
    var_42 = {}
    var_43 = {var_9: var_41, var_10: var_42}
    var_44 = {var_33: var_39, var_8: var_43}
    var_45 = module_0.ParsedContent()
    var_46 = [var_0]
    var_47 = [var_23]
    var_48 = [var_12]
    var_49 = {var_22: var_47, var_11: var_48}
    var_50 = {}
    var_51 = {var_9: var_49, var_10: var_50}
    var_52 = {var_8: var_51}
    var_53 = module_0.ParsedContent()
    var_54 = module_1.Config()
    var_55 = module_2.sorted_imports(var_53, var_54)
    assert var_55 == "import os\nimport sys\n\n\nprint('hello')\n"
    var_56 = [var_0]
    var_57 = {}
    var_58 = 'from os import path'
    var_59 = [var_58]
    var_60 = {var_11: var_59}
    var_61 = {var_9: var_57, var_10: var_60}
    var_62 = {var_8: var_61}
    var_63 = module_0.ParsedContent()
    var_64 = module_2.sorted_imports(var_63, var_54)
    assert var_64 == "from os import path\n\nprint('hello')\n"
    var_65 = [var_0]
    var_66 = {}
    var_67 = 'from os import *'
    var_68 = [var_67]
    var_69 = 'from sys import path'
    var_70 = [var_69]
    var_71 = {var_11: var_68, var_22: var_70}
    var_72 = {var_9: var_66, var_10: var_71}
    var_73 = {var_8: var_72}
    var_74 = module_0.ParsedContent()
    var_75 = True
    var_76 = module_1.Config()
    var_77 = module_2.sorted_imports(var_74, var_76)
    assert var_77 == "from os import *\nfrom sys import path\n\nprint('hello')\n"
    var_78 = [var_0]
    var_79 = [var_35]
    var_80 = {var_34: var_79}
    var_81 = {}
    var_82 = {var_9: var_80, var_10: var_81}
    var_83 = [var_12]
    var_84 = {var_11: var_83}
    var_85 = {}
    var_86 = {var_9: var_84, var_10: var_85}
    var_87 = {var_33: var_82, var_8: var_86}
    var_88 = module_0.ParsedContent()
    var_89 = True
    var_90 = module_1.Config()
    var_91 = module_2.sorted_imports(var_88, var_90)
    assert var_91 == "from __future__ import annotations\nimport os\n\nprint('hello')\n"
    var_92 = [var_0]
    var_93 = [var_23]
    var_94 = [var_12]
    var_95 = {var_22: var_93, var_11: var_94}
    var_96 = {}
    var_97 = {var_9: var_95, var_10: var_96}
    var_98 = {var_8: var_97}
    var_99 = module_0.ParsedContent()
    var_100 = True
    var_101 = module_1.Config()
    var_102 = module_2.sorted_imports(var_99, var_101)
    assert var_102 == "import os\nimport sys\n\nprint('hello')\n"
    var_103 = [var_0]
    var_104 = [var_12]
    var_105 = {var_11: var_104}
    var_106 = {}
    var_107 = {var_9: var_105, var_10: var_106}
    var_108 = {var_8: var_107}
    var_109 = module_0.ParsedContent()
    var_110 = 'thirdparty'
    var_111 = 'Third Party Imports'
    var_112 = {var_110: var_111}
    var_113 = module_1.Config()
    var_114 = module_2.sorted_imports(var_109, var_113)
    assert var_114 == "# Third Party Imports\nimport os\n\nprint('hello')\n"



# Parsed testcases at query #14
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = 'x = 1'
    var_2 = [var_0, var_1]
    var_3 = 'THIRDPARTY'
    var_4 = 'straight'
    var_5 = 'from'
    var_6 = 'os'
    var_7 = 'sys'
    var_8 = []
    var_9 = []
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'collections'
    var_12 = 'itertools'
    var_13 = 'defaultdict'
    var_14 = [var_13]
    var_15 = 'chain'
    var_16 = [var_15]
    var_17 = {var_11: var_14, var_12: var_16}
    var_18 = {var_4: var_10, var_5: var_17}
    var_19 = {var_3: var_18}
    var_20 = 0
    var_21 = '\n'
    var_22 = module_0.ParsedContent()
    var_23 = module_1.Config()
    var_24 = module_2.sorted_imports(var_22, var_23)
    assert var_24 == '\nimport os\nimport sys\n\nfrom collections import defaultdict\nfrom itertools import chain\n\nx = 1\n'
    var_25 = [var_0, var_1]
    var_26 = 'FUTURE'
    var_27 = '__future__'
    var_28 = []
    var_29 = {var_27: var_28}
    var_30 = {}
    var_31 = {var_4: var_29, var_5: var_30}
    var_32 = []
    var_33 = []
    var_34 = {var_6: var_32, var_7: var_33}
    var_35 = [var_13]
    var_36 = {var_11: var_35}
    var_37 = {var_4: var_34, var_5: var_36}
    var_38 = {var_26: var_31, var_3: var_37}
    var_39 = module_0.ParsedContent()
    var_40 = True
    var_41 = module_1.Config()
    var_42 = module_2.sorted_imports(var_39, var_41)
    assert var_42 == '\nfrom __future__ import absolute_import\nimport os\nimport sys\n\nfrom collections import defaultdict\n\nx = 1\n'
    var_43 = [var_0, var_1]
    var_44 = []
    var_45 = []
    var_46 = {var_6: var_44, var_7: var_45}
    var_47 = [var_13]
    var_48 = {var_11: var_47}
    var_49 = {var_4: var_46, var_5: var_48}
    var_50 = {var_3: var_49}
    var_51 = module_0.ParsedContent()
    var_52 = module_1.Config()
    var_53 = module_2.sorted_imports(var_51, var_52)
    assert var_53 == '\nimport os\nimport sys\n\nfrom collections import defaultdict\n\nx = 1\n'
    var_54 = [var_0, var_1]
    var_55 = {}
    var_56 = 'module1'
    var_57 = 'module2'
    var_58 = '*'
    var_59 = [var_58]
    var_60 = 'func'
    var_61 = [var_60]
    var_62 = {var_56: var_59, var_57: var_61}
    var_63 = {var_4: var_55, var_5: var_62}
    var_64 = {var_3: var_63}
    var_65 = module_0.ParsedContent()
    var_66 = module_1.Config()
    var_67 = module_2.sorted_imports(var_65, var_66)
    assert var_67 == '\nfrom module1 import *\nfrom module2 import func\n\nx = 1\n'
    var_68 = [var_0, var_1]
    var_69 = []
    var_70 = {var_6: var_69}
    var_71 = [var_13]
    var_72 = {var_11: var_71}
    var_73 = {var_4: var_70, var_5: var_72}
    var_74 = {var_3: var_73}
    var_75 = module_0.ParsedContent()
    var_76 = module_1.Config()
    var_77 = module_2.sorted_imports(var_75, var_76)
    assert var_77 == '\nfrom collections import defaultdict\n\nimport os\n\nx = 1\n'
    var_78 = [var_0, var_1]
    var_79 = []
    var_80 = {var_6: var_79}
    var_81 = [var_13]
    var_82 = {var_11: var_81}
    var_83 = {var_4: var_80, var_5: var_82}
    var_84 = {var_3: var_83}
    var_85 = module_0.ParsedContent()
    var_86 = 2
    var_87 = module_1.Config()
    var_88 = module_2.sorted_imports(var_85, var_87)
    assert var_88 == '\nimport os\n\n\n\nfrom collections import defaultdict\n\nx = 1\n'
    var_89 = [var_0, var_1]
    var_90 = []
    var_91 = {var_27: var_90}
    var_92 = {}
    var_93 = {var_4: var_91, var_5: var_92}
    var_94 = []
    var_95 = {var_6: var_94}
    var_96 = [var_13]
    var_97 = {var_11: var_96}
    var_98 = {var_4: var_95, var_5: var_97}
    var_99 = {var_26: var_93, var_3: var_98}
    var_100 = module_0.ParsedContent()
    var_101 = module_1.Config()
    var_102 = module_2.sorted_imports(var_100, var_101)
    assert var_102 == '\nfrom __future__ import absolute_import\n\n\nimport os\n\nfrom collections import defaultdict\n\nx = 1\n'
    var_103 = [var_0, var_1]
    var_104 = []
    var_105 = {var_6: var_104}
    var_106 = [var_13]
    var_107 = {var_11: var_106}
    var_108 = {var_4: var_105, var_5: var_107}
    var_109 = {var_3: var_108}
    var_110 = module_0.ParsedContent()
    var_111 = 'thirdparty'
    var_112 = 'Third Party Imports'
    var_113 = {var_111: var_112}
    var_114 = module_1.Config()
    var_115 = module_2.sorted_imports(var_110, var_114)
    assert var_115 == '\n# Third Party Imports\nimport os\n\nfrom collections import defaultdict\n\nx = 1\n'
    var_116 = [var_0, var_1]
    var_117 = []
    var_118 = {var_6: var_117}
    var_119 = [var_13]
    var_120 = {var_11: var_119}
    var_121 = {var_4: var_118, var_5: var_120}
    var_122 = {var_3: var_121}
    var_123 = module_0.ParsedContent()
    var_124 = 'End of Third Party Imports'
    var_125 = {var_111: var_124}
    var_126 = module_1.Config()
    var_127 = module_2.sorted_imports(var_123, var_126)
    assert var_127 == '\nimport os\n\nfrom collections import defaultdict\n\n# End of Third Party Imports\n\nx = 1\n'
    var_128 = [var_0, var_1]
    var_129 = []
    var_130 = {var_6: var_129}
    var_131 = [var_13]
    var_132 = {var_11: var_131}
    var_133 = {var_4: var_130, var_5: var_132}
    var_134 = {var_3: var_133}
    var_135 = module_0.ParsedContent()
    var_136 = module_1.Config()
    var_137 = module_2.sorted_imports(var_135, var_136)
    assert var_137 == '\n\nimport os\n\nfrom collections import defaultdict\n\nx = 1\n'
    var_138 = [var_0, var_1]
    var_139 = []
    var_140 = {var_6: var_139}
    var_141 = [var_13]
    var_142 = {var_11: var_141}
    var_143 = {var_4: var_140, var_5: var_142}
    var_144 = {var_3: var_143}
    var_145 = module_0.ParsedContent()
    var_146 = module_1.Config()
    var_147 = module_2.sorted_imports(var_145, var_146)
    assert var_147 == '\nimport os\n\nfrom collections import defaultdict\n\n\nx = 1\n'
    var_148 = [var_0, var_1]
    var_149 = []
    var_150 = []
    var_151 = {var_6: var_149, var_7: var_150}
    var_152 = [var_13]
    var_153 = {var_11: var_152}
    var_154 = {var_4: var_151, var_5: var_153}
    var_155 = {var_3: var_154}
    var_156 = module_0.ParsedContent()
    var_157 = [var_6, var_11]
    var_158 = module_1.Config()
    var_159 = module_2.sorted_imports(var_156, var_158)
    assert var_159 == '\nimport sys\n\nx = 1\n'
    var_160 = [var_1]
    var_161 = {}
    var_162 = -1
    var_163 = module_0.ParsedContent()
    var_164 = module_1.Config()
    var_165 = module_2.sorted_imports(var_163, var_164)
    assert var_165 == 'x = 1\n'



# Parsed testcases at query #15
#--------------------------


import isort.parse as module_0
import isort.output as module_1
import isort.settings as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = 0
    var_5 = '\n'
    var_6 = {}
    var_7 = {}
    var_8 = module_0.ParsedContent()
    var_9 = module_1.sorted_imports(var_8)
    assert var_9 == ''
    var_10 = "print('hello')"
    var_11 = [var_10]
    var_12 = {}
    var_13 = -1
    var_14 = 1
    var_15 = {}
    var_16 = {}
    var_17 = module_0.ParsedContent()
    var_18 = module_1.sorted_imports(var_17)
    assert var_18 == "print('hello')"
    var_19 = [var_0]
    var_20 = 'THIRDPARTY'
    var_21 = 'straight'
    var_22 = 'from'
    var_23 = 'os'
    var_24 = 'sys'
    var_25 = [var_23]
    var_26 = [var_24]
    var_27 = {var_23: var_25, var_24: var_26}
    var_28 = 'collections'
    var_29 = 'OrderedDict'
    var_30 = [var_29]
    var_31 = {var_28: var_30}
    var_32 = {var_21: var_27, var_22: var_31}
    var_33 = {var_20: var_32}
    var_34 = {}
    var_35 = {}
    var_36 = module_0.ParsedContent()
    var_37 = False
    var_38 = module_2.Config()
    var_39 = 'import os\nimport sys\n\nfrom collections import OrderedDict\n'
    var_40 = module_1.sorted_imports(var_36, var_38)
    var_41 = [var_0]
    var_42 = [var_23]
    var_43 = [var_24]
    var_44 = {var_23: var_42, var_24: var_43}
    var_45 = [var_29]
    var_46 = {var_28: var_45}
    var_47 = {var_21: var_44, var_22: var_46}
    var_48 = {var_20: var_47}
    var_49 = {}
    var_50 = {}
    var_51 = module_0.ParsedContent()
    var_52 = True
    var_53 = module_2.Config()
    var_54 = 'import os\nimport sys\n\nfrom collections import OrderedDict\n'
    var_55 = module_1.sorted_imports(var_51, var_53)
    var_56 = [var_0]
    var_57 = 'FUTURE'
    var_58 = '__future__'
    var_59 = 'annotations'
    var_60 = [var_59]
    var_61 = {var_58: var_60}
    var_62 = {var_21: var_61}
    var_63 = [var_23]
    var_64 = [var_24]
    var_65 = {var_23: var_63, var_24: var_64}
    var_66 = [var_29]
    var_67 = {var_28: var_66}
    var_68 = {var_21: var_65, var_22: var_67}
    var_69 = {var_57: var_62, var_20: var_68}
    var_70 = {}
    var_71 = {}
    var_72 = module_0.ParsedContent()
    var_73 = 'future'
    var_74 = 'thirdparty'
    var_75 = 'Future imports'
    var_76 = 'Third party imports'
    var_77 = {var_73: var_75, var_74: var_76}
    var_78 = True
    var_79 = module_2.Config()
    var_80 = '# Future imports\nfrom __future__ import annotations\n\n# Third party imports\nimport os\nimport sys\n\nfrom collections import OrderedDict\n'
    var_81 = module_1.sorted_imports(var_72, var_79)
    var_82 = [var_0]
    var_83 = [var_23]
    var_84 = [var_24]
    var_85 = {var_23: var_83, var_24: var_84}
    var_86 = [var_29]
    var_87 = {var_28: var_86}
    var_88 = {var_21: var_85, var_22: var_87}
    var_89 = {var_20: var_88}
    var_90 = {}
    var_91 = {}
    var_92 = module_0.ParsedContent()
    var_93 = [var_23]
    var_94 = module_2.Config()
    var_95 = 'import sys\n\nfrom collections import OrderedDict\n'
    var_96 = module_1.sorted_imports(var_92, var_94)
    var_97 = '# Placeholder'
    var_98 = [var_97, var_10]
    var_99 = [var_23]
    var_100 = [var_24]
    var_101 = {var_23: var_99, var_24: var_100}
    var_102 = [var_29]
    var_103 = {var_28: var_102}
    var_104 = {var_21: var_101, var_22: var_103}
    var_105 = {var_20: var_104}
    var_106 = 2
    var_107 = 'import os'
    var_108 = 'import sys'
    var_109 = [var_107, var_108]
    var_110 = {var_20: var_109}
    var_111 = {var_97: var_20}
    var_112 = module_0.ParsedContent()
    var_113 = module_2.Config()
    var_114 = "# Placeholder\nimport os\nimport sys\n\nprint('hello')"
    var_115 = module_1.sorted_imports(var_112, var_113)



# Parsed testcases at query #16
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 'THIRDPARTY'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = 'sys'
    var_7 = []
    var_8 = []
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'collections'
    var_11 = 'defaultdict'
    var_12 = [var_11]
    var_13 = {var_10: var_12}
    var_14 = {var_3: var_9, var_4: var_13}
    var_15 = {var_2: var_14}
    var_16 = 0
    var_17 = 1
    var_18 = '\n'
    var_19 = module_0.ParsedContent()
    var_20 = module_1.Config()
    var_21 = module_2.sorted_imports(var_19, var_20)
    assert var_21 == 'from collections import defaultdict\nimport os\nimport sys\n'
    var_22 = [var_0]
    var_23 = 'FUTURE'
    var_24 = '__future__'
    var_25 = []
    var_26 = {var_24: var_25}
    var_27 = {}
    var_28 = {var_3: var_26, var_4: var_27}
    var_29 = []
    var_30 = []
    var_31 = {var_5: var_29, var_6: var_30}
    var_32 = {}
    var_33 = {var_3: var_31, var_4: var_32}
    var_34 = {var_23: var_28, var_2: var_33}
    var_35 = module_0.ParsedContent()
    var_36 = True
    var_37 = module_1.Config()
    var_38 = module_2.sorted_imports(var_35, var_37)
    assert var_38 == 'import __future__\nimport os\nimport sys\n'
    var_39 = [var_0]
    var_40 = {}
    var_41 = -1
    var_42 = module_0.ParsedContent()
    var_43 = module_1.Config()
    var_44 = module_2.sorted_imports(var_42, var_43)
    assert var_44 == ''
    var_45 = [var_0]
    var_46 = []
    var_47 = []
    var_48 = {var_5: var_46, var_6: var_47}
    var_49 = [var_11]
    var_50 = {var_10: var_49}
    var_51 = {var_3: var_48, var_4: var_50}
    var_52 = {var_2: var_51}
    var_53 = module_0.ParsedContent()
    var_54 = 'thirdparty'
    var_55 = 'Third-party imports'
    var_56 = {var_54: var_55}
    var_57 = module_1.Config()
    var_58 = module_2.sorted_imports(var_53, var_57)
    assert var_58 == '# Third-party imports\nfrom collections import defaultdict\nimport os\nimport sys\n'
    var_59 = [var_0]
    var_60 = []
    var_61 = {var_24: var_60}
    var_62 = {}
    var_63 = {var_3: var_61, var_4: var_62}
    var_64 = []
    var_65 = []
    var_66 = {var_5: var_64, var_6: var_65}
    var_67 = {}
    var_68 = {var_3: var_66, var_4: var_67}
    var_69 = {var_23: var_63, var_2: var_68}
    var_70 = module_0.ParsedContent()
    var_71 = 2
    var_72 = module_1.Config()
    var_73 = module_2.sorted_imports(var_70, var_72)
    assert var_73 == 'import __future__\n\n\nimport os\nimport sys\n'



# Parsed testcases at query #17
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = 'THIRDPARTY'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = 'sys'
    var_7 = set()
    var_8 = set()
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'collections'
    var_11 = 'defaultdict'
    var_12 = {var_11}
    var_13 = {var_10: var_12}
    var_14 = {var_3: var_9, var_4: var_13}
    var_15 = {var_2: var_14}
    var_16 = 0
    var_17 = 1
    var_18 = '\n'
    var_19 = module_0.ParsedContent()
    var_20 = module_1.Config()
    var_21 = module_2.sorted_imports(var_19, var_20)
    assert var_21 == "import os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')"
    var_22 = [var_0]
    var_23 = 'FIRSTPARTY'
    var_24 = set()
    var_25 = set()
    var_26 = {var_5: var_24, var_6: var_25}
    var_27 = {var_11}
    var_28 = {var_10: var_27}
    var_29 = {var_3: var_26, var_4: var_28}
    var_30 = 'my_module'
    var_31 = set()
    var_32 = {var_30: var_31}
    var_33 = {}
    var_34 = {var_3: var_32, var_4: var_33}
    var_35 = {var_2: var_29, var_23: var_34}
    var_36 = module_0.ParsedContent()
    var_37 = True
    var_38 = module_1.Config()
    var_39 = module_2.sorted_imports(var_36, var_38)
    assert var_39 == "import my_module\nimport os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')"
    var_40 = [var_0]
    var_41 = set()
    var_42 = {var_5: var_41}
    var_43 = {var_11}
    var_44 = {var_10: var_43}
    var_45 = {var_3: var_42, var_4: var_44}
    var_46 = {var_2: var_45}
    var_47 = module_0.ParsedContent()
    var_48 = True
    var_49 = module_1.Config()
    var_50 = module_2.sorted_imports(var_47, var_49)
    assert var_50 == "from collections import defaultdict\n\nimport os\n\nprint('hello')"
    var_51 = [var_0]
    var_52 = {}
    var_53 = {var_11}
    var_54 = '*'
    var_55 = {var_54}
    var_56 = {var_10: var_53, var_5: var_55}
    var_57 = {var_3: var_52, var_4: var_56}
    var_58 = {var_2: var_57}
    var_59 = module_0.ParsedContent()
    var_60 = True
    var_61 = module_1.Config()
    var_62 = module_2.sorted_imports(var_59, var_61)
    assert var_62 == "from os import *\nfrom collections import defaultdict\n\nprint('hello')"
    var_63 = [var_0]
    var_64 = set()
    var_65 = {var_5: var_64}
    var_66 = {}
    var_67 = {var_3: var_65, var_4: var_66}
    var_68 = {var_2: var_67}
    var_69 = module_0.ParsedContent()
    var_70 = 'thirdparty'
    var_71 = 'Third Party Imports'
    var_72 = {var_70: var_71}
    var_73 = module_1.Config()
    var_74 = module_2.sorted_imports(var_69, var_73)
    assert var_74 == "# Third Party Imports\nimport os\n\nprint('hello')"
    var_75 = [var_0]
    var_76 = {}
    var_77 = -1
    var_78 = module_0.ParsedContent()
    var_79 = module_1.Config()
    var_80 = module_2.sorted_imports(var_78, var_79)
    assert var_80 == "print('hello')"
    var_81 = [var_0]
    var_82 = set()
    var_83 = set()
    var_84 = {var_5: var_82, var_6: var_83}
    var_85 = {var_11}
    var_86 = {var_10: var_85}
    var_87 = {var_3: var_84, var_4: var_86}
    var_88 = {var_2: var_87}
    var_89 = module_0.ParsedContent()
    var_90 = [var_5, var_10]
    var_91 = module_1.Config()
    var_92 = module_2.sorted_imports(var_89, var_91)
    assert var_92 == "import sys\n\nprint('hello')"
    var_93 = [var_0]
    var_94 = set()
    var_95 = {var_5: var_94}
    var_96 = {}
    var_97 = {var_3: var_95, var_4: var_96}
    var_98 = {var_2: var_97}
    var_99 = module_0.ParsedContent()
    var_100 = 'import os'
    var_101 = 'import os  # formatted'
    var_102 = lambda code, ext, cfg: code.replace(var_100, var_101)
    var_103 = module_1.Config()
    var_104 = module_2.sorted_imports(var_99, var_103)
    assert var_104 == "import os  # formatted\n\nprint('hello')"
    var_105 = [var_0]
    var_106 = set()
    var_107 = {var_5: var_106}
    var_108 = {}
    var_109 = {var_3: var_107, var_4: var_108}
    var_110 = {var_2: var_109}
    var_111 = module_0.ParsedContent()
    var_112 = 2
    var_113 = module_1.Config()
    var_114 = module_2.sorted_imports(var_111, var_113)
    assert var_114 == "import os\n\n\n\nprint('hello')"
    var_115 = [var_0]
    var_116 = set()
    var_117 = {var_5: var_116}
    var_118 = {}
    var_119 = {var_3: var_117, var_4: var_118}
    var_120 = {var_2: var_119}
    var_121 = module_0.ParsedContent()
    var_122 = module_1.Config()
    var_123 = module_2.sorted_imports(var_121, var_122)
    assert var_123 == "\n\nimport os\n\nprint('hello')"



# Parsed testcases at query #18
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = "print('hello')"
    var_2 = [var_0, var_1]
    var_3 = 'THIRDPARTY'
    var_4 = 'straight'
    var_5 = 'from'
    var_6 = 'zlib'
    var_7 = 'os'
    var_8 = set()
    var_9 = set()
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'sys'
    var_12 = 'collections'
    var_13 = 'path'
    var_14 = set()
    var_15 = {var_13: var_14}
    var_16 = 'defaultdict'
    var_17 = set()
    var_18 = {var_16: var_17}
    var_19 = {var_11: var_15, var_12: var_18}
    var_20 = {var_4: var_10, var_5: var_19}
    var_21 = {var_3: var_20}
    var_22 = 0
    var_23 = 2
    var_24 = '\n'
    var_25 = module_0.ParsedContent()
    var_26 = module_1.Config()
    var_27 = module_2.sorted_imports(var_25, var_26)
    var_28 = "import os\nimport zlib\n\nfrom collections import defaultdict\nfrom sys import path\n\nprint('hello')"
    var_29 = [var_0, var_1]
    var_30 = 'FUTURE'
    var_31 = '__future__'
    var_32 = set()
    var_33 = {var_31: var_32}
    var_34 = {}
    var_35 = {var_4: var_33, var_5: var_34}
    var_36 = set()
    var_37 = set()
    var_38 = {var_6: var_36, var_7: var_37}
    var_39 = {}
    var_40 = {var_4: var_38, var_5: var_39}
    var_41 = {var_30: var_35, var_3: var_40}
    var_42 = module_0.ParsedContent()
    var_43 = True
    var_44 = module_1.Config()
    var_45 = module_2.sorted_imports(var_42, var_44)
    var_46 = "from __future__ import\nimport os\nimport zlib\n\nprint('hello')"
    var_47 = [var_0, var_1]
    var_48 = set()
    var_49 = set()
    var_50 = {var_6: var_48, var_7: var_49}
    var_51 = set()
    var_52 = {var_13: var_51}
    var_53 = set()
    var_54 = {var_16: var_53}
    var_55 = {var_11: var_52, var_12: var_54}
    var_56 = {var_4: var_50, var_5: var_55}
    var_57 = {var_3: var_56}
    var_58 = module_0.ParsedContent()
    var_59 = [var_12]
    var_60 = module_1.Config()
    var_61 = module_2.sorted_imports(var_58, var_60)
    var_62 = [var_0, var_1]
    var_63 = set()
    var_64 = set()
    var_65 = {var_6: var_63, var_7: var_64}
    var_66 = set()
    var_67 = {var_13: var_66}
    var_68 = set()
    var_69 = {var_16: var_68}
    var_70 = {var_11: var_67, var_12: var_69}
    var_71 = {var_4: var_65, var_5: var_70}
    var_72 = {var_3: var_71}
    var_73 = module_0.ParsedContent()
    var_74 = [var_7, var_11]
    var_75 = module_1.Config()
    var_76 = module_2.sorted_imports(var_73, var_75)
    var_77 = [var_0, var_1]
    var_78 = {}
    var_79 = '*'
    var_80 = set()
    var_81 = set()
    var_82 = {var_79: var_80, var_13: var_81}
    var_83 = set()
    var_84 = {var_16: var_83}
    var_85 = {var_11: var_82, var_12: var_84}
    var_86 = {var_4: var_78, var_5: var_85}
    var_87 = {var_3: var_86}
    var_88 = module_0.ParsedContent()
    var_89 = module_1.Config()
    var_90 = module_2.sorted_imports(var_88, var_89)
    var_91 = 'from sys import *'
    var_92 = 'from sys import path'
    var_93 = [var_0, var_1]
    var_94 = set()
    var_95 = set()
    var_96 = {var_6: var_94, var_7: var_95}
    var_97 = set()
    var_98 = {var_13: var_97}
    var_99 = set()
    var_100 = {var_16: var_99}
    var_101 = {var_11: var_98, var_12: var_100}
    var_102 = {var_4: var_96, var_5: var_101}
    var_103 = {var_3: var_102}
    var_104 = module_0.ParsedContent()
    var_105 = module_1.Config()
    var_106 = module_2.sorted_imports(var_104, var_105)
    var_107 = "from collections import defaultdict\nfrom sys import path\n\nimport os\nimport zlib\n\nprint('hello')"
    var_108 = [var_0, var_1]
    var_109 = set()
    var_110 = set()
    var_111 = {var_6: var_109, var_7: var_110}
    var_112 = set()
    var_113 = {var_13: var_112}
    var_114 = set()
    var_115 = {var_16: var_114}
    var_116 = {var_11: var_113, var_12: var_115}
    var_117 = {var_4: var_111, var_5: var_116}
    var_118 = {var_3: var_117}
    var_119 = module_0.ParsedContent()
    var_120 = 'thirdparty'
    var_121 = 'Third Party Imports'
    var_122 = {var_120: var_121}
    var_123 = module_1.Config()
    var_124 = module_2.sorted_imports(var_119, var_123)
    var_125 = [var_0, var_1]
    var_126 = set()
    var_127 = {var_31: var_126}
    var_128 = {}
    var_129 = {var_4: var_127, var_5: var_128}
    var_130 = set()
    var_131 = set()
    var_132 = {var_6: var_130, var_7: var_131}
    var_133 = {}
    var_134 = {var_4: var_132, var_5: var_133}
    var_135 = {var_30: var_129, var_3: var_134}
    var_136 = module_0.ParsedContent()
    var_137 = module_1.Config()
    var_138 = module_2.sorted_imports(var_136, var_137)
    var_139 = '\n\n'
    var_140 = [var_0, var_1]
    var_141 = set()
    var_142 = set()
    var_143 = {var_6: var_141, var_7: var_142}
    var_144 = set()
    var_145 = {var_13: var_144}
    var_146 = set()
    var_147 = {var_16: var_146}
    var_148 = {var_11: var_145, var_12: var_147}
    var_149 = {var_4: var_143, var_5: var_148}
    var_150 = {var_3: var_149}
    var_151 = module_0.ParsedContent()
    var_152 = module_1.Config()
    var_153 = module_2.sorted_imports(var_151, var_152)
    var_154 = [var_0, var_1]
    var_155 = set()
    var_156 = set()
    var_157 = {var_6: var_155, var_7: var_156}
    var_158 = set()
    var_159 = {var_13: var_158}
    var_160 = set()
    var_161 = {var_16: var_160}
    var_162 = {var_11: var_159, var_12: var_161}
    var_163 = {var_4: var_157, var_5: var_162}
    var_164 = {var_3: var_163}
    var_165 = module_0.ParsedContent()



# Parsed testcases at query #19
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 'THIRDPARTY'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = 'sys'
    var_7 = set()
    var_8 = set()
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'collections'
    var_11 = 'defaultdict'
    var_12 = {var_11}
    var_13 = {var_10: var_12}
    var_14 = {var_3: var_9, var_4: var_13}
    var_15 = {var_2: var_14}
    var_16 = 0
    var_17 = 1
    var_18 = '\n'
    var_19 = module_0.ParsedContent()
    var_20 = module_1.Config()
    var_21 = module_2.sorted_imports(var_19, var_20)
    var_22 = [var_0]
    var_23 = 'FUTURE'
    var_24 = 'FIRSTPARTY'
    var_25 = '__future__'
    var_26 = 'print_function'
    var_27 = {var_26}
    var_28 = {var_25: var_27}
    var_29 = {}
    var_30 = {var_3: var_28, var_4: var_29}
    var_31 = set()
    var_32 = {var_5: var_31}
    var_33 = {}
    var_34 = {var_3: var_32, var_4: var_33}
    var_35 = 'my_module'
    var_36 = set()
    var_37 = {var_35: var_36}
    var_38 = {}
    var_39 = {var_3: var_37, var_4: var_38}
    var_40 = {var_23: var_30, var_2: var_34, var_24: var_39}
    var_41 = module_0.ParsedContent()
    var_42 = True
    var_43 = module_1.Config()
    var_44 = module_2.sorted_imports(var_41, var_43)
    var_45 = [var_0]
    var_46 = 'django'
    var_47 = set()
    var_48 = {var_46: var_47}
    var_49 = {}
    var_50 = {var_3: var_48, var_4: var_49}
    var_51 = 'my_app'
    var_52 = set()
    var_53 = {var_51: var_52}
    var_54 = {}
    var_55 = {var_3: var_53, var_4: var_54}
    var_56 = {var_2: var_50, var_24: var_55}
    var_57 = module_0.ParsedContent()
    var_58 = 'DJANGO'
    var_59 = [var_58]
    var_60 = module_1.Config()
    var_61 = module_2.sorted_imports(var_57, var_60)
    var_62 = [var_0]
    var_63 = set()
    var_64 = set()
    var_65 = {var_5: var_63, var_6: var_64}
    var_66 = {}
    var_67 = {var_3: var_65, var_4: var_66}
    var_68 = {var_2: var_67}
    var_69 = module_0.ParsedContent()
    var_70 = [var_6]
    var_71 = module_1.Config()
    var_72 = module_2.sorted_imports(var_69, var_71)
    var_73 = [var_0]
    var_74 = set()
    var_75 = {var_5: var_74}
    var_76 = 'exit'
    var_77 = {var_76}
    var_78 = {var_6: var_77}
    var_79 = {var_3: var_75, var_4: var_78}
    var_80 = {var_2: var_79}
    var_81 = module_0.ParsedContent()
    var_82 = True
    var_83 = module_1.Config()
    var_84 = module_2.sorted_imports(var_81, var_83)
    var_85 = 'from sys import exit'
    var_86 = 'import os'
    var_87 = [var_0]
    var_88 = {}
    var_89 = 'numpy'
    var_90 = '*'
    var_91 = {var_90}
    var_92 = 'path'
    var_93 = {var_92}
    var_94 = {var_89: var_91, var_5: var_93}
    var_95 = {var_3: var_88, var_4: var_94}
    var_96 = {var_2: var_95}
    var_97 = module_0.ParsedContent()
    var_98 = True
    var_99 = module_1.Config()
    var_100 = module_2.sorted_imports(var_97, var_99)
    var_101 = 'from numpy import *'
    var_102 = 'from os import path'
    var_103 = [var_0]
    var_104 = 'requests'
    var_105 = set()
    var_106 = {var_104: var_105}
    var_107 = {}
    var_108 = {var_3: var_106, var_4: var_107}
    var_109 = {var_2: var_108}
    var_110 = module_0.ParsedContent()
    var_111 = 'thirdparty'
    var_112 = 'Third Party Imports'
    var_113 = {var_111: var_112}
    var_114 = module_1.Config()
    var_115 = module_2.sorted_imports(var_110, var_114)
    var_116 = "print('hello')"
    var_117 = [var_116]
    var_118 = {}
    var_119 = -1
    var_120 = module_0.ParsedContent()
    var_121 = module_2.sorted_imports(var_120)
    assert var_121 == "print('hello')\n"
    var_122 = '# Place imports here'
    var_123 = [var_122, var_0]
    var_124 = set()
    var_125 = {var_5: var_124}
    var_126 = {}
    var_127 = {var_3: var_125, var_4: var_126}
    var_128 = {var_2: var_127}
    var_129 = 2
    var_130 = []
    var_131 = {var_2: var_130}
    var_132 = {var_122: var_2}
    var_133 = module_0.ParsedContent()
    var_134 = module_1.Config()
    var_135 = module_2.sorted_imports(var_133, var_134)



# Parsed testcases at query #20
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 'THIRDPARTY'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'zlib'
    var_6 = 'os'
    var_7 = []
    var_8 = []
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'sys'
    var_11 = 'json'
    var_12 = 'path'
    var_13 = [var_12]
    var_14 = 'load'
    var_15 = [var_14]
    var_16 = {var_10: var_13, var_11: var_15}
    var_17 = {var_3: var_9, var_4: var_16}
    var_18 = {var_2: var_17}
    var_19 = 0
    var_20 = 1
    var_21 = '\n'
    var_22 = {}
    var_23 = {}
    var_24 = module_0.ParsedContent()
    var_25 = False
    var_26 = False
    var_27 = module_1.Config()
    var_28 = module_2.sorted_imports(var_24, var_27)
    assert var_28 == 'import os\nimport zlib\n\nfrom json import load\nfrom sys import path\n'
    var_29 = [var_0]
    var_30 = 'FIRSTPARTY'
    var_31 = []
    var_32 = []
    var_33 = {var_5: var_31, var_6: var_32}
    var_34 = [var_12]
    var_35 = [var_14]
    var_36 = {var_10: var_34, var_11: var_35}
    var_37 = {var_3: var_33, var_4: var_36}
    var_38 = 'myapp'
    var_39 = []
    var_40 = {var_38: var_39}
    var_41 = 'myapp.utils'
    var_42 = 'helper'
    var_43 = [var_42]
    var_44 = {var_41: var_43}
    var_45 = {var_3: var_40, var_4: var_44}
    var_46 = {var_2: var_37, var_30: var_45}
    var_47 = {}
    var_48 = {}
    var_49 = module_0.ParsedContent()
    var_50 = True
    var_51 = False
    var_52 = module_1.Config()
    var_53 = module_2.sorted_imports(var_49, var_52)
    var_54 = [var_6, var_10]
    var_55 = module_1.Config()
    var_56 = module_2.sorted_imports(var_24, var_55)
    var_57 = [var_0]
    var_58 = {}
    var_59 = 'numpy'
    var_60 = 'pandas'
    var_61 = '*'
    var_62 = [var_61]
    var_63 = 'DataFrame'
    var_64 = [var_63]
    var_65 = {var_59: var_62, var_60: var_64}
    var_66 = {var_3: var_58, var_4: var_65}
    var_67 = {var_2: var_66}
    var_68 = {}
    var_69 = {}
    var_70 = module_0.ParsedContent()
    var_71 = True
    var_72 = module_1.Config()
    var_73 = module_2.sorted_imports(var_70, var_72)
    assert var_73 == 'from numpy import *\nfrom pandas import DataFrame\n'
    var_74 = True
    var_75 = module_1.Config()
    var_76 = module_2.sorted_imports(var_24, var_75)
    var_77 = 'from json import load\nfrom sys import path\n\nimport os\nimport zlib\n'
    var_78 = 2
    var_79 = module_1.Config()
    var_80 = module_2.sorted_imports(var_24, var_79)
    var_81 = '\n\n'
    var_82 = 'thirdparty'
    var_83 = 'Third Party'
    var_84 = {var_82: var_83}
    var_85 = module_1.Config()
    var_86 = module_2.sorted_imports(var_24, var_85)
    var_87 = False
    var_88 = {var_82: var_83}
    var_89 = module_1.Config()
    var_90 = module_2.sorted_imports(var_24, var_89)
    var_91 = '# Third Party'
    var_92 = '# Comment'
    var_93 = [var_92]
    var_94 = []
    var_95 = {var_6: var_94}
    var_96 = {}
    var_97 = {var_3: var_95, var_4: var_96}
    var_98 = {var_2: var_97}
    var_99 = {}
    var_100 = {}
    var_101 = module_0.ParsedContent()
    var_102 = True
    var_103 = module_1.Config()
    var_104 = module_2.sorted_imports(var_101, var_103)
    assert var_104 == 'import os\n\n# Comment\n'
    var_105 = "print('hello')"
    var_106 = [var_105]
    var_107 = []
    var_108 = {var_6: var_107}
    var_109 = {}
    var_110 = {var_3: var_108, var_4: var_109}
    var_111 = {var_2: var_110}
    var_112 = {}
    var_113 = {}
    var_114 = module_0.ParsedContent()
    var_115 = module_1.Config()
    var_116 = module_2.sorted_imports(var_114, var_115)
    var_117 = "\n\nimport os\n\n\nprint('hello')"



# Parsed testcases at query #21
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = 'x = 1'
    var_2 = [var_0, var_1]
    var_3 = 'THIRDPARTY'
    var_4 = 'straight'
    var_5 = 'from'
    var_6 = 'os'
    var_7 = 'sys'
    var_8 = set()
    var_9 = set()
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'collections'
    var_12 = 'defaultdict'
    var_13 = {var_12}
    var_14 = {var_11: var_13}
    var_15 = {var_4: var_10, var_5: var_14}
    var_16 = {var_3: var_15}
    var_17 = 0
    var_18 = 2
    var_19 = '\n'
    var_20 = module_0.ParsedContent()
    var_21 = module_1.Config()
    var_22 = module_2.sorted_imports(var_20, var_21)
    assert var_22 == '\n\nimport os\nimport sys\n\nfrom collections import defaultdict\n\n\nx = 1'
    var_23 = [var_1]
    var_24 = {}
    var_25 = -1
    var_26 = 1
    var_27 = module_0.ParsedContent()
    var_28 = module_2.sorted_imports(var_27, var_21)
    assert var_28 == 'x = 1'
    var_29 = [var_0, var_1]
    var_30 = 'FUTURE'
    var_31 = 'STDLIB'
    var_32 = '__future__'
    var_33 = 'print_function'
    var_34 = {var_33}
    var_35 = {var_32: var_34}
    var_36 = {}
    var_37 = {var_4: var_35, var_5: var_36}
    var_38 = set()
    var_39 = {var_6: var_38}
    var_40 = 'exit'
    var_41 = {var_40}
    var_42 = {var_7: var_41}
    var_43 = {var_4: var_39, var_5: var_42}
    var_44 = {var_30: var_37, var_31: var_43}
    var_45 = module_0.ParsedContent()
    var_46 = 'future'
    var_47 = 'stdlib'
    var_48 = 'Future'
    var_49 = 'Standard Library'
    var_50 = {var_46: var_48, var_47: var_49}
    var_51 = module_1.Config()
    var_52 = module_2.sorted_imports(var_45, var_51)
    assert var_52 == '\n\n# Future\nfrom __future__ import print_function\n\n# Standard Library\nimport os\n\nfrom sys import exit\n\n\nx = 1'
    var_53 = [var_0, var_1]
    var_54 = 'django'
    var_55 = set()
    var_56 = {var_54: var_55}
    var_57 = {}
    var_58 = {var_4: var_56, var_5: var_57}
    var_59 = {var_3: var_58}
    var_60 = module_0.ParsedContent()
    var_61 = 'DJANGO'
    var_62 = [var_61]
    var_63 = module_1.Config()
    var_64 = module_2.sorted_imports(var_60, var_63)
    assert var_64 == '\n\nimport django\n\n\nx = 1'
    var_65 = [var_0, var_1]
    var_66 = set()
    var_67 = set()
    var_68 = {var_6: var_66, var_7: var_67}
    var_69 = {}
    var_70 = {var_4: var_68, var_5: var_69}
    var_71 = {var_3: var_70}
    var_72 = module_0.ParsedContent()
    var_73 = [var_6]
    var_74 = module_1.Config()
    var_75 = module_2.sorted_imports(var_72, var_74)
    assert var_75 == '\n\nimport sys\n\n\nx = 1'



# Parsed testcases at query #22
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'x = 1'
    var_1 = [var_0]
    var_2 = 'THIRDPARTY'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = 'sys'
    var_7 = []
    var_8 = []
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'collections'
    var_11 = 'defaultdict'
    var_12 = [var_11]
    var_13 = {var_10: var_12}
    var_14 = {var_3: var_9, var_4: var_13}
    var_15 = {var_2: var_14}
    var_16 = 0
    var_17 = '\n'
    var_18 = 1
    var_19 = {}
    var_20 = {}
    var_21 = module_0.ParsedContent()
    var_22 = module_1.Config()
    var_23 = module_2.sorted_imports(var_21, var_22)
    assert var_23 == 'import os\nimport sys\n\nfrom collections import defaultdict\n\nx = 1\n'
    var_24 = [var_0]
    var_25 = {}
    var_26 = -1
    var_27 = {}
    var_28 = {}
    var_29 = module_0.ParsedContent()
    var_30 = module_1.Config()
    var_31 = module_2.sorted_imports(var_29, var_30)
    assert var_31 == 'x = 1\n'
    var_32 = [var_0]
    var_33 = []
    var_34 = []
    var_35 = {var_5: var_33, var_6: var_34}
    var_36 = [var_11]
    var_37 = {var_10: var_36}
    var_38 = {var_3: var_35, var_4: var_37}
    var_39 = {var_2: var_38}
    var_40 = {}
    var_41 = {}
    var_42 = module_0.ParsedContent()
    var_43 = [var_5]
    var_44 = module_1.Config()
    var_45 = module_2.sorted_imports(var_42, var_44)
    assert var_45 == 'import sys\n\nfrom collections import defaultdict\n\nx = 1\n'
    var_46 = [var_0]
    var_47 = []
    var_48 = []
    var_49 = {var_5: var_47, var_6: var_48}
    var_50 = [var_11]
    var_51 = {var_10: var_50}
    var_52 = {var_3: var_49, var_4: var_51}
    var_53 = {var_2: var_52}
    var_54 = {}
    var_55 = {}
    var_56 = module_0.ParsedContent()
    var_57 = [var_5]
    var_58 = module_1.Config()
    var_59 = module_2.sorted_imports(var_56, var_58)
    assert var_59 == 'import os\n\nimport sys\n\nfrom collections import defaultdict\n\nx = 1\n'
    var_60 = [var_0]
    var_61 = []
    var_62 = []
    var_63 = {var_5: var_61, var_6: var_62}
    var_64 = [var_11]
    var_65 = {var_10: var_64}
    var_66 = {var_3: var_63, var_4: var_65}
    var_67 = {var_2: var_66}
    var_68 = {}
    var_69 = {}
    var_70 = module_0.ParsedContent()
    var_71 = True
    var_72 = module_1.Config()
    var_73 = module_2.sorted_imports(var_70, var_72)
    assert var_73 == 'import os\nimport sys\n\nfrom collections import defaultdict\n\nx = 1\n'
    var_74 = [var_0]
    var_75 = []
    var_76 = {var_5: var_75}
    var_77 = '*'
    var_78 = [var_77]
    var_79 = 'path'
    var_80 = [var_79]
    var_81 = {var_10: var_78, var_6: var_80}
    var_82 = {var_3: var_76, var_4: var_81}
    var_83 = {var_2: var_82}
    var_84 = {}
    var_85 = {}
    var_86 = module_0.ParsedContent()
    var_87 = True
    var_88 = module_1.Config()
    var_89 = module_2.sorted_imports(var_86, var_88)
    assert var_89 == 'import os\n\nfrom collections import *\nfrom sys import path\n\nx = 1\n'
    var_90 = [var_0]
    var_91 = []
    var_92 = {var_5: var_91}
    var_93 = [var_11]
    var_94 = {var_10: var_93}
    var_95 = {var_3: var_92, var_4: var_94}
    var_96 = {var_2: var_95}
    var_97 = {}
    var_98 = {}
    var_99 = module_0.ParsedContent()
    var_100 = True
    var_101 = module_1.Config()
    var_102 = module_2.sorted_imports(var_99, var_101)
    assert var_102 == 'from collections import defaultdict\n\nimport os\n\nx = 1\n'
    var_103 = [var_0]
    var_104 = []
    var_105 = []
    var_106 = {var_5: var_104, var_6: var_105}
    var_107 = [var_11]
    var_108 = {var_10: var_107}
    var_109 = {var_3: var_106, var_4: var_108}
    var_110 = {var_2: var_109}
    var_111 = {}
    var_112 = {}
    var_113 = module_0.ParsedContent()
    var_114 = 'thirdparty'
    var_115 = 'Third Party Imports'
    var_116 = {var_114: var_115}
    var_117 = module_1.Config()
    var_118 = module_2.sorted_imports(var_113, var_117)
    assert var_118 == '# Third Party Imports\nimport os\nimport sys\n\nfrom collections import defaultdict\n\nx = 1\n'
    var_119 = [var_0]
    var_120 = []
    var_121 = []
    var_122 = {var_5: var_120, var_6: var_121}
    var_123 = [var_11]
    var_124 = {var_10: var_123}
    var_125 = {var_3: var_122, var_4: var_124}
    var_126 = {var_2: var_125}
    var_127 = {}
    var_128 = {}
    var_129 = module_0.ParsedContent()
    var_130 = 'End of Third Party Imports'
    var_131 = {var_114: var_130}
    var_132 = module_1.Config()
    var_133 = module_2.sorted_imports(var_129, var_132)
    assert var_133 == 'import os\nimport sys\n\nfrom collections import defaultdict\n\n# End of Third Party Imports\n\nx = 1\n'
    var_134 = [var_0]
    var_135 = 'FUTURE'
    var_136 = '__future__'
    var_137 = 'annotations'
    var_138 = [var_137]
    var_139 = {var_136: var_138}
    var_140 = {}
    var_141 = {var_3: var_139, var_4: var_140}
    var_142 = []
    var_143 = []
    var_144 = {var_5: var_142, var_6: var_143}
    var_145 = [var_11]
    var_146 = {var_10: var_145}
    var_147 = {var_3: var_144, var_4: var_146}
    var_148 = {var_135: var_141, var_2: var_147}
    var_149 = {}
    var_150 = {}
    var_151 = module_0.ParsedContent()
    var_152 = 2
    var_153 = module_1.Config()
    var_154 = module_2.sorted_imports(var_151, var_153)
    assert var_154 == 'from __future__ import annotations\n\n\n\nimport os\nimport sys\n\nfrom collections import defaultdict\n\nx = 1\n'



# Parsed testcases at query #23
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 'THIRDPARTY'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = 'sys'
    var_7 = []
    var_8 = []
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'collections'
    var_11 = 'itertools'
    var_12 = 'defaultdict'
    var_13 = [var_12]
    var_14 = 'chain'
    var_15 = [var_14]
    var_16 = {var_10: var_13, var_11: var_15}
    var_17 = {var_3: var_9, var_4: var_16}
    var_18 = {var_2: var_17}
    var_19 = 0
    var_20 = 1
    var_21 = '\n'
    var_22 = module_0.ParsedContent()
    var_23 = module_1.Config()
    var_24 = module_2.sorted_imports(var_22, var_23)
    var_25 = [var_0]
    var_26 = 'FUTURE'
    var_27 = '__future__'
    var_28 = 'annotations'
    var_29 = [var_28]
    var_30 = {var_27: var_29}
    var_31 = {}
    var_32 = {var_3: var_30, var_4: var_31}
    var_33 = []
    var_34 = []
    var_35 = {var_5: var_33, var_6: var_34}
    var_36 = {}
    var_37 = {var_3: var_35, var_4: var_36}
    var_38 = {var_26: var_32, var_2: var_37}
    var_39 = module_0.ParsedContent()
    var_40 = True
    var_41 = module_1.Config()
    var_42 = module_2.sorted_imports(var_39, var_41)
    var_43 = [var_0]
    var_44 = []
    var_45 = []
    var_46 = {var_5: var_44, var_6: var_45}
    var_47 = [var_12]
    var_48 = [var_14]
    var_49 = {var_10: var_47, var_11: var_48}
    var_50 = {var_3: var_46, var_4: var_49}
    var_51 = {var_2: var_50}
    var_52 = module_0.ParsedContent()
    var_53 = [var_5]
    var_54 = module_1.Config()
    var_55 = module_2.sorted_imports(var_52, var_54)
    var_56 = "print('hello')"
    var_57 = [var_56]
    var_58 = {}
    var_59 = -1
    var_60 = module_0.ParsedContent()
    var_61 = module_2.sorted_imports(var_60)
    assert var_61 == "print('hello')\n"
    var_62 = [var_0]
    var_63 = {}
    var_64 = 'module1'
    var_65 = 'module2'
    var_66 = '*'
    var_67 = [var_66]
    var_68 = 'func'
    var_69 = [var_68]
    var_70 = {var_64: var_67, var_65: var_69}
    var_71 = {var_3: var_63, var_4: var_70}
    var_72 = {var_2: var_71}
    var_73 = module_0.ParsedContent()
    var_74 = True
    var_75 = module_1.Config()
    var_76 = module_2.sorted_imports(var_73, var_75)
    var_77 = 'from module1 import *'
    var_78 = 'from module2 import func'
    var_79 = [var_0]
    var_80 = []
    var_81 = {var_5: var_80}
    var_82 = [var_12]
    var_83 = {var_10: var_82}
    var_84 = {var_3: var_81, var_4: var_83}
    var_85 = {var_2: var_84}
    var_86 = module_0.ParsedContent()
    var_87 = True
    var_88 = module_1.Config()
    var_89 = module_2.sorted_imports(var_86, var_88)
    var_90 = 'from collections import defaultdict'
    var_91 = 'import os'
    var_92 = [var_0]
    var_93 = [var_28]
    var_94 = {var_27: var_93}
    var_95 = {}
    var_96 = {var_3: var_94, var_4: var_95}
    var_97 = []
    var_98 = {var_5: var_97}
    var_99 = {}
    var_100 = {var_3: var_98, var_4: var_99}
    var_101 = {var_26: var_96, var_2: var_100}
    var_102 = module_0.ParsedContent()
    var_103 = 2
    var_104 = module_1.Config()
    var_105 = module_2.sorted_imports(var_102, var_104)
    var_106 = '\n\n'
    var_107 = [var_0]
    var_108 = []
    var_109 = []
    var_110 = {var_5: var_108, var_6: var_109}
    var_111 = {}
    var_112 = {var_3: var_110, var_4: var_111}
    var_113 = {var_2: var_112}
    var_114 = module_0.ParsedContent()
    var_115 = True
    var_116 = module_1.Config()
    var_117 = module_2.sorted_imports(var_114, var_116)
    var_118 = 'import sys'



# Parsed testcases at query #24
#--------------------------


import isort.parse as module_0
import isort.output as module_1
import isort.settings as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = 0
    var_5 = '\n'
    var_6 = {}
    var_7 = {}
    var_8 = module_0.ParsedContent()
    var_9 = module_1.sorted_imports(var_8)
    assert var_9 == ''
    var_10 = "print('hello')"
    var_11 = [var_10]
    var_12 = {}
    var_13 = -1
    var_14 = 1
    var_15 = {}
    var_16 = {}
    var_17 = module_0.ParsedContent()
    var_18 = module_1.sorted_imports(var_17)
    assert var_18 == "print('hello')"
    var_19 = [var_0]
    var_20 = 'THIRDPARTY'
    var_21 = 'straight'
    var_22 = 'from'
    var_23 = 'os'
    var_24 = 'sys'
    var_25 = []
    var_26 = []
    var_27 = {var_23: var_25, var_24: var_26}
    var_28 = 'collections'
    var_29 = 'itertools'
    var_30 = 'defaultdict'
    var_31 = [var_30]
    var_32 = 'chain'
    var_33 = [var_32]
    var_34 = {var_28: var_31, var_29: var_33}
    var_35 = {var_21: var_27, var_22: var_34}
    var_36 = {var_20: var_35}
    var_37 = {}
    var_38 = {}
    var_39 = module_0.ParsedContent()
    var_40 = module_2.Config()
    var_41 = module_1.sorted_imports(var_39, var_40)
    var_42 = [var_0]
    var_43 = []
    var_44 = []
    var_45 = {var_23: var_43, var_24: var_44}
    var_46 = [var_30]
    var_47 = [var_32]
    var_48 = {var_28: var_46, var_29: var_47}
    var_49 = {var_21: var_45, var_22: var_48}
    var_50 = {var_20: var_49}
    var_51 = {}
    var_52 = {}
    var_53 = module_0.ParsedContent()
    var_54 = True
    var_55 = module_2.Config()
    var_56 = module_1.sorted_imports(var_53, var_55)
    var_57 = [var_0]
    var_58 = 'FUTURE'
    var_59 = '__future__'
    var_60 = 'annotations'
    var_61 = [var_60]
    var_62 = {var_59: var_61}
    var_63 = {}
    var_64 = {var_21: var_62, var_22: var_63}
    var_65 = []
    var_66 = []
    var_67 = {var_23: var_65, var_24: var_66}
    var_68 = [var_30]
    var_69 = [var_32]
    var_70 = {var_28: var_68, var_29: var_69}
    var_71 = {var_21: var_67, var_22: var_70}
    var_72 = {var_58: var_64, var_20: var_71}
    var_73 = {}
    var_74 = {}
    var_75 = module_0.ParsedContent()
    var_76 = module_2.Config()
    var_77 = module_1.sorted_imports(var_75, var_76)
    var_78 = [var_0]
    var_79 = [var_60]
    var_80 = {var_59: var_79}
    var_81 = {}
    var_82 = {var_21: var_80, var_22: var_81}
    var_83 = []
    var_84 = []
    var_85 = {var_23: var_83, var_24: var_84}
    var_86 = [var_30]
    var_87 = [var_32]
    var_88 = {var_28: var_86, var_29: var_87}
    var_89 = {var_21: var_85, var_22: var_88}
    var_90 = {var_58: var_82, var_20: var_89}
    var_91 = {}
    var_92 = {}
    var_93 = module_0.ParsedContent()
    var_94 = True
    var_95 = module_2.Config()
    var_96 = module_1.sorted_imports(var_93, var_95)
    var_97 = [var_0]
    var_98 = []
    var_99 = []
    var_100 = {var_23: var_98, var_24: var_99}
    var_101 = [var_30]
    var_102 = [var_32]
    var_103 = {var_28: var_101, var_29: var_102}
    var_104 = {var_21: var_100, var_22: var_103}
    var_105 = {var_20: var_104}
    var_106 = {}
    var_107 = {}
    var_108 = module_0.ParsedContent()
    var_109 = [var_23, var_28]
    var_110 = module_2.Config()
    var_111 = module_1.sorted_imports(var_108, var_110)
    var_112 = [var_0]
    var_113 = []
    var_114 = []
    var_115 = {var_23: var_113, var_24: var_114}
    var_116 = [var_30]
    var_117 = [var_32]
    var_118 = {var_28: var_116, var_29: var_117}
    var_119 = {var_21: var_115, var_22: var_118}
    var_120 = {var_20: var_119}
    var_121 = {}
    var_122 = {}
    var_123 = module_0.ParsedContent()
    var_124 = 2
    var_125 = module_2.Config()
    var_126 = module_1.sorted_imports(var_123, var_125)
    var_127 = [var_0]
    var_128 = [var_60]
    var_129 = {var_59: var_128}
    var_130 = {}
    var_131 = {var_21: var_129, var_22: var_130}
    var_132 = []
    var_133 = []
    var_134 = {var_23: var_132, var_24: var_133}
    var_135 = [var_30]
    var_136 = [var_32]
    var_137 = {var_28: var_135, var_29: var_136}
    var_138 = {var_21: var_134, var_22: var_137}
    var_139 = {var_58: var_131, var_20: var_138}
    var_140 = {}
    var_141 = {}
    var_142 = module_0.ParsedContent()
    var_143 = module_2.Config()
    var_144 = module_1.sorted_imports(var_142, var_143)
    var_145 = [var_0]
    var_146 = []
    var_147 = []
    var_148 = {var_23: var_146, var_24: var_147}
    var_149 = [var_30]
    var_150 = [var_32]
    var_151 = {var_28: var_149, var_29: var_150}
    var_152 = {var_21: var_148, var_22: var_151}
    var_153 = {var_20: var_152}
    var_154 = {}
    var_155 = {}
    var_156 = module_0.ParsedContent()
    var_157 = 'thirdparty'
    var_158 = 'Third Party Imports'
    var_159 = {var_157: var_158}
    var_160 = module_2.Config()
    var_161 = module_1.sorted_imports(var_156, var_160)



# Parsed testcases at query #25
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2
import re as module_3

def test_case_0():
    var_0 = ''
    var_1 = 'def main():'
    var_2 = '    pass'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = 'sys'
    var_9 = []
    var_10 = []
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = 'collections'
    var_13 = 'itertools'
    var_14 = 'defaultdict'
    var_15 = [var_14]
    var_16 = 'chain'
    var_17 = [var_16]
    var_18 = {var_12: var_15, var_13: var_17}
    var_19 = {var_5: var_11, var_6: var_18}
    var_20 = {var_4: var_19}
    var_21 = 0
    var_22 = '\n'
    var_23 = 3
    var_24 = module_0.ParsedContent()
    var_25 = False
    var_26 = module_1.Config()
    var_27 = module_2.sorted_imports(var_24, var_26)
    var_28 = 'import os\nimport sys\n\nfrom collections import defaultdict\nfrom itertools import chain\n\n\ndef main():'
    var_29 = 'pass'
    var_30 = [var_1, var_2]
    var_31 = {}
    var_32 = -1
    var_33 = 2
    var_34 = module_0.ParsedContent()
    var_35 = module_2.sorted_imports(var_34)
    assert var_35 == 'def main():\n    pass'
    var_36 = [var_0, var_1, var_2]
    var_37 = 'FUTURE'
    var_38 = 'STDLIB'
    var_39 = '__future__'
    var_40 = 'annotations'
    var_41 = [var_40]
    var_42 = {var_39: var_41}
    var_43 = {}
    var_44 = {var_5: var_42, var_6: var_43}
    var_45 = []
    var_46 = {var_7: var_45}
    var_47 = 'argv'
    var_48 = [var_47]
    var_49 = {var_8: var_48}
    var_50 = {var_5: var_46, var_6: var_49}
    var_51 = {}
    var_52 = 'django'
    var_53 = 'conf'
    var_54 = [var_53]
    var_55 = {var_52: var_54}
    var_56 = {var_5: var_51, var_6: var_55}
    var_57 = {var_37: var_44, var_38: var_50, var_4: var_56}
    var_58 = module_0.ParsedContent()
    var_59 = 'future'
    var_60 = 'stdlib'
    var_61 = 'Future imports'
    var_62 = 'Standard library'
    var_63 = {var_59: var_61, var_60: var_62}
    var_64 = 1
    var_65 = module_1.Config()
    var_66 = module_2.sorted_imports(var_58, var_65)
    var_67 = [var_0, var_1, var_2]
    var_68 = 'FIRSTPARTY'
    var_69 = 'numpy'
    var_70 = []
    var_71 = {var_69: var_70}
    var_72 = {}
    var_73 = {var_5: var_71, var_6: var_72}
    var_74 = 'my_module'
    var_75 = []
    var_76 = {var_74: var_75}
    var_77 = {}
    var_78 = {var_5: var_76, var_6: var_77}
    var_79 = {var_4: var_73, var_68: var_78}
    var_80 = module_0.ParsedContent()
    var_81 = 'LOCALFOLDER'
    var_82 = [var_81]
    var_83 = module_1.Config()
    var_84 = 'local'
    var_85 = []
    var_86 = {var_84: var_85}
    var_87 = {}
    var_88 = module_2.sorted_imports(var_80, var_83)
    var_89 = [var_0, var_1, var_2]
    var_90 = 'pandas'
    var_91 = []
    var_92 = []
    var_93 = {var_69: var_91, var_90: var_92}
    var_94 = {}
    var_95 = {var_5: var_93, var_6: var_94}
    var_96 = {var_4: var_95}
    var_97 = module_0.ParsedContent()
    var_98 = [var_69]
    var_99 = module_1.Config()
    var_100 = module_2.sorted_imports(var_97, var_99)
    var_101 = [var_0, var_1, var_2]
    var_102 = [var_40]
    var_103 = {var_39: var_102}
    var_104 = {}
    var_105 = {var_5: var_103, var_6: var_104}
    var_106 = []
    var_107 = {var_7: var_106}
    var_108 = {}
    var_109 = {var_5: var_107, var_6: var_108}
    var_110 = []
    var_111 = {var_69: var_110}
    var_112 = {}
    var_113 = {var_5: var_111, var_6: var_112}
    var_114 = {var_37: var_105, var_38: var_109, var_4: var_113}
    var_115 = module_0.ParsedContent()
    var_116 = True
    var_117 = module_1.Config()
    var_118 = module_2.sorted_imports(var_115, var_117)
    var_119 = [var_0, var_1, var_2]
    var_120 = {}
    var_121 = 'scipy'
    var_122 = '*'
    var_123 = [var_122]
    var_124 = 'DataFrame'
    var_125 = [var_124]
    var_126 = [var_122]
    var_127 = {var_69: var_123, var_90: var_125, var_121: var_126}
    var_128 = {var_5: var_120, var_6: var_127}
    var_129 = {var_4: var_128}
    var_130 = module_0.ParsedContent()
    var_131 = True
    var_132 = module_1.Config()
    var_133 = module_2.sorted_imports(var_130, var_132)
    var_134 = module_3.split(var_22)
    var_135 = enumerate(var_134)
    var_136 = [i for (i, line) in var_135 if var_122 in line]
    var_137 = enumerate(var_134)
    var_138 = [i for (i, line) in var_137 if var_124 in line]
    var_139 = [var_0, var_1, var_2]
    var_140 = []
    var_141 = {var_69: var_140}
    var_142 = {}
    var_143 = {var_5: var_141, var_6: var_142}
    var_144 = {var_4: var_143}
    var_145 = module_0.ParsedContent()
    var_146 = module_2.sorted_imports(var_145, var_132)
    var_147 = [var_1, var_2]
    var_148 = []
    var_149 = {var_69: var_148}
    var_150 = {}
    var_151 = {var_5: var_149, var_6: var_150}
    var_152 = {var_4: var_151}
    var_153 = module_0.ParsedContent()
    var_154 = module_1.Config()
    var_155 = module_2.sorted_imports(var_153, var_154)
    var_156 = module_3.split(var_22)



# Parsed testcases at query #26
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = 0
    var_5 = '\n'
    var_6 = {}
    var_7 = {}
    var_8 = module_0.ParsedContent()
    var_9 = "print('hello')"
    var_10 = [var_9]
    var_11 = {}
    var_12 = -1
    var_13 = 1
    var_14 = {}
    var_15 = {}
    var_16 = module_0.ParsedContent()
    var_17 = [var_0]
    var_18 = 'THIRDPARTY'
    var_19 = 'straight'
    var_20 = 'from'
    var_21 = 'os'
    var_22 = 'sys'
    var_23 = [var_21]
    var_24 = [var_22]
    var_25 = {var_21: var_23, var_22: var_24}
    var_26 = 'collections'
    var_27 = 'defaultdict'
    var_28 = 'OrderedDict'
    var_29 = [var_27, var_28]
    var_30 = {var_26: var_29}
    var_31 = {var_19: var_25, var_20: var_30}
    var_32 = {var_18: var_31}
    var_33 = {}
    var_34 = {}
    var_35 = module_0.ParsedContent()
    var_36 = 'from collections import defaultdict, OrderedDict\nimport os\nimport sys\n'
    var_37 = [var_0]
    var_38 = [var_21]
    var_39 = [var_22]
    var_40 = {var_21: var_38, var_22: var_39}
    var_41 = [var_27, var_28]
    var_42 = {var_26: var_41}
    var_43 = {var_19: var_40, var_20: var_42}
    var_44 = {var_18: var_43}
    var_45 = {}
    var_46 = {}
    var_47 = module_0.ParsedContent()
    var_48 = 2
    var_49 = True
    var_50 = module_1.Config()
    var_51 = module_2.sorted_imports(var_47, var_50)
    var_52 = 'from collections import defaultdict, OrderedDict\n\nimport os\nimport sys\n'
    var_53 = [var_0]
    var_54 = [var_21]
    var_55 = [var_22]
    var_56 = {var_21: var_54, var_22: var_55}
    var_57 = [var_27, var_28]
    var_58 = {var_26: var_57}
    var_59 = {var_19: var_56, var_20: var_58}
    var_60 = {var_18: var_59}
    var_61 = {}
    var_62 = {}
    var_63 = module_0.ParsedContent()
    var_64 = [var_21]
    var_65 = module_1.Config()
    var_66 = module_2.sorted_imports(var_63, var_65)
    var_67 = 'from collections import defaultdict, OrderedDict\nimport sys\n'



# Parsed testcases at query #27
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = -1
    var_3 = 0
    var_4 = '\n'
    var_5 = {}
    var_6 = {}
    var_7 = []
    var_8 = module_0.ParsedContent()
    var_9 = module_1.Config()
    var_10 = module_2.sorted_imports(var_8, var_9)
    assert var_10 == ''
    var_11 = "print('hello')"
    var_12 = [var_11]
    var_13 = {}
    var_14 = -1
    var_15 = 1
    var_16 = {}
    var_17 = {}
    var_18 = []
    var_19 = module_0.ParsedContent()
    var_20 = module_1.Config()
    var_21 = module_2.sorted_imports(var_19, var_20)
    assert var_21 == "print('hello')\n"
    var_22 = [var_11]
    var_23 = 'THIRDPARTY'
    var_24 = 'straight'
    var_25 = 'from'
    var_26 = 'os'
    var_27 = 'sys'
    var_28 = [var_26]
    var_29 = [var_27]
    var_30 = {var_26: var_28, var_27: var_29}
    var_31 = 'collections'
    var_32 = 'defaultdict'
    var_33 = 'OrderedDict'
    var_34 = [var_32, var_33]
    var_35 = {var_31: var_34}
    var_36 = {var_24: var_30, var_25: var_35}
    var_37 = {var_23: var_36}
    var_38 = {}
    var_39 = {}
    var_40 = [var_23]
    var_41 = module_0.ParsedContent()
    var_42 = module_1.Config()
    var_43 = module_2.sorted_imports(var_41, var_42)
    var_44 = "from collections import defaultdict, OrderedDict\nimport os\nimport sys\n\nprint('hello')\n"
    var_45 = [var_11]
    var_46 = [var_26]
    var_47 = [var_27]
    var_48 = {var_26: var_46, var_27: var_47}
    var_49 = [var_32, var_33]
    var_50 = {var_31: var_49}
    var_51 = {var_24: var_48, var_25: var_50}
    var_52 = {var_23: var_51}
    var_53 = {}
    var_54 = {}
    var_55 = [var_23]
    var_56 = module_0.ParsedContent()
    var_57 = 'FUTURE'
    var_58 = [var_57]
    var_59 = module_1.Config()
    var_60 = module_2.sorted_imports(var_56, var_59)
    var_61 = "from collections import defaultdict, OrderedDict\nimport os\nimport sys\n\nprint('hello')\n"
    var_62 = [var_11]
    var_63 = [var_26]
    var_64 = [var_27]
    var_65 = {var_26: var_63, var_27: var_64}
    var_66 = [var_32, var_33]
    var_67 = {var_31: var_66}
    var_68 = {var_24: var_65, var_25: var_67}
    var_69 = '__future__'
    var_70 = 'annotations'
    var_71 = [var_70]
    var_72 = {var_69: var_71}
    var_73 = {}
    var_74 = {var_24: var_72, var_25: var_73}
    var_75 = {var_23: var_68, var_57: var_74}
    var_76 = {}
    var_77 = {}
    var_78 = [var_23, var_57]
    var_79 = module_0.ParsedContent()
    var_80 = True
    var_81 = module_1.Config()
    var_82 = module_2.sorted_imports(var_79, var_81)
    var_83 = "from __future__ import annotations\nfrom collections import defaultdict, OrderedDict\nimport os\nimport sys\n\nprint('hello')\n"
    var_84 = [var_11]
    var_85 = [var_26]
    var_86 = [var_27]
    var_87 = {var_26: var_85, var_27: var_86}
    var_88 = [var_32, var_33]
    var_89 = {var_31: var_88}
    var_90 = {var_24: var_87, var_25: var_89}
    var_91 = {var_23: var_90}
    var_92 = {}
    var_93 = {}
    var_94 = [var_23]
    var_95 = module_0.ParsedContent()
    var_96 = True
    var_97 = module_1.Config()
    var_98 = module_2.sorted_imports(var_95, var_97)
    var_99 = "from collections import defaultdict, OrderedDict\nimport sys\nimport os\n\nprint('hello')\n"
    var_100 = [var_11]
    var_101 = {}
    var_102 = [var_32]
    var_103 = '*'
    var_104 = [var_103]
    var_105 = 'path'
    var_106 = [var_105]
    var_107 = {var_31: var_102, var_26: var_104, var_27: var_106}
    var_108 = {var_24: var_101, var_25: var_107}
    var_109 = {var_23: var_108}
    var_110 = {}
    var_111 = {}
    var_112 = [var_23]
    var_113 = module_0.ParsedContent()
    var_114 = True
    var_115 = module_1.Config()
    var_116 = module_2.sorted_imports(var_113, var_115)
    var_117 = "from os import *\nfrom collections import defaultdict\nfrom sys import path\n\nprint('hello')\n"
    var_118 = [var_11]
    var_119 = [var_26]
    var_120 = [var_27]
    var_121 = {var_26: var_119, var_27: var_120}
    var_122 = [var_32, var_33]
    var_123 = {var_31: var_122}
    var_124 = {var_24: var_121, var_25: var_123}
    var_125 = {var_23: var_124}
    var_126 = {}
    var_127 = {}
    var_128 = [var_23]
    var_129 = module_0.ParsedContent()
    var_130 = True
    var_131 = module_1.Config()
    var_132 = module_2.sorted_imports(var_129, var_131)
    var_133 = "from collections import defaultdict, OrderedDict\n\nimport os\nimport sys\n\nprint('hello')\n"
    var_134 = [var_11]
    var_135 = [var_26]
    var_136 = [var_27]
    var_137 = {var_26: var_135, var_27: var_136}
    var_138 = [var_32, var_33]
    var_139 = {var_31: var_138}
    var_140 = {var_24: var_137, var_25: var_139}
    var_141 = {var_23: var_140}
    var_142 = {}
    var_143 = {}
    var_144 = [var_23]
    var_145 = module_0.ParsedContent()
    var_146 = 'thirdparty'
    var_147 = 'Third Party Imports'
    var_148 = {var_146: var_147}
    var_149 = module_1.Config()
    var_150 = module_2.sorted_imports(var_145, var_149)
    var_151 = "# Third Party Imports\nfrom collections import defaultdict, OrderedDict\nimport os\nimport sys\n\nprint('hello')\n"
    var_152 = [var_11]
    var_153 = 'FIRSTPARTY'
    var_154 = [var_26]
    var_155 = [var_27]
    var_156 = {var_26: var_154, var_27: var_155}
    var_157 = [var_32, var_33]
    var_158 = {var_31: var_157}
    var_159 = {var_24: var_156, var_25: var_158}
    var_160 = 'my_module'
    var_161 = [var_160]
    var_162 = {var_160: var_161}
    var_163 = {}
    var_164 = {var_24: var_162, var_25: var_163}
    var_165 = {var_23: var_159, var_153: var_164}
    var_166 = {}
    var_167 = {}
    var_168 = [var_23, var_153]
    var_169 = module_0.ParsedContent()
    var_170 = 2
    var_171 = module_1.Config()
    var_172 = module_2.sorted_imports(var_169, var_171)
    var_173 = "from collections import defaultdict, OrderedDict\nimport os\nimport sys\n\n\nimport my_module\n\nprint('hello')\n"



# Parsed testcases at query #28
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2
import re as module_3

def test_case_0():
    var_0 = 'x = 1'
    var_1 = [var_0]
    var_2 = 'THIRDPARTY'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = 'sys'
    var_7 = []
    var_8 = []
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'collections'
    var_11 = 'defaultdict'
    var_12 = [var_11]
    var_13 = {var_10: var_12}
    var_14 = {var_3: var_9, var_4: var_13}
    var_15 = {var_2: var_14}
    var_16 = 0
    var_17 = 1
    var_18 = '\n'
    var_19 = module_0.ParsedContent()
    var_20 = module_1.Config()
    var_21 = module_2.sorted_imports(var_19, var_20)
    assert var_21 == 'import os\nimport sys\n\nfrom collections import defaultdict\n\nx = 1'
    var_22 = [var_0]
    var_23 = {}
    var_24 = -1
    var_25 = module_0.ParsedContent()
    var_26 = module_2.sorted_imports(var_25, var_20)
    assert var_26 == 'x = 1'
    var_27 = 'LOCALFOLDER'
    var_28 = [var_0]
    var_29 = []
    var_30 = []
    var_31 = {var_5: var_29, var_6: var_30}
    var_32 = [var_11]
    var_33 = {var_10: var_32}
    var_34 = {var_3: var_31, var_4: var_33}
    var_35 = 'local'
    var_36 = []
    var_37 = {var_35: var_36}
    var_38 = {}
    var_39 = {var_3: var_37, var_4: var_38}
    var_40 = {var_2: var_34, var_27: var_39}
    var_41 = module_0.ParsedContent()
    var_42 = module_2.sorted_imports(var_41, var_20)
    var_43 = [var_0]
    var_44 = 'FUTURE'
    var_45 = '__future__'
    var_46 = 'annotations'
    var_47 = [var_46]
    var_48 = {var_45: var_47}
    var_49 = {}
    var_50 = {var_3: var_48, var_4: var_49}
    var_51 = []
    var_52 = []
    var_53 = {var_5: var_51, var_6: var_52}
    var_54 = [var_11]
    var_55 = {var_10: var_54}
    var_56 = {var_3: var_53, var_4: var_55}
    var_57 = {var_44: var_50, var_2: var_56}
    var_58 = module_0.ParsedContent()
    var_59 = module_2.sorted_imports(var_58, var_20)
    var_60 = [var_0]
    var_61 = []
    var_62 = []
    var_63 = {var_5: var_61, var_6: var_62}
    var_64 = [var_11]
    var_65 = {var_10: var_64}
    var_66 = {var_3: var_63, var_4: var_65}
    var_67 = {var_2: var_66}
    var_68 = module_0.ParsedContent()
    var_69 = module_2.sorted_imports(var_68, var_20)
    var_70 = [var_0]
    var_71 = {}
    var_72 = 'module1'
    var_73 = 'module2'
    var_74 = '*'
    var_75 = [var_74]
    var_76 = 'func'
    var_77 = [var_76]
    var_78 = {var_72: var_75, var_73: var_77}
    var_79 = {var_3: var_71, var_4: var_78}
    var_80 = {var_2: var_79}
    var_81 = module_0.ParsedContent()
    var_82 = module_2.sorted_imports(var_81, var_20)
    var_83 = 'from module1 import *'
    var_84 = 'from module2 import func'
    var_85 = [var_0]
    var_86 = []
    var_87 = {var_5: var_86}
    var_88 = [var_11]
    var_89 = {var_10: var_88}
    var_90 = {var_3: var_87, var_4: var_89}
    var_91 = {var_2: var_90}
    var_92 = module_0.ParsedContent()
    var_93 = module_2.sorted_imports(var_92, var_20)
    var_94 = 'from collections import defaultdict'
    var_95 = 'import os'
    var_96 = 'thirdparty'
    var_97 = 'Third Party Imports'
    var_98 = [var_0]
    var_99 = []
    var_100 = {var_5: var_99}
    var_101 = {}
    var_102 = {var_3: var_100, var_4: var_101}
    var_103 = {var_2: var_102}
    var_104 = module_0.ParsedContent()
    var_105 = module_2.sorted_imports(var_104, var_20)
    var_106 = [var_0]
    var_107 = [var_46]
    var_108 = {var_45: var_107}
    var_109 = {}
    var_110 = {var_3: var_108, var_4: var_109}
    var_111 = []
    var_112 = {var_5: var_111}
    var_113 = {}
    var_114 = {var_3: var_112, var_4: var_113}
    var_115 = {var_44: var_110, var_2: var_114}
    var_116 = module_0.ParsedContent()
    var_117 = module_2.sorted_imports(var_116, var_20)
    var_118 = '\n\n'
    var_119 = [var_0]
    var_120 = []
    var_121 = {var_5: var_120}
    var_122 = {}
    var_123 = {var_3: var_121, var_4: var_122}
    var_124 = {var_2: var_123}
    var_125 = module_0.ParsedContent()
    var_126 = module_2.sorted_imports(var_125, var_20)
    var_127 = module_3.split(var_18)
    var_128 = [var_0]
    var_129 = []
    var_130 = []
    var_131 = {var_5: var_129, var_6: var_130}
    var_132 = {}
    var_133 = {var_3: var_131, var_4: var_132}
    var_134 = {var_2: var_133}
    var_135 = module_0.ParsedContent()
    var_136 = module_2.sorted_imports(var_135, var_20)



# Parsed testcases at query #29
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = 'x = 1'
    var_2 = [var_0, var_1]
    var_3 = 'THIRDPARTY'
    var_4 = 'straight'
    var_5 = 'from'
    var_6 = 'os'
    var_7 = 'sys'
    var_8 = [var_6]
    var_9 = [var_7]
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'collections'
    var_12 = 'OrderedDict'
    var_13 = 'defaultdict'
    var_14 = [var_12, var_13]
    var_15 = {var_11: var_14}
    var_16 = {var_4: var_10, var_5: var_15}
    var_17 = {var_3: var_16}
    var_18 = 0
    var_19 = 2
    var_20 = '\n'
    var_21 = module_0.ParsedContent()
    var_22 = module_1.Config()
    var_23 = module_2.sorted_imports(var_21, var_22)
    assert var_23 == 'import os\nimport sys\n\nfrom collections import OrderedDict, defaultdict\n\nx = 1\n'
    var_24 = [var_1]
    var_25 = {}
    var_26 = -1
    var_27 = 1
    var_28 = module_0.ParsedContent()
    var_29 = module_1.Config()
    var_30 = module_2.sorted_imports(var_28, var_29)
    assert var_30 == 'x = 1\n'
    var_31 = [var_0, var_1]
    var_32 = 'FUTURE'
    var_33 = 'STDLIB'
    var_34 = '__future__'
    var_35 = [var_34]
    var_36 = {var_34: var_35}
    var_37 = {}
    var_38 = {var_4: var_36, var_5: var_37}
    var_39 = [var_6]
    var_40 = {var_6: var_39}
    var_41 = {}
    var_42 = {var_4: var_40, var_5: var_41}
    var_43 = {var_32: var_38, var_33: var_42}
    var_44 = module_0.ParsedContent()
    var_45 = 'future'
    var_46 = 'stdlib'
    var_47 = 'Future'
    var_48 = 'Standard Library'
    var_49 = {var_45: var_47, var_46: var_48}
    var_50 = module_1.Config()
    var_51 = module_2.sorted_imports(var_44, var_50)
    var_52 = [var_0, var_1]
    var_53 = 'django'
    var_54 = 'flask'
    var_55 = [var_53]
    var_56 = [var_54]
    var_57 = {var_53: var_55, var_54: var_56}
    var_58 = {}
    var_59 = {var_4: var_57, var_5: var_58}
    var_60 = {var_3: var_59}
    var_61 = module_0.ParsedContent()
    var_62 = [var_53]
    var_63 = module_1.Config()
    var_64 = module_2.sorted_imports(var_61, var_63)
    assert var_64 == 'import django\nimport flask\n\nx = 1\n'
    var_65 = [var_0, var_1]
    var_66 = [var_34]
    var_67 = {var_34: var_66}
    var_68 = {}
    var_69 = {var_4: var_67, var_5: var_68}
    var_70 = [var_6]
    var_71 = {var_6: var_70}
    var_72 = {}
    var_73 = {var_4: var_71, var_5: var_72}
    var_74 = {var_32: var_69, var_33: var_73}
    var_75 = module_0.ParsedContent()
    var_76 = True
    var_77 = module_1.Config()
    var_78 = module_2.sorted_imports(var_75, var_77)
    assert var_78 == 'from __future__ import __future__\nimport os\n\nx = 1\n'
    var_79 = [var_0, var_1]
    var_80 = [var_6]
    var_81 = [var_7]
    var_82 = {var_6: var_80, var_7: var_81}
    var_83 = [var_12, var_13]
    var_84 = {var_11: var_83}
    var_85 = {var_4: var_82, var_5: var_84}
    var_86 = {var_3: var_85}
    var_87 = module_0.ParsedContent()
    var_88 = [var_6]
    var_89 = module_1.Config()
    var_90 = module_2.sorted_imports(var_87, var_89)
    assert var_90 == 'import sys\n\nfrom collections import OrderedDict, defaultdict\n\nx = 1\n'
    var_91 = [var_0, var_1]
    var_92 = {}
    var_93 = 'module'
    var_94 = '*'
    var_95 = 'func'
    var_96 = [var_94, var_95]
    var_97 = {var_93: var_96}
    var_98 = {var_4: var_92, var_5: var_97}
    var_99 = {var_3: var_98}
    var_100 = module_0.ParsedContent()
    var_101 = True
    var_102 = module_1.Config()
    var_103 = module_2.sorted_imports(var_100, var_102)
    assert var_103 == 'from module import *\nfrom module import func\n\nx = 1\n'
    var_104 = [var_0, var_1]
    var_105 = [var_6]
    var_106 = {var_6: var_105}
    var_107 = [var_12]
    var_108 = {var_11: var_107}
    var_109 = {var_4: var_106, var_5: var_108}
    var_110 = {var_3: var_109}
    var_111 = module_0.ParsedContent()
    var_112 = True
    var_113 = module_1.Config()
    var_114 = module_2.sorted_imports(var_111, var_113)
    assert var_114 == 'from collections import OrderedDict\n\nimport os\n\nx = 1\n'
    var_115 = [var_0, var_1]
    var_116 = [var_6]
    var_117 = {var_6: var_116}
    var_118 = [var_12]
    var_119 = {var_11: var_118}
    var_120 = {var_4: var_117, var_5: var_119}
    var_121 = {var_3: var_120}
    var_122 = module_0.ParsedContent()
    var_123 = module_1.Config()
    var_124 = module_2.sorted_imports(var_122, var_123)
    assert var_124 == 'import os\n\n\nfrom collections import OrderedDict\n\nx = 1\n'
    var_125 = [var_0, var_1]
    var_126 = [var_34]
    var_127 = {var_34: var_126}
    var_128 = {}
    var_129 = {var_4: var_127, var_5: var_128}
    var_130 = [var_6]
    var_131 = {var_6: var_130}
    var_132 = {}
    var_133 = {var_4: var_131, var_5: var_132}
    var_134 = {var_32: var_129, var_33: var_133}
    var_135 = module_0.ParsedContent()
    var_136 = 3
    var_137 = module_1.Config()
    var_138 = module_2.sorted_imports(var_135, var_137)
    assert var_138 == 'from __future__ import __future__\n\n\n\nimport os\n\nx = 1\n'



# Parsed testcases at query #30
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = -1
    var_3 = '\n'
    var_4 = 1
    var_5 = []
    var_6 = {}
    var_7 = {}
    var_8 = {}
    var_9 = module_0.ParsedContent()
    var_10 = module_1.Config()
    var_11 = module_2.sorted_imports(var_9, var_10)
    assert var_11 == "print('hello')"
    var_12 = [var_0]
    var_13 = 0
    var_14 = 2
    var_15 = 'THIRDPARTY'
    var_16 = [var_15]
    var_17 = 'straight'
    var_18 = 'from'
    var_19 = 'os'
    var_20 = 'sys'
    var_21 = [var_19]
    var_22 = [var_20]
    var_23 = {var_19: var_21, var_20: var_22}
    var_24 = 'collections'
    var_25 = 'OrderedDict'
    var_26 = [var_25]
    var_27 = {var_24: var_26}
    var_28 = {var_17: var_23, var_18: var_27}
    var_29 = {var_15: var_28}
    var_30 = {}
    var_31 = {}
    var_32 = module_0.ParsedContent()
    var_33 = module_1.Config()
    var_34 = module_2.sorted_imports(var_32, var_33)
    assert var_34 == "import os\nimport sys\nfrom collections import OrderedDict\n\nprint('hello')"
    var_35 = [var_0]
    var_36 = [var_15]
    var_37 = [var_19]
    var_38 = [var_20]
    var_39 = {var_19: var_37, var_20: var_38}
    var_40 = [var_25]
    var_41 = {var_24: var_40}
    var_42 = {var_17: var_39, var_18: var_41}
    var_43 = {var_15: var_42}
    var_44 = {}
    var_45 = {}
    var_46 = module_0.ParsedContent()
    var_47 = True
    var_48 = module_1.Config()
    var_49 = module_2.sorted_imports(var_46, var_48)
    assert var_49 == "import os\nimport sys\nfrom collections import OrderedDict\n\nprint('hello')"
    var_50 = [var_0]
    var_51 = [var_15]
    var_52 = [var_19]
    var_53 = [var_20]
    var_54 = {var_19: var_52, var_20: var_53}
    var_55 = [var_25]
    var_56 = {var_24: var_55}
    var_57 = {var_17: var_54, var_18: var_56}
    var_58 = {var_15: var_57}
    var_59 = {}
    var_60 = {}
    var_61 = module_0.ParsedContent()
    var_62 = True
    var_63 = module_1.Config()
    var_64 = module_2.sorted_imports(var_61, var_63)
    assert var_64 == "import os\nimport sys\nfrom collections import OrderedDict\n\nprint('hello')"
    var_65 = [var_0]
    var_66 = 'FIRSTPARTY'
    var_67 = [var_15, var_66]
    var_68 = [var_19]
    var_69 = {var_19: var_68}
    var_70 = {}
    var_71 = {var_17: var_69, var_18: var_70}
    var_72 = [var_20]
    var_73 = {var_20: var_72}
    var_74 = {}
    var_75 = {var_17: var_73, var_18: var_74}
    var_76 = {var_15: var_71, var_66: var_75}
    var_77 = {}
    var_78 = {}
    var_79 = module_0.ParsedContent()
    var_80 = module_1.Config()
    var_81 = module_2.sorted_imports(var_79, var_80)
    assert var_81 == "import os\n\n\nimport sys\n\nprint('hello')"
    var_82 = [var_0]
    var_83 = [var_15]
    var_84 = [var_19]
    var_85 = {var_19: var_84}
    var_86 = {}
    var_87 = {var_17: var_85, var_18: var_86}
    var_88 = {var_15: var_87}
    var_89 = {}
    var_90 = {}
    var_91 = module_0.ParsedContent()
    var_92 = 'thirdparty'
    var_93 = 'Third Party Imports'
    var_94 = {var_92: var_93}
    var_95 = module_1.Config()
    var_96 = module_2.sorted_imports(var_91, var_95)
    assert var_96 == "# Third Party Imports\nimport os\n\nprint('hello')"
    var_97 = [var_0]
    var_98 = [var_15]
    var_99 = [var_19]
    var_100 = {var_19: var_99}
    var_101 = {}
    var_102 = {var_17: var_100, var_18: var_101}
    var_103 = {var_15: var_102}
    var_104 = {}
    var_105 = {}
    var_106 = module_0.ParsedContent()
    var_107 = module_1.Config()
    var_108 = module_2.sorted_imports(var_106, var_107)
    assert var_108 == "import os\n\n\nprint('hello')"
    var_109 = "print('world')"
    var_110 = [var_0, var_109]
    var_111 = 3
    var_112 = [var_15]
    var_113 = [var_19]
    var_114 = {var_19: var_113}
    var_115 = {}
    var_116 = {var_17: var_114, var_18: var_115}
    var_117 = {var_15: var_116}
    var_118 = 'import sys'
    var_119 = [var_118]
    var_120 = {var_15: var_119}
    var_121 = {var_0: var_15}
    var_122 = module_0.ParsedContent()
    var_123 = module_1.Config()
    var_124 = module_2.sorted_imports(var_122, var_123)
    assert var_124 == "import os\n\nprint('hello')\nimport sys\n\nprint('world')"



# Parsed testcases at query #31
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 'THIRDPARTY'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = 'sys'
    var_7 = set()
    var_8 = set()
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'collections'
    var_11 = 'defaultdict'
    var_12 = 'OrderedDict'
    var_13 = {var_11, var_12}
    var_14 = {var_10: var_13}
    var_15 = {var_3: var_9, var_4: var_14}
    var_16 = {var_2: var_15}
    var_17 = 0
    var_18 = 1
    var_19 = '\n'
    var_20 = module_0.ParsedContent()
    var_21 = False
    var_22 = module_1.Config()
    var_23 = module_2.sorted_imports(var_20, var_22)
    assert var_23 == 'import os\nimport sys\n\nfrom collections import OrderedDict, defaultdict\n'
    var_24 = "print('hello')"
    var_25 = [var_24]
    var_26 = {}
    var_27 = -1
    var_28 = module_0.ParsedContent()
    var_29 = module_2.sorted_imports(var_28, var_22)
    assert var_29 == "print('hello')\n"
    var_30 = 'LOCALFOLDER'
    var_31 = [var_30]
    var_32 = module_1.Config()
    var_33 = [var_0]
    var_34 = set()
    var_35 = {var_5: var_34}
    var_36 = {}
    var_37 = {var_3: var_35, var_4: var_36}
    var_38 = set()
    var_39 = {var_6: var_38}
    var_40 = {}
    var_41 = {var_3: var_39, var_4: var_40}
    var_42 = {var_2: var_37, var_30: var_41}
    var_43 = module_0.ParsedContent()
    var_44 = module_2.sorted_imports(var_43, var_32)
    var_45 = True
    var_46 = module_1.Config()
    var_47 = [var_0]
    var_48 = {}
    var_49 = 'module1'
    var_50 = 'module2'
    var_51 = '*'
    var_52 = {var_51}
    var_53 = 'function1'
    var_54 = {var_53}
    var_55 = {var_49: var_52, var_50: var_54}
    var_56 = {var_3: var_48, var_4: var_55}
    var_57 = {var_2: var_56}
    var_58 = module_0.ParsedContent()
    var_59 = module_2.sorted_imports(var_58, var_46)
    var_60 = 'from module1 import *'
    var_61 = 'from module2 import function1'
    var_62 = 'thirdparty'
    var_63 = 'Third Party Imports'
    var_64 = {var_62: var_63}
    var_65 = module_1.Config()
    var_66 = [var_0]
    var_67 = set()
    var_68 = {var_5: var_67}
    var_69 = {}
    var_70 = {var_3: var_68, var_4: var_69}
    var_71 = {var_2: var_70}
    var_72 = module_0.ParsedContent()
    var_73 = module_2.sorted_imports(var_72, var_65)
    var_74 = 2
    var_75 = module_1.Config()
    var_76 = [var_0]
    var_77 = 'FUTURE'
    var_78 = '__future__'
    var_79 = 'print_function'
    var_80 = {var_79}
    var_81 = {var_78: var_80}
    var_82 = {}
    var_83 = {var_3: var_81, var_4: var_82}
    var_84 = set()
    var_85 = {var_5: var_84}
    var_86 = {}
    var_87 = {var_3: var_85, var_4: var_86}
    var_88 = {var_77: var_83, var_2: var_87}
    var_89 = module_0.ParsedContent()
    var_90 = module_2.sorted_imports(var_89, var_75)
    var_91 = '\n\n'
    var_92 = 'import sys'
    var_93 = [var_92]
    var_94 = module_1.Config()
    var_95 = [var_0]
    var_96 = set()
    var_97 = set()
    var_98 = {var_5: var_96, var_6: var_97}
    var_99 = {}
    var_100 = {var_3: var_98, var_4: var_99}
    var_101 = {var_2: var_100}
    var_102 = module_0.ParsedContent()
    var_103 = module_2.sorted_imports(var_102, var_94)
    var_104 = [var_0]
    var_105 = set()
    var_106 = {var_5: var_105}
    var_107 = {}
    var_108 = {var_3: var_106, var_4: var_107}
    var_109 = {var_2: var_108}
    var_110 = module_0.ParsedContent()



# Parsed testcases at query #32
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = 0
    var_5 = '\n'
    var_6 = {}
    var_7 = {}
    var_8 = module_0.ParsedContent()
    var_9 = "print('hello')"
    var_10 = [var_9]
    var_11 = {}
    var_12 = -1
    var_13 = 1
    var_14 = {}
    var_15 = {}
    var_16 = module_0.ParsedContent()
    var_17 = [var_0]
    var_18 = 'THIRDPARTY'
    var_19 = 'straight'
    var_20 = 'from'
    var_21 = 'os'
    var_22 = 'sys'
    var_23 = [var_21]
    var_24 = [var_22]
    var_25 = {var_21: var_23, var_22: var_24}
    var_26 = 'collections'
    var_27 = 'defaultdict'
    var_28 = 'OrderedDict'
    var_29 = [var_27, var_28]
    var_30 = {var_26: var_29}
    var_31 = {var_19: var_25, var_20: var_30}
    var_32 = {var_18: var_31}
    var_33 = {}
    var_34 = {}
    var_35 = module_0.ParsedContent()
    var_36 = 'import os\nimport sys\n\nfrom collections import defaultdict, OrderedDict\n\n'
    var_37 = [var_0]
    var_38 = [var_21]
    var_39 = [var_22]
    var_40 = {var_21: var_38, var_22: var_39}
    var_41 = [var_27, var_28]
    var_42 = {var_26: var_41}
    var_43 = {var_19: var_40, var_20: var_42}
    var_44 = {var_18: var_43}
    var_45 = {}
    var_46 = {}
    var_47 = module_0.ParsedContent()
    var_48 = 2
    var_49 = True
    var_50 = True
    var_51 = module_1.Config()
    var_52 = module_2.sorted_imports(var_47, var_51)
    var_53 = 'from collections import defaultdict, OrderedDict\n\nimport sys\nimport os\n\n\n'
    var_54 = [var_0]
    var_55 = 'FIRSTPARTY'
    var_56 = [var_21]
    var_57 = [var_22]
    var_58 = {var_21: var_56, var_22: var_57}
    var_59 = [var_27, var_28]
    var_60 = {var_26: var_59}
    var_61 = {var_19: var_58, var_20: var_60}
    var_62 = 'my_module'
    var_63 = [var_62]
    var_64 = {var_62: var_63}
    var_65 = {}
    var_66 = {var_19: var_64, var_20: var_65}
    var_67 = {var_18: var_61, var_55: var_66}
    var_68 = {}
    var_69 = {}
    var_70 = module_0.ParsedContent()
    var_71 = 'LOCALFOLDER'
    var_72 = [var_71]
    var_73 = module_1.Config()
    var_74 = module_2.sorted_imports(var_70, var_73)
    var_75 = 'import os\nimport sys\n\nfrom collections import defaultdict, OrderedDict\n\nimport my_module\n\n'



# Parsed testcases at query #33
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 'THIRDPARTY'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = 'sys'
    var_7 = []
    var_8 = []
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'collections'
    var_11 = 'itertools'
    var_12 = 'defaultdict'
    var_13 = [var_12]
    var_14 = 'chain'
    var_15 = [var_14]
    var_16 = {var_10: var_13, var_11: var_15}
    var_17 = {var_3: var_9, var_4: var_16}
    var_18 = {var_2: var_17}
    var_19 = 0
    var_20 = 1
    var_21 = '\n'
    var_22 = {}
    var_23 = {}
    var_24 = module_0.ParsedContent()
    var_25 = module_1.Config()
    var_26 = module_2.sorted_imports(var_24, var_25)
    var_27 = [var_0]
    var_28 = 'FUTURE'
    var_29 = '__future__'
    var_30 = 'print_function'
    var_31 = [var_30]
    var_32 = {var_29: var_31}
    var_33 = {}
    var_34 = {var_3: var_32, var_4: var_33}
    var_35 = []
    var_36 = []
    var_37 = {var_5: var_35, var_6: var_36}
    var_38 = [var_12]
    var_39 = {var_10: var_38}
    var_40 = {var_3: var_37, var_4: var_39}
    var_41 = {var_28: var_34, var_2: var_40}
    var_42 = {}
    var_43 = {}
    var_44 = module_0.ParsedContent()
    var_45 = True
    var_46 = module_1.Config()
    var_47 = module_2.sorted_imports(var_44, var_46)
    var_48 = [var_0]
    var_49 = []
    var_50 = {var_5: var_49}
    var_51 = 'argv'
    var_52 = [var_51]
    var_53 = {var_6: var_52}
    var_54 = {var_3: var_50, var_4: var_53}
    var_55 = {var_2: var_54}
    var_56 = {}
    var_57 = {}
    var_58 = module_0.ParsedContent()
    var_59 = True
    var_60 = module_1.Config()
    var_61 = module_2.sorted_imports(var_58, var_60)
    var_62 = 'from sys import argv'
    var_63 = 'import os'
    var_64 = [var_0]
    var_65 = {}
    var_66 = '*'
    var_67 = [var_66]
    var_68 = [var_51]
    var_69 = {var_5: var_67, var_6: var_68}
    var_70 = {var_3: var_65, var_4: var_69}
    var_71 = {var_2: var_70}
    var_72 = {}
    var_73 = {}
    var_74 = module_0.ParsedContent()
    var_75 = True
    var_76 = module_1.Config()
    var_77 = module_2.sorted_imports(var_74, var_76)
    var_78 = 'from os import *'
    var_79 = [var_0]
    var_80 = 'zlib'
    var_81 = []
    var_82 = []
    var_83 = {var_80: var_81, var_5: var_82}
    var_84 = [var_51]
    var_85 = [var_12]
    var_86 = {var_6: var_84, var_10: var_85}
    var_87 = {var_3: var_83, var_4: var_86}
    var_88 = {var_2: var_87}
    var_89 = {}
    var_90 = {}
    var_91 = module_0.ParsedContent()
    var_92 = True
    var_93 = module_1.Config()
    var_94 = module_2.sorted_imports(var_91, var_93)
    var_95 = 'import zlib'
    var_96 = 'from collections import defaultdict'
    var_97 = [var_0]
    var_98 = []
    var_99 = []
    var_100 = {var_5: var_98, var_6: var_99}
    var_101 = [var_12]
    var_102 = {var_10: var_101}
    var_103 = {var_3: var_100, var_4: var_102}
    var_104 = {var_2: var_103}
    var_105 = {}
    var_106 = {}
    var_107 = module_0.ParsedContent()
    var_108 = [var_6, var_96]
    var_109 = module_1.Config()
    var_110 = module_2.sorted_imports(var_107, var_109)
    var_111 = [var_0]
    var_112 = []
    var_113 = {var_5: var_112}
    var_114 = {}
    var_115 = {var_3: var_113, var_4: var_114}
    var_116 = {var_2: var_115}
    var_117 = {}
    var_118 = {}
    var_119 = module_0.ParsedContent()
    var_120 = 'thirdparty'
    var_121 = 'Third Party Imports'
    var_122 = {var_120: var_121}
    var_123 = module_1.Config()
    var_124 = module_2.sorted_imports(var_119, var_123)
    var_125 = '# Third Party Imports'
    var_126 = "print('hello')"
    var_127 = [var_126]
    var_128 = {}
    var_129 = -1
    var_130 = {}
    var_131 = {}
    var_132 = module_0.ParsedContent()
    var_133 = module_1.Config()
    var_134 = module_2.sorted_imports(var_132, var_133)
    assert var_134 == "print('hello')"
    var_135 = '# PLACEHOLDER'
    var_136 = [var_135, var_126]
    var_137 = []
    var_138 = {var_5: var_137}
    var_139 = {}
    var_140 = {var_3: var_138, var_4: var_139}
    var_141 = {var_2: var_140}
    var_142 = 2
    var_143 = [var_63]
    var_144 = {var_2: var_143}
    var_145 = {var_135: var_2}
    var_146 = module_0.ParsedContent()
    var_147 = module_1.Config()
    var_148 = module_2.sorted_imports(var_146, var_147)
    var_149 = [var_0]
    var_150 = []
    var_151 = {var_5: var_150}
    var_152 = {}
    var_153 = {var_3: var_151, var_4: var_152}
    var_154 = {var_2: var_153}
    var_155 = {}
    var_156 = {}
    var_157 = module_0.ParsedContent()



# Parsed testcases at query #34
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'x = 1'
    var_1 = [var_0]
    var_2 = 'THIRDPARTY'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'zlib'
    var_6 = 'os'
    var_7 = []
    var_8 = []
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'sys'
    var_11 = 'json'
    var_12 = 'path'
    var_13 = [var_12]
    var_14 = 'load'
    var_15 = [var_14]
    var_16 = {var_10: var_13, var_11: var_15}
    var_17 = {var_3: var_9, var_4: var_16}
    var_18 = {var_2: var_17}
    var_19 = 0
    var_20 = 1
    var_21 = '\n'
    var_22 = module_0.ParsedContent()
    var_23 = False
    var_24 = module_1.Config()
    var_25 = module_2.sorted_imports(var_22, var_24)
    assert var_25 == 'import os\nimport zlib\n\nfrom json import load\nfrom sys import path\n\nx = 1\n'
    var_26 = [var_0]
    var_27 = []
    var_28 = []
    var_29 = {var_5: var_27, var_6: var_28}
    var_30 = [var_12]
    var_31 = [var_14]
    var_32 = {var_10: var_30, var_11: var_31}
    var_33 = {var_3: var_29, var_4: var_32}
    var_34 = {var_2: var_33}
    var_35 = module_0.ParsedContent()
    var_36 = True
    var_37 = module_1.Config()
    var_38 = module_2.sorted_imports(var_35, var_37)
    assert var_38 == 'import os\nimport zlib\n\nfrom json import load\nfrom sys import path\n\nx = 1\n'
    var_39 = [var_0]
    var_40 = []
    var_41 = []
    var_42 = {var_5: var_40, var_6: var_41}
    var_43 = [var_12]
    var_44 = [var_14]
    var_45 = {var_10: var_43, var_11: var_44}
    var_46 = {var_3: var_42, var_4: var_45}
    var_47 = {var_2: var_46}
    var_48 = module_0.ParsedContent()
    var_49 = [var_5, var_10]
    var_50 = module_1.Config()
    var_51 = module_2.sorted_imports(var_48, var_50)
    assert var_51 == 'import os\n\nfrom json import load\n\nimport zlib\n\nfrom sys import path\n\nx = 1\n'
    var_52 = [var_0]
    var_53 = {}
    var_54 = '*'
    var_55 = [var_54]
    var_56 = [var_14]
    var_57 = {var_10: var_55, var_11: var_56}
    var_58 = {var_3: var_53, var_4: var_57}
    var_59 = {var_2: var_58}
    var_60 = module_0.ParsedContent()
    var_61 = True
    var_62 = module_1.Config()
    var_63 = module_2.sorted_imports(var_60, var_62)
    assert var_63 == 'from sys import *\nfrom json import load\n\nx = 1\n'
    var_64 = [var_0]
    var_65 = []
    var_66 = {var_6: var_65}
    var_67 = [var_12]
    var_68 = {var_10: var_67}
    var_69 = {var_3: var_66, var_4: var_68}
    var_70 = {var_2: var_69}
    var_71 = module_0.ParsedContent()
    var_72 = True
    var_73 = module_1.Config()
    var_74 = module_2.sorted_imports(var_71, var_73)
    assert var_74 == 'from sys import path\n\nimport os\n\nx = 1\n'
    var_75 = [var_0]
    var_76 = []
    var_77 = []
    var_78 = {var_6: var_76, var_10: var_77}
    var_79 = [var_14]
    var_80 = {var_11: var_79}
    var_81 = {var_3: var_78, var_4: var_80}
    var_82 = {var_2: var_81}
    var_83 = module_0.ParsedContent()
    var_84 = [var_6, var_11]
    var_85 = module_1.Config()
    var_86 = module_2.sorted_imports(var_83, var_85)
    assert var_86 == 'import sys\n\nx = 1\n'
    var_87 = [var_0]
    var_88 = []
    var_89 = {var_6: var_88}
    var_90 = [var_12]
    var_91 = {var_10: var_90}
    var_92 = {var_3: var_89, var_4: var_91}
    var_93 = {var_2: var_92}
    var_94 = module_0.ParsedContent()
    var_95 = 'thirdparty'
    var_96 = 'Third Party Imports'
    var_97 = {var_95: var_96}
    var_98 = module_1.Config()
    var_99 = module_2.sorted_imports(var_94, var_98)
    assert var_99 == '# Third Party Imports\nimport os\n\nfrom sys import path\n\nx = 1\n'
    var_100 = [var_0]
    var_101 = 'FUTURE'
    var_102 = '__future__'
    var_103 = []
    var_104 = {var_102: var_103}
    var_105 = {}
    var_106 = {var_3: var_104, var_4: var_105}
    var_107 = []
    var_108 = {var_6: var_107}
    var_109 = {}
    var_110 = {var_3: var_108, var_4: var_109}
    var_111 = {var_101: var_106, var_2: var_110}
    var_112 = module_0.ParsedContent()
    var_113 = 2
    var_114 = module_1.Config()
    var_115 = module_2.sorted_imports(var_112, var_114)
    assert var_115 == 'from __future__ import absolute_import\n\n\n\nimport os\n\nx = 1\n'
    var_116 = [var_0]
    var_117 = []
    var_118 = {var_6: var_117}
    var_119 = {}
    var_120 = {var_3: var_118, var_4: var_119}
    var_121 = {var_2: var_120}
    var_122 = module_0.ParsedContent()
    var_123 = module_1.Config()
    var_124 = module_2.sorted_imports(var_122, var_123)
    assert var_124 == 'import os\n\n\nx = 1\n'
    var_125 = [var_0]
    var_126 = []
    var_127 = {var_6: var_126}
    var_128 = {}
    var_129 = {var_3: var_127, var_4: var_128}
    var_130 = {var_2: var_129}
    var_131 = module_0.ParsedContent()
    var_132 = module_2.sorted_imports(var_131, var_123)
    assert var_132 == 'import os\r\n\r\nx = 1\r\n'



# Parsed testcases at query #35
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'x = 1'
    var_1 = [var_0]
    var_2 = 'THIRDPARTY'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = 'sys'
    var_7 = []
    var_8 = []
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'collections'
    var_11 = 'defaultdict'
    var_12 = [var_11]
    var_13 = {var_10: var_12}
    var_14 = {var_3: var_9, var_4: var_13}
    var_15 = {var_2: var_14}
    var_16 = 0
    var_17 = 1
    var_18 = '\n'
    var_19 = {}
    var_20 = {}
    var_21 = module_0.ParsedContent()
    var_22 = module_1.Config()
    var_23 = module_2.sorted_imports(var_21, var_22)
    assert var_23 == 'import os\nimport sys\n\nfrom collections import defaultdict\n\nx = 1'
    var_24 = [var_0]
    var_25 = 'FUTURE'
    var_26 = 'FIRSTPARTY'
    var_27 = '__future__'
    var_28 = 'print_function'
    var_29 = [var_28]
    var_30 = {var_27: var_29}
    var_31 = {}
    var_32 = {var_3: var_30, var_4: var_31}
    var_33 = []
    var_34 = []
    var_35 = {var_5: var_33, var_6: var_34}
    var_36 = {}
    var_37 = {var_3: var_35, var_4: var_36}
    var_38 = 'my_module'
    var_39 = []
    var_40 = {var_38: var_39}
    var_41 = {}
    var_42 = {var_3: var_40, var_4: var_41}
    var_43 = {var_25: var_32, var_2: var_37, var_26: var_42}
    var_44 = {}
    var_45 = {}
    var_46 = module_0.ParsedContent()
    var_47 = True
    var_48 = module_1.Config()
    var_49 = module_2.sorted_imports(var_46, var_48)
    var_50 = True
    var_51 = module_1.Config()
    var_52 = module_2.sorted_imports(var_21, var_51)
    var_53 = 'from collections import defaultdict\n\nimport os\nimport sys'
    var_54 = [var_0]
    var_55 = {}
    var_56 = 'module1'
    var_57 = 'module2'
    var_58 = 'module3'
    var_59 = '*'
    var_60 = [var_59]
    var_61 = 'function1'
    var_62 = [var_61]
    var_63 = [var_59]
    var_64 = {var_56: var_60, var_57: var_62, var_58: var_63}
    var_65 = {var_3: var_55, var_4: var_64}
    var_66 = {var_2: var_65}
    var_67 = {}
    var_68 = {}
    var_69 = module_0.ParsedContent()
    var_70 = True
    var_71 = module_1.Config()
    var_72 = module_2.sorted_imports(var_69, var_71)
    var_73 = 'from module1 import *'
    var_74 = 'from module2 import function1'
    var_75 = 'Third Party Imports'
    var_76 = {var_2: var_75}
    var_77 = True
    var_78 = module_1.Config()
    var_79 = module_2.sorted_imports(var_21, var_78)
    var_80 = 2
    var_81 = module_1.Config()
    var_82 = module_2.sorted_imports(var_21, var_81)
    var_83 = '# comment'
    var_84 = [var_83, var_0]
    var_85 = []
    var_86 = {var_5: var_85}
    var_87 = {}
    var_88 = {var_3: var_86, var_4: var_87}
    var_89 = {var_2: var_88}
    var_90 = {}
    var_91 = {}
    var_92 = module_0.ParsedContent()
    var_93 = True
    var_94 = module_1.Config()
    var_95 = module_2.sorted_imports(var_92, var_94)
    var_96 = '# PLACE_HOLDER'
    var_97 = [var_0, var_96]
    var_98 = []
    var_99 = {var_5: var_98}
    var_100 = {}
    var_101 = {var_3: var_99, var_4: var_100}
    var_102 = {var_2: var_101}
    var_103 = 'PLACE_HOLDER'
    var_104 = 'import sys'
    var_105 = [var_104]
    var_106 = {var_103: var_105}
    var_107 = {var_96: var_103}
    var_108 = module_0.ParsedContent()
    var_109 = module_2.sorted_imports(var_108, var_22)



# Parsed testcases at query #36
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2
import re as module_3

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 'THIRDPARTY'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'zlib'
    var_6 = 'os'
    var_7 = []
    var_8 = []
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'sys'
    var_11 = 'json'
    var_12 = 'path'
    var_13 = [var_12]
    var_14 = 'load'
    var_15 = [var_14]
    var_16 = {var_10: var_13, var_11: var_15}
    var_17 = {var_3: var_9, var_4: var_16}
    var_18 = {var_2: var_17}
    var_19 = 0
    var_20 = 1
    var_21 = '\n'
    var_22 = {}
    var_23 = {}
    var_24 = module_0.ParsedContent()
    var_25 = module_1.Config()
    var_26 = module_2.sorted_imports(var_24, var_25)
    assert var_26 == '\nimport os\nimport zlib\n\nfrom json import load\nfrom sys import path\n'
    var_27 = "print('hello')"
    var_28 = [var_27]
    var_29 = {}
    var_30 = -1
    var_31 = {}
    var_32 = {}
    var_33 = module_0.ParsedContent()
    var_34 = module_2.sorted_imports(var_33, var_25)
    assert var_34 == "print('hello')"
    var_35 = 'LOCALFOLDER'
    var_36 = [var_35]
    var_37 = module_1.Config()
    var_38 = [var_0]
    var_39 = []
    var_40 = {var_5: var_39}
    var_41 = [var_12]
    var_42 = {var_10: var_41}
    var_43 = {var_3: var_40, var_4: var_42}
    var_44 = 'local'
    var_45 = []
    var_46 = {var_44: var_45}
    var_47 = 'local_func'
    var_48 = [var_47]
    var_49 = {var_0: var_48}
    var_50 = {var_3: var_46, var_4: var_49}
    var_51 = {var_2: var_43, var_35: var_50}
    var_52 = {}
    var_53 = {}
    var_54 = module_0.ParsedContent()
    var_55 = module_2.sorted_imports(var_54, var_37)
    var_56 = True
    var_57 = module_1.Config()
    var_58 = [var_0]
    var_59 = 'FUTURE'
    var_60 = '__future__'
    var_61 = 'annotations'
    var_62 = [var_61]
    var_63 = {var_60: var_62}
    var_64 = {}
    var_65 = {var_3: var_63, var_4: var_64}
    var_66 = []
    var_67 = {var_5: var_66}
    var_68 = [var_12]
    var_69 = {var_10: var_68}
    var_70 = {var_3: var_67, var_4: var_69}
    var_71 = {var_59: var_65, var_2: var_70}
    var_72 = {}
    var_73 = {}
    var_74 = module_0.ParsedContent()
    var_75 = module_2.sorted_imports(var_74, var_57)
    var_76 = 'from sys import *'
    var_77 = [var_76]
    var_78 = module_1.Config()
    var_79 = [var_0]
    var_80 = []
    var_81 = {var_5: var_80}
    var_82 = '*'
    var_83 = [var_82]
    var_84 = {var_10: var_83}
    var_85 = {var_3: var_81, var_4: var_84}
    var_86 = {var_2: var_85}
    var_87 = {}
    var_88 = {}
    var_89 = module_0.ParsedContent()
    var_90 = module_2.sorted_imports(var_89, var_78)
    var_91 = True
    var_92 = module_1.Config()
    var_93 = [var_0]
    var_94 = {}
    var_95 = [var_82]
    var_96 = [var_12]
    var_97 = {var_10: var_95, var_6: var_96}
    var_98 = {var_3: var_94, var_4: var_97}
    var_99 = {var_2: var_98}
    var_100 = {}
    var_101 = {}
    var_102 = module_0.ParsedContent()
    var_103 = module_2.sorted_imports(var_102, var_92)
    var_104 = 'from os import path'
    var_105 = True
    var_106 = module_1.Config()
    var_107 = [var_0]
    var_108 = []
    var_109 = {var_5: var_108}
    var_110 = [var_12]
    var_111 = {var_10: var_110}
    var_112 = {var_3: var_109, var_4: var_111}
    var_113 = {var_2: var_112}
    var_114 = {}
    var_115 = {}
    var_116 = module_0.ParsedContent()
    var_117 = module_2.sorted_imports(var_116, var_106)
    var_118 = 'from sys import path'
    var_119 = 'import zlib'
    var_120 = 'thirdparty'
    var_121 = 'Third Party Imports'
    var_122 = {var_120: var_121}
    var_123 = module_1.Config()
    var_124 = [var_0]
    var_125 = []
    var_126 = {var_5: var_125}
    var_127 = {}
    var_128 = {var_3: var_126, var_4: var_127}
    var_129 = {var_2: var_128}
    var_130 = {}
    var_131 = {}
    var_132 = module_0.ParsedContent()
    var_133 = module_2.sorted_imports(var_132, var_123)
    var_134 = 2
    var_135 = module_1.Config()
    var_136 = [var_0]
    var_137 = [var_61]
    var_138 = {var_60: var_137}
    var_139 = {}
    var_140 = {var_3: var_138, var_4: var_139}
    var_141 = []
    var_142 = {var_5: var_141}
    var_143 = {}
    var_144 = {var_3: var_142, var_4: var_143}
    var_145 = {var_59: var_140, var_2: var_144}
    var_146 = {}
    var_147 = {}
    var_148 = module_0.ParsedContent()
    var_149 = module_2.sorted_imports(var_148, var_135)
    var_150 = module_3.split(var_21)
    var_151 = 'from __future__ import annotations'
    var_152 = module_1.Config()
    var_153 = [var_0, var_27]
    var_154 = []
    var_155 = {var_5: var_154}
    var_156 = {}
    var_157 = {var_3: var_155, var_4: var_156}
    var_158 = {var_2: var_157}
    var_159 = {}
    var_160 = {}
    var_161 = module_0.ParsedContent()
    var_162 = module_2.sorted_imports(var_161, var_152)
    var_163 = module_3.split(var_21)
    var_164 = module_1.Config()
    var_165 = 'def func():'
    var_166 = '    pass'
    var_167 = [var_0, var_165, var_166]
    var_168 = []
    var_169 = {var_5: var_168}
    var_170 = {}
    var_171 = {var_3: var_169, var_4: var_170}
    var_172 = {var_2: var_171}
    var_173 = 3
    var_174 = [var_119]
    var_175 = {var_2: var_174}
    var_176 = {var_165}
    var_177 = module_0.ParsedContent()
    var_178 = module_2.sorted_imports(var_177, var_164)
    var_179 = module_3.split(var_21)



# Parsed testcases at query #37
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = 'def main():'
    var_2 = '    pass'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'FUTURE'
    var_5 = 'STDLIB'
    var_6 = 'THIRDPARTY'
    var_7 = 'straight'
    var_8 = 'from'
    var_9 = '__future__'
    var_10 = 'import annotations'
    var_11 = [var_10]
    var_12 = {var_9: var_11}
    var_13 = {}
    var_14 = {var_7: var_12, var_8: var_13}
    var_15 = 'os'
    var_16 = 'import os'
    var_17 = [var_16]
    var_18 = {var_15: var_17}
    var_19 = {}
    var_20 = {var_7: var_18, var_8: var_19}
    var_21 = 'django'
    var_22 = 'import django'
    var_23 = [var_22]
    var_24 = {var_21: var_23}
    var_25 = {}
    var_26 = {var_7: var_24, var_8: var_25}
    var_27 = {var_4: var_14, var_5: var_20, var_6: var_26}
    var_28 = 0
    var_29 = 3
    var_30 = '\n'
    var_31 = {}
    var_32 = {}
    var_33 = module_0.ParsedContent()
    var_34 = []
    var_35 = []
    var_36 = False
    var_37 = False
    var_38 = False
    var_39 = False
    var_40 = False
    var_41 = False
    var_42 = 1
    var_43 = {}
    var_44 = {}
    var_45 = False
    var_46 = False
    var_47 = 2
    var_48 = 'black'
    var_49 = False
    var_50 = None
    var_51 = module_1.Config()
    var_52 = 'py'
    var_53 = 'import'
    var_54 = module_2.sorted_imports(var_33, var_51, var_52, var_53)
    var_55 = '\n\nfrom __future__ import annotations\n\nimport os\n\nimport django\n\n\ndef main():\n    pass'



# Parsed testcases at query #38
#--------------------------


import isort.parse as module_0
import isort.output as module_1
import isort.settings as module_2

def test_case_0():
    var_0 = ''
    var_1 = 'def foo():'
    var_2 = '    pass'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = 'sys'
    var_9 = set()
    var_10 = set()
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = 'collections'
    var_13 = 'itertools'
    var_14 = 'defaultdict'
    var_15 = {var_14}
    var_16 = 'chain'
    var_17 = {var_16}
    var_18 = {var_12: var_15, var_13: var_17}
    var_19 = {var_5: var_11, var_6: var_18}
    var_20 = {var_4: var_19}
    var_21 = 0
    var_22 = 3
    var_23 = '\n'
    var_24 = module_0.ParsedContent()
    var_25 = module_1.sorted_imports(var_24)
    assert var_25 == 'import os\nimport sys\n\nfrom collections import defaultdict\nfrom itertools import chain\n\n\ndef foo():\n    pass\n'
    var_26 = 'LOCALFOLDER'
    var_27 = [var_26]
    var_28 = module_2.Config()
    var_29 = [var_0, var_1, var_2]
    var_30 = set()
    var_31 = set()
    var_32 = {var_7: var_30, var_8: var_31}
    var_33 = {var_14}
    var_34 = {var_12: var_33}
    var_35 = {var_5: var_32, var_6: var_34}
    var_36 = 'my_module'
    var_37 = set()
    var_38 = {var_36: var_37}
    var_39 = 'my_package'
    var_40 = 'my_function'
    var_41 = {var_40}
    var_42 = {var_39: var_41}
    var_43 = {var_5: var_38, var_6: var_42}
    var_44 = {var_4: var_35, var_26: var_43}
    var_45 = module_0.ParsedContent()
    var_46 = module_1.sorted_imports(var_45, var_28)
    assert var_46 == 'import os\nimport sys\n\nfrom collections import defaultdict\n\nimport my_module\n\nfrom my_package import my_function\n\n\ndef foo():\n    pass\n'
    var_47 = True
    var_48 = module_2.Config()
    var_49 = [var_0, var_1, var_2]
    var_50 = 'FUTURE'
    var_51 = '__future__'
    var_52 = 'print_function'
    var_53 = {var_52}
    var_54 = {var_51: var_53}
    var_55 = {}
    var_56 = {var_5: var_54, var_6: var_55}
    var_57 = set()
    var_58 = set()
    var_59 = {var_7: var_57, var_8: var_58}
    var_60 = {var_14}
    var_61 = {var_12: var_60}
    var_62 = {var_5: var_59, var_6: var_61}
    var_63 = {var_50: var_56, var_4: var_62}
    var_64 = module_0.ParsedContent()
    var_65 = module_1.sorted_imports(var_64, var_48)
    assert var_65 == 'from __future__ import print_function\n\nimport os\nimport sys\n\nfrom collections import defaultdict\n\n\ndef foo():\n    pass\n'
    var_66 = module_2.Config()
    var_67 = [var_0, var_1, var_2]
    var_68 = {}
    var_69 = 'module1'
    var_70 = 'module2'
    var_71 = '*'
    var_72 = {var_71}
    var_73 = 'function1'
    var_74 = 'function2'
    var_75 = {var_73, var_74}
    var_76 = {var_69: var_72, var_70: var_75}
    var_77 = {var_5: var_68, var_6: var_76}
    var_78 = {var_4: var_77}
    var_79 = module_0.ParsedContent()
    var_80 = module_1.sorted_imports(var_79, var_66)
    assert var_80 == 'from module1 import *\nfrom module2 import function1, function2\n\n\ndef foo():\n    pass\n'
    var_81 = module_2.Config()
    var_82 = [var_0, var_1, var_2]
    var_83 = set()
    var_84 = {var_7: var_83}
    var_85 = {var_14}
    var_86 = {var_12: var_85}
    var_87 = {var_5: var_84, var_6: var_86}
    var_88 = {var_4: var_87}
    var_89 = module_0.ParsedContent()
    var_90 = module_1.sorted_imports(var_89, var_81)
    assert var_90 == 'from collections import defaultdict\n\nimport os\n\n\ndef foo():\n    pass\n'
    var_91 = 2
    var_92 = module_2.Config()
    var_93 = [var_0, var_1, var_2]
    var_94 = set()
    var_95 = {var_7: var_94}
    var_96 = {var_14}
    var_97 = {var_12: var_96}
    var_98 = {var_5: var_95, var_6: var_97}
    var_99 = {var_4: var_98}
    var_100 = module_0.ParsedContent()
    var_101 = module_1.sorted_imports(var_100, var_92)
    assert var_101 == 'import os\n\n\nfrom collections import defaultdict\n\n\ndef foo():\n    pass\n'
    var_102 = 'thirdparty'
    var_103 = 'Third Party Imports'
    var_104 = {var_102: var_103}
    var_105 = module_2.Config()
    var_106 = [var_0, var_1, var_2]
    var_107 = set()
    var_108 = {var_7: var_107}
    var_109 = {}
    var_110 = {var_5: var_108, var_6: var_109}
    var_111 = {var_4: var_110}
    var_112 = module_0.ParsedContent()
    var_113 = module_1.sorted_imports(var_112, var_105)
    assert var_113 == '# Third Party Imports\nimport os\n\n\ndef foo():\n    pass\n'
    var_114 = [var_7]
    var_115 = module_2.Config()
    var_116 = [var_0, var_1, var_2]
    var_117 = set()
    var_118 = set()
    var_119 = {var_7: var_117, var_8: var_118}
    var_120 = {}
    var_121 = {var_5: var_119, var_6: var_120}
    var_122 = {var_4: var_121}
    var_123 = module_0.ParsedContent()
    var_124 = module_1.sorted_imports(var_123, var_115)
    assert var_124 == 'import sys\n\n\ndef foo():\n    pass\n'
    var_125 = module_2.Config()
    var_126 = [var_0, var_1, var_2]
    var_127 = set()
    var_128 = {var_7: var_127}
    var_129 = {}
    var_130 = {var_5: var_128, var_6: var_129}
    var_131 = {var_4: var_130}
    var_132 = module_0.ParsedContent()
    var_133 = module_1.sorted_imports(var_132, var_125)
    assert var_133 == 'import os\n\n\n\ndef foo():\n    pass\n'
    var_134 = [var_0, var_1, var_2]
    var_135 = set()
    var_136 = {var_7: var_135}
    var_137 = {}
    var_138 = {var_5: var_136, var_6: var_137}
    var_139 = {var_4: var_138}
    var_140 = module_0.ParsedContent()
    var_141 = module_1.sorted_imports(var_140, var_125)
    assert var_141 == 'IMPORT OS\n\n\ndef foo():\n    pass\n'



# Parsed testcases at query #39
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = 'x = 1'
    var_2 = [var_0, var_1]
    var_3 = 'THIRDPARTY'
    var_4 = 'straight'
    var_5 = 'from'
    var_6 = 'zlib'
    var_7 = 'os'
    var_8 = []
    var_9 = []
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'sys'
    var_12 = 'json'
    var_13 = 'path'
    var_14 = [var_13]
    var_15 = 'load'
    var_16 = [var_15]
    var_17 = {var_11: var_14, var_12: var_16}
    var_18 = {var_4: var_10, var_5: var_17}
    var_19 = {var_3: var_18}
    var_20 = 0
    var_21 = 2
    var_22 = '\n'
    var_23 = module_0.ParsedContent()
    var_24 = module_1.Config()
    var_25 = module_2.sorted_imports(var_23, var_24)
    assert var_25 == '\n\nimport os\nimport zlib\n\nfrom json import load\nfrom sys import path\n\nx = 1\n'
    var_26 = [var_0, var_1]
    var_27 = 'FUTURE'
    var_28 = '__future__'
    var_29 = 'annotations'
    var_30 = [var_29]
    var_31 = {var_28: var_30}
    var_32 = {}
    var_33 = {var_4: var_31, var_5: var_32}
    var_34 = []
    var_35 = []
    var_36 = {var_6: var_34, var_7: var_35}
    var_37 = [var_13]
    var_38 = [var_15]
    var_39 = {var_11: var_37, var_12: var_38}
    var_40 = {var_4: var_36, var_5: var_39}
    var_41 = {var_27: var_33, var_3: var_40}
    var_42 = module_0.ParsedContent()
    var_43 = True
    var_44 = module_1.Config()
    var_45 = module_2.sorted_imports(var_42, var_44)
    assert var_45 == '\n\nfrom __future__ import annotations\n\nimport os\nimport zlib\n\nfrom json import load\nfrom sys import path\n\nx = 1\n'
    var_46 = [var_0, var_1]
    var_47 = []
    var_48 = []
    var_49 = {var_6: var_47, var_7: var_48}
    var_50 = [var_13]
    var_51 = [var_15]
    var_52 = {var_11: var_50, var_12: var_51}
    var_53 = {var_4: var_49, var_5: var_52}
    var_54 = {var_3: var_53}
    var_55 = module_0.ParsedContent()
    var_56 = 'LOCALFOLDER'
    var_57 = [var_56]
    var_58 = module_1.Config()
    var_59 = module_2.sorted_imports(var_55, var_58)
    assert var_59 == '\n\nimport os\nimport zlib\n\nfrom json import load\nfrom sys import path\n\nx = 1\n'
    var_60 = [var_0, var_1]
    var_61 = []
    var_62 = []
    var_63 = {var_6: var_61, var_7: var_62}
    var_64 = [var_13]
    var_65 = [var_15]
    var_66 = {var_11: var_64, var_12: var_65}
    var_67 = {var_4: var_63, var_5: var_66}
    var_68 = {var_3: var_67}
    var_69 = module_0.ParsedContent()
    var_70 = 'import zlib'
    var_71 = 'from sys import path'
    var_72 = [var_70, var_71]
    var_73 = module_1.Config()
    var_74 = module_2.sorted_imports(var_69, var_73)
    assert var_74 == '\n\nimport os\n\nfrom json import load\n\nx = 1\n'
    var_75 = [var_0, var_1]
    var_76 = {}
    var_77 = '*'
    var_78 = [var_77]
    var_79 = [var_15]
    var_80 = {var_11: var_78, var_12: var_79}
    var_81 = {var_4: var_76, var_5: var_80}
    var_82 = {var_3: var_81}
    var_83 = module_0.ParsedContent()
    var_84 = module_1.Config()
    var_85 = module_2.sorted_imports(var_83, var_84)
    assert var_85 == '\n\nfrom sys import *\nfrom json import load\n\nx = 1\n'
    var_86 = [var_0, var_1]
    var_87 = []
    var_88 = []
    var_89 = {var_6: var_87, var_7: var_88}
    var_90 = [var_13]
    var_91 = [var_15]
    var_92 = {var_11: var_90, var_12: var_91}
    var_93 = {var_4: var_89, var_5: var_92}
    var_94 = {var_3: var_93}
    var_95 = module_0.ParsedContent()
    var_96 = module_1.Config()
    var_97 = module_2.sorted_imports(var_95, var_96)
    assert var_97 == '\n\nfrom json import load\nfrom sys import path\n\nimport os\nimport zlib\n\nx = 1\n'
    var_98 = [var_0, var_1]
    var_99 = []
    var_100 = []
    var_101 = {var_6: var_99, var_7: var_100}
    var_102 = [var_13]
    var_103 = [var_15]
    var_104 = {var_11: var_102, var_12: var_103}
    var_105 = {var_4: var_101, var_5: var_104}
    var_106 = {var_3: var_105}
    var_107 = module_0.ParsedContent()
    var_108 = 'thirdparty'
    var_109 = 'Third Party Imports'
    var_110 = {var_108: var_109}
    var_111 = module_1.Config()
    var_112 = module_2.sorted_imports(var_107, var_111)
    assert var_112 == '\n\n# Third Party Imports\nimport os\nimport zlib\n\nfrom json import load\nfrom sys import path\n\nx = 1\n'
    var_113 = [var_0, var_1]
    var_114 = [var_29]
    var_115 = {var_28: var_114}
    var_116 = {}
    var_117 = {var_4: var_115, var_5: var_116}
    var_118 = []
    var_119 = []
    var_120 = {var_6: var_118, var_7: var_119}
    var_121 = [var_13]
    var_122 = [var_15]
    var_123 = {var_11: var_121, var_12: var_122}
    var_124 = {var_4: var_120, var_5: var_123}
    var_125 = {var_27: var_117, var_3: var_124}
    var_126 = module_0.ParsedContent()
    var_127 = module_1.Config()
    var_128 = module_2.sorted_imports(var_126, var_127)
    assert var_128 == '\n\nfrom __future__ import annotations\n\n\n\nimport os\nimport zlib\n\nfrom json import load\nfrom sys import path\n\nx = 1\n'
    var_129 = [var_0, var_1]
    var_130 = []
    var_131 = []
    var_132 = {var_6: var_130, var_7: var_131}
    var_133 = [var_13]
    var_134 = [var_15]
    var_135 = {var_11: var_133, var_12: var_134}
    var_136 = {var_4: var_132, var_5: var_135}
    var_137 = {var_3: var_136}
    var_138 = module_0.ParsedContent()
    var_139 = module_1.Config()
    var_140 = module_2.sorted_imports(var_138, var_139)
    assert var_140 == '\n\n\n\nimport os\nimport zlib\n\nfrom json import load\nfrom sys import path\n\nx = 1\n'
    var_141 = [var_0, var_1]
    var_142 = []
    var_143 = []
    var_144 = {var_6: var_142, var_7: var_143}
    var_145 = [var_13]
    var_146 = [var_15]
    var_147 = {var_11: var_145, var_12: var_146}
    var_148 = {var_4: var_144, var_5: var_147}
    var_149 = {var_3: var_148}
    var_150 = module_0.ParsedContent()
    var_151 = module_1.Config()
    var_152 = module_2.sorted_imports(var_150, var_151)
    assert var_152 == '\n\nimport os\nimport zlib\n\nfrom json import load\nfrom sys import path\n\n\n\nx = 1\n'
    var_153 = [var_1]
    var_154 = {}
    var_155 = -1
    var_156 = module_0.ParsedContent()
    var_157 = module_1.Config()
    var_158 = module_2.sorted_imports(var_156, var_157)
    assert var_158 == 'x = 1\n'



# Parsed testcases at query #40
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 'THIRDPARTY'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = 'sys'
    var_7 = []
    var_8 = []
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'collections'
    var_11 = 'defaultdict'
    var_12 = [var_11]
    var_13 = {var_10: var_12}
    var_14 = {var_3: var_9, var_4: var_13}
    var_15 = {var_2: var_14}
    var_16 = 0
    var_17 = 1
    var_18 = '\n'
    var_19 = {}
    var_20 = {}
    var_21 = module_0.ParsedContent()
    var_22 = []
    var_23 = []
    var_24 = False
    var_25 = module_1.Config()
    var_26 = module_2.sorted_imports(var_21, var_25)
    assert var_26 == 'from collections import defaultdict\nimport os\nimport sys\n'
    var_27 = [var_0]
    var_28 = 'FUTURE'
    var_29 = 'FIRSTPARTY'
    var_30 = '__future__'
    var_31 = []
    var_32 = {var_30: var_31}
    var_33 = {}
    var_34 = {var_3: var_32, var_4: var_33}
    var_35 = []
    var_36 = []
    var_37 = {var_5: var_35, var_6: var_36}
    var_38 = {}
    var_39 = {var_3: var_37, var_4: var_38}
    var_40 = 'my_module'
    var_41 = []
    var_42 = {var_40: var_41}
    var_43 = {}
    var_44 = {var_3: var_42, var_4: var_43}
    var_45 = {var_28: var_34, var_2: var_39, var_29: var_44}
    var_46 = {}
    var_47 = {}
    var_48 = module_0.ParsedContent()
    var_49 = []
    var_50 = []
    var_51 = True
    var_52 = module_1.Config()
    var_53 = module_2.sorted_imports(var_48, var_52)
    assert var_53 == 'from __future__ import absolute_import\nimport my_module\nimport os\nimport sys\n'
    var_54 = [var_0]
    var_55 = []
    var_56 = []
    var_57 = {var_5: var_55, var_6: var_56}
    var_58 = [var_11]
    var_59 = {var_10: var_58}
    var_60 = {var_3: var_57, var_4: var_59}
    var_61 = {var_2: var_60}
    var_62 = {}
    var_63 = {}
    var_64 = module_0.ParsedContent()
    var_65 = [var_5]
    var_66 = []
    var_67 = False
    var_68 = module_1.Config()
    var_69 = module_2.sorted_imports(var_64, var_68)
    assert var_69 == 'from collections import defaultdict\nimport sys\n'
    var_70 = [var_0]
    var_71 = {}
    var_72 = 'module1'
    var_73 = 'module2'
    var_74 = '*'
    var_75 = [var_74]
    var_76 = 'func'
    var_77 = [var_76]
    var_78 = {var_72: var_75, var_73: var_77}
    var_79 = {var_3: var_71, var_4: var_78}
    var_80 = {var_2: var_79}
    var_81 = {}
    var_82 = {}
    var_83 = module_0.ParsedContent()
    var_84 = []
    var_85 = []
    var_86 = False
    var_87 = True
    var_88 = module_1.Config()
    var_89 = module_2.sorted_imports(var_83, var_88)
    assert var_89 == 'from module1 import *\nfrom module2 import func\n'
    var_90 = [var_0]
    var_91 = []
    var_92 = {var_5: var_91}
    var_93 = {}
    var_94 = {var_3: var_92, var_4: var_93}
    var_95 = {var_2: var_94}
    var_96 = {}
    var_97 = {}
    var_98 = module_0.ParsedContent()
    var_99 = []
    var_100 = []
    var_101 = False
    var_102 = 'thirdparty'
    var_103 = 'Third Party Imports'
    var_104 = {var_102: var_103}
    var_105 = module_1.Config()
    var_106 = module_2.sorted_imports(var_98, var_105)
    assert var_106 == '# Third Party Imports\nimport os\n'
    var_107 = [var_0]
    var_108 = []
    var_109 = {var_30: var_108}
    var_110 = {}
    var_111 = {var_3: var_109, var_4: var_110}
    var_112 = []
    var_113 = {var_5: var_112}
    var_114 = {}
    var_115 = {var_3: var_113, var_4: var_114}
    var_116 = {var_28: var_111, var_2: var_115}
    var_117 = {}
    var_118 = {}
    var_119 = module_0.ParsedContent()
    var_120 = []
    var_121 = []
    var_122 = False
    var_123 = 2
    var_124 = module_1.Config()
    var_125 = module_2.sorted_imports(var_119, var_124)
    assert var_125 == 'from __future__ import absolute_import\n\n\nimport os\n'
    var_126 = 'def main():'
    var_127 = '    pass'
    var_128 = [var_0, var_126, var_127]
    var_129 = []
    var_130 = {var_5: var_129}
    var_131 = {}
    var_132 = {var_3: var_130, var_4: var_131}
    var_133 = {var_2: var_132}
    var_134 = 3
    var_135 = 'import os'
    var_136 = [var_135]
    var_137 = {var_2: var_136}
    var_138 = {var_126}
    var_139 = module_0.ParsedContent()
    var_140 = []
    var_141 = []
    var_142 = False
    var_143 = module_1.Config()
    var_144 = module_2.sorted_imports(var_139, var_143)
    assert var_144 == 'def main():\nimport os\n    pass\n'



# Parsed testcases at query #41
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 'THIRDPARTY'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = 'sys'
    var_7 = []
    var_8 = []
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'collections'
    var_11 = 'itertools'
    var_12 = 'defaultdict'
    var_13 = [var_12]
    var_14 = 'chain'
    var_15 = [var_14]
    var_16 = {var_10: var_13, var_11: var_15}
    var_17 = {var_3: var_9, var_4: var_16}
    var_18 = {var_2: var_17}
    var_19 = 0
    var_20 = 1
    var_21 = '\n'
    var_22 = module_0.ParsedContent()
    var_23 = module_1.Config()
    var_24 = module_2.sorted_imports(var_22, var_23)
    assert var_24 == 'from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n'
    var_25 = [var_0]
    var_26 = 'FUTURE'
    var_27 = {}
    var_28 = {}
    var_29 = {var_3: var_27, var_4: var_28}
    var_30 = []
    var_31 = []
    var_32 = {var_5: var_30, var_6: var_31}
    var_33 = [var_12]
    var_34 = [var_14]
    var_35 = {var_10: var_33, var_11: var_34}
    var_36 = {var_3: var_32, var_4: var_35}
    var_37 = {var_26: var_29, var_2: var_36}
    var_38 = module_0.ParsedContent()
    var_39 = True
    var_40 = module_1.Config()
    var_41 = module_2.sorted_imports(var_38, var_40)
    assert var_41 == 'from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n'
    var_42 = [var_0]
    var_43 = []
    var_44 = []
    var_45 = {var_5: var_43, var_6: var_44}
    var_46 = [var_12]
    var_47 = [var_14]
    var_48 = {var_10: var_46, var_11: var_47}
    var_49 = {var_3: var_45, var_4: var_48}
    var_50 = {var_2: var_49}
    var_51 = module_0.ParsedContent()
    var_52 = 'LOCALFOLDER'
    var_53 = [var_52]
    var_54 = module_1.Config()
    var_55 = module_2.sorted_imports(var_51, var_54)
    assert var_55 == 'from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n'
    var_56 = [var_0]
    var_57 = []
    var_58 = []
    var_59 = {var_5: var_57, var_6: var_58}
    var_60 = [var_12]
    var_61 = [var_14]
    var_62 = {var_10: var_60, var_11: var_61}
    var_63 = {var_3: var_59, var_4: var_62}
    var_64 = {var_2: var_63}
    var_65 = module_0.ParsedContent()
    var_66 = [var_5, var_6]
    var_67 = module_1.Config()
    var_68 = module_2.sorted_imports(var_65, var_67)
    assert var_68 == 'from collections import defaultdict\nfrom itertools import chain\n'
    var_69 = [var_0]
    var_70 = []
    var_71 = []
    var_72 = {var_5: var_70, var_6: var_71}
    var_73 = [var_12]
    var_74 = [var_14]
    var_75 = {var_10: var_73, var_11: var_74}
    var_76 = {var_3: var_72, var_4: var_75}
    var_77 = {var_2: var_76}
    var_78 = module_0.ParsedContent()
    var_79 = module_1.Config()
    var_80 = module_2.sorted_imports(var_78, var_79)
    assert var_80 == 'from collections import defaultdict\nfrom itertools import chain\n\nimport os\nimport sys\n'
    var_81 = [var_0]
    var_82 = []
    var_83 = []
    var_84 = {var_5: var_82, var_6: var_83}
    var_85 = [var_12]
    var_86 = [var_14]
    var_87 = {var_10: var_85, var_11: var_86}
    var_88 = {var_3: var_84, var_4: var_87}
    var_89 = {var_2: var_88}
    var_90 = module_0.ParsedContent()
    var_91 = True
    var_92 = module_1.Config()
    var_93 = module_2.sorted_imports(var_90, var_92)
    assert var_93 == 'from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n'
    var_94 = [var_0]
    var_95 = {}
    var_96 = '*'
    var_97 = [var_96]
    var_98 = [var_14]
    var_99 = {var_10: var_97, var_11: var_98}
    var_100 = {var_3: var_95, var_4: var_99}
    var_101 = {var_2: var_100}
    var_102 = module_0.ParsedContent()
    var_103 = True
    var_104 = module_1.Config()
    var_105 = module_2.sorted_imports(var_102, var_104)
    assert var_105 == 'from collections import *\nfrom itertools import chain\n'
    var_106 = [var_0]
    var_107 = []
    var_108 = []
    var_109 = {var_5: var_107, var_6: var_108}
    var_110 = [var_12]
    var_111 = [var_14]
    var_112 = {var_10: var_110, var_11: var_111}
    var_113 = {var_3: var_109, var_4: var_112}
    var_114 = {var_2: var_113}
    var_115 = module_0.ParsedContent()
    var_116 = 'thirdparty'
    var_117 = 'Third Party Imports'
    var_118 = {var_116: var_117}
    var_119 = module_1.Config()
    var_120 = module_2.sorted_imports(var_115, var_119)
    assert var_120 == '# Third Party Imports\nfrom collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n'
    var_121 = [var_0]
    var_122 = []
    var_123 = []
    var_124 = {var_5: var_122, var_6: var_123}
    var_125 = [var_12]
    var_126 = [var_14]
    var_127 = {var_10: var_125, var_11: var_126}
    var_128 = {var_3: var_124, var_4: var_127}
    var_129 = {var_2: var_128}
    var_130 = module_0.ParsedContent()
    var_131 = 'End of Third Party Imports'
    var_132 = {var_116: var_131}
    var_133 = module_1.Config()
    var_134 = module_2.sorted_imports(var_130, var_133)
    assert var_134 == 'from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n\n# End of Third Party Imports\n'
    var_135 = [var_0]
    var_136 = {}
    var_137 = {}
    var_138 = {var_3: var_136, var_4: var_137}
    var_139 = []
    var_140 = []
    var_141 = {var_5: var_139, var_6: var_140}
    var_142 = [var_12]
    var_143 = [var_14]
    var_144 = {var_10: var_142, var_11: var_143}
    var_145 = {var_3: var_141, var_4: var_144}
    var_146 = {var_26: var_138, var_2: var_145}
    var_147 = module_0.ParsedContent()
    var_148 = module_1.Config()
    var_149 = module_2.sorted_imports(var_147, var_148)
    assert var_149 == 'from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n'
    var_150 = [var_0]
    var_151 = []
    var_152 = []
    var_153 = {var_5: var_151, var_6: var_152}
    var_154 = [var_12]
    var_155 = [var_14]
    var_156 = {var_10: var_154, var_11: var_155}
    var_157 = {var_3: var_153, var_4: var_156}
    var_158 = {var_2: var_157}
    var_159 = module_0.ParsedContent()
    var_160 = True
    var_161 = module_1.Config()
    var_162 = module_2.sorted_imports(var_159, var_161)
    assert var_162 == 'from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n'
    var_163 = [var_0]
    var_164 = []
    var_165 = []
    var_166 = {var_5: var_164, var_6: var_165}
    var_167 = [var_12]
    var_168 = [var_14]
    var_169 = {var_10: var_167, var_11: var_168}
    var_170 = {var_3: var_166, var_4: var_169}
    var_171 = {var_2: var_170}
    var_172 = module_0.ParsedContent()
    var_173 = lambda x, y, z: x
    var_174 = module_1.Config()
    var_175 = module_2.sorted_imports(var_172, var_174)
    assert var_175 == 'from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n'
    var_176 = [var_0]
    var_177 = []
    var_178 = []
    var_179 = {var_5: var_177, var_6: var_178}
    var_180 = [var_12]
    var_181 = [var_14]
    var_182 = {var_10: var_180, var_11: var_181}
    var_183 = {var_3: var_179, var_4: var_182}
    var_184 = {var_2: var_183}
    var_185 = module_0.ParsedContent()
    var_186 = module_1.Config()
    var_187 = module_2.sorted_imports(var_185, var_186)
    assert var_187 == '\nfrom collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n'
    var_188 = [var_0]
    var_189 = []
    var_190 = []
    var_191 = {var_5: var_189, var_6: var_190}
    var_192 = [var_12]
    var_193 = [var_14]
    var_194 = {var_10: var_192, var_11: var_193}
    var_195 = {var_3: var_191, var_4: var_194}
    var_196 = {var_2: var_195}
    var_197 = module_0.ParsedContent()
    var_198 = module_1.Config()
    var_199 = module_2.sorted_imports(var_197, var_198)
    assert var_199 == 'from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n\n'



# Parsed testcases at query #42
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = 'def main():'
    var_2 = '    pass'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = 'sys'
    var_9 = []
    var_10 = []
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = 'collections'
    var_13 = 'itertools'
    var_14 = 'defaultdict'
    var_15 = [var_14]
    var_16 = 'chain'
    var_17 = [var_16]
    var_18 = {var_12: var_15, var_13: var_17}
    var_19 = {var_5: var_11, var_6: var_18}
    var_20 = {var_4: var_19}
    var_21 = 0
    var_22 = '\n'
    var_23 = 3
    var_24 = module_0.ParsedContent()
    var_25 = 'import os\nimport sys\nfrom collections import defaultdict\nfrom itertools import chain\n\n\ndef main():'
    var_26 = [var_1, var_2]
    var_27 = {}
    var_28 = -1
    var_29 = 2
    var_30 = module_0.ParsedContent()
    var_31 = [var_0, var_1, var_2]
    var_32 = []
    var_33 = []
    var_34 = {var_7: var_32, var_8: var_33}
    var_35 = [var_14]
    var_36 = [var_16]
    var_37 = {var_12: var_35, var_13: var_36}
    var_38 = {var_5: var_34, var_6: var_37}
    var_39 = {var_4: var_38}
    var_40 = module_0.ParsedContent()
    var_41 = True
    var_42 = module_1.Config()
    var_43 = module_2.sorted_imports(var_40, var_42)
    var_44 = 'import sys\nimport os\nfrom itertools import chain\nfrom collections import defaultdict\n\n\ndef main():'
    var_45 = [var_0, var_1, var_2]
    var_46 = {}
    var_47 = 'module1'
    var_48 = 'module2'
    var_49 = '*'
    var_50 = [var_49]
    var_51 = 'func1'
    var_52 = [var_51]
    var_53 = {var_47: var_50, var_48: var_52}
    var_54 = {var_5: var_46, var_6: var_53}
    var_55 = {var_4: var_54}
    var_56 = module_0.ParsedContent()
    var_57 = module_1.Config()
    var_58 = module_2.sorted_imports(var_56, var_57)
    var_59 = 'from module1 import *\nfrom module2 import func1\n\n\ndef main():'
    var_60 = [var_0, var_1, var_2]
    var_61 = []
    var_62 = {var_7: var_61}
    var_63 = {}
    var_64 = {var_5: var_62, var_6: var_63}
    var_65 = {var_4: var_64}
    var_66 = module_0.ParsedContent()
    var_67 = 'thirdparty'
    var_68 = 'Third Party Imports'
    var_69 = {var_67: var_68}
    var_70 = module_1.Config()
    var_71 = module_2.sorted_imports(var_66, var_70)
    var_72 = '# Third Party Imports\nimport os\n\n\ndef main():'



# Parsed testcases at query #43
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 'THIRDPARTY'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = 'sys'
    var_7 = []
    var_8 = []
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'collections'
    var_11 = 'itertools'
    var_12 = 'defaultdict'
    var_13 = [var_12]
    var_14 = 'chain'
    var_15 = [var_14]
    var_16 = {var_10: var_13, var_11: var_15}
    var_17 = {var_3: var_9, var_4: var_16}
    var_18 = {var_2: var_17}
    var_19 = 0
    var_20 = 1
    var_21 = '\n'
    var_22 = {}
    var_23 = {}
    var_24 = module_0.ParsedContent()
    var_25 = module_1.Config()
    var_26 = module_2.sorted_imports(var_24, var_25)
    assert var_26 == 'from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n'
    var_27 = [var_0]
    var_28 = 'FIRSTPARTY'
    var_29 = []
    var_30 = []
    var_31 = {var_5: var_29, var_6: var_30}
    var_32 = [var_12]
    var_33 = [var_14]
    var_34 = {var_10: var_32, var_11: var_33}
    var_35 = {var_3: var_31, var_4: var_34}
    var_36 = 'my_module'
    var_37 = []
    var_38 = {var_36: var_37}
    var_39 = 'my_package'
    var_40 = 'MyClass'
    var_41 = [var_40]
    var_42 = {var_39: var_41}
    var_43 = {var_3: var_38, var_4: var_42}
    var_44 = {var_2: var_35, var_28: var_43}
    var_45 = {}
    var_46 = {}
    var_47 = module_0.ParsedContent()
    var_48 = True
    var_49 = module_1.Config()
    var_50 = module_2.sorted_imports(var_47, var_49)
    assert var_50 == 'from collections import defaultdict\nfrom itertools import chain\nfrom my_package import MyClass\nimport my_module\nimport os\nimport sys\n'
    var_51 = [var_0]
    var_52 = []
    var_53 = []
    var_54 = {var_5: var_52, var_6: var_53}
    var_55 = [var_12]
    var_56 = [var_14]
    var_57 = {var_10: var_55, var_11: var_56}
    var_58 = {var_3: var_54, var_4: var_57}
    var_59 = {var_2: var_58}
    var_60 = {}
    var_61 = {}
    var_62 = module_0.ParsedContent()
    var_63 = True
    var_64 = module_1.Config()
    var_65 = module_2.sorted_imports(var_62, var_64)
    assert var_65 == 'from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n'
    var_66 = [var_0]
    var_67 = {}
    var_68 = 'module1'
    var_69 = 'module2'
    var_70 = '*'
    var_71 = [var_70]
    var_72 = 'function'
    var_73 = [var_72]
    var_74 = {var_68: var_71, var_69: var_73}
    var_75 = {var_3: var_67, var_4: var_74}
    var_76 = {var_2: var_75}
    var_77 = {}
    var_78 = {}
    var_79 = module_0.ParsedContent()
    var_80 = True
    var_81 = module_1.Config()
    var_82 = module_2.sorted_imports(var_79, var_81)
    assert var_82 == 'from module1 import *\nfrom module2 import function\n'
    var_83 = [var_0]
    var_84 = []
    var_85 = {var_5: var_84}
    var_86 = {}
    var_87 = {var_3: var_85, var_4: var_86}
    var_88 = {var_2: var_87}
    var_89 = {}
    var_90 = {}
    var_91 = module_0.ParsedContent()
    var_92 = 'thirdparty'
    var_93 = 'Third Party Imports'
    var_94 = {var_92: var_93}
    var_95 = module_1.Config()
    var_96 = module_2.sorted_imports(var_91, var_95)
    assert var_96 == '# Third Party Imports\nimport os\n'
    var_97 = [var_0]
    var_98 = 'FUTURE'
    var_99 = '__future__'
    var_100 = 'print_function'
    var_101 = [var_100]
    var_102 = {var_99: var_101}
    var_103 = {}
    var_104 = {var_3: var_102, var_4: var_103}
    var_105 = []
    var_106 = {var_5: var_105}
    var_107 = {}
    var_108 = {var_3: var_106, var_4: var_107}
    var_109 = {var_98: var_104, var_2: var_108}
    var_110 = {}
    var_111 = {}
    var_112 = module_0.ParsedContent()
    var_113 = 2
    var_114 = module_1.Config()
    var_115 = module_2.sorted_imports(var_112, var_114)
    assert var_115 == 'from __future__ import print_function\n\n\nimport os\n'
    var_116 = [var_0]
    var_117 = []
    var_118 = []
    var_119 = {var_5: var_117, var_6: var_118}
    var_120 = {}
    var_121 = {var_3: var_119, var_4: var_120}
    var_122 = {var_2: var_121}
    var_123 = {}
    var_124 = {}
    var_125 = module_0.ParsedContent()
    var_126 = [var_5]
    var_127 = module_1.Config()
    var_128 = module_2.sorted_imports(var_125, var_127)
    assert var_128 == 'import sys\n'
    var_129 = "print('hello')"
    var_130 = [var_129]
    var_131 = {}
    var_132 = -1
    var_133 = {}
    var_134 = {}
    var_135 = module_0.ParsedContent()
    var_136 = module_1.Config()
    var_137 = module_2.sorted_imports(var_135, var_136)
    assert var_137 == "print('hello')\n"



# Parsed testcases at query #44
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = '# Test file'
    var_1 = ''
    var_2 = [var_0, var_1]
    var_3 = 'FUTURE'
    var_4 = 'straight'
    var_5 = 'from'
    var_6 = '__future__'
    var_7 = 'print_function'
    var_8 = [var_7]
    var_9 = {var_6: var_8}
    var_10 = {}
    var_11 = {var_4: var_9, var_5: var_10}
    var_12 = {var_3: var_11}
    var_13 = 1
    var_14 = 2
    var_15 = '\n'
    var_16 = module_0.ParsedContent()
    var_17 = True
    var_18 = module_1.Config()
    var_19 = module_2.sorted_imports(var_16, var_18)
    assert var_19 == '# Test file\n\nfrom __future__ import print_function'
    var_20 = [var_0, var_1]
    var_21 = 'STDLIB'
    var_22 = [var_7]
    var_23 = {var_6: var_22}
    var_24 = {}
    var_25 = {var_4: var_23, var_5: var_24}
    var_26 = 'os'
    var_27 = 'sys'
    var_28 = []
    var_29 = []
    var_30 = {var_26: var_28, var_27: var_29}
    var_31 = {}
    var_32 = {var_4: var_30, var_5: var_31}
    var_33 = {var_3: var_25, var_21: var_32}
    var_34 = module_0.ParsedContent()
    var_35 = False
    var_36 = module_1.Config()
    var_37 = module_2.sorted_imports(var_34, var_36)
    assert var_37 == '# Test file\n\nfrom __future__ import print_function\n\nimport os\nimport sys'
    var_38 = [var_0, var_1]
    var_39 = 'THIRDPARTY'
    var_40 = [var_7]
    var_41 = {var_6: var_40}
    var_42 = {}
    var_43 = {var_4: var_41, var_5: var_42}
    var_44 = 'django'
    var_45 = 'flask'
    var_46 = []
    var_47 = []
    var_48 = {var_44: var_46, var_45: var_47}
    var_49 = {}
    var_50 = {var_4: var_48, var_5: var_49}
    var_51 = {var_3: var_43, var_39: var_50}
    var_52 = module_0.ParsedContent()
    var_53 = [var_44]
    var_54 = module_1.Config()
    var_55 = module_2.sorted_imports(var_52, var_54)
    assert var_55 == '# Test file\n\nfrom __future__ import print_function\n\nimport flask'
    var_56 = [var_0, var_1]
    var_57 = [var_7]
    var_58 = {var_6: var_57}
    var_59 = {}
    var_60 = {var_4: var_58, var_5: var_59}
    var_61 = []
    var_62 = []
    var_63 = {var_26: var_61, var_27: var_62}
    var_64 = {}
    var_65 = {var_4: var_63, var_5: var_64}
    var_66 = {var_3: var_60, var_21: var_65}
    var_67 = module_0.ParsedContent()
    var_68 = [var_26]
    var_69 = module_1.Config()
    var_70 = module_2.sorted_imports(var_67, var_69)
    assert var_70 == '# Test file\n\nfrom __future__ import print_function\n\nimport os\n\nimport sys'
    var_71 = [var_0, var_1]
    var_72 = [var_7]
    var_73 = {var_6: var_72}
    var_74 = {}
    var_75 = {var_4: var_73, var_5: var_74}
    var_76 = []
    var_77 = []
    var_78 = {var_26: var_76, var_27: var_77}
    var_79 = {}
    var_80 = {var_4: var_78, var_5: var_79}
    var_81 = {var_3: var_75, var_21: var_80}
    var_82 = module_0.ParsedContent()
    var_83 = module_1.Config()
    var_84 = module_2.sorted_imports(var_82, var_83)
    assert var_84 == '# Test file\n\nfrom __future__ import print_function\n\n\n\nimport os\nimport sys'
    var_85 = [var_0, var_1]
    var_86 = {}
    var_87 = [var_7]
    var_88 = {var_6: var_87}
    var_89 = {var_4: var_86, var_5: var_88}
    var_90 = []
    var_91 = {var_26: var_90}
    var_92 = 'argv'
    var_93 = [var_92]
    var_94 = {var_27: var_93}
    var_95 = {var_4: var_91, var_5: var_94}
    var_96 = {var_3: var_89, var_21: var_95}
    var_97 = module_0.ParsedContent()
    var_98 = True
    var_99 = module_1.Config()
    var_100 = module_2.sorted_imports(var_97, var_99)
    assert var_100 == '# Test file\n\nfrom __future__ import print_function\nfrom sys import argv\n\nimport os'
    var_101 = [var_0, var_1]
    var_102 = {}
    var_103 = {}
    var_104 = {var_4: var_102, var_5: var_103}
    var_105 = {}
    var_106 = '*'
    var_107 = [var_106]
    var_108 = [var_92]
    var_109 = {var_26: var_107, var_27: var_108}
    var_110 = {var_4: var_105, var_5: var_109}
    var_111 = {var_3: var_104, var_21: var_110}
    var_112 = module_0.ParsedContent()
    var_113 = True
    var_114 = module_1.Config()
    var_115 = module_2.sorted_imports(var_112, var_114)
    assert var_115 == '# Test file\n\nfrom os import *\nfrom sys import argv'
    var_116 = [var_0, var_1]
    var_117 = [var_7]
    var_118 = {var_6: var_117}
    var_119 = {}
    var_120 = {var_4: var_118, var_5: var_119}
    var_121 = []
    var_122 = {var_26: var_121}
    var_123 = {}
    var_124 = {var_4: var_122, var_5: var_123}
    var_125 = {var_3: var_120, var_21: var_124}
    var_126 = module_0.ParsedContent()
    var_127 = 'future'
    var_128 = 'stdlib'
    var_129 = 'Future imports'
    var_130 = 'Standard library'
    var_131 = {var_127: var_129, var_128: var_130}
    var_132 = module_1.Config()
    var_133 = module_2.sorted_imports(var_126, var_132)
    assert var_133 == '# Test file\n\n# Future imports\nfrom __future__ import print_function\n\n# Standard library\nimport os'
    var_134 = [var_0, var_1]
    var_135 = {}
    var_136 = {}
    var_137 = {var_4: var_135, var_5: var_136}
    var_138 = []
    var_139 = []
    var_140 = {var_26: var_138, var_27: var_139}
    var_141 = {}
    var_142 = {var_4: var_140, var_5: var_141}
    var_143 = {var_3: var_137, var_21: var_142}
    var_144 = module_0.ParsedContent()
    var_145 = True
    var_146 = module_1.Config()
    var_147 = module_2.sorted_imports(var_144, var_146)
    assert var_147 == '# Test file\n\nimport os\nimport sys'
    var_148 = [var_0, var_1]
    var_149 = [var_7]
    var_150 = {var_6: var_149}
    var_151 = {}
    var_152 = {var_4: var_150, var_5: var_151}
    var_153 = []
    var_154 = {var_26: var_153}
    var_155 = {}
    var_156 = {var_4: var_154, var_5: var_155}
    var_157 = {var_3: var_152, var_21: var_156}
    var_158 = module_0.ParsedContent()
    var_159 = module_1.Config()
    var_160 = module_2.sorted_imports(var_158, var_159)
    assert var_160 == '\n\n# Test file\n\nfrom __future__ import print_function\n\nimport os'



# Parsed testcases at query #45
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = -1
    var_3 = '\n'
    var_4 = {}
    var_5 = {}
    var_6 = {}
    var_7 = []
    var_8 = 1
    var_9 = module_0.ParsedContent()
    var_10 = module_1.Config()
    var_11 = module_2.sorted_imports(var_9, var_10)
    assert var_11 == "print('hello')"
    var_12 = [var_0]
    var_13 = 0
    var_14 = 'THIRDPARTY'
    var_15 = 'straight'
    var_16 = 'from'
    var_17 = 'os'
    var_18 = 'sys'
    var_19 = 'os.path'
    var_20 = [var_19]
    var_21 = 'sys.path'
    var_22 = [var_21]
    var_23 = {var_17: var_20, var_18: var_22}
    var_24 = 'collections'
    var_25 = 'defaultdict'
    var_26 = [var_25]
    var_27 = {var_24: var_26}
    var_28 = {var_15: var_23, var_16: var_27}
    var_29 = {var_14: var_28}
    var_30 = {}
    var_31 = {}
    var_32 = [var_14]
    var_33 = module_0.ParsedContent()
    var_34 = module_1.Config()
    var_35 = module_2.sorted_imports(var_33, var_34)
    var_36 = "import os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')"
    var_37 = [var_0]
    var_38 = [var_19]
    var_39 = [var_21]
    var_40 = {var_17: var_38, var_18: var_39}
    var_41 = [var_25]
    var_42 = {var_24: var_41}
    var_43 = {var_15: var_40, var_16: var_42}
    var_44 = {var_14: var_43}
    var_45 = {}
    var_46 = {}
    var_47 = [var_14]
    var_48 = module_0.ParsedContent()
    var_49 = 2
    var_50 = True
    var_51 = module_1.Config()
    var_52 = module_2.sorted_imports(var_48, var_51)
    var_53 = "from collections import defaultdict\n\nimport os\nimport sys\n\nprint('hello')"
    var_54 = [var_0]
    var_55 = [var_19]
    var_56 = [var_21]
    var_57 = {var_17: var_55, var_18: var_56}
    var_58 = [var_25]
    var_59 = {var_24: var_58}
    var_60 = {var_15: var_57, var_16: var_59}
    var_61 = {var_14: var_60}
    var_62 = {}
    var_63 = {}
    var_64 = [var_14]
    var_65 = module_0.ParsedContent()
    var_66 = 'FUTURE'
    var_67 = [var_66]
    var_68 = module_1.Config()
    var_69 = module_2.sorted_imports(var_65, var_68)
    var_70 = "import os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')"
    var_71 = [var_0]
    var_72 = [var_19]
    var_73 = [var_21]
    var_74 = {var_17: var_72, var_18: var_73}
    var_75 = [var_25]
    var_76 = {var_24: var_75}
    var_77 = {var_15: var_74, var_16: var_76}
    var_78 = {var_14: var_77}
    var_79 = {}
    var_80 = {}
    var_81 = [var_14]
    var_82 = module_0.ParsedContent()
    var_83 = True
    var_84 = module_1.Config()
    var_85 = module_2.sorted_imports(var_82, var_84)
    var_86 = "import os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')"



# Parsed testcases at query #46
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 'THIRDPARTY'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'zlib'
    var_6 = 'os'
    var_7 = []
    var_8 = []
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'sys'
    var_11 = 'json'
    var_12 = 'path'
    var_13 = [var_12]
    var_14 = 'load'
    var_15 = [var_14]
    var_16 = {var_10: var_13, var_11: var_15}
    var_17 = {var_3: var_9, var_4: var_16}
    var_18 = {var_2: var_17}
    var_19 = 0
    var_20 = 1
    var_21 = '\n'
    var_22 = module_0.ParsedContent()
    var_23 = module_1.Config()
    var_24 = module_2.sorted_imports(var_22, var_23)
    assert var_24 == 'import os\nimport zlib\n\nfrom json import load\nfrom sys import path\n'
    var_25 = [var_0]
    var_26 = 'FUTURE'
    var_27 = '__future__'
    var_28 = 'annotations'
    var_29 = [var_28]
    var_30 = {var_27: var_29}
    var_31 = {}
    var_32 = {var_3: var_30, var_4: var_31}
    var_33 = []
    var_34 = []
    var_35 = {var_5: var_33, var_6: var_34}
    var_36 = {}
    var_37 = {var_3: var_35, var_4: var_36}
    var_38 = {var_26: var_32, var_2: var_37}
    var_39 = module_0.ParsedContent()
    var_40 = True
    var_41 = module_1.Config()
    var_42 = module_2.sorted_imports(var_39, var_41)
    assert var_42 == 'from __future__ import annotations\nimport os\nimport zlib\n'
    var_43 = [var_0]
    var_44 = 'FIRSTPARTY'
    var_45 = []
    var_46 = []
    var_47 = {var_5: var_45, var_6: var_46}
    var_48 = {}
    var_49 = {var_3: var_47, var_4: var_48}
    var_50 = 'my_module'
    var_51 = []
    var_52 = {var_50: var_51}
    var_53 = {}
    var_54 = {var_3: var_52, var_4: var_53}
    var_55 = {var_2: var_49, var_44: var_54}
    var_56 = module_0.ParsedContent()
    var_57 = 'LOCALFOLDER'
    var_58 = [var_57]
    var_59 = module_1.Config()
    var_60 = module_2.sorted_imports(var_56, var_59)
    assert var_60 == 'import os\nimport zlib\n\nimport my_module\n'
    var_61 = [var_0]
    var_62 = []
    var_63 = []
    var_64 = {var_5: var_62, var_6: var_63}
    var_65 = {}
    var_66 = {var_3: var_64, var_4: var_65}
    var_67 = {var_2: var_66}
    var_68 = module_0.ParsedContent()
    var_69 = [var_5]
    var_70 = module_1.Config()
    var_71 = module_2.sorted_imports(var_68, var_70)
    assert var_71 == 'import os\n'
    var_72 = [var_0]
    var_73 = {}
    var_74 = 'module1'
    var_75 = 'module2'
    var_76 = '*'
    var_77 = [var_76]
    var_78 = 'func'
    var_79 = [var_78]
    var_80 = {var_74: var_77, var_75: var_79}
    var_81 = {var_3: var_73, var_4: var_80}
    var_82 = {var_2: var_81}
    var_83 = module_0.ParsedContent()
    var_84 = True
    var_85 = module_1.Config()
    var_86 = module_2.sorted_imports(var_83, var_85)
    assert var_86 == 'from module1 import *\nfrom module2 import func\n'
    var_87 = [var_0]
    var_88 = []
    var_89 = {var_6: var_88}
    var_90 = [var_12]
    var_91 = {var_10: var_90}
    var_92 = {var_3: var_89, var_4: var_91}
    var_93 = {var_2: var_92}
    var_94 = module_0.ParsedContent()
    var_95 = True
    var_96 = module_1.Config()
    var_97 = module_2.sorted_imports(var_94, var_96)
    assert var_97 == 'from sys import path\n\nimport os\n'
    var_98 = [var_0]
    var_99 = []
    var_100 = {var_6: var_99}
    var_101 = {}
    var_102 = {var_3: var_100, var_4: var_101}
    var_103 = {var_2: var_102}
    var_104 = module_0.ParsedContent()
    var_105 = 'thirdparty'
    var_106 = 'Third Party Imports'
    var_107 = {var_105: var_106}
    var_108 = module_1.Config()
    var_109 = module_2.sorted_imports(var_104, var_108)
    assert var_109 == '# Third Party Imports\nimport os\n'
    var_110 = [var_0]
    var_111 = [var_28]
    var_112 = {var_27: var_111}
    var_113 = {}
    var_114 = {var_3: var_112, var_4: var_113}
    var_115 = []
    var_116 = {var_6: var_115}
    var_117 = {}
    var_118 = {var_3: var_116, var_4: var_117}
    var_119 = {var_26: var_114, var_2: var_118}
    var_120 = module_0.ParsedContent()
    var_121 = 2
    var_122 = module_1.Config()
    var_123 = module_2.sorted_imports(var_120, var_122)
    assert var_123 == 'from __future__ import annotations\n\n\nimport os\n'
    var_124 = 'def main():'
    var_125 = '    pass'
    var_126 = [var_124, var_125]
    var_127 = []
    var_128 = {var_6: var_127}
    var_129 = {}
    var_130 = {var_3: var_128, var_4: var_129}
    var_131 = {var_2: var_130}
    var_132 = module_0.ParsedContent()
    var_133 = module_1.Config()
    var_134 = module_2.sorted_imports(var_132, var_133)
    assert var_134 == 'import os\n\n\ndef main():\n    pass\n'
    var_135 = [var_0]
    var_136 = []
    var_137 = {var_6: var_136}
    var_138 = {}
    var_139 = {var_3: var_137, var_4: var_138}
    var_140 = {var_2: var_139}
    var_141 = module_0.ParsedContent()
    var_142 = 'import'
    var_143 = lambda code, ext, cfg: code.replace(var_142, var_4)
    var_144 = module_1.Config()
    var_145 = module_2.sorted_imports(var_141, var_144)
    assert var_145 == 'from os\n'



# Parsed testcases at query #47
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = 0
    var_5 = '\n'
    var_6 = {}
    var_7 = {}
    var_8 = module_0.ParsedContent()
    var_9 = "print('hello')"
    var_10 = [var_9]
    var_11 = {}
    var_12 = -1
    var_13 = 1
    var_14 = {}
    var_15 = {}
    var_16 = module_0.ParsedContent()
    var_17 = [var_0]
    var_18 = 'THIRDPARTY'
    var_19 = 'straight'
    var_20 = 'from'
    var_21 = 'os'
    var_22 = 'sys'
    var_23 = [var_21]
    var_24 = [var_22]
    var_25 = {var_21: var_23, var_22: var_24}
    var_26 = 'collections'
    var_27 = 'defaultdict'
    var_28 = 'OrderedDict'
    var_29 = [var_27, var_28]
    var_30 = {var_26: var_29}
    var_31 = {var_19: var_25, var_20: var_30}
    var_32 = {var_18: var_31}
    var_33 = {}
    var_34 = {}
    var_35 = module_0.ParsedContent()
    var_36 = 'from collections import defaultdict, OrderedDict\nimport os\nimport sys\n'
    var_37 = [var_0]
    var_38 = 'FIRSTPARTY'
    var_39 = [var_21]
    var_40 = {var_21: var_39}
    var_41 = [var_27]
    var_42 = {var_26: var_41}
    var_43 = {var_19: var_40, var_20: var_42}
    var_44 = [var_22]
    var_45 = {var_22: var_44}
    var_46 = 'typing'
    var_47 = 'List'
    var_48 = [var_47]
    var_49 = {var_46: var_48}
    var_50 = {var_19: var_45, var_20: var_49}
    var_51 = {var_18: var_43, var_38: var_50}
    var_52 = {}
    var_53 = {}
    var_54 = module_0.ParsedContent()
    var_55 = [var_38]
    var_56 = module_1.Config()
    var_57 = module_2.sorted_imports(var_54, var_56)
    var_58 = 'from collections import defaultdict\nimport os\n\nfrom typing import List\nimport sys\n'
    var_59 = [var_0]
    var_60 = [var_21]
    var_61 = {var_21: var_60}
    var_62 = [var_27]
    var_63 = {var_26: var_62}
    var_64 = {var_19: var_61, var_20: var_63}
    var_65 = [var_22]
    var_66 = {var_22: var_65}
    var_67 = [var_47]
    var_68 = {var_46: var_67}
    var_69 = {var_19: var_66, var_20: var_68}
    var_70 = {var_18: var_64, var_38: var_69}
    var_71 = {}
    var_72 = {}
    var_73 = module_0.ParsedContent()
    var_74 = True
    var_75 = module_1.Config()
    var_76 = module_2.sorted_imports(var_73, var_75)
    var_77 = 'from collections import defaultdict\nfrom typing import List\nimport os\nimport sys\n'
    var_78 = [var_0]
    var_79 = {}
    var_80 = '*'
    var_81 = [var_80]
    var_82 = 'path'
    var_83 = [var_82]
    var_84 = [var_80]
    var_85 = {var_26: var_81, var_21: var_83, var_22: var_84}
    var_86 = {var_19: var_79, var_20: var_85}
    var_87 = {var_18: var_86}
    var_88 = {}
    var_89 = {}
    var_90 = module_0.ParsedContent()
    var_91 = True
    var_92 = module_1.Config()
    var_93 = module_2.sorted_imports(var_90, var_92)
    var_94 = 'from collections import *\nfrom sys import *\nfrom os import path\n'
    var_95 = [var_0]
    var_96 = [var_21]
    var_97 = {var_21: var_96}
    var_98 = [var_27]
    var_99 = {var_26: var_98}
    var_100 = {var_19: var_97, var_20: var_99}
    var_101 = {var_18: var_100}
    var_102 = {}
    var_103 = {}
    var_104 = module_0.ParsedContent()
    var_105 = True
    var_106 = module_1.Config()
    var_107 = module_2.sorted_imports(var_104, var_106)
    var_108 = 'from collections import defaultdict\n\nimport os\n'
    var_109 = [var_0]
    var_110 = [var_21]
    var_111 = {var_21: var_110}
    var_112 = [var_27]
    var_113 = {var_26: var_112}
    var_114 = {var_19: var_111, var_20: var_113}
    var_115 = {var_18: var_114}
    var_116 = {}
    var_117 = {}
    var_118 = module_0.ParsedContent()
    var_119 = 'thirdparty'
    var_120 = 'Third Party Imports'
    var_121 = {var_119: var_120}
    var_122 = module_1.Config()
    var_123 = module_2.sorted_imports(var_118, var_122)
    var_124 = '# Third Party Imports\nfrom collections import defaultdict\nimport os\n'
    var_125 = [var_0]
    var_126 = [var_21]
    var_127 = {var_21: var_126}
    var_128 = {}
    var_129 = {var_19: var_127, var_20: var_128}
    var_130 = [var_22]
    var_131 = {var_22: var_130}
    var_132 = {}
    var_133 = {var_19: var_131, var_20: var_132}
    var_134 = {var_18: var_129, var_38: var_133}
    var_135 = {}
    var_136 = {}
    var_137 = module_0.ParsedContent()
    var_138 = 2
    var_139 = module_1.Config()
    var_140 = module_2.sorted_imports(var_137, var_139)
    var_141 = 'import os\n\n\nimport sys\n'
    var_142 = [var_9]
    var_143 = [var_21]
    var_144 = {var_21: var_143}
    var_145 = {}
    var_146 = {var_19: var_144, var_20: var_145}
    var_147 = {var_18: var_146}
    var_148 = {}
    var_149 = {}
    var_150 = module_0.ParsedContent()
    var_151 = module_1.Config()
    var_152 = module_2.sorted_imports(var_150, var_151)
    var_153 = "import os\n\n\nprint('hello')\n"



# Parsed testcases at query #48
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = 0
    var_5 = '\n'
    var_6 = {}
    var_7 = {}
    var_8 = module_0.ParsedContent()
    var_9 = "print('hello')"
    var_10 = [var_9]
    var_11 = {}
    var_12 = -1
    var_13 = 1
    var_14 = {}
    var_15 = {}
    var_16 = module_0.ParsedContent()
    var_17 = [var_0]
    var_18 = 'THIRDPARTY'
    var_19 = 'straight'
    var_20 = 'from'
    var_21 = 'zlib'
    var_22 = 'os'
    var_23 = []
    var_24 = []
    var_25 = {var_21: var_23, var_22: var_24}
    var_26 = 'sys'
    var_27 = 'json'
    var_28 = 'path'
    var_29 = [var_28]
    var_30 = 'load'
    var_31 = [var_30]
    var_32 = {var_26: var_29, var_27: var_31}
    var_33 = {var_19: var_25, var_20: var_32}
    var_34 = {var_18: var_33}
    var_35 = {}
    var_36 = {}
    var_37 = module_0.ParsedContent()
    var_38 = 'import os\nimport zlib\n\nfrom json import load\nfrom sys import path\n'
    var_39 = [var_0]
    var_40 = []
    var_41 = []
    var_42 = {var_21: var_40, var_22: var_41}
    var_43 = [var_28]
    var_44 = [var_30]
    var_45 = {var_26: var_43, var_27: var_44}
    var_46 = {var_19: var_42, var_20: var_45}
    var_47 = {var_18: var_46}
    var_48 = {}
    var_49 = {}
    var_50 = module_0.ParsedContent()
    var_51 = True
    var_52 = True
    var_53 = 2
    var_54 = module_1.Config()
    var_55 = module_2.sorted_imports(var_50, var_54)
    var_56 = 'from sys import path\nfrom json import load\n\nimport zlib\nimport os\n'
    var_57 = [var_0]
    var_58 = []
    var_59 = []
    var_60 = {var_21: var_58, var_22: var_59}
    var_61 = [var_28]
    var_62 = [var_30]
    var_63 = {var_26: var_61, var_27: var_62}
    var_64 = {var_19: var_60, var_20: var_63}
    var_65 = {var_18: var_64}
    var_66 = {}
    var_67 = {}
    var_68 = module_0.ParsedContent()
    var_69 = [var_21, var_22]
    var_70 = module_1.Config()
    var_71 = module_2.sorted_imports(var_68, var_70)
    var_72 = 'import os\nimport zlib\n\nfrom json import load\nfrom sys import path\n'
    var_73 = [var_0]
    var_74 = []
    var_75 = []
    var_76 = {var_21: var_74, var_22: var_75}
    var_77 = [var_28]
    var_78 = [var_30]
    var_79 = {var_26: var_77, var_27: var_78}
    var_80 = {var_19: var_76, var_20: var_79}
    var_81 = {var_18: var_80}
    var_82 = {}
    var_83 = {}
    var_84 = module_0.ParsedContent()
    var_85 = True
    var_86 = module_1.Config()
    var_87 = module_2.sorted_imports(var_84, var_86)
    var_88 = 'import json\nimport os\nimport sys\nimport zlib\n'
    var_89 = [var_0]
    var_90 = '*'
    var_91 = [var_90]
    var_92 = [var_30]
    var_93 = {var_26: var_91, var_27: var_92}
    var_94 = {var_20: var_93}
    var_95 = {var_18: var_94}
    var_96 = {}
    var_97 = {}
    var_98 = module_0.ParsedContent()
    var_99 = True
    var_100 = module_1.Config()
    var_101 = module_2.sorted_imports(var_98, var_100)
    var_102 = 'from sys import *\nfrom json import load\n'
    var_103 = [var_0]
    var_104 = []
    var_105 = []
    var_106 = {var_21: var_104, var_22: var_105}
    var_107 = [var_28]
    var_108 = [var_30]
    var_109 = {var_26: var_107, var_27: var_108}
    var_110 = {var_19: var_106, var_20: var_109}
    var_111 = {var_18: var_110}
    var_112 = {}
    var_113 = {}
    var_114 = module_0.ParsedContent()
    var_115 = 'thirdparty'
    var_116 = 'Third Party Imports'
    var_117 = {var_115: var_116}
    var_118 = module_1.Config()
    var_119 = module_2.sorted_imports(var_114, var_118)
    var_120 = '# Third Party Imports\nimport os\nimport zlib\n\nfrom json import load\nfrom sys import path\n'
    var_121 = [var_0]
    var_122 = []
    var_123 = []
    var_124 = {var_21: var_122, var_22: var_123}
    var_125 = [var_28]
    var_126 = [var_30]
    var_127 = {var_26: var_125, var_27: var_126}
    var_128 = {var_19: var_124, var_20: var_127}
    var_129 = {var_18: var_128}
    var_130 = {}
    var_131 = {}
    var_132 = module_0.ParsedContent()
    var_133 = lambda code, ext, cfg: code.upper()
    var_134 = module_1.Config()
    var_135 = module_2.sorted_imports(var_132, var_134)
    var_136 = 'IMPORT OS\nIMPORT ZLIB\n\nFROM JSON IMPORT LOAD\nFROM SYS IMPORT PATH\n'
    var_137 = '# Placeholder'
    var_138 = [var_137, var_9]
    var_139 = []
    var_140 = []
    var_141 = {var_21: var_139, var_22: var_140}
    var_142 = [var_28]
    var_143 = [var_30]
    var_144 = {var_26: var_142, var_27: var_143}
    var_145 = {var_19: var_141, var_20: var_144}
    var_146 = {var_18: var_145}
    var_147 = 'import os'
    var_148 = 'import zlib'
    var_149 = [var_147, var_148]
    var_150 = {var_137: var_149}
    var_151 = {var_137: var_137}
    var_152 = module_0.ParsedContent()
    var_153 = module_2.sorted_imports(var_152, var_134)
    var_154 = "# Placeholder\nimport os\nimport zlib\n\nprint('hello')\n"



# Parsed testcases at query #49
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'x = 1'
    var_1 = [var_0]
    var_2 = 'THIRDPARTY'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = 'sys'
    var_7 = []
    var_8 = []
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'collections'
    var_11 = 'defaultdict'
    var_12 = [var_11]
    var_13 = {var_10: var_12}
    var_14 = {var_3: var_9, var_4: var_13}
    var_15 = {var_2: var_14}
    var_16 = 0
    var_17 = 1
    var_18 = '\n'
    var_19 = module_0.ParsedContent()
    var_20 = module_1.Config()
    var_21 = module_2.sorted_imports(var_19, var_20)
    assert var_21 == 'import os\nimport sys\n\nfrom collections import defaultdict\n\nx = 1\n'
    var_22 = [var_0]
    var_23 = 'FUTURE'
    var_24 = '__future__'
    var_25 = []
    var_26 = {var_24: var_25}
    var_27 = {}
    var_28 = {var_3: var_26, var_4: var_27}
    var_29 = []
    var_30 = []
    var_31 = {var_5: var_29, var_6: var_30}
    var_32 = {}
    var_33 = {var_3: var_31, var_4: var_32}
    var_34 = {var_23: var_28, var_2: var_33}
    var_35 = module_0.ParsedContent()
    var_36 = True
    var_37 = module_1.Config()
    var_38 = module_2.sorted_imports(var_35, var_37)
    assert var_38 == 'import __future__\nimport os\nimport sys\n\nx = 1\n'
    var_39 = [var_0]
    var_40 = []
    var_41 = {var_5: var_40}
    var_42 = [var_11]
    var_43 = {var_10: var_42}
    var_44 = {var_3: var_41, var_4: var_43}
    var_45 = {var_2: var_44}
    var_46 = module_0.ParsedContent()
    var_47 = True
    var_48 = module_1.Config()
    var_49 = module_2.sorted_imports(var_46, var_48)
    assert var_49 == 'from collections import defaultdict\n\nimport os\n\nx = 1\n'
    var_50 = [var_0]
    var_51 = {}
    var_52 = '*'
    var_53 = [var_52]
    var_54 = 'path'
    var_55 = [var_54]
    var_56 = {var_10: var_53, var_5: var_55}
    var_57 = {var_3: var_51, var_4: var_56}
    var_58 = {var_2: var_57}
    var_59 = module_0.ParsedContent()
    var_60 = True
    var_61 = module_1.Config()
    var_62 = module_2.sorted_imports(var_59, var_61)
    assert var_62 == 'from collections import *\nfrom os import path\n\nx = 1\n'
    var_63 = [var_0]
    var_64 = []
    var_65 = {var_5: var_64}
    var_66 = {}
    var_67 = {var_3: var_65, var_4: var_66}
    var_68 = {var_2: var_67}
    var_69 = module_0.ParsedContent()
    var_70 = 'thirdparty'
    var_71 = 'Third Party Imports'
    var_72 = {var_70: var_71}
    var_73 = module_1.Config()
    var_74 = module_2.sorted_imports(var_69, var_73)
    assert var_74 == '# Third Party Imports\nimport os\n\nx = 1\n'
    var_75 = [var_0]
    var_76 = {}
    var_77 = -1
    var_78 = module_0.ParsedContent()
    var_79 = module_1.Config()
    var_80 = module_2.sorted_imports(var_78, var_79)
    assert var_80 == 'x = 1\n'
    var_81 = [var_0]
    var_82 = []
    var_83 = {var_24: var_82}
    var_84 = {}
    var_85 = {var_3: var_83, var_4: var_84}
    var_86 = []
    var_87 = {var_5: var_86}
    var_88 = {}
    var_89 = {var_3: var_87, var_4: var_88}
    var_90 = {var_23: var_85, var_2: var_89}
    var_91 = module_0.ParsedContent()
    var_92 = 2
    var_93 = module_1.Config()
    var_94 = module_2.sorted_imports(var_91, var_93)
    assert var_94 == 'import __future__\n\n\nimport os\n\nx = 1\n'
    var_95 = [var_0]
    var_96 = []
    var_97 = {var_5: var_96}
    var_98 = {}
    var_99 = {var_3: var_97, var_4: var_98}
    var_100 = {var_2: var_99}
    var_101 = module_0.ParsedContent()
    var_102 = module_1.Config()
    var_103 = module_2.sorted_imports(var_101, var_102)
    assert var_103 == 'import os\n\n\nx = 1\n'
    var_104 = [var_0]
    var_105 = []
    var_106 = []
    var_107 = {var_5: var_105, var_6: var_106}
    var_108 = {}
    var_109 = {var_3: var_107, var_4: var_108}
    var_110 = {var_2: var_109}
    var_111 = module_0.ParsedContent()
    var_112 = [var_6]
    var_113 = module_1.Config()
    var_114 = module_2.sorted_imports(var_111, var_113)
    assert var_114 == 'import os\n\nx = 1\n'



# Parsed testcases at query #50
#--------------------------


import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = ''
    var_3 = 'def foo():'
    var_4 = '    pass'
    var_5 = [var_2, var_3, var_4]
    var_6 = 'FUTURE'
    var_7 = 'THIRDPARTY'
    var_8 = 'straight'
    var_9 = 'from'
    var_10 = '__future__'
    var_11 = 'import annotations'
    var_12 = [var_11]
    var_13 = {var_10: var_12}
    var_14 = {}
    var_15 = {var_8: var_13, var_9: var_14}
    var_16 = 'os'
    var_17 = 'sys'
    var_18 = 'import os'
    var_19 = [var_18]
    var_20 = 'import sys'
    var_21 = [var_20]
    var_22 = {var_16: var_19, var_17: var_21}
    var_23 = {}
    var_24 = {var_8: var_22, var_9: var_23}
    var_25 = {var_6: var_15, var_7: var_24}
    var_26 = 0
    var_27 = 3
    var_28 = '\n'
    var_29 = module_1.ParsedContent()
    var_30 = module_2.sorted_imports(var_29, var_1)
    assert var_30 == '\nimport os\nimport sys\n\ndef foo():\n    pass'
    var_31 = 'Future Imports'
    var_32 = {var_6: var_31}
    var_33 = module_0.Config()
    var_34 = [var_2, var_3, var_4]
    var_35 = [var_11]
    var_36 = {var_10: var_35}
    var_37 = {}
    var_38 = {var_8: var_36, var_9: var_37}
    var_39 = [var_18]
    var_40 = [var_20]
    var_41 = {var_16: var_39, var_17: var_40}
    var_42 = {}
    var_43 = {var_8: var_41, var_9: var_42}
    var_44 = {var_6: var_38, var_7: var_43}
    var_45 = module_1.ParsedContent()
    var_46 = module_2.sorted_imports(var_45, var_33)
    assert var_46 == '\n# Future Imports\nfrom __future__ import annotations\n\nimport os\nimport sys\n\ndef foo():\n    pass'
    var_47 = [var_16]
    var_48 = module_0.Config()
    var_49 = [var_2, var_3, var_4]
    var_50 = [var_18]
    var_51 = [var_20]
    var_52 = {var_16: var_50, var_17: var_51}
    var_53 = {}
    var_54 = {var_8: var_52, var_9: var_53}
    var_55 = {var_7: var_54}
    var_56 = module_1.ParsedContent()
    var_57 = module_2.sorted_imports(var_56, var_48)
    assert var_57 == '\nimport sys\n\ndef foo():\n    pass'
    var_58 = module_0.Config()
    var_59 = [var_2, var_3, var_4]
    var_60 = [var_18]
    var_61 = [var_20]
    var_62 = {var_16: var_60, var_17: var_61}
    var_63 = 'numpy'
    var_64 = 'pandas'
    var_65 = 'import numpy as np'
    var_66 = [var_65]
    var_67 = 'import pandas as pd'
    var_68 = [var_67]
    var_69 = {var_63: var_66, var_64: var_68}
    var_70 = {var_8: var_62, var_9: var_69}
    var_71 = {var_7: var_70}
    var_72 = module_1.ParsedContent()
    var_73 = module_2.sorted_imports(var_72, var_58)
    assert var_73 == '\nimport os\nimport sys\nfrom numpy import numpy as np\nfrom pandas import pandas as pd\n\ndef foo():\n    pass'
    var_74 = module_0.Config()
    var_75 = [var_2, var_3, var_4]
    var_76 = [var_18]
    var_77 = [var_20]
    var_78 = {var_16: var_76, var_17: var_77}
    var_79 = [var_65]
    var_80 = [var_67]
    var_81 = {var_63: var_79, var_64: var_80}
    var_82 = {var_8: var_78, var_9: var_81}
    var_83 = {var_7: var_82}
    var_84 = module_1.ParsedContent()
    var_85 = module_2.sorted_imports(var_84, var_74)
    assert var_85 == '\nfrom numpy import numpy as np\nfrom pandas import pandas as pd\n\nimport os\nimport sys\n\ndef foo():\n    pass'



# Parsed testcases at query #51
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 'THIRDPARTY'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = 'sys'
    var_7 = []
    var_8 = []
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'itertools'
    var_11 = 'collections'
    var_12 = 'chain'
    var_13 = [var_12]
    var_14 = 'abc'
    var_15 = [var_14]
    var_16 = {var_10: var_13, var_11: var_15}
    var_17 = {var_3: var_9, var_4: var_16}
    var_18 = {var_2: var_17}
    var_19 = 0
    var_20 = 1
    var_21 = '\n'
    var_22 = module_0.ParsedContent()
    var_23 = module_1.Config()
    var_24 = module_2.sorted_imports(var_22, var_23)
    var_25 = [var_0]
    var_26 = 'FUTURE'
    var_27 = '__future__'
    var_28 = 'annotations'
    var_29 = [var_28]
    var_30 = {var_27: var_29}
    var_31 = {}
    var_32 = {var_3: var_30, var_4: var_31}
    var_33 = []
    var_34 = []
    var_35 = {var_5: var_33, var_6: var_34}
    var_36 = [var_12]
    var_37 = [var_14]
    var_38 = {var_10: var_36, var_11: var_37}
    var_39 = {var_3: var_35, var_4: var_38}
    var_40 = {var_26: var_32, var_2: var_39}
    var_41 = module_0.ParsedContent()
    var_42 = True
    var_43 = module_1.Config()
    var_44 = module_2.sorted_imports(var_41, var_43)
    var_45 = [var_0]
    var_46 = []
    var_47 = []
    var_48 = {var_5: var_46, var_6: var_47}
    var_49 = [var_12]
    var_50 = [var_14]
    var_51 = {var_10: var_49, var_11: var_50}
    var_52 = {var_3: var_48, var_4: var_51}
    var_53 = {var_2: var_52}
    var_54 = module_0.ParsedContent()
    var_55 = [var_5]
    var_56 = module_1.Config()
    var_57 = module_2.sorted_imports(var_54, var_56)
    var_58 = [var_0]
    var_59 = [var_28]
    var_60 = {var_27: var_59}
    var_61 = {}
    var_62 = {var_3: var_60, var_4: var_61}
    var_63 = []
    var_64 = []
    var_65 = {var_5: var_63, var_6: var_64}
    var_66 = [var_12]
    var_67 = [var_14]
    var_68 = {var_10: var_66, var_11: var_67}
    var_69 = {var_3: var_65, var_4: var_68}
    var_70 = {var_26: var_62, var_2: var_69}
    var_71 = module_0.ParsedContent()
    var_72 = 2
    var_73 = module_1.Config()
    var_74 = module_2.sorted_imports(var_71, var_73)
    var_75 = [var_0]
    var_76 = []
    var_77 = []
    var_78 = {var_5: var_76, var_6: var_77}
    var_79 = [var_12]
    var_80 = [var_14]
    var_81 = {var_10: var_79, var_11: var_80}
    var_82 = {var_3: var_78, var_4: var_81}
    var_83 = {var_2: var_82}
    var_84 = module_0.ParsedContent()
    var_85 = 'thirdparty'
    var_86 = 'Third Party Imports'
    var_87 = {var_85: var_86}
    var_88 = module_1.Config()
    var_89 = module_2.sorted_imports(var_84, var_88)
    var_90 = "print('hello')"
    var_91 = [var_90]
    var_92 = {}
    var_93 = -1
    var_94 = module_0.ParsedContent()
    var_95 = module_1.Config()
    var_96 = module_2.sorted_imports(var_94, var_95)
    assert var_96 == "print('hello')"



# Parsed testcases at query #52
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = 0
    var_5 = '\n'
    var_6 = {}
    var_7 = {}
    var_8 = module_0.ParsedContent()
    var_9 = "print('hello')"
    var_10 = [var_9]
    var_11 = {}
    var_12 = -1
    var_13 = 1
    var_14 = {}
    var_15 = {}
    var_16 = module_0.ParsedContent()
    var_17 = [var_0]
    var_18 = 'THIRDPARTY'
    var_19 = 'straight'
    var_20 = 'from'
    var_21 = 'zlib'
    var_22 = 'os'
    var_23 = []
    var_24 = []
    var_25 = {var_21: var_23, var_22: var_24}
    var_26 = 'sys'
    var_27 = 'json'
    var_28 = 'path'
    var_29 = [var_28]
    var_30 = 'load'
    var_31 = [var_30]
    var_32 = {var_26: var_29, var_27: var_31}
    var_33 = {var_19: var_25, var_20: var_32}
    var_34 = {var_18: var_33}
    var_35 = {}
    var_36 = {}
    var_37 = module_0.ParsedContent()
    var_38 = 'import os\nimport zlib\n\nfrom json import load\nfrom sys import path\n\n'
    var_39 = [var_0]
    var_40 = []
    var_41 = []
    var_42 = {var_21: var_40, var_22: var_41}
    var_43 = [var_28]
    var_44 = [var_30]
    var_45 = {var_26: var_43, var_27: var_44}
    var_46 = {var_19: var_42, var_20: var_45}
    var_47 = {var_18: var_46}
    var_48 = {}
    var_49 = {}
    var_50 = module_0.ParsedContent()
    var_51 = 2
    var_52 = True
    var_53 = module_1.Config()
    var_54 = module_2.sorted_imports(var_50, var_53)
    var_55 = 'from json import load\nfrom sys import path\n\nimport os\nimport zlib\n\n\n'
    var_56 = [var_0]
    var_57 = 'FIRSTPARTY'
    var_58 = []
    var_59 = {var_21: var_58}
    var_60 = [var_28]
    var_61 = {var_26: var_60}
    var_62 = {var_19: var_59, var_20: var_61}
    var_63 = []
    var_64 = {var_22: var_63}
    var_65 = [var_30]
    var_66 = {var_27: var_65}
    var_67 = {var_19: var_64, var_20: var_66}
    var_68 = {var_18: var_62, var_57: var_67}
    var_69 = {}
    var_70 = {}
    var_71 = module_0.ParsedContent()
    var_72 = [var_57]
    var_73 = module_1.Config()
    var_74 = module_2.sorted_imports(var_71, var_73)
    var_75 = 'import zlib\n\nfrom sys import path\n\n\nimport os\n\nfrom json import load\n\n'
    var_76 = [var_0]
    var_77 = {}
    var_78 = '*'
    var_79 = [var_78]
    var_80 = [var_30]
    var_81 = {var_26: var_79, var_27: var_80}
    var_82 = {var_19: var_77, var_20: var_81}
    var_83 = {var_18: var_82}
    var_84 = {}
    var_85 = {}
    var_86 = module_0.ParsedContent()
    var_87 = True
    var_88 = module_1.Config()
    var_89 = module_2.sorted_imports(var_86, var_88)
    var_90 = 'from sys import *\nfrom json import load\n\n'
    var_91 = [var_0]
    var_92 = []
    var_93 = {var_21: var_92}
    var_94 = [var_28]
    var_95 = {var_26: var_94}
    var_96 = {var_19: var_93, var_20: var_95}
    var_97 = {var_18: var_96}
    var_98 = {}
    var_99 = {}
    var_100 = module_0.ParsedContent()
    var_101 = 'thirdparty'
    var_102 = 'Third Party Imports'
    var_103 = {var_101: var_102}
    var_104 = module_1.Config()
    var_105 = module_2.sorted_imports(var_100, var_104)
    var_106 = '# Third Party Imports\nimport zlib\n\nfrom sys import path\n\n'
    var_107 = [var_0]
    var_108 = []
    var_109 = {var_21: var_108}
    var_110 = [var_28]
    var_111 = {var_26: var_110}
    var_112 = {var_19: var_109, var_20: var_111}
    var_113 = []
    var_114 = {var_22: var_113}
    var_115 = [var_30]
    var_116 = {var_27: var_115}
    var_117 = {var_19: var_114, var_20: var_116}
    var_118 = {var_18: var_112, var_57: var_117}
    var_119 = {}
    var_120 = {}
    var_121 = module_0.ParsedContent()
    var_122 = True
    var_123 = module_1.Config()
    var_124 = module_2.sorted_imports(var_121, var_123)
    var_125 = 'import os\nimport zlib\n\nfrom json import load\nfrom sys import path\n\n'



# Parsed testcases at query #53
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = '\n'
    var_5 = 1
    var_6 = {}
    var_7 = {}
    var_8 = module_0.ParsedContent()
    var_9 = module_1.Config()
    var_10 = module_2.sorted_imports(var_8, var_9)
    assert var_10 == "print('hello')\n"
    var_11 = [var_0]
    var_12 = 'THIRDPARTY'
    var_13 = 'straight'
    var_14 = 'from'
    var_15 = 'os'
    var_16 = 'sys'
    var_17 = set()
    var_18 = set()
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = 'os.path'
    var_21 = 'join'
    var_22 = set()
    var_23 = {var_21: var_22}
    var_24 = {var_20: var_23}
    var_25 = {var_13: var_19, var_14: var_24}
    var_26 = {var_12: var_25}
    var_27 = 0
    var_28 = 2
    var_29 = {}
    var_30 = {}
    var_31 = module_0.ParsedContent()
    var_32 = module_1.Config()
    var_33 = module_2.sorted_imports(var_31, var_32)
    var_34 = "import os\nimport sys\nfrom os.path import join\n\nprint('hello')\n"
    var_35 = '# Main code'
    var_36 = [var_35, var_0]
    var_37 = 'FUTURE'
    var_38 = '__future__'
    var_39 = 'print_function'
    var_40 = {var_39}
    var_41 = {var_38: var_40}
    var_42 = {}
    var_43 = {var_13: var_41, var_14: var_42}
    var_44 = set()
    var_45 = set()
    var_46 = {var_15: var_44, var_16: var_45}
    var_47 = set()
    var_48 = {var_21: var_47}
    var_49 = {var_20: var_48}
    var_50 = {var_13: var_46, var_14: var_49}
    var_51 = {var_37: var_43, var_12: var_50}
    var_52 = 3
    var_53 = {}
    var_54 = {}
    var_55 = module_0.ParsedContent()
    var_56 = 'future'
    var_57 = 'Future imports'
    var_58 = {var_56: var_57}
    var_59 = module_1.Config()
    var_60 = module_2.sorted_imports(var_55, var_59)
    var_61 = "# Future imports\nfrom __future__ import print_function\n\nimport os\nimport sys\nfrom os.path import join\n\n# Main code\nprint('hello')\n"
    var_62 = [var_0]
    var_63 = {var_39}
    var_64 = {var_38: var_63}
    var_65 = {}
    var_66 = {var_13: var_64, var_14: var_65}
    var_67 = set()
    var_68 = set()
    var_69 = {var_15: var_67, var_16: var_68}
    var_70 = set()
    var_71 = {var_21: var_70}
    var_72 = {var_20: var_71}
    var_73 = {var_13: var_69, var_14: var_72}
    var_74 = {var_37: var_66, var_12: var_73}
    var_75 = {}
    var_76 = {}
    var_77 = module_0.ParsedContent()
    var_78 = True
    var_79 = [var_37]
    var_80 = module_1.Config()
    var_81 = module_2.sorted_imports(var_77, var_80)
    var_82 = "from __future__ import print_function\n\nimport os\nimport sys\nfrom os.path import join\n\nprint('hello')\n"
    var_83 = [var_0]
    var_84 = set()
    var_85 = {var_15: var_84}
    var_86 = 'module'
    var_87 = set()
    var_88 = {var_21: var_87}
    var_89 = '*'
    var_90 = {var_89}
    var_91 = {var_20: var_88, var_86: var_90}
    var_92 = {var_13: var_85, var_14: var_91}
    var_93 = {var_12: var_92}
    var_94 = {}
    var_95 = {}
    var_96 = module_0.ParsedContent()
    var_97 = True
    var_98 = True
    var_99 = module_1.Config()
    var_100 = module_2.sorted_imports(var_96, var_99)
    var_101 = "from module import *\nfrom os.path import join\n\nimport os\n\nprint('hello')\n"



# Parsed testcases at query #54
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = '# test'
    var_1 = 'x = 1'
    var_2 = [var_0, var_1]
    var_3 = 'THIRDPARTY'
    var_4 = 'straight'
    var_5 = 'from'
    var_6 = 'os'
    var_7 = 'sys'
    var_8 = []
    var_9 = []
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'collections'
    var_12 = 'itertools'
    var_13 = 'defaultdict'
    var_14 = [var_13]
    var_15 = 'chain'
    var_16 = [var_15]
    var_17 = {var_11: var_14, var_12: var_16}
    var_18 = {var_4: var_10, var_5: var_17}
    var_19 = {var_3: var_18}
    var_20 = 1
    var_21 = '\n'
    var_22 = 2
    var_23 = module_0.ParsedContent()
    var_24 = module_1.Config()
    var_25 = module_2.sorted_imports(var_23, var_24)
    assert var_25 == '# test\n\nimport os\nimport sys\n\nfrom collections import defaultdict\nfrom itertools import chain\n\nx = 1'
    var_26 = [var_1]
    var_27 = {}
    var_28 = -1
    var_29 = module_0.ParsedContent()
    var_30 = module_2.sorted_imports(var_29, var_24)
    assert var_30 == 'x = 1'
    var_31 = True
    var_32 = True
    var_33 = module_1.Config()
    var_34 = [var_0, var_1]
    var_35 = []
    var_36 = []
    var_37 = {var_6: var_35, var_7: var_36}
    var_38 = [var_13]
    var_39 = [var_15]
    var_40 = {var_11: var_38, var_12: var_39}
    var_41 = {var_4: var_37, var_5: var_40}
    var_42 = {var_3: var_41}
    var_43 = module_0.ParsedContent()
    var_44 = module_2.sorted_imports(var_43, var_33)
    assert var_44 == '# test\n\nfrom itertools import chain\nfrom collections import defaultdict\n\nimport sys\nimport os\n\nx = 1'
    var_45 = [var_0, var_1]
    var_46 = 'FUTURE'
    var_47 = 'STDLIB'
    var_48 = '__future__'
    var_49 = 'annotations'
    var_50 = [var_49]
    var_51 = {var_48: var_50}
    var_52 = {}
    var_53 = {var_4: var_51, var_5: var_52}
    var_54 = []
    var_55 = {var_6: var_54}
    var_56 = [var_13]
    var_57 = {var_11: var_56}
    var_58 = {var_4: var_55, var_5: var_57}
    var_59 = 'numpy'
    var_60 = []
    var_61 = {var_59: var_60}
    var_62 = {}
    var_63 = {var_4: var_61, var_5: var_62}
    var_64 = {var_46: var_53, var_47: var_58, var_3: var_63}
    var_65 = module_0.ParsedContent()
    var_66 = module_2.sorted_imports(var_65, var_24)
    assert var_66 == '# test\n\nfrom __future__ import annotations\n\nimport os\n\nfrom collections import defaultdict\n\nimport numpy\n\nx = 1'



