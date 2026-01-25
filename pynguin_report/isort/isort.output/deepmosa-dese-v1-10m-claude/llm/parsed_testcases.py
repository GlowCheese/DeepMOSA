####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import isort.output as module_0

def test_case_0():
    var_0 = 'line1'
    var_1 = 'line2'
    var_2 = 'line3'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._ensure_newline_before_comment(var_3)

import isort.output as module_0

def test_case_0():
    var_0 = 'line1'
    var_1 = '#comment'
    var_2 = 'line2'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._ensure_newline_before_comment(var_3)

import isort.output as module_0

def test_case_0():
    var_0 = '#comment'
    var_1 = 'line1'
    var_2 = [var_0, var_1]
    var_3 = module_0._ensure_newline_before_comment(var_2)

import isort.output as module_0

def test_case_0():
    var_0 = 'line1'
    var_1 = '#comment1'
    var_2 = '#comment2'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._ensure_newline_before_comment(var_3)

import isort.output as module_0

def test_case_0():
    var_0 = 'line1'
    var_1 = ''
    var_2 = '#comment'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._ensure_newline_before_comment(var_3)

import isort.output as module_0

def test_case_0():
    var_0 = '#comment1'
    var_1 = '#comment2'
    var_2 = '#comment3'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._ensure_newline_before_comment(var_3)

import isort.output as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._ensure_newline_before_comment(var_0)

import isort.output as module_0

def test_case_0():
    var_0 = 'line1'
    var_1 = [var_0]
    var_2 = module_0._ensure_newline_before_comment(var_1)

import isort.output as module_0

def test_case_0():
    var_0 = '#comment'
    var_1 = [var_0]
    var_2 = module_0._ensure_newline_before_comment(var_1)

import isort.output as module_0

def test_case_0():
    var_0 = 'line1'
    var_1 = '#comment1'
    var_2 = 'line2'
    var_3 = '#comment2'
    var_4 = 'line3'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0._ensure_newline_before_comment(var_5)

import isort.output as module_0

def test_case_0():
    var_0 = ''
    var_1 = '#comment'
    var_2 = [var_0, var_1, var_0]
    var_3 = module_0._ensure_newline_before_comment(var_2)

import isort.output as module_0

def test_case_0():
    var_0 = 'code1'
    var_1 = '#comment1'
    var_2 = ''
    var_3 = 'code2'
    var_4 = '#comment2'
    var_5 = '#comment3'
    var_6 = 'code3'
    var_7 = [var_0, var_1, var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0._ensure_newline_before_comment(var_7)



# Parsed testcases at query #2
#--------------------------




import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = 'straight'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 'STDLIB'
    var_5 = {}
    var_6 = {var_1: var_5}
    var_7 = {var_4: var_6}
    var_8 = 'above'
    var_9 = {}
    var_10 = {var_1: var_9}
    var_11 = {}
    var_12 = {var_8: var_10, var_1: var_11}
    var_13 = 0
    var_14 = {}
    var_15 = {}
    var_16 = {}
    var_17 = module_0.ParsedContent()
    var_18 = True
    var_19 = module_1.Config()
    var_20 = 'os'
    var_21 = 'sys'
    var_22 = [var_20, var_21]
    var_23 = 'STDLIB'
    var_24 = []
    var_25 = 'import'
    var_26 = module_2._with_straight_imports(var_17, var_19, var_22, var_23, var_24, var_25)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = 'straight'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 'STDLIB'
    var_5 = {}
    var_6 = {var_1: var_5}
    var_7 = {var_4: var_6}
    var_8 = 'above'
    var_9 = {}
    var_10 = {var_1: var_9}
    var_11 = 'os'
    var_12 = 'sys'
    var_13 = 'comment1'
    var_14 = [var_13]
    var_15 = 'comment2'
    var_16 = [var_15]
    var_17 = {var_11: var_14, var_12: var_16}
    var_18 = {var_8: var_10, var_1: var_17}
    var_19 = 0
    var_20 = {}
    var_21 = {}
    var_22 = {}
    var_23 = module_0.ParsedContent()
    var_24 = True
    var_25 = module_1.Config()
    var_26 = [var_11, var_12]
    var_27 = 'STDLIB'
    var_28 = []
    var_29 = 'import'
    var_30 = module_2._with_straight_imports(var_23, var_25, var_26, var_27, var_28, var_29)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = 'straight'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 'STDLIB'
    var_5 = {}
    var_6 = {var_1: var_5}
    var_7 = {var_4: var_6}
    var_8 = 'above'
    var_9 = 'os'
    var_10 = '# above comment'
    var_11 = [var_10]
    var_12 = {var_9: var_11}
    var_13 = {var_1: var_12}
    var_14 = {}
    var_15 = {var_8: var_13, var_1: var_14}
    var_16 = 0
    var_17 = {}
    var_18 = {}
    var_19 = {}
    var_20 = module_0.ParsedContent()
    var_21 = True
    var_22 = module_1.Config()
    var_23 = [var_9]
    var_24 = 'STDLIB'
    var_25 = []
    var_26 = 'import'
    var_27 = module_2._with_straight_imports(var_20, var_22, var_23, var_24, var_25, var_26)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = 'straight'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 'STDLIB'
    var_5 = {}
    var_6 = {var_1: var_5}
    var_7 = {var_4: var_6}
    var_8 = 'above'
    var_9 = {}
    var_10 = {var_1: var_9}
    var_11 = {}
    var_12 = {var_8: var_10, var_1: var_11}
    var_13 = 0
    var_14 = {}
    var_15 = {}
    var_16 = {}
    var_17 = module_0.ParsedContent()
    var_18 = True
    var_19 = module_1.Config()
    var_20 = []
    var_21 = 'STDLIB'
    var_22 = []
    var_23 = 'import'
    var_24 = module_2._with_straight_imports(var_17, var_19, var_20, var_21, var_22, var_23)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = 'straight'
    var_2 = 'os'
    var_3 = 'operating_system'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = 'STDLIB'
    var_8 = False
    var_9 = {var_2: var_8}
    var_10 = {var_1: var_9}
    var_11 = {var_7: var_10}
    var_12 = 'above'
    var_13 = {}
    var_14 = {var_1: var_13}
    var_15 = {}
    var_16 = {var_12: var_14, var_1: var_15}
    var_17 = {}
    var_18 = {}
    var_19 = {}
    var_20 = module_0.ParsedContent()
    var_21 = True
    var_22 = module_1.Config()
    var_23 = [var_2]
    var_24 = 'STDLIB'
    var_25 = []
    var_26 = 'import'
    var_27 = module_2._with_straight_imports(var_20, var_22, var_23, var_24, var_25, var_26)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = 'straight'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 'STDLIB'
    var_5 = 'os'
    var_6 = 'sys'
    var_7 = False
    var_8 = {var_5: var_7, var_6: var_7}
    var_9 = {var_1: var_8}
    var_10 = {var_4: var_9}
    var_11 = 'above'
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = {}
    var_15 = {var_11: var_13, var_1: var_14}
    var_16 = {}
    var_17 = {}
    var_18 = {}
    var_19 = module_0.ParsedContent()
    var_20 = module_1.Config()
    var_21 = [var_5, var_6]
    var_22 = 'STDLIB'
    var_23 = []
    var_24 = 'import'
    var_25 = module_2._with_straight_imports(var_19, var_20, var_21, var_22, var_23, var_24)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = 'straight'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 'STDLIB'
    var_5 = 'os'
    var_6 = 'sys'
    var_7 = False
    var_8 = {var_5: var_7, var_6: var_7}
    var_9 = {var_1: var_8}
    var_10 = {var_4: var_9}
    var_11 = 'above'
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = {}
    var_15 = {var_11: var_13, var_1: var_14}
    var_16 = {}
    var_17 = {}
    var_18 = {}
    var_19 = module_0.ParsedContent()
    var_20 = module_1.Config()
    var_21 = [var_5, var_6]
    var_22 = 'STDLIB'
    var_23 = [var_5]
    var_24 = 'import'
    var_25 = module_2._with_straight_imports(var_19, var_20, var_21, var_22, var_23, var_24)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = 'straight'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 'STDLIB'
    var_5 = 'os'
    var_6 = False
    var_7 = {var_5: var_6}
    var_8 = {var_1: var_7}
    var_9 = {var_4: var_8}
    var_10 = 'above'
    var_11 = {}
    var_12 = {var_1: var_11}
    var_13 = 'comment'
    var_14 = [var_13]
    var_15 = {var_5: var_14}
    var_16 = {var_10: var_12, var_1: var_15}
    var_17 = {}
    var_18 = {}
    var_19 = {}
    var_20 = module_0.ParsedContent()
    var_21 = True
    var_22 = module_1.Config()
    var_23 = [var_5]
    var_24 = 'STDLIB'
    var_25 = []
    var_26 = 'import'
    var_27 = module_2._with_straight_imports(var_20, var_22, var_23, var_24, var_25, var_26)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_with_from_imports_basic. Retrieved 30/34 statements.
# Partially parsed test_with_from_imports_with_star_import. Retrieved 29/33 statements.
# Partially parsed test_with_from_imports_force_single_line. Retrieved 31/35 statements.


import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'module1'
    var_4 = 'import1'
    var_5 = 'import2'
    var_6 = False
    var_7 = {var_4: var_6, var_5: var_6}
    var_8 = {var_3: var_7}
    var_9 = {var_2: var_8}
    var_10 = {var_1: var_9}
    var_11 = {}
    var_12 = {var_2: var_11}
    var_13 = 'above'
    var_14 = 'nested'
    var_15 = 'straight'
    var_16 = {}
    var_17 = {}
    var_18 = {var_2: var_17}
    var_19 = {}
    var_20 = {}
    var_21 = {var_2: var_16, var_13: var_18, var_14: var_19, var_15: var_20}
    var_22 = '\n'
    var_23 = set()
    var_24 = module_1.ParsedContent()
    var_25 = [var_3]
    var_26 = []
    var_27 = 'import'
    var_28 = module_2._with_from_imports(var_24, var_0, var_25, var_1, var_26, var_27)
    var_29 = len(var_28)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = {var_2: var_6}
    var_8 = 'above'
    var_9 = 'nested'
    var_10 = 'straight'
    var_11 = {}
    var_12 = {}
    var_13 = {var_2: var_12}
    var_14 = {}
    var_15 = {}
    var_16 = {var_2: var_11, var_8: var_13, var_9: var_14, var_10: var_15}
    var_17 = '\n'
    var_18 = set()
    var_19 = module_1.ParsedContent()
    var_20 = []
    var_21 = []
    var_22 = 'import'
    var_23 = module_2._with_from_imports(var_19, var_0, var_20, var_1, var_21, var_22)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'module1'
    var_4 = 'import1'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = {var_1: var_8}
    var_10 = {}
    var_11 = {var_2: var_10}
    var_12 = 'above'
    var_13 = 'nested'
    var_14 = 'straight'
    var_15 = {}
    var_16 = {}
    var_17 = {var_2: var_16}
    var_18 = {}
    var_19 = {}
    var_20 = {var_2: var_15, var_12: var_17, var_13: var_18, var_14: var_19}
    var_21 = '\n'
    var_22 = set()
    var_23 = module_1.ParsedContent()
    var_24 = [var_3]
    var_25 = [var_3]
    var_26 = 'import'
    var_27 = module_2._with_from_imports(var_23, var_0, var_24, var_1, var_25, var_26)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = 'THIRDPARTY'
    var_3 = 'from'
    var_4 = 'module1'
    var_5 = '*'
    var_6 = False
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = {var_3: var_8}
    var_10 = {var_2: var_9}
    var_11 = {}
    var_12 = {var_3: var_11}
    var_13 = 'above'
    var_14 = 'nested'
    var_15 = 'straight'
    var_16 = {}
    var_17 = {}
    var_18 = {var_3: var_17}
    var_19 = {}
    var_20 = {}
    var_21 = {var_3: var_16, var_13: var_18, var_14: var_19, var_15: var_20}
    var_22 = '\n'
    var_23 = set()
    var_24 = module_1.ParsedContent()
    var_25 = [var_4]
    var_26 = []
    var_27 = 'import'
    var_28 = module_2._with_from_imports(var_24, var_1, var_25, var_2, var_26, var_27)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = 'THIRDPARTY'
    var_3 = 'from'
    var_4 = 'module1'
    var_5 = 'import1'
    var_6 = 'import2'
    var_7 = False
    var_8 = {var_5: var_7, var_6: var_7}
    var_9 = {var_4: var_8}
    var_10 = {var_3: var_9}
    var_11 = {var_2: var_10}
    var_12 = {}
    var_13 = {var_3: var_12}
    var_14 = 'above'
    var_15 = 'nested'
    var_16 = 'straight'
    var_17 = {}
    var_18 = {}
    var_19 = {var_3: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = {var_3: var_17, var_14: var_19, var_15: var_20, var_16: var_21}
    var_23 = '\n'
    var_24 = set()
    var_25 = module_1.ParsedContent()
    var_26 = [var_4]
    var_27 = []
    var_28 = 'import'
    var_29 = module_2._with_from_imports(var_25, var_1, var_26, var_2, var_27, var_28)
    var_30 = len(var_29)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_sorted_imports_empty. Retrieved 41/44 statements.
# Partially parsed test_sorted_imports_with_comments. Retrieved 47/50 statements.


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = "print('hello')\n"
    var_2 = [var_1]
    var_3 = -1
    var_4 = {}
    var_5 = {}
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = {}
    var_9 = {}
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'FUTURE'
    var_12 = 'STDLIB'
    var_13 = 'THIRDPARTY'
    var_14 = 'FIRSTPARTY'
    var_15 = 'LOCALFOLDER'
    var_16 = {}
    var_17 = {}
    var_18 = {var_6: var_16, var_7: var_17}
    var_19 = {}
    var_20 = {}
    var_21 = {var_6: var_19, var_7: var_20}
    var_22 = {}
    var_23 = {}
    var_24 = {var_6: var_22, var_7: var_23}
    var_25 = {}
    var_26 = {}
    var_27 = {var_6: var_25, var_7: var_26}
    var_28 = {}
    var_29 = {}
    var_30 = {var_6: var_28, var_7: var_29}
    var_31 = {var_11: var_18, var_12: var_21, var_13: var_24, var_14: var_27, var_15: var_30}
    var_32 = 'above'
    var_33 = {}
    var_34 = {var_6: var_33}
    var_35 = {}
    var_36 = {var_32: var_34, var_6: var_35}
    var_37 = [var_11, var_12, var_13, var_14, var_15]
    var_38 = 1
    var_39 = '\n'
    var_40 = module_0.ParsedContent()
    var_41 = module_1.Config()
    var_42 = module_2.sorted_imports(var_40, var_41)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 0
    var_3 = {}
    var_4 = {}
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = {}
    var_8 = {}
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'FUTURE'
    var_11 = 'STDLIB'
    var_12 = 'THIRDPARTY'
    var_13 = 'FIRSTPARTY'
    var_14 = 'LOCALFOLDER'
    var_15 = {}
    var_16 = {}
    var_17 = {var_5: var_15, var_6: var_16}
    var_18 = 'os'
    var_19 = None
    var_20 = {var_18: var_19}
    var_21 = {}
    var_22 = {var_5: var_20, var_6: var_21}
    var_23 = {}
    var_24 = {}
    var_25 = {var_5: var_23, var_6: var_24}
    var_26 = {}
    var_27 = {}
    var_28 = {var_5: var_26, var_6: var_27}
    var_29 = {}
    var_30 = {}
    var_31 = {var_5: var_29, var_6: var_30}
    var_32 = {var_10: var_17, var_11: var_22, var_12: var_25, var_13: var_28, var_14: var_31}
    var_33 = 'above'
    var_34 = {}
    var_35 = {var_5: var_34}
    var_36 = {}
    var_37 = {var_33: var_35, var_5: var_36}
    var_38 = [var_10, var_11, var_12, var_13, var_14]
    var_39 = '\n'
    var_40 = module_0.ParsedContent()
    var_41 = module_1.Config()
    var_42 = module_2.sorted_imports(var_40, var_41)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 0
    var_3 = {}
    var_4 = {}
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = {}
    var_8 = {}
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'FUTURE'
    var_11 = 'STDLIB'
    var_12 = 'THIRDPARTY'
    var_13 = 'FIRSTPARTY'
    var_14 = 'LOCALFOLDER'
    var_15 = {}
    var_16 = {}
    var_17 = {var_5: var_15, var_6: var_16}
    var_18 = {}
    var_19 = 'os'
    var_20 = 'path'
    var_21 = [var_20]
    var_22 = {var_19: var_21}
    var_23 = {var_5: var_18, var_6: var_22}
    var_24 = {}
    var_25 = {}
    var_26 = {var_5: var_24, var_6: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = {var_5: var_27, var_6: var_28}
    var_30 = {}
    var_31 = {}
    var_32 = {var_5: var_30, var_6: var_31}
    var_33 = {var_10: var_17, var_11: var_23, var_12: var_26, var_13: var_29, var_14: var_32}
    var_34 = 'above'
    var_35 = {}
    var_36 = {var_5: var_35}
    var_37 = {}
    var_38 = {var_34: var_36, var_5: var_37}
    var_39 = [var_10, var_11, var_12, var_13, var_14]
    var_40 = '\n'
    var_41 = module_0.ParsedContent()
    var_42 = module_1.Config()
    var_43 = module_2.sorted_imports(var_41, var_42)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 0
    var_3 = {}
    var_4 = {}
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = {}
    var_8 = {}
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'FUTURE'
    var_11 = 'STDLIB'
    var_12 = 'THIRDPARTY'
    var_13 = 'FIRSTPARTY'
    var_14 = 'LOCALFOLDER'
    var_15 = {}
    var_16 = {}
    var_17 = {var_5: var_15, var_6: var_16}
    var_18 = {}
    var_19 = {}
    var_20 = {var_5: var_18, var_6: var_19}
    var_21 = {}
    var_22 = {}
    var_23 = {var_5: var_21, var_6: var_22}
    var_24 = {}
    var_25 = {}
    var_26 = {var_5: var_24, var_6: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = {var_5: var_27, var_6: var_28}
    var_30 = {var_10: var_17, var_11: var_20, var_12: var_23, var_13: var_26, var_14: var_29}
    var_31 = 'above'
    var_32 = {}
    var_33 = {var_5: var_32}
    var_34 = {}
    var_35 = {var_31: var_33, var_5: var_34}
    var_36 = [var_10, var_11, var_12, var_13, var_14]
    var_37 = '\n'
    var_38 = module_0.ParsedContent()
    var_39 = module_1.Config()
    var_40 = module_2.sorted_imports(var_38, var_39)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 0
    var_3 = {}
    var_4 = {}
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = {}
    var_8 = {}
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'FUTURE'
    var_11 = 'STDLIB'
    var_12 = 'THIRDPARTY'
    var_13 = 'FIRSTPARTY'
    var_14 = 'LOCALFOLDER'
    var_15 = {}
    var_16 = {}
    var_17 = {var_5: var_15, var_6: var_16}
    var_18 = 'os'
    var_19 = None
    var_20 = {var_18: var_19}
    var_21 = {}
    var_22 = {var_5: var_20, var_6: var_21}
    var_23 = {}
    var_24 = {}
    var_25 = {var_5: var_23, var_6: var_24}
    var_26 = {}
    var_27 = {}
    var_28 = {var_5: var_26, var_6: var_27}
    var_29 = {}
    var_30 = {}
    var_31 = {var_5: var_29, var_6: var_30}
    var_32 = {var_10: var_17, var_11: var_22, var_12: var_25, var_13: var_28, var_14: var_31}
    var_33 = 'above'
    var_34 = '# comment'
    var_35 = [var_34]
    var_36 = {var_18: var_35}
    var_37 = {var_5: var_36}
    var_38 = '# inline'
    var_39 = [var_38]
    var_40 = {var_18: var_39}
    var_41 = {var_33: var_37, var_5: var_40}
    var_42 = [var_10, var_11, var_12, var_13, var_14]
    var_43 = '\n'
    var_44 = module_0.ParsedContent()
    var_45 = module_1.Config()
    var_46 = module_2.sorted_imports(var_44, var_45)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 0
    var_3 = {}
    var_4 = {}
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = {}
    var_8 = {}
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'FUTURE'
    var_11 = 'STDLIB'
    var_12 = 'THIRDPARTY'
    var_13 = 'FIRSTPARTY'
    var_14 = 'LOCALFOLDER'
    var_15 = {}
    var_16 = {}
    var_17 = {var_5: var_15, var_6: var_16}
    var_18 = 'os'
    var_19 = 'sys'
    var_20 = None
    var_21 = {var_18: var_20, var_19: var_20}
    var_22 = {}
    var_23 = {var_5: var_21, var_6: var_22}
    var_24 = {}
    var_25 = {}
    var_26 = {var_5: var_24, var_6: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = {var_5: var_27, var_6: var_28}
    var_30 = {}
    var_31 = {}
    var_32 = {var_5: var_30, var_6: var_31}
    var_33 = {var_10: var_17, var_11: var_23, var_12: var_26, var_13: var_29, var_14: var_32}
    var_34 = 'above'
    var_35 = {}
    var_36 = {var_5: var_35}
    var_37 = {}
    var_38 = {var_34: var_36, var_5: var_37}
    var_39 = [var_10, var_11, var_12, var_13, var_14]
    var_40 = '\n'
    var_41 = module_0.ParsedContent()
    var_42 = module_1.Config()
    var_43 = module_2.sorted_imports(var_41, var_42)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_with_star_comments_with_star_comment. Retrieved 9/13 statements.
# Partially parsed test_with_star_comments_without_star_comment. Retrieved 7/11 statements.
# Partially parsed test_with_star_comments_module_not_found. Retrieved 5/9 statements.
# Partially parsed test_with_star_comments_empty_comments_list. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 'nested'
    var_1 = 'test_module'
    var_2 = '*'
    var_3 = 'star comment text'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'comment1'
    var_7 = 'comment2'
    var_8 = [var_6, var_7]

def test_case_0():
    var_0 = 'nested'
    var_1 = 'test_module'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 'comment1'
    var_5 = 'comment2'
    var_6 = [var_4, var_5]

def test_case_0():
    var_0 = 'nested'
    var_1 = {}
    var_2 = 'comment1'
    var_3 = [var_2]
    var_4 = 'nonexistent_module'

def test_case_0():
    var_0 = 'nested'
    var_1 = 'test_module'
    var_2 = '*'
    var_3 = 'star comment'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = []



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_sorted_imports_normalize_empty_lines. Retrieved 41/45 statements.


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = -1
    var_1 = "print('hello')\n"
    var_2 = [var_1]
    var_3 = [var_1]
    var_4 = {}
    var_5 = {}
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = {}
    var_9 = {}
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = {}
    var_12 = 'above'
    var_13 = {}
    var_14 = {var_6: var_13}
    var_15 = {}
    var_16 = {var_12: var_14, var_6: var_15}
    var_17 = []
    var_18 = '\n'
    var_19 = module_0.ParsedContent()
    var_20 = module_1.Config()
    var_21 = module_2.sorted_imports(var_19, var_20)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = "print('hello')\n"
    var_2 = [var_1]
    var_3 = 'import os\n'
    var_4 = [var_3, var_1]
    var_5 = {}
    var_6 = {}
    var_7 = 'straight'
    var_8 = 'from'
    var_9 = {}
    var_10 = {}
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = 'STDLIB'
    var_13 = 'THIRDPARTY'
    var_14 = 'FIRSTPARTY'
    var_15 = 'LOCALFOLDER'
    var_16 = 'os'
    var_17 = None
    var_18 = {var_16: var_17}
    var_19 = {}
    var_20 = {var_7: var_18, var_8: var_19}
    var_21 = {}
    var_22 = {}
    var_23 = {var_7: var_21, var_8: var_22}
    var_24 = {}
    var_25 = {}
    var_26 = {var_7: var_24, var_8: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = {var_7: var_27, var_8: var_28}
    var_30 = {var_12: var_20, var_13: var_23, var_14: var_26, var_15: var_29}
    var_31 = 'above'
    var_32 = {}
    var_33 = {var_7: var_32}
    var_34 = {}
    var_35 = {var_31: var_33, var_7: var_34}
    var_36 = [var_12, var_13, var_14, var_15]
    var_37 = '\n'
    var_38 = module_0.ParsedContent()
    var_39 = module_1.Config()
    var_40 = module_2.sorted_imports(var_38, var_39)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'x = 1\n'
    var_2 = [var_1]
    var_3 = 'from os import path\n'
    var_4 = [var_3, var_1]
    var_5 = {}
    var_6 = {}
    var_7 = 'straight'
    var_8 = 'from'
    var_9 = {}
    var_10 = {}
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = 'STDLIB'
    var_13 = 'THIRDPARTY'
    var_14 = 'FIRSTPARTY'
    var_15 = 'LOCALFOLDER'
    var_16 = {}
    var_17 = 'os'
    var_18 = 'path'
    var_19 = [var_18]
    var_20 = {var_17: var_19}
    var_21 = {var_7: var_16, var_8: var_20}
    var_22 = {}
    var_23 = {}
    var_24 = {var_7: var_22, var_8: var_23}
    var_25 = {}
    var_26 = {}
    var_27 = {var_7: var_25, var_8: var_26}
    var_28 = {}
    var_29 = {}
    var_30 = {var_7: var_28, var_8: var_29}
    var_31 = {var_12: var_21, var_13: var_24, var_14: var_27, var_15: var_30}
    var_32 = 'above'
    var_33 = {}
    var_34 = {var_7: var_33}
    var_35 = {}
    var_36 = {var_32: var_34, var_7: var_35}
    var_37 = [var_12, var_13, var_14, var_15]
    var_38 = '\n'
    var_39 = module_0.ParsedContent()
    var_40 = module_1.Config()
    var_41 = module_2.sorted_imports(var_39, var_40)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'x = 1\n'
    var_2 = [var_1]
    var_3 = 'import os\n'
    var_4 = [var_3, var_1]
    var_5 = {}
    var_6 = {}
    var_7 = 'straight'
    var_8 = 'from'
    var_9 = {}
    var_10 = {}
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = 'STDLIB'
    var_13 = 'THIRDPARTY'
    var_14 = 'FIRSTPARTY'
    var_15 = 'LOCALFOLDER'
    var_16 = 'os'
    var_17 = None
    var_18 = {var_16: var_17}
    var_19 = {}
    var_20 = {var_7: var_18, var_8: var_19}
    var_21 = {}
    var_22 = {}
    var_23 = {var_7: var_21, var_8: var_22}
    var_24 = {}
    var_25 = {}
    var_26 = {var_7: var_24, var_8: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = {var_7: var_27, var_8: var_28}
    var_30 = {var_12: var_20, var_13: var_23, var_14: var_26, var_15: var_29}
    var_31 = 'above'
    var_32 = {}
    var_33 = {var_7: var_32}
    var_34 = {}
    var_35 = {var_31: var_33, var_7: var_34}
    var_36 = [var_12, var_13, var_14, var_15]
    var_37 = '\n'
    var_38 = module_0.ParsedContent()
    var_39 = module_1.Config()
    var_40 = module_2.sorted_imports(var_38, var_39)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'x = 1\n'
    var_2 = [var_1]
    var_3 = 'import os\n'
    var_4 = 'import custom\n'
    var_5 = [var_3, var_4, var_1]
    var_6 = {}
    var_7 = {}
    var_8 = 'straight'
    var_9 = 'from'
    var_10 = {}
    var_11 = {}
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = 'STDLIB'
    var_14 = 'THIRDPARTY'
    var_15 = 'FIRSTPARTY'
    var_16 = 'LOCALFOLDER'
    var_17 = 'os'
    var_18 = None
    var_19 = {var_17: var_18}
    var_20 = {}
    var_21 = {var_8: var_19, var_9: var_20}
    var_22 = 'custom'
    var_23 = {var_22: var_18}
    var_24 = {}
    var_25 = {var_8: var_23, var_9: var_24}
    var_26 = {}
    var_27 = {}
    var_28 = {var_8: var_26, var_9: var_27}
    var_29 = {}
    var_30 = {}
    var_31 = {var_8: var_29, var_9: var_30}
    var_32 = {var_13: var_21, var_14: var_25, var_15: var_28, var_16: var_31}
    var_33 = 'above'
    var_34 = {}
    var_35 = {var_8: var_34}
    var_36 = {}
    var_37 = {var_33: var_35, var_8: var_36}
    var_38 = [var_13, var_14, var_15, var_16]
    var_39 = '\n'
    var_40 = module_0.ParsedContent()
    var_41 = module_1.Config()
    var_42 = module_2.sorted_imports(var_40, var_41)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'x = 1\n'
    var_2 = [var_1]
    var_3 = 'import os\n'
    var_4 = [var_3, var_1]
    var_5 = {}
    var_6 = {}
    var_7 = 'straight'
    var_8 = 'from'
    var_9 = {}
    var_10 = {}
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = 'STDLIB'
    var_13 = 'THIRDPARTY'
    var_14 = 'FIRSTPARTY'
    var_15 = 'LOCALFOLDER'
    var_16 = 'os'
    var_17 = None
    var_18 = {var_16: var_17}
    var_19 = {}
    var_20 = {var_7: var_18, var_8: var_19}
    var_21 = {}
    var_22 = {}
    var_23 = {var_7: var_21, var_8: var_22}
    var_24 = {}
    var_25 = {}
    var_26 = {var_7: var_24, var_8: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = {var_7: var_27, var_8: var_28}
    var_30 = {var_12: var_20, var_13: var_23, var_14: var_26, var_15: var_29}
    var_31 = 'above'
    var_32 = {}
    var_33 = {var_7: var_32}
    var_34 = {}
    var_35 = {var_31: var_33, var_7: var_34}
    var_36 = [var_12, var_13, var_14, var_15]
    var_37 = '\n'
    var_38 = module_0.ParsedContent()
    var_39 = True
    var_40 = module_1.Config()
    var_41 = module_2.sorted_imports(var_38, var_40)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'x = 1\n'
    var_2 = [var_1]
    var_3 = 'import os\n'
    var_4 = [var_3, var_1]
    var_5 = {}
    var_6 = {}
    var_7 = 'straight'
    var_8 = 'from'
    var_9 = {}
    var_10 = {}
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = 'STDLIB'
    var_13 = 'THIRDPARTY'
    var_14 = 'FIRSTPARTY'
    var_15 = 'LOCALFOLDER'
    var_16 = 'os'
    var_17 = None
    var_18 = {var_16: var_17}
    var_19 = {}
    var_20 = {var_7: var_18, var_8: var_19}
    var_21 = {}
    var_22 = {}
    var_23 = {var_7: var_21, var_8: var_22}
    var_24 = {}
    var_25 = {}
    var_26 = {var_7: var_24, var_8: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = {var_7: var_27, var_8: var_28}
    var_30 = {var_12: var_20, var_13: var_23, var_14: var_26, var_15: var_29}
    var_31 = 'above'
    var_32 = {}
    var_33 = {var_7: var_32}
    var_34 = {}
    var_35 = {var_31: var_33, var_7: var_34}
    var_36 = [var_12, var_13, var_14, var_15]
    var_37 = '\n'
    var_38 = module_0.ParsedContent()
    var_39 = 'stdlib'
    var_40 = 'Standard Library'
    var_41 = {var_39: var_40}
    var_42 = module_1.Config()
    var_43 = module_2.sorted_imports(var_38, var_42)

def test_case_0():
    pass



# Parsed testcases at query #7
#--------------------------




import isort.output as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = 'world'
    var_2 = ''
    var_3 = [var_0, var_1, var_2, var_2]
    var_4 = module_0._normalize_empty_lines(var_3)

import isort.output as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = [var_0]
    var_2 = module_0._normalize_empty_lines(var_1)

import isort.output as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._normalize_empty_lines(var_0)

import isort.output as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0._normalize_empty_lines(var_1)

import isort.output as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = 'world'
    var_2 = '   '
    var_3 = '\t'
    var_4 = ''
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0._normalize_empty_lines(var_5)

import isort.output as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = 'world'
    var_2 = [var_0, var_1]
    var_3 = module_0._normalize_empty_lines(var_2)

import isort.output as module_0

def test_case_0():
    var_0 = 'line1'
    var_1 = ''
    var_2 = 'line2'
    var_3 = [var_0, var_1, var_2, var_1, var_1, var_1]
    var_4 = module_0._normalize_empty_lines(var_3)



# Parsed testcases at query #8
#--------------------------




import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'import os'
    var_1 = 'import sys'
    var_2 = [var_0, var_1]
    var_3 = -1
    var_4 = {}
    var_5 = {}
    var_6 = {}
    var_7 = {}
    var_8 = "print('hello')"
    var_9 = [var_8]
    var_10 = '\n'
    var_11 = 'FUTURE'
    var_12 = 'STDLIB'
    var_13 = 'THIRDPARTY'
    var_14 = 'FIRSTPARTY'
    var_15 = 'LOCALFOLDER'
    var_16 = [var_11, var_12, var_13, var_14, var_15]
    var_17 = {}
    var_18 = 1
    var_19 = module_0.ParsedContent()
    var_20 = module_1.Config()
    var_21 = module_2.sorted_imports(var_19, var_20)
    assert var_21 == "print('hello')"



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_sorted_imports_preserves_line_separator. Retrieved 24/28 statements.
# Partially parsed test_sorted_imports_with_multiple_sections. Retrieved 31/35 statements.


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = -1
    var_1 = "print('hello')\n"
    var_2 = [var_1]
    var_3 = -1
    var_4 = {}
    var_5 = {}
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = {}
    var_9 = {}
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = {}
    var_12 = 'above'
    var_13 = {}
    var_14 = {var_6: var_13}
    var_15 = {}
    var_16 = {var_12: var_14, var_6: var_15}
    var_17 = []
    var_18 = '\n'
    var_19 = module_0.ParsedContent()
    var_20 = module_1.Config()
    var_21 = module_2.sorted_imports(var_19, var_20)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = []
    var_2 = {}
    var_3 = {}
    var_4 = 'straight'
    var_5 = 'from'
    var_6 = {}
    var_7 = {}
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = 'STDLIB'
    var_10 = 'os'
    var_11 = 'sys'
    var_12 = None
    var_13 = {var_10: var_12, var_11: var_12}
    var_14 = {}
    var_15 = {var_4: var_13, var_5: var_14}
    var_16 = {var_9: var_15}
    var_17 = 'above'
    var_18 = {}
    var_19 = {var_4: var_18}
    var_20 = {}
    var_21 = {var_17: var_19, var_4: var_20}
    var_22 = [var_9]
    var_23 = '\n'
    var_24 = module_0.ParsedContent()
    var_25 = module_1.Config()
    var_26 = module_2.sorted_imports(var_24, var_25)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = []
    var_2 = {}
    var_3 = {}
    var_4 = 'straight'
    var_5 = 'from'
    var_6 = {}
    var_7 = {}
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = 'STDLIB'
    var_10 = {}
    var_11 = 'os'
    var_12 = 'path'
    var_13 = [var_12]
    var_14 = {var_11: var_13}
    var_15 = {var_4: var_10, var_5: var_14}
    var_16 = {var_9: var_15}
    var_17 = 'above'
    var_18 = {}
    var_19 = {var_4: var_18}
    var_20 = {}
    var_21 = {var_17: var_19, var_4: var_20}
    var_22 = [var_9]
    var_23 = '\n'
    var_24 = module_0.ParsedContent()
    var_25 = module_1.Config()
    var_26 = module_2.sorted_imports(var_24, var_25)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = -1
    var_1 = 'line1'
    var_2 = 'line2'
    var_3 = [var_1, var_2]
    var_4 = -1
    var_5 = {}
    var_6 = {}
    var_7 = 'straight'
    var_8 = 'from'
    var_9 = {}
    var_10 = {}
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_7: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_7: var_16}
    var_18 = []
    var_19 = '\r\n'
    var_20 = module_0.ParsedContent()
    var_21 = module_1.Config()
    var_22 = module_2.sorted_imports(var_20, var_21)
    var_23 = ''

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = []
    var_2 = {}
    var_3 = {}
    var_4 = 'straight'
    var_5 = 'from'
    var_6 = {}
    var_7 = {}
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = 'STDLIB'
    var_10 = 'THIRDPARTY'
    var_11 = 'os'
    var_12 = None
    var_13 = {var_11: var_12}
    var_14 = {}
    var_15 = {var_4: var_13, var_5: var_14}
    var_16 = 'requests'
    var_17 = {var_16: var_12}
    var_18 = {}
    var_19 = {var_4: var_17, var_5: var_18}
    var_20 = {var_9: var_15, var_10: var_19}
    var_21 = 'above'
    var_22 = {}
    var_23 = {var_4: var_22}
    var_24 = {}
    var_25 = {var_21: var_23, var_4: var_24}
    var_26 = [var_9, var_10]
    var_27 = '\n'
    var_28 = module_0.ParsedContent()
    var_29 = module_1.Config()
    var_30 = module_2.sorted_imports(var_28, var_29)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_with_from_imports_basic. Retrieved 10/12 statements.
# Partially parsed test_with_from_imports_remove_imports. Retrieved 10/12 statements.
# Partially parsed test_with_from_imports_with_comments. Retrieved 9/11 statements.
# Partially parsed test_with_from_imports_force_single_line. Retrieved 10/12 statements.
# Partially parsed test_with_from_imports_star_import. Retrieved 9/11 statements.
# Partially parsed test_with_from_imports_combine_as_imports. Retrieved 10/12 statements.
# Partially parsed test_with_from_imports_no_inline_sort. Retrieved 10/12 statements.
# Partially parsed test_with_from_imports_multiple_modules. Retrieved 10/12 statements.
# Partially parsed test_with_from_imports_line_length. Retrieved 10/12 statements.


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'from os import path'
    var_1 = module_0.file_contents(var_0)
    var_2 = module_1.Config()
    var_3 = 'os'
    var_4 = [var_3]
    var_5 = 'STDLIB'
    var_6 = []
    var_7 = 'import'
    var_8 = module_2._with_from_imports(var_1, var_2, var_4, var_5, var_6, var_7)
    var_9 = len(var_8)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = module_1.Config()
    var_3 = []
    var_4 = 'STDLIB'
    var_5 = []
    var_6 = 'import'
    var_7 = module_2._with_from_imports(var_1, var_2, var_3, var_4, var_5, var_6)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'from os import path, getcwd'
    var_1 = module_0.file_contents(var_0)
    var_2 = module_1.Config()
    var_3 = 'os'
    var_4 = [var_3]
    var_5 = 'STDLIB'
    var_6 = 'os.path'
    var_7 = [var_6]
    var_8 = 'import'
    var_9 = module_2._with_from_imports(var_1, var_2, var_4, var_5, var_7, var_8)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'from os import path  # noqa'
    var_1 = module_0.file_contents(var_0)
    var_2 = module_1.Config()
    var_3 = 'os'
    var_4 = [var_3]
    var_5 = 'STDLIB'
    var_6 = []
    var_7 = 'import'
    var_8 = module_2._with_from_imports(var_1, var_2, var_4, var_5, var_6, var_7)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'from os import path, getcwd'
    var_1 = module_0.file_contents(var_0)
    var_2 = True
    var_3 = module_1.Config()
    var_4 = 'os'
    var_5 = [var_4]
    var_6 = 'STDLIB'
    var_7 = []
    var_8 = 'import'
    var_9 = module_2._with_from_imports(var_1, var_3, var_5, var_6, var_7, var_8)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'from os import *'
    var_1 = module_0.file_contents(var_0)
    var_2 = module_1.Config()
    var_3 = 'os'
    var_4 = [var_3]
    var_5 = 'STDLIB'
    var_6 = []
    var_7 = 'import'
    var_8 = module_2._with_from_imports(var_1, var_2, var_4, var_5, var_6, var_7)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'from os import path as p'
    var_1 = module_0.file_contents(var_0)
    var_2 = True
    var_3 = module_1.Config()
    var_4 = 'os'
    var_5 = [var_4]
    var_6 = 'STDLIB'
    var_7 = []
    var_8 = 'import'
    var_9 = module_2._with_from_imports(var_1, var_3, var_5, var_6, var_7, var_8)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'from os import getcwd, path'
    var_1 = module_0.file_contents(var_0)
    var_2 = True
    var_3 = module_1.Config()
    var_4 = 'os'
    var_5 = [var_4]
    var_6 = 'STDLIB'
    var_7 = []
    var_8 = 'import'
    var_9 = module_2._with_from_imports(var_1, var_3, var_5, var_6, var_7, var_8)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'from os import path\nfrom sys import argv'
    var_1 = module_0.file_contents(var_0)
    var_2 = module_1.Config()
    var_3 = 'os'
    var_4 = 'sys'
    var_5 = [var_3, var_4]
    var_6 = 'STDLIB'
    var_7 = []
    var_8 = 'import'
    var_9 = module_2._with_from_imports(var_1, var_2, var_5, var_6, var_7, var_8)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'from os import path, getcwd, listdir, makedirs'
    var_1 = module_0.file_contents(var_0)
    var_2 = 40
    var_3 = module_1.Config()
    var_4 = 'os'
    var_5 = [var_4]
    var_6 = 'STDLIB'
    var_7 = []
    var_8 = 'import'
    var_9 = module_2._with_from_imports(var_1, var_3, var_5, var_6, var_7, var_8)



# Parsed testcases at query #11
#--------------------------




def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = 0
    var_3 = var_0[var_2]
    var_4 = var_3 in var_1
    var_5 = var_0 and var_4
    assert var_5 is False



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_with_star_comments_with_star_comment. Retrieved 16/20 statements.


import isort.parse as module_0
import isort.output as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = {}
    var_3 = 'nested'
    var_4 = 'module1'
    var_5 = '*'
    var_6 = 'star comment'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = {var_3: var_8}
    var_10 = module_0.ParsedContent()
    var_11 = 'comment1'
    var_12 = 'comment2'
    var_13 = [var_11, var_12]
    var_14 = module_1._with_star_comments(var_10, var_4, var_13)
    var_15 = var_10.categorized_comments[var_3][var_4]

import isort.parse as module_0
import isort.output as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = {}
    var_3 = 'nested'
    var_4 = 'module1'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = module_0.ParsedContent()
    var_9 = 'comment1'
    var_10 = 'comment2'
    var_11 = [var_9, var_10]
    var_12 = module_1._with_star_comments(var_8, var_4, var_11)

import isort.parse as module_0
import isort.output as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = {}
    var_3 = 'nested'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = module_0.ParsedContent()
    var_7 = 'comment1'
    var_8 = [var_7]
    var_9 = 'nonexistent_module'
    var_10 = module_1._with_star_comments(var_6, var_9, var_8)

import isort.parse as module_0
import isort.output as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = {}
    var_3 = 'nested'
    var_4 = 'module1'
    var_5 = '*'
    var_6 = 'star comment'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = {var_3: var_8}
    var_10 = module_0.ParsedContent()
    var_11 = []
    var_12 = module_1._with_star_comments(var_10, var_4, var_11)



# Parsed testcases at query #13
#--------------------------




import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = 'straight'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'above'
    var_6 = {}
    var_7 = {var_2: var_6}
    var_8 = {}
    var_9 = {var_5: var_7, var_2: var_8}
    var_10 = ''
    var_11 = False
    var_12 = {}
    var_13 = {}
    var_14 = module_0.ParsedContent()
    var_15 = True
    var_16 = module_1.Config()
    var_17 = []
    var_18 = 'STDLIB'
    var_19 = []
    var_20 = 'import'
    var_21 = module_2._with_straight_imports(var_14, var_16, var_17, var_18, var_19, var_20)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = 'straight'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'above'
    var_6 = {}
    var_7 = {var_2: var_6}
    var_8 = {}
    var_9 = {var_5: var_7, var_2: var_8}
    var_10 = ''
    var_11 = False
    var_12 = {}
    var_13 = {}
    var_14 = module_0.ParsedContent()
    var_15 = True
    var_16 = module_1.Config()
    var_17 = 'os'
    var_18 = 'sys'
    var_19 = [var_17, var_18]
    var_20 = 'STDLIB'
    var_21 = []
    var_22 = 'import'
    var_23 = module_2._with_straight_imports(var_14, var_16, var_19, var_20, var_21, var_22)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = 'straight'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'above'
    var_6 = {}
    var_7 = {var_2: var_6}
    var_8 = 'os'
    var_9 = 'sys'
    var_10 = 'comment1'
    var_11 = [var_10]
    var_12 = 'comment2'
    var_13 = [var_12]
    var_14 = {var_8: var_11, var_9: var_13}
    var_15 = {var_5: var_7, var_2: var_14}
    var_16 = ''
    var_17 = False
    var_18 = {}
    var_19 = {}
    var_20 = module_0.ParsedContent()
    var_21 = True
    var_22 = module_1.Config()
    var_23 = [var_8, var_9]
    var_24 = 'STDLIB'
    var_25 = []
    var_26 = 'import'
    var_27 = module_2._with_straight_imports(var_20, var_22, var_23, var_24, var_25, var_26)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = 'straight'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'above'
    var_6 = 'os'
    var_7 = '# above comment'
    var_8 = [var_7]
    var_9 = {var_6: var_8}
    var_10 = {var_2: var_9}
    var_11 = {}
    var_12 = {var_5: var_10, var_2: var_11}
    var_13 = ''
    var_14 = False
    var_15 = {}
    var_16 = {}
    var_17 = module_0.ParsedContent()
    var_18 = True
    var_19 = module_1.Config()
    var_20 = 'sys'
    var_21 = [var_6, var_20]
    var_22 = 'STDLIB'
    var_23 = []
    var_24 = 'import'
    var_25 = module_2._with_straight_imports(var_17, var_19, var_21, var_22, var_23, var_24)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = 'straight'
    var_3 = 'os'
    var_4 = 'operating_system'
    var_5 = [var_4]
    var_6 = {var_3: var_5}
    var_7 = {var_2: var_6}
    var_8 = 'above'
    var_9 = {}
    var_10 = {var_2: var_9}
    var_11 = {}
    var_12 = {var_8: var_10, var_2: var_11}
    var_13 = ''
    var_14 = False
    var_15 = {}
    var_16 = {}
    var_17 = module_0.ParsedContent()
    var_18 = True
    var_19 = module_1.Config()
    var_20 = [var_3]
    var_21 = 'STDLIB'
    var_22 = []
    var_23 = 'import'
    var_24 = module_2._with_straight_imports(var_17, var_19, var_20, var_21, var_22, var_23)
    var_25 = len(var_24)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'STDLIB'
    var_2 = 'straight'
    var_3 = 'os'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {}
    var_9 = {var_2: var_8}
    var_10 = 'above'
    var_11 = {}
    var_12 = {var_2: var_11}
    var_13 = {}
    var_14 = {var_10: var_12, var_2: var_13}
    var_15 = ''
    var_16 = False
    var_17 = {}
    var_18 = {}
    var_19 = module_0.ParsedContent()
    var_20 = False
    var_21 = module_1.Config()
    var_22 = 'sys'
    var_23 = [var_3, var_22]
    var_24 = [var_3]
    var_25 = 'import'
    var_26 = module_2._with_straight_imports(var_19, var_21, var_23, var_1, var_24, var_25)
    var_27 = str(var_26)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'STDLIB'
    var_2 = 'straight'
    var_3 = 'os'
    var_4 = 'sys'
    var_5 = None
    var_6 = {var_3: var_5, var_4: var_5}
    var_7 = {var_2: var_6}
    var_8 = {var_1: var_7}
    var_9 = {}
    var_10 = {var_2: var_9}
    var_11 = 'above'
    var_12 = {}
    var_13 = {var_2: var_12}
    var_14 = {}
    var_15 = {var_11: var_13, var_2: var_14}
    var_16 = ''
    var_17 = False
    var_18 = {}
    var_19 = {}
    var_20 = module_0.ParsedContent()
    var_21 = False
    var_22 = module_1.Config()
    var_23 = [var_3, var_4]
    var_24 = []
    var_25 = 'import'
    var_26 = module_2._with_straight_imports(var_20, var_22, var_23, var_1, var_24, var_25)
    var_27 = len(var_26)
    assert var_27 == 2

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'STDLIB'
    var_2 = 'straight'
    var_3 = 'os'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = 'operating_system'
    var_9 = [var_8]
    var_10 = {var_3: var_9}
    var_11 = {var_2: var_10}
    var_12 = 'above'
    var_13 = {}
    var_14 = {var_2: var_13}
    var_15 = {}
    var_16 = {var_12: var_14, var_2: var_15}
    var_17 = ''
    var_18 = False
    var_19 = {}
    var_20 = {}
    var_21 = module_0.ParsedContent()
    var_22 = False
    var_23 = module_1.Config()
    var_24 = [var_3]
    var_25 = []
    var_26 = 'import'
    var_27 = module_2._with_straight_imports(var_21, var_23, var_24, var_1, var_25, var_26)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_section_footer_predicate_line_129. Retrieved 13/16 statements.


import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'import os'
    var_1 = 'from typing import Dict'
    var_2 = [var_0, var_1]
    var_3 = module_0.Config()
    var_4 = 'py'
    var_5 = 'import'
    var_6 = module_1.ParsedContent()
    var_7 = 'stdlib'
    var_8 = 'End of stdlib imports'
    var_9 = {var_7: var_8}
    var_10 = True
    var_11 = module_0.Config()
    var_12 = module_2.sorted_imports(var_6, var_11)



# Parsed testcases at query #15
#--------------------------

# Failed to parse test_predicate_at_line_142_evaluates_to_true.




# Parsed testcases at query #16
#--------------------------

# Partially parsed test_with_from_imports_basic_import. Retrieved 23/31 statements.
# Partially parsed test_with_from_imports_with_remove_imports. Retrieved 22/29 statements.
# Partially parsed test_with_from_imports_multiple_modules. Retrieved 25/33 statements.
# Partially parsed test_with_from_imports_with_star_import. Retrieved 22/30 statements.
# Partially parsed test_with_from_imports_force_single_line. Retrieved 24/32 statements.
# Partially parsed test_with_from_imports_with_comments. Retrieved 24/32 statements.
# Partially parsed test_with_from_imports_combine_as_imports. Retrieved 25/33 statements.


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = module_1.Config()
    var_2 = []
    var_3 = 'THIRDPARTY'
    var_4 = []
    var_5 = 'import'
    var_6 = module_2._with_from_imports(var_0, var_1, var_2, var_3, var_4, var_5)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = {}
    var_10 = 'above'
    var_11 = 'nested'
    var_12 = {}
    var_13 = {}
    var_14 = {var_2: var_13}
    var_15 = {}
    var_16 = module_1.Config()
    var_17 = [var_3]
    var_18 = 'THIRDPARTY'
    var_19 = []
    var_20 = 'import'
    var_21 = module_2._with_from_imports(var_0, var_16, var_17, var_18, var_19, var_20)
    var_22 = len(var_21)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = {}
    var_10 = 'above'
    var_11 = 'nested'
    var_12 = {}
    var_13 = {}
    var_14 = {var_2: var_13}
    var_15 = {}
    var_16 = module_1.Config()
    var_17 = [var_3]
    var_18 = 'THIRDPARTY'
    var_19 = [var_3]
    var_20 = 'import'
    var_21 = module_2._with_from_imports(var_0, var_16, var_17, var_18, var_19, var_20)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'sys'
    var_5 = 'path'
    var_6 = False
    var_7 = {var_5: var_6}
    var_8 = 'argv'
    var_9 = {var_8: var_6}
    var_10 = {var_3: var_7, var_4: var_9}
    var_11 = {var_2: var_10}
    var_12 = {}
    var_13 = 'above'
    var_14 = 'nested'
    var_15 = {}
    var_16 = {}
    var_17 = {var_2: var_16}
    var_18 = {}
    var_19 = module_1.Config()
    var_20 = [var_3, var_4]
    var_21 = 'THIRDPARTY'
    var_22 = []
    var_23 = 'import'
    var_24 = module_2._with_from_imports(var_0, var_19, var_20, var_21, var_22, var_23)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = '*'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = {}
    var_10 = 'above'
    var_11 = 'nested'
    var_12 = {}
    var_13 = {}
    var_14 = {var_2: var_13}
    var_15 = {}
    var_16 = module_1.Config()
    var_17 = [var_3]
    var_18 = 'THIRDPARTY'
    var_19 = []
    var_20 = 'import'
    var_21 = module_2._with_from_imports(var_0, var_16, var_17, var_18, var_19, var_20)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = 'environ'
    var_6 = False
    var_7 = {var_4: var_6, var_5: var_6}
    var_8 = {var_3: var_7}
    var_9 = {var_2: var_8}
    var_10 = {}
    var_11 = 'above'
    var_12 = 'nested'
    var_13 = {}
    var_14 = {}
    var_15 = {var_2: var_14}
    var_16 = {}
    var_17 = True
    var_18 = module_1.Config()
    var_19 = [var_3]
    var_20 = 'THIRDPARTY'
    var_21 = []
    var_22 = 'import'
    var_23 = module_2._with_from_imports(var_0, var_18, var_19, var_20, var_21, var_22)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = {}
    var_10 = 'above'
    var_11 = 'nested'
    var_12 = 'important comment'
    var_13 = [var_12]
    var_14 = {var_3: var_13}
    var_15 = {}
    var_16 = {var_2: var_15}
    var_17 = {}
    var_18 = module_1.Config()
    var_19 = [var_3]
    var_20 = 'THIRDPARTY'
    var_21 = []
    var_22 = 'import'
    var_23 = module_2._with_from_imports(var_0, var_18, var_19, var_20, var_21, var_22)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = True
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'os.path'
    var_10 = 'Path'
    var_11 = [var_10]
    var_12 = {var_9: var_11}
    var_13 = 'above'
    var_14 = 'nested'
    var_15 = {}
    var_16 = {}
    var_17 = {var_2: var_16}
    var_18 = {}
    var_19 = module_1.Config()
    var_20 = [var_3]
    var_21 = 'THIRDPARTY'
    var_22 = []
    var_23 = 'import'
    var_24 = module_2._with_from_imports(var_0, var_19, var_20, var_21, var_22, var_23)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_with_from_imports_basic. Retrieved 23/34 statements.
# Partially parsed test_with_from_imports_with_remove_imports. Retrieved 22/32 statements.
# Partially parsed test_with_from_imports_empty_from_modules. Retrieved 18/28 statements.
# Partially parsed test_with_from_imports_with_star_import. Retrieved 24/35 statements.
# Partially parsed test_with_from_imports_multiple_modules. Retrieved 25/36 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'THIRDPARTY'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'environ'
    var_5 = False
    var_6 = {var_3: var_5, var_4: var_5}
    var_7 = {var_2: var_6}
    var_8 = {var_1: var_7}
    var_9 = {}
    var_10 = 'above'
    var_11 = 'nested'
    var_12 = 'straight'
    var_13 = {}
    var_14 = {}
    var_15 = {var_1: var_14}
    var_16 = {}
    var_17 = {}
    var_18 = module_0.Config()
    var_19 = [var_2]
    var_20 = 'THIRDPARTY'
    var_21 = []
    var_22 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'THIRDPARTY'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = False
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = 'straight'
    var_12 = {}
    var_13 = {}
    var_14 = {var_1: var_13}
    var_15 = {}
    var_16 = {}
    var_17 = module_0.Config()
    var_18 = [var_2]
    var_19 = 'THIRDPARTY'
    var_20 = [var_2]
    var_21 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'THIRDPARTY'
    var_1 = 'from'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = 'above'
    var_6 = 'nested'
    var_7 = 'straight'
    var_8 = {}
    var_9 = {}
    var_10 = {var_1: var_9}
    var_11 = {}
    var_12 = {}
    var_13 = module_0.Config()
    var_14 = []
    var_15 = 'THIRDPARTY'
    var_16 = []
    var_17 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'THIRDPARTY'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = '*'
    var_4 = False
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = 'straight'
    var_12 = {}
    var_13 = {}
    var_14 = {var_1: var_13}
    var_15 = {}
    var_16 = {var_2: var_15}
    var_17 = {}
    var_18 = True
    var_19 = module_0.Config()
    var_20 = [var_2]
    var_21 = 'THIRDPARTY'
    var_22 = []
    var_23 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'THIRDPARTY'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'sys'
    var_4 = 'path'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = 'argv'
    var_8 = {var_7: var_5}
    var_9 = {var_2: var_6, var_3: var_8}
    var_10 = {var_1: var_9}
    var_11 = {}
    var_12 = 'above'
    var_13 = 'nested'
    var_14 = 'straight'
    var_15 = {}
    var_16 = {}
    var_17 = {var_1: var_16}
    var_18 = {}
    var_19 = {}
    var_20 = module_0.Config()
    var_21 = [var_2, var_3]
    var_22 = 'THIRDPARTY'
    var_23 = []
    var_24 = 'import'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_predicate_at_line_248. Retrieved 1/5 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_predicate_at_line_311_evaluates_to_false. Retrieved 3/13 statements.


def test_case_0():
    var_0 = 100
    var_1 = '\n'
    var_2 = 'from module import (\n    item1,\n    item2\n)'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_line_153_predicate_evaluates_to_true. Retrieved 7/9 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 153 (output and output[0].strip() == "") evaluates to True.'
    var_1 = ''
    var_2 = 'import os'
    var_3 = 'import sys'
    var_4 = [var_1, var_2, var_3]
    var_5 = 0
    var_6 = var_4[var_5]



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_line_205_predicate_evaluates_to_true. Retrieved 13/21 statements.


import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'import os'
    var_1 = 'import sys'
    var_2 = ''
    var_3 = 'def hello():'
    var_4 = '    pass'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.Config()
    var_7 = 'py'
    var_8 = 'import'
    var_9 = module_1.ParsedContent()
    var_10 = 2
    var_11 = module_0.Config()
    var_12 = module_2.sorted_imports(var_9, var_11)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_with_from_imports_single_import. Retrieved 34/38 statements.
# Partially parsed test_with_from_imports_with_remove_imports. Retrieved 35/39 statements.
# Partially parsed test_with_from_imports_multiple_modules. Retrieved 37/41 statements.
# Partially parsed test_with_from_imports_with_comments. Retrieved 36/40 statements.
# Partially parsed test_with_from_imports_force_single_line. Retrieved 36/40 statements.
# Partially parsed test_with_from_imports_with_star_import. Retrieved 35/39 statements.


import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'FUTURE'
    var_2 = 'from'
    var_3 = 'straight'
    var_4 = {}
    var_5 = {}
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = {}
    var_9 = {}
    var_10 = {var_2: var_8, var_3: var_9}
    var_11 = 'above'
    var_12 = 'nested'
    var_13 = {}
    var_14 = {}
    var_15 = {var_2: var_14}
    var_16 = {}
    var_17 = {}
    var_18 = {var_2: var_13, var_11: var_15, var_12: var_16, var_3: var_17}
    var_19 = {}
    var_20 = {}
    var_21 = ''
    var_22 = set()
    var_23 = False
    var_24 = '\n'
    var_25 = set()
    var_26 = module_1.ParsedContent()
    var_27 = []
    var_28 = []
    var_29 = 'import'
    var_30 = module_2._with_from_imports(var_26, var_0, var_27, var_1, var_28, var_29)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'FUTURE'
    var_2 = 'from'
    var_3 = 'straight'
    var_4 = 'os'
    var_5 = 'path'
    var_6 = False
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = {}
    var_10 = {var_2: var_8, var_3: var_9}
    var_11 = {var_1: var_10}
    var_12 = {}
    var_13 = {}
    var_14 = {var_2: var_12, var_3: var_13}
    var_15 = 'above'
    var_16 = 'nested'
    var_17 = {}
    var_18 = {}
    var_19 = {var_2: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = {var_2: var_17, var_15: var_19, var_16: var_20, var_3: var_21}
    var_23 = {}
    var_24 = {}
    var_25 = ''
    var_26 = set()
    var_27 = '\n'
    var_28 = set()
    var_29 = module_1.ParsedContent()
    var_30 = [var_4]
    var_31 = []
    var_32 = 'import'
    var_33 = module_2._with_from_imports(var_29, var_0, var_30, var_1, var_31, var_32)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'FUTURE'
    var_2 = 'from'
    var_3 = 'straight'
    var_4 = 'os'
    var_5 = 'path'
    var_6 = False
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = {}
    var_10 = {var_2: var_8, var_3: var_9}
    var_11 = {var_1: var_10}
    var_12 = {}
    var_13 = {}
    var_14 = {var_2: var_12, var_3: var_13}
    var_15 = 'above'
    var_16 = 'nested'
    var_17 = {}
    var_18 = {}
    var_19 = {var_2: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = {var_2: var_17, var_15: var_19, var_16: var_20, var_3: var_21}
    var_23 = {}
    var_24 = {}
    var_25 = ''
    var_26 = set()
    var_27 = '\n'
    var_28 = set()
    var_29 = module_1.ParsedContent()
    var_30 = [var_4]
    var_31 = 'os.path'
    var_32 = [var_31]
    var_33 = 'import'
    var_34 = module_2._with_from_imports(var_29, var_0, var_30, var_1, var_32, var_33)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'FUTURE'
    var_2 = 'from'
    var_3 = 'straight'
    var_4 = 'os'
    var_5 = 'path'
    var_6 = False
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = {}
    var_10 = {var_2: var_8, var_3: var_9}
    var_11 = {var_1: var_10}
    var_12 = {}
    var_13 = {}
    var_14 = {var_2: var_12, var_3: var_13}
    var_15 = 'above'
    var_16 = 'nested'
    var_17 = {}
    var_18 = {}
    var_19 = {var_2: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = {var_2: var_17, var_15: var_19, var_16: var_20, var_3: var_21}
    var_23 = {}
    var_24 = {}
    var_25 = ''
    var_26 = set()
    var_27 = '\n'
    var_28 = set()
    var_29 = module_1.ParsedContent()
    var_30 = [var_4]
    var_31 = [var_4]
    var_32 = 'import'
    var_33 = module_2._with_from_imports(var_29, var_0, var_30, var_1, var_31, var_32)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'FUTURE'
    var_2 = 'from'
    var_3 = 'straight'
    var_4 = 'os'
    var_5 = 'sys'
    var_6 = 'path'
    var_7 = False
    var_8 = {var_6: var_7}
    var_9 = 'argv'
    var_10 = {var_9: var_7}
    var_11 = {var_4: var_8, var_5: var_10}
    var_12 = {}
    var_13 = {var_2: var_11, var_3: var_12}
    var_14 = {var_1: var_13}
    var_15 = {}
    var_16 = {}
    var_17 = {var_2: var_15, var_3: var_16}
    var_18 = 'above'
    var_19 = 'nested'
    var_20 = {}
    var_21 = {}
    var_22 = {var_2: var_21}
    var_23 = {}
    var_24 = {}
    var_25 = {var_2: var_20, var_18: var_22, var_19: var_23, var_3: var_24}
    var_26 = {}
    var_27 = {}
    var_28 = ''
    var_29 = set()
    var_30 = '\n'
    var_31 = set()
    var_32 = module_1.ParsedContent()
    var_33 = [var_4, var_5]
    var_34 = []
    var_35 = 'import'
    var_36 = module_2._with_from_imports(var_32, var_0, var_33, var_1, var_34, var_35)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'FUTURE'
    var_2 = 'from'
    var_3 = 'straight'
    var_4 = 'os'
    var_5 = 'path'
    var_6 = False
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = {}
    var_10 = {var_2: var_8, var_3: var_9}
    var_11 = {var_1: var_10}
    var_12 = {}
    var_13 = {}
    var_14 = {var_2: var_12, var_3: var_13}
    var_15 = 'above'
    var_16 = 'nested'
    var_17 = 'test comment'
    var_18 = [var_17]
    var_19 = {var_4: var_18}
    var_20 = {}
    var_21 = {var_2: var_20}
    var_22 = {}
    var_23 = {}
    var_24 = {var_2: var_19, var_15: var_21, var_16: var_22, var_3: var_23}
    var_25 = {}
    var_26 = {}
    var_27 = ''
    var_28 = set()
    var_29 = '\n'
    var_30 = set()
    var_31 = module_1.ParsedContent()
    var_32 = [var_4]
    var_33 = []
    var_34 = 'import'
    var_35 = module_2._with_from_imports(var_31, var_0, var_32, var_1, var_33, var_34)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = 'FUTURE'
    var_3 = 'from'
    var_4 = 'straight'
    var_5 = 'os'
    var_6 = 'path'
    var_7 = 'environ'
    var_8 = False
    var_9 = {var_6: var_8, var_7: var_8}
    var_10 = {var_5: var_9}
    var_11 = {}
    var_12 = {var_3: var_10, var_4: var_11}
    var_13 = {var_2: var_12}
    var_14 = {}
    var_15 = {}
    var_16 = {var_3: var_14, var_4: var_15}
    var_17 = 'above'
    var_18 = 'nested'
    var_19 = {}
    var_20 = {}
    var_21 = {var_3: var_20}
    var_22 = {}
    var_23 = {}
    var_24 = {var_3: var_19, var_17: var_21, var_18: var_22, var_4: var_23}
    var_25 = {}
    var_26 = {}
    var_27 = ''
    var_28 = set()
    var_29 = '\n'
    var_30 = set()
    var_31 = module_1.ParsedContent()
    var_32 = [var_5]
    var_33 = []
    var_34 = 'import'
    var_35 = module_2._with_from_imports(var_31, var_1, var_32, var_2, var_33, var_34)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = 'FUTURE'
    var_3 = 'from'
    var_4 = 'straight'
    var_5 = 'os'
    var_6 = '*'
    var_7 = False
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = {}
    var_11 = {var_3: var_9, var_4: var_10}
    var_12 = {var_2: var_11}
    var_13 = {}
    var_14 = {}
    var_15 = {var_3: var_13, var_4: var_14}
    var_16 = 'above'
    var_17 = 'nested'
    var_18 = {}
    var_19 = {}
    var_20 = {var_3: var_19}
    var_21 = {}
    var_22 = {}
    var_23 = {var_3: var_18, var_16: var_20, var_17: var_21, var_4: var_22}
    var_24 = {}
    var_25 = {}
    var_26 = ''
    var_27 = set()
    var_28 = '\n'
    var_29 = set()
    var_30 = module_1.ParsedContent()
    var_31 = [var_5]
    var_32 = []
    var_33 = 'import'
    var_34 = module_2._with_from_imports(var_30, var_1, var_31, var_2, var_32, var_33)



# Parsed testcases at query #23
#--------------------------




import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'import os'
    var_1 = 'import sys'
    var_2 = ''
    var_3 = 'def foo():'
    var_4 = '    pass'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.Config()
    var_7 = 'py'
    var_8 = 'import'
    var_9 = module_1.ParsedContent()
    var_10 = 2
    var_11 = module_0.Config()
    var_12 = module_2.sorted_imports(var_9, var_11, var_7, var_8)



# Parsed testcases at query #24
#--------------------------






# Parsed testcases at query #25
#--------------------------




def test_case_0():
    var_0 = 'module1'
    var_1 = 'module2'
    var_2 = [var_0, var_1]



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_with_from_imports_basic. Retrieved 33/37 statements.
# Partially parsed test_with_from_imports_with_comments. Retrieved 34/38 statements.
# Partially parsed test_with_from_imports_force_single_line. Retrieved 34/38 statements.
# Partially parsed test_with_from_imports_star_import. Retrieved 31/37 statements.


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'module1'
    var_4 = 'func1'
    var_5 = 'func2'
    var_6 = False
    var_7 = {var_4: var_6, var_5: var_6}
    var_8 = {var_3: var_7}
    var_9 = {var_2: var_8}
    var_10 = {var_1: var_9}
    var_11 = 'straight'
    var_12 = {}
    var_13 = {}
    var_14 = {var_2: var_12, var_11: var_13}
    var_15 = 'above'
    var_16 = 'nested'
    var_17 = {}
    var_18 = {}
    var_19 = {}
    var_20 = {var_2: var_19}
    var_21 = {}
    var_22 = {var_2: var_17, var_11: var_18, var_15: var_20, var_16: var_21}
    var_23 = lambda x: var_1
    var_24 = '\n'
    var_25 = set()
    var_26 = module_0.ParsedContent()
    var_27 = module_1.Config()
    var_28 = [var_3]
    var_29 = []
    var_30 = 'import'
    var_31 = module_2._with_from_imports(var_26, var_27, var_28, var_1, var_29, var_30)
    var_32 = len(var_31)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'straight'
    var_7 = {}
    var_8 = {}
    var_9 = {var_2: var_7, var_6: var_8}
    var_10 = 'above'
    var_11 = 'nested'
    var_12 = {}
    var_13 = {}
    var_14 = {}
    var_15 = {var_2: var_14}
    var_16 = {}
    var_17 = {var_2: var_12, var_6: var_13, var_10: var_15, var_11: var_16}
    var_18 = 0
    var_19 = lambda x: var_1
    var_20 = '\n'
    var_21 = set()
    var_22 = module_0.ParsedContent()
    var_23 = module_1.Config()
    var_24 = []
    var_25 = []
    var_26 = 'import'
    var_27 = module_2._with_from_imports(var_22, var_23, var_24, var_1, var_25, var_26)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'module1'
    var_4 = 'func1'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = {var_1: var_8}
    var_10 = 'straight'
    var_11 = {}
    var_12 = {}
    var_13 = {var_2: var_11, var_10: var_12}
    var_14 = 'above'
    var_15 = 'nested'
    var_16 = {}
    var_17 = {}
    var_18 = {}
    var_19 = {var_2: var_18}
    var_20 = {}
    var_21 = {var_2: var_16, var_10: var_17, var_14: var_19, var_15: var_20}
    var_22 = lambda x: var_1
    var_23 = '\n'
    var_24 = set()
    var_25 = module_0.ParsedContent()
    var_26 = module_1.Config()
    var_27 = [var_3]
    var_28 = [var_3]
    var_29 = 'import'
    var_30 = module_2._with_from_imports(var_25, var_26, var_27, var_1, var_28, var_29)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'module1'
    var_4 = 'func1'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = {var_1: var_8}
    var_10 = 'straight'
    var_11 = {}
    var_12 = {}
    var_13 = {var_2: var_11, var_10: var_12}
    var_14 = 'above'
    var_15 = 'nested'
    var_16 = 'test comment'
    var_17 = [var_16]
    var_18 = {var_3: var_17}
    var_19 = {}
    var_20 = {}
    var_21 = {var_2: var_20}
    var_22 = {}
    var_23 = {var_2: var_18, var_10: var_19, var_14: var_21, var_15: var_22}
    var_24 = lambda x: var_1
    var_25 = '\n'
    var_26 = set()
    var_27 = module_0.ParsedContent()
    var_28 = module_1.Config()
    var_29 = [var_3]
    var_30 = []
    var_31 = 'import'
    var_32 = module_2._with_from_imports(var_27, var_28, var_29, var_1, var_30, var_31)
    var_33 = len(var_32)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'module1'
    var_4 = 'func1'
    var_5 = 'func2'
    var_6 = False
    var_7 = {var_4: var_6, var_5: var_6}
    var_8 = {var_3: var_7}
    var_9 = {var_2: var_8}
    var_10 = {var_1: var_9}
    var_11 = 'straight'
    var_12 = {}
    var_13 = {}
    var_14 = {var_2: var_12, var_11: var_13}
    var_15 = 'above'
    var_16 = 'nested'
    var_17 = {}
    var_18 = {}
    var_19 = {}
    var_20 = {var_2: var_19}
    var_21 = {}
    var_22 = {var_2: var_17, var_11: var_18, var_15: var_20, var_16: var_21}
    var_23 = lambda x: var_1
    var_24 = '\n'
    var_25 = set()
    var_26 = module_0.ParsedContent()
    var_27 = True
    var_28 = module_1.Config()
    var_29 = [var_3]
    var_30 = []
    var_31 = 'import'
    var_32 = module_2._with_from_imports(var_26, var_28, var_29, var_1, var_30, var_31)
    var_33 = len(var_32)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'module1'
    var_4 = '*'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = {var_1: var_8}
    var_10 = 'straight'
    var_11 = {}
    var_12 = {}
    var_13 = {var_2: var_11, var_10: var_12}
    var_14 = 'above'
    var_15 = 'nested'
    var_16 = {}
    var_17 = {}
    var_18 = {}
    var_19 = {var_2: var_18}
    var_20 = {}
    var_21 = {var_2: var_16, var_10: var_17, var_14: var_19, var_15: var_20}
    var_22 = lambda x: var_1
    var_23 = '\n'
    var_24 = set()
    var_25 = module_0.ParsedContent()
    var_26 = module_1.Config()
    var_27 = [var_3]
    var_28 = []
    var_29 = 'import'
    var_30 = module_2._with_from_imports(var_25, var_26, var_27, var_1, var_28, var_29)



# Parsed testcases at query #27
#--------------------------






# Parsed testcases at query #28
#--------------------------




def test_case_0():
    var_0 = 'some comment'
    var_1 = bool(var_0)
    assert var_1 is True
    var_2 = '# noqa'
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = 'type: ignore'
    var_5 = bool(var_4)
    assert var_5 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_from_first_predicate_true. Retrieved 37/42 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = 'from sys import path'
    var_2 = ''
    var_3 = [var_0, var_1, var_2]
    var_4 = True
    var_5 = module_0.Config()
    var_6 = 0
    var_7 = module_1.ParsedContent()
    var_8 = 'FUTURE'
    var_9 = 'STDLIB'
    var_10 = 'THIRDPARTY'
    var_11 = 'FIRSTPARTY'
    var_12 = 'LOCALFOLDER'
    var_13 = 'straight'
    var_14 = 'from'
    var_15 = {}
    var_16 = {}
    var_17 = {var_13: var_15, var_14: var_16}
    var_18 = 'os'
    var_19 = []
    var_20 = {var_18: var_19}
    var_21 = 'sys'
    var_22 = 'path'
    var_23 = []
    var_24 = {var_22: var_23}
    var_25 = {var_21: var_24}
    var_26 = {var_13: var_20, var_14: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = {var_13: var_27, var_14: var_28}
    var_30 = {}
    var_31 = {}
    var_32 = {var_13: var_30, var_14: var_31}
    var_33 = {}
    var_34 = {}
    var_35 = {var_13: var_33, var_14: var_34}
    var_36 = module_0.Config()



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_with_straight_imports_combine_straight_imports_enabled. Retrieved 24/30 statements.
# Partially parsed test_with_straight_imports_combine_with_inline_comments. Retrieved 28/34 statements.
# Partially parsed test_with_straight_imports_as_imports_present. Retrieved 25/31 statements.
# Partially parsed test_with_straight_imports_remove_imports. Retrieved 24/30 statements.
# Partially parsed test_with_straight_imports_combine_disabled. Retrieved 24/30 statements.
# Partially parsed test_with_straight_imports_empty_modules. Retrieved 20/26 statements.
# Partially parsed test_with_straight_imports_above_comments. Retrieved 25/31 statements.


import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = 'STDLIB'
    var_3 = lambda x: var_2
    var_4 = module_0.Config()
    var_5 = module_1.ParsedContent()
    var_6 = 'straight'
    var_7 = 'os'
    var_8 = 'sys'
    var_9 = None
    var_10 = {var_7: var_9, var_8: var_9}
    var_11 = {var_6: var_10}
    var_12 = {}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_6: var_14}
    var_16 = {}
    var_17 = True
    var_18 = module_0.Config()
    var_19 = [var_7, var_8]
    var_20 = []
    var_21 = 'import'
    var_22 = module_2._with_straight_imports(var_5, var_18, var_19, var_2, var_20, var_21)
    var_23 = len(var_22)
    assert var_23 == 1

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = 'STDLIB'
    var_3 = lambda x: var_2
    var_4 = module_0.Config()
    var_5 = module_1.ParsedContent()
    var_6 = 'straight'
    var_7 = 'os'
    var_8 = 'sys'
    var_9 = None
    var_10 = {var_7: var_9, var_8: var_9}
    var_11 = {var_6: var_10}
    var_12 = {}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_6: var_14}
    var_16 = 'for file operations'
    var_17 = [var_16]
    var_18 = 'for system info'
    var_19 = [var_18]
    var_20 = {var_7: var_17, var_8: var_19}
    var_21 = True
    var_22 = module_0.Config()
    var_23 = [var_7, var_8]
    var_24 = []
    var_25 = 'import'
    var_26 = module_2._with_straight_imports(var_5, var_22, var_23, var_2, var_24, var_25)
    var_27 = len(var_26)
    assert var_27 == 1

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = 'STDLIB'
    var_3 = lambda x: var_2
    var_4 = module_0.Config()
    var_5 = module_1.ParsedContent()
    var_6 = 'straight'
    var_7 = 'os'
    var_8 = None
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = 'operating_system'
    var_12 = [var_11]
    var_13 = {var_7: var_12}
    var_14 = 'above'
    var_15 = {}
    var_16 = {var_6: var_15}
    var_17 = {}
    var_18 = True
    var_19 = module_0.Config()
    var_20 = [var_7]
    var_21 = []
    var_22 = 'import'
    var_23 = module_2._with_straight_imports(var_5, var_19, var_20, var_2, var_21, var_22)
    var_24 = len(var_23)
    assert var_24 == 2

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = 'STDLIB'
    var_3 = lambda x: var_2
    var_4 = module_0.Config()
    var_5 = module_1.ParsedContent()
    var_6 = 'straight'
    var_7 = 'os'
    var_8 = 'sys'
    var_9 = None
    var_10 = {var_7: var_9, var_8: var_9}
    var_11 = {var_6: var_10}
    var_12 = {}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_6: var_14}
    var_16 = {}
    var_17 = False
    var_18 = module_0.Config()
    var_19 = [var_7, var_8]
    var_20 = [var_7]
    var_21 = 'import'
    var_22 = module_2._with_straight_imports(var_5, var_18, var_19, var_2, var_20, var_21)
    var_23 = len(var_22)
    assert var_23 == 1

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = 'STDLIB'
    var_3 = lambda x: var_2
    var_4 = module_0.Config()
    var_5 = module_1.ParsedContent()
    var_6 = 'straight'
    var_7 = 'os'
    var_8 = 'sys'
    var_9 = None
    var_10 = {var_7: var_9, var_8: var_9}
    var_11 = {var_6: var_10}
    var_12 = {}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_6: var_14}
    var_16 = {}
    var_17 = False
    var_18 = module_0.Config()
    var_19 = [var_7, var_8]
    var_20 = []
    var_21 = 'import'
    var_22 = module_2._with_straight_imports(var_5, var_18, var_19, var_2, var_20, var_21)
    var_23 = len(var_22)
    assert var_23 == 2

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = 'STDLIB'
    var_3 = lambda x: var_2
    var_4 = module_0.Config()
    var_5 = module_1.ParsedContent()
    var_6 = 'straight'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = {}
    var_10 = 'above'
    var_11 = {}
    var_12 = {var_6: var_11}
    var_13 = {}
    var_14 = True
    var_15 = module_0.Config()
    var_16 = []
    var_17 = []
    var_18 = 'import'
    var_19 = module_2._with_straight_imports(var_5, var_15, var_16, var_2, var_17, var_18)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = 'STDLIB'
    var_3 = lambda x: var_2
    var_4 = module_0.Config()
    var_5 = module_1.ParsedContent()
    var_6 = 'straight'
    var_7 = 'os'
    var_8 = None
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = {}
    var_12 = 'above'
    var_13 = '# type: ignore'
    var_14 = [var_13]
    var_15 = {var_7: var_14}
    var_16 = {var_6: var_15}
    var_17 = {}
    var_18 = False
    var_19 = module_0.Config()
    var_20 = [var_7]
    var_21 = []
    var_22 = 'import'
    var_23 = module_2._with_straight_imports(var_5, var_19, var_20, var_2, var_21, var_22)
    var_24 = len(var_23)
    assert var_24 == 2



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_ensure_newline_before_comments_predicate. Retrieved 9/12 statements.


import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = 'import os'
    var_3 = '# This is a comment'
    var_4 = 'import sys'
    var_5 = [var_2, var_3, var_4]
    var_6 = 'py'
    var_7 = module_1.ParsedContent()
    var_8 = module_2.sorted_imports(var_7, var_1)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_predicate_at_line_81_evaluates_to_true. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 'test_module'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_predicate_at_line_210_evaluates_to_true. Retrieved 15/21 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'import os\n'
    var_1 = 'def foo():\n'
    var_2 = '    pass\n'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Config()
    var_5 = 'py'
    var_6 = module_1.ParsedContent()
    var_7 = 'class Foo:\n'
    var_8 = [var_0, var_7, var_2]
    var_9 = module_0.Config()
    var_10 = module_1.ParsedContent()
    var_11 = 'py'
    var_12 = 'class Foo:'
    var_13 = 'pyi'
    var_14 = var_11 != var_13



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_sorted_imports_with_line_separator. Retrieved 29/33 statements.
# Partially parsed test_line_with_comments. Retrieved 4/7 statements.
# Partially parsed test_sorted_imports_no_sections. Retrieved 34/38 statements.


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = -1
    var_1 = 'x = 1'
    var_2 = [var_1]
    var_3 = 1
    var_4 = {}
    var_5 = {}
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = {}
    var_9 = {}
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = {}
    var_12 = 'above'
    var_13 = {}
    var_14 = {var_6: var_13}
    var_15 = {}
    var_16 = {var_12: var_14, var_6: var_15}
    var_17 = []
    var_18 = '\n'
    var_19 = module_0.ParsedContent()
    var_20 = module_1.Config()
    var_21 = module_2.sorted_imports(var_19, var_20)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'x = 1'
    var_2 = [var_1]
    var_3 = 1
    var_4 = 2
    var_5 = {}
    var_6 = {}
    var_7 = 'straight'
    var_8 = 'from'
    var_9 = {}
    var_10 = {}
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = 'FUTURE'
    var_13 = 'os'
    var_14 = ''
    var_15 = {var_13: var_14}
    var_16 = {}
    var_17 = {var_7: var_15, var_8: var_16}
    var_18 = {var_12: var_17}
    var_19 = 'above'
    var_20 = {}
    var_21 = {var_7: var_20}
    var_22 = {}
    var_23 = {var_19: var_21, var_7: var_22}
    var_24 = [var_12]
    var_25 = '\n'
    var_26 = module_0.ParsedContent()
    var_27 = module_1.Config()
    var_28 = module_2.sorted_imports(var_26, var_27)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'x = 1'
    var_2 = [var_1]
    var_3 = 1
    var_4 = 2
    var_5 = {}
    var_6 = {}
    var_7 = 'straight'
    var_8 = 'from'
    var_9 = {}
    var_10 = {}
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = 'STDLIB'
    var_13 = {}
    var_14 = 'os'
    var_15 = 'path'
    var_16 = {var_15}
    var_17 = {var_14: var_16}
    var_18 = {var_7: var_13, var_8: var_17}
    var_19 = {var_12: var_18}
    var_20 = 'above'
    var_21 = {}
    var_22 = {var_7: var_21}
    var_23 = {}
    var_24 = {var_20: var_22, var_7: var_23}
    var_25 = [var_12]
    var_26 = '\n'
    var_27 = module_0.ParsedContent()
    var_28 = module_1.Config()
    var_29 = module_2.sorted_imports(var_27, var_28)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'x = 1'
    var_2 = [var_1]
    var_3 = 1
    var_4 = 2
    var_5 = {}
    var_6 = {}
    var_7 = 'straight'
    var_8 = 'from'
    var_9 = {}
    var_10 = {}
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = 'FUTURE'
    var_13 = 'os'
    var_14 = ''
    var_15 = {var_13: var_14}
    var_16 = {}
    var_17 = {var_7: var_15, var_8: var_16}
    var_18 = {var_12: var_17}
    var_19 = 'above'
    var_20 = {}
    var_21 = {var_7: var_20}
    var_22 = {}
    var_23 = {var_19: var_21, var_7: var_22}
    var_24 = [var_12]
    var_25 = '\r\n'
    var_26 = module_0.ParsedContent()
    var_27 = module_1.Config()
    var_28 = module_2.sorted_imports(var_26, var_27)

import isort.output as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = 'import sys'
    var_2 = ''
    var_3 = [var_0, var_1, var_2, var_2]
    var_4 = module_0._normalize_empty_lines(var_3)

import isort.output as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = '# This is a comment'
    var_2 = 'x = 1'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._ensure_newline_before_comment(var_3)

def test_case_0():
    var_0 = 'import os'
    var_1 = 'comment1'
    var_2 = 'comment2'
    var_3 = [var_1, var_2]

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'x = 1'
    var_2 = [var_1]
    var_3 = 1
    var_4 = 2
    var_5 = {}
    var_6 = {}
    var_7 = 'straight'
    var_8 = 'from'
    var_9 = {}
    var_10 = {}
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = 'FUTURE'
    var_13 = 'STDLIB'
    var_14 = 'os'
    var_15 = ''
    var_16 = {var_14: var_15}
    var_17 = {}
    var_18 = {var_7: var_16, var_8: var_17}
    var_19 = {}
    var_20 = {}
    var_21 = {var_7: var_19, var_8: var_20}
    var_22 = {var_12: var_18, var_13: var_21}
    var_23 = 'above'
    var_24 = {}
    var_25 = {var_7: var_24}
    var_26 = {}
    var_27 = {var_23: var_25, var_7: var_26}
    var_28 = [var_12, var_13]
    var_29 = '\n'
    var_30 = module_0.ParsedContent()
    var_31 = True
    var_32 = module_1.Config()
    var_33 = module_2.sorted_imports(var_30, var_32)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'x = 1'
    var_2 = [var_1]
    var_3 = 1
    var_4 = 2
    var_5 = {}
    var_6 = {}
    var_7 = 'straight'
    var_8 = 'from'
    var_9 = {}
    var_10 = {}
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = 'FUTURE'
    var_13 = 'os'
    var_14 = ''
    var_15 = {var_13: var_14}
    var_16 = {}
    var_17 = {var_7: var_15, var_8: var_16}
    var_18 = {var_12: var_17}
    var_19 = 'above'
    var_20 = {}
    var_21 = {var_7: var_20}
    var_22 = {}
    var_23 = {var_19: var_21, var_7: var_22}
    var_24 = [var_12]
    var_25 = '\n'
    var_26 = module_0.ParsedContent()
    var_27 = 'future'
    var_28 = 'Future imports'
    var_29 = {var_27: var_28}
    var_30 = module_1.Config()
    var_31 = module_2.sorted_imports(var_26, var_30)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'x = 1'
    var_2 = [var_1]
    var_3 = 1
    var_4 = 2
    var_5 = {}
    var_6 = {}
    var_7 = 'straight'
    var_8 = 'from'
    var_9 = {}
    var_10 = {}
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = 'STDLIB'
    var_13 = 'os'
    var_14 = 'sys'
    var_15 = ''
    var_16 = {var_13: var_15, var_14: var_15}
    var_17 = {}
    var_18 = {var_7: var_16, var_8: var_17}
    var_19 = {var_12: var_18}
    var_20 = 'above'
    var_21 = {}
    var_22 = {var_7: var_21}
    var_23 = {}
    var_24 = {var_20: var_22, var_7: var_23}
    var_25 = [var_12]
    var_26 = '\n'
    var_27 = module_0.ParsedContent()
    var_28 = True
    var_29 = module_1.Config()
    var_30 = module_2.sorted_imports(var_27, var_29)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_predicate_at_line_311_evaluates_to_false. Retrieved 4/7 statements.


import re as module_0

def test_case_0():
    var_0 = 'from module import a, b, c'
    var_1 = '\n'
    var_2 = 50
    var_3 = module_0.split(var_1)



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_with_from_imports_predicate_at_line_1. Retrieved 7/10 statements.


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = module_1.Config()
    var_2 = []
    var_3 = 'THIRDPARTY'
    var_4 = []
    var_5 = 'import'
    var_6 = module_2._with_from_imports(var_0, var_1, var_2, var_3, var_4, var_5)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_predicate_at_line_192_evaluates_to_true. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = False
    var_2 = ''



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_sorted_imports_no_sections. Retrieved 49/53 statements.


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = -1
    var_2 = {}
    var_3 = {}
    var_4 = 'straight'
    var_5 = 'from'
    var_6 = {}
    var_7 = {}
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {}
    var_10 = 'above'
    var_11 = {}
    var_12 = {var_4: var_11}
    var_13 = {}
    var_14 = {var_10: var_12, var_4: var_13}
    var_15 = "print('hello')"
    var_16 = [var_15]
    var_17 = [var_15]
    var_18 = '\n'
    var_19 = []
    var_20 = 1
    var_21 = module_0.ParsedContent()
    var_22 = module_1.Config()
    var_23 = module_2.sorted_imports(var_21, var_22)
    assert var_23 == "print('hello')"

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = 0
    var_2 = {}
    var_3 = {}
    var_4 = 'straight'
    var_5 = 'from'
    var_6 = {}
    var_7 = {}
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = 'FUTURE'
    var_10 = 'STDLIB'
    var_11 = 'THIRDPARTY'
    var_12 = 'FIRSTPARTY'
    var_13 = 'LOCALFOLDER'
    var_14 = {}
    var_15 = {}
    var_16 = {var_4: var_14, var_5: var_15}
    var_17 = 'os'
    var_18 = 'sys'
    var_19 = {}
    var_20 = {}
    var_21 = {var_17: var_19, var_18: var_20}
    var_22 = {}
    var_23 = {var_4: var_21, var_5: var_22}
    var_24 = {}
    var_25 = {}
    var_26 = {var_4: var_24, var_5: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = {var_4: var_27, var_5: var_28}
    var_30 = {}
    var_31 = {}
    var_32 = {var_4: var_30, var_5: var_31}
    var_33 = {var_9: var_16, var_10: var_23, var_11: var_26, var_12: var_29, var_13: var_32}
    var_34 = 'above'
    var_35 = {}
    var_36 = {var_4: var_35}
    var_37 = {}
    var_38 = {var_34: var_36, var_4: var_37}
    var_39 = "print('hello')"
    var_40 = [var_39]
    var_41 = 'import os'
    var_42 = 'import sys'
    var_43 = [var_41, var_42, var_39]
    var_44 = '\n'
    var_45 = [var_9, var_10, var_11, var_12, var_13]
    var_46 = 3
    var_47 = module_0.ParsedContent()
    var_48 = module_1.Config()
    var_49 = module_2.sorted_imports(var_47, var_48)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = 0
    var_2 = {}
    var_3 = {}
    var_4 = 'straight'
    var_5 = 'from'
    var_6 = {}
    var_7 = {}
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = 'FUTURE'
    var_10 = 'STDLIB'
    var_11 = 'THIRDPARTY'
    var_12 = 'FIRSTPARTY'
    var_13 = 'LOCALFOLDER'
    var_14 = {}
    var_15 = {}
    var_16 = {var_4: var_14, var_5: var_15}
    var_17 = {}
    var_18 = 'os'
    var_19 = 'path'
    var_20 = [var_19]
    var_21 = {var_18: var_20}
    var_22 = {var_4: var_17, var_5: var_21}
    var_23 = {}
    var_24 = {}
    var_25 = {var_4: var_23, var_5: var_24}
    var_26 = {}
    var_27 = {}
    var_28 = {var_4: var_26, var_5: var_27}
    var_29 = {}
    var_30 = {}
    var_31 = {var_4: var_29, var_5: var_30}
    var_32 = {var_9: var_16, var_10: var_22, var_11: var_25, var_12: var_28, var_13: var_31}
    var_33 = 'above'
    var_34 = {}
    var_35 = {var_4: var_34}
    var_36 = {}
    var_37 = {var_33: var_35, var_4: var_36}
    var_38 = "print('hello')"
    var_39 = [var_38]
    var_40 = 'from os import path'
    var_41 = [var_40, var_38]
    var_42 = '\n'
    var_43 = [var_9, var_10, var_11, var_12, var_13]
    var_44 = 2
    var_45 = module_0.ParsedContent()
    var_46 = module_1.Config()
    var_47 = module_2.sorted_imports(var_45, var_46)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = 0
    var_2 = {}
    var_3 = {}
    var_4 = 'straight'
    var_5 = 'from'
    var_6 = {}
    var_7 = {}
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = 'FUTURE'
    var_10 = 'STDLIB'
    var_11 = 'THIRDPARTY'
    var_12 = 'FIRSTPARTY'
    var_13 = 'LOCALFOLDER'
    var_14 = '__future__'
    var_15 = {}
    var_16 = {var_14: var_15}
    var_17 = {}
    var_18 = {var_4: var_16, var_5: var_17}
    var_19 = 'os'
    var_20 = {}
    var_21 = {var_19: var_20}
    var_22 = {}
    var_23 = {var_4: var_21, var_5: var_22}
    var_24 = {}
    var_25 = {}
    var_26 = {var_4: var_24, var_5: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = {var_4: var_27, var_5: var_28}
    var_30 = {}
    var_31 = {}
    var_32 = {var_4: var_30, var_5: var_31}
    var_33 = {var_9: var_18, var_10: var_23, var_11: var_26, var_12: var_29, var_13: var_32}
    var_34 = 'above'
    var_35 = {}
    var_36 = {var_4: var_35}
    var_37 = {}
    var_38 = {var_34: var_36, var_4: var_37}
    var_39 = []
    var_40 = 'import os'
    var_41 = [var_40]
    var_42 = '\n'
    var_43 = [var_9, var_10, var_11, var_12, var_13]
    var_44 = 1
    var_45 = module_0.ParsedContent()
    var_46 = True
    var_47 = module_1.Config()
    var_48 = module_2.sorted_imports(var_45, var_47)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = 0
    var_2 = {}
    var_3 = {}
    var_4 = 'straight'
    var_5 = 'from'
    var_6 = {}
    var_7 = {}
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = 'FUTURE'
    var_10 = 'STDLIB'
    var_11 = 'THIRDPARTY'
    var_12 = 'FIRSTPARTY'
    var_13 = 'LOCALFOLDER'
    var_14 = {}
    var_15 = {}
    var_16 = {var_4: var_14, var_5: var_15}
    var_17 = 'os'
    var_18 = 'sys'
    var_19 = {}
    var_20 = {}
    var_21 = {var_17: var_19, var_18: var_20}
    var_22 = {}
    var_23 = {var_4: var_21, var_5: var_22}
    var_24 = {}
    var_25 = {}
    var_26 = {var_4: var_24, var_5: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = {var_4: var_27, var_5: var_28}
    var_30 = {}
    var_31 = {}
    var_32 = {var_4: var_30, var_5: var_31}
    var_33 = {var_9: var_16, var_10: var_23, var_11: var_26, var_12: var_29, var_13: var_32}
    var_34 = 'above'
    var_35 = {}
    var_36 = {var_4: var_35}
    var_37 = {}
    var_38 = {var_34: var_36, var_4: var_37}
    var_39 = "print('hello')"
    var_40 = [var_39]
    var_41 = 'import os'
    var_42 = 'import sys'
    var_43 = [var_41, var_42, var_39]
    var_44 = '\n'
    var_45 = [var_9, var_10, var_11, var_12, var_13]
    var_46 = 3
    var_47 = module_0.ParsedContent()
    var_48 = [var_41]
    var_49 = module_1.Config()
    var_50 = module_2.sorted_imports(var_47, var_49)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = ''
    var_1 = 0
    var_2 = {}
    var_3 = {}
    var_4 = 'straight'
    var_5 = 'from'
    var_6 = {}
    var_7 = {}
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = 'FUTURE'
    var_10 = 'STDLIB'
    var_11 = 'THIRDPARTY'
    var_12 = 'FIRSTPARTY'
    var_13 = 'LOCALFOLDER'
    var_14 = {}
    var_15 = {}
    var_16 = {var_4: var_14, var_5: var_15}
    var_17 = 'os'
    var_18 = {}
    var_19 = {var_17: var_18}
    var_20 = {}
    var_21 = {var_4: var_19, var_5: var_20}
    var_22 = {}
    var_23 = {}
    var_24 = {var_4: var_22, var_5: var_23}
    var_25 = {}
    var_26 = {}
    var_27 = {var_4: var_25, var_5: var_26}
    var_28 = {}
    var_29 = {}
    var_30 = {var_4: var_28, var_5: var_29}
    var_31 = {var_9: var_16, var_10: var_21, var_11: var_24, var_12: var_27, var_13: var_30}
    var_32 = 'above'
    var_33 = {}
    var_34 = {var_4: var_33}
    var_35 = {}
    var_36 = {var_32: var_34, var_4: var_35}
    var_37 = "print('hello')"
    var_38 = [var_37]
    var_39 = 'import os'
    var_40 = [var_39, var_37]
    var_41 = '\n'
    var_42 = [var_9, var_10, var_11, var_12, var_13]
    var_43 = 2
    var_44 = module_0.ParsedContent()
    var_45 = 'stdlib'
    var_46 = 'Standard Library'
    var_47 = {var_45: var_46}
    var_48 = module_1.Config()
    var_49 = module_2.sorted_imports(var_44, var_48)

def test_case_0():
    pass



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_predicate_at_line_151_evaluates_to_true. Retrieved 6/8 statements.


def test_case_0():
    var_0 = 'import a'
    var_1 = 'import b'
    var_2 = ''
    var_3 = [var_0, var_1, var_2]
    var_4 = -1
    var_5 = var_3[var_4]



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_with_from_imports_empty_from_modules. Retrieved 19/25 statements.
# Partially parsed test_with_from_imports_single_module. Retrieved 24/30 statements.
# Partially parsed test_with_from_imports_with_remove_imports. Retrieved 23/29 statements.
# Partially parsed test_with_from_imports_with_star_import. Retrieved 24/30 statements.
# Partially parsed test_with_from_imports_force_single_line. Retrieved 25/31 statements.
# Partially parsed test_with_from_imports_with_as_imports. Retrieved 27/33 statements.
# Partially parsed test_with_from_imports_with_comments. Retrieved 26/32 statements.
# Partially parsed test_with_from_imports_multiple_modules. Retrieved 27/33 statements.


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = {}
    var_6 = 'above'
    var_7 = 'nested'
    var_8 = 'straight'
    var_9 = {}
    var_10 = {}
    var_11 = {var_2: var_10}
    var_12 = {}
    var_13 = {}
    var_14 = module_1.Config()
    var_15 = []
    var_16 = []
    var_17 = 'import'
    var_18 = module_2._with_from_imports(var_0, var_14, var_15, var_1, var_16, var_17)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = True
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = {}
    var_10 = 'above'
    var_11 = 'nested'
    var_12 = 'straight'
    var_13 = {}
    var_14 = {}
    var_15 = {var_2: var_14}
    var_16 = {}
    var_17 = {}
    var_18 = module_1.Config()
    var_19 = [var_3]
    var_20 = []
    var_21 = 'import'
    var_22 = module_2._with_from_imports(var_0, var_18, var_19, var_1, var_20, var_21)
    var_23 = len(var_22)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = True
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = {}
    var_10 = 'above'
    var_11 = 'nested'
    var_12 = 'straight'
    var_13 = {}
    var_14 = {}
    var_15 = {var_2: var_14}
    var_16 = {}
    var_17 = {}
    var_18 = module_1.Config()
    var_19 = [var_3]
    var_20 = [var_3]
    var_21 = 'import'
    var_22 = module_2._with_from_imports(var_0, var_18, var_19, var_1, var_20, var_21)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = '*'
    var_5 = True
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = {}
    var_10 = 'above'
    var_11 = 'nested'
    var_12 = 'straight'
    var_13 = {}
    var_14 = {}
    var_15 = {var_2: var_14}
    var_16 = {}
    var_17 = {}
    var_18 = module_1.Config()
    var_19 = [var_3]
    var_20 = []
    var_21 = 'import'
    var_22 = module_2._with_from_imports(var_0, var_18, var_19, var_1, var_20, var_21)
    var_23 = len(var_22)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = 'getcwd'
    var_6 = True
    var_7 = {var_4: var_6, var_5: var_6}
    var_8 = {var_3: var_7}
    var_9 = {var_2: var_8}
    var_10 = {}
    var_11 = 'above'
    var_12 = 'nested'
    var_13 = 'straight'
    var_14 = {}
    var_15 = {}
    var_16 = {var_2: var_15}
    var_17 = {}
    var_18 = {}
    var_19 = module_1.Config()
    var_20 = [var_3]
    var_21 = []
    var_22 = 'import'
    var_23 = module_2._with_from_imports(var_0, var_19, var_20, var_1, var_21, var_22)
    var_24 = len(var_23)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = True
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'os.path'
    var_10 = 'p'
    var_11 = [var_10]
    var_12 = {var_9: var_11}
    var_13 = 'above'
    var_14 = 'nested'
    var_15 = 'straight'
    var_16 = {}
    var_17 = {}
    var_18 = {var_2: var_17}
    var_19 = {}
    var_20 = {}
    var_21 = module_1.Config()
    var_22 = [var_3]
    var_23 = []
    var_24 = 'import'
    var_25 = module_2._with_from_imports(var_0, var_21, var_22, var_1, var_23, var_24)
    var_26 = len(var_25)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = True
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = {}
    var_10 = 'above'
    var_11 = 'nested'
    var_12 = 'straight'
    var_13 = 'important comment'
    var_14 = [var_13]
    var_15 = {var_3: var_14}
    var_16 = {}
    var_17 = {var_2: var_16}
    var_18 = {}
    var_19 = {}
    var_20 = module_1.Config()
    var_21 = [var_3]
    var_22 = []
    var_23 = 'import'
    var_24 = module_2._with_from_imports(var_0, var_20, var_21, var_1, var_22, var_23)
    var_25 = len(var_24)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'sys'
    var_5 = 'path'
    var_6 = True
    var_7 = {var_5: var_6}
    var_8 = 'argv'
    var_9 = {var_8: var_6}
    var_10 = {var_3: var_7, var_4: var_9}
    var_11 = {var_2: var_10}
    var_12 = {}
    var_13 = 'above'
    var_14 = 'nested'
    var_15 = 'straight'
    var_16 = {}
    var_17 = {}
    var_18 = {var_2: var_17}
    var_19 = {}
    var_20 = {}
    var_21 = module_1.Config()
    var_22 = [var_3, var_4]
    var_23 = []
    var_24 = 'import'
    var_25 = module_2._with_from_imports(var_0, var_21, var_22, var_1, var_23, var_24)
    var_26 = len(var_25)



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_comments_above_predicate_evaluates_to_true. Retrieved 26/32 statements.


import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'Test that the predicate at line 60 (if comments_above:) evaluates to True.'
    var_1 = []
    var_2 = 0
    var_3 = 'THIRDPARTY'
    var_4 = lambda x: var_3
    var_5 = module_0.Config()
    var_6 = module_1.ParsedContent()
    var_7 = 'above'
    var_8 = 'straight'
    var_9 = 'os'
    var_10 = '# This is a comment above os import'
    var_11 = [var_10]
    var_12 = {var_9: var_11}
    var_13 = {var_8: var_12}
    var_14 = {}
    var_15 = {}
    var_16 = []
    var_17 = {var_9: var_16}
    var_18 = {var_8: var_17}
    var_19 = module_0.Config()
    var_20 = [var_9]
    var_21 = 'THIRDPARTY'
    var_22 = []
    var_23 = 'import'
    var_24 = module_2._with_straight_imports(var_6, var_19, var_20, var_21, var_22, var_23)
    var_25 = len(var_24)



# Parsed testcases at query #42
#--------------------------




def test_case_0():
    var_0 = '# comment 1'
    var_1 = '# comment 2'
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = False
    assert var_4 is True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_predicate_at_line_116_evaluates_to_true. Retrieved 19/30 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'line1'
    var_1 = 'line2'
    var_2 = 'FUTURE'
    var_3 = 'STDLIB'
    var_4 = 'straight'
    var_5 = 'from'
    var_6 = {}
    var_7 = {}
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = 'os'
    var_10 = []
    var_11 = {var_9: var_10}
    var_12 = {}
    var_13 = {var_4: var_11, var_5: var_12}
    var_14 = 'import os'
    var_15 = [var_14]
    var_16 = module_0.Config()
    var_17 = 'STDLIB'
    var_18 = [var_14]



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_with_from_imports_basic_structure. Retrieved 38/42 statements.
# Partially parsed test_with_from_imports_returns_list. Retrieved 35/39 statements.


import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = module_0.Config()
    var_2 = 'from'
    var_3 = False
    var_4 = None
    var_5 = ''
    var_6 = {}
    var_7 = {}
    var_8 = {}
    var_9 = {}
    var_10 = {}
    var_11 = set()
    var_12 = set()
    var_13 = {}
    var_14 = {}
    var_15 = []
    var_16 = module_1.ParsedContent()
    var_17 = module_0.Config()
    var_18 = []
    var_19 = 'STDLIB'
    var_20 = []
    var_21 = 'import'
    var_22 = module_2._with_from_imports(var_16, var_17, var_18, var_19, var_20, var_21)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = module_0.Config()
    var_2 = 'from'
    var_3 = False
    var_4 = None
    var_5 = ''
    var_6 = 'STDLIB'
    var_7 = 'os'
    var_8 = 'path'
    var_9 = {var_8: var_3}
    var_10 = {var_7: var_9}
    var_11 = {var_2: var_10}
    var_12 = {var_6: var_11}
    var_13 = {}
    var_14 = {}
    var_15 = {}
    var_16 = 'above'
    var_17 = 'nested'
    var_18 = 'straight'
    var_19 = {}
    var_20 = {}
    var_21 = {var_2: var_20}
    var_22 = {}
    var_23 = {}
    var_24 = {var_2: var_19, var_16: var_21, var_17: var_22, var_18: var_23}
    var_25 = set()
    var_26 = set()
    var_27 = {}
    var_28 = {}
    var_29 = []
    var_30 = {}
    var_31 = {var_2: var_30}
    var_32 = module_1.ParsedContent()
    var_33 = module_0.Config()
    var_34 = [var_7]
    var_35 = []
    var_36 = 'import'
    var_37 = module_2._with_from_imports(var_32, var_33, var_34, var_6, var_35, var_36)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = module_0.Config()
    var_2 = 'from'
    var_3 = False
    var_4 = None
    var_5 = ''
    var_6 = 'STDLIB'
    var_7 = 'os'
    var_8 = 'path'
    var_9 = {var_8: var_3}
    var_10 = {var_7: var_9}
    var_11 = {var_2: var_10}
    var_12 = {var_6: var_11}
    var_13 = {}
    var_14 = {}
    var_15 = {}
    var_16 = 'above'
    var_17 = 'nested'
    var_18 = 'straight'
    var_19 = {}
    var_20 = {}
    var_21 = {var_2: var_20}
    var_22 = {}
    var_23 = {}
    var_24 = {var_2: var_19, var_16: var_21, var_17: var_22, var_18: var_23}
    var_25 = set()
    var_26 = set()
    var_27 = {}
    var_28 = {}
    var_29 = []
    var_30 = {}
    var_31 = {var_2: var_30}
    var_32 = module_1.ParsedContent()
    var_33 = module_0.Config()
    var_34 = [var_7]
    var_35 = [var_7]
    var_36 = 'import'
    var_37 = module_2._with_from_imports(var_32, var_33, var_34, var_6, var_35, var_36)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = module_0.Config()
    var_2 = 'from'
    var_3 = False
    var_4 = None
    var_5 = ''
    var_6 = 'STDLIB'
    var_7 = {}
    var_8 = {var_2: var_7}
    var_9 = {var_6: var_8}
    var_10 = {}
    var_11 = {}
    var_12 = {}
    var_13 = 'above'
    var_14 = 'nested'
    var_15 = 'straight'
    var_16 = {}
    var_17 = {}
    var_18 = {var_2: var_17}
    var_19 = {}
    var_20 = {}
    var_21 = {var_2: var_16, var_13: var_18, var_14: var_19, var_15: var_20}
    var_22 = set()
    var_23 = set()
    var_24 = {}
    var_25 = {}
    var_26 = []
    var_27 = {}
    var_28 = {var_2: var_27}
    var_29 = module_1.ParsedContent()
    var_30 = module_0.Config()
    var_31 = []
    var_32 = []
    var_33 = 'import'
    var_34 = module_2._with_from_imports(var_29, var_30, var_31, var_6, var_32, var_33)



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_predicate_at_line_311_evaluates_to_false. Retrieved 4/7 statements.
# Partially parsed test_predicate_at_line_311_with_multiline_import. Retrieved 4/7 statements.
# Partially parsed test_predicate_at_line_311_with_short_lines. Retrieved 4/7 statements.


import re as module_0

def test_case_0():
    var_0 = 'from module import a, b, c'
    var_1 = '\n'
    var_2 = 80
    var_3 = module_0.split(var_1)

import re as module_0

def test_case_0():
    var_0 = 'from module import (\n    a,\n    b\n)'
    var_1 = '\n'
    var_2 = 100
    var_3 = module_0.split(var_1)

import re as module_0

def test_case_0():
    var_0 = 'from x import a\nfrom y import b'
    var_1 = '\n'
    var_2 = 50
    var_3 = module_0.split(var_1)



# Parsed testcases at query #46
#--------------------------






# Parsed testcases at query #47
#--------------------------

# Partially parsed test_with_from_imports_with_removed_module. Retrieved 9/13 statements.
# Partially parsed test_with_from_imports_basic_structure. Retrieved 22/31 statements.
# Partially parsed test_with_from_imports_returns_list. Retrieved 21/30 statements.


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = module_1.Config()
    var_2 = []
    var_3 = 'THIRDPARTY'
    var_4 = []
    var_5 = 'import'
    var_6 = module_2._with_from_imports(var_0, var_1, var_2, var_3, var_4, var_5)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = module_1.Config()
    var_2 = 'os'
    var_3 = 'sys'
    var_4 = [var_2, var_3]
    var_5 = 'STDLIB'
    var_6 = [var_2]
    var_7 = 'import'
    var_8 = module_2._with_from_imports(var_0, var_1, var_4, var_5, var_6, var_7)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'module1'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {}
    var_8 = 'above'
    var_9 = 'nested'
    var_10 = 'straight'
    var_11 = {}
    var_12 = {}
    var_13 = {var_2: var_12}
    var_14 = {}
    var_15 = {}
    var_16 = module_1.Config()
    var_17 = [var_3]
    var_18 = 'THIRDPARTY'
    var_19 = []
    var_20 = 'import'
    var_21 = module_2._with_from_imports(var_0, var_16, var_17, var_18, var_19, var_20)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = {}
    var_6 = 'above'
    var_7 = 'nested'
    var_8 = 'straight'
    var_9 = {}
    var_10 = {}
    var_11 = {var_2: var_10}
    var_12 = {}
    var_13 = {}
    var_14 = module_1.Config()
    var_15 = []
    var_16 = 'THIRDPARTY'
    var_17 = []
    var_18 = 'import'
    var_19 = module_2._with_from_imports(var_0, var_14, var_15, var_16, var_17, var_18)
    var_20 = len(var_19)
    assert var_20 == 0



# Parsed testcases at query #48
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = []
    var_3 = 0
    var_4 = False
    var_5 = frozenset()
    var_6 = {}
    var_7 = {}
    var_8 = 'FUTURE'
    var_9 = 'STDLIB'
    var_10 = (var_8, var_9)
    var_11 = module_1.ParsedContent()



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_with_from_imports_basic. Retrieved 25/33 statements.
# Partially parsed test_with_from_imports_with_remove_imports. Retrieved 25/33 statements.
# Partially parsed test_with_from_imports_empty_modules. Retrieved 20/28 statements.
# Partially parsed test_with_from_imports_with_comments. Retrieved 25/33 statements.
# Partially parsed test_with_from_imports_with_star_imports. Retrieved 27/35 statements.


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = 'sep'
    var_6 = False
    var_7 = {var_4: var_6, var_5: var_6}
    var_8 = {var_3: var_7}
    var_9 = {var_2: var_8}
    var_10 = {}
    var_11 = 'above'
    var_12 = 'nested'
    var_13 = 'straight'
    var_14 = {}
    var_15 = {}
    var_16 = {var_2: var_15}
    var_17 = {}
    var_18 = {}
    var_19 = module_1.Config()
    var_20 = [var_3]
    var_21 = []
    var_22 = 'import'
    var_23 = module_2._with_from_imports(var_0, var_19, var_20, var_1, var_21, var_22)
    var_24 = len(var_23)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = 'sep'
    var_6 = False
    var_7 = {var_4: var_6, var_5: var_6}
    var_8 = {var_3: var_7}
    var_9 = {var_2: var_8}
    var_10 = {}
    var_11 = 'above'
    var_12 = 'nested'
    var_13 = 'straight'
    var_14 = {}
    var_15 = {}
    var_16 = {var_2: var_15}
    var_17 = {}
    var_18 = {}
    var_19 = module_1.Config()
    var_20 = [var_3]
    var_21 = [var_3]
    var_22 = 'import'
    var_23 = module_2._with_from_imports(var_0, var_19, var_20, var_1, var_21, var_22)
    var_24 = len(var_23)
    assert var_24 == 0

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = {}
    var_6 = 'above'
    var_7 = 'nested'
    var_8 = 'straight'
    var_9 = {}
    var_10 = {}
    var_11 = {var_2: var_10}
    var_12 = {}
    var_13 = {}
    var_14 = module_1.Config()
    var_15 = []
    var_16 = []
    var_17 = 'import'
    var_18 = module_2._with_from_imports(var_0, var_14, var_15, var_1, var_16, var_17)
    var_19 = len(var_18)
    assert var_19 == 0

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = {}
    var_10 = 'above'
    var_11 = 'nested'
    var_12 = 'straight'
    var_13 = '# important module'
    var_14 = [var_13]
    var_15 = {var_3: var_14}
    var_16 = {}
    var_17 = {var_2: var_16}
    var_18 = {}
    var_19 = {}
    var_20 = module_1.Config()
    var_21 = [var_3]
    var_22 = []
    var_23 = 'import'
    var_24 = module_2._with_from_imports(var_0, var_20, var_21, var_1, var_22, var_23)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = '*'
    var_5 = 'path'
    var_6 = False
    var_7 = {var_4: var_6, var_5: var_6}
    var_8 = {var_3: var_7}
    var_9 = {var_2: var_8}
    var_10 = {}
    var_11 = 'above'
    var_12 = 'nested'
    var_13 = 'straight'
    var_14 = {}
    var_15 = {}
    var_16 = {var_2: var_15}
    var_17 = '# star comment'
    var_18 = {var_4: var_17}
    var_19 = {var_3: var_18}
    var_20 = {}
    var_21 = True
    var_22 = module_1.Config()
    var_23 = [var_3]
    var_24 = []
    var_25 = 'import'
    var_26 = module_2._with_from_imports(var_0, var_22, var_23, var_1, var_24, var_25)



# Parsed testcases at query #50
#--------------------------




def test_case_0():
    var_0 = False
    assert var_0 is False



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_predicate_at_line_178_evaluates_to_true. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_import'
    var_2 = 'nested'
    var_3 = {}
    var_4 = None



# Parsed testcases at query #52
#--------------------------




import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = {}
    var_3 = {}
    var_4 = 'FUTURE'
    var_5 = 'STDLIB'
    var_6 = 'THIRDPARTY'
    var_7 = 'FIRSTPARTY'
    var_8 = 'LOCALFOLDER'
    var_9 = 'straight'
    var_10 = 'from'
    var_11 = {}
    var_12 = {}
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = 'os'
    var_15 = []
    var_16 = {var_14: var_15}
    var_17 = {}
    var_18 = {var_9: var_16, var_10: var_17}
    var_19 = {}
    var_20 = {}
    var_21 = {var_9: var_19, var_10: var_20}
    var_22 = {}
    var_23 = {}
    var_24 = {var_9: var_22, var_10: var_23}
    var_25 = {}
    var_26 = {}
    var_27 = {var_9: var_25, var_10: var_26}
    var_28 = {var_4: var_13, var_5: var_18, var_6: var_21, var_7: var_24, var_8: var_27}
    var_29 = {}
    var_30 = 1
    var_31 = ''
    var_32 = [var_31]
    var_33 = {}
    var_34 = {}
    var_35 = module_0.ParsedContent()
    var_36 = 'stdlib'
    var_37 = 'Standard Library'
    var_38 = {var_36: var_37}
    var_39 = module_1.Config()
    var_40 = 'py'
    var_41 = 'import'
    var_42 = module_2.sorted_imports(var_35, var_39, var_40, var_41)



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_sorted_imports_returns_string. Retrieved 21/24 statements.


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = -1
    var_3 = {}
    var_4 = {}
    var_5 = {}
    var_6 = {}
    var_7 = 0
    var_8 = 'FUTURE'
    var_9 = 'STDLIB'
    var_10 = 'THIRDPARTY'
    var_11 = 'FIRSTPARTY'
    var_12 = 'LOCALFOLDER'
    var_13 = [var_8, var_9, var_10, var_11, var_12]
    var_14 = "print('hello')"
    var_15 = [var_14]
    var_16 = {}
    var_17 = '\n'
    var_18 = module_0.ParsedContent()
    var_19 = module_1.Config()
    var_20 = module_2.sorted_imports(var_18, var_19)



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_predicate_at_line_49_evaluates_to_true. Retrieved 27/34 statements.


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = 'THIRDPARTY'
    var_3 = lambda x: var_2
    var_4 = ''
    var_5 = False
    var_6 = module_0.ParsedContent()
    var_7 = 'straight'
    var_8 = 'test_module'
    var_9 = 'alias1'
    var_10 = 'alias2'
    var_11 = [var_9, var_10]
    var_12 = {var_8: var_11}
    var_13 = []
    var_14 = {var_8: var_13}
    var_15 = {var_7: var_14}
    var_16 = 'above'
    var_17 = {}
    var_18 = {var_7: var_17}
    var_19 = {}
    var_20 = module_1.Config()
    var_21 = [var_8]
    var_22 = 'THIRDPARTY'
    var_23 = []
    var_24 = 'import'
    var_25 = module_2._with_straight_imports(var_6, var_20, var_21, var_22, var_23, var_24)
    var_26 = len(var_25)



# Parsed testcases at query #55
#--------------------------






####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_with_from_imports_empty_from_modules. Retrieved 19/27 statements.
# Partially parsed test_with_from_imports_single_import. Retrieved 24/32 statements.
# Partially parsed test_with_from_imports_remove_imports. Retrieved 24/33 statements.
# Partially parsed test_with_from_imports_skip_removed_module. Retrieved 23/31 statements.
# Partially parsed test_with_from_imports_with_comments. Retrieved 26/34 statements.
# Partially parsed test_with_from_imports_force_single_line. Retrieved 25/34 statements.
# Partially parsed test_with_from_imports_combine_star. Retrieved 24/33 statements.
# Partially parsed test_with_from_imports_with_as_imports. Retrieved 26/35 statements.


import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_1.ParsedContent()
    var_2 = 'STDLIB'
    var_3 = 'from'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'above'
    var_7 = 'nested'
    var_8 = 'straight'
    var_9 = {}
    var_10 = {}
    var_11 = {var_3: var_10}
    var_12 = {}
    var_13 = {}
    var_14 = {}
    var_15 = []
    var_16 = []
    var_17 = 'import'
    var_18 = module_2._with_from_imports(var_1, var_0, var_15, var_2, var_16, var_17)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_1.ParsedContent()
    var_2 = 'STDLIB'
    var_3 = 'from'
    var_4 = 'os'
    var_5 = 'path'
    var_6 = False
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = {var_3: var_8}
    var_10 = 'above'
    var_11 = 'nested'
    var_12 = 'straight'
    var_13 = {}
    var_14 = {}
    var_15 = {var_3: var_14}
    var_16 = {}
    var_17 = {}
    var_18 = {}
    var_19 = [var_4]
    var_20 = []
    var_21 = 'import'
    var_22 = module_2._with_from_imports(var_1, var_0, var_19, var_2, var_20, var_21)
    var_23 = len(var_22)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_1.ParsedContent()
    var_2 = 'STDLIB'
    var_3 = 'from'
    var_4 = 'os'
    var_5 = 'path'
    var_6 = False
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = {var_3: var_8}
    var_10 = 'above'
    var_11 = 'nested'
    var_12 = 'straight'
    var_13 = {}
    var_14 = {}
    var_15 = {var_3: var_14}
    var_16 = {}
    var_17 = {}
    var_18 = {}
    var_19 = [var_4]
    var_20 = 'os.path'
    var_21 = [var_20]
    var_22 = 'import'
    var_23 = module_2._with_from_imports(var_1, var_0, var_19, var_2, var_21, var_22)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_1.ParsedContent()
    var_2 = 'STDLIB'
    var_3 = 'from'
    var_4 = 'os'
    var_5 = 'path'
    var_6 = False
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = {var_3: var_8}
    var_10 = 'above'
    var_11 = 'nested'
    var_12 = 'straight'
    var_13 = {}
    var_14 = {}
    var_15 = {var_3: var_14}
    var_16 = {}
    var_17 = {}
    var_18 = {}
    var_19 = [var_4]
    var_20 = [var_4]
    var_21 = 'import'
    var_22 = module_2._with_from_imports(var_1, var_0, var_19, var_2, var_20, var_21)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_1.ParsedContent()
    var_2 = 'STDLIB'
    var_3 = 'from'
    var_4 = 'os'
    var_5 = 'path'
    var_6 = False
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = {var_3: var_8}
    var_10 = 'above'
    var_11 = 'nested'
    var_12 = 'straight'
    var_13 = '# important'
    var_14 = [var_13]
    var_15 = {var_4: var_14}
    var_16 = {}
    var_17 = {var_3: var_16}
    var_18 = {}
    var_19 = {}
    var_20 = {}
    var_21 = [var_4]
    var_22 = []
    var_23 = 'import'
    var_24 = module_2._with_from_imports(var_1, var_0, var_21, var_2, var_22, var_23)
    var_25 = len(var_24)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = module_1.ParsedContent()
    var_3 = 'STDLIB'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = 'path'
    var_7 = 'environ'
    var_8 = False
    var_9 = {var_6: var_8, var_7: var_8}
    var_10 = {var_5: var_9}
    var_11 = {var_4: var_10}
    var_12 = 'above'
    var_13 = 'nested'
    var_14 = 'straight'
    var_15 = {}
    var_16 = {}
    var_17 = {var_4: var_16}
    var_18 = {}
    var_19 = {}
    var_20 = {}
    var_21 = [var_5]
    var_22 = []
    var_23 = 'import'
    var_24 = module_2._with_from_imports(var_2, var_1, var_21, var_3, var_22, var_23)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = module_1.ParsedContent()
    var_3 = 'STDLIB'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = '*'
    var_7 = False
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = {var_4: var_9}
    var_11 = 'above'
    var_12 = 'nested'
    var_13 = 'straight'
    var_14 = {}
    var_15 = {}
    var_16 = {var_4: var_15}
    var_17 = {}
    var_18 = {}
    var_19 = {}
    var_20 = [var_5]
    var_21 = []
    var_22 = 'import'
    var_23 = module_2._with_from_imports(var_2, var_1, var_20, var_3, var_21, var_22)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_1.ParsedContent()
    var_2 = 'STDLIB'
    var_3 = 'from'
    var_4 = 'os'
    var_5 = 'path'
    var_6 = False
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = {var_3: var_8}
    var_10 = 'above'
    var_11 = 'nested'
    var_12 = 'straight'
    var_13 = {}
    var_14 = {}
    var_15 = {var_3: var_14}
    var_16 = {}
    var_17 = {}
    var_18 = 'os.path'
    var_19 = 'p'
    var_20 = [var_19]
    var_21 = {var_18: var_20}
    var_22 = [var_4]
    var_23 = []
    var_24 = 'import'
    var_25 = module_2._with_from_imports(var_1, var_0, var_22, var_2, var_23, var_24)



# Parsed testcases at query #2
#--------------------------




def test_case_0():
    var_0 = []
    var_1 = list(var_0)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_with_star_comments_with_star_comment. Retrieved 9/14 statements.
# Partially parsed test_with_star_comments_without_star_comment. Retrieved 7/12 statements.
# Partially parsed test_with_star_comments_module_not_found. Retrieved 6/11 statements.
# Partially parsed test_with_star_comments_empty_comments_list. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 'nested'
    var_1 = 'test_module'
    var_2 = '*'
    var_3 = 'star comment text'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'comment1'
    var_7 = 'comment2'
    var_8 = [var_6, var_7]

def test_case_0():
    var_0 = 'nested'
    var_1 = 'test_module'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 'comment1'
    var_5 = 'comment2'
    var_6 = [var_4, var_5]

def test_case_0():
    var_0 = 'nested'
    var_1 = {}
    var_2 = 'comment1'
    var_3 = 'comment2'
    var_4 = [var_2, var_3]
    var_5 = 'nonexistent_module'

def test_case_0():
    var_0 = 'nested'
    var_1 = 'test_module'
    var_2 = '*'
    var_3 = 'star comment text'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = []



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_sorted_imports_normalizes_empty_lines. Retrieved 22/26 statements.
# Partially parsed test_sorted_imports_with_no_sections. Retrieved 33/37 statements.
# Partially parsed test_sorted_imports_with_place_imports. Retrieved 30/34 statements.
# Partially parsed test_sorted_imports_preserves_line_separator. Retrieved 22/26 statements.


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = -1
    var_1 = "print('hello')\n"
    var_2 = [var_1]
    var_3 = '\n'
    var_4 = []
    var_5 = {}
    var_6 = {}
    var_7 = 'straight'
    var_8 = 'from'
    var_9 = {}
    var_10 = {}
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_7: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_7: var_16}
    var_18 = module_0.ParsedContent()
    var_19 = module_1.Config()
    var_20 = module_2.sorted_imports(var_18, var_19)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = "print('hello')\n"
    var_2 = [var_1]
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = [var_4]
    var_6 = {}
    var_7 = {}
    var_8 = 'straight'
    var_9 = 'from'
    var_10 = {}
    var_11 = {}
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = 'os'
    var_14 = None
    var_15 = {var_13: var_14}
    var_16 = {}
    var_17 = {var_8: var_15, var_9: var_16}
    var_18 = {var_4: var_17}
    var_19 = 'above'
    var_20 = {}
    var_21 = {var_8: var_20}
    var_22 = {}
    var_23 = {var_19: var_21, var_8: var_22}
    var_24 = module_0.ParsedContent()
    var_25 = module_1.Config()
    var_26 = module_2.sorted_imports(var_24, var_25)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = -1
    var_1 = ''
    var_2 = 'code\n'
    var_3 = [var_1, var_1, var_2]
    var_4 = '\n'
    var_5 = []
    var_6 = {}
    var_7 = {}
    var_8 = 'straight'
    var_9 = 'from'
    var_10 = {}
    var_11 = {}
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = {}
    var_14 = 'above'
    var_15 = {}
    var_16 = {var_8: var_15}
    var_17 = {}
    var_18 = {var_14: var_16, var_8: var_17}
    var_19 = module_0.ParsedContent()
    var_20 = module_1.Config()
    var_21 = module_2.sorted_imports(var_19, var_20)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'code\n'
    var_2 = [var_1]
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = [var_4]
    var_6 = {}
    var_7 = {}
    var_8 = 'straight'
    var_9 = 'from'
    var_10 = {}
    var_11 = {}
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = {}
    var_14 = 'os'
    var_15 = 'path'
    var_16 = [var_15]
    var_17 = {var_14: var_16}
    var_18 = {var_8: var_13, var_9: var_17}
    var_19 = {var_4: var_18}
    var_20 = 'above'
    var_21 = {}
    var_22 = {var_8: var_21}
    var_23 = {}
    var_24 = {var_20: var_22, var_8: var_23}
    var_25 = module_0.ParsedContent()
    var_26 = module_1.Config()
    var_27 = module_2.sorted_imports(var_25, var_26)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'code\n'
    var_2 = [var_1]
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = [var_4]
    var_6 = {}
    var_7 = {}
    var_8 = 'straight'
    var_9 = 'from'
    var_10 = {}
    var_11 = {}
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = 'os'
    var_14 = None
    var_15 = {var_13: var_14}
    var_16 = {}
    var_17 = {var_8: var_15, var_9: var_16}
    var_18 = {var_4: var_17}
    var_19 = 'above'
    var_20 = {}
    var_21 = {var_8: var_20}
    var_22 = {}
    var_23 = {var_19: var_21, var_8: var_22}
    var_24 = module_0.ParsedContent()
    var_25 = 'import os'
    var_26 = [var_25]
    var_27 = module_1.Config()
    var_28 = module_2.sorted_imports(var_24, var_27)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'code\n'
    var_2 = [var_1]
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'THIRDPARTY'
    var_6 = [var_4, var_5]
    var_7 = {}
    var_8 = {}
    var_9 = 'straight'
    var_10 = 'from'
    var_11 = {}
    var_12 = {}
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = 'os'
    var_15 = None
    var_16 = {var_14: var_15}
    var_17 = {}
    var_18 = {var_9: var_16, var_10: var_17}
    var_19 = 'requests'
    var_20 = {var_19: var_15}
    var_21 = {}
    var_22 = {var_9: var_20, var_10: var_21}
    var_23 = {var_4: var_18, var_5: var_22}
    var_24 = 'above'
    var_25 = {}
    var_26 = {var_9: var_25}
    var_27 = {}
    var_28 = {var_24: var_26, var_9: var_27}
    var_29 = module_0.ParsedContent()
    var_30 = True
    var_31 = module_1.Config()
    var_32 = module_2.sorted_imports(var_29, var_31)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'code\n'
    var_2 = [var_1]
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = [var_4]
    var_6 = {}
    var_7 = {}
    var_8 = 'straight'
    var_9 = 'from'
    var_10 = {}
    var_11 = {}
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = 'os'
    var_14 = None
    var_15 = {var_13: var_14}
    var_16 = {}
    var_17 = {var_8: var_15, var_9: var_16}
    var_18 = {var_4: var_17}
    var_19 = 'above'
    var_20 = {}
    var_21 = {var_8: var_20}
    var_22 = {}
    var_23 = {var_19: var_21, var_8: var_22}
    var_24 = module_0.ParsedContent()
    var_25 = 'stdlib'
    var_26 = 'Standard Library'
    var_27 = {var_25: var_26}
    var_28 = module_1.Config()
    var_29 = module_2.sorted_imports(var_24, var_28)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = '# isort: split\n'
    var_2 = 'code\n'
    var_3 = [var_1, var_2]
    var_4 = '\n'
    var_5 = 'STDLIB'
    var_6 = [var_5]
    var_7 = 'import os'
    var_8 = [var_7]
    var_9 = {var_5: var_8}
    var_10 = {var_1: var_5}
    var_11 = 'straight'
    var_12 = 'from'
    var_13 = {}
    var_14 = {}
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = 'os'
    var_17 = None
    var_18 = {var_16: var_17}
    var_19 = {}
    var_20 = {var_11: var_18, var_12: var_19}
    var_21 = {var_5: var_20}
    var_22 = 'above'
    var_23 = {}
    var_24 = {var_11: var_23}
    var_25 = {}
    var_26 = {var_22: var_24, var_11: var_25}
    var_27 = module_0.ParsedContent()
    var_28 = module_1.Config()
    var_29 = module_2.sorted_imports(var_27, var_28)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = -1
    var_1 = 'code'
    var_2 = [var_1]
    var_3 = '\r\n'
    var_4 = []
    var_5 = {}
    var_6 = {}
    var_7 = 'straight'
    var_8 = 'from'
    var_9 = {}
    var_10 = {}
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_7: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_7: var_16}
    var_18 = module_0.ParsedContent()
    var_19 = module_1.Config()
    var_20 = module_2.sorted_imports(var_18, var_19)
    var_21 = '\n'



# Parsed testcases at query #5
#--------------------------




import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = {}
    var_2 = 'from'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'above'
    var_6 = 'nested'
    var_7 = 'straight'
    var_8 = {}
    var_9 = {}
    var_10 = {var_2: var_9}
    var_11 = {}
    var_12 = {}
    var_13 = {var_2: var_8, var_5: var_10, var_6: var_11, var_7: var_12}
    var_14 = '\n'
    var_15 = set()
    var_16 = module_1.ParsedContent()
    var_17 = []
    var_18 = 'STDLIB'
    var_19 = []
    var_20 = 'import'
    var_21 = module_2._with_from_imports(var_16, var_0, var_17, var_18, var_19, var_20)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'STDLIB'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = {var_1: var_8}
    var_10 = {}
    var_11 = {var_2: var_10}
    var_12 = 'above'
    var_13 = 'nested'
    var_14 = 'straight'
    var_15 = {}
    var_16 = {}
    var_17 = {var_2: var_16}
    var_18 = {}
    var_19 = {}
    var_20 = {var_2: var_15, var_12: var_17, var_13: var_18, var_14: var_19}
    var_21 = '\n'
    var_22 = set()
    var_23 = module_1.ParsedContent()
    var_24 = [var_3]
    var_25 = []
    var_26 = 'import'
    var_27 = module_2._with_from_imports(var_23, var_0, var_24, var_1, var_25, var_26)
    var_28 = len(var_27)
    assert var_28 == 1

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'STDLIB'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = {var_1: var_8}
    var_10 = {}
    var_11 = {var_2: var_10}
    var_12 = 'above'
    var_13 = 'nested'
    var_14 = 'straight'
    var_15 = {}
    var_16 = {}
    var_17 = {var_2: var_16}
    var_18 = {}
    var_19 = {}
    var_20 = {var_2: var_15, var_12: var_17, var_13: var_18, var_14: var_19}
    var_21 = '\n'
    var_22 = set()
    var_23 = module_1.ParsedContent()
    var_24 = [var_3]
    var_25 = [var_3]
    var_26 = 'import'
    var_27 = module_2._with_from_imports(var_23, var_0, var_24, var_1, var_25, var_26)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'STDLIB'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = 'environ'
    var_6 = False
    var_7 = {var_4: var_6, var_5: var_6}
    var_8 = {var_3: var_7}
    var_9 = {var_2: var_8}
    var_10 = {var_1: var_9}
    var_11 = {}
    var_12 = {var_2: var_11}
    var_13 = 'above'
    var_14 = 'nested'
    var_15 = 'straight'
    var_16 = {}
    var_17 = {}
    var_18 = {var_2: var_17}
    var_19 = {}
    var_20 = {}
    var_21 = {var_2: var_16, var_13: var_18, var_14: var_19, var_15: var_20}
    var_22 = '\n'
    var_23 = set()
    var_24 = module_1.ParsedContent()
    var_25 = [var_3]
    var_26 = []
    var_27 = 'import'
    var_28 = module_2._with_from_imports(var_24, var_0, var_25, var_1, var_26, var_27)
    var_29 = len(var_28)
    assert var_29 == 1

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'STDLIB'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = {var_1: var_8}
    var_10 = {}
    var_11 = {var_2: var_10}
    var_12 = 'above'
    var_13 = 'nested'
    var_14 = 'straight'
    var_15 = 'important comment'
    var_16 = [var_15]
    var_17 = {var_3: var_16}
    var_18 = {}
    var_19 = {var_2: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = {var_2: var_17, var_12: var_19, var_13: var_20, var_14: var_21}
    var_23 = '\n'
    var_24 = set()
    var_25 = module_1.ParsedContent()
    var_26 = [var_3]
    var_27 = []
    var_28 = 'import'
    var_29 = module_2._with_from_imports(var_25, var_0, var_26, var_1, var_27, var_28)
    var_30 = len(var_29)
    assert var_30 == 1

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = 'STDLIB'
    var_3 = 'from'
    var_4 = 'os'
    var_5 = 'path'
    var_6 = 'environ'
    var_7 = False
    var_8 = {var_5: var_7, var_6: var_7}
    var_9 = {var_4: var_8}
    var_10 = {var_3: var_9}
    var_11 = {var_2: var_10}
    var_12 = {}
    var_13 = {var_3: var_12}
    var_14 = 'above'
    var_15 = 'nested'
    var_16 = 'straight'
    var_17 = {}
    var_18 = {}
    var_19 = {var_3: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = {var_3: var_17, var_14: var_19, var_15: var_20, var_16: var_21}
    var_23 = '\n'
    var_24 = set()
    var_25 = module_1.ParsedContent()
    var_26 = [var_4]
    var_27 = []
    var_28 = 'import'
    var_29 = module_2._with_from_imports(var_25, var_1, var_26, var_2, var_27, var_28)
    var_30 = len(var_29)
    assert var_30 == 2

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'STDLIB'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = '*'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = {var_1: var_8}
    var_10 = {}
    var_11 = {var_2: var_10}
    var_12 = 'above'
    var_13 = 'nested'
    var_14 = 'straight'
    var_15 = {}
    var_16 = {}
    var_17 = {var_2: var_16}
    var_18 = {}
    var_19 = {}
    var_20 = {var_2: var_15, var_12: var_17, var_13: var_18, var_14: var_19}
    var_21 = '\n'
    var_22 = set()
    var_23 = module_1.ParsedContent()
    var_24 = [var_3]
    var_25 = []
    var_26 = 'import'
    var_27 = module_2._with_from_imports(var_23, var_0, var_24, var_1, var_25, var_26)
    var_28 = len(var_27)
    assert var_28 == 1



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_with_from_imports_basic. Retrieved 29/38 statements.
# Partially parsed test_with_from_imports_with_star. Retrieved 27/36 statements.
# Partially parsed test_with_from_imports_remove_imports. Retrieved 22/30 statements.
# Partially parsed test_with_from_imports_with_comments. Retrieved 27/36 statements.
# Partially parsed test_with_from_imports_force_single_line. Retrieved 27/36 statements.
# Partially parsed test_with_from_imports_multiple_modules. Retrieved 28/37 statements.


import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_1.ParsedContent()
    var_2 = 'FUTURE'
    var_3 = 'STDLIB'
    var_4 = 'from'
    var_5 = 'straight'
    var_6 = {}
    var_7 = {}
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = 'os'
    var_10 = 'path'
    var_11 = None
    var_12 = {var_10: var_11}
    var_13 = {var_9: var_12}
    var_14 = {}
    var_15 = {var_4: var_13, var_5: var_14}
    var_16 = {}
    var_17 = {}
    var_18 = 'nested'
    var_19 = 'above'
    var_20 = {}
    var_21 = {}
    var_22 = {}
    var_23 = {}
    var_24 = {var_4: var_23}
    var_25 = [var_9]
    var_26 = []
    var_27 = 'import'
    var_28 = module_2._with_from_imports(var_1, var_0, var_25, var_3, var_26, var_27)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_1.ParsedContent()
    var_2 = 'STDLIB'
    var_3 = 'from'
    var_4 = 'straight'
    var_5 = 'module'
    var_6 = '*'
    var_7 = None
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = {}
    var_11 = {var_3: var_9, var_4: var_10}
    var_12 = {}
    var_13 = {}
    var_14 = 'nested'
    var_15 = 'above'
    var_16 = []
    var_17 = {var_5: var_16}
    var_18 = {}
    var_19 = {}
    var_20 = {var_5: var_19}
    var_21 = {}
    var_22 = {var_3: var_21}
    var_23 = [var_5]
    var_24 = []
    var_25 = 'import'
    var_26 = module_2._with_from_imports(var_1, var_0, var_23, var_2, var_24, var_25)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_1.ParsedContent()
    var_2 = 'STDLIB'
    var_3 = 'from'
    var_4 = 'straight'
    var_5 = {}
    var_6 = {}
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {}
    var_9 = {}
    var_10 = 'nested'
    var_11 = 'above'
    var_12 = {}
    var_13 = {}
    var_14 = {}
    var_15 = {}
    var_16 = {var_3: var_15}
    var_17 = []
    var_18 = 'os.path'
    var_19 = [var_18]
    var_20 = 'import'
    var_21 = module_2._with_from_imports(var_1, var_0, var_17, var_2, var_19, var_20)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_1.ParsedContent()
    var_2 = 'STDLIB'
    var_3 = 'from'
    var_4 = 'straight'
    var_5 = 'os'
    var_6 = 'path'
    var_7 = None
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = {}
    var_11 = {var_3: var_9, var_4: var_10}
    var_12 = {}
    var_13 = {}
    var_14 = 'nested'
    var_15 = 'above'
    var_16 = 'important comment'
    var_17 = [var_16]
    var_18 = {var_5: var_17}
    var_19 = {}
    var_20 = {}
    var_21 = {}
    var_22 = {var_3: var_21}
    var_23 = [var_5]
    var_24 = []
    var_25 = 'import'
    var_26 = module_2._with_from_imports(var_1, var_0, var_23, var_2, var_24, var_25)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = module_1.ParsedContent()
    var_3 = 'STDLIB'
    var_4 = 'from'
    var_5 = 'straight'
    var_6 = 'os'
    var_7 = 'path'
    var_8 = 'getcwd'
    var_9 = None
    var_10 = {var_7: var_9, var_8: var_9}
    var_11 = {var_6: var_10}
    var_12 = {}
    var_13 = {var_4: var_11, var_5: var_12}
    var_14 = {}
    var_15 = {}
    var_16 = 'nested'
    var_17 = 'above'
    var_18 = {}
    var_19 = {}
    var_20 = {}
    var_21 = {}
    var_22 = {var_4: var_21}
    var_23 = [var_6]
    var_24 = []
    var_25 = 'import'
    var_26 = module_2._with_from_imports(var_2, var_1, var_23, var_3, var_24, var_25)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_1.ParsedContent()
    var_2 = 'STDLIB'
    var_3 = 'from'
    var_4 = 'straight'
    var_5 = 'os'
    var_6 = 'sys'
    var_7 = 'path'
    var_8 = None
    var_9 = {var_7: var_8}
    var_10 = 'argv'
    var_11 = {var_10: var_8}
    var_12 = {var_5: var_9, var_6: var_11}
    var_13 = {}
    var_14 = {var_3: var_12, var_4: var_13}
    var_15 = {}
    var_16 = {}
    var_17 = 'nested'
    var_18 = 'above'
    var_19 = {}
    var_20 = {}
    var_21 = {}
    var_22 = {}
    var_23 = {var_3: var_22}
    var_24 = [var_5, var_6]
    var_25 = []
    var_26 = 'import'
    var_27 = module_2._with_from_imports(var_1, var_0, var_24, var_2, var_25, var_26)



# Parsed testcases at query #7
#--------------------------




import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = {}
    var_1 = 'from'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 'above'
    var_5 = 'nested'
    var_6 = 'straight'
    var_7 = {}
    var_8 = {}
    var_9 = {var_1: var_8}
    var_10 = {}
    var_11 = {}
    var_12 = {var_1: var_7, var_4: var_9, var_5: var_10, var_6: var_11}
    var_13 = 0
    var_14 = ''
    var_15 = {}
    var_16 = {}
    var_17 = []
    var_18 = False
    var_19 = False
    var_20 = []
    var_21 = '\n'
    var_22 = {}
    var_23 = module_0.ParsedContent()
    var_24 = module_1.Config()
    var_25 = []
    var_26 = 'FUTURE'
    var_27 = []
    var_28 = 'import'
    var_29 = module_2._with_from_imports(var_23, var_24, var_25, var_26, var_27, var_28)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'FUTURE'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = False
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = {}
    var_10 = {var_1: var_9}
    var_11 = 'above'
    var_12 = 'nested'
    var_13 = 'straight'
    var_14 = {}
    var_15 = {}
    var_16 = {var_1: var_15}
    var_17 = {}
    var_18 = {}
    var_19 = {var_1: var_14, var_11: var_16, var_12: var_17, var_13: var_18}
    var_20 = ''
    var_21 = {}
    var_22 = {}
    var_23 = []
    var_24 = []
    var_25 = '\n'
    var_26 = {}
    var_27 = module_0.ParsedContent()
    var_28 = module_1.Config()
    var_29 = [var_2]
    var_30 = [var_2]
    var_31 = 'import'
    var_32 = module_2._with_from_imports(var_27, var_28, var_29, var_0, var_30, var_31)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'FUTURE'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = False
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = {}
    var_10 = {var_1: var_9}
    var_11 = 'above'
    var_12 = 'nested'
    var_13 = 'straight'
    var_14 = {}
    var_15 = {}
    var_16 = {var_1: var_15}
    var_17 = {}
    var_18 = {}
    var_19 = {var_1: var_14, var_11: var_16, var_12: var_17, var_13: var_18}
    var_20 = ''
    var_21 = {}
    var_22 = {}
    var_23 = []
    var_24 = []
    var_25 = '\n'
    var_26 = {}
    var_27 = module_0.ParsedContent()
    var_28 = module_1.Config()
    var_29 = [var_2]
    var_30 = []
    var_31 = 'import'
    var_32 = module_2._with_from_imports(var_27, var_28, var_29, var_0, var_30, var_31)
    var_33 = len(var_32)
    assert var_33 == 1

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'FUTURE'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = '*'
    var_4 = False
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = {}
    var_10 = {var_1: var_9}
    var_11 = 'above'
    var_12 = 'nested'
    var_13 = 'straight'
    var_14 = {}
    var_15 = {}
    var_16 = {var_1: var_15}
    var_17 = {}
    var_18 = {var_2: var_17}
    var_19 = {}
    var_20 = {var_1: var_14, var_11: var_16, var_12: var_18, var_13: var_19}
    var_21 = ''
    var_22 = {}
    var_23 = {}
    var_24 = []
    var_25 = []
    var_26 = '\n'
    var_27 = {}
    var_28 = module_0.ParsedContent()
    var_29 = True
    var_30 = module_1.Config()
    var_31 = [var_2]
    var_32 = []
    var_33 = 'import'
    var_34 = module_2._with_from_imports(var_28, var_30, var_31, var_0, var_32, var_33)
    var_35 = len(var_34)
    assert var_35 == 1

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'FUTURE'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'environ'
    var_5 = False
    var_6 = {var_3: var_5, var_4: var_5}
    var_7 = {var_2: var_6}
    var_8 = {var_1: var_7}
    var_9 = {var_0: var_8}
    var_10 = {}
    var_11 = {var_1: var_10}
    var_12 = 'above'
    var_13 = 'nested'
    var_14 = 'straight'
    var_15 = {}
    var_16 = {}
    var_17 = {var_1: var_16}
    var_18 = {}
    var_19 = {}
    var_20 = {var_1: var_15, var_12: var_17, var_13: var_18, var_14: var_19}
    var_21 = ''
    var_22 = {}
    var_23 = {}
    var_24 = []
    var_25 = []
    var_26 = '\n'
    var_27 = {}
    var_28 = module_0.ParsedContent()
    var_29 = True
    var_30 = module_1.Config()
    var_31 = [var_2]
    var_32 = []
    var_33 = 'import'
    var_34 = module_2._with_from_imports(var_28, var_30, var_31, var_0, var_32, var_33)
    var_35 = len(var_34)
    assert var_35 == 2

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'FUTURE'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = True
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'os.path'
    var_10 = 'p'
    var_11 = [var_10]
    var_12 = {var_9: var_11}
    var_13 = {var_1: var_12}
    var_14 = 'above'
    var_15 = 'nested'
    var_16 = 'straight'
    var_17 = {}
    var_18 = {}
    var_19 = {var_1: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = {var_1: var_17, var_14: var_19, var_15: var_20, var_16: var_21}
    var_23 = 0
    var_24 = ''
    var_25 = {}
    var_26 = {}
    var_27 = []
    var_28 = False
    var_29 = False
    var_30 = []
    var_31 = '\n'
    var_32 = {}
    var_33 = module_0.ParsedContent()
    var_34 = module_1.Config()
    var_35 = [var_2]
    var_36 = []
    var_37 = 'import'
    var_38 = module_2._with_from_imports(var_33, var_34, var_35, var_0, var_36, var_37)
    var_39 = len(var_38)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'FUTURE'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = False
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = {}
    var_10 = {var_1: var_9}
    var_11 = 'above'
    var_12 = 'nested'
    var_13 = 'straight'
    var_14 = 'important comment'
    var_15 = [var_14]
    var_16 = {var_2: var_15}
    var_17 = {}
    var_18 = {var_1: var_17}
    var_19 = {}
    var_20 = {}
    var_21 = {var_1: var_16, var_11: var_18, var_12: var_19, var_13: var_20}
    var_22 = ''
    var_23 = {}
    var_24 = {}
    var_25 = []
    var_26 = []
    var_27 = '\n'
    var_28 = {}
    var_29 = module_0.ParsedContent()
    var_30 = module_1.Config()
    var_31 = [var_2]
    var_32 = []
    var_33 = 'import'
    var_34 = module_2._with_from_imports(var_29, var_30, var_31, var_0, var_32, var_33)
    var_35 = len(var_34)
    assert var_35 == 1

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'FUTURE'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'sys'
    var_4 = 'path'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = 'argv'
    var_8 = {var_7: var_5}
    var_9 = {var_2: var_6, var_3: var_8}
    var_10 = {var_1: var_9}
    var_11 = {var_0: var_10}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = 'above'
    var_15 = 'nested'
    var_16 = 'straight'
    var_17 = {}
    var_18 = {}
    var_19 = {var_1: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = {var_1: var_17, var_14: var_19, var_15: var_20, var_16: var_21}
    var_23 = ''
    var_24 = {}
    var_25 = {}
    var_26 = []
    var_27 = []
    var_28 = '\n'
    var_29 = {}
    var_30 = module_0.ParsedContent()
    var_31 = module_1.Config()
    var_32 = [var_2, var_3]
    var_33 = []
    var_34 = 'import'
    var_35 = module_2._with_from_imports(var_30, var_31, var_32, var_0, var_33, var_34)
    var_36 = len(var_35)
    assert var_36 == 2



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_with_from_imports_single_import. Retrieved 25/34 statements.
# Partially parsed test_with_from_imports_with_remove_imports. Retrieved 24/32 statements.
# Partially parsed test_with_from_imports_multiple_modules. Retrieved 28/37 statements.
# Partially parsed test_with_from_imports_with_star_import. Retrieved 24/34 statements.
# Partially parsed test_with_from_imports_force_single_line. Retrieved 26/37 statements.


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = module_1.Config()
    var_2 = []
    var_3 = 'THIRDPARTY'
    var_4 = []
    var_5 = 'import'
    var_6 = module_2._with_from_imports(var_0, var_1, var_2, var_3, var_4, var_5)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = {}
    var_10 = 'above'
    var_11 = 'nested'
    var_12 = 'straight'
    var_13 = {}
    var_14 = {}
    var_15 = {var_2: var_14}
    var_16 = {}
    var_17 = {}
    var_18 = module_1.Config()
    var_19 = [var_3]
    var_20 = 'THIRDPARTY'
    var_21 = []
    var_22 = 'import'
    var_23 = module_2._with_from_imports(var_0, var_18, var_19, var_20, var_21, var_22)
    var_24 = len(var_23)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = {}
    var_10 = 'above'
    var_11 = 'nested'
    var_12 = 'straight'
    var_13 = {}
    var_14 = {}
    var_15 = {var_2: var_14}
    var_16 = {}
    var_17 = {}
    var_18 = module_1.Config()
    var_19 = [var_3]
    var_20 = 'THIRDPARTY'
    var_21 = [var_3]
    var_22 = 'import'
    var_23 = module_2._with_from_imports(var_0, var_18, var_19, var_20, var_21, var_22)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'sys'
    var_5 = 'path'
    var_6 = False
    var_7 = {var_5: var_6}
    var_8 = 'argv'
    var_9 = {var_8: var_6}
    var_10 = {var_3: var_7, var_4: var_9}
    var_11 = {var_2: var_10}
    var_12 = {}
    var_13 = 'above'
    var_14 = 'nested'
    var_15 = 'straight'
    var_16 = {}
    var_17 = {}
    var_18 = {var_2: var_17}
    var_19 = {}
    var_20 = {}
    var_21 = module_1.Config()
    var_22 = [var_3, var_4]
    var_23 = 'THIRDPARTY'
    var_24 = []
    var_25 = 'import'
    var_26 = module_2._with_from_imports(var_0, var_21, var_22, var_23, var_24, var_25)
    var_27 = len(var_26)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = '*'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = {}
    var_10 = 'above'
    var_11 = 'nested'
    var_12 = 'straight'
    var_13 = {}
    var_14 = {}
    var_15 = {var_2: var_14}
    var_16 = {}
    var_17 = {}
    var_18 = module_1.Config()
    var_19 = [var_3]
    var_20 = 'THIRDPARTY'
    var_21 = []
    var_22 = 'import'
    var_23 = module_2._with_from_imports(var_0, var_18, var_19, var_20, var_21, var_22)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = 'getcwd'
    var_6 = False
    var_7 = {var_4: var_6, var_5: var_6}
    var_8 = {var_3: var_7}
    var_9 = {var_2: var_8}
    var_10 = {}
    var_11 = 'above'
    var_12 = 'nested'
    var_13 = 'straight'
    var_14 = {}
    var_15 = {}
    var_16 = {var_2: var_15}
    var_17 = {}
    var_18 = {}
    var_19 = module_1.Config()
    var_20 = [var_3]
    var_21 = 'THIRDPARTY'
    var_22 = []
    var_23 = 'import'
    var_24 = module_2._with_from_imports(var_0, var_19, var_20, var_21, var_22, var_23)
    var_25 = len(var_24)
    assert var_25 == 2



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_with_from_imports_basic. Retrieved 21/46 statements.
# Partially parsed test_with_from_imports_with_remove_imports. Retrieved 21/47 statements.
# Partially parsed test_with_from_imports_empty_from_modules. Retrieved 16/40 statements.
# Partially parsed test_with_from_imports_force_single_line. Retrieved 21/46 statements.


def test_case_0():
    var_0 = 'THIRDPARTY'
    var_1 = 'from'
    var_2 = 'module1'
    var_3 = 'func1'
    var_4 = 'func2'
    var_5 = False
    var_6 = {var_3: var_5, var_4: var_5}
    var_7 = {var_2: var_6}
    var_8 = {var_1: var_7}
    var_9 = {}
    var_10 = 'nested'
    var_11 = 'above'
    var_12 = 'straight'
    var_13 = {}
    var_14 = {}
    var_15 = {}
    var_16 = {var_1: var_15}
    var_17 = {}
    var_18 = [var_2]
    var_19 = []
    var_20 = 'import'

def test_case_0():
    var_0 = 'THIRDPARTY'
    var_1 = 'from'
    var_2 = 'module1'
    var_3 = 'func1'
    var_4 = 'func2'
    var_5 = False
    var_6 = {var_3: var_5, var_4: var_5}
    var_7 = {var_2: var_6}
    var_8 = {var_1: var_7}
    var_9 = {}
    var_10 = 'nested'
    var_11 = 'above'
    var_12 = 'straight'
    var_13 = {}
    var_14 = {}
    var_15 = {}
    var_16 = {var_1: var_15}
    var_17 = {}
    var_18 = [var_2]
    var_19 = [var_2]
    var_20 = 'import'

def test_case_0():
    var_0 = 'THIRDPARTY'
    var_1 = 'from'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = 'nested'
    var_6 = 'above'
    var_7 = 'straight'
    var_8 = {}
    var_9 = {}
    var_10 = {}
    var_11 = {var_1: var_10}
    var_12 = {}
    var_13 = []
    var_14 = []
    var_15 = 'import'

def test_case_0():
    var_0 = 'THIRDPARTY'
    var_1 = 'from'
    var_2 = 'module1'
    var_3 = 'func1'
    var_4 = 'func2'
    var_5 = False
    var_6 = {var_3: var_5, var_4: var_5}
    var_7 = {var_2: var_6}
    var_8 = {var_1: var_7}
    var_9 = {}
    var_10 = 'nested'
    var_11 = 'above'
    var_12 = 'straight'
    var_13 = {}
    var_14 = {}
    var_15 = {}
    var_16 = {var_1: var_15}
    var_17 = {}
    var_18 = [var_2]
    var_19 = []
    var_20 = 'import'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_sorted_imports_empty_output. Retrieved 40/44 statements.
# Partially parsed test_sorted_imports_with_from_imports. Retrieved 47/51 statements.


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = -1
    var_1 = 'x = 1'
    var_2 = 'y = 2'
    var_3 = [var_1, var_2]
    var_4 = '\n'
    var_5 = 'FUTURE'
    var_6 = 'STDLIB'
    var_7 = 'THIRDPARTY'
    var_8 = 'FIRSTPARTY'
    var_9 = 'LOCALFOLDER'
    var_10 = [var_5, var_6, var_7, var_8, var_9]
    var_11 = {}
    var_12 = {}
    var_13 = {}
    var_14 = {}
    var_15 = {}
    var_16 = 2
    var_17 = module_0.ParsedContent()
    var_18 = module_1.Config()
    var_19 = module_2.sorted_imports(var_17, var_18)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'x = 1'
    var_2 = [var_1]
    var_3 = '\n'
    var_4 = 'FUTURE'
    var_5 = 'STDLIB'
    var_6 = 'THIRDPARTY'
    var_7 = 'FIRSTPARTY'
    var_8 = 'LOCALFOLDER'
    var_9 = [var_4, var_5, var_6, var_7, var_8]
    var_10 = 'straight'
    var_11 = 'from'
    var_12 = {}
    var_13 = {}
    var_14 = {var_10: var_12, var_11: var_13}
    var_15 = 'os'
    var_16 = None
    var_17 = {var_15: var_16}
    var_18 = {}
    var_19 = {var_10: var_17, var_11: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = {var_10: var_20, var_11: var_21}
    var_23 = {}
    var_24 = {}
    var_25 = {var_10: var_23, var_11: var_24}
    var_26 = {}
    var_27 = {}
    var_28 = {var_10: var_26, var_11: var_27}
    var_29 = {var_4: var_14, var_5: var_19, var_6: var_22, var_7: var_25, var_8: var_28}
    var_30 = {}
    var_31 = {}
    var_32 = {var_10: var_30, var_11: var_31}
    var_33 = 'above'
    var_34 = {}
    var_35 = {var_10: var_34}
    var_36 = {}
    var_37 = {var_33: var_35, var_10: var_36}
    var_38 = {}
    var_39 = {}
    var_40 = 1
    var_41 = module_0.ParsedContent()
    var_42 = module_1.Config()
    var_43 = module_2.sorted_imports(var_41, var_42)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = []
    var_2 = '\n'
    var_3 = 'FUTURE'
    var_4 = 'STDLIB'
    var_5 = 'THIRDPARTY'
    var_6 = 'FIRSTPARTY'
    var_7 = 'LOCALFOLDER'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = 'straight'
    var_10 = 'from'
    var_11 = {}
    var_12 = {}
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = {}
    var_15 = {}
    var_16 = {var_9: var_14, var_10: var_15}
    var_17 = {}
    var_18 = {}
    var_19 = {var_9: var_17, var_10: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = {var_9: var_20, var_10: var_21}
    var_23 = {}
    var_24 = {}
    var_25 = {var_9: var_23, var_10: var_24}
    var_26 = {var_3: var_13, var_4: var_16, var_5: var_19, var_6: var_22, var_7: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = {var_9: var_27, var_10: var_28}
    var_30 = 'above'
    var_31 = {}
    var_32 = {var_9: var_31}
    var_33 = {}
    var_34 = {var_30: var_32, var_9: var_33}
    var_35 = {}
    var_36 = {}
    var_37 = module_0.ParsedContent()
    var_38 = module_1.Config()
    var_39 = module_2.sorted_imports(var_37, var_38)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'x = 1'
    var_2 = [var_1]
    var_3 = '\n'
    var_4 = 'FUTURE'
    var_5 = 'STDLIB'
    var_6 = 'THIRDPARTY'
    var_7 = 'FIRSTPARTY'
    var_8 = 'LOCALFOLDER'
    var_9 = [var_4, var_5, var_6, var_7, var_8]
    var_10 = 'straight'
    var_11 = 'from'
    var_12 = {}
    var_13 = {}
    var_14 = {var_10: var_12, var_11: var_13}
    var_15 = {}
    var_16 = 'os'
    var_17 = 'path'
    var_18 = [var_17]
    var_19 = {var_16: var_18}
    var_20 = {var_10: var_15, var_11: var_19}
    var_21 = {}
    var_22 = {}
    var_23 = {var_10: var_21, var_11: var_22}
    var_24 = {}
    var_25 = {}
    var_26 = {var_10: var_24, var_11: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = {var_10: var_27, var_11: var_28}
    var_30 = {var_4: var_14, var_5: var_20, var_6: var_23, var_7: var_26, var_8: var_29}
    var_31 = {}
    var_32 = {}
    var_33 = {var_10: var_31, var_11: var_32}
    var_34 = 'above'
    var_35 = {}
    var_36 = {}
    var_37 = {var_10: var_35, var_11: var_36}
    var_38 = {}
    var_39 = {}
    var_40 = {var_34: var_37, var_10: var_38, var_11: var_39}
    var_41 = {}
    var_42 = {}
    var_43 = 1
    var_44 = module_0.ParsedContent()
    var_45 = module_1.Config()
    var_46 = module_2.sorted_imports(var_44, var_45)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_with_from_imports_predicate. Retrieved 17/42 statements.


def test_case_0():
    var_0 = 'FUTURE'
    var_1 = 'from'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = 'above'
    var_6 = 'nested'
    var_7 = 'straight'
    var_8 = {}
    var_9 = {}
    var_10 = {var_1: var_9}
    var_11 = {}
    var_12 = {}
    var_13 = []
    var_14 = 'FUTURE'
    var_15 = []
    var_16 = 'import'



# Parsed testcases at query #12
#--------------------------




import isort.output as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._ensure_newline_before_comment(var_0)

import isort.output as module_0

def test_case_0():
    var_0 = 'line1'
    var_1 = 'line2'
    var_2 = 'line3'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._ensure_newline_before_comment(var_3)

import isort.output as module_0

def test_case_0():
    var_0 = '# comment1'
    var_1 = '# comment2'
    var_2 = [var_0, var_1]
    var_3 = module_0._ensure_newline_before_comment(var_2)

import isort.output as module_0

def test_case_0():
    var_0 = '# comment'
    var_1 = 'line1'
    var_2 = [var_0, var_1]
    var_3 = module_0._ensure_newline_before_comment(var_2)

import isort.output as module_0

def test_case_0():
    var_0 = 'line1'
    var_1 = '# comment'
    var_2 = [var_0, var_1]
    var_3 = module_0._ensure_newline_before_comment(var_2)

import isort.output as module_0

def test_case_0():
    var_0 = 'line1'
    var_1 = 'line2'
    var_2 = '# comment1'
    var_3 = 'line3'
    var_4 = '# comment2'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0._ensure_newline_before_comment(var_5)

import isort.output as module_0

def test_case_0():
    var_0 = 'line1'
    var_1 = ''
    var_2 = '# comment'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._ensure_newline_before_comment(var_3)

import isort.output as module_0

def test_case_0():
    var_0 = 'line1'
    var_1 = '# comment1'
    var_2 = '# comment2'
    var_3 = '# comment3'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._ensure_newline_before_comment(var_4)

import isort.output as module_0

def test_case_0():
    var_0 = '# comment'
    var_1 = [var_0]
    var_2 = module_0._ensure_newline_before_comment(var_1)

import isort.output as module_0

def test_case_0():
    var_0 = 'line1'
    var_1 = [var_0]
    var_2 = module_0._ensure_newline_before_comment(var_1)

import isort.output as module_0

def test_case_0():
    var_0 = 'line1'
    var_1 = ''
    var_2 = '# comment'
    var_3 = [var_0, var_1, var_1, var_2]
    var_4 = module_0._ensure_newline_before_comment(var_3)

import isort.output as module_0

def test_case_0():
    var_0 = ''
    var_1 = '# comment'
    var_2 = [var_0, var_1]
    var_3 = module_0._ensure_newline_before_comment(var_2)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_sorted_imports_with_future_imports. Retrieved 32/36 statements.
# Partially parsed test_sorted_imports_normalize_output. Retrieved 28/33 statements.
# Partially parsed test_sorted_imports_with_remove_imports. Retrieved 32/36 statements.
# Partially parsed test_sorted_imports_lines_between_sections. Retrieved 35/39 statements.
# Partially parsed test_sorted_imports_no_sections. Retrieved 35/39 statements.


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = -1
    var_1 = "print('hello')"
    var_2 = ''
    var_3 = [var_1, var_2]
    var_4 = '\n'
    var_5 = []
    var_6 = {}
    var_7 = {}
    var_8 = {}
    var_9 = {}
    var_10 = {}
    var_11 = module_0.ParsedContent()
    var_12 = module_1.Config()
    var_13 = module_2.sorted_imports(var_11, var_12)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = "print('hello')"
    var_2 = ''
    var_3 = [var_1, var_2]
    var_4 = '\n'
    var_5 = 'FUTURE'
    var_6 = 'STDLIB'
    var_7 = [var_5, var_6]
    var_8 = 'straight'
    var_9 = 'from'
    var_10 = {}
    var_11 = {}
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = '__future__'
    var_14 = {}
    var_15 = {var_13: var_14}
    var_16 = {}
    var_17 = {var_8: var_15, var_9: var_16}
    var_18 = {}
    var_19 = {}
    var_20 = {var_8: var_18, var_9: var_19}
    var_21 = {var_5: var_17, var_6: var_20}
    var_22 = 'above'
    var_23 = {}
    var_24 = {var_8: var_23}
    var_25 = {}
    var_26 = {var_22: var_24, var_8: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = module_0.ParsedContent()
    var_30 = module_1.Config()
    var_31 = module_2.sorted_imports(var_29, var_30)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'x = 1'
    var_2 = ''
    var_3 = [var_1, var_2]
    var_4 = '\n'
    var_5 = 'STDLIB'
    var_6 = [var_5]
    var_7 = 'straight'
    var_8 = 'from'
    var_9 = {}
    var_10 = {}
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = 'os'
    var_13 = 'sys'
    var_14 = {}
    var_15 = {}
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = {}
    var_18 = {var_7: var_16, var_8: var_17}
    var_19 = {var_5: var_18}
    var_20 = 'above'
    var_21 = {}
    var_22 = {var_7: var_21}
    var_23 = {}
    var_24 = {var_20: var_22, var_7: var_23}
    var_25 = {}
    var_26 = {}
    var_27 = module_0.ParsedContent()
    var_28 = module_1.Config()
    var_29 = module_2.sorted_imports(var_27, var_28)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'x = 1'
    var_2 = ''
    var_3 = [var_1, var_2]
    var_4 = '\n'
    var_5 = 'STDLIB'
    var_6 = [var_5]
    var_7 = 'straight'
    var_8 = 'from'
    var_9 = {}
    var_10 = {}
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = 'os'
    var_14 = 'path'
    var_15 = 'sys'
    var_16 = [var_14, var_15]
    var_17 = {var_13: var_16}
    var_18 = {var_7: var_12, var_8: var_17}
    var_19 = {var_5: var_18}
    var_20 = 'above'
    var_21 = {}
    var_22 = {var_7: var_21}
    var_23 = {}
    var_24 = {var_20: var_22, var_7: var_23}
    var_25 = {}
    var_26 = {}
    var_27 = module_0.ParsedContent()
    var_28 = module_1.Config()
    var_29 = module_2.sorted_imports(var_27, var_28)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = ''
    var_2 = 'x = 1'
    var_3 = [var_1, var_1, var_2]
    var_4 = '\n'
    var_5 = 'STDLIB'
    var_6 = [var_5]
    var_7 = 'straight'
    var_8 = 'from'
    var_9 = {}
    var_10 = {}
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = 'os'
    var_13 = {}
    var_14 = {var_12: var_13}
    var_15 = {}
    var_16 = {var_7: var_14, var_8: var_15}
    var_17 = {var_5: var_16}
    var_18 = 'above'
    var_19 = {}
    var_20 = {var_7: var_19}
    var_21 = {}
    var_22 = {var_18: var_20, var_7: var_21}
    var_23 = {}
    var_24 = {}
    var_25 = module_0.ParsedContent()
    var_26 = module_1.Config()
    var_27 = module_2.sorted_imports(var_25, var_26)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'x = 1'
    var_2 = ''
    var_3 = [var_1, var_2]
    var_4 = '\n'
    var_5 = 'STDLIB'
    var_6 = [var_5]
    var_7 = 'straight'
    var_8 = 'from'
    var_9 = {}
    var_10 = {}
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = 'os'
    var_13 = 'sys'
    var_14 = {}
    var_15 = {}
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = {}
    var_18 = {var_7: var_16, var_8: var_17}
    var_19 = {var_5: var_18}
    var_20 = 'above'
    var_21 = {}
    var_22 = {var_7: var_21}
    var_23 = {}
    var_24 = {var_20: var_22, var_7: var_23}
    var_25 = {}
    var_26 = {}
    var_27 = module_0.ParsedContent()
    var_28 = 'import os'
    var_29 = [var_28]
    var_30 = module_1.Config()
    var_31 = module_2.sorted_imports(var_27, var_30)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'x = 1'
    var_2 = ''
    var_3 = [var_1, var_2]
    var_4 = '\n'
    var_5 = 'STDLIB'
    var_6 = [var_5]
    var_7 = 'straight'
    var_8 = 'from'
    var_9 = {}
    var_10 = {}
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = 'os'
    var_13 = {}
    var_14 = {var_12: var_13}
    var_15 = {}
    var_16 = {var_7: var_14, var_8: var_15}
    var_17 = {var_5: var_16}
    var_18 = 'above'
    var_19 = {}
    var_20 = {var_7: var_19}
    var_21 = {}
    var_22 = {var_18: var_20, var_7: var_21}
    var_23 = {}
    var_24 = {}
    var_25 = module_0.ParsedContent()
    var_26 = 'stdlib'
    var_27 = 'Standard Library'
    var_28 = {var_26: var_27}
    var_29 = module_1.Config()
    var_30 = module_2.sorted_imports(var_25, var_29)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'x = 1'
    var_2 = ''
    var_3 = [var_1, var_2]
    var_4 = '\n'
    var_5 = 'FUTURE'
    var_6 = 'STDLIB'
    var_7 = [var_5, var_6]
    var_8 = 'straight'
    var_9 = 'from'
    var_10 = {}
    var_11 = {}
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = '__future__'
    var_14 = {}
    var_15 = {var_13: var_14}
    var_16 = {}
    var_17 = {var_8: var_15, var_9: var_16}
    var_18 = 'os'
    var_19 = {}
    var_20 = {var_18: var_19}
    var_21 = {}
    var_22 = {var_8: var_20, var_9: var_21}
    var_23 = {var_5: var_17, var_6: var_22}
    var_24 = 'above'
    var_25 = {}
    var_26 = {var_8: var_25}
    var_27 = {}
    var_28 = {var_24: var_26, var_8: var_27}
    var_29 = {}
    var_30 = {}
    var_31 = module_0.ParsedContent()
    var_32 = 2
    var_33 = module_1.Config()
    var_34 = module_2.sorted_imports(var_31, var_33)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'x = 1'
    var_2 = ''
    var_3 = [var_1, var_2]
    var_4 = '\n'
    var_5 = 'STDLIB'
    var_6 = 'THIRDPARTY'
    var_7 = [var_5, var_6]
    var_8 = 'straight'
    var_9 = 'from'
    var_10 = {}
    var_11 = {}
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = 'os'
    var_14 = {}
    var_15 = {var_13: var_14}
    var_16 = {}
    var_17 = {var_8: var_15, var_9: var_16}
    var_18 = 'django'
    var_19 = {}
    var_20 = {var_18: var_19}
    var_21 = {}
    var_22 = {var_8: var_20, var_9: var_21}
    var_23 = {var_5: var_17, var_6: var_22}
    var_24 = 'above'
    var_25 = {}
    var_26 = {var_8: var_25}
    var_27 = {}
    var_28 = {var_24: var_26, var_8: var_27}
    var_29 = {}
    var_30 = {}
    var_31 = module_0.ParsedContent()
    var_32 = True
    var_33 = module_1.Config()
    var_34 = module_2.sorted_imports(var_31, var_33)

def test_case_0():
    pass



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_with_from_imports_with_star_import. Retrieved 40/44 statements.


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = {}
    var_3 = 'from'
    var_4 = 'straight'
    var_5 = 'nested'
    var_6 = 'above'
    var_7 = {}
    var_8 = {}
    var_9 = {}
    var_10 = {}
    var_11 = {var_3: var_10}
    var_12 = {var_3: var_7, var_4: var_8, var_5: var_9, var_6: var_11}
    var_13 = 0
    var_14 = 'THIRDPARTY'
    var_15 = lambda x: var_14
    var_16 = '    '
    var_17 = set()
    var_18 = set()
    var_19 = set()
    var_20 = set()
    var_21 = set()
    var_22 = set()
    var_23 = set()
    var_24 = set()
    var_25 = set()
    var_26 = set()
    var_27 = {}
    var_28 = {}
    var_29 = {var_3: var_27, var_4: var_28}
    var_30 = '\n'
    var_31 = set()
    var_32 = module_0.ParsedContent()
    var_33 = module_1.Config()
    var_34 = []
    var_35 = []
    var_36 = 'import'
    var_37 = module_2._with_from_imports(var_32, var_33, var_34, var_14, var_35, var_36)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = {var_1: var_8}
    var_10 = {}
    var_11 = 'straight'
    var_12 = 'nested'
    var_13 = 'above'
    var_14 = {}
    var_15 = {}
    var_16 = {}
    var_17 = {}
    var_18 = {var_2: var_17}
    var_19 = {var_2: var_14, var_11: var_15, var_12: var_16, var_13: var_18}
    var_20 = lambda x: var_1
    var_21 = '    '
    var_22 = set()
    var_23 = set()
    var_24 = set()
    var_25 = set()
    var_26 = set()
    var_27 = set()
    var_28 = set()
    var_29 = set()
    var_30 = set()
    var_31 = set()
    var_32 = {}
    var_33 = {}
    var_34 = {var_2: var_32, var_11: var_33}
    var_35 = '\n'
    var_36 = set()
    var_37 = module_0.ParsedContent()
    var_38 = module_1.Config()
    var_39 = [var_3]
    var_40 = []
    var_41 = 'import'
    var_42 = module_2._with_from_imports(var_37, var_38, var_39, var_1, var_40, var_41)
    var_43 = len(var_42)
    assert var_43 == 1

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = {var_1: var_8}
    var_10 = {}
    var_11 = 'straight'
    var_12 = 'nested'
    var_13 = 'above'
    var_14 = {}
    var_15 = {}
    var_16 = {}
    var_17 = {}
    var_18 = {var_2: var_17}
    var_19 = {var_2: var_14, var_11: var_15, var_12: var_16, var_13: var_18}
    var_20 = lambda x: var_1
    var_21 = '    '
    var_22 = set()
    var_23 = set()
    var_24 = set()
    var_25 = set()
    var_26 = set()
    var_27 = set()
    var_28 = set()
    var_29 = set()
    var_30 = set()
    var_31 = set()
    var_32 = {}
    var_33 = {}
    var_34 = {var_2: var_32, var_11: var_33}
    var_35 = '\n'
    var_36 = set()
    var_37 = module_0.ParsedContent()
    var_38 = module_1.Config()
    var_39 = [var_3]
    var_40 = [var_3]
    var_41 = 'import'
    var_42 = module_2._with_from_imports(var_37, var_38, var_39, var_1, var_40, var_41)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = {var_1: var_8}
    var_10 = {}
    var_11 = 'straight'
    var_12 = 'nested'
    var_13 = 'above'
    var_14 = {}
    var_15 = {}
    var_16 = {}
    var_17 = {}
    var_18 = {var_2: var_17}
    var_19 = {var_2: var_14, var_11: var_15, var_12: var_16, var_13: var_18}
    var_20 = lambda x: var_1
    var_21 = '    '
    var_22 = set()
    var_23 = set()
    var_24 = set()
    var_25 = set()
    var_26 = set()
    var_27 = set()
    var_28 = set()
    var_29 = set()
    var_30 = set()
    var_31 = set()
    var_32 = {}
    var_33 = {}
    var_34 = {var_2: var_32, var_11: var_33}
    var_35 = '\n'
    var_36 = set()
    var_37 = module_0.ParsedContent()
    var_38 = module_1.Config()
    var_39 = [var_3]
    var_40 = [var_3]
    var_41 = 'import'
    var_42 = module_2._with_from_imports(var_37, var_38, var_39, var_1, var_40, var_41)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = 'environ'
    var_6 = False
    var_7 = {var_4: var_6, var_5: var_6}
    var_8 = {var_3: var_7}
    var_9 = {var_2: var_8}
    var_10 = {var_1: var_9}
    var_11 = {}
    var_12 = 'straight'
    var_13 = 'nested'
    var_14 = 'above'
    var_15 = {}
    var_16 = {}
    var_17 = {}
    var_18 = {}
    var_19 = {var_2: var_18}
    var_20 = {var_2: var_15, var_12: var_16, var_13: var_17, var_14: var_19}
    var_21 = lambda x: var_1
    var_22 = '    '
    var_23 = set()
    var_24 = set()
    var_25 = set()
    var_26 = set()
    var_27 = set()
    var_28 = set()
    var_29 = set()
    var_30 = set()
    var_31 = set()
    var_32 = set()
    var_33 = {}
    var_34 = {}
    var_35 = {var_2: var_33, var_12: var_34}
    var_36 = '\n'
    var_37 = set()
    var_38 = module_0.ParsedContent()
    var_39 = module_1.Config()
    var_40 = [var_3]
    var_41 = []
    var_42 = 'import'
    var_43 = module_2._with_from_imports(var_38, var_39, var_40, var_1, var_41, var_42)
    var_44 = len(var_43)
    assert var_44 == 1

import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = []
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = '*'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = {var_1: var_8}
    var_10 = {}
    var_11 = 'straight'
    var_12 = 'nested'
    var_13 = 'above'
    var_14 = {}
    var_15 = {}
    var_16 = {}
    var_17 = {}
    var_18 = {var_2: var_17}
    var_19 = {var_2: var_14, var_11: var_15, var_12: var_16, var_13: var_18}
    var_20 = lambda x: var_1
    var_21 = '    '
    var_22 = set()
    var_23 = set()
    var_24 = set()
    var_25 = set()
    var_26 = set()
    var_27 = set()
    var_28 = set()
    var_29 = set()
    var_30 = set()
    var_31 = set()
    var_32 = {}
    var_33 = {}
    var_34 = {var_2: var_32, var_11: var_33}
    var_35 = '\n'
    var_36 = set()
    var_37 = module_0.ParsedContent()
    var_38 = True
    var_39 = module_1.Config()



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_sorted_imports_with_empty_imports. Retrieved 38/42 statements.
# Partially parsed test_sorted_imports_with_line_separator. Retrieved 29/33 statements.


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = -1
    var_1 = "print('hello')"
    var_2 = 'x = 1'
    var_3 = [var_1, var_2]
    var_4 = '\n'
    var_5 = []
    var_6 = {}
    var_7 = {}
    var_8 = {}
    var_9 = {}
    var_10 = {}
    var_11 = 2
    var_12 = module_0.ParsedContent()
    var_13 = module_1.Config()
    var_14 = module_2.sorted_imports(var_12, var_13)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = "print('hello')"
    var_2 = [var_1]
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'THIRDPARTY'
    var_6 = 'FIRSTPARTY'
    var_7 = 'LOCALFOLDER'
    var_8 = [var_4, var_5, var_6, var_7]
    var_9 = 'straight'
    var_10 = 'from'
    var_11 = {}
    var_12 = {}
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = {}
    var_15 = {}
    var_16 = {var_9: var_14, var_10: var_15}
    var_17 = {}
    var_18 = {}
    var_19 = {var_9: var_17, var_10: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = {var_9: var_20, var_10: var_21}
    var_23 = {var_4: var_13, var_5: var_16, var_6: var_19, var_7: var_22}
    var_24 = {}
    var_25 = {}
    var_26 = {var_9: var_24, var_10: var_25}
    var_27 = 'above'
    var_28 = {}
    var_29 = {var_9: var_28}
    var_30 = {}
    var_31 = {var_27: var_29, var_9: var_30}
    var_32 = {}
    var_33 = {}
    var_34 = 1
    var_35 = module_0.ParsedContent()
    var_36 = module_1.Config()
    var_37 = module_2.sorted_imports(var_35, var_36)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = "print('hello')"
    var_2 = [var_1]
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'THIRDPARTY'
    var_6 = 'FIRSTPARTY'
    var_7 = 'LOCALFOLDER'
    var_8 = [var_4, var_5, var_6, var_7]
    var_9 = 'straight'
    var_10 = 'from'
    var_11 = 'os'
    var_12 = None
    var_13 = {var_11: var_12}
    var_14 = {}
    var_15 = {var_9: var_13, var_10: var_14}
    var_16 = {}
    var_17 = {}
    var_18 = {var_9: var_16, var_10: var_17}
    var_19 = {}
    var_20 = {}
    var_21 = {var_9: var_19, var_10: var_20}
    var_22 = {}
    var_23 = {}
    var_24 = {var_9: var_22, var_10: var_23}
    var_25 = {var_4: var_15, var_5: var_18, var_6: var_21, var_7: var_24}
    var_26 = {}
    var_27 = {}
    var_28 = {var_9: var_26, var_10: var_27}
    var_29 = 'above'
    var_30 = {}
    var_31 = {var_9: var_30}
    var_32 = {}
    var_33 = {var_29: var_31, var_9: var_32}
    var_34 = {}
    var_35 = {}
    var_36 = 1
    var_37 = module_0.ParsedContent()
    var_38 = module_1.Config()
    var_39 = module_2.sorted_imports(var_37, var_38)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'code'
    var_2 = [var_1]
    var_3 = '\r\n'
    var_4 = 'STDLIB'
    var_5 = [var_4]
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = 'sys'
    var_9 = None
    var_10 = {var_8: var_9}
    var_11 = {}
    var_12 = {var_6: var_10, var_7: var_11}
    var_13 = {var_4: var_12}
    var_14 = {}
    var_15 = {}
    var_16 = {var_6: var_14, var_7: var_15}
    var_17 = 'above'
    var_18 = {}
    var_19 = {var_6: var_18}
    var_20 = {}
    var_21 = {var_17: var_19, var_6: var_20}
    var_22 = {}
    var_23 = {}
    var_24 = 1
    var_25 = module_0.ParsedContent()
    var_26 = module_1.Config()
    var_27 = 'py'
    var_28 = module_2.sorted_imports(var_25, var_26, var_27)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'x = 1'
    var_2 = [var_1]
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'THIRDPARTY'
    var_6 = [var_4, var_5]
    var_7 = 'straight'
    var_8 = 'from'
    var_9 = 'os'
    var_10 = None
    var_11 = {var_9: var_10}
    var_12 = {}
    var_13 = {var_7: var_11, var_8: var_12}
    var_14 = 'requests'
    var_15 = {var_14: var_10}
    var_16 = {}
    var_17 = {var_7: var_15, var_8: var_16}
    var_18 = {var_4: var_13, var_5: var_17}
    var_19 = {}
    var_20 = {}
    var_21 = {var_7: var_19, var_8: var_20}
    var_22 = 'above'
    var_23 = {}
    var_24 = {var_7: var_23}
    var_25 = {}
    var_26 = {var_22: var_24, var_7: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = 1
    var_30 = module_0.ParsedContent()
    var_31 = module_1.Config()
    var_32 = module_2.sorted_imports(var_30, var_31)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'code'
    var_2 = [var_1]
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = [var_4]
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = {}
    var_9 = 'os'
    var_10 = 'path'
    var_11 = None
    var_12 = {var_10: var_11}
    var_13 = {var_9: var_12}
    var_14 = {var_6: var_8, var_7: var_13}
    var_15 = {var_4: var_14}
    var_16 = {}
    var_17 = {}
    var_18 = {var_6: var_16, var_7: var_17}
    var_19 = 'above'
    var_20 = {}
    var_21 = {var_6: var_20}
    var_22 = {}
    var_23 = {var_19: var_21, var_6: var_22}
    var_24 = {}
    var_25 = {}
    var_26 = 1
    var_27 = module_0.ParsedContent()
    var_28 = module_1.Config()
    var_29 = module_2.sorted_imports(var_27, var_28)



# Parsed testcases at query #16
#--------------------------




import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = {}
    var_1 = 'straight'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 'above'
    var_5 = {}
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = {var_4: var_6, var_1: var_7}
    var_9 = 0
    var_10 = {}
    var_11 = {}
    var_12 = {}
    var_13 = module_0.ParsedContent()
    var_14 = module_1.Config()
    var_15 = []
    var_16 = 'THIRDPARTY'
    var_17 = []
    var_18 = 'import'
    var_19 = module_2._with_straight_imports(var_13, var_14, var_15, var_16, var_17, var_18)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'THIRDPARTY'
    var_1 = 'straight'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = {}
    var_6 = {var_1: var_5}
    var_7 = 'above'
    var_8 = {}
    var_9 = {var_1: var_8}
    var_10 = {}
    var_11 = {var_7: var_9, var_1: var_10}
    var_12 = 0
    var_13 = {}
    var_14 = {}
    var_15 = {}
    var_16 = module_0.ParsedContent()
    var_17 = True
    var_18 = module_1.Config()
    var_19 = 'module1'
    var_20 = 'module2'
    var_21 = [var_19, var_20]
    var_22 = []
    var_23 = 'import'
    var_24 = module_2._with_straight_imports(var_16, var_18, var_21, var_0, var_22, var_23)
    var_25 = len(var_24)
    assert var_25 == 1

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'THIRDPARTY'
    var_1 = 'straight'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = {}
    var_6 = {var_1: var_5}
    var_7 = 'above'
    var_8 = {}
    var_9 = {var_1: var_8}
    var_10 = 'module1'
    var_11 = 'module2'
    var_12 = 'comment1'
    var_13 = [var_12]
    var_14 = 'comment2'
    var_15 = [var_14]
    var_16 = {var_10: var_13, var_11: var_15}
    var_17 = {var_7: var_9, var_1: var_16}
    var_18 = 0
    var_19 = {}
    var_20 = {}
    var_21 = {}
    var_22 = module_0.ParsedContent()
    var_23 = True
    var_24 = module_1.Config()
    var_25 = [var_10, var_11]
    var_26 = []
    var_27 = 'import'
    var_28 = module_2._with_straight_imports(var_22, var_24, var_25, var_0, var_26, var_27)
    var_29 = len(var_28)
    assert var_29 == 1

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'THIRDPARTY'
    var_1 = 'straight'
    var_2 = 'module1'
    var_3 = None
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = {}
    var_8 = {var_1: var_7}
    var_9 = 'above'
    var_10 = {}
    var_11 = {var_1: var_10}
    var_12 = {}
    var_13 = {var_9: var_11, var_1: var_12}
    var_14 = 0
    var_15 = {}
    var_16 = {}
    var_17 = {}
    var_18 = module_0.ParsedContent()
    var_19 = False
    var_20 = module_1.Config()
    var_21 = [var_2]
    var_22 = [var_2]
    var_23 = 'import'
    var_24 = module_2._with_straight_imports(var_18, var_20, var_21, var_0, var_22, var_23)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'THIRDPARTY'
    var_1 = 'straight'
    var_2 = 'module1'
    var_3 = None
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = {}
    var_8 = {var_1: var_7}
    var_9 = 'above'
    var_10 = {}
    var_11 = {var_1: var_10}
    var_12 = {}
    var_13 = {var_9: var_11, var_1: var_12}
    var_14 = 0
    var_15 = {}
    var_16 = {}
    var_17 = {}
    var_18 = module_0.ParsedContent()
    var_19 = False
    var_20 = module_1.Config()
    var_21 = [var_2]
    var_22 = []
    var_23 = 'import'
    var_24 = module_2._with_straight_imports(var_18, var_20, var_21, var_0, var_22, var_23)
    var_25 = len(var_24)
    assert var_25 == 1

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'THIRDPARTY'
    var_1 = 'straight'
    var_2 = 'module1'
    var_3 = None
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'alias1'
    var_8 = [var_7]
    var_9 = {var_2: var_8}
    var_10 = {var_1: var_9}
    var_11 = 'above'
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = {}
    var_15 = {var_11: var_13, var_1: var_14}
    var_16 = 0
    var_17 = {}
    var_18 = {}
    var_19 = {}
    var_20 = module_0.ParsedContent()
    var_21 = True
    var_22 = module_1.Config()
    var_23 = [var_2]
    var_24 = []
    var_25 = 'import'
    var_26 = module_2._with_straight_imports(var_20, var_22, var_23, var_0, var_24, var_25)
    var_27 = len(var_26)
    assert var_27 == 2

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'THIRDPARTY'
    var_1 = 'straight'
    var_2 = 'module1'
    var_3 = None
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = {}
    var_8 = {var_1: var_7}
    var_9 = 'above'
    var_10 = '# above comment'
    var_11 = [var_10]
    var_12 = {var_2: var_11}
    var_13 = {var_1: var_12}
    var_14 = {}
    var_15 = {var_9: var_13, var_1: var_14}
    var_16 = 0
    var_17 = {}
    var_18 = {}
    var_19 = {}
    var_20 = module_0.ParsedContent()
    var_21 = False
    var_22 = module_1.Config()
    var_23 = [var_2]
    var_24 = []
    var_25 = 'import'
    var_26 = module_2._with_straight_imports(var_20, var_22, var_23, var_0, var_24, var_25)
    var_27 = len(var_26)
    assert var_27 == 2



# Parsed testcases at query #17
#--------------------------

# Failed to parse test_predicate_at_line_1_evaluates_to_false.




# Parsed testcases at query #18
#--------------------------

# Partially parsed test_with_from_imports_with_removed_module. Retrieved 22/29 statements.
# Partially parsed test_with_from_imports_single_import. Retrieved 25/48 statements.
# Partially parsed test_with_from_imports_star_import_without_combine. Retrieved 25/48 statements.
# Partially parsed test_with_from_imports_force_single_line. Retrieved 26/49 statements.


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = module_1.Config()
    var_2 = []
    var_3 = 'THIRDPARTY'
    var_4 = []
    var_5 = 'import'
    var_6 = module_2._with_from_imports(var_0, var_1, var_2, var_3, var_4, var_5)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'above'
    var_8 = 'nested'
    var_9 = 'straight'
    var_10 = {}
    var_11 = {}
    var_12 = {var_2: var_11}
    var_13 = {}
    var_14 = {}
    var_15 = {}
    var_16 = module_1.Config()
    var_17 = [var_3]
    var_18 = 'THIRDPARTY'
    var_19 = [var_3]
    var_20 = 'import'
    var_21 = module_2._with_from_imports(var_0, var_16, var_17, var_18, var_19, var_20)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = None
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = 'straight'
    var_12 = {}
    var_13 = {}
    var_14 = {var_2: var_13}
    var_15 = {}
    var_16 = {}
    var_17 = {}
    var_18 = module_1.Config()
    var_19 = [var_3]
    var_20 = 'THIRDPARTY'
    var_21 = []
    var_22 = 'import'
    var_23 = module_2._with_from_imports(var_0, var_18, var_19, var_20, var_21, var_22)
    var_24 = len(var_23)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = '*'
    var_5 = None
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = 'straight'
    var_12 = {}
    var_13 = {}
    var_14 = {var_2: var_13}
    var_15 = {}
    var_16 = {}
    var_17 = {}
    var_18 = module_1.Config()
    var_19 = [var_3]
    var_20 = 'THIRDPARTY'
    var_21 = []
    var_22 = 'import'
    var_23 = module_2._with_from_imports(var_0, var_18, var_19, var_20, var_21, var_22)
    var_24 = len(var_23)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = 'environ'
    var_6 = None
    var_7 = {var_4: var_6, var_5: var_6}
    var_8 = {var_3: var_7}
    var_9 = {var_2: var_8}
    var_10 = 'above'
    var_11 = 'nested'
    var_12 = 'straight'
    var_13 = {}
    var_14 = {}
    var_15 = {var_2: var_14}
    var_16 = {}
    var_17 = {}
    var_18 = {}
    var_19 = module_1.Config()
    var_20 = [var_3]
    var_21 = 'THIRDPARTY'
    var_22 = []
    var_23 = 'import'
    var_24 = module_2._with_from_imports(var_0, var_19, var_20, var_21, var_22, var_23)
    var_25 = len(var_24)
    assert var_25 == 2



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_with_star_comments_with_star_comment. Retrieved 12/15 statements.
# Partially parsed test_with_star_comments_without_star_comment. Retrieved 9/11 statements.
# Partially parsed test_with_star_comments_module_not_in_nested. Retrieved 7/9 statements.
# Partially parsed test_with_star_comments_empty_comments_list. Retrieved 9/11 statements.


import isort.parse as module_0
import isort.output as module_1

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'nested'
    var_2 = 'test_module'
    var_3 = '*'
    var_4 = 'star comment text'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'comment1'
    var_8 = 'comment2'
    var_9 = [var_7, var_8]
    var_10 = module_1._with_star_comments(var_0, var_2, var_9)
    var_11 = var_0.categorized_comments[var_1][var_2]

import isort.parse as module_0
import isort.output as module_1

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'nested'
    var_2 = 'test_module'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'comment1'
    var_6 = 'comment2'
    var_7 = [var_5, var_6]
    var_8 = module_1._with_star_comments(var_0, var_2, var_7)

import isort.parse as module_0
import isort.output as module_1

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'nested'
    var_2 = {}
    var_3 = 'comment1'
    var_4 = [var_3]
    var_5 = 'missing_module'
    var_6 = module_1._with_star_comments(var_0, var_5, var_4)

import isort.parse as module_0
import isort.output as module_1

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'nested'
    var_2 = 'test_module'
    var_3 = '*'
    var_4 = 'star comment'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = []
    var_8 = module_1._with_star_comments(var_0, var_2, var_7)



# Parsed testcases at query #20
#--------------------------




def test_case_0():
    var_0 = 'module.func1'
    var_1 = 'module.func2'
    var_2 = [var_0, var_1]
    var_3 = bool(var_2)
    assert var_3 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_with_straight_imports_predicate_line_1. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 1 (function definition) evaluates to True.'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_sorted_imports_with_no_imports. Retrieved 13/16 statements.


import isort.parse as module_0

def test_case_0():
    var_0 = -1
    var_1 = "print('hello')"
    var_2 = 'x = 1'
    var_3 = [var_1, var_2]
    var_4 = '\n'
    var_5 = []
    var_6 = {}
    var_7 = {}
    var_8 = {}
    var_9 = 2
    var_10 = module_0.ParsedContent()
    var_11 = 'py'
    var_12 = 'import'



# Parsed testcases at query #23
#--------------------------




import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = module_0.Config()
    var_2 = -1
    var_3 = {}
    var_4 = {}
    var_5 = "print('hello')"
    var_6 = [var_5]
    var_7 = 0
    var_8 = 1
    var_9 = []
    var_10 = module_1.ParsedContent()
    var_11 = module_0.Config()
    var_12 = module_2.sorted_imports(var_10, var_11)
    assert var_12 == "print('hello')\n"



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_with_from_imports_single_module. Retrieved 25/48 statements.
# Partially parsed test_with_from_imports_with_remove_imports. Retrieved 24/46 statements.
# Partially parsed test_with_from_imports_force_single_line. Retrieved 25/48 statements.
# Partially parsed test_with_from_imports_combine_star. Retrieved 24/47 statements.
# Partially parsed test_with_from_imports_with_comments. Retrieved 21/41 statements.


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = module_1.Config()
    var_2 = []
    var_3 = 'THIRDPARTY'
    var_4 = []
    var_5 = 'import'
    var_6 = module_2._with_from_imports(var_0, var_1, var_2, var_3, var_4, var_5)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = 'straight'
    var_12 = {}
    var_13 = {}
    var_14 = {var_2: var_13}
    var_15 = {}
    var_16 = {}
    var_17 = {}
    var_18 = module_1.Config()
    var_19 = [var_3]
    var_20 = 'THIRDPARTY'
    var_21 = []
    var_22 = 'import'
    var_23 = module_2._with_from_imports(var_0, var_18, var_19, var_20, var_21, var_22)
    var_24 = len(var_23)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = 'straight'
    var_12 = {}
    var_13 = {}
    var_14 = {var_2: var_13}
    var_15 = {}
    var_16 = {}
    var_17 = {}
    var_18 = module_1.Config()
    var_19 = [var_3]
    var_20 = 'THIRDPARTY'
    var_21 = [var_3]
    var_22 = 'import'
    var_23 = module_2._with_from_imports(var_0, var_18, var_19, var_20, var_21, var_22)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = 'getcwd'
    var_6 = False
    var_7 = {var_4: var_6, var_5: var_6}
    var_8 = {var_3: var_7}
    var_9 = {var_2: var_8}
    var_10 = 'above'
    var_11 = 'nested'
    var_12 = 'straight'
    var_13 = {}
    var_14 = {}
    var_15 = {var_2: var_14}
    var_16 = {}
    var_17 = {}
    var_18 = {}
    var_19 = module_1.Config()
    var_20 = [var_3]
    var_21 = 'THIRDPARTY'
    var_22 = []
    var_23 = 'import'
    var_24 = module_2._with_from_imports(var_0, var_19, var_20, var_21, var_22, var_23)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = '*'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = 'straight'
    var_12 = {}
    var_13 = {}
    var_14 = {var_2: var_13}
    var_15 = {}
    var_16 = {}
    var_17 = {}
    var_18 = module_1.Config()
    var_19 = [var_3]
    var_20 = 'THIRDPARTY'
    var_21 = []
    var_22 = 'import'
    var_23 = module_2._with_from_imports(var_0, var_18, var_19, var_20, var_21, var_22)

import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = 'straight'
    var_12 = 'test comment'
    var_13 = [var_12]
    var_14 = {var_3: var_13}
    var_15 = {}
    var_16 = {var_2: var_15}
    var_17 = {}
    var_18 = {}
    var_19 = {}
    var_20 = module_1.Config()



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_with_from_imports_basic. Retrieved 40/43 statements.
# Partially parsed test_with_from_imports_with_comments. Retrieved 43/46 statements.
# Partially parsed test_with_from_imports_remove_imports. Retrieved 40/43 statements.
# Partially parsed test_with_from_imports_multiple_imports. Retrieved 41/44 statements.
# Partially parsed test_with_from_imports_with_star. Retrieved 42/45 statements.


import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 0
    var_2 = 'FUTURE'
    var_3 = 'STDLIB'
    var_4 = 'from'
    var_5 = 'straight'
    var_6 = {}
    var_7 = {}
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = 'os'
    var_10 = 'path'
    var_11 = True
    var_12 = {var_10: var_11}
    var_13 = {var_9: var_12}
    var_14 = {}
    var_15 = {var_4: var_13, var_5: var_14}
    var_16 = {var_2: var_8, var_3: var_15}
    var_17 = {}
    var_18 = {}
    var_19 = {var_4: var_17, var_5: var_18}
    var_20 = 'nested'
    var_21 = 'above'
    var_22 = {}
    var_23 = {}
    var_24 = {}
    var_25 = {}
    var_26 = {var_4: var_25}
    var_27 = {var_4: var_22, var_5: var_23, var_20: var_24, var_21: var_26}
    var_28 = {}
    var_29 = {}
    var_30 = ''
    var_31 = False
    var_32 = lambda x: var_3
    var_33 = '\n'
    var_34 = set()
    var_35 = module_1.ParsedContent()
    var_36 = [var_9]
    var_37 = []
    var_38 = 'import'
    var_39 = module_2._with_from_imports(var_35, var_0, var_36, var_3, var_37, var_38)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 0
    var_2 = 'FUTURE'
    var_3 = 'STDLIB'
    var_4 = 'from'
    var_5 = 'straight'
    var_6 = {}
    var_7 = {}
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = 'sys'
    var_10 = 'argv'
    var_11 = True
    var_12 = {var_10: var_11}
    var_13 = {var_9: var_12}
    var_14 = {}
    var_15 = {var_4: var_13, var_5: var_14}
    var_16 = {var_2: var_8, var_3: var_15}
    var_17 = {}
    var_18 = {}
    var_19 = {var_4: var_17, var_5: var_18}
    var_20 = 'nested'
    var_21 = 'above'
    var_22 = 'system module'
    var_23 = [var_22]
    var_24 = {var_9: var_23}
    var_25 = {}
    var_26 = {}
    var_27 = {}
    var_28 = {var_4: var_27}
    var_29 = {var_4: var_24, var_5: var_25, var_20: var_26, var_21: var_28}
    var_30 = {}
    var_31 = {}
    var_32 = ''
    var_33 = False
    var_34 = lambda x: var_3
    var_35 = '\n'
    var_36 = set()
    var_37 = module_1.ParsedContent()
    var_38 = [var_9]
    var_39 = []
    var_40 = 'import'
    var_41 = module_2._with_from_imports(var_37, var_0, var_38, var_3, var_39, var_40)
    var_42 = len(var_41)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 0
    var_2 = 'FUTURE'
    var_3 = 'STDLIB'
    var_4 = 'from'
    var_5 = 'straight'
    var_6 = {}
    var_7 = {}
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = 'os'
    var_10 = 'path'
    var_11 = True
    var_12 = {var_10: var_11}
    var_13 = {var_9: var_12}
    var_14 = {}
    var_15 = {var_4: var_13, var_5: var_14}
    var_16 = {var_2: var_8, var_3: var_15}
    var_17 = {}
    var_18 = {}
    var_19 = {var_4: var_17, var_5: var_18}
    var_20 = 'nested'
    var_21 = 'above'
    var_22 = {}
    var_23 = {}
    var_24 = {}
    var_25 = {}
    var_26 = {var_4: var_25}
    var_27 = {var_4: var_22, var_5: var_23, var_20: var_24, var_21: var_26}
    var_28 = {}
    var_29 = {}
    var_30 = ''
    var_31 = False
    var_32 = lambda x: var_3
    var_33 = '\n'
    var_34 = set()
    var_35 = module_1.ParsedContent()
    var_36 = [var_9]
    var_37 = [var_9]
    var_38 = 'import'
    var_39 = module_2._with_from_imports(var_35, var_0, var_36, var_3, var_37, var_38)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 0
    var_2 = 'FUTURE'
    var_3 = 'STDLIB'
    var_4 = 'from'
    var_5 = 'straight'
    var_6 = {}
    var_7 = {}
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {}
    var_10 = {}
    var_11 = {var_4: var_9, var_5: var_10}
    var_12 = {var_2: var_8, var_3: var_11}
    var_13 = {}
    var_14 = {}
    var_15 = {var_4: var_13, var_5: var_14}
    var_16 = 'nested'
    var_17 = 'above'
    var_18 = {}
    var_19 = {}
    var_20 = {}
    var_21 = {}
    var_22 = {var_4: var_21}
    var_23 = {var_4: var_18, var_5: var_19, var_16: var_20, var_17: var_22}
    var_24 = {}
    var_25 = {}
    var_26 = ''
    var_27 = False
    var_28 = lambda x: var_3
    var_29 = '\n'
    var_30 = set()
    var_31 = module_1.ParsedContent()
    var_32 = []
    var_33 = []
    var_34 = 'import'
    var_35 = module_2._with_from_imports(var_31, var_0, var_32, var_3, var_33, var_34)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 0
    var_2 = 'FUTURE'
    var_3 = 'STDLIB'
    var_4 = 'from'
    var_5 = 'straight'
    var_6 = {}
    var_7 = {}
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = 'os'
    var_10 = 'path'
    var_11 = 'environ'
    var_12 = True
    var_13 = {var_10: var_12, var_11: var_12}
    var_14 = {var_9: var_13}
    var_15 = {}
    var_16 = {var_4: var_14, var_5: var_15}
    var_17 = {var_2: var_8, var_3: var_16}
    var_18 = {}
    var_19 = {}
    var_20 = {var_4: var_18, var_5: var_19}
    var_21 = 'nested'
    var_22 = 'above'
    var_23 = {}
    var_24 = {}
    var_25 = {}
    var_26 = {}
    var_27 = {var_4: var_26}
    var_28 = {var_4: var_23, var_5: var_24, var_21: var_25, var_22: var_27}
    var_29 = {}
    var_30 = {}
    var_31 = ''
    var_32 = False
    var_33 = lambda x: var_3
    var_34 = '\n'
    var_35 = set()
    var_36 = module_1.ParsedContent()
    var_37 = [var_9]
    var_38 = []
    var_39 = 'import'
    var_40 = module_2._with_from_imports(var_36, var_0, var_37, var_3, var_38, var_39)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = 0
    var_3 = 'FUTURE'
    var_4 = 'STDLIB'
    var_5 = 'from'
    var_6 = 'straight'
    var_7 = {}
    var_8 = {}
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'os'
    var_11 = '*'
    var_12 = {var_11: var_0}
    var_13 = {var_10: var_12}
    var_14 = {}
    var_15 = {var_5: var_13, var_6: var_14}
    var_16 = {var_3: var_9, var_4: var_15}
    var_17 = {}
    var_18 = {}
    var_19 = {var_5: var_17, var_6: var_18}
    var_20 = 'nested'
    var_21 = 'above'
    var_22 = {}
    var_23 = {}
    var_24 = 'star import'
    var_25 = {var_11: var_24}
    var_26 = {var_10: var_25}
    var_27 = {}
    var_28 = {var_5: var_27}
    var_29 = {var_5: var_22, var_6: var_23, var_20: var_26, var_21: var_28}
    var_30 = {}
    var_31 = {}
    var_32 = ''
    var_33 = False
    var_34 = lambda x: var_4
    var_35 = '\n'
    var_36 = set()
    var_37 = module_1.ParsedContent()
    var_38 = [var_10]
    var_39 = []
    var_40 = 'import'
    var_41 = module_2._with_from_imports(var_37, var_1, var_38, var_4, var_39, var_40)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_with_from_imports_basic. Retrieved 22/33 statements.
# Partially parsed test_with_from_imports_with_star_import. Retrieved 22/33 statements.
# Partially parsed test_with_from_imports_removed_modules. Retrieved 21/31 statements.
# Partially parsed test_with_from_imports_force_single_line. Retrieved 23/35 statements.
# Partially parsed test_with_from_imports_with_comments. Retrieved 24/35 statements.
# Partially parsed test_with_from_imports_empty_from_modules. Retrieved 17/27 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'THIRDPARTY'
    var_1 = 'from'
    var_2 = 'module1'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = False
    var_6 = {var_3: var_5, var_4: var_5}
    var_7 = {var_2: var_6}
    var_8 = {var_1: var_7}
    var_9 = {}
    var_10 = 'above'
    var_11 = 'nested'
    var_12 = 'straight'
    var_13 = {}
    var_14 = {}
    var_15 = {var_1: var_14}
    var_16 = {}
    var_17 = {}
    var_18 = module_0.Config()
    var_19 = [var_2]
    var_20 = []
    var_21 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'THIRDPARTY'
    var_1 = 'from'
    var_2 = 'module1'
    var_3 = '*'
    var_4 = False
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = 'straight'
    var_12 = {}
    var_13 = {}
    var_14 = {var_1: var_13}
    var_15 = {}
    var_16 = {}
    var_17 = True
    var_18 = module_0.Config()
    var_19 = [var_2]
    var_20 = []
    var_21 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'THIRDPARTY'
    var_1 = 'from'
    var_2 = 'module1'
    var_3 = 'import1'
    var_4 = False
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = 'straight'
    var_12 = {}
    var_13 = {}
    var_14 = {var_1: var_13}
    var_15 = {}
    var_16 = {}
    var_17 = module_0.Config()
    var_18 = [var_2]
    var_19 = [var_2]
    var_20 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'THIRDPARTY'
    var_1 = 'from'
    var_2 = 'module1'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = False
    var_6 = {var_3: var_5, var_4: var_5}
    var_7 = {var_2: var_6}
    var_8 = {var_1: var_7}
    var_9 = {}
    var_10 = 'above'
    var_11 = 'nested'
    var_12 = 'straight'
    var_13 = {}
    var_14 = {}
    var_15 = {var_1: var_14}
    var_16 = {}
    var_17 = {}
    var_18 = True
    var_19 = module_0.Config()
    var_20 = [var_2]
    var_21 = []
    var_22 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'THIRDPARTY'
    var_1 = 'from'
    var_2 = 'module1'
    var_3 = 'import1'
    var_4 = False
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = 'straight'
    var_12 = 'comment1'
    var_13 = [var_12]
    var_14 = {var_2: var_13}
    var_15 = {}
    var_16 = {var_1: var_15}
    var_17 = {}
    var_18 = {var_2: var_17}
    var_19 = {}
    var_20 = module_0.Config()
    var_21 = [var_2]
    var_22 = []
    var_23 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'THIRDPARTY'
    var_1 = 'from'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = 'above'
    var_6 = 'nested'
    var_7 = 'straight'
    var_8 = {}
    var_9 = {}
    var_10 = {var_1: var_9}
    var_11 = {}
    var_12 = {}
    var_13 = module_0.Config()
    var_14 = []
    var_15 = []
    var_16 = 'import'



# Parsed testcases at query #27
#--------------------------




import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = -1
    var_3 = {}
    var_4 = {}
    var_5 = 'x = 1'
    var_6 = [var_5]
    var_7 = ''
    var_8 = [var_7]
    var_9 = 'FUTURE'
    var_10 = 'STDLIB'
    var_11 = 'THIRDPARTY'
    var_12 = 'FIRSTPARTY'
    var_13 = 'LOCALFOLDER'
    var_14 = [var_9, var_10, var_11, var_12, var_13]
    var_15 = {}
    var_16 = {}
    var_17 = {}
    var_18 = []
    var_19 = False
    var_20 = False
    var_21 = '\n'
    var_22 = module_0.ParsedContent()
    var_23 = module_1.Config()
    var_24 = module_2.sorted_imports(var_22, var_23)
    assert var_24 == 'x = 1'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_false. Retrieved 6/14 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 1 (_with_straight_imports) evaluates to False when called.'
    var_1 = module_0.Config()
    var_2 = []
    var_3 = 'THIRDPARTY'
    var_4 = []
    var_5 = 'import'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_with_from_imports_predicate_line_1. Retrieved 12/16 statements.


import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = module_0.Config()
    var_2 = 'import'
    var_3 = module_1.ParsedContent()
    var_4 = module_0.Config()
    var_5 = 'os'
    var_6 = 'sys'
    var_7 = [var_5, var_6]
    var_8 = 'STDLIB'
    var_9 = []
    var_10 = 'import'
    var_11 = module_2._with_from_imports(var_3, var_4, var_7, var_8, var_9, var_10)



# Parsed testcases at query #30
#--------------------------




def test_case_0():
    var_0 = []
    var_1 = bool(var_0)
    assert var_1 is False



# Parsed testcases at query #31
#--------------------------




import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = {}
    var_1 = 'straight'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 'above'
    var_5 = {}
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = {var_4: var_6, var_1: var_7}
    var_9 = module_0.ParsedContent()
    var_10 = True
    var_11 = module_1.Config()
    var_12 = []
    var_13 = 'THIRDPARTY'
    var_14 = []
    var_15 = 'import'
    var_16 = module_2._with_straight_imports(var_9, var_11, var_12, var_13, var_14, var_15)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = {}
    var_1 = 'straight'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 'above'
    var_5 = {}
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = {var_4: var_6, var_1: var_7}
    var_9 = module_0.ParsedContent()
    var_10 = True
    var_11 = module_1.Config()
    var_12 = 'os'
    var_13 = 'sys'
    var_14 = [var_12, var_13]
    var_15 = 'STDLIB'
    var_16 = []
    var_17 = 'import'
    var_18 = module_2._with_straight_imports(var_9, var_11, var_14, var_15, var_16, var_17)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = {}
    var_1 = 'straight'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 'above'
    var_5 = {}
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = {var_4: var_6, var_1: var_7}
    var_9 = module_0.ParsedContent()
    var_10 = True
    var_11 = module_1.Config()
    var_12 = 'os'
    var_13 = 'sys'
    var_14 = [var_12, var_13]
    var_15 = 'STDLIB'
    var_16 = []
    var_17 = 'import'
    var_18 = module_2._with_straight_imports(var_9, var_11, var_14, var_15, var_16, var_17)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'straight'
    var_2 = 'os'
    var_3 = False
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'O'
    var_8 = [var_7]
    var_9 = {var_2: var_8}
    var_10 = {var_1: var_9}
    var_11 = 'above'
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = {}
    var_15 = {var_11: var_13, var_1: var_14}
    var_16 = module_0.ParsedContent()
    var_17 = True
    var_18 = module_1.Config()
    var_19 = [var_2]
    var_20 = []
    var_21 = 'import'
    var_22 = module_2._with_straight_imports(var_16, var_18, var_19, var_0, var_20, var_21)
    var_23 = len(var_22)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'straight'
    var_2 = 'os'
    var_3 = 'sys'
    var_4 = False
    var_5 = {var_2: var_4, var_3: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = {}
    var_9 = {var_1: var_8}
    var_10 = 'above'
    var_11 = {}
    var_12 = {var_1: var_11}
    var_13 = {}
    var_14 = {var_10: var_12, var_1: var_13}
    var_15 = module_0.ParsedContent()
    var_16 = module_1.Config()
    var_17 = [var_2, var_3]
    var_18 = [var_2]
    var_19 = 'import'
    var_20 = module_2._with_straight_imports(var_15, var_16, var_17, var_0, var_18, var_19)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'straight'
    var_2 = 'os'
    var_3 = False
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = {}
    var_8 = {var_1: var_7}
    var_9 = 'above'
    var_10 = '# noqa'
    var_11 = [var_10]
    var_12 = {var_2: var_11}
    var_13 = {var_1: var_12}
    var_14 = {}
    var_15 = {var_9: var_13, var_1: var_14}
    var_16 = module_0.ParsedContent()
    var_17 = True
    var_18 = module_1.Config()
    var_19 = [var_2]
    var_20 = []
    var_21 = 'import'
    var_22 = module_2._with_straight_imports(var_16, var_18, var_19, var_0, var_20, var_21)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_line_45_predicate_evaluates_to_true. Retrieved 9/21 statements.


def test_case_0():
    var_0 = 'foo'
    var_1 = 'bar'
    var_2 = [var_0, var_1]
    var_3 = 'baz'
    var_4 = 'baz as b'
    var_5 = [var_4]
    var_6 = {var_3: var_5}
    var_7 = '*'
    var_8 = var_7 in var_2



# Parsed testcases at query #33
#--------------------------




def test_case_0():
    var_0 = 'module1'
    var_1 = 'module2'
    var_2 = [var_0, var_1]
    var_3 = bool(var_2)
    assert var_3 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_with_from_imports_basic_import. Retrieved 25/34 statements.
# Partially parsed test_with_from_imports_with_star_import. Retrieved 24/33 statements.
# Partially parsed test_with_from_imports_force_single_line. Retrieved 25/34 statements.


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = module_1.Config()
    var_2 = []
    var_3 = 'THIRDPARTY'
    var_4 = []
    var_5 = 'import'
    var_6 = module_2._with_from_imports(var_0, var_1, var_2, var_3, var_4, var_5)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = module_1.Config()
    var_2 = 'os'
    var_3 = [var_2]
    var_4 = 'STDLIB'
    var_5 = [var_2]
    var_6 = 'import'
    var_7 = module_2._with_from_imports(var_0, var_1, var_3, var_4, var_5, var_6)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'STDLIB'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = True
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = {}
    var_10 = 'above'
    var_11 = 'nested'
    var_12 = 'straight'
    var_13 = {}
    var_14 = {}
    var_15 = {var_2: var_14}
    var_16 = {}
    var_17 = {}
    var_18 = module_1.Config()
    var_19 = [var_3]
    var_20 = 'STDLIB'
    var_21 = []
    var_22 = 'import'
    var_23 = module_2._with_from_imports(var_0, var_18, var_19, var_20, var_21, var_22)
    var_24 = len(var_23)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'django'
    var_4 = '*'
    var_5 = True
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = {}
    var_10 = 'above'
    var_11 = 'nested'
    var_12 = 'straight'
    var_13 = {}
    var_14 = {}
    var_15 = {var_2: var_14}
    var_16 = {}
    var_17 = {}
    var_18 = module_1.Config()
    var_19 = [var_3]
    var_20 = 'THIRDPARTY'
    var_21 = []
    var_22 = 'import'
    var_23 = module_2._with_from_imports(var_0, var_18, var_19, var_20, var_21, var_22)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'requests'
    var_4 = 'get'
    var_5 = 'post'
    var_6 = True
    var_7 = {var_4: var_6, var_5: var_6}
    var_8 = {var_3: var_7}
    var_9 = {var_2: var_8}
    var_10 = {}
    var_11 = 'above'
    var_12 = 'nested'
    var_13 = 'straight'
    var_14 = {}
    var_15 = {}
    var_16 = {var_2: var_15}
    var_17 = {}
    var_18 = {}
    var_19 = module_1.Config()
    var_20 = [var_3]
    var_21 = 'THIRDPARTY'
    var_22 = []
    var_23 = 'import'
    var_24 = module_2._with_from_imports(var_0, var_19, var_20, var_21, var_22, var_23)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_with_star_comments_predicate_false. Retrieved 8/12 statements.


def test_case_0():
    var_0 = 'nested'
    var_1 = 'test_module'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 'test_module'
    var_5 = 'comment1'
    var_6 = 'comment2'
    var_7 = [var_5, var_6]



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_with_from_imports_empty_from_modules. Retrieved 19/27 statements.
# Partially parsed test_with_from_imports_single_import. Retrieved 23/32 statements.
# Partially parsed test_with_from_imports_with_remove_imports. Retrieved 23/31 statements.
# Partially parsed test_with_from_imports_module_in_remove_imports. Retrieved 23/31 statements.
# Partially parsed test_with_from_imports_with_comments. Retrieved 26/35 statements.
# Partially parsed test_with_from_imports_force_single_line. Retrieved 25/34 statements.
# Partially parsed test_with_from_imports_with_star_import. Retrieved 24/33 statements.
# Partially parsed test_with_from_imports_with_as_imports. Retrieved 26/35 statements.
# Partially parsed test_with_from_imports_above_comments. Retrieved 25/34 statements.


import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_1.ParsedContent()
    var_2 = 'THIRDPARTY'
    var_3 = 'from'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = 'above'
    var_8 = 'nested'
    var_9 = 'straight'
    var_10 = {}
    var_11 = {}
    var_12 = {var_3: var_11}
    var_13 = {}
    var_14 = {}
    var_15 = []
    var_16 = []
    var_17 = 'import'
    var_18 = module_2._with_from_imports(var_1, var_0, var_15, var_2, var_16, var_17)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_1.ParsedContent()
    var_2 = 'THIRDPARTY'
    var_3 = 'from'
    var_4 = 'os'
    var_5 = 'path'
    var_6 = False
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = {var_3: var_8}
    var_10 = {}
    var_11 = 'above'
    var_12 = 'nested'
    var_13 = 'straight'
    var_14 = {}
    var_15 = {}
    var_16 = {var_3: var_15}
    var_17 = {}
    var_18 = {}
    var_19 = [var_4]
    var_20 = []
    var_21 = 'import'
    var_22 = module_2._with_from_imports(var_1, var_0, var_19, var_2, var_20, var_21)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_1.ParsedContent()
    var_2 = 'THIRDPARTY'
    var_3 = 'from'
    var_4 = 'os'
    var_5 = 'path'
    var_6 = False
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = {var_3: var_8}
    var_10 = {}
    var_11 = 'above'
    var_12 = 'nested'
    var_13 = 'straight'
    var_14 = {}
    var_15 = {}
    var_16 = {var_3: var_15}
    var_17 = {}
    var_18 = {}
    var_19 = [var_4]
    var_20 = [var_4]
    var_21 = 'import'
    var_22 = module_2._with_from_imports(var_1, var_0, var_19, var_2, var_20, var_21)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_1.ParsedContent()
    var_2 = 'THIRDPARTY'
    var_3 = 'from'
    var_4 = 'sys'
    var_5 = 'exit'
    var_6 = False
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = {var_3: var_8}
    var_10 = {}
    var_11 = 'above'
    var_12 = 'nested'
    var_13 = 'straight'
    var_14 = {}
    var_15 = {}
    var_16 = {var_3: var_15}
    var_17 = {}
    var_18 = {}
    var_19 = [var_4]
    var_20 = [var_4]
    var_21 = 'import'
    var_22 = module_2._with_from_imports(var_1, var_0, var_19, var_2, var_20, var_21)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_1.ParsedContent()
    var_2 = 'THIRDPARTY'
    var_3 = 'from'
    var_4 = 'os'
    var_5 = 'path'
    var_6 = False
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = {var_3: var_8}
    var_10 = {}
    var_11 = 'above'
    var_12 = 'nested'
    var_13 = 'straight'
    var_14 = '# important'
    var_15 = [var_14]
    var_16 = {var_4: var_15}
    var_17 = {}
    var_18 = {var_3: var_17}
    var_19 = {}
    var_20 = {}
    var_21 = [var_4]
    var_22 = []
    var_23 = 'import'
    var_24 = module_2._with_from_imports(var_1, var_0, var_21, var_2, var_22, var_23)
    var_25 = len(var_24)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = module_1.ParsedContent()
    var_3 = 'THIRDPARTY'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = 'path'
    var_7 = 'getcwd'
    var_8 = False
    var_9 = {var_6: var_8, var_7: var_8}
    var_10 = {var_5: var_9}
    var_11 = {var_4: var_10}
    var_12 = {}
    var_13 = 'above'
    var_14 = 'nested'
    var_15 = 'straight'
    var_16 = {}
    var_17 = {}
    var_18 = {var_4: var_17}
    var_19 = {}
    var_20 = {}
    var_21 = [var_5]
    var_22 = []
    var_23 = 'import'
    var_24 = module_2._with_from_imports(var_2, var_1, var_21, var_3, var_22, var_23)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = module_1.ParsedContent()
    var_3 = 'THIRDPARTY'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = '*'
    var_7 = False
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = {var_4: var_9}
    var_11 = {}
    var_12 = 'above'
    var_13 = 'nested'
    var_14 = 'straight'
    var_15 = {}
    var_16 = {}
    var_17 = {var_4: var_16}
    var_18 = {}
    var_19 = {}
    var_20 = [var_5]
    var_21 = []
    var_22 = 'import'
    var_23 = module_2._with_from_imports(var_2, var_1, var_20, var_3, var_21, var_22)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = module_1.ParsedContent()
    var_3 = 'THIRDPARTY'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = 'path'
    var_7 = {var_6: var_0}
    var_8 = {var_5: var_7}
    var_9 = {var_4: var_8}
    var_10 = 'os.path'
    var_11 = 'p'
    var_12 = [var_11]
    var_13 = {var_10: var_12}
    var_14 = 'above'
    var_15 = 'nested'
    var_16 = 'straight'
    var_17 = {}
    var_18 = {}
    var_19 = {var_4: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = [var_5]
    var_23 = []
    var_24 = 'import'
    var_25 = module_2._with_from_imports(var_2, var_1, var_22, var_3, var_23, var_24)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_1.ParsedContent()
    var_2 = 'THIRDPARTY'
    var_3 = 'from'
    var_4 = 'os'
    var_5 = 'path'
    var_6 = False
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = {var_3: var_8}
    var_10 = {}
    var_11 = 'above'
    var_12 = 'nested'
    var_13 = 'straight'
    var_14 = {}
    var_15 = '# above comment'
    var_16 = [var_15]
    var_17 = {var_4: var_16}
    var_18 = {var_3: var_17}
    var_19 = {}
    var_20 = {}
    var_21 = [var_4]
    var_22 = []
    var_23 = 'import'
    var_24 = module_2._with_from_imports(var_1, var_0, var_21, var_2, var_22, var_23)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_sorted_imports_normalizes_output. Retrieved 23/27 statements.


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = -1
    var_1 = "print('hello')"
    var_2 = ''
    var_3 = [var_1, var_2]
    var_4 = {}
    var_5 = {}
    var_6 = {}
    var_7 = 'straight'
    var_8 = 'from'
    var_9 = {}
    var_10 = {}
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = 'above'
    var_13 = {}
    var_14 = {var_7: var_13}
    var_15 = {}
    var_16 = {var_12: var_14, var_7: var_15}
    var_17 = []
    var_18 = '\n'
    var_19 = 2
    var_20 = module_0.ParsedContent()
    var_21 = module_1.Config()
    var_22 = module_2.sorted_imports(var_20, var_21)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = ''
    var_2 = "print('hello')"
    var_3 = [var_1, var_2]
    var_4 = {}
    var_5 = {}
    var_6 = 'FUTURE'
    var_7 = 'STDLIB'
    var_8 = 'THIRDPARTY'
    var_9 = 'FIRSTPARTY'
    var_10 = 'LOCALFOLDER'
    var_11 = 'straight'
    var_12 = 'from'
    var_13 = {}
    var_14 = {}
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = 'os'
    var_17 = None
    var_18 = {var_16: var_17}
    var_19 = {}
    var_20 = {var_11: var_18, var_12: var_19}
    var_21 = {}
    var_22 = {}
    var_23 = {var_11: var_21, var_12: var_22}
    var_24 = {}
    var_25 = {}
    var_26 = {var_11: var_24, var_12: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = {var_11: var_27, var_12: var_28}
    var_30 = {var_6: var_15, var_7: var_20, var_8: var_23, var_9: var_26, var_10: var_29}
    var_31 = {}
    var_32 = {}
    var_33 = {var_11: var_31, var_12: var_32}
    var_34 = 'above'
    var_35 = {}
    var_36 = {var_11: var_35}
    var_37 = {}
    var_38 = {var_34: var_36, var_11: var_37}
    var_39 = [var_6, var_7, var_8, var_9, var_10]
    var_40 = '\n'
    var_41 = 2
    var_42 = module_0.ParsedContent()
    var_43 = module_1.Config()
    var_44 = module_2.sorted_imports(var_42, var_43)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = -1
    var_1 = 'line1'
    var_2 = ''
    var_3 = [var_1, var_2]
    var_4 = {}
    var_5 = {}
    var_6 = {}
    var_7 = 'straight'
    var_8 = 'from'
    var_9 = {}
    var_10 = {}
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = 'above'
    var_13 = {}
    var_14 = {var_7: var_13}
    var_15 = {}
    var_16 = {var_12: var_14, var_7: var_15}
    var_17 = []
    var_18 = '\n'
    var_19 = 2
    var_20 = module_0.ParsedContent()
    var_21 = module_1.Config()
    var_22 = module_2.sorted_imports(var_20, var_21)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = ''
    var_2 = [var_1]
    var_3 = {}
    var_4 = {}
    var_5 = 'FUTURE'
    var_6 = 'STDLIB'
    var_7 = 'THIRDPARTY'
    var_8 = 'FIRSTPARTY'
    var_9 = 'LOCALFOLDER'
    var_10 = 'straight'
    var_11 = 'from'
    var_12 = {}
    var_13 = {}
    var_14 = {var_10: var_12, var_11: var_13}
    var_15 = 'os'
    var_16 = 'sys'
    var_17 = None
    var_18 = {var_15: var_17, var_16: var_17}
    var_19 = {}
    var_20 = {var_10: var_18, var_11: var_19}
    var_21 = {}
    var_22 = {}
    var_23 = {var_10: var_21, var_11: var_22}
    var_24 = {}
    var_25 = {}
    var_26 = {var_10: var_24, var_11: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = {var_10: var_27, var_11: var_28}
    var_30 = {var_5: var_14, var_6: var_20, var_7: var_23, var_8: var_26, var_9: var_29}
    var_31 = {}
    var_32 = {}
    var_33 = {var_10: var_31, var_11: var_32}
    var_34 = 'above'
    var_35 = {}
    var_36 = {var_10: var_35}
    var_37 = {}
    var_38 = {var_34: var_36, var_10: var_37}
    var_39 = [var_5, var_6, var_7, var_8, var_9]
    var_40 = '\n'
    var_41 = 1
    var_42 = module_0.ParsedContent()
    var_43 = 'import os'
    var_44 = [var_43]
    var_45 = module_1.Config()
    var_46 = module_2.sorted_imports(var_42, var_45)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = ''
    var_2 = [var_1]
    var_3 = {}
    var_4 = {}
    var_5 = 'FUTURE'
    var_6 = 'STDLIB'
    var_7 = 'THIRDPARTY'
    var_8 = 'FIRSTPARTY'
    var_9 = 'LOCALFOLDER'
    var_10 = 'straight'
    var_11 = 'from'
    var_12 = {}
    var_13 = {}
    var_14 = {var_10: var_12, var_11: var_13}
    var_15 = 'os'
    var_16 = None
    var_17 = {var_15: var_16}
    var_18 = {}
    var_19 = {var_10: var_17, var_11: var_18}
    var_20 = 'django'
    var_21 = {var_20: var_16}
    var_22 = {}
    var_23 = {var_10: var_21, var_11: var_22}
    var_24 = {}
    var_25 = {}
    var_26 = {var_10: var_24, var_11: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = {var_10: var_27, var_11: var_28}
    var_30 = {var_5: var_14, var_6: var_19, var_7: var_23, var_8: var_26, var_9: var_29}
    var_31 = {}
    var_32 = {}
    var_33 = {var_10: var_31, var_11: var_32}
    var_34 = 'above'
    var_35 = {}
    var_36 = {var_10: var_35}
    var_37 = {}
    var_38 = {var_34: var_36, var_10: var_37}
    var_39 = [var_5, var_6, var_7, var_8, var_9]
    var_40 = '\n'
    var_41 = 1
    var_42 = module_0.ParsedContent()
    var_43 = 2
    var_44 = module_1.Config()
    var_45 = module_2.sorted_imports(var_42, var_44)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_with_from_imports_basic. Retrieved 29/33 statements.
# Partially parsed test_with_from_imports_remove_imports. Retrieved 30/34 statements.
# Partially parsed test_with_from_imports_with_comments. Retrieved 30/34 statements.
# Partially parsed test_with_from_imports_force_single_line. Retrieved 30/34 statements.
# Partially parsed test_with_from_imports_combine_as_imports. Retrieved 32/36 statements.
# Partially parsed test_with_from_imports_star_import. Retrieved 31/35 statements.


import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'module'
    var_4 = 'func'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = {var_1: var_8}
    var_10 = 'above'
    var_11 = 'nested'
    var_12 = 'straight'
    var_13 = {}
    var_14 = {}
    var_15 = {var_2: var_14}
    var_16 = {}
    var_17 = {}
    var_18 = {var_2: var_13, var_10: var_15, var_11: var_16, var_12: var_17}
    var_19 = {}
    var_20 = {var_2: var_19}
    var_21 = set()
    var_22 = '\n'
    var_23 = module_1.ParsedContent()
    var_24 = [var_3]
    var_25 = []
    var_26 = 'import'
    var_27 = module_2._with_from_imports(var_23, var_0, var_24, var_1, var_25, var_26)
    var_28 = len(var_27)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'module1'
    var_4 = 'module2'
    var_5 = 'func'
    var_6 = False
    var_7 = {var_5: var_6}
    var_8 = {var_5: var_6}
    var_9 = {var_3: var_7, var_4: var_8}
    var_10 = {var_2: var_9}
    var_11 = {var_1: var_10}
    var_12 = 'above'
    var_13 = 'nested'
    var_14 = 'straight'
    var_15 = {}
    var_16 = {}
    var_17 = {var_2: var_16}
    var_18 = {}
    var_19 = {}
    var_20 = {var_2: var_15, var_12: var_17, var_13: var_18, var_14: var_19}
    var_21 = {}
    var_22 = {var_2: var_21}
    var_23 = set()
    var_24 = '\n'
    var_25 = module_1.ParsedContent()
    var_26 = [var_3, var_4]
    var_27 = [var_3]
    var_28 = 'import'
    var_29 = module_2._with_from_imports(var_25, var_0, var_26, var_1, var_27, var_28)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'above'
    var_7 = 'nested'
    var_8 = 'straight'
    var_9 = {}
    var_10 = {}
    var_11 = {var_2: var_10}
    var_12 = {}
    var_13 = {}
    var_14 = {var_2: var_9, var_6: var_11, var_7: var_12, var_8: var_13}
    var_15 = {}
    var_16 = {var_2: var_15}
    var_17 = set()
    var_18 = '\n'
    var_19 = module_1.ParsedContent()
    var_20 = []
    var_21 = []
    var_22 = 'import'
    var_23 = module_2._with_from_imports(var_19, var_0, var_20, var_1, var_21, var_22)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'module'
    var_4 = 'func'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = {var_1: var_8}
    var_10 = 'above'
    var_11 = 'nested'
    var_12 = 'straight'
    var_13 = '# test comment'
    var_14 = [var_13]
    var_15 = {var_3: var_14}
    var_16 = {}
    var_17 = {var_2: var_16}
    var_18 = {}
    var_19 = {}
    var_20 = {var_2: var_15, var_10: var_17, var_11: var_18, var_12: var_19}
    var_21 = {}
    var_22 = {var_2: var_21}
    var_23 = set()
    var_24 = '\n'
    var_25 = module_1.ParsedContent()
    var_26 = [var_3]
    var_27 = []
    var_28 = 'import'
    var_29 = module_2._with_from_imports(var_25, var_0, var_26, var_1, var_27, var_28)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = 'THIRDPARTY'
    var_3 = 'from'
    var_4 = 'module'
    var_5 = 'func1'
    var_6 = 'func2'
    var_7 = False
    var_8 = {var_5: var_7, var_6: var_7}
    var_9 = {var_4: var_8}
    var_10 = {var_3: var_9}
    var_11 = {var_2: var_10}
    var_12 = 'above'
    var_13 = 'nested'
    var_14 = 'straight'
    var_15 = {}
    var_16 = {}
    var_17 = {var_3: var_16}
    var_18 = {}
    var_19 = {}
    var_20 = {var_3: var_15, var_12: var_17, var_13: var_18, var_14: var_19}
    var_21 = {}
    var_22 = {var_3: var_21}
    var_23 = set()
    var_24 = '\n'
    var_25 = module_1.ParsedContent()
    var_26 = [var_4]
    var_27 = []
    var_28 = 'import'
    var_29 = module_2._with_from_imports(var_25, var_1, var_26, var_2, var_27, var_28)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = 'THIRDPARTY'
    var_3 = 'from'
    var_4 = 'module'
    var_5 = 'func'
    var_6 = False
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = {var_3: var_8}
    var_10 = {var_2: var_9}
    var_11 = 'above'
    var_12 = 'nested'
    var_13 = 'straight'
    var_14 = {}
    var_15 = {}
    var_16 = {var_3: var_15}
    var_17 = {}
    var_18 = {}
    var_19 = {var_3: var_14, var_11: var_16, var_12: var_17, var_13: var_18}
    var_20 = 'module.func'
    var_21 = 'alias'
    var_22 = [var_21]
    var_23 = {var_20: var_22}
    var_24 = {var_3: var_23}
    var_25 = set()
    var_26 = '\n'
    var_27 = module_1.ParsedContent()
    var_28 = [var_4]
    var_29 = []
    var_30 = 'import'
    var_31 = module_2._with_from_imports(var_27, var_1, var_28, var_2, var_29, var_30)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = 'THIRDPARTY'
    var_3 = 'from'
    var_4 = 'module'
    var_5 = '*'
    var_6 = False
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = {var_3: var_8}
    var_10 = {var_2: var_9}
    var_11 = 'above'
    var_12 = 'nested'
    var_13 = 'straight'
    var_14 = {}
    var_15 = {}
    var_16 = {var_3: var_15}
    var_17 = '# star'
    var_18 = {var_5: var_17}
    var_19 = {var_4: var_18}
    var_20 = {}
    var_21 = {var_3: var_14, var_11: var_16, var_12: var_19, var_13: var_20}
    var_22 = {}
    var_23 = {var_3: var_22}
    var_24 = set()
    var_25 = '\n'
    var_26 = module_1.ParsedContent()
    var_27 = [var_4]
    var_28 = []
    var_29 = 'import'
    var_30 = module_2._with_from_imports(var_26, var_1, var_27, var_2, var_28, var_29)



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_predicate_at_line_16_evaluates_to_false. Retrieved 1/22 statements.


def test_case_0():
    var_0 = 'os'



# Parsed testcases at query #40
#--------------------------




def test_case_0():
    var_0 = []
    var_1 = bool(var_0)
    assert var_1 is False



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_with_from_imports_basic. Retrieved 22/47 statements.
# Partially parsed test_with_from_imports_remove_imports. Retrieved 22/48 statements.
# Partially parsed test_with_from_imports_force_single_line. Retrieved 22/47 statements.
# Partially parsed test_with_from_imports_empty_modules. Retrieved 17/43 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'environ'
    var_5 = False
    var_6 = {var_3: var_5, var_4: var_5}
    var_7 = {var_2: var_6}
    var_8 = {var_1: var_7}
    var_9 = {}
    var_10 = 'nested'
    var_11 = 'above'
    var_12 = 'straight'
    var_13 = {}
    var_14 = {}
    var_15 = {}
    var_16 = {var_1: var_15}
    var_17 = {}
    var_18 = module_0.Config()
    var_19 = [var_2]
    var_20 = []
    var_21 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'environ'
    var_5 = False
    var_6 = {var_3: var_5, var_4: var_5}
    var_7 = {var_2: var_6}
    var_8 = {var_1: var_7}
    var_9 = {}
    var_10 = 'nested'
    var_11 = 'above'
    var_12 = 'straight'
    var_13 = {}
    var_14 = {}
    var_15 = {}
    var_16 = {var_1: var_15}
    var_17 = {}
    var_18 = module_0.Config()
    var_19 = [var_2]
    var_20 = [var_2]
    var_21 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'environ'
    var_5 = False
    var_6 = {var_3: var_5, var_4: var_5}
    var_7 = {var_2: var_6}
    var_8 = {var_1: var_7}
    var_9 = {}
    var_10 = 'nested'
    var_11 = 'above'
    var_12 = 'straight'
    var_13 = {}
    var_14 = {}
    var_15 = {}
    var_16 = {var_1: var_15}
    var_17 = {}
    var_18 = module_0.Config()
    var_19 = [var_2]
    var_20 = []
    var_21 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = 'nested'
    var_6 = 'above'
    var_7 = 'straight'
    var_8 = {}
    var_9 = {}
    var_10 = {}
    var_11 = {var_1: var_10}
    var_12 = {}
    var_13 = module_0.Config()
    var_14 = []
    var_15 = []
    var_16 = 'import'



# Parsed testcases at query #42
#--------------------------




def test_case_0():
    var_0 = []
    var_1 = bool(var_0)
    assert var_1 is False



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_predicate_at_line_45_evaluates_to_false. Retrieved 8/18 statements.


def test_case_0():
    var_0 = '*'
    var_1 = 'foo'
    var_2 = 'bar'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'foo as f'
    var_5 = [var_4]
    var_6 = {var_1: var_5}
    var_7 = var_0 in var_3



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_predicate_at_line_45_evaluates_to_false. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 'module1'
    var_1 = 'module2'
    var_2 = [var_0, var_1]
    var_3 = '*'
    var_4 = var_3 in var_2



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_with_from_imports_remove_imports_module. Retrieved 12/16 statements.
# Partially parsed test_with_from_imports_with_comments. Retrieved 26/34 statements.
# Partially parsed test_with_from_imports_star_import. Retrieved 25/34 statements.
# Partially parsed test_with_from_imports_force_single_line. Retrieved 27/36 statements.
# Partially parsed test_with_from_imports_as_imports. Retrieved 28/37 statements.
# Partially parsed test_with_from_imports_multiple_modules. Retrieved 27/36 statements.
# Partially parsed test_with_from_imports_with_nested_comments. Retrieved 25/34 statements.


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = module_1.Config()
    var_2 = []
    var_3 = 'THIRDPARTY'
    var_4 = []
    var_5 = 'import'
    var_6 = module_2._with_from_imports(var_0, var_1, var_2, var_3, var_4, var_5)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'module1'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_1.Config()
    var_8 = [var_3]
    var_9 = []
    var_10 = 'import'
    var_11 = module_2._with_from_imports(var_0, var_7, var_8, var_1, var_9, var_10)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'module1'
    var_4 = 'func1'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = 'straight'
    var_12 = '# comment1'
    var_13 = [var_12]
    var_14 = {var_3: var_13}
    var_15 = {}
    var_16 = {var_2: var_15}
    var_17 = {}
    var_18 = {}
    var_19 = {}
    var_20 = module_1.Config()
    var_21 = [var_3]
    var_22 = []
    var_23 = 'import'
    var_24 = module_2._with_from_imports(var_0, var_20, var_21, var_1, var_22, var_23)
    var_25 = len(var_24)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'module1'
    var_4 = '*'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = 'straight'
    var_12 = {}
    var_13 = {}
    var_14 = {var_2: var_13}
    var_15 = {}
    var_16 = {var_3: var_15}
    var_17 = {}
    var_18 = {}
    var_19 = True
    var_20 = module_1.Config()
    var_21 = [var_3]
    var_22 = []
    var_23 = 'import'
    var_24 = module_2._with_from_imports(var_0, var_20, var_21, var_1, var_22, var_23)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'module1'
    var_4 = 'func1'
    var_5 = 'func2'
    var_6 = False
    var_7 = {var_4: var_6, var_5: var_6}
    var_8 = {var_3: var_7}
    var_9 = {var_2: var_8}
    var_10 = 'above'
    var_11 = 'nested'
    var_12 = 'straight'
    var_13 = {}
    var_14 = {}
    var_15 = {var_2: var_14}
    var_16 = {}
    var_17 = {var_3: var_16}
    var_18 = {}
    var_19 = {}
    var_20 = True
    var_21 = []
    var_22 = module_1.Config()
    var_23 = [var_3]
    var_24 = []
    var_25 = 'import'
    var_26 = module_2._with_from_imports(var_0, var_22, var_23, var_1, var_24, var_25)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'module1'
    var_4 = 'func1'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = 'straight'
    var_12 = {}
    var_13 = {}
    var_14 = {var_2: var_13}
    var_15 = {}
    var_16 = {var_3: var_15}
    var_17 = {}
    var_18 = 'module1.func1'
    var_19 = 'alias1'
    var_20 = [var_19]
    var_21 = {var_18: var_20}
    var_22 = True
    var_23 = module_1.Config()
    var_24 = [var_3]
    var_25 = []
    var_26 = 'import'
    var_27 = module_2._with_from_imports(var_0, var_23, var_24, var_1, var_25, var_26)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'module1'
    var_4 = 'module2'
    var_5 = 'func1'
    var_6 = False
    var_7 = {var_5: var_6}
    var_8 = 'func2'
    var_9 = {var_8: var_6}
    var_10 = {var_3: var_7, var_4: var_9}
    var_11 = {var_2: var_10}
    var_12 = 'above'
    var_13 = 'nested'
    var_14 = 'straight'
    var_15 = {}
    var_16 = {}
    var_17 = {var_2: var_16}
    var_18 = {}
    var_19 = {}
    var_20 = {}
    var_21 = module_1.Config()
    var_22 = [var_3, var_4]
    var_23 = []
    var_24 = 'import'
    var_25 = module_2._with_from_imports(var_0, var_21, var_22, var_1, var_23, var_24)
    var_26 = len(var_25)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'module1'
    var_4 = 'func1'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = 'straight'
    var_12 = {}
    var_13 = {}
    var_14 = {var_2: var_13}
    var_15 = '# nested comment'
    var_16 = {var_4: var_15}
    var_17 = {var_3: var_16}
    var_18 = {}
    var_19 = {}
    var_20 = module_1.Config()
    var_21 = [var_3]
    var_22 = []
    var_23 = 'import'
    var_24 = module_2._with_from_imports(var_0, var_20, var_21, var_1, var_22, var_23)



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_with_straight_imports_predicate_line_1. Retrieved 8/17 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 1 evaluates to True for the function definition.'
    var_1 = 'parsed'
    var_2 = 'config'
    var_3 = 'straight_modules'
    var_4 = 'section'
    var_5 = 'remove_imports'
    var_6 = 'import_type'
    var_7 = [var_1, var_2, var_3, var_4, var_5, var_6]



# Parsed testcases at query #47
#--------------------------




import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = module_1.Config()
    var_2 = []
    var_3 = 'THIRDPARTY'
    var_4 = []
    var_5 = 'import'
    var_6 = None
    var_7 = []
    var_8 = False
    assert var_8 is False



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_with_from_imports_basic. Retrieved 22/47 statements.
# Partially parsed test_with_from_imports_with_remove_imports. Retrieved 23/48 statements.
# Partially parsed test_with_from_imports_empty_modules. Retrieved 17/41 statements.
# Partially parsed test_with_from_imports_force_single_line. Retrieved 22/47 statements.
# Partially parsed test_with_from_imports_with_star_imports. Retrieved 21/46 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'THIRDPARTY'
    var_1 = 'from'
    var_2 = 'module1'
    var_3 = 'func1'
    var_4 = 'func2'
    var_5 = False
    var_6 = {var_3: var_5, var_4: var_5}
    var_7 = {var_2: var_6}
    var_8 = {var_1: var_7}
    var_9 = {}
    var_10 = 'nested'
    var_11 = 'above'
    var_12 = 'straight'
    var_13 = {}
    var_14 = {}
    var_15 = {}
    var_16 = {var_1: var_15}
    var_17 = {}
    var_18 = module_0.Config()
    var_19 = [var_2]
    var_20 = []
    var_21 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'THIRDPARTY'
    var_1 = 'from'
    var_2 = 'module1'
    var_3 = 'func1'
    var_4 = 'func2'
    var_5 = False
    var_6 = {var_3: var_5, var_4: var_5}
    var_7 = {var_2: var_6}
    var_8 = {var_1: var_7}
    var_9 = {}
    var_10 = 'nested'
    var_11 = 'above'
    var_12 = 'straight'
    var_13 = {}
    var_14 = {}
    var_15 = {}
    var_16 = {var_1: var_15}
    var_17 = {}
    var_18 = module_0.Config()
    var_19 = [var_2]
    var_20 = 'module1.func1'
    var_21 = [var_20]
    var_22 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'THIRDPARTY'
    var_1 = 'from'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = 'nested'
    var_6 = 'above'
    var_7 = 'straight'
    var_8 = {}
    var_9 = {}
    var_10 = {}
    var_11 = {var_1: var_10}
    var_12 = {}
    var_13 = module_0.Config()
    var_14 = []
    var_15 = []
    var_16 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'THIRDPARTY'
    var_1 = 'from'
    var_2 = 'module1'
    var_3 = 'func1'
    var_4 = 'func2'
    var_5 = False
    var_6 = {var_3: var_5, var_4: var_5}
    var_7 = {var_2: var_6}
    var_8 = {var_1: var_7}
    var_9 = {}
    var_10 = 'nested'
    var_11 = 'above'
    var_12 = 'straight'
    var_13 = {}
    var_14 = {}
    var_15 = {}
    var_16 = {var_1: var_15}
    var_17 = {}
    var_18 = module_0.Config()
    var_19 = [var_2]
    var_20 = []
    var_21 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'THIRDPARTY'
    var_1 = 'from'
    var_2 = 'module1'
    var_3 = '*'
    var_4 = False
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {}
    var_9 = 'nested'
    var_10 = 'above'
    var_11 = 'straight'
    var_12 = {}
    var_13 = {}
    var_14 = {}
    var_15 = {var_1: var_14}
    var_16 = {}
    var_17 = module_0.Config()
    var_18 = [var_2]
    var_19 = []
    var_20 = 'import'



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_with_from_imports_single_module. Retrieved 24/33 statements.
# Partially parsed test_with_from_imports_with_remove_imports. Retrieved 25/34 statements.
# Partially parsed test_with_from_imports_with_star_import. Retrieved 25/35 statements.
# Partially parsed test_with_from_imports_force_single_line. Retrieved 26/37 statements.
# Partially parsed test_with_from_imports_with_as_imports. Retrieved 27/37 statements.


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = module_1.Config()
    var_2 = []
    var_3 = 'THIRDPARTY'
    var_4 = []
    var_5 = 'import'
    var_6 = module_2._with_from_imports(var_0, var_1, var_2, var_3, var_4, var_5)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'module'
    var_4 = 'func'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = 'straight'
    var_12 = {}
    var_13 = {}
    var_14 = {var_2: var_13}
    var_15 = {}
    var_16 = {}
    var_17 = {}
    var_18 = module_1.Config()
    var_19 = [var_3]
    var_20 = 'THIRDPARTY'
    var_21 = []
    var_22 = 'import'
    var_23 = module_2._with_from_imports(var_0, var_18, var_19, var_20, var_21, var_22)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'module'
    var_4 = 'func'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = 'straight'
    var_12 = {}
    var_13 = {}
    var_14 = {var_2: var_13}
    var_15 = {}
    var_16 = {}
    var_17 = {}
    var_18 = module_1.Config()
    var_19 = 'other'
    var_20 = [var_3, var_19]
    var_21 = 'THIRDPARTY'
    var_22 = [var_3]
    var_23 = 'import'
    var_24 = module_2._with_from_imports(var_0, var_18, var_20, var_21, var_22, var_23)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'module'
    var_4 = '*'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = 'straight'
    var_12 = {}
    var_13 = {}
    var_14 = {var_2: var_13}
    var_15 = {}
    var_16 = {var_3: var_15}
    var_17 = {}
    var_18 = {}
    var_19 = module_1.Config()
    var_20 = [var_3]
    var_21 = 'THIRDPARTY'
    var_22 = []
    var_23 = 'import'
    var_24 = module_2._with_from_imports(var_0, var_19, var_20, var_21, var_22, var_23)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'module'
    var_4 = 'func1'
    var_5 = 'func2'
    var_6 = False
    var_7 = {var_4: var_6, var_5: var_6}
    var_8 = {var_3: var_7}
    var_9 = {var_2: var_8}
    var_10 = 'above'
    var_11 = 'nested'
    var_12 = 'straight'
    var_13 = {}
    var_14 = {}
    var_15 = {var_2: var_14}
    var_16 = {}
    var_17 = {var_3: var_16}
    var_18 = {}
    var_19 = {}
    var_20 = module_1.Config()
    var_21 = [var_3]
    var_22 = 'THIRDPARTY'
    var_23 = []
    var_24 = 'import'
    var_25 = module_2._with_from_imports(var_0, var_20, var_21, var_22, var_23, var_24)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'THIRDPARTY'
    var_2 = 'from'
    var_3 = 'module'
    var_4 = 'func'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = 'straight'
    var_12 = {}
    var_13 = {}
    var_14 = {var_2: var_13}
    var_15 = {}
    var_16 = {}
    var_17 = 'module.func'
    var_18 = 'alias'
    var_19 = [var_18]
    var_20 = {var_17: var_19}
    var_21 = module_1.Config()
    var_22 = [var_3]
    var_23 = 'THIRDPARTY'
    var_24 = []
    var_25 = 'import'
    var_26 = module_2._with_from_imports(var_0, var_21, var_22, var_23, var_24, var_25)



# Parsed testcases at query #50
#--------------------------






# Parsed testcases at query #51
#--------------------------

# Partially parsed test_with_from_imports_with_single_module. Retrieved 36/40 statements.
# Partially parsed test_with_from_imports_with_remove_imports. Retrieved 36/40 statements.
# Partially parsed test_with_from_imports_with_star_import. Retrieved 36/40 statements.


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = {}
    var_3 = 'from'
    var_4 = 'straight'
    var_5 = {}
    var_6 = {}
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'above'
    var_9 = 'nested'
    var_10 = {}
    var_11 = {}
    var_12 = {var_3: var_11}
    var_13 = {}
    var_14 = {}
    var_15 = {var_3: var_10, var_8: var_12, var_9: var_13, var_4: var_14}
    var_16 = '\n'
    var_17 = set()
    var_18 = {}
    var_19 = {}
    var_20 = ''
    var_21 = set()
    var_22 = module_0.ParsedContent()
    var_23 = module_1.Config()
    var_24 = []
    var_25 = 'STDLIB'
    var_26 = []
    var_27 = 'import'
    var_28 = module_2._with_from_imports(var_22, var_23, var_24, var_25, var_26, var_27)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = 'STDLIB'
    var_3 = 'from'
    var_4 = 'os'
    var_5 = 'path'
    var_6 = False
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = {var_3: var_8}
    var_10 = {var_2: var_9}
    var_11 = 'straight'
    var_12 = {}
    var_13 = {}
    var_14 = {var_3: var_12, var_11: var_13}
    var_15 = 'above'
    var_16 = 'nested'
    var_17 = {}
    var_18 = {}
    var_19 = {var_3: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = {var_3: var_17, var_15: var_19, var_16: var_20, var_11: var_21}
    var_23 = '\n'
    var_24 = set()
    var_25 = {}
    var_26 = {}
    var_27 = ''
    var_28 = set()
    var_29 = module_0.ParsedContent()
    var_30 = module_1.Config()
    var_31 = [var_4]
    var_32 = []
    var_33 = 'import'
    var_34 = module_2._with_from_imports(var_29, var_30, var_31, var_2, var_32, var_33)
    var_35 = len(var_34)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = 'STDLIB'
    var_3 = 'from'
    var_4 = 'os'
    var_5 = 'path'
    var_6 = False
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = {var_3: var_8}
    var_10 = {var_2: var_9}
    var_11 = 'straight'
    var_12 = {}
    var_13 = {}
    var_14 = {var_3: var_12, var_11: var_13}
    var_15 = 'above'
    var_16 = 'nested'
    var_17 = {}
    var_18 = {}
    var_19 = {var_3: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = {var_3: var_17, var_15: var_19, var_16: var_20, var_11: var_21}
    var_23 = '\n'
    var_24 = set()
    var_25 = {}
    var_26 = {}
    var_27 = ''
    var_28 = set()
    var_29 = module_0.ParsedContent()
    var_30 = module_1.Config()
    var_31 = [var_4]
    var_32 = 'os.path'
    var_33 = [var_32]
    var_34 = 'import'
    var_35 = module_2._with_from_imports(var_29, var_30, var_31, var_2, var_33, var_34)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = 'STDLIB'
    var_3 = 'from'
    var_4 = 'os'
    var_5 = 'path'
    var_6 = False
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = {var_3: var_8}
    var_10 = {var_2: var_9}
    var_11 = 'straight'
    var_12 = {}
    var_13 = {}
    var_14 = {var_3: var_12, var_11: var_13}
    var_15 = 'above'
    var_16 = 'nested'
    var_17 = {}
    var_18 = {}
    var_19 = {var_3: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = {var_3: var_17, var_15: var_19, var_16: var_20, var_11: var_21}
    var_23 = '\n'
    var_24 = set()
    var_25 = {}
    var_26 = {}
    var_27 = ''
    var_28 = set()
    var_29 = module_0.ParsedContent()
    var_30 = module_1.Config()
    var_31 = [var_4]
    var_32 = [var_4]
    var_33 = 'import'
    var_34 = module_2._with_from_imports(var_29, var_30, var_31, var_2, var_32, var_33)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = 'STDLIB'
    var_3 = 'from'
    var_4 = 'os'
    var_5 = '*'
    var_6 = False
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = {var_3: var_8}
    var_10 = {var_2: var_9}
    var_11 = 'straight'
    var_12 = {}
    var_13 = {}
    var_14 = {var_3: var_12, var_11: var_13}
    var_15 = 'above'
    var_16 = 'nested'
    var_17 = {}
    var_18 = {}
    var_19 = {var_3: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = {var_3: var_17, var_15: var_19, var_16: var_20, var_11: var_21}
    var_23 = '\n'
    var_24 = set()
    var_25 = {}
    var_26 = {}
    var_27 = ''
    var_28 = set()
    var_29 = module_0.ParsedContent()
    var_30 = True
    var_31 = module_1.Config()
    var_32 = [var_4]
    var_33 = []
    var_34 = 'import'
    var_35 = module_2._with_from_imports(var_29, var_31, var_32, var_2, var_33, var_34)



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_no_sections_config_creates_no_sections_key. Retrieved 31/39 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.Config()
    var_3 = 'py'
    var_4 = 'import'
    var_5 = module_1.ParsedContent()
    var_6 = 'FUTURE'
    var_7 = 'STDLIB'
    var_8 = 'THIRDPARTY'
    var_9 = 'FIRSTPARTY'
    var_10 = 'LOCALFOLDER'
    var_11 = 'straight'
    var_12 = 'from'
    var_13 = {}
    var_14 = {}
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = 'os'
    var_17 = []
    var_18 = {var_16: var_17}
    var_19 = {}
    var_20 = {var_11: var_18, var_12: var_19}
    var_21 = {}
    var_22 = {}
    var_23 = {var_11: var_21, var_12: var_22}
    var_24 = {}
    var_25 = {}
    var_26 = {var_11: var_24, var_12: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = {var_11: var_27, var_12: var_28}
    var_30 = module_0.Config()



# Parsed testcases at query #53
#--------------------------






# Parsed testcases at query #54
#--------------------------

# Partially parsed test_with_from_imports_basic. Retrieved 43/47 statements.
# Partially parsed test_with_from_imports_with_remove_imports. Retrieved 40/44 statements.
# Partially parsed test_with_from_imports_force_single_line. Retrieved 38/42 statements.
# Partially parsed test_with_from_imports_combine_star. Retrieved 38/42 statements.


import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'FUTURE'
    var_2 = 'STDLIB'
    var_3 = 'from'
    var_4 = 'straight'
    var_5 = {}
    var_6 = {}
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'os'
    var_9 = 'path'
    var_10 = True
    var_11 = {var_9: var_10}
    var_12 = {var_8: var_11}
    var_13 = {}
    var_14 = {var_3: var_12, var_4: var_13}
    var_15 = {var_1: var_7, var_2: var_14}
    var_16 = {}
    var_17 = {}
    var_18 = {var_3: var_16, var_4: var_17}
    var_19 = 'above'
    var_20 = 'nested'
    var_21 = {}
    var_22 = {}
    var_23 = {var_3: var_22}
    var_24 = {}
    var_25 = {}
    var_26 = {var_3: var_21, var_19: var_23, var_20: var_24, var_4: var_25}
    var_27 = 0
    var_28 = lambda x: var_2
    var_29 = '\n'
    var_30 = False
    var_31 = [var_1, var_2]
    var_32 = '    '
    var_33 = None
    var_34 = {}
    var_35 = {}
    var_36 = set()
    var_37 = module_1.ParsedContent()
    var_38 = [var_8]
    var_39 = []
    var_40 = 'import'
    var_41 = module_2._with_from_imports(var_37, var_0, var_38, var_2, var_39, var_40)
    var_42 = len(var_41)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'STDLIB'
    var_2 = 'from'
    var_3 = 'straight'
    var_4 = 'os'
    var_5 = 'path'
    var_6 = 'getcwd'
    var_7 = True
    var_8 = {var_5: var_7, var_6: var_7}
    var_9 = {var_4: var_8}
    var_10 = {}
    var_11 = {var_2: var_9, var_3: var_10}
    var_12 = {var_1: var_11}
    var_13 = {}
    var_14 = {}
    var_15 = {var_2: var_13, var_3: var_14}
    var_16 = 'above'
    var_17 = 'nested'
    var_18 = {}
    var_19 = {}
    var_20 = {var_2: var_19}
    var_21 = {}
    var_22 = {}
    var_23 = {var_2: var_18, var_16: var_20, var_17: var_21, var_3: var_22}
    var_24 = 0
    var_25 = lambda x: var_1
    var_26 = '\n'
    var_27 = False
    var_28 = [var_1]
    var_29 = '    '
    var_30 = None
    var_31 = {}
    var_32 = {}
    var_33 = set()
    var_34 = module_1.ParsedContent()
    var_35 = [var_4]
    var_36 = 'os.path'
    var_37 = [var_36]
    var_38 = 'import'
    var_39 = module_2._with_from_imports(var_34, var_0, var_35, var_1, var_37, var_38)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'STDLIB'
    var_2 = 'from'
    var_3 = 'straight'
    var_4 = {}
    var_5 = {}
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = {}
    var_9 = {}
    var_10 = {var_2: var_8, var_3: var_9}
    var_11 = 'above'
    var_12 = 'nested'
    var_13 = {}
    var_14 = {}
    var_15 = {var_2: var_14}
    var_16 = {}
    var_17 = {}
    var_18 = {var_2: var_13, var_11: var_15, var_12: var_16, var_3: var_17}
    var_19 = 0
    var_20 = lambda x: var_1
    var_21 = '\n'
    var_22 = False
    var_23 = [var_1]
    var_24 = '    '
    var_25 = None
    var_26 = {}
    var_27 = {}
    var_28 = set()
    var_29 = module_1.ParsedContent()
    var_30 = []
    var_31 = []
    var_32 = 'import'
    var_33 = module_2._with_from_imports(var_29, var_0, var_30, var_1, var_31, var_32)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = 'STDLIB'
    var_3 = 'from'
    var_4 = 'straight'
    var_5 = 'os'
    var_6 = 'path'
    var_7 = {var_6: var_0}
    var_8 = {var_5: var_7}
    var_9 = {}
    var_10 = {var_3: var_8, var_4: var_9}
    var_11 = {var_2: var_10}
    var_12 = {}
    var_13 = {}
    var_14 = {var_3: var_12, var_4: var_13}
    var_15 = 'above'
    var_16 = 'nested'
    var_17 = {}
    var_18 = {}
    var_19 = {var_3: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = {var_3: var_17, var_15: var_19, var_16: var_20, var_4: var_21}
    var_23 = 0
    var_24 = lambda x: var_2
    var_25 = '\n'
    var_26 = False
    var_27 = [var_2]
    var_28 = '    '
    var_29 = None
    var_30 = {}
    var_31 = {}
    var_32 = set()
    var_33 = module_1.ParsedContent()
    var_34 = [var_5]
    var_35 = []
    var_36 = 'import'
    var_37 = module_2._with_from_imports(var_33, var_1, var_34, var_2, var_35, var_36)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = 'STDLIB'
    var_3 = 'from'
    var_4 = 'straight'
    var_5 = 'os'
    var_6 = '*'
    var_7 = {var_6: var_0}
    var_8 = {var_5: var_7}
    var_9 = {}
    var_10 = {var_3: var_8, var_4: var_9}
    var_11 = {var_2: var_10}
    var_12 = {}
    var_13 = {}
    var_14 = {var_3: var_12, var_4: var_13}
    var_15 = 'above'
    var_16 = 'nested'
    var_17 = {}
    var_18 = {}
    var_19 = {var_3: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = {var_3: var_17, var_15: var_19, var_16: var_20, var_4: var_21}
    var_23 = 0
    var_24 = lambda x: var_2
    var_25 = '\n'
    var_26 = False
    var_27 = [var_2]
    var_28 = '    '
    var_29 = None
    var_30 = {}
    var_31 = {}
    var_32 = set()
    var_33 = module_1.ParsedContent()
    var_34 = [var_5]
    var_35 = []
    var_36 = 'import'
    var_37 = module_2._with_from_imports(var_33, var_1, var_34, var_2, var_35, var_36)



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_with_from_imports_basic. Retrieved 10/12 statements.
# Partially parsed test_with_from_imports_with_remove_imports. Retrieved 10/12 statements.
# Partially parsed test_with_from_imports_force_single_line. Retrieved 10/12 statements.
# Partially parsed test_with_from_imports_combine_star. Retrieved 10/12 statements.
# Partially parsed test_with_from_imports_no_inline_sort. Retrieved 10/12 statements.
# Partially parsed test_with_from_imports_split_on_trailing_comma. Retrieved 10/12 statements.
# Partially parsed test_with_from_imports_line_length_exceeded. Retrieved 10/12 statements.
# Partially parsed test_with_from_imports_ignore_comments. Retrieved 10/12 statements.
# Partially parsed test_with_from_imports_multiple_modules. Retrieved 10/12 statements.


import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'from module import a, b'
    var_2 = module_1.file_contents(var_1)
    var_3 = 'module'
    var_4 = [var_3]
    var_5 = 'THIRDPARTY'
    var_6 = []
    var_7 = 'import'
    var_8 = module_2._with_from_imports(var_2, var_0, var_4, var_5, var_6, var_7)
    var_9 = len(var_8)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = ''
    var_2 = module_1.file_contents(var_1)
    var_3 = []
    var_4 = 'THIRDPARTY'
    var_5 = []
    var_6 = 'import'
    var_7 = module_2._with_from_imports(var_2, var_0, var_3, var_4, var_5, var_6)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'from module import a, b'
    var_2 = module_1.file_contents(var_1)
    var_3 = 'module'
    var_4 = [var_3]
    var_5 = 'THIRDPARTY'
    var_6 = 'module.a'
    var_7 = [var_6]
    var_8 = 'import'
    var_9 = module_2._with_from_imports(var_2, var_0, var_4, var_5, var_7, var_8)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = 'from module import a, b'
    var_3 = module_1.file_contents(var_2)
    var_4 = 'module'
    var_5 = [var_4]
    var_6 = 'THIRDPARTY'
    var_7 = []
    var_8 = 'import'
    var_9 = module_2._with_from_imports(var_3, var_1, var_5, var_6, var_7, var_8)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = 'from module import *'
    var_3 = module_1.file_contents(var_2)
    var_4 = 'module'
    var_5 = [var_4]
    var_6 = 'THIRDPARTY'
    var_7 = []
    var_8 = 'import'
    var_9 = module_2._with_from_imports(var_3, var_1, var_5, var_6, var_7, var_8)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = 'from module import z, a, m'
    var_3 = module_1.file_contents(var_2)
    var_4 = 'module'
    var_5 = [var_4]
    var_6 = 'THIRDPARTY'
    var_7 = []
    var_8 = 'import'
    var_9 = module_2._with_from_imports(var_3, var_1, var_5, var_6, var_7, var_8)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = 'from module import a, b,'
    var_3 = module_1.file_contents(var_2)
    var_4 = 'module'
    var_5 = [var_4]
    var_6 = 'THIRDPARTY'
    var_7 = []
    var_8 = 'import'
    var_9 = module_2._with_from_imports(var_3, var_1, var_5, var_6, var_7, var_8)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 20
    var_1 = module_0.Config()
    var_2 = 'from module import verylongname1, verylongname2, verylongname3'
    var_3 = module_1.file_contents(var_2)
    var_4 = 'module'
    var_5 = [var_4]
    var_6 = 'THIRDPARTY'
    var_7 = []
    var_8 = 'import'
    var_9 = module_2._with_from_imports(var_3, var_1, var_5, var_6, var_7, var_8)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = 'from module import a  # comment'
    var_3 = module_1.file_contents(var_2)
    var_4 = 'module'
    var_5 = [var_4]
    var_6 = 'THIRDPARTY'
    var_7 = []
    var_8 = 'import'
    var_9 = module_2._with_from_imports(var_3, var_1, var_5, var_6, var_7, var_8)

import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'from module1 import a\nfrom module2 import b'
    var_2 = module_1.file_contents(var_1)
    var_3 = 'module1'
    var_4 = 'module2'
    var_5 = [var_3, var_4]
    var_6 = 'THIRDPARTY'
    var_7 = []
    var_8 = 'import'
    var_9 = module_2._with_from_imports(var_2, var_0, var_5, var_6, var_7, var_8)



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_predicate_at_line_81_evaluates_to_true. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'test_module'



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_sorted_imports_normalize_empty_lines. Retrieved 43/47 statements.
# Partially parsed test_sorted_imports_with_from_imports. Retrieved 46/50 statements.
# Partially parsed test_sorted_imports_with_line_separator. Retrieved 42/46 statements.


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = -1
    var_1 = "print('hello')"
    var_2 = 'x = 1'
    var_3 = [var_1, var_2]
    var_4 = '\n'
    var_5 = {}
    var_6 = {}
    var_7 = {}
    var_8 = {}
    var_9 = {}
    var_10 = []
    var_11 = 2
    var_12 = module_0.ParsedContent()
    var_13 = module_1.Config()
    var_14 = module_2.sorted_imports(var_12, var_13)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'x = 1'
    var_2 = [var_1]
    var_3 = '\n'
    var_4 = {}
    var_5 = {}
    var_6 = 'FUTURE'
    var_7 = 'STDLIB'
    var_8 = 'THIRDPARTY'
    var_9 = 'FIRSTPARTY'
    var_10 = 'LOCALFOLDER'
    var_11 = 'straight'
    var_12 = 'from'
    var_13 = {}
    var_14 = {}
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = 'os'
    var_17 = 'sys'
    var_18 = None
    var_19 = {var_16: var_18, var_17: var_18}
    var_20 = {}
    var_21 = {var_11: var_19, var_12: var_20}
    var_22 = {}
    var_23 = {}
    var_24 = {var_11: var_22, var_12: var_23}
    var_25 = {}
    var_26 = {}
    var_27 = {var_11: var_25, var_12: var_26}
    var_28 = {}
    var_29 = {}
    var_30 = {var_11: var_28, var_12: var_29}
    var_31 = {var_6: var_15, var_7: var_21, var_8: var_24, var_9: var_27, var_10: var_30}
    var_32 = {}
    var_33 = {}
    var_34 = {var_11: var_32, var_12: var_33}
    var_35 = 'above'
    var_36 = {}
    var_37 = {var_11: var_36}
    var_38 = {}
    var_39 = {var_35: var_37, var_11: var_38}
    var_40 = [var_6, var_7, var_8, var_9, var_10]
    var_41 = 1
    var_42 = module_0.ParsedContent()
    var_43 = module_1.Config()
    var_44 = module_2.sorted_imports(var_42, var_43)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = ''
    var_2 = 'x = 1'
    var_3 = [var_1, var_1, var_2]
    var_4 = '\n'
    var_5 = {}
    var_6 = {}
    var_7 = 'FUTURE'
    var_8 = 'STDLIB'
    var_9 = 'THIRDPARTY'
    var_10 = 'FIRSTPARTY'
    var_11 = 'LOCALFOLDER'
    var_12 = 'straight'
    var_13 = 'from'
    var_14 = {}
    var_15 = {}
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = {}
    var_18 = {}
    var_19 = {var_12: var_17, var_13: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = {var_12: var_20, var_13: var_21}
    var_23 = {}
    var_24 = {}
    var_25 = {var_12: var_23, var_13: var_24}
    var_26 = {}
    var_27 = {}
    var_28 = {var_12: var_26, var_13: var_27}
    var_29 = {var_7: var_16, var_8: var_19, var_9: var_22, var_10: var_25, var_11: var_28}
    var_30 = {}
    var_31 = {}
    var_32 = {var_12: var_30, var_13: var_31}
    var_33 = 'above'
    var_34 = {}
    var_35 = {var_12: var_34}
    var_36 = {}
    var_37 = {var_33: var_35, var_12: var_36}
    var_38 = [var_7, var_8, var_9, var_10, var_11]
    var_39 = 3
    var_40 = module_0.ParsedContent()
    var_41 = module_1.Config()
    var_42 = module_2.sorted_imports(var_40, var_41)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'x = 1'
    var_2 = [var_1]
    var_3 = '\n'
    var_4 = {}
    var_5 = {}
    var_6 = 'FUTURE'
    var_7 = 'STDLIB'
    var_8 = 'THIRDPARTY'
    var_9 = 'FIRSTPARTY'
    var_10 = 'LOCALFOLDER'
    var_11 = 'straight'
    var_12 = 'from'
    var_13 = {}
    var_14 = {}
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = {}
    var_17 = 'os'
    var_18 = 'path'
    var_19 = 'getcwd'
    var_20 = [var_18, var_19]
    var_21 = {var_17: var_20}
    var_22 = {var_11: var_16, var_12: var_21}
    var_23 = {}
    var_24 = {}
    var_25 = {var_11: var_23, var_12: var_24}
    var_26 = {}
    var_27 = {}
    var_28 = {var_11: var_26, var_12: var_27}
    var_29 = {}
    var_30 = {}
    var_31 = {var_11: var_29, var_12: var_30}
    var_32 = {var_6: var_15, var_7: var_22, var_8: var_25, var_9: var_28, var_10: var_31}
    var_33 = {}
    var_34 = {}
    var_35 = {var_11: var_33, var_12: var_34}
    var_36 = 'above'
    var_37 = {}
    var_38 = {var_11: var_37}
    var_39 = {}
    var_40 = {var_36: var_38, var_11: var_39}
    var_41 = [var_6, var_7, var_8, var_9, var_10]
    var_42 = 1
    var_43 = module_0.ParsedContent()
    var_44 = module_1.Config()
    var_45 = module_2.sorted_imports(var_43, var_44)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'x = 1'
    var_2 = [var_1]
    var_3 = '\n'
    var_4 = {}
    var_5 = {}
    var_6 = 'FUTURE'
    var_7 = 'STDLIB'
    var_8 = 'THIRDPARTY'
    var_9 = 'FIRSTPARTY'
    var_10 = 'LOCALFOLDER'
    var_11 = 'straight'
    var_12 = 'from'
    var_13 = {}
    var_14 = {}
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = 'os'
    var_17 = None
    var_18 = {var_16: var_17}
    var_19 = {}
    var_20 = {var_11: var_18, var_12: var_19}
    var_21 = {}
    var_22 = {}
    var_23 = {var_11: var_21, var_12: var_22}
    var_24 = {}
    var_25 = {}
    var_26 = {var_11: var_24, var_12: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = {var_11: var_27, var_12: var_28}
    var_30 = {var_6: var_15, var_7: var_20, var_8: var_23, var_9: var_26, var_10: var_29}
    var_31 = {}
    var_32 = {}
    var_33 = {var_11: var_31, var_12: var_32}
    var_34 = 'above'
    var_35 = {}
    var_36 = {var_11: var_35}
    var_37 = {}
    var_38 = {var_34: var_36, var_11: var_37}
    var_39 = [var_6, var_7, var_8, var_9, var_10]
    var_40 = 1
    var_41 = module_0.ParsedContent()
    var_42 = [var_16]
    var_43 = module_1.Config()
    var_44 = module_2.sorted_imports(var_41, var_43)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'x = 1'
    var_2 = [var_1]
    var_3 = '\r\n'
    var_4 = {}
    var_5 = {}
    var_6 = 'FUTURE'
    var_7 = 'STDLIB'
    var_8 = 'THIRDPARTY'
    var_9 = 'FIRSTPARTY'
    var_10 = 'LOCALFOLDER'
    var_11 = 'straight'
    var_12 = 'from'
    var_13 = {}
    var_14 = {}
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = {}
    var_17 = {}
    var_18 = {var_11: var_16, var_12: var_17}
    var_19 = {}
    var_20 = {}
    var_21 = {var_11: var_19, var_12: var_20}
    var_22 = {}
    var_23 = {}
    var_24 = {var_11: var_22, var_12: var_23}
    var_25 = {}
    var_26 = {}
    var_27 = {var_11: var_25, var_12: var_26}
    var_28 = {var_6: var_15, var_7: var_18, var_8: var_21, var_9: var_24, var_10: var_27}
    var_29 = {}
    var_30 = {}
    var_31 = {var_11: var_29, var_12: var_30}
    var_32 = 'above'
    var_33 = {}
    var_34 = {var_11: var_33}
    var_35 = {}
    var_36 = {var_32: var_34, var_11: var_35}
    var_37 = [var_6, var_7, var_8, var_9, var_10]
    var_38 = 1
    var_39 = module_0.ParsedContent()
    var_40 = module_1.Config()
    var_41 = module_2.sorted_imports(var_39, var_40)



# Parsed testcases at query #58
#--------------------------






# Parsed testcases at query #59
#--------------------------

# Partially parsed test_sorted_imports_normalizes_output. Retrieved 26/30 statements.
# Partially parsed test_sorted_imports_with_no_sections. Retrieved 32/36 statements.
# Partially parsed test_sorted_imports_with_from_first. Retrieved 32/36 statements.
# Partially parsed test_sorted_imports_with_ensure_newline_before_comments. Retrieved 26/30 statements.
# Partially parsed test_sorted_imports_with_lines_between_sections. Retrieved 33/37 statements.


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = -1
    var_1 = "print('hello')"
    var_2 = 'x = 1'
    var_3 = [var_1, var_2]
    var_4 = '\n'
    var_5 = []
    var_6 = {}
    var_7 = {}
    var_8 = 'straight'
    var_9 = 'from'
    var_10 = {}
    var_11 = {}
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = {}
    var_14 = 'above'
    var_15 = {}
    var_16 = {var_8: var_15}
    var_17 = {}
    var_18 = {var_14: var_16, var_8: var_17}
    var_19 = module_0.ParsedContent()
    var_20 = module_1.Config()
    var_21 = module_2.sorted_imports(var_19, var_20)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'x = 1'
    var_2 = [var_1]
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = [var_4]
    var_6 = {}
    var_7 = {}
    var_8 = 'straight'
    var_9 = 'from'
    var_10 = {}
    var_11 = {}
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = 'os'
    var_14 = 'sys'
    var_15 = None
    var_16 = {var_13: var_15, var_14: var_15}
    var_17 = {}
    var_18 = {var_8: var_16, var_9: var_17}
    var_19 = {var_4: var_18}
    var_20 = 'above'
    var_21 = {}
    var_22 = {var_8: var_21}
    var_23 = {}
    var_24 = {var_20: var_22, var_8: var_23}
    var_25 = module_0.ParsedContent()
    var_26 = module_1.Config()
    var_27 = module_2.sorted_imports(var_25, var_26)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = ''
    var_2 = 'x = 1'
    var_3 = [var_1, var_1, var_2]
    var_4 = '\n'
    var_5 = 'STDLIB'
    var_6 = [var_5]
    var_7 = {}
    var_8 = {}
    var_9 = 'straight'
    var_10 = 'from'
    var_11 = {}
    var_12 = {}
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = {}
    var_15 = {}
    var_16 = {var_9: var_14, var_10: var_15}
    var_17 = {var_5: var_16}
    var_18 = 'above'
    var_19 = {}
    var_20 = {var_9: var_19}
    var_21 = {}
    var_22 = {var_18: var_20, var_9: var_21}
    var_23 = module_0.ParsedContent()
    var_24 = module_1.Config()
    var_25 = module_2.sorted_imports(var_23, var_24)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'x = 1'
    var_2 = [var_1]
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'THIRDPARTY'
    var_6 = [var_4, var_5]
    var_7 = {}
    var_8 = {}
    var_9 = 'straight'
    var_10 = 'from'
    var_11 = {}
    var_12 = {}
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = 'os'
    var_15 = None
    var_16 = {var_14: var_15}
    var_17 = {}
    var_18 = {var_9: var_16, var_10: var_17}
    var_19 = {}
    var_20 = {}
    var_21 = {var_9: var_19, var_10: var_20}
    var_22 = {var_4: var_18, var_5: var_21}
    var_23 = 'above'
    var_24 = {}
    var_25 = {var_9: var_24}
    var_26 = {}
    var_27 = {var_23: var_25, var_9: var_26}
    var_28 = module_0.ParsedContent()
    var_29 = True
    var_30 = module_1.Config()
    var_31 = module_2.sorted_imports(var_28, var_30)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'x = 1'
    var_2 = [var_1]
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = [var_4]
    var_6 = {}
    var_7 = {}
    var_8 = 'straight'
    var_9 = 'from'
    var_10 = {}
    var_11 = {}
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = 'os'
    var_14 = None
    var_15 = {var_13: var_14}
    var_16 = {}
    var_17 = {var_8: var_15, var_9: var_16}
    var_18 = {var_4: var_17}
    var_19 = 'above'
    var_20 = {}
    var_21 = {var_8: var_20}
    var_22 = {}
    var_23 = {var_19: var_21, var_8: var_22}
    var_24 = module_0.ParsedContent()
    var_25 = 'import os'
    var_26 = [var_25]
    var_27 = module_1.Config()
    var_28 = module_2.sorted_imports(var_24, var_27)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'x = 1'
    var_2 = [var_1]
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = [var_4]
    var_6 = {}
    var_7 = {}
    var_8 = 'straight'
    var_9 = 'from'
    var_10 = {}
    var_11 = 'os'
    var_12 = []
    var_13 = {var_11: var_12}
    var_14 = {var_8: var_10, var_9: var_13}
    var_15 = 'sys'
    var_16 = None
    var_17 = {var_15: var_16}
    var_18 = 'path'
    var_19 = [var_18]
    var_20 = {var_11: var_19}
    var_21 = {var_8: var_17, var_9: var_20}
    var_22 = {var_4: var_21}
    var_23 = 'above'
    var_24 = {}
    var_25 = {var_8: var_24}
    var_26 = {}
    var_27 = {var_23: var_25, var_8: var_26}
    var_28 = module_0.ParsedContent()
    var_29 = True
    var_30 = module_1.Config()
    var_31 = module_2.sorted_imports(var_28, var_30)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'x = 1'
    var_2 = [var_1]
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = [var_4]
    var_6 = {}
    var_7 = {}
    var_8 = 'straight'
    var_9 = 'from'
    var_10 = {}
    var_11 = {}
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = {}
    var_14 = {}
    var_15 = {var_8: var_13, var_9: var_14}
    var_16 = {var_4: var_15}
    var_17 = 'above'
    var_18 = {}
    var_19 = {var_8: var_18}
    var_20 = {}
    var_21 = {var_17: var_19, var_8: var_20}
    var_22 = module_0.ParsedContent()
    var_23 = True
    var_24 = module_1.Config()
    var_25 = module_2.sorted_imports(var_22, var_24)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'x = 1'
    var_2 = [var_1]
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = [var_4]
    var_6 = {}
    var_7 = {}
    var_8 = 'straight'
    var_9 = 'from'
    var_10 = {}
    var_11 = {}
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = 'os'
    var_14 = None
    var_15 = {var_13: var_14}
    var_16 = {}
    var_17 = {var_8: var_15, var_9: var_16}
    var_18 = {var_4: var_17}
    var_19 = 'above'
    var_20 = {}
    var_21 = {var_8: var_20}
    var_22 = {}
    var_23 = {var_19: var_21, var_8: var_22}
    var_24 = module_0.ParsedContent()
    var_25 = 'stdlib'
    var_26 = 'Standard Library'
    var_27 = {var_25: var_26}
    var_28 = module_1.Config()
    var_29 = module_2.sorted_imports(var_24, var_28)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'x = 1'
    var_2 = [var_1]
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'THIRDPARTY'
    var_6 = [var_4, var_5]
    var_7 = {}
    var_8 = {}
    var_9 = 'straight'
    var_10 = 'from'
    var_11 = {}
    var_12 = {}
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = 'os'
    var_15 = None
    var_16 = {var_14: var_15}
    var_17 = {}
    var_18 = {var_9: var_16, var_10: var_17}
    var_19 = 'django'
    var_20 = {var_19: var_15}
    var_21 = {}
    var_22 = {var_9: var_20, var_10: var_21}
    var_23 = {var_4: var_18, var_5: var_22}
    var_24 = 'above'
    var_25 = {}
    var_26 = {var_9: var_25}
    var_27 = {}
    var_28 = {var_24: var_26, var_9: var_27}
    var_29 = module_0.ParsedContent()
    var_30 = 2
    var_31 = module_1.Config()
    var_32 = module_2.sorted_imports(var_29, var_31)



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_line_153_predicate_evaluates_to_true. Retrieved 6/8 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 153 (output and output[0].strip() == "") evaluates to True.'
    var_1 = ''
    var_2 = 'import os'
    var_3 = [var_1, var_2]
    var_4 = 0
    var_5 = var_3[var_4]



# Parsed testcases at query #61
#--------------------------






# Parsed testcases at query #62
#--------------------------

# Partially parsed test_predicate_line_49_evaluates_to_true. Retrieved 14/49 statements.


def test_case_0():
    var_0 = '# header'
    var_1 = 'FUTURE'
    var_2 = 'STDLIB'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = {}
    var_6 = {}
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {}
    var_9 = 'os'
    var_10 = 'path'
    var_11 = [var_10]
    var_12 = {var_9: var_11}
    var_13 = {var_3: var_8, var_4: var_12}



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_predicate_at_line_44_evaluates_to_true. Retrieved 25/29 statements.


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'Test that the predicate at line 44 (for module in straight_modules:) evaluates to True\n    when straight_modules is not empty and combine_straight_imports is False or as_imports is True.'
    var_1 = 0
    var_2 = 'THIRDPARTY'
    var_3 = 'straight'
    var_4 = 'module1'
    var_5 = None
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = {}
    var_10 = {var_3: var_9}
    var_11 = 'above'
    var_12 = {}
    var_13 = {var_3: var_12}
    var_14 = {}
    var_15 = {var_11: var_13, var_3: var_14}
    var_16 = module_0.ParsedContent()
    var_17 = False
    var_18 = module_1.Config()
    var_19 = [var_4]
    var_20 = 'THIRDPARTY'
    var_21 = []
    var_22 = 'import'
    var_23 = module_2._with_straight_imports(var_16, var_18, var_19, var_20, var_21, var_22)
    var_24 = len(var_23)



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_line_279_predicate_evaluates_true. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 'from module import something'
    var_1 = 'test_module'
    var_2 = 'test_module'



# Parsed testcases at query #65
#--------------------------






# Parsed testcases at query #66
#--------------------------




import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = -1
    var_1 = "print('hello')\n"
    var_2 = [var_1]
    var_3 = {}
    var_4 = {}
    var_5 = {}
    var_6 = {}
    var_7 = {}
    var_8 = []
    var_9 = '\n'
    var_10 = 1
    var_11 = module_0.ParsedContent()
    var_12 = module_1.Config()
    var_13 = module_2.sorted_imports(var_11, var_12)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = "print('hello')\n"
    var_2 = [var_1]
    var_3 = {}
    var_4 = {}
    var_5 = 'STDLIB'
    var_6 = 'THIRDPARTY'
    var_7 = 'FIRSTPARTY'
    var_8 = 'LOCALFOLDER'
    var_9 = 'straight'
    var_10 = 'from'
    var_11 = 'os'
    var_12 = None
    var_13 = {var_11: var_12}
    var_14 = {}
    var_15 = {var_9: var_13, var_10: var_14}
    var_16 = {}
    var_17 = {}
    var_18 = {var_9: var_16, var_10: var_17}
    var_19 = {}
    var_20 = {}
    var_21 = {var_9: var_19, var_10: var_20}
    var_22 = {}
    var_23 = {}
    var_24 = {var_9: var_22, var_10: var_23}
    var_25 = {var_5: var_15, var_6: var_18, var_7: var_21, var_8: var_24}
    var_26 = {}
    var_27 = {}
    var_28 = {var_9: var_26, var_10: var_27}
    var_29 = 'above'
    var_30 = {}
    var_31 = {var_9: var_30}
    var_32 = {}
    var_33 = {var_29: var_31, var_9: var_32}
    var_34 = [var_5, var_6, var_7, var_8]
    var_35 = '\n'
    var_36 = 1
    var_37 = module_0.ParsedContent()
    var_38 = module_1.Config()
    var_39 = module_2.sorted_imports(var_37, var_38)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'code\n'
    var_2 = [var_1]
    var_3 = {}
    var_4 = {}
    var_5 = 'STDLIB'
    var_6 = 'THIRDPARTY'
    var_7 = 'FIRSTPARTY'
    var_8 = 'LOCALFOLDER'
    var_9 = 'straight'
    var_10 = 'from'
    var_11 = {}
    var_12 = 'os'
    var_13 = 'path'
    var_14 = [var_13]
    var_15 = {var_12: var_14}
    var_16 = {var_9: var_11, var_10: var_15}
    var_17 = {}
    var_18 = {}
    var_19 = {var_9: var_17, var_10: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = {var_9: var_20, var_10: var_21}
    var_23 = {}
    var_24 = {}
    var_25 = {var_9: var_23, var_10: var_24}
    var_26 = {var_5: var_16, var_6: var_19, var_7: var_22, var_8: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = {var_9: var_27, var_10: var_28}
    var_30 = 'above'
    var_31 = {}
    var_32 = {var_9: var_31}
    var_33 = {}
    var_34 = {var_30: var_32, var_9: var_33}
    var_35 = [var_5, var_6, var_7, var_8]
    var_36 = '\n'
    var_37 = 1
    var_38 = module_0.ParsedContent()
    var_39 = module_1.Config()
    var_40 = module_2.sorted_imports(var_38, var_39)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'code\n'
    var_2 = [var_1]
    var_3 = {}
    var_4 = {}
    var_5 = 'STDLIB'
    var_6 = 'THIRDPARTY'
    var_7 = 'FIRSTPARTY'
    var_8 = 'LOCALFOLDER'
    var_9 = 'straight'
    var_10 = 'from'
    var_11 = 'os'
    var_12 = 'sys'
    var_13 = None
    var_14 = {var_11: var_13, var_12: var_13}
    var_15 = {}
    var_16 = {var_9: var_14, var_10: var_15}
    var_17 = {}
    var_18 = {}
    var_19 = {var_9: var_17, var_10: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = {var_9: var_20, var_10: var_21}
    var_23 = {}
    var_24 = {}
    var_25 = {var_9: var_23, var_10: var_24}
    var_26 = {var_5: var_16, var_6: var_19, var_7: var_22, var_8: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = {var_9: var_27, var_10: var_28}
    var_30 = 'above'
    var_31 = {}
    var_32 = {var_9: var_31}
    var_33 = {}
    var_34 = {var_30: var_32, var_9: var_33}
    var_35 = [var_5, var_6, var_7, var_8]
    var_36 = '\n'
    var_37 = 1
    var_38 = module_0.ParsedContent()
    var_39 = 'import os'
    var_40 = [var_39]
    var_41 = module_1.Config()
    var_42 = module_2.sorted_imports(var_38, var_41)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'code\n'
    var_2 = [var_1]
    var_3 = {}
    var_4 = {}
    var_5 = 'STDLIB'
    var_6 = 'THIRDPARTY'
    var_7 = 'FIRSTPARTY'
    var_8 = 'LOCALFOLDER'
    var_9 = 'straight'
    var_10 = 'from'
    var_11 = 'os'
    var_12 = None
    var_13 = {var_11: var_12}
    var_14 = {}
    var_15 = {var_9: var_13, var_10: var_14}
    var_16 = 'django'
    var_17 = {var_16: var_12}
    var_18 = {}
    var_19 = {var_9: var_17, var_10: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = {var_9: var_20, var_10: var_21}
    var_23 = {}
    var_24 = {}
    var_25 = {var_9: var_23, var_10: var_24}
    var_26 = {var_5: var_15, var_6: var_19, var_7: var_22, var_8: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = {var_9: var_27, var_10: var_28}
    var_30 = 'above'
    var_31 = {}
    var_32 = {var_9: var_31}
    var_33 = {}
    var_34 = {var_30: var_32, var_9: var_33}
    var_35 = [var_5, var_6, var_7, var_8]
    var_36 = '\n'
    var_37 = 1
    var_38 = module_0.ParsedContent()
    var_39 = module_1.Config()
    var_40 = module_2.sorted_imports(var_38, var_39)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'code\n'
    var_2 = [var_1]
    var_3 = {}
    var_4 = {}
    var_5 = 'FUTURE'
    var_6 = 'STDLIB'
    var_7 = 'THIRDPARTY'
    var_8 = 'FIRSTPARTY'
    var_9 = 'LOCALFOLDER'
    var_10 = 'straight'
    var_11 = 'from'
    var_12 = {}
    var_13 = {}
    var_14 = {var_10: var_12, var_11: var_13}
    var_15 = 'os'
    var_16 = None
    var_17 = {var_15: var_16}
    var_18 = {}
    var_19 = {var_10: var_17, var_11: var_18}
    var_20 = 'django'
    var_21 = {var_20: var_16}
    var_22 = {}
    var_23 = {var_10: var_21, var_11: var_22}
    var_24 = {}
    var_25 = {}
    var_26 = {var_10: var_24, var_11: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = {var_10: var_27, var_11: var_28}
    var_30 = {var_5: var_14, var_6: var_19, var_7: var_23, var_8: var_26, var_9: var_29}
    var_31 = {}
    var_32 = {}
    var_33 = {var_10: var_31, var_11: var_32}
    var_34 = 'above'
    var_35 = {}
    var_36 = {var_10: var_35}
    var_37 = {}
    var_38 = {var_34: var_36, var_10: var_37}
    var_39 = [var_5, var_6, var_7, var_8, var_9]
    var_40 = '\n'
    var_41 = 1
    var_42 = module_0.ParsedContent()
    var_43 = True
    var_44 = module_1.Config()
    var_45 = module_2.sorted_imports(var_42, var_44)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'code\n'
    var_2 = [var_1]
    var_3 = {}
    var_4 = {}
    var_5 = 'STDLIB'
    var_6 = 'THIRDPARTY'
    var_7 = 'FIRSTPARTY'
    var_8 = 'LOCALFOLDER'
    var_9 = 'straight'
    var_10 = 'from'
    var_11 = 'os'
    var_12 = None
    var_13 = {var_11: var_12}
    var_14 = {}
    var_15 = {var_9: var_13, var_10: var_14}
    var_16 = {}
    var_17 = {}
    var_18 = {var_9: var_16, var_10: var_17}
    var_19 = {}
    var_20 = {}
    var_21 = {var_9: var_19, var_10: var_20}
    var_22 = {}
    var_23 = {}
    var_24 = {var_9: var_22, var_10: var_23}
    var_25 = {var_5: var_15, var_6: var_18, var_7: var_21, var_8: var_24}
    var_26 = {}
    var_27 = {}
    var_28 = {var_9: var_26, var_10: var_27}
    var_29 = 'above'
    var_30 = {}
    var_31 = {var_9: var_30}
    var_32 = {}
    var_33 = {var_29: var_31, var_9: var_32}
    var_34 = [var_5, var_6, var_7, var_8]
    var_35 = '\n'
    var_36 = 1
    var_37 = module_0.ParsedContent()
    var_38 = 'stdlib'
    var_39 = 'Standard Library'
    var_40 = {var_38: var_39}
    var_41 = module_1.Config()
    var_42 = module_2.sorted_imports(var_37, var_41)

def test_case_0():
    pass



# Parsed testcases at query #67
#--------------------------






# Parsed testcases at query #68
#--------------------------

# Partially parsed test_predicate_at_line_38_evaluates_to_true. Retrieved 15/23 statements.


import isort.settings as module_0
import isort.parse as module_1
import isort.output as module_2

def test_case_0():
    var_0 = False
    var_1 = module_0.Config()
    var_2 = []
    var_3 = 'py'
    var_4 = 'import'
    var_5 = module_1.ParsedContent()
    var_6 = 'STDLIB'
    var_7 = 'straight'
    var_8 = 'from'
    var_9 = 'os'
    var_10 = []
    var_11 = {var_9: var_10}
    var_12 = {}
    var_13 = {var_7: var_11, var_8: var_12}
    var_14 = module_2.sorted_imports(var_5, var_1)



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_with_from_imports_empty_from_modules. Retrieved 19/25 statements.
# Partially parsed test_with_from_imports_single_import. Retrieved 24/31 statements.
# Partially parsed test_with_from_imports_with_remove_imports. Retrieved 25/32 statements.
# Partially parsed test_with_from_imports_with_comments. Retrieved 26/33 statements.
# Partially parsed test_with_from_imports_force_single_line. Retrieved 26/33 statements.
# Partially parsed test_with_from_imports_star_import. Retrieved 23/30 statements.
# Partially parsed test_with_from_imports_with_as_imports. Retrieved 27/34 statements.
# Partially parsed test_with_from_imports_above_comments. Retrieved 25/32 statements.
# Partially parsed test_with_from_imports_long_line_wrapping. Retrieved 26/33 statements.


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'STDLIB'
    var_2 = 'from'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'above'
    var_6 = 'nested'
    var_7 = 'straight'
    var_8 = {}
    var_9 = {}
    var_10 = {var_2: var_9}
    var_11 = {}
    var_12 = {}
    var_13 = {}
    var_14 = module_1.Config()
    var_15 = []
    var_16 = []
    var_17 = 'import'
    var_18 = module_2._with_from_imports(var_0, var_14, var_15, var_1, var_16, var_17)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'STDLIB'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = None
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = 'straight'
    var_12 = {}
    var_13 = {}
    var_14 = {var_2: var_13}
    var_15 = {}
    var_16 = {}
    var_17 = {}
    var_18 = module_1.Config()
    var_19 = [var_3]
    var_20 = []
    var_21 = 'import'
    var_22 = module_2._with_from_imports(var_0, var_18, var_19, var_1, var_20, var_21)
    var_23 = len(var_22)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'STDLIB'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = 'environ'
    var_6 = None
    var_7 = {var_4: var_6, var_5: var_6}
    var_8 = {var_3: var_7}
    var_9 = {var_2: var_8}
    var_10 = 'above'
    var_11 = 'nested'
    var_12 = 'straight'
    var_13 = {}
    var_14 = {}
    var_15 = {var_2: var_14}
    var_16 = {}
    var_17 = {}
    var_18 = {}
    var_19 = module_1.Config()
    var_20 = [var_3]
    var_21 = 'os.path'
    var_22 = [var_21]
    var_23 = 'import'
    var_24 = module_2._with_from_imports(var_0, var_19, var_20, var_1, var_22, var_23)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'STDLIB'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = None
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = 'straight'
    var_12 = 'important comment'
    var_13 = [var_12]
    var_14 = {var_3: var_13}
    var_15 = {}
    var_16 = {var_2: var_15}
    var_17 = {}
    var_18 = {var_3: var_17}
    var_19 = {}
    var_20 = {}
    var_21 = module_1.Config()
    var_22 = [var_3]
    var_23 = []
    var_24 = 'import'
    var_25 = module_2._with_from_imports(var_0, var_21, var_22, var_1, var_23, var_24)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'STDLIB'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = 'environ'
    var_6 = None
    var_7 = {var_4: var_6, var_5: var_6}
    var_8 = {var_3: var_7}
    var_9 = {var_2: var_8}
    var_10 = 'above'
    var_11 = 'nested'
    var_12 = 'straight'
    var_13 = {}
    var_14 = {}
    var_15 = {var_2: var_14}
    var_16 = {}
    var_17 = {}
    var_18 = {}
    var_19 = True
    var_20 = module_1.Config()
    var_21 = [var_3]
    var_22 = []
    var_23 = 'import'
    var_24 = module_2._with_from_imports(var_0, var_20, var_21, var_1, var_22, var_23)
    var_25 = len(var_24)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'STDLIB'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = '*'
    var_5 = None
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = 'straight'
    var_12 = {}
    var_13 = {}
    var_14 = {var_2: var_13}
    var_15 = {}
    var_16 = {}
    var_17 = {}
    var_18 = module_1.Config()
    var_19 = [var_3]
    var_20 = []
    var_21 = 'import'
    var_22 = module_2._with_from_imports(var_0, var_18, var_19, var_1, var_20, var_21)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'STDLIB'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = None
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = 'straight'
    var_12 = {}
    var_13 = {}
    var_14 = {var_2: var_13}
    var_15 = {}
    var_16 = {}
    var_17 = 'os.path'
    var_18 = 'p'
    var_19 = [var_18]
    var_20 = {var_17: var_19}
    var_21 = True
    var_22 = module_1.Config()
    var_23 = [var_3]
    var_24 = []
    var_25 = 'import'
    var_26 = module_2._with_from_imports(var_0, var_22, var_23, var_1, var_24, var_25)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'STDLIB'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = None
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = 'straight'
    var_12 = {}
    var_13 = '# Above comment'
    var_14 = [var_13]
    var_15 = {var_3: var_14}
    var_16 = {var_2: var_15}
    var_17 = {}
    var_18 = {}
    var_19 = {}
    var_20 = module_1.Config()
    var_21 = [var_3]
    var_22 = []
    var_23 = 'import'
    var_24 = module_2._with_from_imports(var_0, var_20, var_21, var_1, var_22, var_23)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = module_0.ParsedContent()
    var_1 = 'STDLIB'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'path'
    var_5 = 'environ'
    var_6 = 'getcwd'
    var_7 = None
    var_8 = {var_4: var_7, var_5: var_7, var_6: var_7}
    var_9 = {var_3: var_8}
    var_10 = {var_2: var_9}
    var_11 = 'above'
    var_12 = 'nested'
    var_13 = 'straight'
    var_14 = {}
    var_15 = {}
    var_16 = {var_2: var_15}
    var_17 = {}
    var_18 = {}
    var_19 = {}
    var_20 = 40
    var_21 = module_1.Config()
    var_22 = [var_3]
    var_23 = []
    var_24 = 'import'
    var_25 = module_2._with_from_imports(var_0, var_21, var_22, var_1, var_23, var_24)



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_with_straight_imports_predicate_line_1. Retrieved 23/27 statements.


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'Test that the predicate at line 1 (_with_straight_imports function) evaluates to True.'
    var_1 = {}
    var_2 = 'straight'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'above'
    var_6 = {}
    var_7 = {var_2: var_6}
    var_8 = {}
    var_9 = {var_5: var_7, var_2: var_8}
    var_10 = 0
    var_11 = 'THIRDPARTY'
    var_12 = lambda x: var_11
    var_13 = module_0.ParsedContent()
    var_14 = False
    var_15 = module_1.Config()
    var_16 = 'os'
    var_17 = 'sys'
    var_18 = [var_16, var_17]
    var_19 = 'STDLIB'
    var_20 = []
    var_21 = 'import'
    var_22 = module_2._with_straight_imports(var_13, var_15, var_18, var_19, var_20, var_21)



# Parsed testcases at query #71
#--------------------------

# Failed to parse test_formatting_function_predicate.




# Parsed testcases at query #72
#--------------------------






# Parsed testcases at query #73
#--------------------------




import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = -1
    var_1 = 'x = 1'
    var_2 = [var_1]
    var_3 = []
    var_4 = {}
    var_5 = {}
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = {}
    var_9 = {}
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = {}
    var_12 = 'above'
    var_13 = {}
    var_14 = {var_6: var_13}
    var_15 = {}
    var_16 = {var_12: var_14, var_6: var_15}
    var_17 = []
    var_18 = '\n'
    var_19 = 1
    var_20 = module_0.ParsedContent()
    var_21 = module_1.Config()
    var_22 = module_2.sorted_imports(var_20, var_21)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = []
    var_2 = []
    var_3 = {}
    var_4 = {}
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = {}
    var_8 = {}
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'FUTURE'
    var_11 = 'STDLIB'
    var_12 = 'THIRDPARTY'
    var_13 = 'FIRSTPARTY'
    var_14 = 'LOCALFOLDER'
    var_15 = {}
    var_16 = {}
    var_17 = {var_5: var_15, var_6: var_16}
    var_18 = 'os'
    var_19 = None
    var_20 = {var_18: var_19}
    var_21 = {}
    var_22 = {var_5: var_20, var_6: var_21}
    var_23 = {}
    var_24 = {}
    var_25 = {var_5: var_23, var_6: var_24}
    var_26 = {}
    var_27 = {}
    var_28 = {var_5: var_26, var_6: var_27}
    var_29 = {}
    var_30 = {}
    var_31 = {var_5: var_29, var_6: var_30}
    var_32 = {var_10: var_17, var_11: var_22, var_12: var_25, var_13: var_28, var_14: var_31}
    var_33 = 'above'
    var_34 = {}
    var_35 = {var_5: var_34}
    var_36 = {}
    var_37 = {var_33: var_35, var_5: var_36}
    var_38 = [var_10, var_11, var_12, var_13, var_14]
    var_39 = '\n'
    var_40 = module_0.ParsedContent()
    var_41 = module_1.Config()
    var_42 = module_2.sorted_imports(var_40, var_41)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = []
    var_2 = []
    var_3 = {}
    var_4 = {}
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = {}
    var_8 = {}
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'FUTURE'
    var_11 = 'STDLIB'
    var_12 = 'THIRDPARTY'
    var_13 = 'FIRSTPARTY'
    var_14 = 'LOCALFOLDER'
    var_15 = {}
    var_16 = {}
    var_17 = {var_5: var_15, var_6: var_16}
    var_18 = {}
    var_19 = 'os'
    var_20 = 'path'
    var_21 = [var_20]
    var_22 = {var_19: var_21}
    var_23 = {var_5: var_18, var_6: var_22}
    var_24 = {}
    var_25 = {}
    var_26 = {var_5: var_24, var_6: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = {var_5: var_27, var_6: var_28}
    var_30 = {}
    var_31 = {}
    var_32 = {var_5: var_30, var_6: var_31}
    var_33 = {var_10: var_17, var_11: var_23, var_12: var_26, var_13: var_29, var_14: var_32}
    var_34 = 'above'
    var_35 = {}
    var_36 = {var_5: var_35}
    var_37 = {}
    var_38 = {var_34: var_36, var_5: var_37}
    var_39 = [var_10, var_11, var_12, var_13, var_14]
    var_40 = '\n'
    var_41 = module_0.ParsedContent()
    var_42 = module_1.Config()
    var_43 = module_2.sorted_imports(var_41, var_42)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = []
    var_2 = []
    var_3 = {}
    var_4 = {}
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = {}
    var_8 = {}
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'FUTURE'
    var_11 = 'STDLIB'
    var_12 = 'THIRDPARTY'
    var_13 = 'FIRSTPARTY'
    var_14 = 'LOCALFOLDER'
    var_15 = {}
    var_16 = {}
    var_17 = {var_5: var_15, var_6: var_16}
    var_18 = 'os'
    var_19 = 'sys'
    var_20 = None
    var_21 = {var_18: var_20, var_19: var_20}
    var_22 = {}
    var_23 = {var_5: var_21, var_6: var_22}
    var_24 = {}
    var_25 = {}
    var_26 = {var_5: var_24, var_6: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = {var_5: var_27, var_6: var_28}
    var_30 = {}
    var_31 = {}
    var_32 = {var_5: var_30, var_6: var_31}
    var_33 = {var_10: var_17, var_11: var_23, var_12: var_26, var_13: var_29, var_14: var_32}
    var_34 = 'above'
    var_35 = {}
    var_36 = {var_5: var_35}
    var_37 = {}
    var_38 = {var_34: var_36, var_5: var_37}
    var_39 = [var_10, var_11, var_12, var_13, var_14]
    var_40 = '\n'
    var_41 = module_0.ParsedContent()
    var_42 = 'import os'
    var_43 = [var_42]
    var_44 = module_1.Config()
    var_45 = module_2.sorted_imports(var_41, var_44)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = []
    var_2 = []
    var_3 = {}
    var_4 = {}
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = {}
    var_8 = {}
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'FUTURE'
    var_11 = 'STDLIB'
    var_12 = 'THIRDPARTY'
    var_13 = 'FIRSTPARTY'
    var_14 = 'LOCALFOLDER'
    var_15 = {}
    var_16 = {}
    var_17 = {var_5: var_15, var_6: var_16}
    var_18 = 'os'
    var_19 = None
    var_20 = {var_18: var_19}
    var_21 = {}
    var_22 = {var_5: var_20, var_6: var_21}
    var_23 = {}
    var_24 = {}
    var_25 = {var_5: var_23, var_6: var_24}
    var_26 = {}
    var_27 = {}
    var_28 = {var_5: var_26, var_6: var_27}
    var_29 = {}
    var_30 = {}
    var_31 = {var_5: var_29, var_6: var_30}
    var_32 = {var_10: var_17, var_11: var_22, var_12: var_25, var_13: var_28, var_14: var_31}
    var_33 = 'above'
    var_34 = {}
    var_35 = {var_5: var_34}
    var_36 = {}
    var_37 = {var_33: var_35, var_5: var_36}
    var_38 = [var_10, var_11, var_12, var_13, var_14]
    var_39 = '\n'
    var_40 = module_0.ParsedContent()
    var_41 = 'stdlib'
    var_42 = 'Standard Library'
    var_43 = {var_41: var_42}
    var_44 = module_1.Config()
    var_45 = module_2.sorted_imports(var_40, var_44)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = []
    var_2 = []
    var_3 = {}
    var_4 = {}
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = {}
    var_8 = {}
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'FUTURE'
    var_11 = 'STDLIB'
    var_12 = 'THIRDPARTY'
    var_13 = 'FIRSTPARTY'
    var_14 = 'LOCALFOLDER'
    var_15 = {}
    var_16 = {}
    var_17 = {var_5: var_15, var_6: var_16}
    var_18 = 'os'
    var_19 = None
    var_20 = {var_18: var_19}
    var_21 = {}
    var_22 = {var_5: var_20, var_6: var_21}
    var_23 = 'requests'
    var_24 = {var_23: var_19}
    var_25 = {}
    var_26 = {var_5: var_24, var_6: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = {var_5: var_27, var_6: var_28}
    var_30 = {}
    var_31 = {}
    var_32 = {var_5: var_30, var_6: var_31}
    var_33 = {var_10: var_17, var_11: var_22, var_12: var_26, var_13: var_29, var_14: var_32}
    var_34 = 'above'
    var_35 = {}
    var_36 = {var_5: var_35}
    var_37 = {}
    var_38 = {var_34: var_36, var_5: var_37}
    var_39 = [var_10, var_11, var_12, var_13, var_14]
    var_40 = '\n'
    var_41 = module_0.ParsedContent()
    var_42 = 2
    var_43 = module_1.Config()
    var_44 = module_2.sorted_imports(var_41, var_43)

def test_case_0():
    pass



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_sorted_imports_with_no_sections. Retrieved 33/37 statements.
# Partially parsed test_sorted_imports_with_from_imports. Retrieved 29/33 statements.
# Partially parsed test_sorted_imports_normalize_empty_lines. Retrieved 20/24 statements.
# Partially parsed test_sorted_imports_with_remove_imports. Retrieved 32/36 statements.


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = -1
    var_1 = 'line1'
    var_2 = 'line2'
    var_3 = [var_1, var_2]
    var_4 = '\n'
    var_5 = 2
    var_6 = []
    var_7 = {}
    var_8 = {}
    var_9 = {}
    var_10 = 'straight'
    var_11 = 'from'
    var_12 = {}
    var_13 = {}
    var_14 = {var_10: var_12, var_11: var_13}
    var_15 = {}
    var_16 = module_0.ParsedContent()
    var_17 = module_1.Config()
    var_18 = module_2.sorted_imports(var_16, var_17)
    assert var_18 == 'line1\nline2\n'

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'rest of file'
    var_2 = [var_1]
    var_3 = '\n'
    var_4 = 1
    var_5 = 'FUTURE'
    var_6 = 'STDLIB'
    var_7 = [var_5, var_6]
    var_8 = {}
    var_9 = {}
    var_10 = 'straight'
    var_11 = 'from'
    var_12 = {}
    var_13 = {}
    var_14 = {var_10: var_12, var_11: var_13}
    var_15 = 'os'
    var_16 = {}
    var_17 = {var_15: var_16}
    var_18 = {}
    var_19 = {var_10: var_17, var_11: var_18}
    var_20 = {var_5: var_14, var_6: var_19}
    var_21 = {var_15}
    var_22 = {}
    var_23 = {var_10: var_21, var_11: var_22}
    var_24 = 'above'
    var_25 = {}
    var_26 = {var_10: var_25}
    var_27 = {}
    var_28 = {var_24: var_26, var_10: var_27}
    var_29 = module_0.ParsedContent()
    var_30 = True
    var_31 = module_1.Config()
    var_32 = module_2.sorted_imports(var_29, var_31)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'code'
    var_2 = [var_1]
    var_3 = '\n'
    var_4 = 1
    var_5 = 'STDLIB'
    var_6 = [var_5]
    var_7 = {}
    var_8 = {}
    var_9 = 'straight'
    var_10 = 'from'
    var_11 = 'os'
    var_12 = 'sys'
    var_13 = {}
    var_14 = {}
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = {}
    var_17 = {var_9: var_15, var_10: var_16}
    var_18 = {var_5: var_17}
    var_19 = {}
    var_20 = {}
    var_21 = {var_9: var_19, var_10: var_20}
    var_22 = 'above'
    var_23 = {}
    var_24 = {var_9: var_23}
    var_25 = {}
    var_26 = {var_22: var_24, var_9: var_25}
    var_27 = module_0.ParsedContent()
    var_28 = module_1.Config()
    var_29 = module_2.sorted_imports(var_27, var_28)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'code'
    var_2 = [var_1]
    var_3 = '\n'
    var_4 = 1
    var_5 = 'STDLIB'
    var_6 = [var_5]
    var_7 = {}
    var_8 = {}
    var_9 = 'straight'
    var_10 = 'from'
    var_11 = {}
    var_12 = 'os'
    var_13 = 'path'
    var_14 = {var_13}
    var_15 = {var_12: var_14}
    var_16 = {var_9: var_11, var_10: var_15}
    var_17 = {var_5: var_16}
    var_18 = {}
    var_19 = {}
    var_20 = {var_9: var_18, var_10: var_19}
    var_21 = 'above'
    var_22 = {}
    var_23 = {var_10: var_22}
    var_24 = {}
    var_25 = {var_21: var_23, var_10: var_24}
    var_26 = module_0.ParsedContent()
    var_27 = module_1.Config()
    var_28 = module_2.sorted_imports(var_26, var_27)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = -1
    var_1 = 'line1'
    var_2 = ''
    var_3 = 'line2'
    var_4 = [var_1, var_2, var_2, var_3]
    var_5 = '\n'
    var_6 = 4
    var_7 = []
    var_8 = {}
    var_9 = {}
    var_10 = {}
    var_11 = 'straight'
    var_12 = 'from'
    var_13 = {}
    var_14 = {}
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = {}
    var_17 = module_0.ParsedContent()
    var_18 = module_1.Config()
    var_19 = module_2.sorted_imports(var_17, var_18)

import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 0
    var_1 = 'code'
    var_2 = [var_1]
    var_3 = '\n'
    var_4 = 1
    var_5 = 'STDLIB'
    var_6 = [var_5]
    var_7 = {}
    var_8 = {}
    var_9 = 'straight'
    var_10 = 'from'
    var_11 = 'os'
    var_12 = 'sys'
    var_13 = {}
    var_14 = {}
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = {}
    var_17 = {var_9: var_15, var_10: var_16}
    var_18 = {var_5: var_17}
    var_19 = {}
    var_20 = {}
    var_21 = {var_9: var_19, var_10: var_20}
    var_22 = 'above'
    var_23 = {}
    var_24 = {var_9: var_23}
    var_25 = {}
    var_26 = {var_22: var_24, var_9: var_25}
    var_27 = module_0.ParsedContent()
    var_28 = 'import os'
    var_29 = [var_28]
    var_30 = module_1.Config()
    var_31 = module_2.sorted_imports(var_27, var_30)



