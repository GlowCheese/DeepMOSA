####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import isort.output as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._ensure_newline_before_comment(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import isort.output as module_0

def test_case_0():
    var_0 = '# comment'
    var_1 = [var_0]
    var_2 = module_0._ensure_newline_before_comment(var_1)
    var_3 = bool(var_2 == ['# comment'])
    assert var_3 is True

import isort.output as module_0

def test_case_0():
    var_0 = 'code'
    var_1 = [var_0]
    var_2 = module_0._ensure_newline_before_comment(var_1)
    var_3 = bool(var_2 == ['code'])
    assert var_3 is True

import isort.output as module_0

def test_case_0():
    var_0 = 'code'
    var_1 = '# comment'
    var_2 = [var_0, var_1]
    var_3 = module_0._ensure_newline_before_comment(var_2)
    var_4 = bool(var_3 == ['code', '', '# comment'])
    assert var_4 is True

import isort.output as module_0

def test_case_0():
    var_0 = ''
    var_1 = '# comment'
    var_2 = [var_0, var_1]
    var_3 = module_0._ensure_newline_before_comment(var_2)
    var_4 = bool(var_3 == ['', '# comment'])
    assert var_4 is True

import isort.output as module_0

def test_case_0():
    var_0 = '# comment1'
    var_1 = '# comment2'
    var_2 = [var_0, var_1]
    var_3 = module_0._ensure_newline_before_comment(var_2)
    var_4 = bool(var_3 == ['# comment1', '# comment2'])
    assert var_4 is True

import isort.output as module_0

def test_case_0():
    var_0 = 'code'
    var_1 = '# comment1'
    var_2 = '# comment2'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._ensure_newline_before_comment(var_3)
    var_5 = bool(var_4 == ['code', '', '# comment1', '# comment2'])
    assert var_5 is True

import isort.output as module_0

def test_case_0():
    var_0 = 'code1'
    var_1 = '# comment1'
    var_2 = 'code2'
    var_3 = '# comment2'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._ensure_newline_before_comment(var_4)
    var_6 = bool(var_5 == ['code1', '', '# comment1', 'code2', '', '# comment2'])
    assert var_6 is True

import isort.output as module_0

def test_case_0():
    var_0 = '# comment'
    var_1 = 'code'
    var_2 = [var_0, var_1]
    var_3 = module_0._ensure_newline_before_comment(var_2)
    var_4 = bool(var_3 == ['# comment', 'code'])
    assert var_4 is True

import isort.output as module_0

def test_case_0():
    var_0 = None
    var_1 = '# comment'
    var_2 = [var_0, var_1]
    var_3 = module_0._ensure_newline_before_comment(var_2)
    var_4 = bool(var_3 == [None, '', '# comment'])
    assert var_4 is True

import isort.output as module_0

def test_case_0():
    var_0 = ''
    var_1 = '# comment'
    var_2 = [var_0, var_0, var_1]
    var_3 = module_0._ensure_newline_before_comment(var_2)
    var_4 = bool(var_3 == ['', '', '# comment'])
    assert var_4 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_with_straight_imports_empty_straight_modules. Retrieved 15/17 statements.
# Partially parsed test_with_straight_imports_combine_straight_imports_no_as_imports. Retrieved 27/29 statements.
# Partially parsed test_with_straight_imports_combine_straight_imports_with_as_imports. Retrieved 29/31 statements.
# Partially parsed test_with_straight_imports_no_combine_straight_imports. Retrieved 27/29 statements.
# Partially parsed test_with_straight_imports_remove_imports. Retrieved 27/29 statements.
# Partially parsed test_with_straight_imports_ignore_comments. Retrieved 28/30 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'straight'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'above'
    var_4 = {}
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = {var_3: var_5, var_0: var_6}
    var_8 = {}
    var_9 = {var_0: var_8}
    var_10 = []
    var_11 = True
    var_12 = 'combine_straight_imports'
    var_13 = {var_12: var_11}
    var_14 = module_0.Config(**var_13)
    var_15 = []
    var_16 = []
    var_17 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'straight'
    var_1 = 'module1'
    var_2 = 'module2'
    var_3 = []
    var_4 = []
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'above'
    var_8 = '# comment1'
    var_9 = [var_8]
    var_10 = '# comment2'
    var_11 = [var_10]
    var_12 = {var_1: var_9, var_2: var_11}
    var_13 = {var_0: var_12}
    var_14 = '# inline1'
    var_15 = [var_14]
    var_16 = '# inline2'
    var_17 = [var_16]
    var_18 = {var_1: var_15, var_2: var_17}
    var_19 = {var_7: var_13, var_0: var_18}
    var_20 = {}
    var_21 = {var_0: var_20}
    var_22 = []
    var_23 = True
    var_24 = 'combine_straight_imports'
    var_25 = {var_24: var_23}
    var_26 = module_0.Config(**var_25)
    var_27 = [var_1, var_2]
    var_28 = []
    var_29 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'straight'
    var_1 = 'module1'
    var_2 = 'module2'
    var_3 = []
    var_4 = []
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'above'
    var_8 = '# comment1'
    var_9 = [var_8]
    var_10 = '# comment2'
    var_11 = [var_10]
    var_12 = {var_1: var_9, var_2: var_11}
    var_13 = {var_0: var_12}
    var_14 = '# inline1'
    var_15 = [var_14]
    var_16 = '# inline2'
    var_17 = [var_16]
    var_18 = {var_1: var_15, var_2: var_17}
    var_19 = {var_7: var_13, var_0: var_18}
    var_20 = 'alias1'
    var_21 = [var_20]
    var_22 = {var_1: var_21}
    var_23 = {var_0: var_22}
    var_24 = []
    var_25 = True
    var_26 = 'combine_straight_imports'
    var_27 = {var_26: var_25}
    var_28 = module_0.Config(**var_27)
    var_29 = [var_1, var_2]
    var_30 = []
    var_31 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'straight'
    var_1 = 'module1'
    var_2 = 'module2'
    var_3 = []
    var_4 = []
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'above'
    var_8 = '# comment1'
    var_9 = [var_8]
    var_10 = '# comment2'
    var_11 = [var_10]
    var_12 = {var_1: var_9, var_2: var_11}
    var_13 = {var_0: var_12}
    var_14 = '# inline1'
    var_15 = [var_14]
    var_16 = '# inline2'
    var_17 = [var_16]
    var_18 = {var_1: var_15, var_2: var_17}
    var_19 = {var_7: var_13, var_0: var_18}
    var_20 = {}
    var_21 = {var_0: var_20}
    var_22 = []
    var_23 = False
    var_24 = 'combine_straight_imports'
    var_25 = {var_24: var_23}
    var_26 = module_0.Config(**var_25)
    var_27 = [var_1, var_2]
    var_28 = []
    var_29 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'straight'
    var_1 = 'module1'
    var_2 = 'module2'
    var_3 = []
    var_4 = []
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'above'
    var_8 = '# comment1'
    var_9 = [var_8]
    var_10 = '# comment2'
    var_11 = [var_10]
    var_12 = {var_1: var_9, var_2: var_11}
    var_13 = {var_0: var_12}
    var_14 = '# inline1'
    var_15 = [var_14]
    var_16 = '# inline2'
    var_17 = [var_16]
    var_18 = {var_1: var_15, var_2: var_17}
    var_19 = {var_7: var_13, var_0: var_18}
    var_20 = {}
    var_21 = {var_0: var_20}
    var_22 = []
    var_23 = False
    var_24 = 'combine_straight_imports'
    var_25 = {var_24: var_23}
    var_26 = module_0.Config(**var_25)
    var_27 = [var_1, var_2]
    var_28 = [var_1]
    var_29 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'straight'
    var_1 = 'module1'
    var_2 = 'module2'
    var_3 = []
    var_4 = []
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'above'
    var_8 = '# comment1'
    var_9 = [var_8]
    var_10 = '# comment2'
    var_11 = [var_10]
    var_12 = {var_1: var_9, var_2: var_11}
    var_13 = {var_0: var_12}
    var_14 = '# inline1'
    var_15 = [var_14]
    var_16 = '# inline2'
    var_17 = [var_16]
    var_18 = {var_1: var_15, var_2: var_17}
    var_19 = {var_7: var_13, var_0: var_18}
    var_20 = {}
    var_21 = {var_0: var_20}
    var_22 = []
    var_23 = False
    var_24 = True
    var_25 = 'combine_straight_imports'
    var_26 = 'ignore_comments'
    var_27 = {var_25: var_23, var_26: var_24}
    var_28 = module_0.Config(**var_27)
    var_29 = [var_1, var_2]
    var_30 = []
    var_31 = 'import'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_with_from_imports_basic. Retrieved 22/24 statements.
# Partially parsed test_with_from_imports_with_comments. Retrieved 29/31 statements.
# Partially parsed test_with_from_imports_remove_imports. Retrieved 23/25 statements.
# Partially parsed test_with_from_imports_with_as_imports. Retrieved 28/30 statements.
# Partially parsed test_with_from_imports_with_star. Retrieved 27/29 statements.
# Partially parsed test_with_from_imports_force_single_line. Retrieved 23/25 statements.
# Partially parsed test_with_from_imports_combine_as_imports. Retrieved 29/31 statements.
# Partially parsed test_with_from_imports_ignore_comments. Retrieved 30/32 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = {}
    var_11 = {}
    var_12 = {var_1: var_11}
    var_13 = {var_1: var_10, var_9: var_12}
    var_14 = {}
    var_15 = {var_1: var_14}
    var_16 = '\n'
    var_17 = set()
    var_18 = []
    var_19 = {}
    var_20 = module_0.Config(**var_19)
    var_21 = [var_2]
    var_22 = []
    var_23 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = 'straight'
    var_12 = 'comment1'
    var_13 = 'comment2'
    var_14 = [var_12, var_13]
    var_15 = {var_2: var_14}
    var_16 = {}
    var_17 = {var_1: var_16}
    var_18 = {}
    var_19 = {}
    var_20 = {var_1: var_15, var_9: var_17, var_10: var_18, var_11: var_19}
    var_21 = {}
    var_22 = {var_1: var_21}
    var_23 = '\n'
    var_24 = set()
    var_25 = []
    var_26 = {}
    var_27 = module_0.Config(**var_26)
    var_28 = [var_2]
    var_29 = []
    var_30 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = {}
    var_11 = {}
    var_12 = {var_1: var_11}
    var_13 = {var_1: var_10, var_9: var_12}
    var_14 = {}
    var_15 = {var_1: var_14}
    var_16 = '\n'
    var_17 = set()
    var_18 = []
    var_19 = {}
    var_20 = module_0.Config(**var_19)
    var_21 = [var_2]
    var_22 = 'module.import1'
    var_23 = [var_22]
    var_24 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = {}
    var_11 = {}
    var_12 = {var_1: var_11}
    var_13 = {var_1: var_10, var_9: var_12}
    var_14 = 'module.import1'
    var_15 = 'module.import2'
    var_16 = 'alias1'
    var_17 = [var_16]
    var_18 = 'alias2'
    var_19 = [var_18]
    var_20 = {var_14: var_17, var_15: var_19}
    var_21 = {var_1: var_20}
    var_22 = '\n'
    var_23 = set()
    var_24 = []
    var_25 = {}
    var_26 = module_0.Config(**var_25)
    var_27 = [var_2]
    var_28 = []
    var_29 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = '*'
    var_4 = 'import1'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = {}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = 'star comment'
    var_15 = [var_14]
    var_16 = {var_3: var_15}
    var_17 = {var_2: var_16}
    var_18 = {var_1: var_11, var_9: var_13, var_10: var_17}
    var_19 = {}
    var_20 = {var_1: var_19}
    var_21 = '\n'
    var_22 = set()
    var_23 = []
    var_24 = {}
    var_25 = module_0.Config(**var_24)
    var_26 = [var_2]
    var_27 = []
    var_28 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = {}
    var_11 = {}
    var_12 = {var_1: var_11}
    var_13 = {var_1: var_10, var_9: var_12}
    var_14 = {}
    var_15 = {var_1: var_14}
    var_16 = '\n'
    var_17 = set()
    var_18 = []
    var_19 = True
    var_20 = 'force_single_line'
    var_21 = {var_20: var_19}
    var_22 = module_0.Config(**var_21)
    var_23 = [var_2]
    var_24 = []
    var_25 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = {}
    var_11 = {}
    var_12 = {var_1: var_11}
    var_13 = {var_1: var_10, var_9: var_12}
    var_14 = 'module.import1'
    var_15 = 'module.import2'
    var_16 = 'alias1'
    var_17 = [var_16]
    var_18 = 'alias2'
    var_19 = [var_18]
    var_20 = {var_14: var_17, var_15: var_19}
    var_21 = {var_1: var_20}
    var_22 = '\n'
    var_23 = set()
    var_24 = []
    var_25 = True
    var_26 = 'combine_as_imports'
    var_27 = {var_26: var_25}
    var_28 = module_0.Config(**var_27)
    var_29 = [var_2]
    var_30 = []
    var_31 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = 'straight'
    var_12 = 'comment1'
    var_13 = 'comment2'
    var_14 = [var_12, var_13]
    var_15 = {var_2: var_14}
    var_16 = {}
    var_17 = {var_1: var_16}
    var_18 = {}
    var_19 = {}
    var_20 = {var_1: var_15, var_9: var_17, var_10: var_18, var_11: var_19}
    var_21 = {}
    var_22 = {var_1: var_21}
    var_23 = '\n'
    var_24 = set()
    var_25 = []
    var_26 = True
    var_27 = 'ignore_comments'
    var_28 = {var_27: var_26}
    var_29 = module_0.Config(**var_28)
    var_30 = [var_2]
    var_31 = []
    var_32 = 'import'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_sorted_imports_empty_parsed_content. Retrieved 11/13 statements.
# Partially parsed test_sorted_imports_no_imports. Retrieved 11/13 statements.
# Partially parsed test_sorted_imports_single_straight_import. Retrieved 24/26 statements.
# Partially parsed test_sorted_imports_single_from_import. Retrieved 26/28 statements.
# Partially parsed test_sorted_imports_multiple_straight_imports. Retrieved 26/28 statements.
# Partially parsed test_sorted_imports_multiple_from_imports. Retrieved 30/32 statements.
# Partially parsed test_sorted_imports_with_comments. Retrieved 28/30 statements.
# Partially parsed test_sorted_imports_with_as_imports. Retrieved 26/28 statements.
# Partially parsed test_sorted_imports_with_force_sort_within_sections. Retrieved 27/29 statements.
# Partially parsed test_sorted_imports_with_no_sections. Retrieved 31/33 statements.
# Partially parsed test_sorted_imports_with_remove_imports. Retrieved 28/30 statements.
# Partially parsed test_sorted_imports_with_import_headings. Retrieved 28/30 statements.


def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = -1
    var_3 = 0
    var_4 = '\n'
    var_5 = {}
    var_6 = {}
    var_7 = {}
    var_8 = {}
    var_9 = {}
    var_10 = []
    var_11 = []

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = -1
    var_3 = 1
    var_4 = '\n'
    var_5 = {}
    var_6 = {}
    var_7 = {}
    var_8 = {}
    var_9 = {}
    var_10 = []
    var_11 = []

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = 1
    var_4 = '\n'
    var_5 = 'STDLIB'
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = 'os'
    var_9 = set()
    var_10 = {var_8: var_9}
    var_11 = {}
    var_12 = {var_6: var_10, var_7: var_11}
    var_13 = {var_5: var_12}
    var_14 = 'above'
    var_15 = {}
    var_16 = {var_6: var_15}
    var_17 = {}
    var_18 = {var_14: var_16, var_6: var_17}
    var_19 = {}
    var_20 = {var_6: var_19}
    var_21 = {}
    var_22 = {}
    var_23 = [var_5]
    var_24 = []

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = 1
    var_4 = '\n'
    var_5 = 'STDLIB'
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = {}
    var_9 = 'os'
    var_10 = 'path'
    var_11 = set()
    var_12 = {var_10: var_11}
    var_13 = {var_9: var_12}
    var_14 = {var_6: var_8, var_7: var_13}
    var_15 = {var_5: var_14}
    var_16 = 'above'
    var_17 = {}
    var_18 = {var_6: var_17}
    var_19 = {}
    var_20 = {var_16: var_18, var_6: var_19}
    var_21 = {}
    var_22 = {var_6: var_21}
    var_23 = {}
    var_24 = {}
    var_25 = [var_5]
    var_26 = []

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = 1
    var_4 = '\n'
    var_5 = 'STDLIB'
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = 'os'
    var_9 = 'sys'
    var_10 = set()
    var_11 = set()
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = {}
    var_14 = {var_6: var_12, var_7: var_13}
    var_15 = {var_5: var_14}
    var_16 = 'above'
    var_17 = {}
    var_18 = {var_6: var_17}
    var_19 = {}
    var_20 = {var_16: var_18, var_6: var_19}
    var_21 = {}
    var_22 = {var_6: var_21}
    var_23 = {}
    var_24 = {}
    var_25 = [var_5]
    var_26 = []

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = 1
    var_4 = '\n'
    var_5 = 'STDLIB'
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = {}
    var_9 = 'os'
    var_10 = 'sys'
    var_11 = 'path'
    var_12 = set()
    var_13 = {var_11: var_12}
    var_14 = 'argv'
    var_15 = set()
    var_16 = {var_14: var_15}
    var_17 = {var_9: var_13, var_10: var_16}
    var_18 = {var_6: var_8, var_7: var_17}
    var_19 = {var_5: var_18}
    var_20 = 'above'
    var_21 = {}
    var_22 = {var_6: var_21}
    var_23 = {}
    var_24 = {var_20: var_22, var_6: var_23}
    var_25 = {}
    var_26 = {var_6: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = [var_5]
    var_30 = []

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = 1
    var_4 = '\n'
    var_5 = 'STDLIB'
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = 'os'
    var_9 = set()
    var_10 = {var_8: var_9}
    var_11 = {}
    var_12 = {var_6: var_10, var_7: var_11}
    var_13 = {var_5: var_12}
    var_14 = 'above'
    var_15 = '# comment above'
    var_16 = [var_15]
    var_17 = {var_8: var_16}
    var_18 = {var_6: var_17}
    var_19 = '# inline comment'
    var_20 = [var_19]
    var_21 = {var_8: var_20}
    var_22 = {var_14: var_18, var_6: var_21}
    var_23 = {}
    var_24 = {var_6: var_23}
    var_25 = {}
    var_26 = {}
    var_27 = [var_5]
    var_28 = []

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = 1
    var_4 = '\n'
    var_5 = 'STDLIB'
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = 'os'
    var_9 = set()
    var_10 = {var_8: var_9}
    var_11 = {}
    var_12 = {var_6: var_10, var_7: var_11}
    var_13 = {var_5: var_12}
    var_14 = 'above'
    var_15 = {}
    var_16 = {var_6: var_15}
    var_17 = {}
    var_18 = {var_14: var_16, var_6: var_17}
    var_19 = 'ospath'
    var_20 = [var_19]
    var_21 = {var_8: var_20}
    var_22 = {var_6: var_21}
    var_23 = {}
    var_24 = {}
    var_25 = [var_5]
    var_26 = []

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'force_sort_within_sections'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = ''
    var_5 = [var_4]
    var_6 = 0
    var_7 = '\n'
    var_8 = 'STDLIB'
    var_9 = 'straight'
    var_10 = 'from'
    var_11 = 'sys'
    var_12 = 'os'
    var_13 = set()
    var_14 = set()
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = {}
    var_17 = {var_9: var_15, var_10: var_16}
    var_18 = {var_8: var_17}
    var_19 = 'above'
    var_20 = {}
    var_21 = {var_9: var_20}
    var_22 = {}
    var_23 = {var_19: var_21, var_9: var_22}
    var_24 = {}
    var_25 = {var_9: var_24}
    var_26 = {}
    var_27 = {}
    var_28 = [var_8]
    var_29 = []

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'no_sections'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = ''
    var_5 = [var_4]
    var_6 = 0
    var_7 = '\n'
    var_8 = 'STDLIB'
    var_9 = 'THIRDPARTY'
    var_10 = 'straight'
    var_11 = 'from'
    var_12 = 'os'
    var_13 = set()
    var_14 = {var_12: var_13}
    var_15 = {}
    var_16 = {var_10: var_14, var_11: var_15}
    var_17 = 'django'
    var_18 = set()
    var_19 = {var_17: var_18}
    var_20 = {}
    var_21 = {var_10: var_19, var_11: var_20}
    var_22 = {var_8: var_16, var_9: var_21}
    var_23 = 'above'
    var_24 = {}
    var_25 = {var_10: var_24}
    var_26 = {}
    var_27 = {var_23: var_25, var_10: var_26}
    var_28 = {}
    var_29 = {var_10: var_28}
    var_30 = {}
    var_31 = {}
    var_32 = [var_8, var_9]
    var_33 = []

import isort.settings as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'remove_imports'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = ''
    var_6 = [var_5]
    var_7 = 0
    var_8 = 1
    var_9 = '\n'
    var_10 = 'STDLIB'
    var_11 = 'straight'
    var_12 = 'from'
    var_13 = 'sys'
    var_14 = set()
    var_15 = set()
    var_16 = {var_0: var_14, var_13: var_15}
    var_17 = {}
    var_18 = {var_11: var_16, var_12: var_17}
    var_19 = {var_10: var_18}
    var_20 = 'above'
    var_21 = {}
    var_22 = {var_11: var_21}
    var_23 = {}
    var_24 = {var_20: var_22, var_11: var_23}
    var_25 = {}
    var_26 = {var_11: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = [var_10]
    var_30 = []

import isort.settings as module_0

def test_case_0():
    var_0 = 'stdlib'
    var_1 = 'Standard Library'
    var_2 = {var_0: var_1}
    var_3 = 'import_headings'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = ''
    var_7 = [var_6]
    var_8 = 0
    var_9 = 1
    var_10 = '\n'
    var_11 = 'STDLIB'
    var_12 = 'straight'
    var_13 = 'from'
    var_14 = 'os'
    var_15 = set()
    var_16 = {var_14: var_15}
    var_17 = {}
    var_18 = {var_12: var_16, var_13: var_17}
    var_19 = {var_11: var_18}
    var_20 = 'above'
    var_21 = {}
    var_22 = {var_12: var_21}
    var_23 = {}
    var_24 = {var_20: var_22, var_12: var_23}
    var_25 = {}
    var_26 = {var_12: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = [var_11]
    var_30 = []



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_sorted_imports_no_imports. Retrieved 12/14 statements.
# Partially parsed test_sorted_imports_single_import. Retrieved 25/27 statements.
# Partially parsed test_sorted_imports_multiple_imports. Retrieved 27/29 statements.
# Partially parsed test_sorted_imports_with_comments. Retrieved 27/29 statements.
# Partially parsed test_sorted_imports_with_as_imports. Retrieved 27/29 statements.
# Partially parsed test_sorted_imports_with_remove_imports. Retrieved 28/30 statements.
# Partially parsed test_sorted_imports_with_combine_straight_imports. Retrieved 28/30 statements.
# Partially parsed test_sorted_imports_with_from_imports. Retrieved 27/29 statements.
# Partially parsed test_sorted_imports_with_force_sort_within_sections. Retrieved 28/30 statements.
# Partially parsed test_sorted_imports_with_import_headings. Retrieved 28/30 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = -1
    var_3 = '\n'
    var_4 = {}
    var_5 = {}
    var_6 = {}
    var_7 = []
    var_8 = {}
    var_9 = {}
    var_10 = 1
    var_11 = []
    var_12 = {}
    var_13 = module_0.Config(**var_12)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = {}
    var_19 = {var_5: var_18}
    var_20 = [var_4]
    var_21 = {}
    var_22 = {}
    var_23 = 1
    var_24 = []
    var_25 = {}
    var_26 = module_0.Config(**var_25)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = 'sys'
    var_9 = set()
    var_10 = set()
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = {var_5: var_11, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = 'above'
    var_16 = {}
    var_17 = {var_5: var_16}
    var_18 = {}
    var_19 = {var_15: var_17, var_5: var_18}
    var_20 = {}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []
    var_27 = {}
    var_28 = module_0.Config(**var_27)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = 'comment'
    var_17 = [var_16]
    var_18 = {var_7: var_17}
    var_19 = {var_13: var_15, var_5: var_18}
    var_20 = {}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []
    var_27 = {}
    var_28 = module_0.Config(**var_27)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = 'ospath'
    var_19 = {var_18}
    var_20 = {var_7: var_19}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []
    var_27 = {}
    var_28 = module_0.Config(**var_27)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = 'sys'
    var_9 = set()
    var_10 = set()
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = {var_5: var_11, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = 'above'
    var_16 = {}
    var_17 = {var_5: var_16}
    var_18 = {}
    var_19 = {var_15: var_17, var_5: var_18}
    var_20 = {}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []
    var_27 = [var_7]
    var_28 = 'remove_imports'
    var_29 = {var_28: var_27}
    var_30 = module_0.Config(**var_29)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = 'sys'
    var_9 = set()
    var_10 = set()
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = {var_5: var_11, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = 'above'
    var_16 = {}
    var_17 = {var_5: var_16}
    var_18 = {}
    var_19 = {var_15: var_17, var_5: var_18}
    var_20 = {}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []
    var_27 = True
    var_28 = 'combine_straight_imports'
    var_29 = {var_28: var_27}
    var_30 = module_0.Config(**var_29)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = {}
    var_8 = 'os'
    var_9 = 'path'
    var_10 = set()
    var_11 = {var_9: var_10}
    var_12 = {var_8: var_11}
    var_13 = {var_5: var_7, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = 'above'
    var_16 = {}
    var_17 = {var_5: var_16}
    var_18 = {}
    var_19 = {var_15: var_17, var_5: var_18}
    var_20 = {}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []
    var_27 = {}
    var_28 = module_0.Config(**var_27)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'sys'
    var_8 = 'os'
    var_9 = set()
    var_10 = set()
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = {var_5: var_11, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = 'above'
    var_16 = {}
    var_17 = {var_5: var_16}
    var_18 = {}
    var_19 = {var_15: var_17, var_5: var_18}
    var_20 = {}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []
    var_27 = True
    var_28 = 'force_sort_within_sections'
    var_29 = {var_28: var_27}
    var_30 = module_0.Config(**var_29)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = {}
    var_19 = {var_5: var_18}
    var_20 = [var_4]
    var_21 = {}
    var_22 = {}
    var_23 = 1
    var_24 = []
    var_25 = 'stdlib'
    var_26 = 'Standard Library'
    var_27 = {var_25: var_26}
    var_28 = 'import_headings'
    var_29 = {var_28: var_27}
    var_30 = module_0.Config(**var_29)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_with_star_comments_when_star_comment_exists. Retrieved 10/13 statements.
# Partially parsed test_with_star_comments_when_star_comment_does_not_exist. Retrieved 8/11 statements.
# Partially parsed test_with_star_comments_when_module_does_not_exist. Retrieved 6/9 statements.


def test_case_0():
    var_0 = []
    var_1 = 'nested'
    var_2 = 'test_module'
    var_3 = '*'
    var_4 = 'star comment'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_module'
    var_8 = 'comment1'
    var_9 = 'comment2'
    var_10 = [var_8, var_9]

def test_case_0():
    var_0 = []
    var_1 = 'nested'
    var_2 = 'test_module'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'test_module'
    var_6 = 'comment1'
    var_7 = 'comment2'
    var_8 = [var_6, var_7]

def test_case_0():
    var_0 = []
    var_1 = 'nested'
    var_2 = {}
    var_3 = 'non_existent_module'
    var_4 = 'comment1'
    var_5 = 'comment2'
    var_6 = [var_4, var_5]



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 1/5 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = bool(not (var_1.no_inline_sort or (var_1.force_single_line and 'module' not in var_1.single_line_exclusions)) and (not var_1.only_sections))
    assert var_2 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_sorted_imports_no_imports. Retrieved 12/14 statements.
# Partially parsed test_sorted_imports_with_imports. Retrieved 26/28 statements.
# Partially parsed test_sorted_imports_with_combine_straight_imports. Retrieved 23/25 statements.
# Partially parsed test_sorted_imports_with_force_sort_within_sections. Retrieved 27/29 statements.
# Partially parsed test_sorted_imports_with_import_headings. Retrieved 25/27 statements.
# Partially parsed test_sorted_imports_with_remove_imports. Retrieved 23/25 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = -1
    var_3 = '\n'
    var_4 = {}
    var_5 = {}
    var_6 = {}
    var_7 = []
    var_8 = {}
    var_9 = {}
    var_10 = 1
    var_11 = []
    var_12 = {}
    var_13 = module_0.Config(**var_12)

import isort.settings as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = 'sys'
    var_9 = set()
    var_10 = set()
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = 'collections'
    var_13 = 'defaultdict'
    var_14 = set()
    var_15 = {var_13: var_14}
    var_16 = {var_12: var_15}
    var_17 = {var_5: var_11, var_6: var_16}
    var_18 = {var_4: var_17}
    var_19 = {}
    var_20 = {}
    var_21 = [var_4]
    var_22 = {}
    var_23 = {}
    var_24 = 2
    var_25 = []
    var_26 = {}
    var_27 = module_0.Config(**var_26)

import isort.settings as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = 'sys'
    var_9 = set()
    var_10 = set()
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = {var_5: var_11, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = {}
    var_16 = {}
    var_17 = [var_4]
    var_18 = {}
    var_19 = {}
    var_20 = 2
    var_21 = []
    var_22 = True
    var_23 = 'combine_straight_imports'
    var_24 = {var_23: var_22}
    var_25 = module_0.Config(**var_24)

import isort.settings as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = 'sys'
    var_9 = set()
    var_10 = set()
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = 'collections'
    var_13 = 'defaultdict'
    var_14 = set()
    var_15 = {var_13: var_14}
    var_16 = {var_12: var_15}
    var_17 = {var_5: var_11, var_6: var_16}
    var_18 = {var_4: var_17}
    var_19 = {}
    var_20 = {}
    var_21 = [var_4]
    var_22 = {}
    var_23 = {}
    var_24 = 2
    var_25 = []
    var_26 = True
    var_27 = 'force_sort_within_sections'
    var_28 = {var_27: var_26}
    var_29 = module_0.Config(**var_28)

import isort.settings as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = 'sys'
    var_9 = set()
    var_10 = set()
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = {var_5: var_11, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = {}
    var_16 = {}
    var_17 = [var_4]
    var_18 = {}
    var_19 = {}
    var_20 = 2
    var_21 = []
    var_22 = 'thirdparty'
    var_23 = 'Third Party Imports'
    var_24 = {var_22: var_23}
    var_25 = 'import_headings'
    var_26 = {var_25: var_24}
    var_27 = module_0.Config(**var_26)

import isort.settings as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = 'sys'
    var_9 = set()
    var_10 = set()
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = {var_5: var_11, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = {}
    var_16 = {}
    var_17 = [var_4]
    var_18 = {}
    var_19 = {}
    var_20 = 2
    var_21 = []
    var_22 = [var_7]
    var_23 = 'remove_imports'
    var_24 = {var_23: var_22}
    var_25 = module_0.Config(**var_24)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_sorted_imports_with_no_imports. Retrieved 8/10 statements.


def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = 1
    var_5 = '\n'
    var_6 = {}
    var_7 = {}
    var_8 = []



# Parsed testcases at query #10
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = []
    var_3 = 'no_inline_sort'
    var_4 = 'force_single_line'
    var_5 = 'single_line_exclusions'
    var_6 = 'only_sections'
    var_7 = {var_3: var_0, var_4: var_1, var_5: var_2, var_6: var_1}
    var_8 = module_0.Config(**var_7)
    var_9 = bool(not (not var_8.no_inline_sort or (var_8.force_single_line and 'module' not in var_8.single_line_exclusions)))
    assert var_9 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_sorted_imports_no_imports. Retrieved 11/14 statements.
# Partially parsed test_sorted_imports_single_import. Retrieved 24/27 statements.
# Partially parsed test_sorted_imports_multiple_imports. Retrieved 26/29 statements.
# Partially parsed test_sorted_imports_with_comments. Retrieved 28/31 statements.
# Partially parsed test_sorted_imports_combine_straight_imports. Retrieved 28/30 statements.
# Partially parsed test_sorted_imports_with_as_imports. Retrieved 26/29 statements.
# Partially parsed test_sorted_imports_remove_imports. Retrieved 28/30 statements.
# Partially parsed test_sorted_imports_no_sections. Retrieved 32/34 statements.
# Partially parsed test_sorted_imports_force_sort_within_sections. Retrieved 28/30 statements.
# Partially parsed test_sorted_imports_with_import_headings. Retrieved 28/30 statements.
# Partially parsed test_sorted_imports_with_import_footers. Retrieved 28/30 statements.
# Partially parsed test_sorted_imports_ensure_newline_before_comments. Retrieved 26/28 statements.
# Partially parsed test_sorted_imports_with_formatting_function. Retrieved 26/28 statements.


def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = -1
    var_3 = '\n'
    var_4 = {}
    var_5 = {}
    var_6 = {}
    var_7 = []
    var_8 = {}
    var_9 = {}
    var_10 = 1
    var_11 = []

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = {}
    var_19 = {var_5: var_18}
    var_20 = [var_4]
    var_21 = {}
    var_22 = {}
    var_23 = 1
    var_24 = []

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = 'sys'
    var_9 = set()
    var_10 = set()
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = {var_5: var_11, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = 'above'
    var_16 = {}
    var_17 = {var_5: var_16}
    var_18 = {}
    var_19 = {var_15: var_17, var_5: var_18}
    var_20 = {}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = '# OS module'
    var_15 = [var_14]
    var_16 = {var_7: var_15}
    var_17 = {var_5: var_16}
    var_18 = '# For path operations'
    var_19 = [var_18]
    var_20 = {var_7: var_19}
    var_21 = {var_13: var_17, var_5: var_20}
    var_22 = {}
    var_23 = {var_5: var_22}
    var_24 = [var_4]
    var_25 = {}
    var_26 = {}
    var_27 = 1
    var_28 = []

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = 'sys'
    var_9 = set()
    var_10 = set()
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = {var_5: var_11, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = 'above'
    var_16 = {}
    var_17 = {var_5: var_16}
    var_18 = {}
    var_19 = {var_15: var_17, var_5: var_18}
    var_20 = {}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []
    var_27 = True
    var_28 = 'combine_straight_imports'
    var_29 = {var_28: var_27}
    var_30 = module_0.Config(**var_29)

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = 'path'
    var_19 = [var_18]
    var_20 = {var_7: var_19}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = 'sys'
    var_9 = set()
    var_10 = set()
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = {var_5: var_11, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = 'above'
    var_16 = {}
    var_17 = {var_5: var_16}
    var_18 = {}
    var_19 = {var_15: var_17, var_5: var_18}
    var_20 = {}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []
    var_27 = [var_7]
    var_28 = 'remove_imports'
    var_29 = {var_28: var_27}
    var_30 = module_0.Config(**var_29)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'THIRDPARTY'
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = 'os'
    var_9 = set()
    var_10 = {var_8: var_9}
    var_11 = {}
    var_12 = {var_6: var_10, var_7: var_11}
    var_13 = 'django'
    var_14 = set()
    var_15 = {var_13: var_14}
    var_16 = {}
    var_17 = {var_6: var_15, var_7: var_16}
    var_18 = {var_4: var_12, var_5: var_17}
    var_19 = 'above'
    var_20 = {}
    var_21 = {var_6: var_20}
    var_22 = {}
    var_23 = {var_19: var_21, var_6: var_22}
    var_24 = {}
    var_25 = {var_6: var_24}
    var_26 = [var_4, var_5]
    var_27 = {}
    var_28 = {}
    var_29 = 1
    var_30 = []
    var_31 = True
    var_32 = 'no_sections'
    var_33 = {var_32: var_31}
    var_34 = module_0.Config(**var_33)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'sys'
    var_8 = 'os'
    var_9 = set()
    var_10 = set()
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = {var_5: var_11, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = 'above'
    var_16 = {}
    var_17 = {var_5: var_16}
    var_18 = {}
    var_19 = {var_15: var_17, var_5: var_18}
    var_20 = {}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []
    var_27 = True
    var_28 = 'force_sort_within_sections'
    var_29 = {var_28: var_27}
    var_30 = module_0.Config(**var_29)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = {}
    var_19 = {var_5: var_18}
    var_20 = [var_4]
    var_21 = {}
    var_22 = {}
    var_23 = 1
    var_24 = []
    var_25 = 'stdlib'
    var_26 = 'Standard Library'
    var_27 = {var_25: var_26}
    var_28 = 'import_headings'
    var_29 = {var_28: var_27}
    var_30 = module_0.Config(**var_29)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = {}
    var_19 = {var_5: var_18}
    var_20 = [var_4]
    var_21 = {}
    var_22 = {}
    var_23 = 1
    var_24 = []
    var_25 = 'stdlib'
    var_26 = 'End of Standard Library'
    var_27 = {var_25: var_26}
    var_28 = 'import_footers'
    var_29 = {var_28: var_27}
    var_30 = module_0.Config(**var_29)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = {}
    var_19 = {var_5: var_18}
    var_20 = [var_4]
    var_21 = {}
    var_22 = {}
    var_23 = 1
    var_24 = []
    var_25 = True
    var_26 = 'ensure_newline_before_comments'
    var_27 = {var_26: var_25}
    var_28 = module_0.Config(**var_27)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = {}
    var_19 = {var_5: var_18}
    var_20 = [var_4]
    var_21 = {}
    var_22 = {}
    var_23 = 1
    var_24 = []
    var_25 = lambda x, y, z: x.upper()
    var_26 = 'formatting_function'
    var_27 = {var_26: var_25}
    var_28 = module_0.Config(**var_27)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_sorted_imports_with_no_imports. Retrieved 8/11 statements.


def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = 1
    var_5 = '\n'
    var_6 = {}
    var_7 = {}
    var_8 = []



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_sorted_imports_no_imports. Retrieved 12/14 statements.
# Partially parsed test_sorted_imports_single_straight_import. Retrieved 25/27 statements.
# Partially parsed test_sorted_imports_multiple_straight_imports. Retrieved 27/29 statements.
# Partially parsed test_sorted_imports_with_from_imports. Retrieved 26/28 statements.
# Partially parsed test_sorted_imports_with_as_imports. Retrieved 27/29 statements.
# Partially parsed test_sorted_imports_combine_straight_imports. Retrieved 28/30 statements.
# Partially parsed test_sorted_imports_with_comments. Retrieved 29/31 statements.
# Partially parsed test_sorted_imports_remove_imports. Retrieved 28/30 statements.
# Partially parsed test_sorted_imports_no_sections. Retrieved 32/34 statements.
# Partially parsed test_sorted_imports_with_import_headings. Retrieved 28/30 statements.
# Partially parsed test_sorted_imports_with_import_footers. Retrieved 28/30 statements.
# Partially parsed test_sorted_imports_ensure_newline_before_comments. Retrieved 26/28 statements.
# Partially parsed test_sorted_imports_with_formatting_function. Retrieved 26/28 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = -1
    var_3 = '\n'
    var_4 = {}
    var_5 = {}
    var_6 = {}
    var_7 = []
    var_8 = {}
    var_9 = {}
    var_10 = 1
    var_11 = []
    var_12 = {}
    var_13 = module_0.Config(**var_12)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = [var_7]
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = {}
    var_19 = {var_5: var_18}
    var_20 = [var_4]
    var_21 = {}
    var_22 = {}
    var_23 = 1
    var_24 = []
    var_25 = {}
    var_26 = module_0.Config(**var_25)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = 'sys'
    var_9 = [var_7]
    var_10 = [var_8]
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = {var_5: var_11, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = 'above'
    var_16 = {}
    var_17 = {var_5: var_16}
    var_18 = {}
    var_19 = {var_15: var_17, var_5: var_18}
    var_20 = {}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []
    var_27 = {}
    var_28 = module_0.Config(**var_27)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = {}
    var_8 = 'os'
    var_9 = 'path'
    var_10 = [var_9]
    var_11 = {var_8: var_10}
    var_12 = {var_5: var_7, var_6: var_11}
    var_13 = {var_4: var_12}
    var_14 = 'above'
    var_15 = {}
    var_16 = {var_5: var_15}
    var_17 = {}
    var_18 = {var_14: var_16, var_5: var_17}
    var_19 = {}
    var_20 = {var_5: var_19}
    var_21 = [var_4]
    var_22 = {}
    var_23 = {}
    var_24 = 1
    var_25 = []
    var_26 = {}
    var_27 = module_0.Config(**var_26)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = [var_7]
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = 'operating_system'
    var_19 = [var_18]
    var_20 = {var_7: var_19}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []
    var_27 = {}
    var_28 = module_0.Config(**var_27)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = 'sys'
    var_9 = [var_7]
    var_10 = [var_8]
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = {var_5: var_11, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = 'above'
    var_16 = {}
    var_17 = {var_5: var_16}
    var_18 = {}
    var_19 = {var_15: var_17, var_5: var_18}
    var_20 = {}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []
    var_27 = True
    var_28 = 'combine_straight_imports'
    var_29 = {var_28: var_27}
    var_30 = module_0.Config(**var_29)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = [var_7]
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = '# Comment above'
    var_15 = [var_14]
    var_16 = {var_7: var_15}
    var_17 = {var_5: var_16}
    var_18 = '# Inline comment'
    var_19 = [var_18]
    var_20 = {var_7: var_19}
    var_21 = {var_13: var_17, var_5: var_20}
    var_22 = {}
    var_23 = {var_5: var_22}
    var_24 = [var_4]
    var_25 = {}
    var_26 = {}
    var_27 = 1
    var_28 = []
    var_29 = {}
    var_30 = module_0.Config(**var_29)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = 'sys'
    var_9 = [var_7]
    var_10 = [var_8]
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = {var_5: var_11, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = 'above'
    var_16 = {}
    var_17 = {var_5: var_16}
    var_18 = {}
    var_19 = {var_15: var_17, var_5: var_18}
    var_20 = {}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []
    var_27 = [var_7]
    var_28 = 'remove_imports'
    var_29 = {var_28: var_27}
    var_30 = module_0.Config(**var_29)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'THIRDPARTY'
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = 'os'
    var_9 = [var_8]
    var_10 = {var_8: var_9}
    var_11 = {}
    var_12 = {var_6: var_10, var_7: var_11}
    var_13 = 'django'
    var_14 = [var_13]
    var_15 = {var_13: var_14}
    var_16 = {}
    var_17 = {var_6: var_15, var_7: var_16}
    var_18 = {var_4: var_12, var_5: var_17}
    var_19 = 'above'
    var_20 = {}
    var_21 = {var_6: var_20}
    var_22 = {}
    var_23 = {var_19: var_21, var_6: var_22}
    var_24 = {}
    var_25 = {var_6: var_24}
    var_26 = [var_4, var_5]
    var_27 = {}
    var_28 = {}
    var_29 = 1
    var_30 = []
    var_31 = True
    var_32 = 'no_sections'
    var_33 = {var_32: var_31}
    var_34 = module_0.Config(**var_33)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = [var_7]
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = {}
    var_19 = {var_5: var_18}
    var_20 = [var_4]
    var_21 = {}
    var_22 = {}
    var_23 = 1
    var_24 = []
    var_25 = 'stdlib'
    var_26 = 'Standard Library Imports'
    var_27 = {var_25: var_26}
    var_28 = 'import_headings'
    var_29 = {var_28: var_27}
    var_30 = module_0.Config(**var_29)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = [var_7]
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = {}
    var_19 = {var_5: var_18}
    var_20 = [var_4]
    var_21 = {}
    var_22 = {}
    var_23 = 1
    var_24 = []
    var_25 = 'stdlib'
    var_26 = 'End of Standard Library Imports'
    var_27 = {var_25: var_26}
    var_28 = 'import_footers'
    var_29 = {var_28: var_27}
    var_30 = module_0.Config(**var_29)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = [var_7]
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = {}
    var_19 = {var_5: var_18}
    var_20 = [var_4]
    var_21 = {}
    var_22 = {}
    var_23 = 1
    var_24 = []
    var_25 = True
    var_26 = 'ensure_newline_before_comments'
    var_27 = {var_26: var_25}
    var_28 = module_0.Config(**var_27)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = [var_7]
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = {}
    var_19 = {var_5: var_18}
    var_20 = [var_4]
    var_21 = {}
    var_22 = {}
    var_23 = 1
    var_24 = []
    var_25 = lambda x, y, z: x.upper()
    var_26 = 'formatting_function'
    var_27 = {var_26: var_25}
    var_28 = module_0.Config(**var_27)



# Parsed testcases at query #14
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_sorted_imports_with_empty_parsed_content. Retrieved 22/24 statements.
# Partially parsed test_sorted_imports_with_no_imports. Retrieved 22/24 statements.
# Partially parsed test_sorted_imports_with_single_straight_import. Retrieved 28/30 statements.
# Partially parsed test_sorted_imports_with_multiple_straight_imports. Retrieved 30/32 statements.
# Partially parsed test_sorted_imports_with_from_imports. Retrieved 30/32 statements.
# Partially parsed test_sorted_imports_with_combined_straight_imports. Retrieved 31/33 statements.
# Partially parsed test_sorted_imports_with_comments. Retrieved 32/34 statements.
# Partially parsed test_sorted_imports_with_remove_imports. Retrieved 31/33 statements.
# Partially parsed test_sorted_imports_with_as_imports. Retrieved 30/32 statements.
# Partially parsed test_sorted_imports_with_force_sort_within_sections. Retrieved 31/33 statements.
# Partially parsed test_sorted_imports_with_import_headings. Retrieved 31/33 statements.
# Partially parsed test_sorted_imports_with_lines_between_sections. Retrieved 34/36 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = {}
    var_3 = 'above'
    var_4 = 'straight'
    var_5 = 'from'
    var_6 = {}
    var_7 = {}
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {}
    var_10 = {}
    var_11 = {var_3: var_8, var_4: var_9, var_5: var_10}
    var_12 = {}
    var_13 = {}
    var_14 = {var_4: var_12, var_5: var_13}
    var_15 = -1
    var_16 = 0
    var_17 = '\n'
    var_18 = []
    var_19 = {}
    var_20 = {}
    var_21 = []
    var_22 = {}
    var_23 = module_0.Config(**var_22)

import isort.settings as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = {}
    var_3 = 'above'
    var_4 = 'straight'
    var_5 = 'from'
    var_6 = {}
    var_7 = {}
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {}
    var_10 = {}
    var_11 = {var_3: var_8, var_4: var_9, var_5: var_10}
    var_12 = {}
    var_13 = {}
    var_14 = {var_4: var_12, var_5: var_13}
    var_15 = -1
    var_16 = 1
    var_17 = '\n'
    var_18 = []
    var_19 = {}
    var_20 = {}
    var_21 = []
    var_22 = {}
    var_23 = module_0.Config(**var_22)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 'STDLIB'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = set()
    var_7 = {var_5: var_6}
    var_8 = {}
    var_9 = {var_3: var_7, var_4: var_8}
    var_10 = {var_2: var_9}
    var_11 = 'above'
    var_12 = {}
    var_13 = {}
    var_14 = {var_3: var_12, var_4: var_13}
    var_15 = {}
    var_16 = {}
    var_17 = {var_11: var_14, var_3: var_15, var_4: var_16}
    var_18 = {}
    var_19 = {}
    var_20 = {var_3: var_18, var_4: var_19}
    var_21 = 0
    var_22 = 1
    var_23 = '\n'
    var_24 = [var_2]
    var_25 = {}
    var_26 = {}
    var_27 = []
    var_28 = {}
    var_29 = module_0.Config(**var_28)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 'STDLIB'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = 'sys'
    var_7 = set()
    var_8 = set()
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {}
    var_11 = {var_3: var_9, var_4: var_10}
    var_12 = {var_2: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {}
    var_16 = {var_3: var_14, var_4: var_15}
    var_17 = {}
    var_18 = {}
    var_19 = {var_13: var_16, var_3: var_17, var_4: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = {var_3: var_20, var_4: var_21}
    var_23 = 0
    var_24 = 1
    var_25 = '\n'
    var_26 = [var_2]
    var_27 = {}
    var_28 = {}
    var_29 = []
    var_30 = {}
    var_31 = module_0.Config(**var_30)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 'STDLIB'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = {}
    var_6 = 'os'
    var_7 = 'path'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = {var_3: var_5, var_4: var_10}
    var_12 = {var_2: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {}
    var_16 = {var_3: var_14, var_4: var_15}
    var_17 = {}
    var_18 = {}
    var_19 = {var_13: var_16, var_3: var_17, var_4: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = {var_3: var_20, var_4: var_21}
    var_23 = 0
    var_24 = 1
    var_25 = '\n'
    var_26 = [var_2]
    var_27 = {}
    var_28 = {}
    var_29 = []
    var_30 = {}
    var_31 = module_0.Config(**var_30)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 'STDLIB'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = 'sys'
    var_7 = set()
    var_8 = set()
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {}
    var_11 = {var_3: var_9, var_4: var_10}
    var_12 = {var_2: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {}
    var_16 = {var_3: var_14, var_4: var_15}
    var_17 = {}
    var_18 = {}
    var_19 = {var_13: var_16, var_3: var_17, var_4: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = {var_3: var_20, var_4: var_21}
    var_23 = 0
    var_24 = 1
    var_25 = '\n'
    var_26 = [var_2]
    var_27 = {}
    var_28 = {}
    var_29 = []
    var_30 = True
    var_31 = 'combine_straight_imports'
    var_32 = {var_31: var_30}
    var_33 = module_0.Config(**var_32)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 'STDLIB'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = set()
    var_7 = {var_5: var_6}
    var_8 = {}
    var_9 = {var_3: var_7, var_4: var_8}
    var_10 = {var_2: var_9}
    var_11 = 'above'
    var_12 = '# Comment above'
    var_13 = [var_12]
    var_14 = {var_5: var_13}
    var_15 = {}
    var_16 = {var_3: var_14, var_4: var_15}
    var_17 = '# Inline comment'
    var_18 = [var_17]
    var_19 = {var_5: var_18}
    var_20 = {}
    var_21 = {var_11: var_16, var_3: var_19, var_4: var_20}
    var_22 = {}
    var_23 = {}
    var_24 = {var_3: var_22, var_4: var_23}
    var_25 = 0
    var_26 = 1
    var_27 = '\n'
    var_28 = [var_2]
    var_29 = {}
    var_30 = {}
    var_31 = []
    var_32 = {}
    var_33 = module_0.Config(**var_32)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 'STDLIB'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = 'sys'
    var_7 = set()
    var_8 = set()
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {}
    var_11 = {var_3: var_9, var_4: var_10}
    var_12 = {var_2: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {}
    var_16 = {var_3: var_14, var_4: var_15}
    var_17 = {}
    var_18 = {}
    var_19 = {var_13: var_16, var_3: var_17, var_4: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = {var_3: var_20, var_4: var_21}
    var_23 = 0
    var_24 = 1
    var_25 = '\n'
    var_26 = [var_2]
    var_27 = {}
    var_28 = {}
    var_29 = []
    var_30 = [var_5]
    var_31 = 'remove_imports'
    var_32 = {var_31: var_30}
    var_33 = module_0.Config(**var_32)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 'STDLIB'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = set()
    var_7 = {var_5: var_6}
    var_8 = {}
    var_9 = {var_3: var_7, var_4: var_8}
    var_10 = {var_2: var_9}
    var_11 = 'above'
    var_12 = {}
    var_13 = {}
    var_14 = {var_3: var_12, var_4: var_13}
    var_15 = {}
    var_16 = {}
    var_17 = {var_11: var_14, var_3: var_15, var_4: var_16}
    var_18 = 'ospath'
    var_19 = [var_18]
    var_20 = {var_5: var_19}
    var_21 = {}
    var_22 = {var_3: var_20, var_4: var_21}
    var_23 = 0
    var_24 = 1
    var_25 = '\n'
    var_26 = [var_2]
    var_27 = {}
    var_28 = {}
    var_29 = []
    var_30 = {}
    var_31 = module_0.Config(**var_30)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 'STDLIB'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'zlib'
    var_6 = 'os'
    var_7 = set()
    var_8 = set()
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {}
    var_11 = {var_3: var_9, var_4: var_10}
    var_12 = {var_2: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {}
    var_16 = {var_3: var_14, var_4: var_15}
    var_17 = {}
    var_18 = {}
    var_19 = {var_13: var_16, var_3: var_17, var_4: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = {var_3: var_20, var_4: var_21}
    var_23 = 0
    var_24 = 1
    var_25 = '\n'
    var_26 = [var_2]
    var_27 = {}
    var_28 = {}
    var_29 = []
    var_30 = True
    var_31 = 'force_sort_within_sections'
    var_32 = {var_31: var_30}
    var_33 = module_0.Config(**var_32)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 'STDLIB'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = set()
    var_7 = {var_5: var_6}
    var_8 = {}
    var_9 = {var_3: var_7, var_4: var_8}
    var_10 = {var_2: var_9}
    var_11 = 'above'
    var_12 = {}
    var_13 = {}
    var_14 = {var_3: var_12, var_4: var_13}
    var_15 = {}
    var_16 = {}
    var_17 = {var_11: var_14, var_3: var_15, var_4: var_16}
    var_18 = {}
    var_19 = {}
    var_20 = {var_3: var_18, var_4: var_19}
    var_21 = 0
    var_22 = 1
    var_23 = '\n'
    var_24 = [var_2]
    var_25 = {}
    var_26 = {}
    var_27 = []
    var_28 = 'stdlib'
    var_29 = 'Standard Library'
    var_30 = {var_28: var_29}
    var_31 = 'import_headings'
    var_32 = {var_31: var_30}
    var_33 = module_0.Config(**var_32)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 'STDLIB'
    var_3 = 'THIRDPARTY'
    var_4 = 'straight'
    var_5 = 'from'
    var_6 = 'os'
    var_7 = set()
    var_8 = {var_6: var_7}
    var_9 = {}
    var_10 = {var_4: var_8, var_5: var_9}
    var_11 = 'django'
    var_12 = set()
    var_13 = {var_11: var_12}
    var_14 = {}
    var_15 = {var_4: var_13, var_5: var_14}
    var_16 = {var_2: var_10, var_3: var_15}
    var_17 = 'above'
    var_18 = {}
    var_19 = {}
    var_20 = {var_4: var_18, var_5: var_19}
    var_21 = {}
    var_22 = {}
    var_23 = {var_17: var_20, var_4: var_21, var_5: var_22}
    var_24 = {}
    var_25 = {}
    var_26 = {var_4: var_24, var_5: var_25}
    var_27 = 0
    var_28 = 1
    var_29 = '\n'
    var_30 = [var_2, var_3]
    var_31 = {}
    var_32 = {}
    var_33 = []
    var_34 = 'lines_between_sections'
    var_35 = {var_34: var_28}
    var_36 = module_0.Config(**var_35)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_with_straight_imports_empty_modules. Retrieved 15/17 statements.
# Partially parsed test_with_straight_imports_combine_no_as_imports. Retrieved 25/27 statements.
# Partially parsed test_with_straight_imports_combine_with_as_imports. Retrieved 27/29 statements.
# Partially parsed test_with_straight_imports_no_combine. Retrieved 25/27 statements.
# Partially parsed test_with_straight_imports_remove_imports. Retrieved 25/27 statements.
# Partially parsed test_with_straight_imports_ignore_comments. Retrieved 26/28 statements.
# Partially parsed test_with_straight_imports_custom_comment_prefix. Retrieved 26/28 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'straight'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'above'
    var_4 = {}
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = {var_3: var_5, var_0: var_6}
    var_8 = {}
    var_9 = {var_0: var_8}
    var_10 = []
    var_11 = True
    var_12 = 'combine_straight_imports'
    var_13 = {var_12: var_11}
    var_14 = module_0.Config(**var_13)
    var_15 = []
    var_16 = []
    var_17 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'straight'
    var_1 = 'sys'
    var_2 = 'os'
    var_3 = [var_2]
    var_4 = []
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'above'
    var_8 = '# sys comment'
    var_9 = [var_8]
    var_10 = {var_1: var_9}
    var_11 = {var_0: var_10}
    var_12 = '# inline sys'
    var_13 = [var_12]
    var_14 = '# inline os'
    var_15 = [var_14]
    var_16 = {var_1: var_13, var_2: var_15}
    var_17 = {var_7: var_11, var_0: var_16}
    var_18 = {}
    var_19 = {var_0: var_18}
    var_20 = []
    var_21 = True
    var_22 = 'combine_straight_imports'
    var_23 = {var_22: var_21}
    var_24 = module_0.Config(**var_23)
    var_25 = [var_1, var_2]
    var_26 = []
    var_27 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'straight'
    var_1 = 'sys'
    var_2 = 'os'
    var_3 = [var_2]
    var_4 = []
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'above'
    var_8 = '# sys comment'
    var_9 = [var_8]
    var_10 = {var_1: var_9}
    var_11 = {var_0: var_10}
    var_12 = '# inline sys'
    var_13 = [var_12]
    var_14 = '# inline os'
    var_15 = [var_14]
    var_16 = {var_1: var_13, var_2: var_15}
    var_17 = {var_7: var_11, var_0: var_16}
    var_18 = 'sys_alias'
    var_19 = [var_18]
    var_20 = {var_1: var_19}
    var_21 = {var_0: var_20}
    var_22 = []
    var_23 = True
    var_24 = 'combine_straight_imports'
    var_25 = {var_24: var_23}
    var_26 = module_0.Config(**var_25)
    var_27 = [var_1, var_2]
    var_28 = []
    var_29 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'straight'
    var_1 = 'sys'
    var_2 = 'os'
    var_3 = [var_2]
    var_4 = []
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'above'
    var_8 = '# sys comment'
    var_9 = [var_8]
    var_10 = {var_1: var_9}
    var_11 = {var_0: var_10}
    var_12 = '# inline sys'
    var_13 = [var_12]
    var_14 = '# inline os'
    var_15 = [var_14]
    var_16 = {var_1: var_13, var_2: var_15}
    var_17 = {var_7: var_11, var_0: var_16}
    var_18 = {}
    var_19 = {var_0: var_18}
    var_20 = []
    var_21 = False
    var_22 = 'combine_straight_imports'
    var_23 = {var_22: var_21}
    var_24 = module_0.Config(**var_23)
    var_25 = [var_1, var_2]
    var_26 = []
    var_27 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'straight'
    var_1 = 'sys'
    var_2 = 'os'
    var_3 = [var_2]
    var_4 = []
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'above'
    var_8 = '# sys comment'
    var_9 = [var_8]
    var_10 = {var_1: var_9}
    var_11 = {var_0: var_10}
    var_12 = '# inline sys'
    var_13 = [var_12]
    var_14 = '# inline os'
    var_15 = [var_14]
    var_16 = {var_1: var_13, var_2: var_15}
    var_17 = {var_7: var_11, var_0: var_16}
    var_18 = {}
    var_19 = {var_0: var_18}
    var_20 = []
    var_21 = False
    var_22 = 'combine_straight_imports'
    var_23 = {var_22: var_21}
    var_24 = module_0.Config(**var_23)
    var_25 = [var_1, var_2]
    var_26 = [var_1]
    var_27 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'straight'
    var_1 = 'sys'
    var_2 = 'os'
    var_3 = [var_2]
    var_4 = []
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'above'
    var_8 = '# sys comment'
    var_9 = [var_8]
    var_10 = {var_1: var_9}
    var_11 = {var_0: var_10}
    var_12 = '# inline sys'
    var_13 = [var_12]
    var_14 = '# inline os'
    var_15 = [var_14]
    var_16 = {var_1: var_13, var_2: var_15}
    var_17 = {var_7: var_11, var_0: var_16}
    var_18 = {}
    var_19 = {var_0: var_18}
    var_20 = []
    var_21 = False
    var_22 = True
    var_23 = 'combine_straight_imports'
    var_24 = 'ignore_comments'
    var_25 = {var_23: var_21, var_24: var_22}
    var_26 = module_0.Config(**var_25)
    var_27 = [var_1, var_2]
    var_28 = []
    var_29 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'straight'
    var_1 = 'sys'
    var_2 = 'os'
    var_3 = [var_2]
    var_4 = []
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'above'
    var_8 = '# sys comment'
    var_9 = [var_8]
    var_10 = {var_1: var_9}
    var_11 = {var_0: var_10}
    var_12 = '# inline sys'
    var_13 = [var_12]
    var_14 = '# inline os'
    var_15 = [var_14]
    var_16 = {var_1: var_13, var_2: var_15}
    var_17 = {var_7: var_11, var_0: var_16}
    var_18 = {}
    var_19 = {var_0: var_18}
    var_20 = []
    var_21 = False
    var_22 = ' # '
    var_23 = 'combine_straight_imports'
    var_24 = 'comment_prefix'
    var_25 = {var_23: var_21, var_24: var_22}
    var_26 = module_0.Config(**var_25)
    var_27 = [var_1, var_2]
    var_28 = []
    var_29 = 'import'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_sorted_imports_no_imports. Retrieved 12/14 statements.
# Partially parsed test_sorted_imports_single_import. Retrieved 25/27 statements.
# Partially parsed test_sorted_imports_multiple_imports. Retrieved 27/29 statements.
# Partially parsed test_sorted_imports_with_comments. Retrieved 28/30 statements.
# Partially parsed test_sorted_imports_with_as_imports. Retrieved 27/29 statements.
# Partially parsed test_sorted_imports_with_from_imports. Retrieved 27/29 statements.
# Partially parsed test_sorted_imports_with_sections. Retrieved 31/33 statements.
# Partially parsed test_sorted_imports_with_remove_imports. Retrieved 28/30 statements.
# Partially parsed test_sorted_imports_with_combine_straight_imports. Retrieved 28/30 statements.
# Partially parsed test_sorted_imports_with_force_sort_within_sections. Retrieved 28/30 statements.
# Partially parsed test_sorted_imports_with_import_headings. Retrieved 28/30 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = -1
    var_3 = '\n'
    var_4 = {}
    var_5 = {}
    var_6 = {}
    var_7 = []
    var_8 = {}
    var_9 = {}
    var_10 = 1
    var_11 = []
    var_12 = {}
    var_13 = module_0.Config(**var_12)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = {}
    var_19 = {var_5: var_18}
    var_20 = [var_4]
    var_21 = {}
    var_22 = {}
    var_23 = 1
    var_24 = []
    var_25 = {}
    var_26 = module_0.Config(**var_25)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = 'sys'
    var_9 = set()
    var_10 = set()
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = {var_5: var_11, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = 'above'
    var_16 = {}
    var_17 = {var_5: var_16}
    var_18 = {}
    var_19 = {var_15: var_17, var_5: var_18}
    var_20 = {}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []
    var_27 = {}
    var_28 = module_0.Config(**var_27)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = '# comment'
    var_17 = [var_16]
    var_18 = {var_7: var_17}
    var_19 = {var_13: var_15, var_5: var_18}
    var_20 = {}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []
    var_27 = False
    var_28 = 'ignore_comments'
    var_29 = {var_28: var_27}
    var_30 = module_0.Config(**var_29)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = 'path'
    var_19 = [var_18]
    var_20 = {var_7: var_19}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []
    var_27 = {}
    var_28 = module_0.Config(**var_27)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = {}
    var_8 = 'os'
    var_9 = 'path'
    var_10 = set()
    var_11 = {var_9: var_10}
    var_12 = {var_8: var_11}
    var_13 = {var_5: var_7, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = 'above'
    var_16 = {}
    var_17 = {var_5: var_16}
    var_18 = {}
    var_19 = {var_15: var_17, var_5: var_18}
    var_20 = {}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []
    var_27 = {}
    var_28 = module_0.Config(**var_27)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'FUTURE'
    var_5 = 'STDLIB'
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = '__future__'
    var_9 = set()
    var_10 = {var_8: var_9}
    var_11 = {}
    var_12 = {var_6: var_10, var_7: var_11}
    var_13 = 'os'
    var_14 = set()
    var_15 = {var_13: var_14}
    var_16 = {}
    var_17 = {var_6: var_15, var_7: var_16}
    var_18 = {var_4: var_12, var_5: var_17}
    var_19 = 'above'
    var_20 = {}
    var_21 = {var_6: var_20}
    var_22 = {}
    var_23 = {var_19: var_21, var_6: var_22}
    var_24 = {}
    var_25 = {var_6: var_24}
    var_26 = [var_4, var_5]
    var_27 = {}
    var_28 = {}
    var_29 = 1
    var_30 = []
    var_31 = {}
    var_32 = module_0.Config(**var_31)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = 'sys'
    var_9 = set()
    var_10 = set()
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = {var_5: var_11, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = 'above'
    var_16 = {}
    var_17 = {var_5: var_16}
    var_18 = {}
    var_19 = {var_15: var_17, var_5: var_18}
    var_20 = {}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []
    var_27 = [var_7]
    var_28 = 'remove_imports'
    var_29 = {var_28: var_27}
    var_30 = module_0.Config(**var_29)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = 'sys'
    var_9 = set()
    var_10 = set()
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = {var_5: var_11, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = 'above'
    var_16 = {}
    var_17 = {var_5: var_16}
    var_18 = {}
    var_19 = {var_15: var_17, var_5: var_18}
    var_20 = {}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []
    var_27 = True
    var_28 = 'combine_straight_imports'
    var_29 = {var_28: var_27}
    var_30 = module_0.Config(**var_29)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'sys'
    var_8 = 'os'
    var_9 = set()
    var_10 = set()
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = {var_5: var_11, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = 'above'
    var_16 = {}
    var_17 = {var_5: var_16}
    var_18 = {}
    var_19 = {var_15: var_17, var_5: var_18}
    var_20 = {}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []
    var_27 = True
    var_28 = 'force_sort_within_sections'
    var_29 = {var_28: var_27}
    var_30 = module_0.Config(**var_29)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = {}
    var_19 = {var_5: var_18}
    var_20 = [var_4]
    var_21 = {}
    var_22 = {}
    var_23 = 1
    var_24 = []
    var_25 = 'stdlib'
    var_26 = 'Standard Library'
    var_27 = {var_25: var_26}
    var_28 = 'import_headings'
    var_29 = {var_28: var_27}
    var_30 = module_0.Config(**var_29)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_with_straight_imports_empty_modules. Retrieved 15/17 statements.
# Partially parsed test_with_straight_imports_combine_without_as. Retrieved 25/27 statements.
# Partially parsed test_with_straight_imports_combine_with_as. Retrieved 29/31 statements.
# Partially parsed test_with_straight_imports_no_combine. Retrieved 26/28 statements.
# Partially parsed test_with_straight_imports_remove_imports. Retrieved 25/27 statements.
# Partially parsed test_with_straight_imports_ignore_comments. Retrieved 26/28 statements.
# Partially parsed test_with_straight_imports_custom_comment_prefix. Retrieved 26/28 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'above'
    var_2 = 'straight'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = {}
    var_6 = {var_1: var_4, var_2: var_5}
    var_7 = {}
    var_8 = {var_2: var_7}
    var_9 = []
    var_10 = True
    var_11 = 'combine_straight_imports'
    var_12 = {var_11: var_10}
    var_13 = module_0.Config(**var_12)
    var_14 = []
    var_15 = 'standard'
    var_16 = []
    var_17 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'standard'
    var_1 = 'straight'
    var_2 = 'sys'
    var_3 = 'os'
    var_4 = []
    var_5 = []
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = {}
    var_11 = {var_1: var_10}
    var_12 = 'comment1'
    var_13 = [var_12]
    var_14 = 'comment2'
    var_15 = [var_14]
    var_16 = {var_2: var_13, var_3: var_15}
    var_17 = {var_9: var_11, var_1: var_16}
    var_18 = {}
    var_19 = {var_1: var_18}
    var_20 = []
    var_21 = True
    var_22 = 'combine_straight_imports'
    var_23 = {var_22: var_21}
    var_24 = module_0.Config(**var_23)
    var_25 = [var_2, var_3]
    var_26 = []
    var_27 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'standard'
    var_1 = 'straight'
    var_2 = 'sys'
    var_3 = 'os'
    var_4 = []
    var_5 = []
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = {}
    var_11 = {var_1: var_10}
    var_12 = 'comment1'
    var_13 = [var_12]
    var_14 = 'comment2'
    var_15 = [var_14]
    var_16 = {var_2: var_13, var_3: var_15}
    var_17 = {var_9: var_11, var_1: var_16}
    var_18 = 's'
    var_19 = [var_18]
    var_20 = 'o'
    var_21 = [var_20]
    var_22 = {var_2: var_19, var_3: var_21}
    var_23 = {var_1: var_22}
    var_24 = []
    var_25 = True
    var_26 = 'combine_straight_imports'
    var_27 = {var_26: var_25}
    var_28 = module_0.Config(**var_27)
    var_29 = [var_2, var_3]
    var_30 = []
    var_31 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'standard'
    var_1 = 'straight'
    var_2 = 'sys'
    var_3 = 'os'
    var_4 = []
    var_5 = []
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'above comment'
    var_11 = [var_10]
    var_12 = {var_2: var_11}
    var_13 = {var_1: var_12}
    var_14 = 'inline comment'
    var_15 = [var_14]
    var_16 = [var_14]
    var_17 = {var_2: var_15, var_3: var_16}
    var_18 = {var_9: var_13, var_1: var_17}
    var_19 = {}
    var_20 = {var_1: var_19}
    var_21 = []
    var_22 = False
    var_23 = 'combine_straight_imports'
    var_24 = {var_23: var_22}
    var_25 = module_0.Config(**var_24)
    var_26 = [var_2, var_3]
    var_27 = []
    var_28 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'standard'
    var_1 = 'straight'
    var_2 = 'sys'
    var_3 = 'os'
    var_4 = []
    var_5 = []
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = {}
    var_11 = {var_1: var_10}
    var_12 = 'comment1'
    var_13 = [var_12]
    var_14 = 'comment2'
    var_15 = [var_14]
    var_16 = {var_2: var_13, var_3: var_15}
    var_17 = {var_9: var_11, var_1: var_16}
    var_18 = {}
    var_19 = {var_1: var_18}
    var_20 = []
    var_21 = False
    var_22 = 'combine_straight_imports'
    var_23 = {var_22: var_21}
    var_24 = module_0.Config(**var_23)
    var_25 = [var_2, var_3]
    var_26 = [var_2]
    var_27 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'standard'
    var_1 = 'straight'
    var_2 = 'sys'
    var_3 = 'os'
    var_4 = []
    var_5 = []
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = {}
    var_11 = {var_1: var_10}
    var_12 = 'comment1'
    var_13 = [var_12]
    var_14 = 'comment2'
    var_15 = [var_14]
    var_16 = {var_2: var_13, var_3: var_15}
    var_17 = {var_9: var_11, var_1: var_16}
    var_18 = {}
    var_19 = {var_1: var_18}
    var_20 = []
    var_21 = False
    var_22 = True
    var_23 = 'combine_straight_imports'
    var_24 = 'ignore_comments'
    var_25 = {var_23: var_21, var_24: var_22}
    var_26 = module_0.Config(**var_25)
    var_27 = [var_2, var_3]
    var_28 = []
    var_29 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'standard'
    var_1 = 'straight'
    var_2 = 'sys'
    var_3 = 'os'
    var_4 = []
    var_5 = []
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = {}
    var_11 = {var_1: var_10}
    var_12 = 'comment1'
    var_13 = [var_12]
    var_14 = 'comment2'
    var_15 = [var_14]
    var_16 = {var_2: var_13, var_3: var_15}
    var_17 = {var_9: var_11, var_1: var_16}
    var_18 = {}
    var_19 = {var_1: var_18}
    var_20 = []
    var_21 = False
    var_22 = ' # '
    var_23 = 'combine_straight_imports'
    var_24 = 'comment_prefix'
    var_25 = {var_23: var_21, var_24: var_22}
    var_26 = module_0.Config(**var_25)
    var_27 = [var_2, var_3]
    var_28 = []
    var_29 = 'import'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_predicate_at_line_1. Retrieved 7/10 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = 'module1'
    var_4 = 'module2'
    var_5 = [var_3, var_4]
    var_6 = 'section'
    var_7 = []
    var_8 = 'import'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test__with_from_imports_basic_case. Retrieved 21/23 statements.
# Partially parsed test__with_from_imports_with_comments. Retrieved 22/24 statements.
# Partially parsed test__with_from_imports_remove_imports. Retrieved 24/26 statements.
# Partially parsed test__with_from_imports_with_as_imports. Retrieved 22/24 statements.
# Partially parsed test__with_from_imports_with_star. Retrieved 22/24 statements.
# Partially parsed test__with_from_imports_with_force_single_line. Retrieved 25/27 statements.
# Partially parsed test__with_from_imports_with_combine_as_imports. Retrieved 23/25 statements.
# Partially parsed test__with_from_imports_with_ignore_comments. Retrieved 23/25 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = []
    var_9 = {var_2: var_8}
    var_10 = {var_1: var_9}
    var_11 = 'os.path'
    var_12 = []
    var_13 = {var_11: var_12}
    var_14 = {var_1: var_13}
    var_15 = '\n'
    var_16 = set()
    var_17 = []
    var_18 = {}
    var_19 = module_0.Config(**var_18)
    var_20 = [var_2]
    var_21 = []
    var_22 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = '# comment'
    var_9 = [var_8]
    var_10 = {var_2: var_9}
    var_11 = {var_1: var_10}
    var_12 = 'os.path'
    var_13 = []
    var_14 = {var_12: var_13}
    var_15 = {var_1: var_14}
    var_16 = '\n'
    var_17 = set()
    var_18 = []
    var_19 = {}
    var_20 = module_0.Config(**var_19)
    var_21 = [var_2]
    var_22 = []
    var_23 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'sys'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = []
    var_10 = {var_2: var_9}
    var_11 = {var_1: var_10}
    var_12 = 'os.path'
    var_13 = 'os.sys'
    var_14 = []
    var_15 = []
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = {var_1: var_16}
    var_18 = '\n'
    var_19 = set()
    var_20 = []
    var_21 = {}
    var_22 = module_0.Config(**var_21)
    var_23 = [var_2]
    var_24 = [var_13]
    var_25 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = []
    var_9 = {var_2: var_8}
    var_10 = {var_1: var_9}
    var_11 = 'os.path'
    var_12 = 'path as ospath'
    var_13 = [var_12]
    var_14 = {var_11: var_13}
    var_15 = {var_1: var_14}
    var_16 = '\n'
    var_17 = set()
    var_18 = []
    var_19 = {}
    var_20 = module_0.Config(**var_19)
    var_21 = [var_2]
    var_22 = []
    var_23 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = '*'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'nested'
    var_9 = '# star comment'
    var_10 = [var_9]
    var_11 = {var_3: var_10}
    var_12 = {var_2: var_11}
    var_13 = {var_8: var_12}
    var_14 = {}
    var_15 = {var_1: var_14}
    var_16 = '\n'
    var_17 = set()
    var_18 = []
    var_19 = {}
    var_20 = module_0.Config(**var_19)
    var_21 = [var_2]
    var_22 = []
    var_23 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'sys'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = []
    var_10 = {var_2: var_9}
    var_11 = {var_1: var_10}
    var_12 = 'os.path'
    var_13 = 'os.sys'
    var_14 = []
    var_15 = []
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = {var_1: var_16}
    var_18 = '\n'
    var_19 = set()
    var_20 = []
    var_21 = True
    var_22 = 'force_single_line'
    var_23 = {var_22: var_21}
    var_24 = module_0.Config(**var_23)
    var_25 = [var_2]
    var_26 = []
    var_27 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = []
    var_9 = {var_2: var_8}
    var_10 = {var_1: var_9}
    var_11 = 'os.path'
    var_12 = 'path as ospath'
    var_13 = [var_12]
    var_14 = {var_11: var_13}
    var_15 = {var_1: var_14}
    var_16 = '\n'
    var_17 = set()
    var_18 = []
    var_19 = True
    var_20 = 'combine_as_imports'
    var_21 = {var_20: var_19}
    var_22 = module_0.Config(**var_21)
    var_23 = [var_2]
    var_24 = []
    var_25 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = '# comment'
    var_9 = [var_8]
    var_10 = {var_2: var_9}
    var_11 = {var_1: var_10}
    var_12 = 'os.path'
    var_13 = []
    var_14 = {var_12: var_13}
    var_15 = {var_1: var_14}
    var_16 = '\n'
    var_17 = set()
    var_18 = []
    var_19 = True
    var_20 = 'ignore_comments'
    var_21 = {var_20: var_19}
    var_22 = module_0.Config(**var_21)
    var_23 = [var_2]
    var_24 = []
    var_25 = 'import'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_sorted_imports_no_imports. Retrieved 13/15 statements.
# Partially parsed test_sorted_imports_with_imports. Retrieved 28/30 statements.
# Partially parsed test_sorted_imports_with_combined_straight_imports. Retrieved 29/31 statements.
# Partially parsed test_sorted_imports_with_from_imports. Retrieved 28/30 statements.
# Partially parsed test_sorted_imports_with_comments. Retrieved 30/32 statements.
# Partially parsed test_sorted_imports_with_remove_imports. Retrieved 29/31 statements.
# Partially parsed test_sorted_imports_with_no_sections. Retrieved 33/35 statements.
# Partially parsed test_sorted_imports_with_force_sort_within_sections. Retrieved 29/31 statements.
# Partially parsed test_sorted_imports_with_import_headings. Retrieved 29/31 statements.
# Partially parsed test_sorted_imports_with_ensure_newline_before_comments. Retrieved 27/29 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = -1
    var_3 = '\n'
    var_4 = 1
    var_5 = {}
    var_6 = {}
    var_7 = {}
    var_8 = {}
    var_9 = {}
    var_10 = []
    var_11 = []
    var_12 = []
    var_13 = {}
    var_14 = module_0.Config(**var_13)

import isort.settings as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 2
    var_5 = 'THIRDPARTY'
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = 'os'
    var_9 = 'sys'
    var_10 = set()
    var_11 = set()
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = {}
    var_14 = {var_6: var_12, var_7: var_13}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_6: var_16}
    var_18 = 'above'
    var_19 = {}
    var_20 = {var_6: var_19}
    var_21 = {}
    var_22 = {var_18: var_20, var_6: var_21}
    var_23 = {}
    var_24 = {}
    var_25 = [var_5]
    var_26 = []
    var_27 = []
    var_28 = {}
    var_29 = module_0.Config(**var_28)

import isort.settings as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 2
    var_5 = 'THIRDPARTY'
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = 'os'
    var_9 = 'sys'
    var_10 = set()
    var_11 = set()
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = {}
    var_14 = {var_6: var_12, var_7: var_13}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_6: var_16}
    var_18 = 'above'
    var_19 = {}
    var_20 = {var_6: var_19}
    var_21 = {}
    var_22 = {var_18: var_20, var_6: var_21}
    var_23 = {}
    var_24 = {}
    var_25 = [var_5]
    var_26 = []
    var_27 = []
    var_28 = True
    var_29 = 'combine_straight_imports'
    var_30 = {var_29: var_28}
    var_31 = module_0.Config(**var_30)

import isort.settings as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 2
    var_5 = 'THIRDPARTY'
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = {}
    var_9 = 'os'
    var_10 = 'path'
    var_11 = set()
    var_12 = {var_10: var_11}
    var_13 = {var_9: var_12}
    var_14 = {var_6: var_8, var_7: var_13}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_6: var_16}
    var_18 = 'above'
    var_19 = {}
    var_20 = {var_6: var_19}
    var_21 = {}
    var_22 = {var_18: var_20, var_6: var_21}
    var_23 = {}
    var_24 = {}
    var_25 = [var_5]
    var_26 = []
    var_27 = []
    var_28 = {}
    var_29 = module_0.Config(**var_28)

import isort.settings as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 2
    var_5 = 'THIRDPARTY'
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = 'os'
    var_9 = set()
    var_10 = {var_8: var_9}
    var_11 = {}
    var_12 = {var_6: var_10, var_7: var_11}
    var_13 = {var_5: var_12}
    var_14 = {}
    var_15 = {var_6: var_14}
    var_16 = 'above'
    var_17 = '# Comment above os'
    var_18 = [var_17]
    var_19 = {var_8: var_18}
    var_20 = {var_6: var_19}
    var_21 = '# Comment inline os'
    var_22 = [var_21]
    var_23 = {var_8: var_22}
    var_24 = {var_16: var_20, var_6: var_23}
    var_25 = {}
    var_26 = {}
    var_27 = [var_5]
    var_28 = []
    var_29 = []
    var_30 = {}
    var_31 = module_0.Config(**var_30)

import isort.settings as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 2
    var_5 = 'THIRDPARTY'
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = 'os'
    var_9 = 'sys'
    var_10 = set()
    var_11 = set()
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = {}
    var_14 = {var_6: var_12, var_7: var_13}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_6: var_16}
    var_18 = 'above'
    var_19 = {}
    var_20 = {var_6: var_19}
    var_21 = {}
    var_22 = {var_18: var_20, var_6: var_21}
    var_23 = {}
    var_24 = {}
    var_25 = [var_5]
    var_26 = []
    var_27 = []
    var_28 = [var_8]
    var_29 = 'remove_imports'
    var_30 = {var_29: var_28}
    var_31 = module_0.Config(**var_30)

import isort.settings as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 3
    var_5 = 'FUTURE'
    var_6 = 'THIRDPARTY'
    var_7 = 'straight'
    var_8 = 'from'
    var_9 = '__future__'
    var_10 = set()
    var_11 = {var_9: var_10}
    var_12 = {}
    var_13 = {var_7: var_11, var_8: var_12}
    var_14 = 'os'
    var_15 = set()
    var_16 = {var_14: var_15}
    var_17 = {}
    var_18 = {var_7: var_16, var_8: var_17}
    var_19 = {var_5: var_13, var_6: var_18}
    var_20 = {}
    var_21 = {var_7: var_20}
    var_22 = 'above'
    var_23 = {}
    var_24 = {var_7: var_23}
    var_25 = {}
    var_26 = {var_22: var_24, var_7: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = [var_5, var_6]
    var_30 = []
    var_31 = []
    var_32 = True
    var_33 = 'no_sections'
    var_34 = {var_33: var_32}
    var_35 = module_0.Config(**var_34)

import isort.settings as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 3
    var_5 = 'THIRDPARTY'
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = 'sys'
    var_9 = 'os'
    var_10 = set()
    var_11 = set()
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = {}
    var_14 = {var_6: var_12, var_7: var_13}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_6: var_16}
    var_18 = 'above'
    var_19 = {}
    var_20 = {var_6: var_19}
    var_21 = {}
    var_22 = {var_18: var_20, var_6: var_21}
    var_23 = {}
    var_24 = {}
    var_25 = [var_5]
    var_26 = []
    var_27 = []
    var_28 = True
    var_29 = 'force_sort_within_sections'
    var_30 = {var_29: var_28}
    var_31 = module_0.Config(**var_30)

import isort.settings as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 2
    var_5 = 'THIRDPARTY'
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = 'os'
    var_9 = set()
    var_10 = {var_8: var_9}
    var_11 = {}
    var_12 = {var_6: var_10, var_7: var_11}
    var_13 = {var_5: var_12}
    var_14 = {}
    var_15 = {var_6: var_14}
    var_16 = 'above'
    var_17 = {}
    var_18 = {var_6: var_17}
    var_19 = {}
    var_20 = {var_16: var_18, var_6: var_19}
    var_21 = {}
    var_22 = {}
    var_23 = [var_5]
    var_24 = []
    var_25 = []
    var_26 = 'thirdparty'
    var_27 = 'Third Party Imports'
    var_28 = {var_26: var_27}
    var_29 = 'import_headings'
    var_30 = {var_29: var_28}
    var_31 = module_0.Config(**var_30)

import isort.settings as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 3
    var_5 = 'THIRDPARTY'
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = 'os'
    var_9 = set()
    var_10 = {var_8: var_9}
    var_11 = {}
    var_12 = {var_6: var_10, var_7: var_11}
    var_13 = {var_5: var_12}
    var_14 = {}
    var_15 = {var_6: var_14}
    var_16 = 'above'
    var_17 = {}
    var_18 = {var_6: var_17}
    var_19 = {}
    var_20 = {var_16: var_18, var_6: var_19}
    var_21 = {}
    var_22 = {}
    var_23 = [var_5]
    var_24 = []
    var_25 = []
    var_26 = True
    var_27 = 'ensure_newline_before_comments'
    var_28 = {var_27: var_26}
    var_29 = module_0.Config(**var_28)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_with_from_imports_basic. Retrieved 20/25 statements.
# Partially parsed test_with_from_imports_with_comments. Retrieved 21/26 statements.
# Partially parsed test_with_from_imports_remove_imports. Retrieved 21/26 statements.
# Partially parsed test_with_from_imports_with_as_imports. Retrieved 22/27 statements.
# Partially parsed test_with_from_imports_with_star. Retrieved 21/26 statements.
# Partially parsed test_with_from_imports_force_single_line. Retrieved 21/26 statements.
# Partially parsed test_with_from_imports_combine_as_imports. Retrieved 23/28 statements.
# Partially parsed test_with_from_imports_ignore_comments. Retrieved 22/27 statements.
# Partially parsed test_with_from_imports_no_inline_sort. Retrieved 21/26 statements.
# Partially parsed test_with_from_imports_only_sections. Retrieved 20/25 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = 'os'
    var_4 = [var_3]
    var_5 = 'STDLIB'
    var_6 = []
    var_7 = 'import'
    var_8 = 'STDLIB'
    var_9 = 'from'
    var_10 = 'path'
    var_11 = 'sys'
    var_12 = [var_10, var_11]
    var_13 = {var_3: var_12}
    var_14 = {var_9: var_13}
    var_15 = {}
    var_16 = 'above'
    var_17 = 'nested'
    var_18 = {}
    var_19 = {}
    var_20 = {var_9: var_19}
    var_21 = {}

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = 'os'
    var_4 = [var_3]
    var_5 = 'STDLIB'
    var_6 = []
    var_7 = 'import'
    var_8 = 'STDLIB'
    var_9 = 'from'
    var_10 = 'path'
    var_11 = [var_10]
    var_12 = {var_3: var_11}
    var_13 = {var_9: var_12}
    var_14 = {}
    var_15 = 'above'
    var_16 = 'nested'
    var_17 = '# comment'
    var_18 = [var_17]
    var_19 = {var_3: var_18}
    var_20 = {}
    var_21 = {var_9: var_20}
    var_22 = {}

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = 'os'
    var_4 = [var_3]
    var_5 = 'STDLIB'
    var_6 = 'os.path'
    var_7 = [var_6]
    var_8 = 'import'
    var_9 = 'STDLIB'
    var_10 = 'from'
    var_11 = 'path'
    var_12 = 'sys'
    var_13 = [var_11, var_12]
    var_14 = {var_3: var_13}
    var_15 = {var_10: var_14}
    var_16 = {}
    var_17 = 'above'
    var_18 = 'nested'
    var_19 = {}
    var_20 = {}
    var_21 = {var_10: var_20}
    var_22 = {}

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = 'os'
    var_4 = [var_3]
    var_5 = 'STDLIB'
    var_6 = []
    var_7 = 'import'
    var_8 = 'STDLIB'
    var_9 = 'from'
    var_10 = 'path'
    var_11 = [var_10]
    var_12 = {var_3: var_11}
    var_13 = {var_9: var_12}
    var_14 = 'os.path'
    var_15 = 'path as ospath'
    var_16 = [var_15]
    var_17 = {var_14: var_16}
    var_18 = 'above'
    var_19 = 'nested'
    var_20 = {}
    var_21 = {}
    var_22 = {var_9: var_21}
    var_23 = {}

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = 'os'
    var_4 = [var_3]
    var_5 = 'STDLIB'
    var_6 = []
    var_7 = 'import'
    var_8 = 'STDLIB'
    var_9 = 'from'
    var_10 = '*'
    var_11 = [var_10]
    var_12 = {var_3: var_11}
    var_13 = {var_9: var_12}
    var_14 = {}
    var_15 = 'above'
    var_16 = 'nested'
    var_17 = {}
    var_18 = {}
    var_19 = {var_9: var_18}
    var_20 = '# star comment'
    var_21 = {var_10: var_20}
    var_22 = {var_3: var_21}

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = 'force_single_line'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'os'
    var_6 = [var_5]
    var_7 = 'STDLIB'
    var_8 = []
    var_9 = 'import'
    var_10 = 'STDLIB'
    var_11 = 'from'
    var_12 = 'path'
    var_13 = 'sys'
    var_14 = [var_12, var_13]
    var_15 = {var_5: var_14}
    var_16 = {var_11: var_15}
    var_17 = {}
    var_18 = 'above'
    var_19 = 'nested'
    var_20 = {}
    var_21 = {}
    var_22 = {var_11: var_21}
    var_23 = {}

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = 'combine_as_imports'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'os'
    var_6 = [var_5]
    var_7 = 'STDLIB'
    var_8 = []
    var_9 = 'import'
    var_10 = 'STDLIB'
    var_11 = 'from'
    var_12 = 'path'
    var_13 = [var_12]
    var_14 = {var_5: var_13}
    var_15 = {var_11: var_14}
    var_16 = 'os.path'
    var_17 = 'path as ospath'
    var_18 = [var_17]
    var_19 = {var_16: var_18}
    var_20 = 'above'
    var_21 = 'nested'
    var_22 = {}
    var_23 = {}
    var_24 = {var_11: var_23}
    var_25 = {}

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = 'ignore_comments'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'os'
    var_6 = [var_5]
    var_7 = 'STDLIB'
    var_8 = []
    var_9 = 'import'
    var_10 = 'STDLIB'
    var_11 = 'from'
    var_12 = 'path'
    var_13 = [var_12]
    var_14 = {var_5: var_13}
    var_15 = {var_11: var_14}
    var_16 = {}
    var_17 = 'above'
    var_18 = 'nested'
    var_19 = '# comment'
    var_20 = [var_19]
    var_21 = {var_5: var_20}
    var_22 = {}
    var_23 = {var_11: var_22}
    var_24 = {}

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = 'no_inline_sort'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'os'
    var_6 = [var_5]
    var_7 = 'STDLIB'
    var_8 = []
    var_9 = 'import'
    var_10 = 'STDLIB'
    var_11 = 'from'
    var_12 = 'sys'
    var_13 = 'path'
    var_14 = [var_12, var_13]
    var_15 = {var_5: var_14}
    var_16 = {var_11: var_15}
    var_17 = {}
    var_18 = 'above'
    var_19 = 'nested'
    var_20 = {}
    var_21 = {}
    var_22 = {var_11: var_21}
    var_23 = {}

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = 'only_sections'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'os'
    var_6 = [var_5]
    var_7 = 'STDLIB'
    var_8 = []
    var_9 = 'import'
    var_10 = 'STDLIB'
    var_11 = 'from'
    var_12 = 'path'
    var_13 = [var_12]
    var_14 = {var_5: var_13}
    var_15 = {var_11: var_14}
    var_16 = {}
    var_17 = 'above'
    var_18 = 'nested'
    var_19 = {}
    var_20 = {}
    var_21 = {var_11: var_20}
    var_22 = {}



# Parsed testcases at query #23
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #24
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_sorted_imports_no_imports. Retrieved 11/13 statements.
# Partially parsed test_sorted_imports_with_straight_imports. Retrieved 18/20 statements.
# Partially parsed test_sorted_imports_with_from_imports. Retrieved 19/21 statements.
# Partially parsed test_sorted_imports_with_as_imports. Retrieved 18/20 statements.
# Partially parsed test_sorted_imports_with_comments. Retrieved 19/21 statements.
# Partially parsed test_sorted_imports_with_remove_imports. Retrieved 19/21 statements.
# Partially parsed test_sorted_imports_with_combine_straight_imports. Retrieved 19/21 statements.
# Partially parsed test_sorted_imports_with_force_sort_within_sections. Retrieved 19/21 statements.
# Partially parsed test_sorted_imports_with_import_headings. Retrieved 19/21 statements.
# Partially parsed test_sorted_imports_with_lines_between_sections. Retrieved 22/24 statements.
# Partially parsed test_sorted_imports_with_lines_after_imports. Retrieved 18/20 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = -1
    var_3 = '\n'
    var_4 = {}
    var_5 = {}
    var_6 = {}
    var_7 = []
    var_8 = {}
    var_9 = 1
    var_10 = []
    var_11 = {}
    var_12 = module_0.Config(**var_11)

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = '\n'
    var_3 = 'THIRDPARTY'
    var_4 = 'straight'
    var_5 = 'numpy'
    var_6 = 'pandas'
    var_7 = set()
    var_8 = set()
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_4: var_9}
    var_11 = {var_3: var_10}
    var_12 = {}
    var_13 = {}
    var_14 = {var_4: var_13}
    var_15 = [var_3]
    var_16 = {}
    var_17 = []
    var_18 = {}
    var_19 = module_0.Config(**var_18)

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = '\n'
    var_3 = 'THIRDPARTY'
    var_4 = 'from'
    var_5 = 'numpy'
    var_6 = 'array'
    var_7 = set()
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = {var_4: var_9}
    var_11 = {var_3: var_10}
    var_12 = {}
    var_13 = 'straight'
    var_14 = {}
    var_15 = {var_13: var_14}
    var_16 = [var_3]
    var_17 = {}
    var_18 = []
    var_19 = {}
    var_20 = module_0.Config(**var_19)

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = '\n'
    var_3 = 'THIRDPARTY'
    var_4 = 'straight'
    var_5 = 'numpy'
    var_6 = set()
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = {var_3: var_8}
    var_10 = {}
    var_11 = 'np'
    var_12 = [var_11]
    var_13 = {var_5: var_12}
    var_14 = {var_4: var_13}
    var_15 = [var_3]
    var_16 = {}
    var_17 = []
    var_18 = {}
    var_19 = module_0.Config(**var_18)

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = '\n'
    var_3 = 'THIRDPARTY'
    var_4 = 'straight'
    var_5 = 'numpy'
    var_6 = set()
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = {var_3: var_8}
    var_10 = '# comment'
    var_11 = [var_10]
    var_12 = {var_5: var_11}
    var_13 = {var_4: var_12}
    var_14 = {}
    var_15 = {var_4: var_14}
    var_16 = [var_3]
    var_17 = {}
    var_18 = []
    var_19 = {}
    var_20 = module_0.Config(**var_19)

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = '\n'
    var_3 = 'THIRDPARTY'
    var_4 = 'straight'
    var_5 = 'numpy'
    var_6 = 'pandas'
    var_7 = set()
    var_8 = set()
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_4: var_9}
    var_11 = {var_3: var_10}
    var_12 = {}
    var_13 = {}
    var_14 = {var_4: var_13}
    var_15 = [var_3]
    var_16 = {}
    var_17 = []
    var_18 = [var_5]
    var_19 = 'remove_imports'
    var_20 = {var_19: var_18}
    var_21 = module_0.Config(**var_20)

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = '\n'
    var_3 = 'THIRDPARTY'
    var_4 = 'straight'
    var_5 = 'numpy'
    var_6 = 'pandas'
    var_7 = set()
    var_8 = set()
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_4: var_9}
    var_11 = {var_3: var_10}
    var_12 = {}
    var_13 = {}
    var_14 = {var_4: var_13}
    var_15 = [var_3]
    var_16 = {}
    var_17 = []
    var_18 = True
    var_19 = 'combine_straight_imports'
    var_20 = {var_19: var_18}
    var_21 = module_0.Config(**var_20)

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = '\n'
    var_3 = 'THIRDPARTY'
    var_4 = 'straight'
    var_5 = 'pandas'
    var_6 = 'numpy'
    var_7 = set()
    var_8 = set()
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_4: var_9}
    var_11 = {var_3: var_10}
    var_12 = {}
    var_13 = {}
    var_14 = {var_4: var_13}
    var_15 = [var_3]
    var_16 = {}
    var_17 = []
    var_18 = True
    var_19 = 'force_sort_within_sections'
    var_20 = {var_19: var_18}
    var_21 = module_0.Config(**var_20)

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = '\n'
    var_3 = 'THIRDPARTY'
    var_4 = 'straight'
    var_5 = 'numpy'
    var_6 = set()
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = {var_3: var_8}
    var_10 = {}
    var_11 = {}
    var_12 = {var_4: var_11}
    var_13 = [var_3]
    var_14 = {}
    var_15 = []
    var_16 = 'thirdparty'
    var_17 = 'Third Party Imports'
    var_18 = {var_16: var_17}
    var_19 = 'import_headings'
    var_20 = {var_19: var_18}
    var_21 = module_0.Config(**var_20)

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = '\n'
    var_3 = 'FUTURE'
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = '__future__'
    var_7 = set()
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'numpy'
    var_11 = set()
    var_12 = {var_10: var_11}
    var_13 = {var_5: var_12}
    var_14 = {var_3: var_9, var_4: var_13}
    var_15 = {}
    var_16 = {}
    var_17 = {var_5: var_16}
    var_18 = [var_3, var_4]
    var_19 = {}
    var_20 = []
    var_21 = 1
    var_22 = 'lines_between_sections'
    var_23 = {var_22: var_21}
    var_24 = module_0.Config(**var_23)

import isort.settings as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'numpy'
    var_7 = set()
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = {var_4: var_9}
    var_11 = {}
    var_12 = {}
    var_13 = {var_5: var_12}
    var_14 = [var_4]
    var_15 = {}
    var_16 = 1
    var_17 = []
    var_18 = 'lines_after_imports'
    var_19 = {var_18: var_16}
    var_20 = module_0.Config(**var_19)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_with_from_imports_basic. Retrieved 24/26 statements.
# Partially parsed test_with_from_imports_with_comments. Retrieved 27/29 statements.
# Partially parsed test_with_from_imports_remove_imports. Retrieved 25/27 statements.
# Partially parsed test_with_from_imports_with_as_imports. Retrieved 30/32 statements.
# Partially parsed test_with_from_imports_with_star. Retrieved 27/29 statements.
# Partially parsed test_with_from_imports_with_nested_comments. Retrieved 29/31 statements.
# Partially parsed test_with_from_imports_with_force_single_line. Retrieved 25/27 statements.
# Partially parsed test_with_from_imports_with_combine_as_imports. Retrieved 31/33 statements.
# Partially parsed test_with_from_imports_with_ignore_comments. Retrieved 28/30 statements.
# Partially parsed test_with_from_imports_with_comment_prefix. Retrieved 28/30 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = {}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = {}
    var_15 = {var_1: var_11, var_9: var_13, var_10: var_14}
    var_16 = {}
    var_17 = {var_1: var_16}
    var_18 = '\n'
    var_19 = set()
    var_20 = []
    var_21 = {}
    var_22 = module_0.Config(**var_21)
    var_23 = [var_2]
    var_24 = []
    var_25 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = 'comment1'
    var_12 = 'comment2'
    var_13 = [var_11, var_12]
    var_14 = {var_2: var_13}
    var_15 = {}
    var_16 = {var_1: var_15}
    var_17 = {}
    var_18 = {var_1: var_14, var_9: var_16, var_10: var_17}
    var_19 = {}
    var_20 = {var_1: var_19}
    var_21 = '\n'
    var_22 = set()
    var_23 = []
    var_24 = {}
    var_25 = module_0.Config(**var_24)
    var_26 = [var_2]
    var_27 = []
    var_28 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = {}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = {}
    var_15 = {var_1: var_11, var_9: var_13, var_10: var_14}
    var_16 = {}
    var_17 = {var_1: var_16}
    var_18 = '\n'
    var_19 = set()
    var_20 = []
    var_21 = {}
    var_22 = module_0.Config(**var_21)
    var_23 = [var_2]
    var_24 = 'module.import1'
    var_25 = [var_24]
    var_26 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = {}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = {}
    var_15 = {var_1: var_11, var_9: var_13, var_10: var_14}
    var_16 = 'module.import1'
    var_17 = 'module.import2'
    var_18 = 'alias1'
    var_19 = [var_18]
    var_20 = 'alias2'
    var_21 = [var_20]
    var_22 = {var_16: var_19, var_17: var_21}
    var_23 = {var_1: var_22}
    var_24 = '\n'
    var_25 = set()
    var_26 = []
    var_27 = {}
    var_28 = module_0.Config(**var_27)
    var_29 = [var_2]
    var_30 = []
    var_31 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = '*'
    var_4 = 'import1'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = {}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = 'star_comment'
    var_15 = [var_14]
    var_16 = {var_3: var_15}
    var_17 = {var_2: var_16}
    var_18 = {var_1: var_11, var_9: var_13, var_10: var_17}
    var_19 = {}
    var_20 = {var_1: var_19}
    var_21 = '\n'
    var_22 = set()
    var_23 = []
    var_24 = {}
    var_25 = module_0.Config(**var_24)
    var_26 = [var_2]
    var_27 = []
    var_28 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = {}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = 'nested_comment1'
    var_15 = [var_14]
    var_16 = 'nested_comment2'
    var_17 = [var_16]
    var_18 = {var_3: var_15, var_4: var_17}
    var_19 = {var_2: var_18}
    var_20 = {var_1: var_11, var_9: var_13, var_10: var_19}
    var_21 = {}
    var_22 = {var_1: var_21}
    var_23 = '\n'
    var_24 = set()
    var_25 = []
    var_26 = {}
    var_27 = module_0.Config(**var_26)
    var_28 = [var_2]
    var_29 = []
    var_30 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = {}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = {}
    var_15 = {var_1: var_11, var_9: var_13, var_10: var_14}
    var_16 = {}
    var_17 = {var_1: var_16}
    var_18 = '\n'
    var_19 = set()
    var_20 = []
    var_21 = True
    var_22 = 'force_single_line'
    var_23 = {var_22: var_21}
    var_24 = module_0.Config(**var_23)
    var_25 = [var_2]
    var_26 = []
    var_27 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = {}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = {}
    var_15 = {var_1: var_11, var_9: var_13, var_10: var_14}
    var_16 = 'module.import1'
    var_17 = 'module.import2'
    var_18 = 'alias1'
    var_19 = [var_18]
    var_20 = 'alias2'
    var_21 = [var_20]
    var_22 = {var_16: var_19, var_17: var_21}
    var_23 = {var_1: var_22}
    var_24 = '\n'
    var_25 = set()
    var_26 = []
    var_27 = True
    var_28 = 'combine_as_imports'
    var_29 = {var_28: var_27}
    var_30 = module_0.Config(**var_29)
    var_31 = [var_2]
    var_32 = []
    var_33 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = 'comment1'
    var_12 = 'comment2'
    var_13 = [var_11, var_12]
    var_14 = {var_2: var_13}
    var_15 = {}
    var_16 = {var_1: var_15}
    var_17 = {}
    var_18 = {var_1: var_14, var_9: var_16, var_10: var_17}
    var_19 = {}
    var_20 = {var_1: var_19}
    var_21 = '\n'
    var_22 = set()
    var_23 = []
    var_24 = True
    var_25 = 'ignore_comments'
    var_26 = {var_25: var_24}
    var_27 = module_0.Config(**var_26)
    var_28 = [var_2]
    var_29 = []
    var_30 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = 'comment1'
    var_12 = 'comment2'
    var_13 = [var_11, var_12]
    var_14 = {var_2: var_13}
    var_15 = {}
    var_16 = {var_1: var_15}
    var_17 = {}
    var_18 = {var_1: var_14, var_9: var_16, var_10: var_17}
    var_19 = {}
    var_20 = {var_1: var_19}
    var_21 = '\n'
    var_22 = set()
    var_23 = []
    var_24 = '--'
    var_25 = 'comment_prefix'
    var_26 = {var_25: var_24}
    var_27 = module_0.Config(**var_26)
    var_28 = [var_2]
    var_29 = []
    var_30 = 'import'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 5/7 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = []
    var_4 = ''
    var_5 = []
    var_6 = ''



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 7/10 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = 'straight'
    var_2 = []
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = []
    var_6 = 'test'
    var_7 = []
    var_8 = 'import'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_sorted_imports_no_imports. Retrieved 12/14 statements.
# Partially parsed test_sorted_imports_single_import. Retrieved 25/27 statements.
# Partially parsed test_sorted_imports_multiple_imports. Retrieved 27/29 statements.
# Partially parsed test_sorted_imports_with_comments. Retrieved 30/32 statements.
# Partially parsed test_sorted_imports_combine_straight_imports. Retrieved 28/30 statements.
# Partially parsed test_sorted_imports_with_as_imports. Retrieved 27/29 statements.
# Partially parsed test_sorted_imports_with_from_imports. Retrieved 28/30 statements.
# Partially parsed test_sorted_imports_with_sections. Retrieved 35/37 statements.
# Partially parsed test_sorted_imports_with_force_sort_within_sections. Retrieved 28/30 statements.
# Partially parsed test_sorted_imports_with_import_headings. Retrieved 28/30 statements.
# Partially parsed test_sorted_imports_with_lines_between_sections. Retrieved 36/38 statements.
# Partially parsed test_sorted_imports_with_remove_imports. Retrieved 28/30 statements.
# Partially parsed test_sorted_imports_with_only_sections. Retrieved 34/35 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = '\n'
    var_3 = -1
    var_4 = {}
    var_5 = {}
    var_6 = {}
    var_7 = []
    var_8 = {}
    var_9 = {}
    var_10 = 1
    var_11 = []
    var_12 = {}
    var_13 = module_0.Config(**var_12)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = '\n'
    var_3 = 0
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = {}
    var_19 = {var_5: var_18}
    var_20 = [var_4]
    var_21 = {}
    var_22 = {}
    var_23 = 1
    var_24 = []
    var_25 = {}
    var_26 = module_0.Config(**var_25)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = '\n'
    var_3 = 0
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = 'sys'
    var_9 = set()
    var_10 = set()
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = {var_5: var_11, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = 'above'
    var_16 = {}
    var_17 = {var_5: var_16}
    var_18 = {}
    var_19 = {var_15: var_17, var_5: var_18}
    var_20 = {}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []
    var_27 = {}
    var_28 = module_0.Config(**var_27)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = '\n'
    var_3 = 0
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = '# OS module'
    var_15 = [var_14]
    var_16 = {var_7: var_15}
    var_17 = {var_5: var_16}
    var_18 = '# For path operations'
    var_19 = [var_18]
    var_20 = {var_7: var_19}
    var_21 = {var_13: var_17, var_5: var_20}
    var_22 = {}
    var_23 = {var_5: var_22}
    var_24 = [var_4]
    var_25 = {}
    var_26 = {}
    var_27 = 1
    var_28 = []
    var_29 = False
    var_30 = 'ignore_comments'
    var_31 = {var_30: var_29}
    var_32 = module_0.Config(**var_31)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = '\n'
    var_3 = 0
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = 'sys'
    var_9 = set()
    var_10 = set()
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = {var_5: var_11, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = 'above'
    var_16 = {}
    var_17 = {var_5: var_16}
    var_18 = {}
    var_19 = {var_15: var_17, var_5: var_18}
    var_20 = {}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []
    var_27 = True
    var_28 = 'combine_straight_imports'
    var_29 = {var_28: var_27}
    var_30 = module_0.Config(**var_29)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = '\n'
    var_3 = 0
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = 'path'
    var_19 = [var_18]
    var_20 = {var_7: var_19}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []
    var_27 = {}
    var_28 = module_0.Config(**var_27)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = '\n'
    var_3 = 0
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = {}
    var_8 = 'os'
    var_9 = 'path'
    var_10 = set()
    var_11 = {var_9: var_10}
    var_12 = {var_8: var_11}
    var_13 = {var_5: var_7, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = 'above'
    var_16 = {}
    var_17 = {var_6: var_16}
    var_18 = {}
    var_19 = {var_8: var_18}
    var_20 = {var_15: var_17, var_6: var_19}
    var_21 = {}
    var_22 = {var_6: var_21}
    var_23 = [var_4]
    var_24 = {}
    var_25 = {}
    var_26 = 1
    var_27 = []
    var_28 = {}
    var_29 = module_0.Config(**var_28)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = '\n'
    var_3 = 0
    var_4 = 'FUTURE'
    var_5 = 'STDLIB'
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = '__future__'
    var_9 = 'annotations'
    var_10 = {var_9}
    var_11 = {var_8: var_10}
    var_12 = {}
    var_13 = {var_6: var_11, var_7: var_12}
    var_14 = 'os'
    var_15 = set()
    var_16 = {var_14: var_15}
    var_17 = {}
    var_18 = {var_6: var_16, var_7: var_17}
    var_19 = {var_4: var_13, var_5: var_18}
    var_20 = 'above'
    var_21 = {}
    var_22 = {}
    var_23 = {var_6: var_21, var_7: var_22}
    var_24 = {}
    var_25 = {}
    var_26 = {var_20: var_23, var_6: var_24, var_7: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = {var_6: var_27, var_7: var_28}
    var_30 = [var_4, var_5]
    var_31 = {}
    var_32 = {}
    var_33 = 1
    var_34 = []
    var_35 = {}
    var_36 = module_0.Config(**var_35)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = '\n'
    var_3 = 0
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'sys'
    var_8 = 'os'
    var_9 = set()
    var_10 = set()
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = {var_5: var_11, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = 'above'
    var_16 = {}
    var_17 = {var_5: var_16}
    var_18 = {}
    var_19 = {var_15: var_17, var_5: var_18}
    var_20 = {}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []
    var_27 = True
    var_28 = 'force_sort_within_sections'
    var_29 = {var_28: var_27}
    var_30 = module_0.Config(**var_29)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = '\n'
    var_3 = 0
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = {}
    var_19 = {var_5: var_18}
    var_20 = [var_4]
    var_21 = {}
    var_22 = {}
    var_23 = 1
    var_24 = []
    var_25 = 'stdlib'
    var_26 = 'Standard Library'
    var_27 = {var_25: var_26}
    var_28 = 'import_headings'
    var_29 = {var_28: var_27}
    var_30 = module_0.Config(**var_29)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = '\n'
    var_3 = 0
    var_4 = 'FUTURE'
    var_5 = 'STDLIB'
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = '__future__'
    var_9 = 'annotations'
    var_10 = {var_9}
    var_11 = {var_8: var_10}
    var_12 = {}
    var_13 = {var_6: var_11, var_7: var_12}
    var_14 = 'os'
    var_15 = set()
    var_16 = {var_14: var_15}
    var_17 = {}
    var_18 = {var_6: var_16, var_7: var_17}
    var_19 = {var_4: var_13, var_5: var_18}
    var_20 = 'above'
    var_21 = {}
    var_22 = {}
    var_23 = {var_6: var_21, var_7: var_22}
    var_24 = {}
    var_25 = {}
    var_26 = {var_20: var_23, var_6: var_24, var_7: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = {var_6: var_27, var_7: var_28}
    var_30 = [var_4, var_5]
    var_31 = {}
    var_32 = {}
    var_33 = 1
    var_34 = []
    var_35 = 2
    var_36 = 'lines_between_sections'
    var_37 = {var_36: var_35}
    var_38 = module_0.Config(**var_37)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = '\n'
    var_3 = 0
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = 'sys'
    var_9 = set()
    var_10 = set()
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = {var_5: var_11, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = 'above'
    var_16 = {}
    var_17 = {var_5: var_16}
    var_18 = {}
    var_19 = {var_15: var_17, var_5: var_18}
    var_20 = {}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []
    var_27 = [var_7]
    var_28 = 'remove_imports'
    var_29 = {var_28: var_27}
    var_30 = module_0.Config(**var_29)

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = '\n'
    var_3 = 0
    var_4 = 'FUTURE'
    var_5 = 'STDLIB'
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = '__future__'
    var_9 = 'annotations'
    var_10 = {var_9}
    var_11 = {var_8: var_10}
    var_12 = {}
    var_13 = {var_6: var_11, var_7: var_12}
    var_14 = 'os'
    var_15 = set()
    var_16 = {var_14: var_15}
    var_17 = {}
    var_18 = {var_6: var_16, var_7: var_17}
    var_19 = {var_4: var_13, var_5: var_18}
    var_20 = 'above'
    var_21 = {}
    var_22 = {}
    var_23 = {var_6: var_21, var_7: var_22}
    var_24 = {}
    var_25 = {}
    var_26 = {var_20: var_23, var_6: var_24, var_7: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = {var_6: var_27, var_7: var_28}
    var_30 = [var_4, var_5]
    var_31 = {}
    var_32 = {}
    var_33 = 1
    var_34 = []



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_sorted_imports_predicate_false. Retrieved 4/6 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 0
    var_1 = []
    var_2 = '\n'
    var_3 = []
    var_4 = {}
    var_5 = module_0.Config(**var_4)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_with_from_imports_basic. Retrieved 24/26 statements.
# Partially parsed test_with_from_imports_with_comments. Retrieved 27/29 statements.
# Partially parsed test_with_from_imports_remove_imports. Retrieved 25/27 statements.
# Partially parsed test_with_from_imports_with_as_imports. Retrieved 31/33 statements.
# Partially parsed test_with_from_imports_with_star_comment. Retrieved 28/30 statements.
# Partially parsed test_with_from_imports_force_single_line. Retrieved 25/27 statements.
# Partially parsed test_with_from_imports_ignore_comments. Retrieved 28/30 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = {}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = {}
    var_15 = {var_1: var_11, var_9: var_13, var_10: var_14}
    var_16 = {}
    var_17 = {var_1: var_16}
    var_18 = '\n'
    var_19 = set()
    var_20 = []
    var_21 = {}
    var_22 = module_0.Config(**var_21)
    var_23 = [var_2]
    var_24 = []
    var_25 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = 'comment1'
    var_12 = 'comment2'
    var_13 = [var_11, var_12]
    var_14 = {var_2: var_13}
    var_15 = {}
    var_16 = {var_1: var_15}
    var_17 = {}
    var_18 = {var_1: var_14, var_9: var_16, var_10: var_17}
    var_19 = {}
    var_20 = {var_1: var_19}
    var_21 = '\n'
    var_22 = set()
    var_23 = []
    var_24 = {}
    var_25 = module_0.Config(**var_24)
    var_26 = [var_2]
    var_27 = []
    var_28 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = {}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = {}
    var_15 = {var_1: var_11, var_9: var_13, var_10: var_14}
    var_16 = {}
    var_17 = {var_1: var_16}
    var_18 = '\n'
    var_19 = set()
    var_20 = []
    var_21 = {}
    var_22 = module_0.Config(**var_21)
    var_23 = [var_2]
    var_24 = 'module.import1'
    var_25 = [var_24]
    var_26 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = {}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = {}
    var_15 = {var_1: var_11, var_9: var_13, var_10: var_14}
    var_16 = 'module.import1'
    var_17 = 'module.import2'
    var_18 = 'alias1'
    var_19 = [var_18]
    var_20 = 'alias2'
    var_21 = [var_20]
    var_22 = {var_16: var_19, var_17: var_21}
    var_23 = {var_1: var_22}
    var_24 = '\n'
    var_25 = set()
    var_26 = []
    var_27 = True
    var_28 = 'combine_as_imports'
    var_29 = {var_28: var_27}
    var_30 = module_0.Config(**var_29)
    var_31 = [var_2]
    var_32 = []
    var_33 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = '*'
    var_4 = 'import1'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = {}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = 'star comment'
    var_15 = [var_14]
    var_16 = {var_3: var_15}
    var_17 = {var_2: var_16}
    var_18 = {var_1: var_11, var_9: var_13, var_10: var_17}
    var_19 = {}
    var_20 = {var_1: var_19}
    var_21 = '\n'
    var_22 = set()
    var_23 = []
    var_24 = True
    var_25 = 'combine_star'
    var_26 = {var_25: var_24}
    var_27 = module_0.Config(**var_26)
    var_28 = [var_2]
    var_29 = []
    var_30 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = {}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = {}
    var_15 = {var_1: var_11, var_9: var_13, var_10: var_14}
    var_16 = {}
    var_17 = {var_1: var_16}
    var_18 = '\n'
    var_19 = set()
    var_20 = []
    var_21 = True
    var_22 = 'force_single_line'
    var_23 = {var_22: var_21}
    var_24 = module_0.Config(**var_23)
    var_25 = [var_2]
    var_26 = []
    var_27 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = 'comment1'
    var_12 = 'comment2'
    var_13 = [var_11, var_12]
    var_14 = {var_2: var_13}
    var_15 = {}
    var_16 = {var_1: var_15}
    var_17 = {}
    var_18 = {var_1: var_14, var_9: var_16, var_10: var_17}
    var_19 = {}
    var_20 = {var_1: var_19}
    var_21 = '\n'
    var_22 = set()
    var_23 = []
    var_24 = True
    var_25 = 'ignore_comments'
    var_26 = {var_25: var_24}
    var_27 = module_0.Config(**var_26)
    var_28 = [var_2]
    var_29 = []
    var_30 = 'import'



# Parsed testcases at query #32
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_star_comment_is_none. Retrieved 8/11 statements.


def test_case_0():
    var_0 = []
    var_1 = 'nested'
    var_2 = 'module'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'module'
    var_6 = 'comment1'
    var_7 = 'comment2'
    var_8 = [var_6, var_7]



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_as_imports_predicate. Retrieved 5/10 statements.


def test_case_0():
    var_0 = []
    var_1 = 'straight'
    var_2 = 'module1'
    var_3 = 'module2'
    var_4 = [var_2, var_3]
    var_5 = [var_2]



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_with_from_imports_basic. Retrieved 22/24 statements.
# Partially parsed test_with_from_imports_remove_imports. Retrieved 21/23 statements.
# Partially parsed test_with_from_imports_star_import. Retrieved 21/23 statements.
# Partially parsed test_with_from_imports_force_single_line. Retrieved 21/23 statements.
# Partially parsed test_with_from_imports_combine_as_imports. Retrieved 23/25 statements.
# Partially parsed test_with_from_imports_ignore_comments. Retrieved 20/22 statements.
# Partially parsed test_with_from_imports_no_inline_sort. Retrieved 21/23 statements.
# Partially parsed test_with_from_imports_only_sections. Retrieved 20/22 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = '# comment'
    var_9 = (var_8,)
    var_10 = {var_2: var_9}
    var_11 = {var_1: var_10}
    var_12 = 'os.path'
    var_13 = 'os.path as osp'
    var_14 = [var_13]
    var_15 = {var_12: var_14}
    var_16 = {var_1: var_15}
    var_17 = '\n'
    var_18 = []
    var_19 = {}
    var_20 = module_0.Config(**var_19)
    var_21 = [var_2]
    var_22 = []
    var_23 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'sys'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = '# comment'
    var_10 = (var_9,)
    var_11 = {var_2: var_10}
    var_12 = {var_1: var_11}
    var_13 = {}
    var_14 = {var_1: var_13}
    var_15 = '\n'
    var_16 = []
    var_17 = {}
    var_18 = module_0.Config(**var_17)
    var_19 = [var_2]
    var_20 = 'os.sys'
    var_21 = [var_20]
    var_22 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = '*'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'nested'
    var_9 = '# star comment'
    var_10 = {var_3: var_9}
    var_11 = {var_2: var_10}
    var_12 = {var_8: var_11}
    var_13 = {}
    var_14 = {var_1: var_13}
    var_15 = '\n'
    var_16 = []
    var_17 = True
    var_18 = 'combine_star'
    var_19 = {var_18: var_17}
    var_20 = module_0.Config(**var_19)
    var_21 = [var_2]
    var_22 = []
    var_23 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'sys'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = '# comment'
    var_10 = (var_9,)
    var_11 = {var_2: var_10}
    var_12 = {var_1: var_11}
    var_13 = {}
    var_14 = {var_1: var_13}
    var_15 = '\n'
    var_16 = []
    var_17 = True
    var_18 = 'force_single_line'
    var_19 = {var_18: var_17}
    var_20 = module_0.Config(**var_19)
    var_21 = [var_2]
    var_22 = []
    var_23 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = '# comment'
    var_9 = (var_8,)
    var_10 = {var_2: var_9}
    var_11 = {var_1: var_10}
    var_12 = 'os.path'
    var_13 = 'os.path as osp'
    var_14 = [var_13]
    var_15 = {var_12: var_14}
    var_16 = {var_1: var_15}
    var_17 = '\n'
    var_18 = []
    var_19 = True
    var_20 = 'combine_as_imports'
    var_21 = {var_20: var_19}
    var_22 = module_0.Config(**var_21)
    var_23 = [var_2]
    var_24 = []
    var_25 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = '# comment'
    var_9 = (var_8,)
    var_10 = {var_2: var_9}
    var_11 = {var_1: var_10}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = '\n'
    var_15 = []
    var_16 = True
    var_17 = 'ignore_comments'
    var_18 = {var_17: var_16}
    var_19 = module_0.Config(**var_18)
    var_20 = [var_2]
    var_21 = []
    var_22 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'sys'
    var_4 = 'path'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = '# comment'
    var_10 = (var_9,)
    var_11 = {var_2: var_10}
    var_12 = {var_1: var_11}
    var_13 = {}
    var_14 = {var_1: var_13}
    var_15 = '\n'
    var_16 = []
    var_17 = True
    var_18 = 'no_inline_sort'
    var_19 = {var_18: var_17}
    var_20 = module_0.Config(**var_19)
    var_21 = [var_2]
    var_22 = []
    var_23 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = '# comment'
    var_9 = (var_8,)
    var_10 = {var_2: var_9}
    var_11 = {var_1: var_10}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = '\n'
    var_15 = []
    var_16 = True
    var_17 = 'only_sections'
    var_18 = {var_17: var_16}
    var_19 = module_0.Config(**var_18)
    var_20 = [var_2]
    var_21 = []
    var_22 = 'import'



# Parsed testcases at query #36
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_with_from_imports_basic_case. Retrieved 19/21 statements.
# Partially parsed test_with_from_imports_with_comments. Retrieved 22/24 statements.
# Partially parsed test_with_from_imports_with_star. Retrieved 23/25 statements.
# Partially parsed test_with_from_imports_with_as_imports. Retrieved 23/25 statements.
# Partially parsed test_with_from_imports_remove_imports. Retrieved 21/23 statements.
# Partially parsed test_with_from_imports_force_single_line. Retrieved 21/23 statements.
# Partially parsed test_with_from_imports_combine_star. Retrieved 25/27 statements.
# Partially parsed test_with_from_imports_no_inline_sort. Retrieved 21/23 statements.
# Partially parsed test_with_from_imports_only_sections. Retrieved 20/22 statements.
# Partially parsed test_with_from_imports_split_on_trailing_comma. Retrieved 22/24 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = ()
    var_9 = {var_2: var_8}
    var_10 = {var_1: var_9}
    var_11 = {}
    var_12 = {var_1: var_11}
    var_13 = '\n'
    var_14 = set()
    var_15 = []
    var_16 = {}
    var_17 = module_0.Config(**var_16)
    var_18 = [var_2]
    var_19 = []
    var_20 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'comment'
    var_9 = (var_8,)
    var_10 = {var_2: var_9}
    var_11 = {var_1: var_10}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = '\n'
    var_15 = set()
    var_16 = []
    var_17 = False
    var_18 = '# '
    var_19 = 'ignore_comments'
    var_20 = 'comment_prefix'
    var_21 = {var_19: var_17, var_20: var_18}
    var_22 = module_0.Config(**var_21)
    var_23 = [var_2]
    var_24 = []
    var_25 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = '*'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'nested'
    var_9 = 'star comment'
    var_10 = {var_3: var_9}
    var_11 = {var_2: var_10}
    var_12 = {var_8: var_11}
    var_13 = {}
    var_14 = {var_1: var_13}
    var_15 = '\n'
    var_16 = set()
    var_17 = []
    var_18 = False
    var_19 = '# '
    var_20 = 'ignore_comments'
    var_21 = 'comment_prefix'
    var_22 = {var_20: var_18, var_21: var_19}
    var_23 = module_0.Config(**var_22)
    var_24 = [var_2]
    var_25 = []
    var_26 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = ()
    var_9 = {var_2: var_8}
    var_10 = {var_1: var_9}
    var_11 = 'os.path'
    var_12 = 'path as ospath'
    var_13 = [var_12]
    var_14 = {var_11: var_13}
    var_15 = {var_1: var_14}
    var_16 = '\n'
    var_17 = set()
    var_18 = []
    var_19 = True
    var_20 = 'combine_as_imports'
    var_21 = {var_20: var_19}
    var_22 = module_0.Config(**var_21)
    var_23 = [var_2]
    var_24 = []
    var_25 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'sys'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = ()
    var_10 = {var_2: var_9}
    var_11 = {var_1: var_10}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = '\n'
    var_15 = set()
    var_16 = []
    var_17 = {}
    var_18 = module_0.Config(**var_17)
    var_19 = [var_2]
    var_20 = 'os.sys'
    var_21 = [var_20]
    var_22 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'sys'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = ()
    var_10 = {var_2: var_9}
    var_11 = {var_1: var_10}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = '\n'
    var_15 = set()
    var_16 = []
    var_17 = True
    var_18 = 'force_single_line'
    var_19 = {var_18: var_17}
    var_20 = module_0.Config(**var_19)
    var_21 = [var_2]
    var_22 = []
    var_23 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = '*'
    var_4 = 'path'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'nested'
    var_10 = 'star comment'
    var_11 = {var_3: var_10}
    var_12 = {var_2: var_11}
    var_13 = {var_9: var_12}
    var_14 = {}
    var_15 = {var_1: var_14}
    var_16 = '\n'
    var_17 = set()
    var_18 = []
    var_19 = True
    var_20 = False
    var_21 = '# '
    var_22 = 'combine_star'
    var_23 = 'ignore_comments'
    var_24 = 'comment_prefix'
    var_25 = {var_22: var_19, var_23: var_20, var_24: var_21}
    var_26 = module_0.Config(**var_25)
    var_27 = [var_2]
    var_28 = []
    var_29 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'sys'
    var_4 = 'path'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = ()
    var_10 = {var_2: var_9}
    var_11 = {var_1: var_10}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = '\n'
    var_15 = set()
    var_16 = []
    var_17 = True
    var_18 = 'no_inline_sort'
    var_19 = {var_18: var_17}
    var_20 = module_0.Config(**var_19)
    var_21 = [var_2]
    var_22 = []
    var_23 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = ()
    var_9 = {var_2: var_8}
    var_10 = {var_1: var_9}
    var_11 = {}
    var_12 = {var_1: var_11}
    var_13 = '\n'
    var_14 = set()
    var_15 = []
    var_16 = True
    var_17 = 'only_sections'
    var_18 = {var_17: var_16}
    var_19 = module_0.Config(**var_18)
    var_20 = [var_2]
    var_21 = []
    var_22 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'sys'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = ()
    var_10 = {var_2: var_9}
    var_11 = {var_1: var_10}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = '\n'
    var_15 = {var_2}
    var_16 = []
    var_17 = True
    var_18 = 30
    var_19 = 'split_on_trailing_comma'
    var_20 = 'line_length'
    var_21 = {var_19: var_17, var_20: var_18}
    var_22 = module_0.Config(**var_21)
    var_23 = [var_2]
    var_24 = []
    var_25 = 'import'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_sorted_imports_with_no_imports. Retrieved 9/11 statements.


def test_case_0():
    var_0 = -1
    var_1 = "print('hello')"
    var_2 = [var_1]
    var_3 = '\n'
    var_4 = {}
    var_5 = []
    var_6 = {}
    var_7 = {}
    var_8 = 1
    var_9 = []



# Parsed testcases at query #39
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_sorted_imports_with_no_imports. Retrieved 10/12 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = -1
    var_3 = '\n'
    var_4 = {}
    var_5 = []
    var_6 = {}
    var_7 = {}
    var_8 = 1
    var_9 = []
    var_10 = {}
    var_11 = module_0.Config(**var_10)



# Parsed testcases at query #41
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_with_straight_imports_predicate. Retrieved 28/30 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'straight'
    var_1 = 'module1'
    var_2 = 'module2'
    var_3 = 'alias1'
    var_4 = [var_3]
    var_5 = 'alias2'
    var_6 = [var_5]
    var_7 = {var_1: var_4, var_2: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'section'
    var_10 = 'import1'
    var_11 = [var_10]
    var_12 = 'import2'
    var_13 = [var_12]
    var_14 = {var_1: var_11, var_2: var_13}
    var_15 = {var_0: var_14}
    var_16 = {var_9: var_15}
    var_17 = 'above'
    var_18 = {}
    var_19 = {var_0: var_18}
    var_20 = {}
    var_21 = {var_17: var_19, var_0: var_20}
    var_22 = []
    var_23 = True
    var_24 = 'combine_straight_imports'
    var_25 = {var_24: var_23}
    var_26 = module_0.Config(**var_25)
    var_27 = [var_1, var_2]
    var_28 = 'section'
    var_29 = []
    var_30 = 'import'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_sorted_imports_with_no_imports. Retrieved 8/10 statements.


def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = '\n'
    var_5 = 1
    var_6 = {}
    var_7 = {}
    var_8 = []



# Parsed testcases at query #44
#--------------------------




def test_case_0():
    pass



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_with_from_imports_basic. Retrieved 24/26 statements.
# Partially parsed test_with_from_imports_with_comments. Retrieved 27/29 statements.
# Partially parsed test_with_from_imports_remove_imports. Retrieved 25/27 statements.
# Partially parsed test_with_from_imports_star_import. Retrieved 28/30 statements.
# Partially parsed test_with_from_imports_as_imports. Retrieved 31/33 statements.
# Partially parsed test_with_from_imports_force_single_line. Retrieved 25/27 statements.
# Partially parsed test_with_from_imports_multiline_reformat. Retrieved 25/27 statements.
# Partially parsed test_with_from_imports_split_on_trailing_comma. Retrieved 25/27 statements.
# Partially parsed test_with_from_imports_ignore_comments. Retrieved 26/28 statements.
# Partially parsed test_with_from_imports_combine_as_imports_with_star. Retrieved 28/30 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = []
    var_11 = {var_2: var_10}
    var_12 = None
    var_13 = {var_2: var_12}
    var_14 = {var_1: var_13}
    var_15 = {var_1: var_11, var_9: var_14}
    var_16 = {}
    var_17 = {var_1: var_16}
    var_18 = '\n'
    var_19 = set()
    var_20 = []
    var_21 = {}
    var_22 = module_0.Config(**var_21)
    var_23 = [var_2]
    var_24 = []
    var_25 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'comment1'
    var_11 = [var_10]
    var_12 = {var_2: var_11}
    var_13 = None
    var_14 = {var_2: var_13}
    var_15 = {var_1: var_14}
    var_16 = {var_1: var_12, var_9: var_15}
    var_17 = {}
    var_18 = {var_1: var_17}
    var_19 = '\n'
    var_20 = set()
    var_21 = []
    var_22 = False
    var_23 = '# '
    var_24 = 'ignore_comments'
    var_25 = 'comment_prefix'
    var_26 = {var_24: var_22, var_25: var_23}
    var_27 = module_0.Config(**var_26)
    var_28 = [var_2]
    var_29 = []
    var_30 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = []
    var_11 = {var_2: var_10}
    var_12 = None
    var_13 = {var_2: var_12}
    var_14 = {var_1: var_13}
    var_15 = {var_1: var_11, var_9: var_14}
    var_16 = {}
    var_17 = {var_1: var_16}
    var_18 = '\n'
    var_19 = set()
    var_20 = []
    var_21 = {}
    var_22 = module_0.Config(**var_21)
    var_23 = [var_2]
    var_24 = 'module.import1'
    var_25 = [var_24]
    var_26 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = '*'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'nested'
    var_9 = 'above'
    var_10 = []
    var_11 = {var_2: var_10}
    var_12 = 'star comment'
    var_13 = {var_3: var_12}
    var_14 = {var_2: var_13}
    var_15 = None
    var_16 = {var_2: var_15}
    var_17 = {var_1: var_16}
    var_18 = {var_1: var_11, var_8: var_14, var_9: var_17}
    var_19 = {}
    var_20 = {var_1: var_19}
    var_21 = '\n'
    var_22 = set()
    var_23 = []
    var_24 = True
    var_25 = 'combine_star'
    var_26 = {var_25: var_24}
    var_27 = module_0.Config(**var_26)
    var_28 = [var_2]
    var_29 = []
    var_30 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = []
    var_11 = {var_2: var_10}
    var_12 = None
    var_13 = {var_2: var_12}
    var_14 = {var_1: var_13}
    var_15 = {var_1: var_11, var_9: var_14}
    var_16 = 'module.import1'
    var_17 = 'module.import2'
    var_18 = 'import1 as alias1'
    var_19 = [var_18]
    var_20 = 'import2 as alias2'
    var_21 = [var_20]
    var_22 = {var_16: var_19, var_17: var_21}
    var_23 = {var_1: var_22}
    var_24 = '\n'
    var_25 = set()
    var_26 = []
    var_27 = True
    var_28 = 'combine_as_imports'
    var_29 = {var_28: var_27}
    var_30 = module_0.Config(**var_29)
    var_31 = [var_2]
    var_32 = []
    var_33 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = []
    var_11 = {var_2: var_10}
    var_12 = None
    var_13 = {var_2: var_12}
    var_14 = {var_1: var_13}
    var_15 = {var_1: var_11, var_9: var_14}
    var_16 = {}
    var_17 = {var_1: var_16}
    var_18 = '\n'
    var_19 = set()
    var_20 = []
    var_21 = True
    var_22 = 'force_single_line'
    var_23 = {var_22: var_21}
    var_24 = module_0.Config(**var_23)
    var_25 = [var_2]
    var_26 = []
    var_27 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = []
    var_11 = {var_2: var_10}
    var_12 = None
    var_13 = {var_2: var_12}
    var_14 = {var_1: var_13}
    var_15 = {var_1: var_11, var_9: var_14}
    var_16 = {}
    var_17 = {var_1: var_16}
    var_18 = '\n'
    var_19 = set()
    var_20 = []
    var_21 = 20
    var_22 = 'line_length'
    var_23 = {var_22: var_21}
    var_24 = module_0.Config(**var_23)
    var_25 = [var_2]
    var_26 = []
    var_27 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = []
    var_11 = {var_2: var_10}
    var_12 = None
    var_13 = {var_2: var_12}
    var_14 = {var_1: var_13}
    var_15 = {var_1: var_11, var_9: var_14}
    var_16 = {}
    var_17 = {var_1: var_16}
    var_18 = '\n'
    var_19 = {var_2}
    var_20 = []
    var_21 = True
    var_22 = 'split_on_trailing_comma'
    var_23 = {var_22: var_21}
    var_24 = module_0.Config(**var_23)
    var_25 = [var_2]
    var_26 = []
    var_27 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'comment1'
    var_11 = [var_10]
    var_12 = {var_2: var_11}
    var_13 = None
    var_14 = {var_2: var_13}
    var_15 = {var_1: var_14}
    var_16 = {var_1: var_12, var_9: var_15}
    var_17 = {}
    var_18 = {var_1: var_17}
    var_19 = '\n'
    var_20 = set()
    var_21 = []
    var_22 = True
    var_23 = 'ignore_comments'
    var_24 = {var_23: var_22}
    var_25 = module_0.Config(**var_24)
    var_26 = [var_2]
    var_27 = []
    var_28 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = '*'
    var_4 = 'import1'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = []
    var_11 = {var_2: var_10}
    var_12 = None
    var_13 = {var_2: var_12}
    var_14 = {var_1: var_13}
    var_15 = {var_1: var_11, var_9: var_14}
    var_16 = 'module.import1'
    var_17 = 'import1 as alias1'
    var_18 = [var_17]
    var_19 = {var_16: var_18}
    var_20 = {var_1: var_19}
    var_21 = '\n'
    var_22 = set()
    var_23 = []
    var_24 = True
    var_25 = 'combine_as_imports'
    var_26 = 'combine_star'
    var_27 = {var_25: var_24, var_26: var_24}
    var_28 = module_0.Config(**var_27)
    var_29 = [var_2]
    var_30 = []
    var_31 = 'import'



# Parsed testcases at query #2
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #3
#--------------------------




import isort.output as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._normalize_empty_lines(var_0)
    var_2 = bool(var_1 == [''])
    assert var_2 is True

import isort.output as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = module_0._normalize_empty_lines(var_1)
    var_3 = bool(var_2 == ['', ''])
    assert var_3 is True

import isort.output as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = [var_0]
    var_2 = module_0._normalize_empty_lines(var_1)
    var_3 = bool(var_2 == ['hello', ''])
    assert var_3 is True

import isort.output as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = ''
    var_2 = '  '
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._normalize_empty_lines(var_3)
    var_5 = bool(var_4 == ['hello', ''])
    assert var_5 is True

import isort.output as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = 'world'
    var_2 = [var_0, var_1]
    var_3 = module_0._normalize_empty_lines(var_2)
    var_4 = bool(var_3 == ['hello', 'world', ''])
    assert var_4 is True

import isort.output as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = ''
    var_2 = 'world'
    var_3 = '  '
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._normalize_empty_lines(var_4)
    var_6 = bool(var_5 == ['hello', '', 'world', ''])
    assert var_6 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_sorted_imports_with_empty_parsed_content. Retrieved 22/24 statements.
# Partially parsed test_sorted_imports_with_no_imports. Retrieved 22/24 statements.
# Partially parsed test_sorted_imports_with_single_straight_import. Retrieved 28/30 statements.
# Partially parsed test_sorted_imports_with_multiple_straight_imports. Retrieved 30/32 statements.
# Partially parsed test_sorted_imports_with_from_imports. Retrieved 30/32 statements.
# Partially parsed test_sorted_imports_with_as_imports. Retrieved 30/32 statements.
# Partially parsed test_sorted_imports_with_combined_straight_imports. Retrieved 31/33 statements.
# Partially parsed test_sorted_imports_with_comments. Retrieved 32/34 statements.
# Partially parsed test_sorted_imports_with_force_sort_within_sections. Retrieved 31/33 statements.
# Partially parsed test_sorted_imports_with_import_headings. Retrieved 31/33 statements.
# Partially parsed test_sorted_imports_with_lines_between_sections. Retrieved 35/37 statements.
# Partially parsed test_sorted_imports_with_ensure_newline_before_comments. Retrieved 29/31 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = -1
    var_3 = 0
    var_4 = '\n'
    var_5 = {}
    var_6 = 'above'
    var_7 = 'straight'
    var_8 = 'from'
    var_9 = {}
    var_10 = {}
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = {}
    var_14 = {var_6: var_11, var_7: var_12, var_8: var_13}
    var_15 = {}
    var_16 = {}
    var_17 = {var_7: var_15, var_8: var_16}
    var_18 = []
    var_19 = {}
    var_20 = {}
    var_21 = []
    var_22 = {}
    var_23 = module_0.Config(**var_22)

import isort.settings as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = -1
    var_3 = 1
    var_4 = '\n'
    var_5 = {}
    var_6 = 'above'
    var_7 = 'straight'
    var_8 = 'from'
    var_9 = {}
    var_10 = {}
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = {}
    var_14 = {var_6: var_11, var_7: var_12, var_8: var_13}
    var_15 = {}
    var_16 = {}
    var_17 = {var_7: var_15, var_8: var_16}
    var_18 = []
    var_19 = {}
    var_20 = {}
    var_21 = []
    var_22 = {}
    var_23 = module_0.Config(**var_22)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = 1
    var_4 = '\n'
    var_5 = 'THIRDPARTY'
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = 'os'
    var_9 = set()
    var_10 = {var_8: var_9}
    var_11 = {}
    var_12 = {var_6: var_10, var_7: var_11}
    var_13 = {var_5: var_12}
    var_14 = 'above'
    var_15 = {}
    var_16 = {}
    var_17 = {var_6: var_15, var_7: var_16}
    var_18 = {}
    var_19 = {}
    var_20 = {var_14: var_17, var_6: var_18, var_7: var_19}
    var_21 = {}
    var_22 = {}
    var_23 = {var_6: var_21, var_7: var_22}
    var_24 = [var_5]
    var_25 = {}
    var_26 = {}
    var_27 = []
    var_28 = {}
    var_29 = module_0.Config(**var_28)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = 1
    var_4 = '\n'
    var_5 = 'THIRDPARTY'
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = 'os'
    var_9 = 'sys'
    var_10 = set()
    var_11 = set()
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = {}
    var_14 = {var_6: var_12, var_7: var_13}
    var_15 = {var_5: var_14}
    var_16 = 'above'
    var_17 = {}
    var_18 = {}
    var_19 = {var_6: var_17, var_7: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = {var_16: var_19, var_6: var_20, var_7: var_21}
    var_23 = {}
    var_24 = {}
    var_25 = {var_6: var_23, var_7: var_24}
    var_26 = [var_5]
    var_27 = {}
    var_28 = {}
    var_29 = []
    var_30 = {}
    var_31 = module_0.Config(**var_30)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = 1
    var_4 = '\n'
    var_5 = 'THIRDPARTY'
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = {}
    var_9 = 'os'
    var_10 = 'path'
    var_11 = set()
    var_12 = {var_10: var_11}
    var_13 = {var_9: var_12}
    var_14 = {var_6: var_8, var_7: var_13}
    var_15 = {var_5: var_14}
    var_16 = 'above'
    var_17 = {}
    var_18 = {}
    var_19 = {var_6: var_17, var_7: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = {var_16: var_19, var_6: var_20, var_7: var_21}
    var_23 = {}
    var_24 = {}
    var_25 = {var_6: var_23, var_7: var_24}
    var_26 = [var_5]
    var_27 = {}
    var_28 = {}
    var_29 = []
    var_30 = {}
    var_31 = module_0.Config(**var_30)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = 1
    var_4 = '\n'
    var_5 = 'THIRDPARTY'
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = 'numpy'
    var_9 = set()
    var_10 = {var_8: var_9}
    var_11 = {}
    var_12 = {var_6: var_10, var_7: var_11}
    var_13 = {var_5: var_12}
    var_14 = 'above'
    var_15 = {}
    var_16 = {}
    var_17 = {var_6: var_15, var_7: var_16}
    var_18 = {}
    var_19 = {}
    var_20 = {var_14: var_17, var_6: var_18, var_7: var_19}
    var_21 = 'np'
    var_22 = [var_21]
    var_23 = {var_8: var_22}
    var_24 = {}
    var_25 = {var_6: var_23, var_7: var_24}
    var_26 = [var_5]
    var_27 = {}
    var_28 = {}
    var_29 = []
    var_30 = {}
    var_31 = module_0.Config(**var_30)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = 1
    var_4 = '\n'
    var_5 = 'THIRDPARTY'
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = 'os'
    var_9 = 'sys'
    var_10 = set()
    var_11 = set()
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = {}
    var_14 = {var_6: var_12, var_7: var_13}
    var_15 = {var_5: var_14}
    var_16 = 'above'
    var_17 = {}
    var_18 = {}
    var_19 = {var_6: var_17, var_7: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = {var_16: var_19, var_6: var_20, var_7: var_21}
    var_23 = {}
    var_24 = {}
    var_25 = {var_6: var_23, var_7: var_24}
    var_26 = [var_5]
    var_27 = {}
    var_28 = {}
    var_29 = []
    var_30 = True
    var_31 = 'combine_straight_imports'
    var_32 = {var_31: var_30}
    var_33 = module_0.Config(**var_32)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = 1
    var_4 = '\n'
    var_5 = 'THIRDPARTY'
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = 'os'
    var_9 = set()
    var_10 = {var_8: var_9}
    var_11 = {}
    var_12 = {var_6: var_10, var_7: var_11}
    var_13 = {var_5: var_12}
    var_14 = 'above'
    var_15 = '# Comment above'
    var_16 = [var_15]
    var_17 = {var_8: var_16}
    var_18 = {}
    var_19 = {var_6: var_17, var_7: var_18}
    var_20 = '# Inline comment'
    var_21 = [var_20]
    var_22 = {var_8: var_21}
    var_23 = {}
    var_24 = {var_14: var_19, var_6: var_22, var_7: var_23}
    var_25 = {}
    var_26 = {}
    var_27 = {var_6: var_25, var_7: var_26}
    var_28 = [var_5]
    var_29 = {}
    var_30 = {}
    var_31 = []
    var_32 = {}
    var_33 = module_0.Config(**var_32)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = 1
    var_4 = '\n'
    var_5 = 'THIRDPARTY'
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = 'sys'
    var_9 = 'os'
    var_10 = set()
    var_11 = set()
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = {}
    var_14 = {var_6: var_12, var_7: var_13}
    var_15 = {var_5: var_14}
    var_16 = 'above'
    var_17 = {}
    var_18 = {}
    var_19 = {var_6: var_17, var_7: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = {var_16: var_19, var_6: var_20, var_7: var_21}
    var_23 = {}
    var_24 = {}
    var_25 = {var_6: var_23, var_7: var_24}
    var_26 = [var_5]
    var_27 = {}
    var_28 = {}
    var_29 = []
    var_30 = True
    var_31 = 'force_sort_within_sections'
    var_32 = {var_31: var_30}
    var_33 = module_0.Config(**var_32)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = 1
    var_4 = '\n'
    var_5 = 'THIRDPARTY'
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = 'os'
    var_9 = set()
    var_10 = {var_8: var_9}
    var_11 = {}
    var_12 = {var_6: var_10, var_7: var_11}
    var_13 = {var_5: var_12}
    var_14 = 'above'
    var_15 = {}
    var_16 = {}
    var_17 = {var_6: var_15, var_7: var_16}
    var_18 = {}
    var_19 = {}
    var_20 = {var_14: var_17, var_6: var_18, var_7: var_19}
    var_21 = {}
    var_22 = {}
    var_23 = {var_6: var_21, var_7: var_22}
    var_24 = [var_5]
    var_25 = {}
    var_26 = {}
    var_27 = []
    var_28 = 'thirdparty'
    var_29 = 'Third Party Imports'
    var_30 = {var_28: var_29}
    var_31 = 'import_headings'
    var_32 = {var_31: var_30}
    var_33 = module_0.Config(**var_32)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = 1
    var_4 = '\n'
    var_5 = 'FUTURE'
    var_6 = 'THIRDPARTY'
    var_7 = 'straight'
    var_8 = 'from'
    var_9 = '__future__'
    var_10 = set()
    var_11 = {var_9: var_10}
    var_12 = {}
    var_13 = {var_7: var_11, var_8: var_12}
    var_14 = 'os'
    var_15 = set()
    var_16 = {var_14: var_15}
    var_17 = {}
    var_18 = {var_7: var_16, var_8: var_17}
    var_19 = {var_5: var_13, var_6: var_18}
    var_20 = 'above'
    var_21 = {}
    var_22 = {}
    var_23 = {var_7: var_21, var_8: var_22}
    var_24 = {}
    var_25 = {}
    var_26 = {var_20: var_23, var_7: var_24, var_8: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = {var_7: var_27, var_8: var_28}
    var_30 = [var_5, var_6]
    var_31 = {}
    var_32 = {}
    var_33 = []
    var_34 = 2
    var_35 = 'lines_between_sections'
    var_36 = {var_35: var_34}
    var_37 = module_0.Config(**var_36)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = 1
    var_4 = '\n'
    var_5 = 'THIRDPARTY'
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = 'os'
    var_9 = set()
    var_10 = {var_8: var_9}
    var_11 = {}
    var_12 = {var_6: var_10, var_7: var_11}
    var_13 = {var_5: var_12}
    var_14 = 'above'
    var_15 = {}
    var_16 = {}
    var_17 = {var_6: var_15, var_7: var_16}
    var_18 = {}
    var_19 = {}
    var_20 = {var_14: var_17, var_6: var_18, var_7: var_19}
    var_21 = {}
    var_22 = {}
    var_23 = {var_6: var_21, var_7: var_22}
    var_24 = [var_5]
    var_25 = {}
    var_26 = {}
    var_27 = []
    var_28 = True
    var_29 = 'ensure_newline_before_comments'
    var_30 = {var_29: var_28}
    var_31 = module_0.Config(**var_30)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_with_from_imports_basic. Retrieved 24/26 statements.
# Partially parsed test_with_from_imports_with_comments. Retrieved 28/30 statements.
# Partially parsed test_with_from_imports_remove_imports. Retrieved 25/27 statements.
# Partially parsed test_with_from_imports_with_as_imports. Retrieved 27/29 statements.
# Partially parsed test_with_from_imports_force_single_line. Retrieved 25/27 statements.
# Partially parsed test_with_from_imports_star_import. Retrieved 23/25 statements.
# Partially parsed test_with_from_imports_star_import_with_comment. Retrieved 26/28 statements.
# Partially parsed test_with_from_imports_combine_star. Retrieved 25/27 statements.
# Partially parsed test_with_from_imports_no_inline_sort. Retrieved 25/27 statements.
# Partially parsed test_with_from_imports_only_sections. Retrieved 25/27 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'sys'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = {}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = {}
    var_15 = {var_1: var_11, var_9: var_13, var_10: var_14}
    var_16 = {}
    var_17 = {var_1: var_16}
    var_18 = '\n'
    var_19 = set()
    var_20 = []
    var_21 = {}
    var_22 = module_0.Config(**var_21)
    var_23 = [var_2]
    var_24 = []
    var_25 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'sys'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = 'comment1'
    var_12 = 'comment2'
    var_13 = [var_11, var_12]
    var_14 = {var_2: var_13}
    var_15 = {}
    var_16 = {var_1: var_15}
    var_17 = {}
    var_18 = {var_1: var_14, var_9: var_16, var_10: var_17}
    var_19 = {}
    var_20 = {var_1: var_19}
    var_21 = '\n'
    var_22 = set()
    var_23 = []
    var_24 = False
    var_25 = 'ignore_comments'
    var_26 = {var_25: var_24}
    var_27 = module_0.Config(**var_26)
    var_28 = [var_2]
    var_29 = []
    var_30 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'sys'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = {}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = {}
    var_15 = {var_1: var_11, var_9: var_13, var_10: var_14}
    var_16 = {}
    var_17 = {var_1: var_16}
    var_18 = '\n'
    var_19 = set()
    var_20 = []
    var_21 = {}
    var_22 = module_0.Config(**var_21)
    var_23 = [var_2]
    var_24 = 'os.path'
    var_25 = [var_24]
    var_26 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'above'
    var_9 = 'nested'
    var_10 = {}
    var_11 = {}
    var_12 = {var_1: var_11}
    var_13 = {}
    var_14 = {var_1: var_10, var_8: var_12, var_9: var_13}
    var_15 = 'os.path'
    var_16 = 'path as ospath'
    var_17 = [var_16]
    var_18 = {var_15: var_17}
    var_19 = {var_1: var_18}
    var_20 = '\n'
    var_21 = set()
    var_22 = []
    var_23 = True
    var_24 = 'combine_as_imports'
    var_25 = {var_24: var_23}
    var_26 = module_0.Config(**var_25)
    var_27 = [var_2]
    var_28 = []
    var_29 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'sys'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = {}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = {}
    var_15 = {var_1: var_11, var_9: var_13, var_10: var_14}
    var_16 = {}
    var_17 = {var_1: var_16}
    var_18 = '\n'
    var_19 = set()
    var_20 = []
    var_21 = True
    var_22 = 'force_single_line'
    var_23 = {var_22: var_21}
    var_24 = module_0.Config(**var_23)
    var_25 = [var_2]
    var_26 = []
    var_27 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = '*'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'above'
    var_9 = 'nested'
    var_10 = {}
    var_11 = {}
    var_12 = {var_1: var_11}
    var_13 = {}
    var_14 = {var_1: var_10, var_8: var_12, var_9: var_13}
    var_15 = {}
    var_16 = {var_1: var_15}
    var_17 = '\n'
    var_18 = set()
    var_19 = []
    var_20 = {}
    var_21 = module_0.Config(**var_20)
    var_22 = [var_2]
    var_23 = []
    var_24 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = '*'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'above'
    var_9 = 'nested'
    var_10 = {}
    var_11 = {}
    var_12 = {var_1: var_11}
    var_13 = 'star comment'
    var_14 = {var_3: var_13}
    var_15 = {var_2: var_14}
    var_16 = {var_1: var_10, var_8: var_12, var_9: var_15}
    var_17 = {}
    var_18 = {var_1: var_17}
    var_19 = '\n'
    var_20 = set()
    var_21 = []
    var_22 = False
    var_23 = 'ignore_comments'
    var_24 = {var_23: var_22}
    var_25 = module_0.Config(**var_24)
    var_26 = [var_2]
    var_27 = []
    var_28 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = '*'
    var_4 = 'path'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = {}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = {}
    var_15 = {var_1: var_11, var_9: var_13, var_10: var_14}
    var_16 = {}
    var_17 = {var_1: var_16}
    var_18 = '\n'
    var_19 = set()
    var_20 = []
    var_21 = True
    var_22 = 'combine_star'
    var_23 = {var_22: var_21}
    var_24 = module_0.Config(**var_23)
    var_25 = [var_2]
    var_26 = []
    var_27 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'sys'
    var_4 = 'path'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = {}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = {}
    var_15 = {var_1: var_11, var_9: var_13, var_10: var_14}
    var_16 = {}
    var_17 = {var_1: var_16}
    var_18 = '\n'
    var_19 = set()
    var_20 = []
    var_21 = True
    var_22 = 'no_inline_sort'
    var_23 = {var_22: var_21}
    var_24 = module_0.Config(**var_23)
    var_25 = [var_2]
    var_26 = []
    var_27 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'sys'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = {}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = {}
    var_15 = {var_1: var_11, var_9: var_13, var_10: var_14}
    var_16 = {}
    var_17 = {var_1: var_16}
    var_18 = '\n'
    var_19 = set()
    var_20 = []
    var_21 = True
    var_22 = 'only_sections'
    var_23 = {var_22: var_21}
    var_24 = module_0.Config(**var_23)
    var_25 = [var_2]
    var_26 = []
    var_27 = 'import'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_sorted_imports_no_imports. Retrieved 12/14 statements.
# Partially parsed test_sorted_imports_single_straight_import. Retrieved 23/25 statements.
# Partially parsed test_sorted_imports_multiple_straight_imports. Retrieved 25/27 statements.
# Partially parsed test_sorted_imports_with_comments. Retrieved 25/27 statements.
# Partially parsed test_sorted_imports_with_as_import. Retrieved 25/27 statements.
# Partially parsed test_sorted_imports_with_remove_imports. Retrieved 26/28 statements.
# Partially parsed test_sorted_imports_with_combine_straight_imports. Retrieved 26/28 statements.
# Partially parsed test_sorted_imports_with_from_imports. Retrieved 25/27 statements.
# Partially parsed test_sorted_imports_with_import_headings. Retrieved 26/28 statements.
# Partially parsed test_sorted_imports_with_no_sections. Retrieved 29/31 statements.
# Partially parsed test_sorted_imports_with_force_sort_within_sections. Retrieved 26/28 statements.
# Partially parsed test_sorted_imports_with_lines_between_sections. Retrieved 29/31 statements.
# Partially parsed test_sorted_imports_with_ensure_newline_before_comments. Retrieved 24/26 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = '\n'
    var_3 = -1
    var_4 = {}
    var_5 = {}
    var_6 = {}
    var_7 = {}
    var_8 = {}
    var_9 = []
    var_10 = 1
    var_11 = []
    var_12 = {}
    var_13 = module_0.Config(**var_12)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = '\n'
    var_3 = 0
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'os'
    var_7 = set()
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = {var_4: var_9}
    var_11 = 'above'
    var_12 = {}
    var_13 = {var_5: var_12}
    var_14 = {}
    var_15 = {var_11: var_13, var_5: var_14}
    var_16 = {}
    var_17 = {var_5: var_16}
    var_18 = {}
    var_19 = {}
    var_20 = [var_4]
    var_21 = 1
    var_22 = []
    var_23 = {}
    var_24 = module_0.Config(**var_23)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = '\n'
    var_3 = 0
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'os'
    var_7 = 'sys'
    var_8 = set()
    var_9 = set()
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = {var_5: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = {}
    var_19 = {var_5: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = [var_4]
    var_23 = 1
    var_24 = []
    var_25 = {}
    var_26 = module_0.Config(**var_25)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = '\n'
    var_3 = 0
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'os'
    var_7 = set()
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = {var_4: var_9}
    var_11 = 'above'
    var_12 = {}
    var_13 = {var_5: var_12}
    var_14 = 'comment'
    var_15 = [var_14]
    var_16 = {var_6: var_15}
    var_17 = {var_11: var_13, var_5: var_16}
    var_18 = {}
    var_19 = {var_5: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = [var_4]
    var_23 = 1
    var_24 = []
    var_25 = {}
    var_26 = module_0.Config(**var_25)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = '\n'
    var_3 = 0
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'os'
    var_7 = set()
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = {var_4: var_9}
    var_11 = 'above'
    var_12 = {}
    var_13 = {var_5: var_12}
    var_14 = {}
    var_15 = {var_11: var_13, var_5: var_14}
    var_16 = 'os_path'
    var_17 = [var_16]
    var_18 = {var_6: var_17}
    var_19 = {var_5: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = [var_4]
    var_23 = 1
    var_24 = []
    var_25 = {}
    var_26 = module_0.Config(**var_25)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = '\n'
    var_3 = 0
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'os'
    var_7 = 'sys'
    var_8 = set()
    var_9 = set()
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = {var_5: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = {}
    var_19 = {var_5: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = [var_4]
    var_23 = 1
    var_24 = []
    var_25 = [var_6]
    var_26 = 'remove_imports'
    var_27 = {var_26: var_25}
    var_28 = module_0.Config(**var_27)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = '\n'
    var_3 = 0
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'os'
    var_7 = 'sys'
    var_8 = set()
    var_9 = set()
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = {var_5: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = {}
    var_19 = {var_5: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = [var_4]
    var_23 = 1
    var_24 = []
    var_25 = True
    var_26 = 'combine_straight_imports'
    var_27 = {var_26: var_25}
    var_28 = module_0.Config(**var_27)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = '\n'
    var_3 = 0
    var_4 = 'THIRDPARTY'
    var_5 = 'from'
    var_6 = 'os'
    var_7 = 'path'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = {var_5: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = {}
    var_19 = {var_5: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = [var_4]
    var_23 = 1
    var_24 = []
    var_25 = {}
    var_26 = module_0.Config(**var_25)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = '\n'
    var_3 = 0
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'os'
    var_7 = set()
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = {var_4: var_9}
    var_11 = 'above'
    var_12 = {}
    var_13 = {var_5: var_12}
    var_14 = {}
    var_15 = {var_11: var_13, var_5: var_14}
    var_16 = {}
    var_17 = {var_5: var_16}
    var_18 = {}
    var_19 = {}
    var_20 = [var_4]
    var_21 = 1
    var_22 = []
    var_23 = 'thirdparty'
    var_24 = 'Third Party Imports'
    var_25 = {var_23: var_24}
    var_26 = 'import_headings'
    var_27 = {var_26: var_25}
    var_28 = module_0.Config(**var_27)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = '\n'
    var_3 = 0
    var_4 = 'FUTURE'
    var_5 = 'THIRDPARTY'
    var_6 = 'straight'
    var_7 = '__future__'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = 'os'
    var_12 = set()
    var_13 = {var_11: var_12}
    var_14 = {var_6: var_13}
    var_15 = {var_4: var_10, var_5: var_14}
    var_16 = 'above'
    var_17 = {}
    var_18 = {var_6: var_17}
    var_19 = {}
    var_20 = {var_16: var_18, var_6: var_19}
    var_21 = {}
    var_22 = {var_6: var_21}
    var_23 = {}
    var_24 = {}
    var_25 = [var_4, var_5]
    var_26 = 1
    var_27 = []
    var_28 = True
    var_29 = 'no_sections'
    var_30 = {var_29: var_28}
    var_31 = module_0.Config(**var_30)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = '\n'
    var_3 = 0
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'sys'
    var_7 = 'os'
    var_8 = set()
    var_9 = set()
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = {var_5: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = {}
    var_19 = {var_5: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = [var_4]
    var_23 = 1
    var_24 = []
    var_25 = True
    var_26 = 'force_sort_within_sections'
    var_27 = {var_26: var_25}
    var_28 = module_0.Config(**var_27)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = '\n'
    var_3 = 0
    var_4 = 'FUTURE'
    var_5 = 'THIRDPARTY'
    var_6 = 'straight'
    var_7 = '__future__'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = 'os'
    var_12 = set()
    var_13 = {var_11: var_12}
    var_14 = {var_6: var_13}
    var_15 = {var_4: var_10, var_5: var_14}
    var_16 = 'above'
    var_17 = {}
    var_18 = {var_6: var_17}
    var_19 = {}
    var_20 = {var_16: var_18, var_6: var_19}
    var_21 = {}
    var_22 = {var_6: var_21}
    var_23 = {}
    var_24 = {}
    var_25 = [var_4, var_5]
    var_26 = 1
    var_27 = []
    var_28 = 2
    var_29 = 'lines_between_sections'
    var_30 = {var_29: var_28}
    var_31 = module_0.Config(**var_30)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = '\n'
    var_3 = 0
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'os'
    var_7 = set()
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = {var_4: var_9}
    var_11 = 'above'
    var_12 = {}
    var_13 = {var_5: var_12}
    var_14 = {}
    var_15 = {var_11: var_13, var_5: var_14}
    var_16 = {}
    var_17 = {var_5: var_16}
    var_18 = {}
    var_19 = {}
    var_20 = [var_4]
    var_21 = 1
    var_22 = []
    var_23 = True
    var_24 = 'ensure_newline_before_comments'
    var_25 = {var_24: var_23}
    var_26 = module_0.Config(**var_25)



# Parsed testcases at query #7
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_sorted_imports_predicate. Retrieved 10/12 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 0
    var_1 = ''
    var_2 = [var_1]
    var_3 = '\n'
    var_4 = {}
    var_5 = {}
    var_6 = {}
    var_7 = []
    var_8 = 1
    var_9 = []
    var_10 = {}
    var_11 = module_0.Config(**var_10)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_sorted_imports_with_no_imports. Retrieved 9/11 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = "print('Hello')"
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = 1
    var_5 = '\n'
    var_6 = {}
    var_7 = {}
    var_8 = []
    var_9 = {}
    var_10 = module_0.Config(**var_9)



# Parsed testcases at query #10
#--------------------------




import isort.output as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._ensure_newline_before_comment(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import isort.output as module_0

def test_case_0():
    var_0 = '# comment'
    var_1 = [var_0]
    var_2 = module_0._ensure_newline_before_comment(var_1)
    var_3 = bool(var_2 == ['# comment'])
    assert var_3 is True

import isort.output as module_0

def test_case_0():
    var_0 = 'code'
    var_1 = [var_0]
    var_2 = module_0._ensure_newline_before_comment(var_1)
    var_3 = bool(var_2 == ['code'])
    assert var_3 is True

import isort.output as module_0

def test_case_0():
    var_0 = 'code'
    var_1 = '# comment'
    var_2 = [var_0, var_1]
    var_3 = module_0._ensure_newline_before_comment(var_2)
    var_4 = bool(var_3 == ['code', '', '# comment'])
    assert var_4 is True

import isort.output as module_0

def test_case_0():
    var_0 = 'code'
    var_1 = ''
    var_2 = '# comment'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._ensure_newline_before_comment(var_3)
    var_5 = bool(var_4 == ['code', '', '# comment'])
    assert var_5 is True

import isort.output as module_0

def test_case_0():
    var_0 = '# comment1'
    var_1 = '# comment2'
    var_2 = [var_0, var_1]
    var_3 = module_0._ensure_newline_before_comment(var_2)
    var_4 = bool(var_3 == ['# comment1', '# comment2'])
    assert var_4 is True

import isort.output as module_0

def test_case_0():
    var_0 = ''
    var_1 = '# comment'
    var_2 = [var_0, var_1]
    var_3 = module_0._ensure_newline_before_comment(var_2)
    var_4 = bool(var_3 == ['', '# comment'])
    assert var_4 is True

import isort.output as module_0

def test_case_0():
    var_0 = 'code1'
    var_1 = '# comment1'
    var_2 = 'code2'
    var_3 = '# comment2'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._ensure_newline_before_comment(var_4)
    var_6 = bool(var_5 == ['code1', '', '# comment1', 'code2', '', '# comment2'])
    assert var_6 is True

import isort.output as module_0

def test_case_0():
    var_0 = '# comment'
    var_1 = 'code'
    var_2 = [var_0, var_1]
    var_3 = module_0._ensure_newline_before_comment(var_2)
    var_4 = bool(var_3 == ['# comment', 'code'])
    assert var_4 is True

import isort.output as module_0

def test_case_0():
    var_0 = 'code'
    var_1 = '# comment'
    var_2 = [var_0, var_1]
    var_3 = module_0._ensure_newline_before_comment(var_2)
    var_4 = bool(var_3 == ['code', '', '# comment'])
    assert var_4 is True

import isort.output as module_0

def test_case_0():
    var_0 = 'code1'
    var_1 = 'code2'
    var_2 = '# comment'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._ensure_newline_before_comment(var_3)
    var_5 = bool(var_4 == ['code1', 'code2', '', '# comment'])
    assert var_5 is True

import isort.output as module_0

def test_case_0():
    var_0 = 'code'
    var_1 = ''
    var_2 = '# comment'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._ensure_newline_before_comment(var_3)
    var_5 = bool(var_4 == ['code', '', '# comment'])
    assert var_5 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test__with_from_imports_empty. Retrieved 20/22 statements.
# Partially parsed test__with_from_imports_single_import. Retrieved 23/25 statements.
# Partially parsed test__with_from_imports_with_comments. Retrieved 28/30 statements.
# Partially parsed test__with_from_imports_remove_imports. Retrieved 25/27 statements.
# Partially parsed test__with_from_imports_with_as_imports. Retrieved 27/29 statements.
# Partially parsed test__with_from_imports_star_import. Retrieved 26/28 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'above'
    var_6 = 'nested'
    var_7 = {}
    var_8 = {}
    var_9 = {var_1: var_8}
    var_10 = {}
    var_11 = {var_1: var_7, var_5: var_9, var_6: var_10}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = '\n'
    var_15 = set()
    var_16 = []
    var_17 = {}
    var_18 = module_0.Config(**var_17)
    var_19 = []
    var_20 = []
    var_21 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'above'
    var_9 = 'nested'
    var_10 = {}
    var_11 = {}
    var_12 = {var_1: var_11}
    var_13 = {}
    var_14 = {var_1: var_10, var_8: var_12, var_9: var_13}
    var_15 = {}
    var_16 = {var_1: var_15}
    var_17 = '\n'
    var_18 = set()
    var_19 = []
    var_20 = {}
    var_21 = module_0.Config(**var_20)
    var_22 = [var_2]
    var_23 = []
    var_24 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'above'
    var_9 = 'nested'
    var_10 = '# comment'
    var_11 = [var_10]
    var_12 = {var_2: var_11}
    var_13 = {}
    var_14 = {var_1: var_13}
    var_15 = '# nested comment'
    var_16 = {var_3: var_15}
    var_17 = {var_2: var_16}
    var_18 = {var_1: var_12, var_8: var_14, var_9: var_17}
    var_19 = {}
    var_20 = {var_1: var_19}
    var_21 = '\n'
    var_22 = set()
    var_23 = []
    var_24 = False
    var_25 = 'ignore_comments'
    var_26 = {var_25: var_24}
    var_27 = module_0.Config(**var_26)
    var_28 = [var_2]
    var_29 = []
    var_30 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'sys'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = {}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = {}
    var_15 = {var_1: var_11, var_9: var_13, var_10: var_14}
    var_16 = {}
    var_17 = {var_1: var_16}
    var_18 = '\n'
    var_19 = set()
    var_20 = []
    var_21 = {}
    var_22 = module_0.Config(**var_21)
    var_23 = [var_2]
    var_24 = 'os.path'
    var_25 = [var_24]
    var_26 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'above'
    var_9 = 'nested'
    var_10 = {}
    var_11 = {}
    var_12 = {var_1: var_11}
    var_13 = {}
    var_14 = {var_1: var_10, var_8: var_12, var_9: var_13}
    var_15 = 'os.path'
    var_16 = 'path as ospath'
    var_17 = [var_16]
    var_18 = {var_15: var_17}
    var_19 = {var_1: var_18}
    var_20 = '\n'
    var_21 = set()
    var_22 = []
    var_23 = True
    var_24 = 'combine_as_imports'
    var_25 = {var_24: var_23}
    var_26 = module_0.Config(**var_25)
    var_27 = [var_2]
    var_28 = []
    var_29 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = '*'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'above'
    var_9 = 'nested'
    var_10 = {}
    var_11 = {}
    var_12 = {var_1: var_11}
    var_13 = '# star comment'
    var_14 = {var_3: var_13}
    var_15 = {var_2: var_14}
    var_16 = {var_1: var_10, var_8: var_12, var_9: var_15}
    var_17 = {}
    var_18 = {var_1: var_17}
    var_19 = '\n'
    var_20 = set()
    var_21 = []
    var_22 = True
    var_23 = 'combine_star'
    var_24 = {var_23: var_22}
    var_25 = module_0.Config(**var_24)
    var_26 = [var_2]
    var_27 = []
    var_28 = 'import'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_with_from_imports_basic. Retrieved 23/26 statements.
# Partially parsed test_with_from_imports_with_comments. Retrieved 28/31 statements.
# Partially parsed test_with_from_imports_remove_imports. Retrieved 24/27 statements.
# Partially parsed test_with_from_imports_with_as_imports. Retrieved 31/34 statements.
# Partially parsed test_with_from_imports_with_star. Retrieved 28/31 statements.
# Partially parsed test_with_from_imports_force_single_line. Retrieved 25/29 statements.
# Partially parsed test_with_from_imports_ignore_comments. Retrieved 28/32 statements.
# Partially parsed test_with_from_imports_combine_as_imports. Retrieved 31/35 statements.
# Partially parsed test_with_from_imports_combine_star. Retrieved 28/32 statements.
# Partially parsed test_with_from_imports_only_sections. Retrieved 25/29 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = {}
    var_11 = {}
    var_12 = {var_1: var_11}
    var_13 = {var_1: var_10, var_9: var_12}
    var_14 = {}
    var_15 = {var_1: var_14}
    var_16 = '\n'
    var_17 = set()
    var_18 = []
    var_19 = {}
    var_20 = module_0.Config(**var_19)
    var_21 = [var_2]
    var_22 = 'section'
    var_23 = []
    var_24 = 'import'
    var_25 = 'from module import import1, import2'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = 'comment1'
    var_12 = 'comment2'
    var_13 = [var_11, var_12]
    var_14 = {var_2: var_13}
    var_15 = {}
    var_16 = {var_1: var_15}
    var_17 = {}
    var_18 = {var_1: var_14, var_9: var_16, var_10: var_17}
    var_19 = {}
    var_20 = {var_1: var_19}
    var_21 = '\n'
    var_22 = set()
    var_23 = []
    var_24 = {}
    var_25 = module_0.Config(**var_24)
    var_26 = [var_2]
    var_27 = 'section'
    var_28 = []
    var_29 = 'import'
    var_30 = 'from module import import1, import2'
    var_31 = 'comment1'
    var_32 = 'comment2'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = {}
    var_11 = {}
    var_12 = {var_1: var_11}
    var_13 = {var_1: var_10, var_9: var_12}
    var_14 = {}
    var_15 = {var_1: var_14}
    var_16 = '\n'
    var_17 = set()
    var_18 = []
    var_19 = {}
    var_20 = module_0.Config(**var_19)
    var_21 = [var_2]
    var_22 = 'section'
    var_23 = 'module.import1'
    var_24 = [var_23]
    var_25 = 'import'
    var_26 = 'from module import import2'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = {}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = {}
    var_15 = {var_1: var_11, var_9: var_13, var_10: var_14}
    var_16 = 'module.import1'
    var_17 = 'module.import2'
    var_18 = 'import1 as alias1'
    var_19 = [var_18]
    var_20 = 'import2 as alias2'
    var_21 = [var_20]
    var_22 = {var_16: var_19, var_17: var_21}
    var_23 = {var_1: var_22}
    var_24 = '\n'
    var_25 = set()
    var_26 = []
    var_27 = {}
    var_28 = module_0.Config(**var_27)
    var_29 = [var_2]
    var_30 = 'section'
    var_31 = []
    var_32 = 'import'
    var_33 = 'from module import import1 as alias1'
    var_34 = 'from module import import2 as alias2'
    var_35 = 'from module import import1, import2'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = '*'
    var_4 = 'import1'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = {}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = 'star comment'
    var_15 = [var_14]
    var_16 = {var_3: var_15}
    var_17 = {var_2: var_16}
    var_18 = {var_1: var_11, var_9: var_13, var_10: var_17}
    var_19 = {}
    var_20 = {var_1: var_19}
    var_21 = '\n'
    var_22 = set()
    var_23 = []
    var_24 = {}
    var_25 = module_0.Config(**var_24)
    var_26 = [var_2]
    var_27 = 'section'
    var_28 = []
    var_29 = 'import'
    var_30 = 'from module import *'
    var_31 = 'star comment'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = {}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = {}
    var_15 = {var_1: var_11, var_9: var_13, var_10: var_14}
    var_16 = {}
    var_17 = {var_1: var_16}
    var_18 = '\n'
    var_19 = set()
    var_20 = []
    var_21 = {}
    var_22 = module_0.Config(**var_21)
    var_23 = [var_2]
    var_24 = 'section'
    var_25 = []
    var_26 = 'import'
    var_27 = 'from module import import1'
    var_28 = 'from module import import2'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = 'comment1'
    var_12 = 'comment2'
    var_13 = [var_11, var_12]
    var_14 = {var_2: var_13}
    var_15 = {}
    var_16 = {var_1: var_15}
    var_17 = {}
    var_18 = {var_1: var_14, var_9: var_16, var_10: var_17}
    var_19 = {}
    var_20 = {var_1: var_19}
    var_21 = '\n'
    var_22 = set()
    var_23 = []
    var_24 = {}
    var_25 = module_0.Config(**var_24)
    var_26 = [var_2]
    var_27 = 'section'
    var_28 = []
    var_29 = 'import'
    var_30 = 'from module import import1, import2'
    var_31 = 'comment1'
    var_32 = 'comment2'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = {}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = {}
    var_15 = {var_1: var_11, var_9: var_13, var_10: var_14}
    var_16 = 'module.import1'
    var_17 = 'module.import2'
    var_18 = 'import1 as alias1'
    var_19 = [var_18]
    var_20 = 'import2 as alias2'
    var_21 = [var_20]
    var_22 = {var_16: var_19, var_17: var_21}
    var_23 = {var_1: var_22}
    var_24 = '\n'
    var_25 = set()
    var_26 = []
    var_27 = {}
    var_28 = module_0.Config(**var_27)
    var_29 = [var_2]
    var_30 = 'section'
    var_31 = []
    var_32 = 'import'
    var_33 = 'from module import import1 as alias1, import2 as alias2'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = '*'
    var_4 = 'import1'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = {}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = 'star comment'
    var_15 = [var_14]
    var_16 = {var_3: var_15}
    var_17 = {var_2: var_16}
    var_18 = {var_1: var_11, var_9: var_13, var_10: var_17}
    var_19 = {}
    var_20 = {var_1: var_19}
    var_21 = '\n'
    var_22 = set()
    var_23 = []
    var_24 = {}
    var_25 = module_0.Config(**var_24)
    var_26 = [var_2]
    var_27 = 'section'
    var_28 = []
    var_29 = 'import'
    var_30 = 'from module import *'
    var_31 = 'star comment'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = {}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = {}
    var_15 = {var_1: var_11, var_9: var_13, var_10: var_14}
    var_16 = {}
    var_17 = {var_1: var_16}
    var_18 = '\n'
    var_19 = set()
    var_20 = []
    var_21 = {}
    var_22 = module_0.Config(**var_21)
    var_23 = [var_2]
    var_24 = 'section'
    var_25 = []
    var_26 = 'import'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_predicate_at_line_1. Retrieved 5/7 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = []
    var_4 = ''
    var_5 = []
    var_6 = ''



# Parsed testcases at query #14
#--------------------------

# Partially parsed test__with_from_imports_basic. Retrieved 23/25 statements.
# Partially parsed test__with_from_imports_with_comments. Retrieved 27/29 statements.
# Partially parsed test__with_from_imports_remove_imports. Retrieved 25/27 statements.
# Partially parsed test__with_from_imports_with_as_imports. Retrieved 27/29 statements.
# Partially parsed test__with_from_imports_with_star. Retrieved 23/25 statements.
# Partially parsed test__with_from_imports_force_single_line. Retrieved 25/27 statements.
# Partially parsed test__with_from_imports_multiline. Retrieved 26/28 statements.
# Partially parsed test__with_from_imports_with_above_comments. Retrieved 25/27 statements.
# Partially parsed test__with_from_imports_with_nested_comments. Retrieved 26/28 statements.
# Partially parsed test__with_from_imports_with_straight_comments. Retrieved 31/33 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'above'
    var_9 = 'nested'
    var_10 = {}
    var_11 = {}
    var_12 = {var_1: var_11}
    var_13 = {}
    var_14 = {var_1: var_10, var_8: var_12, var_9: var_13}
    var_15 = {}
    var_16 = {var_1: var_15}
    var_17 = '\n'
    var_18 = set()
    var_19 = []
    var_20 = {}
    var_21 = module_0.Config(**var_20)
    var_22 = [var_2]
    var_23 = []
    var_24 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'above'
    var_9 = 'nested'
    var_10 = 'comment1'
    var_11 = 'comment2'
    var_12 = [var_10, var_11]
    var_13 = {var_2: var_12}
    var_14 = {}
    var_15 = {var_1: var_14}
    var_16 = {}
    var_17 = {var_1: var_13, var_8: var_15, var_9: var_16}
    var_18 = {}
    var_19 = {var_1: var_18}
    var_20 = '\n'
    var_21 = set()
    var_22 = []
    var_23 = False
    var_24 = 'ignore_comments'
    var_25 = {var_24: var_23}
    var_26 = module_0.Config(**var_25)
    var_27 = [var_2]
    var_28 = []
    var_29 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'sys'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = {}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = {}
    var_15 = {var_1: var_11, var_9: var_13, var_10: var_14}
    var_16 = {}
    var_17 = {var_1: var_16}
    var_18 = '\n'
    var_19 = set()
    var_20 = []
    var_21 = {}
    var_22 = module_0.Config(**var_21)
    var_23 = [var_2]
    var_24 = 'os.sys'
    var_25 = [var_24]
    var_26 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'above'
    var_9 = 'nested'
    var_10 = {}
    var_11 = {}
    var_12 = {var_1: var_11}
    var_13 = {}
    var_14 = {var_1: var_10, var_8: var_12, var_9: var_13}
    var_15 = 'os.path'
    var_16 = 'path as ospath'
    var_17 = [var_16]
    var_18 = {var_15: var_17}
    var_19 = {var_1: var_18}
    var_20 = '\n'
    var_21 = set()
    var_22 = []
    var_23 = True
    var_24 = 'combine_as_imports'
    var_25 = {var_24: var_23}
    var_26 = module_0.Config(**var_25)
    var_27 = [var_2]
    var_28 = []
    var_29 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = '*'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'nested'
    var_9 = 'star comment'
    var_10 = [var_9]
    var_11 = {var_3: var_10}
    var_12 = {var_2: var_11}
    var_13 = {var_8: var_12}
    var_14 = {}
    var_15 = {var_1: var_14}
    var_16 = '\n'
    var_17 = set()
    var_18 = []
    var_19 = True
    var_20 = 'combine_star'
    var_21 = {var_20: var_19}
    var_22 = module_0.Config(**var_21)
    var_23 = [var_2]
    var_24 = []
    var_25 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'sys'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = {}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = {}
    var_15 = {var_1: var_11, var_9: var_13, var_10: var_14}
    var_16 = {}
    var_17 = {var_1: var_16}
    var_18 = '\n'
    var_19 = set()
    var_20 = []
    var_21 = True
    var_22 = 'force_single_line'
    var_23 = {var_22: var_21}
    var_24 = module_0.Config(**var_23)
    var_25 = [var_2]
    var_26 = []
    var_27 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'sys'
    var_5 = 'env'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_2: var_6}
    var_8 = {var_1: var_7}
    var_9 = {var_0: var_8}
    var_10 = 'above'
    var_11 = 'nested'
    var_12 = {}
    var_13 = {}
    var_14 = {var_1: var_13}
    var_15 = {}
    var_16 = {var_1: var_12, var_10: var_14, var_11: var_15}
    var_17 = {}
    var_18 = {var_1: var_17}
    var_19 = '\n'
    var_20 = set()
    var_21 = []
    var_22 = 20
    var_23 = 'line_length'
    var_24 = {var_23: var_22}
    var_25 = module_0.Config(**var_24)
    var_26 = [var_2]
    var_27 = []
    var_28 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'above'
    var_9 = 'nested'
    var_10 = {}
    var_11 = 'above comment'
    var_12 = [var_11]
    var_13 = {var_2: var_12}
    var_14 = {var_1: var_13}
    var_15 = {}
    var_16 = {var_1: var_10, var_8: var_14, var_9: var_15}
    var_17 = {}
    var_18 = {var_1: var_17}
    var_19 = '\n'
    var_20 = set()
    var_21 = []
    var_22 = {}
    var_23 = module_0.Config(**var_22)
    var_24 = [var_2]
    var_25 = []
    var_26 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'above'
    var_9 = 'nested'
    var_10 = {}
    var_11 = {}
    var_12 = {var_1: var_11}
    var_13 = 'nested comment'
    var_14 = [var_13]
    var_15 = {var_3: var_14}
    var_16 = {var_2: var_15}
    var_17 = {var_1: var_10, var_8: var_12, var_9: var_16}
    var_18 = {}
    var_19 = {var_1: var_18}
    var_20 = '\n'
    var_21 = set()
    var_22 = []
    var_23 = {}
    var_24 = module_0.Config(**var_23)
    var_25 = [var_2]
    var_26 = []
    var_27 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'above'
    var_9 = 'nested'
    var_10 = 'straight'
    var_11 = {}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = {}
    var_15 = 'os.path'
    var_16 = 'straight comment'
    var_17 = [var_16]
    var_18 = {var_15: var_17}
    var_19 = {var_1: var_11, var_8: var_13, var_9: var_14, var_10: var_18}
    var_20 = 'path as ospath'
    var_21 = [var_20]
    var_22 = {var_15: var_21}
    var_23 = {var_1: var_22}
    var_24 = '\n'
    var_25 = set()
    var_26 = []
    var_27 = True
    var_28 = 'combine_as_imports'
    var_29 = {var_28: var_27}
    var_30 = module_0.Config(**var_29)
    var_31 = [var_2]
    var_32 = []
    var_33 = 'import'



# Parsed testcases at query #15
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_with_straight_imports_empty_modules. Retrieved 5/7 statements.
# Partially parsed test_with_straight_imports_single_module_no_comments. Retrieved 15/20 statements.
# Partially parsed test_with_straight_imports_single_module_with_inline_comment. Retrieved 17/22 statements.
# Partially parsed test_with_straight_imports_single_module_with_above_comment. Retrieved 17/22 statements.
# Partially parsed test_with_straight_imports_combine_straight_imports. Retrieved 22/27 statements.
# Partially parsed test_with_straight_imports_ignore_comments. Retrieved 18/23 statements.
# Partially parsed test_with_straight_imports_with_as_import. Retrieved 17/22 statements.
# Partially parsed test_with_straight_imports_remove_import. Retrieved 15/20 statements.
# Partially parsed test_with_straight_imports_custom_comment_prefix. Retrieved 18/23 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = []
    var_4 = 'section'
    var_5 = []
    var_6 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = 'section'
    var_2 = 'straight'
    var_3 = 'module1'
    var_4 = []
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {}
    var_8 = 'above'
    var_9 = {}
    var_10 = {var_2: var_9}
    var_11 = {}
    var_12 = {}
    var_13 = module_0.Config(**var_12)
    var_14 = [var_3]
    var_15 = []
    var_16 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = 'section'
    var_2 = 'straight'
    var_3 = 'module1'
    var_4 = []
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {}
    var_8 = 'above'
    var_9 = {}
    var_10 = {var_2: var_9}
    var_11 = 'comment1'
    var_12 = [var_11]
    var_13 = {var_3: var_12}
    var_14 = {}
    var_15 = module_0.Config(**var_14)
    var_16 = [var_3]
    var_17 = []
    var_18 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = 'section'
    var_2 = 'straight'
    var_3 = 'module1'
    var_4 = []
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {}
    var_8 = 'above'
    var_9 = '# comment1'
    var_10 = [var_9]
    var_11 = {var_3: var_10}
    var_12 = {var_2: var_11}
    var_13 = {}
    var_14 = {}
    var_15 = module_0.Config(**var_14)
    var_16 = [var_3]
    var_17 = []
    var_18 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = 'section'
    var_2 = 'straight'
    var_3 = 'module1'
    var_4 = 'module2'
    var_5 = []
    var_6 = []
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = {}
    var_10 = 'above'
    var_11 = {}
    var_12 = {var_2: var_11}
    var_13 = 'comment1'
    var_14 = [var_13]
    var_15 = 'comment2'
    var_16 = [var_15]
    var_17 = {var_3: var_14, var_4: var_16}
    var_18 = True
    var_19 = 'combine_straight_imports'
    var_20 = {var_19: var_18}
    var_21 = module_0.Config(**var_20)
    var_22 = [var_3, var_4]
    var_23 = []
    var_24 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = 'section'
    var_2 = 'straight'
    var_3 = 'module1'
    var_4 = []
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {}
    var_8 = 'above'
    var_9 = {}
    var_10 = {var_2: var_9}
    var_11 = 'comment1'
    var_12 = [var_11]
    var_13 = {var_3: var_12}
    var_14 = True
    var_15 = 'ignore_comments'
    var_16 = {var_15: var_14}
    var_17 = module_0.Config(**var_16)
    var_18 = [var_3]
    var_19 = []
    var_20 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = 'section'
    var_2 = 'straight'
    var_3 = 'module1'
    var_4 = []
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'alias1'
    var_8 = [var_7]
    var_9 = {var_3: var_8}
    var_10 = 'above'
    var_11 = {}
    var_12 = {var_2: var_11}
    var_13 = {}
    var_14 = {}
    var_15 = module_0.Config(**var_14)
    var_16 = [var_3]
    var_17 = []
    var_18 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = 'section'
    var_2 = 'straight'
    var_3 = 'module1'
    var_4 = []
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {}
    var_8 = 'above'
    var_9 = {}
    var_10 = {var_2: var_9}
    var_11 = {}
    var_12 = {}
    var_13 = module_0.Config(**var_12)
    var_14 = [var_3]
    var_15 = [var_3]
    var_16 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = 'section'
    var_2 = 'straight'
    var_3 = 'module1'
    var_4 = []
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {}
    var_8 = 'above'
    var_9 = {}
    var_10 = {var_2: var_9}
    var_11 = 'comment1'
    var_12 = [var_11]
    var_13 = {var_3: var_12}
    var_14 = '# '
    var_15 = 'comment_prefix'
    var_16 = {var_15: var_14}
    var_17 = module_0.Config(**var_16)
    var_18 = [var_3]
    var_19 = []
    var_20 = 'import'



# Parsed testcases at query #17
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #18
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_with_from_imports_basic. Retrieved 19/21 statements.
# Partially parsed test_with_from_imports_with_comments. Retrieved 20/22 statements.
# Partially parsed test_with_from_imports_remove_imports. Retrieved 20/22 statements.
# Partially parsed test_with_from_imports_with_as_imports. Retrieved 22/24 statements.
# Partially parsed test_with_from_imports_with_star_comment. Retrieved 21/23 statements.
# Partially parsed test_with_from_imports_force_single_line. Retrieved 21/23 statements.
# Partially parsed test_with_from_imports_combine_as_imports. Retrieved 23/25 statements.
# Partially parsed test_with_from_imports_ignore_comments. Retrieved 21/23 statements.
# Partially parsed test_with_from_imports_only_sections. Retrieved 20/22 statements.
# Partially parsed test_with_from_imports_no_inline_sort. Retrieved 21/23 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = ()
    var_9 = {var_2: var_8}
    var_10 = {var_1: var_9}
    var_11 = {}
    var_12 = {var_1: var_11}
    var_13 = '\n'
    var_14 = set()
    var_15 = []
    var_16 = {}
    var_17 = module_0.Config(**var_16)
    var_18 = [var_2]
    var_19 = []
    var_20 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = '# comment'
    var_9 = (var_8,)
    var_10 = {var_2: var_9}
    var_11 = {var_1: var_10}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = '\n'
    var_15 = set()
    var_16 = []
    var_17 = {}
    var_18 = module_0.Config(**var_17)
    var_19 = [var_2]
    var_20 = []
    var_21 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = ()
    var_9 = {var_2: var_8}
    var_10 = {var_1: var_9}
    var_11 = {}
    var_12 = {var_1: var_11}
    var_13 = '\n'
    var_14 = set()
    var_15 = []
    var_16 = {}
    var_17 = module_0.Config(**var_16)
    var_18 = [var_2]
    var_19 = 'os.path'
    var_20 = [var_19]
    var_21 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = ()
    var_9 = {var_2: var_8}
    var_10 = {var_1: var_9}
    var_11 = 'os.path'
    var_12 = 'path as ospath'
    var_13 = [var_12]
    var_14 = {var_11: var_13}
    var_15 = {var_1: var_14}
    var_16 = '\n'
    var_17 = set()
    var_18 = []
    var_19 = {}
    var_20 = module_0.Config(**var_19)
    var_21 = [var_2]
    var_22 = []
    var_23 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = '*'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'nested'
    var_9 = '# star comment'
    var_10 = {var_3: var_9}
    var_11 = {var_2: var_10}
    var_12 = {var_8: var_11}
    var_13 = {}
    var_14 = {var_1: var_13}
    var_15 = '\n'
    var_16 = set()
    var_17 = []
    var_18 = {}
    var_19 = module_0.Config(**var_18)
    var_20 = [var_2]
    var_21 = []
    var_22 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'sys'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = ()
    var_10 = {var_2: var_9}
    var_11 = {var_1: var_10}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = '\n'
    var_15 = set()
    var_16 = []
    var_17 = True
    var_18 = 'force_single_line'
    var_19 = {var_18: var_17}
    var_20 = module_0.Config(**var_19)
    var_21 = [var_2]
    var_22 = []
    var_23 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = ()
    var_9 = {var_2: var_8}
    var_10 = {var_1: var_9}
    var_11 = 'os.path'
    var_12 = 'path as ospath'
    var_13 = [var_12]
    var_14 = {var_11: var_13}
    var_15 = {var_1: var_14}
    var_16 = '\n'
    var_17 = set()
    var_18 = []
    var_19 = True
    var_20 = 'combine_as_imports'
    var_21 = {var_20: var_19}
    var_22 = module_0.Config(**var_21)
    var_23 = [var_2]
    var_24 = []
    var_25 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = '# comment'
    var_9 = (var_8,)
    var_10 = {var_2: var_9}
    var_11 = {var_1: var_10}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = '\n'
    var_15 = set()
    var_16 = []
    var_17 = True
    var_18 = 'ignore_comments'
    var_19 = {var_18: var_17}
    var_20 = module_0.Config(**var_19)
    var_21 = [var_2]
    var_22 = []
    var_23 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = ()
    var_9 = {var_2: var_8}
    var_10 = {var_1: var_9}
    var_11 = {}
    var_12 = {var_1: var_11}
    var_13 = '\n'
    var_14 = set()
    var_15 = []
    var_16 = True
    var_17 = 'only_sections'
    var_18 = {var_17: var_16}
    var_19 = module_0.Config(**var_18)
    var_20 = [var_2]
    var_21 = []
    var_22 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'sys'
    var_4 = 'path'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = ()
    var_10 = {var_2: var_9}
    var_11 = {var_1: var_10}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = '\n'
    var_15 = set()
    var_16 = []
    var_17 = True
    var_18 = 'no_inline_sort'
    var_19 = {var_18: var_17}
    var_20 = module_0.Config(**var_19)
    var_21 = [var_2]
    var_22 = []
    var_23 = 'import'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_with_star_comments_when_star_comment_exists. Retrieved 7/11 statements.
# Partially parsed test_with_star_comments_when_star_comment_does_not_exist. Retrieved 5/9 statements.
# Partially parsed test_with_star_comments_when_module_does_not_exist. Retrieved 6/8 statements.


def test_case_0():
    var_0 = 'nested'
    var_1 = '*'
    var_2 = 'star_comment'
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

def test_case_0():
    var_0 = 'nested'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'comment1'
    var_5 = 'comment2'
    var_6 = [var_4, var_5]



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_sorted_imports_no_imports. Retrieved 12/14 statements.
# Partially parsed test_sorted_imports_single_import. Retrieved 25/27 statements.
# Partially parsed test_sorted_imports_multiple_imports. Retrieved 27/29 statements.
# Partially parsed test_sorted_imports_with_comments. Retrieved 28/30 statements.
# Partially parsed test_sorted_imports_combine_straight_imports. Retrieved 28/30 statements.
# Partially parsed test_sorted_imports_with_as_imports. Retrieved 27/29 statements.
# Partially parsed test_sorted_imports_with_from_imports. Retrieved 27/29 statements.
# Partially parsed test_sorted_imports_with_section_headings. Retrieved 28/30 statements.
# Partially parsed test_sorted_imports_with_force_sort_within_sections. Retrieved 28/30 statements.
# Partially parsed test_sorted_imports_with_lines_between_sections. Retrieved 31/33 statements.
# Partially parsed test_sorted_imports_with_remove_imports. Retrieved 28/30 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = -1
    var_3 = '\n'
    var_4 = {}
    var_5 = {}
    var_6 = {}
    var_7 = []
    var_8 = {}
    var_9 = {}
    var_10 = 1
    var_11 = []
    var_12 = {}
    var_13 = module_0.Config(**var_12)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = {}
    var_19 = {var_5: var_18}
    var_20 = [var_4]
    var_21 = {}
    var_22 = {}
    var_23 = 1
    var_24 = []
    var_25 = {}
    var_26 = module_0.Config(**var_25)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = 'sys'
    var_9 = set()
    var_10 = set()
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = {var_5: var_11, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = 'above'
    var_16 = {}
    var_17 = {var_5: var_16}
    var_18 = {}
    var_19 = {var_15: var_17, var_5: var_18}
    var_20 = {}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []
    var_27 = {}
    var_28 = module_0.Config(**var_27)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = '# comment'
    var_17 = [var_16]
    var_18 = {var_7: var_17}
    var_19 = {var_13: var_15, var_5: var_18}
    var_20 = {}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []
    var_27 = False
    var_28 = 'ignore_comments'
    var_29 = {var_28: var_27}
    var_30 = module_0.Config(**var_29)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = 'sys'
    var_9 = set()
    var_10 = set()
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = {var_5: var_11, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = 'above'
    var_16 = {}
    var_17 = {var_5: var_16}
    var_18 = {}
    var_19 = {var_15: var_17, var_5: var_18}
    var_20 = {}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []
    var_27 = True
    var_28 = 'combine_straight_imports'
    var_29 = {var_28: var_27}
    var_30 = module_0.Config(**var_29)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = 'os_path'
    var_19 = [var_18]
    var_20 = {var_7: var_19}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []
    var_27 = {}
    var_28 = module_0.Config(**var_27)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = {}
    var_8 = 'os'
    var_9 = 'path'
    var_10 = set()
    var_11 = {var_9: var_10}
    var_12 = {var_8: var_11}
    var_13 = {var_5: var_7, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = 'above'
    var_16 = {}
    var_17 = {var_5: var_16}
    var_18 = {}
    var_19 = {var_15: var_17, var_5: var_18}
    var_20 = {}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []
    var_27 = {}
    var_28 = module_0.Config(**var_27)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = {}
    var_19 = {var_5: var_18}
    var_20 = [var_4]
    var_21 = {}
    var_22 = {}
    var_23 = 1
    var_24 = []
    var_25 = 'stdlib'
    var_26 = 'Standard Library'
    var_27 = {var_25: var_26}
    var_28 = 'import_headings'
    var_29 = {var_28: var_27}
    var_30 = module_0.Config(**var_29)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'sys'
    var_8 = 'os'
    var_9 = set()
    var_10 = set()
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = {var_5: var_11, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = 'above'
    var_16 = {}
    var_17 = {var_5: var_16}
    var_18 = {}
    var_19 = {var_15: var_17, var_5: var_18}
    var_20 = {}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []
    var_27 = True
    var_28 = 'force_sort_within_sections'
    var_29 = {var_28: var_27}
    var_30 = module_0.Config(**var_29)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'FUTURE'
    var_5 = 'STDLIB'
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = '__future__'
    var_9 = set()
    var_10 = {var_8: var_9}
    var_11 = {}
    var_12 = {var_6: var_10, var_7: var_11}
    var_13 = 'os'
    var_14 = set()
    var_15 = {var_13: var_14}
    var_16 = {}
    var_17 = {var_6: var_15, var_7: var_16}
    var_18 = {var_4: var_12, var_5: var_17}
    var_19 = 'above'
    var_20 = {}
    var_21 = {var_6: var_20}
    var_22 = {}
    var_23 = {var_19: var_21, var_6: var_22}
    var_24 = {}
    var_25 = {var_6: var_24}
    var_26 = [var_4, var_5]
    var_27 = {}
    var_28 = {}
    var_29 = 1
    var_30 = []
    var_31 = 'lines_between_sections'
    var_32 = {var_31: var_29}
    var_33 = module_0.Config(**var_32)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = 'sys'
    var_9 = set()
    var_10 = set()
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = {var_5: var_11, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = 'above'
    var_16 = {}
    var_17 = {var_5: var_16}
    var_18 = {}
    var_19 = {var_15: var_17, var_5: var_18}
    var_20 = {}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []
    var_27 = [var_7]
    var_28 = 'remove_imports'
    var_29 = {var_28: var_27}
    var_30 = module_0.Config(**var_29)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_sorted_imports_no_imports. Retrieved 11/13 statements.
# Partially parsed test_sorted_imports_single_import. Retrieved 24/26 statements.
# Partially parsed test_sorted_imports_multiple_imports. Retrieved 26/28 statements.
# Partially parsed test_sorted_imports_with_comments. Retrieved 26/28 statements.
# Partially parsed test_sorted_imports_with_as_imports. Retrieved 26/28 statements.
# Partially parsed test_sorted_imports_with_remove_imports. Retrieved 28/30 statements.
# Partially parsed test_sorted_imports_with_combine_straight_imports. Retrieved 27/29 statements.
# Partially parsed test_sorted_imports_with_force_sort_within_sections. Retrieved 27/29 statements.
# Partially parsed test_sorted_imports_with_import_headings. Retrieved 28/30 statements.
# Partially parsed test_sorted_imports_with_lines_between_sections. Retrieved 32/34 statements.
# Partially parsed test_sorted_imports_with_lines_after_imports. Retrieved 25/27 statements.
# Partially parsed test_sorted_imports_with_ensure_newline_before_comments. Retrieved 26/29 statements.


def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = -1
    var_3 = '\n'
    var_4 = {}
    var_5 = {}
    var_6 = {}
    var_7 = []
    var_8 = {}
    var_9 = {}
    var_10 = 1
    var_11 = []

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = {}
    var_19 = {var_5: var_18}
    var_20 = [var_4]
    var_21 = {}
    var_22 = {}
    var_23 = 1
    var_24 = []

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = 'sys'
    var_9 = set()
    var_10 = set()
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = {var_5: var_11, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = 'above'
    var_16 = {}
    var_17 = {var_5: var_16}
    var_18 = {}
    var_19 = {var_15: var_17, var_5: var_18}
    var_20 = {}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = '# comment'
    var_17 = [var_16]
    var_18 = {var_7: var_17}
    var_19 = {var_13: var_15, var_5: var_18}
    var_20 = {}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = 'ospath'
    var_19 = {var_18}
    var_20 = {var_7: var_19}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = {}
    var_24 = {}
    var_25 = 1
    var_26 = []

import isort.settings as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'remove_imports'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = ''
    var_6 = [var_5]
    var_7 = 0
    var_8 = '\n'
    var_9 = 'STDLIB'
    var_10 = 'straight'
    var_11 = 'from'
    var_12 = 'sys'
    var_13 = set()
    var_14 = set()
    var_15 = {var_0: var_13, var_12: var_14}
    var_16 = {}
    var_17 = {var_10: var_15, var_11: var_16}
    var_18 = {var_9: var_17}
    var_19 = 'above'
    var_20 = {}
    var_21 = {var_10: var_20}
    var_22 = {}
    var_23 = {var_19: var_21, var_10: var_22}
    var_24 = {}
    var_25 = {var_10: var_24}
    var_26 = [var_9]
    var_27 = {}
    var_28 = {}
    var_29 = 1
    var_30 = []

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'combine_straight_imports'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = ''
    var_5 = [var_4]
    var_6 = 0
    var_7 = '\n'
    var_8 = 'STDLIB'
    var_9 = 'straight'
    var_10 = 'from'
    var_11 = 'os'
    var_12 = 'sys'
    var_13 = set()
    var_14 = set()
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = {}
    var_17 = {var_9: var_15, var_10: var_16}
    var_18 = {var_8: var_17}
    var_19 = 'above'
    var_20 = {}
    var_21 = {var_9: var_20}
    var_22 = {}
    var_23 = {var_19: var_21, var_9: var_22}
    var_24 = {}
    var_25 = {var_9: var_24}
    var_26 = [var_8]
    var_27 = {}
    var_28 = {}
    var_29 = []

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'force_sort_within_sections'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = ''
    var_5 = [var_4]
    var_6 = 0
    var_7 = '\n'
    var_8 = 'STDLIB'
    var_9 = 'straight'
    var_10 = 'from'
    var_11 = 'sys'
    var_12 = 'os'
    var_13 = set()
    var_14 = set()
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = {}
    var_17 = {var_9: var_15, var_10: var_16}
    var_18 = {var_8: var_17}
    var_19 = 'above'
    var_20 = {}
    var_21 = {var_9: var_20}
    var_22 = {}
    var_23 = {var_19: var_21, var_9: var_22}
    var_24 = {}
    var_25 = {var_9: var_24}
    var_26 = [var_8]
    var_27 = {}
    var_28 = {}
    var_29 = []

import isort.settings as module_0

def test_case_0():
    var_0 = 'stdlib'
    var_1 = 'Standard Library'
    var_2 = {var_0: var_1}
    var_3 = 'import_headings'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = ''
    var_7 = [var_6]
    var_8 = 0
    var_9 = '\n'
    var_10 = 'STDLIB'
    var_11 = 'straight'
    var_12 = 'from'
    var_13 = 'os'
    var_14 = set()
    var_15 = {var_13: var_14}
    var_16 = {}
    var_17 = {var_11: var_15, var_12: var_16}
    var_18 = {var_10: var_17}
    var_19 = 'above'
    var_20 = {}
    var_21 = {var_11: var_20}
    var_22 = {}
    var_23 = {var_19: var_21, var_11: var_22}
    var_24 = {}
    var_25 = {var_11: var_24}
    var_26 = [var_10]
    var_27 = {}
    var_28 = {}
    var_29 = 1
    var_30 = []

import isort.settings as module_0

def test_case_0():
    var_0 = 2
    var_1 = 'lines_between_sections'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = ''
    var_5 = [var_4]
    var_6 = 0
    var_7 = '\n'
    var_8 = 'STDLIB'
    var_9 = 'THIRDPARTY'
    var_10 = 'straight'
    var_11 = 'from'
    var_12 = 'os'
    var_13 = set()
    var_14 = {var_12: var_13}
    var_15 = {}
    var_16 = {var_10: var_14, var_11: var_15}
    var_17 = 'numpy'
    var_18 = set()
    var_19 = {var_17: var_18}
    var_20 = {}
    var_21 = {var_10: var_19, var_11: var_20}
    var_22 = {var_8: var_16, var_9: var_21}
    var_23 = 'above'
    var_24 = {}
    var_25 = {var_10: var_24}
    var_26 = {}
    var_27 = {var_23: var_25, var_10: var_26}
    var_28 = {}
    var_29 = {var_10: var_28}
    var_30 = [var_8, var_9]
    var_31 = {}
    var_32 = {}
    var_33 = 1
    var_34 = []

import isort.settings as module_0

def test_case_0():
    var_0 = 2
    var_1 = 'lines_after_imports'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = "print('hello')"
    var_5 = [var_4]
    var_6 = 0
    var_7 = '\n'
    var_8 = 'STDLIB'
    var_9 = 'straight'
    var_10 = 'from'
    var_11 = 'os'
    var_12 = set()
    var_13 = {var_11: var_12}
    var_14 = {}
    var_15 = {var_9: var_13, var_10: var_14}
    var_16 = {var_8: var_15}
    var_17 = 'above'
    var_18 = {}
    var_19 = {var_9: var_18}
    var_20 = {}
    var_21 = {var_17: var_19, var_9: var_20}
    var_22 = {}
    var_23 = {var_9: var_22}
    var_24 = [var_8]
    var_25 = {}
    var_26 = {}
    var_27 = []

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'ensure_newline_before_comments'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = ''
    var_5 = [var_4]
    var_6 = 0
    var_7 = '\n'
    var_8 = 'STDLIB'
    var_9 = 'straight'
    var_10 = 'from'
    var_11 = 'os'
    var_12 = set()
    var_13 = {var_11: var_12}
    var_14 = {}
    var_15 = {var_9: var_13, var_10: var_14}
    var_16 = {var_8: var_15}
    var_17 = 'above'
    var_18 = {}
    var_19 = {var_9: var_18}
    var_20 = {}
    var_21 = {var_17: var_19, var_9: var_20}
    var_22 = {}
    var_23 = {var_9: var_22}
    var_24 = [var_8]
    var_25 = {}
    var_26 = {}
    var_27 = []
    var_28 = '# comment'



# Parsed testcases at query #23
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #24
#--------------------------

# Partially parsed test__with_from_imports_basic_case. Retrieved 24/26 statements.
# Partially parsed test__with_from_imports_with_star. Retrieved 23/25 statements.
# Partially parsed test__with_from_imports_with_comments. Retrieved 25/27 statements.
# Partially parsed test__with_from_imports_with_as_imports. Retrieved 26/28 statements.
# Partially parsed test__with_from_imports_with_remove_imports. Retrieved 25/27 statements.
# Partially parsed test__with_from_imports_with_force_single_line. Retrieved 25/27 statements.
# Partially parsed test__with_from_imports_with_combine_as_imports. Retrieved 27/29 statements.
# Partially parsed test__with_from_imports_with_ignore_comments. Retrieved 26/28 statements.
# Partially parsed test__with_from_imports_with_comment_prefix. Retrieved 26/28 statements.
# Partially parsed test__with_from_imports_with_above_comments. Retrieved 25/27 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = {}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = {}
    var_15 = {var_1: var_11, var_9: var_13, var_10: var_14}
    var_16 = {}
    var_17 = {var_1: var_16}
    var_18 = '\n'
    var_19 = set()
    var_20 = []
    var_21 = {}
    var_22 = module_0.Config(**var_21)
    var_23 = [var_2]
    var_24 = []
    var_25 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = '*'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'above'
    var_9 = 'nested'
    var_10 = {}
    var_11 = {}
    var_12 = {var_1: var_11}
    var_13 = {}
    var_14 = {var_1: var_10, var_8: var_12, var_9: var_13}
    var_15 = {}
    var_16 = {var_1: var_15}
    var_17 = '\n'
    var_18 = set()
    var_19 = []
    var_20 = {}
    var_21 = module_0.Config(**var_20)
    var_22 = [var_2]
    var_23 = []
    var_24 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'above'
    var_9 = 'nested'
    var_10 = 'comment1'
    var_11 = [var_10]
    var_12 = {var_2: var_11}
    var_13 = {}
    var_14 = {var_1: var_13}
    var_15 = {}
    var_16 = {var_1: var_12, var_8: var_14, var_9: var_15}
    var_17 = {}
    var_18 = {var_1: var_17}
    var_19 = '\n'
    var_20 = set()
    var_21 = []
    var_22 = {}
    var_23 = module_0.Config(**var_22)
    var_24 = [var_2]
    var_25 = []
    var_26 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'above'
    var_9 = 'nested'
    var_10 = {}
    var_11 = {}
    var_12 = {var_1: var_11}
    var_13 = {}
    var_14 = {var_1: var_10, var_8: var_12, var_9: var_13}
    var_15 = 'module.import1'
    var_16 = 'import1 as alias1'
    var_17 = [var_16]
    var_18 = {var_15: var_17}
    var_19 = {var_1: var_18}
    var_20 = '\n'
    var_21 = set()
    var_22 = []
    var_23 = {}
    var_24 = module_0.Config(**var_23)
    var_25 = [var_2]
    var_26 = []
    var_27 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = {}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = {}
    var_15 = {var_1: var_11, var_9: var_13, var_10: var_14}
    var_16 = {}
    var_17 = {var_1: var_16}
    var_18 = '\n'
    var_19 = set()
    var_20 = []
    var_21 = {}
    var_22 = module_0.Config(**var_21)
    var_23 = [var_2]
    var_24 = 'module.import1'
    var_25 = [var_24]
    var_26 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = 'nested'
    var_11 = {}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = {}
    var_15 = {var_1: var_11, var_9: var_13, var_10: var_14}
    var_16 = {}
    var_17 = {var_1: var_16}
    var_18 = '\n'
    var_19 = set()
    var_20 = []
    var_21 = True
    var_22 = 'force_single_line'
    var_23 = {var_22: var_21}
    var_24 = module_0.Config(**var_23)
    var_25 = [var_2]
    var_26 = []
    var_27 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'above'
    var_9 = 'nested'
    var_10 = {}
    var_11 = {}
    var_12 = {var_1: var_11}
    var_13 = {}
    var_14 = {var_1: var_10, var_8: var_12, var_9: var_13}
    var_15 = 'module.import1'
    var_16 = 'import1 as alias1'
    var_17 = [var_16]
    var_18 = {var_15: var_17}
    var_19 = {var_1: var_18}
    var_20 = '\n'
    var_21 = set()
    var_22 = []
    var_23 = True
    var_24 = 'combine_as_imports'
    var_25 = {var_24: var_23}
    var_26 = module_0.Config(**var_25)
    var_27 = [var_2]
    var_28 = []
    var_29 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'above'
    var_9 = 'nested'
    var_10 = 'comment1'
    var_11 = [var_10]
    var_12 = {var_2: var_11}
    var_13 = {}
    var_14 = {var_1: var_13}
    var_15 = {}
    var_16 = {var_1: var_12, var_8: var_14, var_9: var_15}
    var_17 = {}
    var_18 = {var_1: var_17}
    var_19 = '\n'
    var_20 = set()
    var_21 = []
    var_22 = True
    var_23 = 'ignore_comments'
    var_24 = {var_23: var_22}
    var_25 = module_0.Config(**var_24)
    var_26 = [var_2]
    var_27 = []
    var_28 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'above'
    var_9 = 'nested'
    var_10 = 'comment1'
    var_11 = [var_10]
    var_12 = {var_2: var_11}
    var_13 = {}
    var_14 = {var_1: var_13}
    var_15 = {}
    var_16 = {var_1: var_12, var_8: var_14, var_9: var_15}
    var_17 = {}
    var_18 = {var_1: var_17}
    var_19 = '\n'
    var_20 = set()
    var_21 = []
    var_22 = '# '
    var_23 = 'comment_prefix'
    var_24 = {var_23: var_22}
    var_25 = module_0.Config(**var_24)
    var_26 = [var_2]
    var_27 = []
    var_28 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'above'
    var_9 = 'nested'
    var_10 = {}
    var_11 = 'above_comment1'
    var_12 = [var_11]
    var_13 = {var_2: var_12}
    var_14 = {var_1: var_13}
    var_15 = {}
    var_16 = {var_1: var_10, var_8: var_14, var_9: var_15}
    var_17 = {}
    var_18 = {var_1: var_17}
    var_19 = '\n'
    var_20 = set()
    var_21 = []
    var_22 = {}
    var_23 = module_0.Config(**var_22)
    var_24 = [var_2]
    var_25 = []
    var_26 = 'import'



# Parsed testcases at query #25
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_with_straight_imports_empty_modules. Retrieved 15/17 statements.
# Partially parsed test_with_straight_imports_combine_no_as_imports. Retrieved 23/25 statements.
# Partially parsed test_with_straight_imports_combine_with_as_imports. Retrieved 21/23 statements.
# Partially parsed test_with_straight_imports_no_combine. Retrieved 19/21 statements.
# Partially parsed test_with_straight_imports_remove_imports. Retrieved 19/21 statements.
# Partially parsed test_with_straight_imports_ignore_comments. Retrieved 20/22 statements.
# Partially parsed test_with_straight_imports_custom_comment_prefix. Retrieved 20/22 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'straight'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = {var_0: var_3}
    var_5 = 'above'
    var_6 = {}
    var_7 = {var_0: var_6}
    var_8 = {}
    var_9 = {var_5: var_7, var_0: var_8}
    var_10 = []
    var_11 = True
    var_12 = 'combine_straight_imports'
    var_13 = {var_12: var_11}
    var_14 = module_0.Config(**var_13)
    var_15 = []
    var_16 = []
    var_17 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'straight'
    var_1 = 'module1'
    var_2 = 'module2'
    var_3 = []
    var_4 = []
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = {}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = '# comment above'
    var_11 = [var_10]
    var_12 = {var_1: var_11}
    var_13 = {var_0: var_12}
    var_14 = '# inline comment'
    var_15 = [var_14]
    var_16 = {var_1: var_15}
    var_17 = {var_9: var_13, var_0: var_16}
    var_18 = []
    var_19 = True
    var_20 = 'combine_straight_imports'
    var_21 = {var_20: var_19}
    var_22 = module_0.Config(**var_21)
    var_23 = [var_1, var_2]
    var_24 = []
    var_25 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'straight'
    var_1 = 'module1'
    var_2 = 'module2'
    var_3 = []
    var_4 = []
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'alias'
    var_8 = [var_7]
    var_9 = {var_1: var_8}
    var_10 = {var_0: var_9}
    var_11 = 'above'
    var_12 = {}
    var_13 = {var_0: var_12}
    var_14 = {}
    var_15 = {var_11: var_13, var_0: var_14}
    var_16 = []
    var_17 = True
    var_18 = 'combine_straight_imports'
    var_19 = {var_18: var_17}
    var_20 = module_0.Config(**var_19)
    var_21 = [var_1, var_2]
    var_22 = []
    var_23 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'straight'
    var_1 = 'module1'
    var_2 = 'module2'
    var_3 = []
    var_4 = []
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = {}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = {}
    var_11 = {var_0: var_10}
    var_12 = {}
    var_13 = {var_9: var_11, var_0: var_12}
    var_14 = []
    var_15 = False
    var_16 = 'combine_straight_imports'
    var_17 = {var_16: var_15}
    var_18 = module_0.Config(**var_17)
    var_19 = [var_1, var_2]
    var_20 = []
    var_21 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'straight'
    var_1 = 'module1'
    var_2 = 'module2'
    var_3 = []
    var_4 = []
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = {}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = {}
    var_11 = {var_0: var_10}
    var_12 = {}
    var_13 = {var_9: var_11, var_0: var_12}
    var_14 = []
    var_15 = False
    var_16 = 'combine_straight_imports'
    var_17 = {var_16: var_15}
    var_18 = module_0.Config(**var_17)
    var_19 = [var_1, var_2]
    var_20 = [var_1]
    var_21 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'straight'
    var_1 = 'module1'
    var_2 = []
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = {}
    var_6 = {var_0: var_5}
    var_7 = 'above'
    var_8 = {}
    var_9 = {var_0: var_8}
    var_10 = '# comment'
    var_11 = [var_10]
    var_12 = {var_1: var_11}
    var_13 = {var_7: var_9, var_0: var_12}
    var_14 = []
    var_15 = False
    var_16 = True
    var_17 = 'combine_straight_imports'
    var_18 = 'ignore_comments'
    var_19 = {var_17: var_15, var_18: var_16}
    var_20 = module_0.Config(**var_19)
    var_21 = [var_1]
    var_22 = []
    var_23 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'straight'
    var_1 = 'module1'
    var_2 = []
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = {}
    var_6 = {var_0: var_5}
    var_7 = 'above'
    var_8 = {}
    var_9 = {var_0: var_8}
    var_10 = '# comment'
    var_11 = [var_10]
    var_12 = {var_1: var_11}
    var_13 = {var_7: var_9, var_0: var_12}
    var_14 = []
    var_15 = False
    var_16 = ' # '
    var_17 = 'combine_straight_imports'
    var_18 = 'comment_prefix'
    var_19 = {var_17: var_15, var_18: var_16}
    var_20 = module_0.Config(**var_19)
    var_21 = [var_1]
    var_22 = []
    var_23 = 'import'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_sorted_imports_no_imports. Retrieved 12/14 statements.
# Partially parsed test_sorted_imports_basic_straight_imports. Retrieved 25/27 statements.
# Partially parsed test_sorted_imports_with_from_imports. Retrieved 25/27 statements.
# Partially parsed test_sorted_imports_combine_straight_imports. Retrieved 26/28 statements.
# Partially parsed test_sorted_imports_with_comments. Retrieved 25/27 statements.
# Partially parsed test_sorted_imports_remove_imports. Retrieved 26/28 statements.
# Partially parsed test_sorted_imports_with_as_imports. Retrieved 25/27 statements.
# Partially parsed test_sorted_imports_with_section_headings. Retrieved 26/28 statements.
# Partially parsed test_sorted_imports_with_force_sort_within_sections. Retrieved 26/28 statements.
# Partially parsed test_sorted_imports_with_lines_between_sections. Retrieved 28/30 statements.
# Partially parsed test_sorted_imports_with_ensure_newline_before_comments. Retrieved 24/26 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = -1
    var_3 = '\n'
    var_4 = {}
    var_5 = {}
    var_6 = {}
    var_7 = []
    var_8 = {}
    var_9 = {}
    var_10 = 1
    var_11 = []
    var_12 = {}
    var_13 = module_0.Config(**var_12)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'os'
    var_7 = 'sys'
    var_8 = set()
    var_9 = set()
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = {var_5: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = {}
    var_19 = {var_5: var_18}
    var_20 = [var_4]
    var_21 = {}
    var_22 = {}
    var_23 = 1
    var_24 = []
    var_25 = {}
    var_26 = module_0.Config(**var_25)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'THIRDPARTY'
    var_5 = 'from'
    var_6 = 'os'
    var_7 = 'path'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = {var_5: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = {}
    var_19 = {var_5: var_18}
    var_20 = [var_4]
    var_21 = {}
    var_22 = {}
    var_23 = 1
    var_24 = []
    var_25 = {}
    var_26 = module_0.Config(**var_25)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'os'
    var_7 = 'sys'
    var_8 = set()
    var_9 = set()
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = {var_5: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = {}
    var_19 = {var_5: var_18}
    var_20 = [var_4]
    var_21 = {}
    var_22 = {}
    var_23 = 1
    var_24 = []
    var_25 = True
    var_26 = 'combine_straight_imports'
    var_27 = {var_26: var_25}
    var_28 = module_0.Config(**var_27)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'os'
    var_7 = set()
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = {var_4: var_9}
    var_11 = 'above'
    var_12 = {}
    var_13 = {var_5: var_12}
    var_14 = 'comment'
    var_15 = [var_14]
    var_16 = {var_6: var_15}
    var_17 = {var_11: var_13, var_5: var_16}
    var_18 = {}
    var_19 = {var_5: var_18}
    var_20 = [var_4]
    var_21 = {}
    var_22 = {}
    var_23 = 1
    var_24 = []
    var_25 = {}
    var_26 = module_0.Config(**var_25)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'os'
    var_7 = 'sys'
    var_8 = set()
    var_9 = set()
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = {var_5: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = {}
    var_19 = {var_5: var_18}
    var_20 = [var_4]
    var_21 = {}
    var_22 = {}
    var_23 = 1
    var_24 = []
    var_25 = [var_6]
    var_26 = 'remove_imports'
    var_27 = {var_26: var_25}
    var_28 = module_0.Config(**var_27)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'os'
    var_7 = set()
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = {var_4: var_9}
    var_11 = 'above'
    var_12 = {}
    var_13 = {var_5: var_12}
    var_14 = {}
    var_15 = {var_11: var_13, var_5: var_14}
    var_16 = 'ospath'
    var_17 = [var_16]
    var_18 = {var_6: var_17}
    var_19 = {var_5: var_18}
    var_20 = [var_4]
    var_21 = {}
    var_22 = {}
    var_23 = 1
    var_24 = []
    var_25 = {}
    var_26 = module_0.Config(**var_25)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'os'
    var_7 = set()
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = {var_4: var_9}
    var_11 = 'above'
    var_12 = {}
    var_13 = {var_5: var_12}
    var_14 = {}
    var_15 = {var_11: var_13, var_5: var_14}
    var_16 = {}
    var_17 = {var_5: var_16}
    var_18 = [var_4]
    var_19 = {}
    var_20 = {}
    var_21 = 1
    var_22 = []
    var_23 = 'thirdparty'
    var_24 = 'Third Party Imports'
    var_25 = {var_23: var_24}
    var_26 = 'import_headings'
    var_27 = {var_26: var_25}
    var_28 = module_0.Config(**var_27)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'os'
    var_7 = 'sys'
    var_8 = set()
    var_9 = set()
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = {var_5: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = {}
    var_19 = {var_5: var_18}
    var_20 = [var_4]
    var_21 = {}
    var_22 = {}
    var_23 = 1
    var_24 = []
    var_25 = True
    var_26 = 'force_sort_within_sections'
    var_27 = {var_26: var_25}
    var_28 = module_0.Config(**var_27)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'THIRDPARTY'
    var_5 = 'FIRSTPARTY'
    var_6 = 'straight'
    var_7 = 'os'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = 'sys'
    var_12 = set()
    var_13 = {var_11: var_12}
    var_14 = {var_6: var_13}
    var_15 = {var_4: var_10, var_5: var_14}
    var_16 = 'above'
    var_17 = {}
    var_18 = {var_6: var_17}
    var_19 = {}
    var_20 = {var_16: var_18, var_6: var_19}
    var_21 = {}
    var_22 = {var_6: var_21}
    var_23 = [var_4, var_5]
    var_24 = {}
    var_25 = {}
    var_26 = 1
    var_27 = []
    var_28 = 'lines_between_sections'
    var_29 = {var_28: var_26}
    var_30 = module_0.Config(**var_29)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'os'
    var_7 = set()
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = {var_4: var_9}
    var_11 = 'above'
    var_12 = {}
    var_13 = {var_5: var_12}
    var_14 = {}
    var_15 = {var_11: var_13, var_5: var_14}
    var_16 = {}
    var_17 = {var_5: var_16}
    var_18 = [var_4]
    var_19 = {}
    var_20 = {}
    var_21 = 1
    var_22 = []
    var_23 = True
    var_24 = 'ensure_newline_before_comments'
    var_25 = {var_24: var_23}
    var_26 = module_0.Config(**var_25)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test__with_from_imports_basic_case. Retrieved 22/24 statements.
# Partially parsed test__with_from_imports_with_star. Retrieved 26/28 statements.
# Partially parsed test__with_from_imports_with_as_imports. Retrieved 25/27 statements.
# Partially parsed test__with_from_imports_remove_imports. Retrieved 23/25 statements.
# Partially parsed test__with_from_imports_force_single_line. Retrieved 23/25 statements.
# Partially parsed test__with_from_imports_with_comments. Retrieved 23/25 statements.
# Partially parsed test__with_from_imports_ignore_comments. Retrieved 24/26 statements.
# Partially parsed test__with_from_imports_multiline_reformat. Retrieved 24/27 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = {}
    var_11 = {}
    var_12 = {var_1: var_11}
    var_13 = {var_1: var_10, var_9: var_12}
    var_14 = {}
    var_15 = {var_1: var_14}
    var_16 = '\n'
    var_17 = set()
    var_18 = []
    var_19 = {}
    var_20 = module_0.Config(**var_19)
    var_21 = [var_2]
    var_22 = []
    var_23 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = '*'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'nested'
    var_9 = 'above'
    var_10 = {}
    var_11 = 'star comment'
    var_12 = {var_3: var_11}
    var_13 = {var_2: var_12}
    var_14 = {}
    var_15 = {var_1: var_14}
    var_16 = {var_1: var_10, var_8: var_13, var_9: var_15}
    var_17 = {}
    var_18 = {var_1: var_17}
    var_19 = '\n'
    var_20 = set()
    var_21 = []
    var_22 = True
    var_23 = 'combine_star'
    var_24 = {var_23: var_22}
    var_25 = module_0.Config(**var_24)
    var_26 = [var_2]
    var_27 = []
    var_28 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'above'
    var_9 = {}
    var_10 = {}
    var_11 = {var_1: var_10}
    var_12 = {var_1: var_9, var_8: var_11}
    var_13 = 'module.import1'
    var_14 = 'import1 as alias1'
    var_15 = [var_14]
    var_16 = {var_13: var_15}
    var_17 = {var_1: var_16}
    var_18 = '\n'
    var_19 = set()
    var_20 = []
    var_21 = True
    var_22 = 'combine_as_imports'
    var_23 = {var_22: var_21}
    var_24 = module_0.Config(**var_23)
    var_25 = [var_2]
    var_26 = []
    var_27 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = {}
    var_11 = {}
    var_12 = {var_1: var_11}
    var_13 = {var_1: var_10, var_9: var_12}
    var_14 = {}
    var_15 = {var_1: var_14}
    var_16 = '\n'
    var_17 = set()
    var_18 = []
    var_19 = {}
    var_20 = module_0.Config(**var_19)
    var_21 = [var_2]
    var_22 = 'module.import1'
    var_23 = [var_22]
    var_24 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'above'
    var_10 = {}
    var_11 = {}
    var_12 = {var_1: var_11}
    var_13 = {var_1: var_10, var_9: var_12}
    var_14 = {}
    var_15 = {var_1: var_14}
    var_16 = '\n'
    var_17 = set()
    var_18 = []
    var_19 = True
    var_20 = 'force_single_line'
    var_21 = {var_20: var_19}
    var_22 = module_0.Config(**var_21)
    var_23 = [var_2]
    var_24 = []
    var_25 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'above'
    var_9 = 'comment1'
    var_10 = [var_9]
    var_11 = {var_2: var_10}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = {var_1: var_11, var_8: var_13}
    var_15 = {}
    var_16 = {var_1: var_15}
    var_17 = '\n'
    var_18 = set()
    var_19 = []
    var_20 = {}
    var_21 = module_0.Config(**var_20)
    var_22 = [var_2]
    var_23 = []
    var_24 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'above'
    var_9 = 'comment1'
    var_10 = [var_9]
    var_11 = {var_2: var_10}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = {var_1: var_11, var_8: var_13}
    var_15 = {}
    var_16 = {var_1: var_15}
    var_17 = '\n'
    var_18 = set()
    var_19 = []
    var_20 = True
    var_21 = 'ignore_comments'
    var_22 = {var_21: var_20}
    var_23 = module_0.Config(**var_22)
    var_24 = [var_2]
    var_25 = []
    var_26 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'section'
    var_1 = 'from'
    var_2 = 'module'
    var_3 = 'import1'
    var_4 = 'import2'
    var_5 = 'import3'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_2: var_6}
    var_8 = {var_1: var_7}
    var_9 = {var_0: var_8}
    var_10 = 'above'
    var_11 = {}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = {var_1: var_11, var_10: var_13}
    var_15 = {}
    var_16 = {var_1: var_15}
    var_17 = '\n'
    var_18 = set()
    var_19 = []
    var_20 = 20
    var_21 = 'line_length'
    var_22 = {var_21: var_20}
    var_23 = module_0.Config(**var_22)
    var_24 = [var_2]
    var_25 = []
    var_26 = 'import'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_predicate_at_line_1. Retrieved 16/18 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'straight'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'above'
    var_4 = {}
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = {var_3: var_5, var_0: var_6}
    var_8 = {}
    var_9 = {var_0: var_8}
    var_10 = []
    var_11 = False
    var_12 = 'combine_straight_imports'
    var_13 = {var_12: var_11}
    var_14 = module_0.Config(**var_13)
    var_15 = []
    var_16 = 'straight'
    var_17 = []
    var_18 = 'import'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_sorted_imports_predicate_false. Retrieved 10/12 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = '\n'
    var_5 = 1
    var_6 = {}
    var_7 = {}
    var_8 = []
    var_9 = []
    var_10 = {}
    var_11 = module_0.Config(**var_10)



# Parsed testcases at query #31
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_with_straight_imports_combine_straight_imports. Retrieved 24/26 statements.
# Partially parsed test_with_straight_imports_combine_straight_imports_with_inline_comments. Retrieved 28/30 statements.
# Partially parsed test_with_straight_imports_combine_straight_imports_with_above_comments. Retrieved 26/28 statements.
# Partially parsed test_with_straight_imports_combine_straight_imports_with_as_imports. Retrieved 26/28 statements.
# Partially parsed test_with_straight_imports_combine_straight_imports_removed. Retrieved 23/25 statements.
# Partially parsed test_with_straight_imports_no_combine_straight_imports. Retrieved 23/25 statements.
# Partially parsed test_with_straight_imports_no_combine_straight_imports_with_inline_comments. Retrieved 27/29 statements.
# Partially parsed test_with_straight_imports_no_combine_straight_imports_with_above_comments. Retrieved 25/27 statements.
# Partially parsed test_with_straight_imports_no_combine_straight_imports_with_as_imports. Retrieved 25/27 statements.
# Partially parsed test_with_straight_imports_no_combine_straight_imports_removed. Retrieved 24/26 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'straight'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'STDLIB'
    var_4 = 'os'
    var_5 = 'sys'
    var_6 = [var_5]
    var_7 = []
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_0: var_8}
    var_10 = {var_3: var_9}
    var_11 = 'above'
    var_12 = {}
    var_13 = {var_0: var_12}
    var_14 = {}
    var_15 = {var_11: var_13, var_0: var_14}
    var_16 = []
    var_17 = True
    var_18 = False
    var_19 = '# '
    var_20 = 'combine_straight_imports'
    var_21 = 'ignore_comments'
    var_22 = 'comment_prefix'
    var_23 = {var_20: var_17, var_21: var_18, var_22: var_19}
    var_24 = module_0.Config(**var_23)
    var_25 = [var_4, var_5]
    var_26 = 'STDLIB'
    var_27 = []
    var_28 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'straight'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'STDLIB'
    var_4 = 'os'
    var_5 = 'sys'
    var_6 = [var_5]
    var_7 = []
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_0: var_8}
    var_10 = {var_3: var_9}
    var_11 = 'above'
    var_12 = {}
    var_13 = {var_0: var_12}
    var_14 = 'comment1'
    var_15 = [var_14]
    var_16 = 'comment2'
    var_17 = [var_16]
    var_18 = {var_4: var_15, var_5: var_17}
    var_19 = {var_11: var_13, var_0: var_18}
    var_20 = []
    var_21 = True
    var_22 = False
    var_23 = '# '
    var_24 = 'combine_straight_imports'
    var_25 = 'ignore_comments'
    var_26 = 'comment_prefix'
    var_27 = {var_24: var_21, var_25: var_22, var_26: var_23}
    var_28 = module_0.Config(**var_27)
    var_29 = [var_4, var_5]
    var_30 = 'STDLIB'
    var_31 = []
    var_32 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'straight'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'STDLIB'
    var_4 = 'os'
    var_5 = 'sys'
    var_6 = [var_5]
    var_7 = []
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_0: var_8}
    var_10 = {var_3: var_9}
    var_11 = 'above'
    var_12 = '# above comment'
    var_13 = [var_12]
    var_14 = {var_4: var_13}
    var_15 = {var_0: var_14}
    var_16 = {}
    var_17 = {var_11: var_15, var_0: var_16}
    var_18 = []
    var_19 = True
    var_20 = False
    var_21 = '# '
    var_22 = 'combine_straight_imports'
    var_23 = 'ignore_comments'
    var_24 = 'comment_prefix'
    var_25 = {var_22: var_19, var_23: var_20, var_24: var_21}
    var_26 = module_0.Config(**var_25)
    var_27 = [var_4, var_5]
    var_28 = 'STDLIB'
    var_29 = []
    var_30 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'straight'
    var_1 = 'os'
    var_2 = 'os_alias'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = 'STDLIB'
    var_7 = 'sys'
    var_8 = [var_7]
    var_9 = []
    var_10 = {var_1: var_8, var_7: var_9}
    var_11 = {var_0: var_10}
    var_12 = {var_6: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_0: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_0: var_16}
    var_18 = []
    var_19 = True
    var_20 = False
    var_21 = '# '
    var_22 = 'combine_straight_imports'
    var_23 = 'ignore_comments'
    var_24 = 'comment_prefix'
    var_25 = {var_22: var_19, var_23: var_20, var_24: var_21}
    var_26 = module_0.Config(**var_25)
    var_27 = [var_1, var_7]
    var_28 = 'STDLIB'
    var_29 = []
    var_30 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'straight'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'STDLIB'
    var_4 = 'os'
    var_5 = 'sys'
    var_6 = [var_5]
    var_7 = []
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_0: var_8}
    var_10 = {var_3: var_9}
    var_11 = 'above'
    var_12 = {}
    var_13 = {var_0: var_12}
    var_14 = {}
    var_15 = {var_11: var_13, var_0: var_14}
    var_16 = []
    var_17 = True
    var_18 = '# '
    var_19 = 'combine_straight_imports'
    var_20 = 'ignore_comments'
    var_21 = 'comment_prefix'
    var_22 = {var_19: var_17, var_20: var_17, var_21: var_18}
    var_23 = module_0.Config(**var_22)
    var_24 = [var_4, var_5]
    var_25 = 'STDLIB'
    var_26 = []
    var_27 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'straight'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'STDLIB'
    var_4 = 'os'
    var_5 = 'sys'
    var_6 = [var_5]
    var_7 = []
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_0: var_8}
    var_10 = {var_3: var_9}
    var_11 = 'above'
    var_12 = {}
    var_13 = {var_0: var_12}
    var_14 = {}
    var_15 = {var_11: var_13, var_0: var_14}
    var_16 = []
    var_17 = False
    var_18 = '# '
    var_19 = 'combine_straight_imports'
    var_20 = 'ignore_comments'
    var_21 = 'comment_prefix'
    var_22 = {var_19: var_17, var_20: var_17, var_21: var_18}
    var_23 = module_0.Config(**var_22)
    var_24 = [var_4, var_5]
    var_25 = 'STDLIB'
    var_26 = []
    var_27 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'straight'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'STDLIB'
    var_4 = 'os'
    var_5 = 'sys'
    var_6 = [var_5]
    var_7 = []
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_0: var_8}
    var_10 = {var_3: var_9}
    var_11 = 'above'
    var_12 = {}
    var_13 = {var_0: var_12}
    var_14 = 'comment1'
    var_15 = [var_14]
    var_16 = 'comment2'
    var_17 = [var_16]
    var_18 = {var_4: var_15, var_5: var_17}
    var_19 = {var_11: var_13, var_0: var_18}
    var_20 = []
    var_21 = False
    var_22 = '# '
    var_23 = 'combine_straight_imports'
    var_24 = 'ignore_comments'
    var_25 = 'comment_prefix'
    var_26 = {var_23: var_21, var_24: var_21, var_25: var_22}
    var_27 = module_0.Config(**var_26)
    var_28 = [var_4, var_5]
    var_29 = 'STDLIB'
    var_30 = []
    var_31 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'straight'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'STDLIB'
    var_4 = 'os'
    var_5 = 'sys'
    var_6 = [var_5]
    var_7 = []
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_0: var_8}
    var_10 = {var_3: var_9}
    var_11 = 'above'
    var_12 = '# above comment'
    var_13 = [var_12]
    var_14 = {var_4: var_13}
    var_15 = {var_0: var_14}
    var_16 = {}
    var_17 = {var_11: var_15, var_0: var_16}
    var_18 = []
    var_19 = False
    var_20 = '# '
    var_21 = 'combine_straight_imports'
    var_22 = 'ignore_comments'
    var_23 = 'comment_prefix'
    var_24 = {var_21: var_19, var_22: var_19, var_23: var_20}
    var_25 = module_0.Config(**var_24)
    var_26 = [var_4, var_5]
    var_27 = 'STDLIB'
    var_28 = []
    var_29 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'straight'
    var_1 = 'os'
    var_2 = 'os_alias'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = 'STDLIB'
    var_7 = 'sys'
    var_8 = [var_7]
    var_9 = []
    var_10 = {var_1: var_8, var_7: var_9}
    var_11 = {var_0: var_10}
    var_12 = {var_6: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_0: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_0: var_16}
    var_18 = []
    var_19 = False
    var_20 = '# '
    var_21 = 'combine_straight_imports'
    var_22 = 'ignore_comments'
    var_23 = 'comment_prefix'
    var_24 = {var_21: var_19, var_22: var_19, var_23: var_20}
    var_25 = module_0.Config(**var_24)
    var_26 = [var_1, var_7]
    var_27 = 'STDLIB'
    var_28 = []
    var_29 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'straight'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'STDLIB'
    var_4 = 'os'
    var_5 = 'sys'
    var_6 = [var_5]
    var_7 = []
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_0: var_8}
    var_10 = {var_3: var_9}
    var_11 = 'above'
    var_12 = {}
    var_13 = {var_0: var_12}
    var_14 = {}
    var_15 = {var_11: var_13, var_0: var_14}
    var_16 = []
    var_17 = False
    var_18 = True
    var_19 = '# '
    var_20 = 'combine_straight_imports'
    var_21 = 'ignore_comments'
    var_22 = 'comment_prefix'
    var_23 = {var_20: var_17, var_21: var_18, var_22: var_19}
    var_24 = module_0.Config(**var_23)
    var_25 = [var_4, var_5]
    var_26 = 'STDLIB'
    var_27 = []
    var_28 = 'import'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_star_comment_is_none. Retrieved 8/11 statements.


def test_case_0():
    var_0 = []
    var_1 = 'nested'
    var_2 = 'module'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'module'
    var_6 = 'comment1'
    var_7 = 'comment2'
    var_8 = [var_6, var_7]



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_sorted_imports_empty_parsed_content. Retrieved 21/23 statements.
# Partially parsed test_sorted_imports_no_imports. Retrieved 21/23 statements.
# Partially parsed test_sorted_imports_single_straight_import. Retrieved 27/29 statements.
# Partially parsed test_sorted_imports_multiple_straight_imports. Retrieved 29/31 statements.
# Partially parsed test_sorted_imports_single_from_import. Retrieved 29/31 statements.
# Partially parsed test_sorted_imports_multiple_from_imports. Retrieved 31/33 statements.
# Partially parsed test_sorted_imports_with_comments. Retrieved 29/31 statements.
# Partially parsed test_sorted_imports_with_as_import. Retrieved 29/31 statements.
# Partially parsed test_sorted_imports_with_remove_imports. Retrieved 31/33 statements.
# Partially parsed test_sorted_imports_with_combine_straight_imports. Retrieved 30/32 statements.
# Partially parsed test_sorted_imports_with_force_sort_within_sections. Retrieved 30/32 statements.
# Partially parsed test_sorted_imports_with_import_headings. Retrieved 31/33 statements.


def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = 0
    var_5 = '\n'
    var_6 = 'above'
    var_7 = 'straight'
    var_8 = 'from'
    var_9 = {}
    var_10 = {}
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = {}
    var_14 = {var_6: var_11, var_7: var_12, var_8: var_13}
    var_15 = {}
    var_16 = {}
    var_17 = {var_7: var_15, var_8: var_16}
    var_18 = []
    var_19 = {}
    var_20 = {}
    var_21 = []

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = 1
    var_5 = '\n'
    var_6 = 'above'
    var_7 = 'straight'
    var_8 = 'from'
    var_9 = {}
    var_10 = {}
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = {}
    var_14 = {var_6: var_11, var_7: var_12, var_8: var_13}
    var_15 = {}
    var_16 = {}
    var_17 = {var_7: var_15, var_8: var_16}
    var_18 = []
    var_19 = {}
    var_20 = {}
    var_21 = []

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 'STDLIB'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = set()
    var_7 = {var_5: var_6}
    var_8 = {}
    var_9 = {var_3: var_7, var_4: var_8}
    var_10 = {var_2: var_9}
    var_11 = 0
    var_12 = 1
    var_13 = '\n'
    var_14 = 'above'
    var_15 = {}
    var_16 = {}
    var_17 = {var_3: var_15, var_4: var_16}
    var_18 = {}
    var_19 = {}
    var_20 = {var_14: var_17, var_3: var_18, var_4: var_19}
    var_21 = {}
    var_22 = {}
    var_23 = {var_3: var_21, var_4: var_22}
    var_24 = [var_2]
    var_25 = {}
    var_26 = {}
    var_27 = []

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 'STDLIB'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = 'sys'
    var_7 = set()
    var_8 = set()
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {}
    var_11 = {var_3: var_9, var_4: var_10}
    var_12 = {var_2: var_11}
    var_13 = 0
    var_14 = 1
    var_15 = '\n'
    var_16 = 'above'
    var_17 = {}
    var_18 = {}
    var_19 = {var_3: var_17, var_4: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = {var_16: var_19, var_3: var_20, var_4: var_21}
    var_23 = {}
    var_24 = {}
    var_25 = {var_3: var_23, var_4: var_24}
    var_26 = [var_2]
    var_27 = {}
    var_28 = {}
    var_29 = []

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 'STDLIB'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = {}
    var_6 = 'os'
    var_7 = 'path'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = {var_3: var_5, var_4: var_10}
    var_12 = {var_2: var_11}
    var_13 = 0
    var_14 = 1
    var_15 = '\n'
    var_16 = 'above'
    var_17 = {}
    var_18 = {}
    var_19 = {var_3: var_17, var_4: var_18}
    var_20 = {}
    var_21 = {}
    var_22 = {var_16: var_19, var_3: var_20, var_4: var_21}
    var_23 = {}
    var_24 = {}
    var_25 = {var_3: var_23, var_4: var_24}
    var_26 = [var_2]
    var_27 = {}
    var_28 = {}
    var_29 = []

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 'STDLIB'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = {}
    var_6 = 'os'
    var_7 = 'path'
    var_8 = 'sys'
    var_9 = set()
    var_10 = set()
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {var_6: var_11}
    var_13 = {var_3: var_5, var_4: var_12}
    var_14 = {var_2: var_13}
    var_15 = 0
    var_16 = 1
    var_17 = '\n'
    var_18 = 'above'
    var_19 = {}
    var_20 = {}
    var_21 = {var_3: var_19, var_4: var_20}
    var_22 = {}
    var_23 = {}
    var_24 = {var_18: var_21, var_3: var_22, var_4: var_23}
    var_25 = {}
    var_26 = {}
    var_27 = {var_3: var_25, var_4: var_26}
    var_28 = [var_2]
    var_29 = {}
    var_30 = {}
    var_31 = []

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 'STDLIB'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = set()
    var_7 = {var_5: var_6}
    var_8 = {}
    var_9 = {var_3: var_7, var_4: var_8}
    var_10 = {var_2: var_9}
    var_11 = 0
    var_12 = 1
    var_13 = '\n'
    var_14 = 'above'
    var_15 = {}
    var_16 = {}
    var_17 = {var_3: var_15, var_4: var_16}
    var_18 = '# comment'
    var_19 = [var_18]
    var_20 = {var_5: var_19}
    var_21 = {}
    var_22 = {var_14: var_17, var_3: var_20, var_4: var_21}
    var_23 = {}
    var_24 = {}
    var_25 = {var_3: var_23, var_4: var_24}
    var_26 = [var_2]
    var_27 = {}
    var_28 = {}
    var_29 = []

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 'STDLIB'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = set()
    var_7 = {var_5: var_6}
    var_8 = {}
    var_9 = {var_3: var_7, var_4: var_8}
    var_10 = {var_2: var_9}
    var_11 = 0
    var_12 = 1
    var_13 = '\n'
    var_14 = 'above'
    var_15 = {}
    var_16 = {}
    var_17 = {var_3: var_15, var_4: var_16}
    var_18 = {}
    var_19 = {}
    var_20 = {var_14: var_17, var_3: var_18, var_4: var_19}
    var_21 = 'alias'
    var_22 = [var_21]
    var_23 = {var_5: var_22}
    var_24 = {}
    var_25 = {var_3: var_23, var_4: var_24}
    var_26 = [var_2]
    var_27 = {}
    var_28 = {}
    var_29 = []

import isort.settings as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'remove_imports'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = ''
    var_6 = [var_5]
    var_7 = 'STDLIB'
    var_8 = 'straight'
    var_9 = 'from'
    var_10 = 'sys'
    var_11 = set()
    var_12 = set()
    var_13 = {var_0: var_11, var_10: var_12}
    var_14 = {}
    var_15 = {var_8: var_13, var_9: var_14}
    var_16 = {var_7: var_15}
    var_17 = 0
    var_18 = 1
    var_19 = '\n'
    var_20 = 'above'
    var_21 = {}
    var_22 = {}
    var_23 = {var_8: var_21, var_9: var_22}
    var_24 = {}
    var_25 = {}
    var_26 = {var_20: var_23, var_8: var_24, var_9: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = {var_8: var_27, var_9: var_28}
    var_30 = [var_7]
    var_31 = {}
    var_32 = {}
    var_33 = []

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'combine_straight_imports'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = ''
    var_5 = [var_4]
    var_6 = 'STDLIB'
    var_7 = 'straight'
    var_8 = 'from'
    var_9 = 'os'
    var_10 = 'sys'
    var_11 = set()
    var_12 = set()
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = {}
    var_15 = {var_7: var_13, var_8: var_14}
    var_16 = {var_6: var_15}
    var_17 = 0
    var_18 = '\n'
    var_19 = 'above'
    var_20 = {}
    var_21 = {}
    var_22 = {var_7: var_20, var_8: var_21}
    var_23 = {}
    var_24 = {}
    var_25 = {var_19: var_22, var_7: var_23, var_8: var_24}
    var_26 = {}
    var_27 = {}
    var_28 = {var_7: var_26, var_8: var_27}
    var_29 = [var_6]
    var_30 = {}
    var_31 = {}
    var_32 = []

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'force_sort_within_sections'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = ''
    var_5 = [var_4]
    var_6 = 'STDLIB'
    var_7 = 'straight'
    var_8 = 'from'
    var_9 = 'sys'
    var_10 = 'os'
    var_11 = set()
    var_12 = set()
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = {}
    var_15 = {var_7: var_13, var_8: var_14}
    var_16 = {var_6: var_15}
    var_17 = 0
    var_18 = '\n'
    var_19 = 'above'
    var_20 = {}
    var_21 = {}
    var_22 = {var_7: var_20, var_8: var_21}
    var_23 = {}
    var_24 = {}
    var_25 = {var_19: var_22, var_7: var_23, var_8: var_24}
    var_26 = {}
    var_27 = {}
    var_28 = {var_7: var_26, var_8: var_27}
    var_29 = [var_6]
    var_30 = {}
    var_31 = {}
    var_32 = []

import isort.settings as module_0

def test_case_0():
    var_0 = 'stdlib'
    var_1 = 'Standard Library'
    var_2 = {var_0: var_1}
    var_3 = 'import_headings'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = ''
    var_7 = [var_6]
    var_8 = 'STDLIB'
    var_9 = 'straight'
    var_10 = 'from'
    var_11 = 'os'
    var_12 = set()
    var_13 = {var_11: var_12}
    var_14 = {}
    var_15 = {var_9: var_13, var_10: var_14}
    var_16 = {var_8: var_15}
    var_17 = 0
    var_18 = 1
    var_19 = '\n'
    var_20 = 'above'
    var_21 = {}
    var_22 = {}
    var_23 = {var_9: var_21, var_10: var_22}
    var_24 = {}
    var_25 = {}
    var_26 = {var_20: var_23, var_9: var_24, var_10: var_25}
    var_27 = {}
    var_28 = {}
    var_29 = {var_9: var_27, var_10: var_28}
    var_30 = [var_8]
    var_31 = {}
    var_32 = {}
    var_33 = []



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_sorted_imports_no_imports. Retrieved 5/7 statements.
# Partially parsed test_sorted_imports_single_straight_import. Retrieved 14/16 statements.
# Partially parsed test_sorted_imports_single_from_import. Retrieved 16/18 statements.
# Partially parsed test_sorted_imports_multiple_straight_imports. Retrieved 16/18 statements.
# Partially parsed test_sorted_imports_multiple_from_imports. Retrieved 20/22 statements.
# Partially parsed test_sorted_imports_with_comments. Retrieved 23/25 statements.
# Partially parsed test_sorted_imports_with_as_import. Retrieved 18/20 statements.
# Partially parsed test_sorted_imports_with_remove_imports. Retrieved 18/20 statements.
# Partially parsed test_sorted_imports_with_combine_straight_imports. Retrieved 18/20 statements.
# Partially parsed test_sorted_imports_with_force_sort_within_sections. Retrieved 18/20 statements.
# Partially parsed test_sorted_imports_with_import_headings. Retrieved 18/20 statements.
# Partially parsed test_sorted_imports_with_ensure_newline_before_comments. Retrieved 16/18 statements.
# Partially parsed test_sorted_imports_with_lines_after_imports. Retrieved 16/18 statements.
# Partially parsed test_sorted_imports_with_no_sections. Retrieved 22/24 statements.
# Partially parsed test_sorted_imports_with_from_first. Retrieved 20/22 statements.
# Partially parsed test_sorted_imports_with_star_first. Retrieved 22/24 statements.


def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = -1
    var_3 = '\n'
    var_4 = 1
    var_5 = []

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = '\n'
    var_3 = 1
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = [var_4]
    var_14 = []

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = '\n'
    var_3 = 1
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = {}
    var_8 = 'os'
    var_9 = 'path'
    var_10 = set()
    var_11 = {var_9: var_10}
    var_12 = {var_8: var_11}
    var_13 = {var_5: var_7, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = [var_4]
    var_16 = []

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = '\n'
    var_3 = 1
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'sys'
    var_8 = 'os'
    var_9 = set()
    var_10 = set()
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = {var_5: var_11, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = [var_4]
    var_16 = []

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = '\n'
    var_3 = 1
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = {}
    var_8 = 'os'
    var_9 = 'sys'
    var_10 = 'path'
    var_11 = set()
    var_12 = {var_10: var_11}
    var_13 = 'argv'
    var_14 = set()
    var_15 = {var_13: var_14}
    var_16 = {var_8: var_12, var_9: var_15}
    var_17 = {var_5: var_7, var_6: var_16}
    var_18 = {var_4: var_17}
    var_19 = [var_4]
    var_20 = []

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = '\n'
    var_3 = 1
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = [var_4]
    var_14 = 'above'
    var_15 = '# Comment above'
    var_16 = [var_15]
    var_17 = {var_7: var_16}
    var_18 = {var_5: var_17}
    var_19 = '# Inline comment'
    var_20 = [var_19]
    var_21 = {var_7: var_20}
    var_22 = {var_14: var_18, var_5: var_21}
    var_23 = []

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = '\n'
    var_3 = 1
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = [var_4]
    var_14 = 'osp'
    var_15 = [var_14]
    var_16 = {var_7: var_15}
    var_17 = {var_5: var_16}
    var_18 = []

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = '\n'
    var_3 = 1
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = 'sys'
    var_9 = set()
    var_10 = set()
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = {var_5: var_11, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = [var_4]
    var_16 = []
    var_17 = [var_7]
    var_18 = 'remove_imports'
    var_19 = {var_18: var_17}
    var_20 = module_0.Config(**var_19)

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = '\n'
    var_3 = 1
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = 'sys'
    var_9 = set()
    var_10 = set()
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = {var_5: var_11, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = [var_4]
    var_16 = []
    var_17 = True
    var_18 = 'combine_straight_imports'
    var_19 = {var_18: var_17}
    var_20 = module_0.Config(**var_19)

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = '\n'
    var_3 = 1
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'sys'
    var_8 = 'os'
    var_9 = set()
    var_10 = set()
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = {var_5: var_11, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = [var_4]
    var_16 = []
    var_17 = True
    var_18 = 'force_sort_within_sections'
    var_19 = {var_18: var_17}
    var_20 = module_0.Config(**var_19)

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = '\n'
    var_3 = 1
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = [var_4]
    var_14 = []
    var_15 = 'thirdparty'
    var_16 = 'Third Party Imports'
    var_17 = {var_15: var_16}
    var_18 = 'import_headings'
    var_19 = {var_18: var_17}
    var_20 = module_0.Config(**var_19)

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = '\n'
    var_3 = 1
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = [var_4]
    var_14 = []
    var_15 = True
    var_16 = 'ensure_newline_before_comments'
    var_17 = {var_16: var_15}
    var_18 = module_0.Config(**var_17)

import isort.settings as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 2
    var_5 = 'THIRDPARTY'
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = 'os'
    var_9 = set()
    var_10 = {var_8: var_9}
    var_11 = {}
    var_12 = {var_6: var_10, var_7: var_11}
    var_13 = {var_5: var_12}
    var_14 = [var_5]
    var_15 = []
    var_16 = 'lines_after_imports'
    var_17 = {var_16: var_4}
    var_18 = module_0.Config(**var_17)

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = '\n'
    var_3 = 1
    var_4 = 'FUTURE'
    var_5 = 'THIRDPARTY'
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = '__future__'
    var_9 = set()
    var_10 = {var_8: var_9}
    var_11 = {}
    var_12 = {var_6: var_10, var_7: var_11}
    var_13 = 'os'
    var_14 = set()
    var_15 = {var_13: var_14}
    var_16 = {}
    var_17 = {var_6: var_15, var_7: var_16}
    var_18 = {var_4: var_12, var_5: var_17}
    var_19 = [var_4, var_5]
    var_20 = []
    var_21 = True
    var_22 = 'no_sections'
    var_23 = {var_22: var_21}
    var_24 = module_0.Config(**var_23)

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = '\n'
    var_3 = 1
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = 'sys'
    var_11 = 'argv'
    var_12 = set()
    var_13 = {var_11: var_12}
    var_14 = {var_10: var_13}
    var_15 = {var_5: var_9, var_6: var_14}
    var_16 = {var_4: var_15}
    var_17 = [var_4]
    var_18 = []
    var_19 = True
    var_20 = 'from_first'
    var_21 = {var_20: var_19}
    var_22 = module_0.Config(**var_21)

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = '\n'
    var_3 = 1
    var_4 = 'THIRDPARTY'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = {}
    var_8 = 'os'
    var_9 = 'sys'
    var_10 = '*'
    var_11 = set()
    var_12 = {var_10: var_11}
    var_13 = 'argv'
    var_14 = set()
    var_15 = {var_13: var_14}
    var_16 = {var_8: var_12, var_9: var_15}
    var_17 = {var_5: var_7, var_6: var_16}
    var_18 = {var_4: var_17}
    var_19 = [var_4]
    var_20 = []
    var_21 = True
    var_22 = 'star_first'
    var_23 = {var_22: var_21}
    var_24 = module_0.Config(**var_23)



# Parsed testcases at query #36
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 1/4 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = bool(not (not var_1.no_inline_sort or (var_1.force_single_line and 'module' not in var_1.single_line_exclusions)) and (not var_1.only_sections))
    assert var_2 is True



# Parsed testcases at query #38
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_combine_straight_imports_without_as_imports. Retrieved 22/24 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'combine_straight_imports'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'straight'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = 'above'
    var_8 = {}
    var_9 = {var_4: var_8}
    var_10 = {}
    var_11 = {var_7: var_9, var_4: var_10}
    var_12 = 'STANDARD_LIBRARY'
    var_13 = 'module1'
    var_14 = 'module2'
    var_15 = []
    var_16 = []
    var_17 = {var_13: var_15, var_14: var_16}
    var_18 = {var_4: var_17}
    var_19 = {var_12: var_18}
    var_20 = []
    var_21 = [var_13, var_14]
    var_22 = 'STANDARD_LIBRARY'
    var_23 = []
    var_24 = 'import'



# Parsed testcases at query #40
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = []
    var_3 = 'no_inline_sort'
    var_4 = 'force_single_line'
    var_5 = 'single_line_exclusions'
    var_6 = 'only_sections'
    var_7 = {var_3: var_0, var_4: var_1, var_5: var_2, var_6: var_1}
    var_8 = module_0.Config(**var_7)
    var_9 = bool(not (not var_8.no_inline_sort or (var_8.force_single_line and 'module' not in var_8.single_line_exclusions)))
    assert var_9 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_with_straight_imports_predicate_false. Retrieved 22/24 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'straight'
    var_1 = 'module1'
    var_2 = 'alias1'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = 'above'
    var_7 = {}
    var_8 = {var_0: var_7}
    var_9 = {}
    var_10 = {var_6: var_8, var_0: var_9}
    var_11 = 'section'
    var_12 = []
    var_13 = {var_1: var_12}
    var_14 = {var_0: var_13}
    var_15 = {var_11: var_14}
    var_16 = []
    var_17 = True
    var_18 = 'combine_straight_imports'
    var_19 = {var_18: var_17}
    var_20 = module_0.Config(**var_19)
    var_21 = [var_1]
    var_22 = 'section'
    var_23 = []
    var_24 = 'import'



# Parsed testcases at query #42
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_sorted_imports_no_imports. Retrieved 12/14 statements.
# Partially parsed test_sorted_imports_single_straight_import. Retrieved 25/27 statements.
# Partially parsed test_sorted_imports_multiple_straight_imports. Retrieved 27/29 statements.
# Partially parsed test_sorted_imports_with_comments. Retrieved 29/31 statements.
# Partially parsed test_sorted_imports_with_as_imports. Retrieved 27/29 statements.
# Partially parsed test_sorted_imports_with_from_imports. Retrieved 28/30 statements.
# Partially parsed test_sorted_imports_with_force_sort_within_sections. Retrieved 28/30 statements.
# Partially parsed test_sorted_imports_with_no_sections. Retrieved 32/34 statements.


import isort.settings as module_0

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
    var_9 = {}
    var_10 = {}
    var_11 = []
    var_12 = {}
    var_13 = module_0.Config(**var_12)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = {}
    var_19 = {var_5: var_18}
    var_20 = [var_4]
    var_21 = 1
    var_22 = {}
    var_23 = {}
    var_24 = []
    var_25 = {}
    var_26 = module_0.Config(**var_25)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = 'sys'
    var_9 = set()
    var_10 = set()
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = {var_5: var_11, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = 'above'
    var_16 = {}
    var_17 = {var_5: var_16}
    var_18 = {}
    var_19 = {var_15: var_17, var_5: var_18}
    var_20 = {}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = 1
    var_24 = {}
    var_25 = {}
    var_26 = []
    var_27 = {}
    var_28 = module_0.Config(**var_27)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = '# Comment above'
    var_15 = [var_14]
    var_16 = {var_7: var_15}
    var_17 = {var_5: var_16}
    var_18 = '# Inline comment'
    var_19 = [var_18]
    var_20 = {var_7: var_19}
    var_21 = {var_13: var_17, var_5: var_20}
    var_22 = {}
    var_23 = {var_5: var_22}
    var_24 = [var_4]
    var_25 = 1
    var_26 = {}
    var_27 = {}
    var_28 = []
    var_29 = {}
    var_30 = module_0.Config(**var_29)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'os'
    var_8 = set()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = {var_4: var_11}
    var_13 = 'above'
    var_14 = {}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = {var_13: var_15, var_5: var_16}
    var_18 = 'path'
    var_19 = [var_18]
    var_20 = {var_7: var_19}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = 1
    var_24 = {}
    var_25 = {}
    var_26 = []
    var_27 = {}
    var_28 = module_0.Config(**var_27)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = {}
    var_8 = 'os'
    var_9 = 'path'
    var_10 = set()
    var_11 = {var_9: var_10}
    var_12 = {var_8: var_11}
    var_13 = {var_5: var_7, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = 'above'
    var_16 = {}
    var_17 = {var_6: var_16}
    var_18 = {}
    var_19 = {var_8: var_18}
    var_20 = {var_15: var_17, var_6: var_19}
    var_21 = {}
    var_22 = {var_6: var_21}
    var_23 = [var_4]
    var_24 = 1
    var_25 = {}
    var_26 = {}
    var_27 = []
    var_28 = {}
    var_29 = module_0.Config(**var_28)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'straight'
    var_6 = 'from'
    var_7 = 'sys'
    var_8 = 'os'
    var_9 = set()
    var_10 = set()
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}
    var_13 = {var_5: var_11, var_6: var_12}
    var_14 = {var_4: var_13}
    var_15 = 'above'
    var_16 = {}
    var_17 = {var_5: var_16}
    var_18 = {}
    var_19 = {var_15: var_17, var_5: var_18}
    var_20 = {}
    var_21 = {var_5: var_20}
    var_22 = [var_4]
    var_23 = 1
    var_24 = {}
    var_25 = {}
    var_26 = []
    var_27 = True
    var_28 = 'force_sort_within_sections'
    var_29 = {var_28: var_27}
    var_30 = module_0.Config(**var_29)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = 0
    var_3 = '\n'
    var_4 = 'STDLIB'
    var_5 = 'THIRDPARTY'
    var_6 = 'straight'
    var_7 = 'from'
    var_8 = 'os'
    var_9 = set()
    var_10 = {var_8: var_9}
    var_11 = {}
    var_12 = {var_6: var_10, var_7: var_11}
    var_13 = 'django'
    var_14 = set()
    var_15 = {var_13: var_14}
    var_16 = {}
    var_17 = {var_6: var_15, var_7: var_16}
    var_18 = {var_4: var_12, var_5: var_17}
    var_19 = 'above'
    var_20 = {}
    var_21 = {var_6: var_20}
    var_22 = {}
    var_23 = {var_19: var_21, var_6: var_22}
    var_24 = {}
    var_25 = {var_6: var_24}
    var_26 = [var_4, var_5]
    var_27 = 1
    var_28 = {}
    var_29 = {}
    var_30 = []
    var_31 = True
    var_32 = 'no_sections'
    var_33 = {var_32: var_31}
    var_34 = module_0.Config(**var_33)



# Parsed testcases at query #44
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #45
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 1/4 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_with_from_imports_basic_case. Retrieved 21/23 statements.
# Partially parsed test_with_from_imports_with_comments. Retrieved 23/25 statements.
# Partially parsed test_with_from_imports_remove_imports. Retrieved 24/26 statements.
# Partially parsed test_with_from_imports_with_as_imports. Retrieved 23/25 statements.
# Partially parsed test_with_from_imports_with_star. Retrieved 23/25 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = []
    var_9 = {var_2: var_8}
    var_10 = {var_1: var_9}
    var_11 = 'os.path'
    var_12 = []
    var_13 = {var_11: var_12}
    var_14 = {var_1: var_13}
    var_15 = '\n'
    var_16 = set()
    var_17 = []
    var_18 = {}
    var_19 = module_0.Config(**var_18)
    var_20 = [var_2]
    var_21 = []
    var_22 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = '# comment'
    var_9 = [var_8]
    var_10 = {var_2: var_9}
    var_11 = {var_1: var_10}
    var_12 = 'os.path'
    var_13 = []
    var_14 = {var_12: var_13}
    var_15 = {var_1: var_14}
    var_16 = '\n'
    var_17 = set()
    var_18 = []
    var_19 = False
    var_20 = 'ignore_comments'
    var_21 = {var_20: var_19}
    var_22 = module_0.Config(**var_21)
    var_23 = [var_2]
    var_24 = []
    var_25 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'sys'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = []
    var_10 = {var_2: var_9}
    var_11 = {var_1: var_10}
    var_12 = 'os.path'
    var_13 = 'os.sys'
    var_14 = []
    var_15 = []
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = {var_1: var_16}
    var_18 = '\n'
    var_19 = set()
    var_20 = []
    var_21 = {}
    var_22 = module_0.Config(**var_21)
    var_23 = [var_2]
    var_24 = [var_13]
    var_25 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = []
    var_9 = {var_2: var_8}
    var_10 = {var_1: var_9}
    var_11 = 'os.path'
    var_12 = 'path as ospath'
    var_13 = [var_12]
    var_14 = {var_11: var_13}
    var_15 = {var_1: var_14}
    var_16 = '\n'
    var_17 = set()
    var_18 = []
    var_19 = True
    var_20 = 'combine_as_imports'
    var_21 = {var_20: var_19}
    var_22 = module_0.Config(**var_21)
    var_23 = [var_2]
    var_24 = []
    var_25 = 'import'

import isort.settings as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = '*'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'nested'
    var_9 = '# all'
    var_10 = [var_9]
    var_11 = {var_3: var_10}
    var_12 = {var_2: var_11}
    var_13 = {var_8: var_12}
    var_14 = {}
    var_15 = {var_1: var_14}
    var_16 = '\n'
    var_17 = set()
    var_18 = []
    var_19 = True
    var_20 = 'combine_star'
    var_21 = {var_20: var_19}
    var_22 = module_0.Config(**var_21)
    var_23 = [var_2]
    var_24 = []
    var_25 = 'import'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_with_straight_imports_predicate. Retrieved 19/21 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'straight'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'above'
    var_4 = {}
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = {var_3: var_5, var_0: var_6}
    var_8 = 'section'
    var_9 = {}
    var_10 = {var_0: var_9}
    var_11 = {var_8: var_10}
    var_12 = []
    var_13 = True
    var_14 = 'combine_straight_imports'
    var_15 = {var_14: var_13}
    var_16 = module_0.Config(**var_15)
    var_17 = 'module1'
    var_18 = [var_17]
    var_19 = 'section'
    var_20 = []
    var_21 = 'import'



