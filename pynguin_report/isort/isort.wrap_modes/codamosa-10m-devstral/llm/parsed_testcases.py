####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '    '
    var_6 = 88
    var_7 = []
    var_8 = '\n'
    var_9 = '#'
    var_10 = False
    var_11 = module_0.vertical(var_0, var_4, var_5, var_5, var_6, var_7, var_8, var_9, var_10, var_10)
    assert var_11 == 'from module import(\n    a,\n    b,\n    c)'
    var_12 = [var_1, var_2, var_3]
    var_13 = []
    var_14 = True
    var_15 = module_0.vertical(var_0, var_12, var_5, var_5, var_6, var_13, var_8, var_9, var_14, var_10)
    assert var_15 == 'from module import(\n    a,\n    b,\n    c,)'



# Parsed testcases at query #2
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = []
    var_2 = '    '
    var_3 = ''
    var_4 = 88
    var_5 = []
    var_6 = '\n'
    var_7 = '#'
    var_8 = False
    var_9 = module_0.backslash_grid(var_0, var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8, var_8)
    assert var_9 == ''
    var_10 = 'thing'
    var_11 = [var_10]
    var_12 = []
    var_13 = module_0.backslash_grid(var_0, var_11, var_2, var_3, var_4, var_12, var_6, var_7, var_8, var_8)
    assert var_13 == 'from module import thing'
    var_14 = 'thing1'
    var_15 = 'thing2'
    var_16 = 'thing3'
    var_17 = [var_14, var_15, var_16]
    var_18 = []
    var_19 = module_0.backslash_grid(var_0, var_17, var_2, var_3, var_4, var_18, var_6, var_7, var_8, var_8)
    assert var_19 == 'from module import thing1, thing2, thing3'
    var_20 = 'thing4'
    var_21 = 'thing5'
    var_22 = [var_14, var_15, var_16, var_20, var_21]
    var_23 = 30
    var_24 = []
    var_25 = module_0.backslash_grid(var_0, var_22, var_2, var_3, var_23, var_24, var_6, var_7, var_8, var_8)
    var_26 = 'from module import thing1, thing2, thing3, \\\n    thing4, thing5'
    var_27 = [var_14, var_15]
    var_28 = 'comment1'
    var_29 = 'comment2'
    var_30 = [var_28, var_29]
    var_31 = module_0.backslash_grid(var_0, var_27, var_2, var_3, var_4, var_30, var_6, var_7, var_8, var_8)
    assert var_31 == 'from module import thing1, thing2  # comment1 comment2'
    var_32 = [var_14, var_15]
    var_33 = []
    var_34 = True
    var_35 = module_0.backslash_grid(var_0, var_32, var_2, var_3, var_4, var_33, var_6, var_7, var_34, var_8)
    assert var_35 == 'from module import thing1, thing2,'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'statement'
    var_1 = 'imports'
    var_2 = 'white_space'
    var_3 = 'indent'
    var_4 = 'line_length'
    var_5 = 'comments'
    var_6 = 'line_separator'
    var_7 = 'comment_prefix'
    var_8 = 'include_trailing_comma'
    var_9 = 'remove_comments'
    var_10 = 'from module import'
    var_11 = 'a'
    var_12 = 'b'
    var_13 = 'c'
    var_14 = [var_11, var_12, var_13]
    var_15 = '    '
    var_16 = 88
    var_17 = []
    var_18 = '\n'
    var_19 = '  #'
    var_20 = False
    var_21 = {var_0: var_10, var_1: var_14, var_2: var_15, var_3: var_15, var_4: var_16, var_5: var_17, var_6: var_18, var_7: var_19, var_8: var_20, var_9: var_20}
    var_22 = 'from module import(\n    a,\n    b,\n    c\n)'
    var_23 = 'comment1'
    var_24 = 'comment2'
    var_25 = 'from module import(# comment1\n    # comment2\n    a,\n    b,\n    c\n)'
    var_26 = 'from module import(\n    a,\n    b,\n    c,\n)'



# Parsed testcases at query #4
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = []
    var_2 = '    '
    var_3 = 88
    var_4 = []
    var_5 = '\n'
    var_6 = '#'
    var_7 = False
    var_8 = module_0.vertical_hanging_indent(var_0, var_1, var_2, var_2, var_3, var_4, var_5, var_6, var_7, var_7)
    assert var_8 == ''
    var_9 = 'single_import'
    var_10 = [var_9]
    var_11 = []
    var_12 = module_0.vertical_hanging_indent(var_0, var_10, var_2, var_2, var_3, var_11, var_5, var_6, var_7, var_7)
    assert var_12 == 'from module import(\n    single_import)'
    var_13 = 'first_import'
    var_14 = 'second_import'
    var_15 = 'third_import'
    var_16 = [var_13, var_14, var_15]
    var_17 = []
    var_18 = module_0.vertical_hanging_indent(var_0, var_16, var_2, var_2, var_3, var_17, var_5, var_6, var_7, var_7)
    assert var_18 == 'from module import(\n    first_import,\n    second_import,\n    third_import\n)'
    var_19 = [var_13, var_14]
    var_20 = []
    var_21 = True
    var_22 = module_0.vertical_hanging_indent(var_0, var_19, var_2, var_2, var_3, var_20, var_5, var_6, var_21, var_7)
    assert var_22 == 'from module import(\n    first_import,\n    second_import,\n)'
    var_23 = [var_13, var_14]
    var_24 = '# This is a comment'
    var_25 = [var_24]
    var_26 = module_0.vertical_hanging_indent(var_0, var_23, var_2, var_2, var_3, var_25, var_5, var_6, var_7, var_7)
    assert var_26 == 'from module import(# This is a comment\n    first_import,\n    second_import\n)'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'statement'
    var_1 = 'imports'
    var_2 = 'white_space'
    var_3 = 'indent'
    var_4 = 'line_length'
    var_5 = 'comments'
    var_6 = 'line_separator'
    var_7 = 'comment_prefix'
    var_8 = 'include_trailing_comma'
    var_9 = 'remove_comments'
    var_10 = 'from module import'
    var_11 = 'a'
    var_12 = 'b'
    var_13 = 'c'
    var_14 = [var_11, var_12, var_13]
    var_15 = '    '
    var_16 = 79
    var_17 = []
    var_18 = '\n'
    var_19 = '#'
    var_20 = False
    var_21 = {var_0: var_10, var_1: var_14, var_2: var_15, var_3: var_15, var_4: var_16, var_5: var_17, var_6: var_18, var_7: var_19, var_8: var_20, var_9: var_20}
    var_22 = 'from module import(\n    a,\n    b,\n    c)'
    var_23 = 'from module import(\n    a,\n    b,\n    c,)'
    var_24 = '# comment'
    var_25 = 'from module import(\n    # comment\na,\n    b,\n    c,)'
    var_26 = ''



# Parsed testcases at query #6
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = []
    var_2 = '    '
    var_3 = 88
    var_4 = []
    var_5 = '\n'
    var_6 = '#'
    var_7 = False
    var_8 = module_0.hanging_indent_with_parentheses(var_0, var_1, var_2, var_2, var_3, var_4, var_5, var_6, var_7, var_7)
    assert var_8 == ''
    var_9 = 'a'
    var_10 = [var_9]
    var_11 = []
    var_12 = module_0.hanging_indent_with_parentheses(var_0, var_10, var_2, var_2, var_3, var_11, var_5, var_6, var_7, var_7)
    assert var_12 == 'from module import(a)'
    var_13 = 'b'
    var_14 = 'c'
    var_15 = [var_9, var_13, var_14]
    var_16 = []
    var_17 = module_0.hanging_indent_with_parentheses(var_0, var_15, var_2, var_2, var_3, var_16, var_5, var_6, var_7, var_7)
    assert var_17 == 'from module import(a, b, c)'
    var_18 = 'd'
    var_19 = 'e'
    var_20 = [var_9, var_13, var_14, var_18, var_19]
    var_21 = 20
    var_22 = []
    var_23 = module_0.hanging_indent_with_parentheses(var_0, var_20, var_2, var_2, var_21, var_22, var_5, var_6, var_7, var_7)
    assert var_23 == 'from module import(\n    a, b, c,\n    d, e)'
    var_24 = [var_9, var_13, var_14]
    var_25 = []
    var_26 = True
    var_27 = module_0.hanging_indent_with_parentheses(var_0, var_24, var_2, var_2, var_3, var_25, var_5, var_6, var_26, var_7)
    assert var_27 == 'from module import(a, b, c,)'
    var_28 = [var_9, var_13, var_14]
    var_29 = 'comment'
    var_30 = [var_29]
    var_31 = module_0.hanging_indent_with_parentheses(var_0, var_28, var_2, var_2, var_3, var_30, var_5, var_6, var_7, var_7)
    assert var_31 == 'from module import(a, b, c) # comment'
    var_32 = [var_9, var_13, var_14]
    var_33 = [var_29]
    var_34 = module_0.hanging_indent_with_parentheses(var_0, var_32, var_2, var_2, var_21, var_33, var_5, var_6, var_7, var_7)
    assert var_34 == 'from module import(\n    a, b, c) # comment'
    var_35 = [var_9, var_13, var_14]
    var_36 = [var_29]
    var_37 = module_0.hanging_indent_with_parentheses(var_0, var_35, var_2, var_2, var_21, var_36, var_5, var_6, var_26, var_7)
    assert var_37 == 'from module import(\n    a, b, c,) # comment'
    var_38 = [var_9, var_13, var_14]
    var_39 = [var_29]
    var_40 = module_0.hanging_indent_with_parentheses(var_0, var_38, var_2, var_2, var_3, var_39, var_5, var_6, var_7, var_26)
    assert var_40 == 'from module import(a, b, c)'



# Parsed testcases at query #7
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = []
    var_2 = '    '
    var_3 = 88
    var_4 = []
    var_5 = '\n'
    var_6 = '#'
    var_7 = False
    var_8 = module_0.vertical_hanging_indent_bracket(var_0, var_1, var_2, var_2, var_3, var_4, var_5, var_6, var_7, var_7)
    assert var_8 == ''
    var_9 = 'a'
    var_10 = [var_9]
    var_11 = []
    var_12 = module_0.vertical_hanging_indent_bracket(var_0, var_10, var_2, var_2, var_3, var_11, var_5, var_6, var_7, var_7)
    assert var_12 == 'from module import(\n    a\n)'
    var_13 = 'b'
    var_14 = 'c'
    var_15 = [var_9, var_13, var_14]
    var_16 = []
    var_17 = True
    var_18 = module_0.vertical_hanging_indent_bracket(var_0, var_15, var_2, var_2, var_3, var_16, var_5, var_6, var_17, var_7)
    assert var_18 == 'from module import(\n    a,\n    b,\n    c,\n)'
    var_19 = [var_9, var_13]
    var_20 = '# comment'
    var_21 = [var_20]
    var_22 = module_0.vertical_hanging_indent_bracket(var_0, var_19, var_2, var_2, var_3, var_21, var_5, var_6, var_7, var_7)
    assert var_22 == 'from module import(\n# comment\n    a,\n    b\n)'
    var_23 = 'd'
    var_24 = 'e'
    var_25 = [var_9, var_13, var_14, var_23, var_24]
    var_26 = 20
    var_27 = []
    var_28 = module_0.vertical_hanging_indent_bracket(var_0, var_25, var_2, var_2, var_26, var_27, var_5, var_6, var_7, var_7)
    assert var_28 == 'from module import(\n    a,\n    b,\n    c,\n    d,\n    e\n)'



# Parsed testcases at query #8
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '    '
    var_6 = 10
    var_7 = []
    var_8 = '\n'
    var_9 = '#'
    var_10 = False
    var_11 = module_0.vertical_grid(var_0, var_4, var_5, var_5, var_6, var_7, var_8, var_9, var_10, var_10)
    assert var_11 == 'from module import(\n    a, b, c)'
    var_12 = 'very_long_name'
    var_13 = 'another_long_name'
    var_14 = [var_12, var_13]
    var_15 = 20
    var_16 = []
    var_17 = module_0.vertical_grid(var_0, var_14, var_5, var_5, var_15, var_16, var_8, var_9, var_10, var_10)
    assert var_17 == 'from module import(\n    very_long_name,\n    another_long_name)'
    var_18 = [var_1, var_2]
    var_19 = []
    var_20 = True
    var_21 = module_0.vertical_grid(var_0, var_18, var_5, var_5, var_6, var_19, var_8, var_9, var_20, var_10)
    assert var_21 == 'from module import(\n    a, b,)'



# Parsed testcases at query #9
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = []
    var_2 = '    '
    var_3 = 88
    var_4 = []
    var_5 = '\n'
    var_6 = '#'
    var_7 = False
    var_8 = module_0.vertical(var_0, var_1, var_2, var_2, var_3, var_4, var_5, var_6, var_7, var_7)
    assert var_8 == ''
    var_9 = 'a'
    var_10 = [var_9]
    var_11 = []
    var_12 = module_0.vertical(var_0, var_10, var_2, var_2, var_3, var_11, var_5, var_6, var_7, var_7)
    assert var_12 == 'from module import(\n    a)'
    var_13 = 'b'
    var_14 = 'c'
    var_15 = [var_9, var_13, var_14]
    var_16 = []
    var_17 = module_0.vertical(var_0, var_15, var_2, var_2, var_3, var_16, var_5, var_6, var_7, var_7)
    assert var_17 == 'from module import(\n    a,\n    b,\n    c)'
    var_18 = [var_9, var_13, var_14]
    var_19 = []
    var_20 = True
    var_21 = module_0.vertical(var_0, var_18, var_2, var_2, var_3, var_19, var_5, var_6, var_20, var_7)
    assert var_21 == 'from module import(\n    a,\n    b,\n    c,)'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'statement'
    var_1 = 'imports'
    var_2 = 'white_space'
    var_3 = 'indent'
    var_4 = 'line_length'
    var_5 = 'comments'
    var_6 = 'line_separator'
    var_7 = 'comment_prefix'
    var_8 = 'include_trailing_comma'
    var_9 = 'remove_comments'
    var_10 = 'from module import'
    var_11 = 'a'
    var_12 = 'b'
    var_13 = 'c'
    var_14 = [var_11, var_12, var_13]
    var_15 = '    '
    var_16 = 79
    var_17 = []
    var_18 = '\n'
    var_19 = '#'
    var_20 = False
    var_21 = {var_0: var_10, var_1: var_14, var_2: var_15, var_3: var_15, var_4: var_16, var_5: var_17, var_6: var_18, var_7: var_19, var_8: var_20, var_9: var_20}
    var_22 = 'from module import(\n    a, b, c)'
    var_23 = 'from module import(\n    a, b, c,)'
    var_24 = 'comment'
    var_25 = 'from module import(# comment\n    a, b, c)'
    var_26 = 'very_long_import_name_1'
    var_27 = 'very_long_import_name_2'
    var_28 = 'very_long_import_name_3'
    var_29 = 'from module import(\n    very_long_import_name_1,\n    very_long_import_name_2,\n    very_long_import_name_3)'
    var_30 = 'single_import'
    var_31 = 'from module import(\n    single_import)'
    var_32 = ''



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'statement'
    var_1 = 'imports'
    var_2 = 'white_space'
    var_3 = 'indent'
    var_4 = 'line_length'
    var_5 = 'comments'
    var_6 = 'line_separator'
    var_7 = 'comment_prefix'
    var_8 = 'include_trailing_comma'
    var_9 = 'remove_comments'
    var_10 = 'from module import'
    var_11 = 'a'
    var_12 = 'b'
    var_13 = 'c'
    var_14 = [var_11, var_12, var_13]
    var_15 = '    '
    var_16 = 10
    var_17 = []
    var_18 = '\n'
    var_19 = '#'
    var_20 = False
    var_21 = {var_0: var_10, var_1: var_14, var_2: var_15, var_3: var_15, var_4: var_16, var_5: var_17, var_6: var_18, var_7: var_19, var_8: var_20, var_9: var_20}
    var_22 = 'from module import(\n    a, b, c\n)'
    var_23 = 'from module import(\n    a, b, c,\n)'
    var_24 = 'comment'
    var_25 = 'from module import( # comment\n    a, b, c\n)'
    var_26 = 'bb'
    var_27 = 'ccc'
    var_28 = 'dddd'
    var_29 = [var_11, var_26, var_27, var_28]
    var_30 = []
    var_31 = {var_0: var_10, var_1: var_29, var_2: var_15, var_3: var_15, var_4: var_16, var_5: var_30, var_6: var_18, var_7: var_19, var_8: var_20, var_9: var_20}
    var_32 = 'from module import(\n    a, bb,\n    ccc,\n    dddd\n)'
    var_33 = ''



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'statement'
    var_1 = 'imports'
    var_2 = 'white_space'
    var_3 = 'indent'
    var_4 = 'line_length'
    var_5 = 'comments'
    var_6 = 'line_separator'
    var_7 = 'comment_prefix'
    var_8 = 'include_trailing_comma'
    var_9 = 'remove_comments'
    var_10 = 'from module import'
    var_11 = 'a'
    var_12 = 'b'
    var_13 = 'c'
    var_14 = [var_11, var_12, var_13]
    var_15 = '    '
    var_16 = 80
    var_17 = []
    var_18 = '\n'
    var_19 = '  #'
    var_20 = False
    var_21 = {var_0: var_10, var_1: var_14, var_2: var_15, var_3: var_15, var_4: var_16, var_5: var_17, var_6: var_18, var_7: var_19, var_8: var_20, var_9: var_20}
    var_22 = 'comment1'
    var_23 = 'comment2'
    var_24 = 'from module import a, b  # comment1 comment2\nfrom module import c'
    var_25 = 'single_import'



# Parsed testcases at query #13
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'from module import ('
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '    '
    var_6 = 88
    var_7 = []
    var_8 = '\n'
    var_9 = '#'
    var_10 = False
    var_11 = module_0.vertical_grid_grouped_no_comma(var_0, var_4, var_5, var_5, var_6, var_7, var_8, var_9, var_10, var_10)



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'statement'
    var_1 = 'imports'
    var_2 = 'white_space'
    var_3 = 'indent'
    var_4 = 'line_length'
    var_5 = 'comments'
    var_6 = 'line_separator'
    var_7 = 'comment_prefix'
    var_8 = 'include_trailing_comma'
    var_9 = 'remove_comments'
    var_10 = 'from module import'
    var_11 = 'a'
    var_12 = 'b'
    var_13 = 'c'
    var_14 = [var_11, var_12, var_13]
    var_15 = ' '
    var_16 = '    '
    var_17 = 50
    var_18 = []
    var_19 = '\n'
    var_20 = '  #'
    var_21 = False
    var_22 = {var_0: var_10, var_1: var_14, var_2: var_15, var_3: var_16, var_4: var_17, var_5: var_18, var_6: var_19, var_7: var_20, var_8: var_21, var_9: var_21}
    var_23 = 'NOQA'
    var_24 = 'some comment'
    var_25 = 'comment'



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'statement'
    var_1 = 'imports'
    var_2 = 'white_space'
    var_3 = 'indent'
    var_4 = 'line_length'
    var_5 = 'comments'
    var_6 = 'line_separator'
    var_7 = 'comment_prefix'
    var_8 = 'include_trailing_comma'
    var_9 = 'remove_comments'
    var_10 = 'from module import'
    var_11 = 'a'
    var_12 = 'b'
    var_13 = 'c'
    var_14 = [var_11, var_12, var_13]
    var_15 = '    '
    var_16 = 79
    var_17 = []
    var_18 = '\n'
    var_19 = '#'
    var_20 = False
    var_21 = {var_0: var_10, var_1: var_14, var_2: var_15, var_3: var_15, var_4: var_16, var_5: var_17, var_6: var_18, var_7: var_19, var_8: var_20, var_9: var_20}
    var_22 = '# comment'



# Parsed testcases at query #16
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = []
    var_2 = '    '
    var_3 = 88
    var_4 = []
    var_5 = '\n'
    var_6 = '#'
    var_7 = False
    var_8 = module_0.grid(var_0, var_1, var_2, var_2, var_3, var_4, var_5, var_6, var_7, var_7)
    assert var_8 == ''
    var_9 = 'a'
    var_10 = [var_9]
    var_11 = []
    var_12 = module_0.grid(var_0, var_10, var_2, var_2, var_3, var_11, var_5, var_6, var_7, var_7)
    assert var_12 == 'from module import(a)'
    var_13 = 'b'
    var_14 = 'c'
    var_15 = [var_9, var_13, var_14]
    var_16 = []
    var_17 = module_0.grid(var_0, var_15, var_2, var_2, var_3, var_16, var_5, var_6, var_7, var_7)
    assert var_17 == 'from module import(a, b, c)'
    var_18 = 'very_long_import_name'
    var_19 = 'another_long_import'
    var_20 = 'short'
    var_21 = [var_18, var_19, var_20]
    var_22 = 30
    var_23 = []
    var_24 = module_0.grid(var_0, var_21, var_2, var_2, var_22, var_23, var_5, var_6, var_7, var_7)
    var_25 = 'from module import(very_long_import_name,\n    another_long_import,\n    short)'
    var_26 = [var_9, var_13]
    var_27 = 'comment1'
    var_28 = 'comment2'
    var_29 = [var_27, var_28]
    var_30 = module_0.grid(var_0, var_26, var_2, var_2, var_3, var_29, var_5, var_6, var_7, var_7)
    var_31 = 'from module import(a, # comment1\n    b)'
    var_32 = [var_9, var_13]
    var_33 = []
    var_34 = True
    var_35 = module_0.grid(var_0, var_32, var_2, var_2, var_3, var_33, var_5, var_6, var_34, var_7)
    assert var_35 == 'from module import(a, b,)'



# Parsed testcases at query #17
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'from module import ('
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '    '
    var_6 = 88
    var_7 = []
    var_8 = '\n'
    var_9 = '#'
    var_10 = False
    var_11 = module_0.vertical_grid_grouped_no_comma(var_0, var_4, var_5, var_5, var_6, var_7, var_8, var_9, var_10, var_10)



# Parsed testcases at query #18
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = []
    var_2 = '    '
    var_3 = 88
    var_4 = []
    var_5 = '\n'
    var_6 = '#'
    var_7 = False
    var_8 = module_0.hanging_indent_with_parentheses(var_0, var_1, var_2, var_2, var_3, var_4, var_5, var_6, var_7, var_7)
    assert var_8 == ''
    var_9 = 'something'
    var_10 = [var_9]
    var_11 = []
    var_12 = module_0.hanging_indent_with_parentheses(var_0, var_10, var_2, var_2, var_3, var_11, var_5, var_6, var_7, var_7)
    assert var_12 == 'from module import(something)'
    var_13 = 'another'
    var_14 = 'thing'
    var_15 = [var_9, var_13, var_14]
    var_16 = []
    var_17 = module_0.hanging_indent_with_parentheses(var_0, var_15, var_2, var_2, var_3, var_16, var_5, var_6, var_7, var_7)
    assert var_17 == 'from module import(something, another, thing)'
    var_18 = 'something_very_long'
    var_19 = 'another_long_import'
    var_20 = [var_18, var_19, var_14]
    var_21 = 30
    var_22 = []
    var_23 = module_0.hanging_indent_with_parentheses(var_0, var_20, var_2, var_2, var_21, var_22, var_5, var_6, var_7, var_7)
    var_24 = 'from module import(\n    something_very_long,\n    another_long_import,\n    thing)'
    var_25 = [var_9, var_13]
    var_26 = []
    var_27 = True
    var_28 = module_0.hanging_indent_with_parentheses(var_0, var_25, var_2, var_2, var_3, var_26, var_5, var_6, var_27, var_7)
    assert var_28 == 'from module import(something, another,)'
    var_29 = [var_9, var_13]
    var_30 = '# comment'
    var_31 = [var_30]
    var_32 = module_0.hanging_indent_with_parentheses(var_0, var_29, var_2, var_2, var_3, var_31, var_5, var_6, var_7, var_7)
    assert var_32 == 'from module import(something, another) # comment'
    var_33 = [var_18, var_19]
    var_34 = [var_30]
    var_35 = module_0.hanging_indent_with_parentheses(var_0, var_33, var_2, var_2, var_21, var_34, var_5, var_6, var_7, var_7)
    var_36 = 'from module import(\n    something_very_long,\n    another_long_import\n    ) # comment'
    var_37 = [var_9, var_13]
    var_38 = [var_30]
    var_39 = module_0.hanging_indent_with_parentheses(var_0, var_37, var_2, var_2, var_3, var_38, var_5, var_6, var_7, var_27)
    assert var_39 == 'from module import(something, another)'



# Parsed testcases at query #19
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = ' '
    var_6 = '    '
    var_7 = 88
    var_8 = []
    var_9 = '\n'
    var_10 = '#'
    var_11 = False
    var_12 = module_0.vertical_grid(var_0, var_4, var_5, var_6, var_7, var_8, var_9, var_10, var_11, var_11)
    assert var_12 == 'from module import(\n    a, b, c)'
    var_13 = [var_1, var_2, var_3]
    var_14 = []
    var_15 = True
    var_16 = module_0.vertical_grid(var_0, var_13, var_5, var_6, var_7, var_14, var_9, var_10, var_15, var_11)
    assert var_16 == 'from module import(\n    a, b, c,)'
    var_17 = ','
    var_18 = [var_1, var_2, var_3]
    var_19 = 'comment'
    var_20 = [var_19]
    var_21 = module_0.vertical_grid(var_0, var_18, var_5, var_6, var_7, var_20, var_9, var_10, var_11, var_11)
    assert var_21 == 'from module import( # comment\n    a, b, c)'
    var_22 = 'very_long_name_a'
    var_23 = 'very_long_name_b'
    var_24 = 'very_long_name_c'
    var_25 = [var_22, var_23, var_24]
    var_26 = 30
    var_27 = []
    var_28 = module_0.vertical_grid(var_0, var_25, var_5, var_6, var_26, var_27, var_9, var_10, var_11, var_11)
    assert var_28 == 'from module import(\n    very_long_name_a,\n    very_long_name_b,\n    very_long_name_c)'
    var_29 = []
    var_30 = []
    var_31 = module_0.vertical_grid(var_0, var_29, var_5, var_6, var_7, var_30, var_9, var_10, var_11, var_11)
    assert var_31 == ''
    var_32 = [var_1]
    var_33 = []
    var_34 = module_0.vertical_grid(var_0, var_32, var_5, var_6, var_7, var_33, var_9, var_10, var_11, var_11)
    assert var_34 == 'from module import(\n    a)'



# Parsed testcases at query #20
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'from module import ('
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '    '
    var_6 = 10
    var_7 = []
    var_8 = '\n'
    var_9 = '#'
    var_10 = False
    var_11 = module_0.backslash_grid(var_0, var_4, var_5, var_5, var_6, var_7, var_8, var_9, var_10, var_10)
    var_12 = 'from module import (\n    a, b, c)'
    var_13 = 'very_long_import_name'
    var_14 = 'another_long_one'
    var_15 = 'short'
    var_16 = [var_13, var_14, var_15]
    var_17 = 20
    var_18 = []
    var_19 = module_0.backslash_grid(var_0, var_16, var_5, var_5, var_17, var_18, var_8, var_9, var_10, var_10)
    var_20 = 'from module import (\n    very_long_import_name, \\\n    another_long_one, \\\n    short'
    var_21 = [var_1, var_2]
    var_22 = 'comment1'
    var_23 = 'comment2'
    var_24 = [var_22, var_23]
    var_25 = module_0.backslash_grid(var_0, var_21, var_5, var_5, var_17, var_24, var_8, var_9, var_10, var_10)
    var_26 = 'from module import (\n    a, b  # comment1 comment2'
    var_27 = [var_1, var_2]
    var_28 = []
    var_29 = True
    var_30 = module_0.backslash_grid(var_0, var_27, var_5, var_5, var_17, var_28, var_8, var_9, var_29, var_10)
    var_31 = 'from module import (\n    a, b,'
    var_32 = []
    var_33 = []
    var_34 = module_0.backslash_grid(var_0, var_32, var_5, var_5, var_17, var_33, var_8, var_9, var_10, var_10)
    var_35 = ''



# Parsed testcases at query #21
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'from x import'
    var_1 = []
    var_2 = '    '
    var_3 = 88
    var_4 = []
    var_5 = '\n'
    var_6 = '#'
    var_7 = False
    var_8 = module_0.grid(var_0, var_1, var_2, var_2, var_3, var_4, var_5, var_6, var_7, var_7)
    assert var_8 == ''
    var_9 = 'a'
    var_10 = [var_9]
    var_11 = []
    var_12 = module_0.grid(var_0, var_10, var_2, var_2, var_3, var_11, var_5, var_6, var_7, var_7)
    assert var_12 == 'from x import(a)'
    var_13 = 'b'
    var_14 = 'c'
    var_15 = [var_9, var_13, var_14]
    var_16 = []
    var_17 = module_0.grid(var_0, var_15, var_2, var_2, var_3, var_16, var_5, var_6, var_7, var_7)
    assert var_17 == 'from x import(a, b, c)'
    var_18 = 'd'
    var_19 = 'e'
    var_20 = 'f'
    var_21 = [var_9, var_13, var_14, var_18, var_19, var_20]
    var_22 = 20
    var_23 = []
    var_24 = module_0.grid(var_0, var_21, var_2, var_2, var_22, var_23, var_5, var_6, var_7, var_7)
    assert var_24 == 'from x import(a, b,\n    c, d,\n    e, f)'
    var_25 = [var_9, var_13, var_14]
    var_26 = 'comment'
    var_27 = [var_26]
    var_28 = module_0.grid(var_0, var_25, var_2, var_2, var_3, var_27, var_5, var_6, var_7, var_7)
    assert var_28 == 'from x import(a, b, c) # comment'
    var_29 = [var_9, var_13, var_14]
    var_30 = []
    var_31 = True
    var_32 = module_0.grid(var_0, var_29, var_2, var_2, var_3, var_30, var_5, var_6, var_31, var_7)
    assert var_32 == 'from x import(a, b, c,)'



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'statement'
    var_1 = 'imports'
    var_2 = 'white_space'
    var_3 = 'indent'
    var_4 = 'line_length'
    var_5 = 'comments'
    var_6 = 'line_separator'
    var_7 = 'comment_prefix'
    var_8 = 'include_trailing_comma'
    var_9 = 'remove_comments'
    var_10 = 'from module import'
    var_11 = 'a'
    var_12 = 'b'
    var_13 = 'c'
    var_14 = [var_11, var_12, var_13]
    var_15 = '    '
    var_16 = 79
    var_17 = []
    var_18 = '\n'
    var_19 = '#'
    var_20 = False
    var_21 = {var_0: var_10, var_1: var_14, var_2: var_15, var_3: var_15, var_4: var_16, var_5: var_17, var_6: var_18, var_7: var_19, var_8: var_20, var_9: var_20}
    var_22 = 'from module import(\n    a,\n    b,\n    c\n    )'
    var_23 = [var_11, var_12, var_13]
    var_24 = '# comment'
    var_25 = [var_24]
    var_26 = {var_0: var_10, var_1: var_23, var_2: var_15, var_3: var_15, var_4: var_16, var_5: var_25, var_6: var_18, var_7: var_19, var_8: var_20, var_9: var_20}
    var_27 = 'from module import(# comment\n    a,\n    b,\n    c\n    )'
    var_28 = [var_11, var_12, var_13]
    var_29 = []
    var_30 = True
    var_31 = {var_0: var_10, var_1: var_28, var_2: var_15, var_3: var_15, var_4: var_16, var_5: var_29, var_6: var_18, var_7: var_19, var_8: var_30, var_9: var_20}
    var_32 = 'from module import(\n    a,\n    b,\n    c,\n    )'
    var_33 = []
    var_34 = []
    var_35 = {var_0: var_10, var_1: var_33, var_2: var_15, var_3: var_15, var_4: var_16, var_5: var_34, var_6: var_18, var_7: var_19, var_8: var_20, var_9: var_20}
    var_36 = ''



# Parsed testcases at query #23
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = []
    var_2 = '    '
    var_3 = 88
    var_4 = []
    var_5 = '\n'
    var_6 = '#'
    var_7 = False
    var_8 = module_0.noqa(var_0, var_1, var_2, var_2, var_3, var_4, var_5, var_6, var_7, var_7)
    assert var_8 == 'from module import'
    var_9 = 'A'
    var_10 = [var_9]
    var_11 = []
    var_12 = module_0.noqa(var_0, var_10, var_2, var_2, var_3, var_11, var_5, var_6, var_7, var_7)
    assert var_12 == 'from module import A'
    var_13 = 'B'
    var_14 = 'C'
    var_15 = [var_9, var_13, var_14]
    var_16 = []
    var_17 = module_0.noqa(var_0, var_15, var_2, var_2, var_3, var_16, var_5, var_6, var_7, var_7)
    assert var_17 == 'from module import A, B, C'
    var_18 = [var_9, var_13, var_14]
    var_19 = 'Comment'
    var_20 = [var_19]
    var_21 = module_0.noqa(var_0, var_18, var_2, var_2, var_3, var_20, var_5, var_6, var_7, var_7)
    assert var_21 == 'from module import A, B, C # Comment'
    var_22 = [var_9, var_13, var_14]
    var_23 = 20
    var_24 = [var_19]
    var_25 = module_0.noqa(var_0, var_22, var_2, var_2, var_23, var_24, var_5, var_6, var_7, var_7)
    assert var_25 == 'from module import A, B, C # NOQA Comment'
    var_26 = [var_9, var_13, var_14]
    var_27 = 'NOQA'
    var_28 = [var_27]
    var_29 = module_0.noqa(var_0, var_26, var_2, var_2, var_23, var_28, var_5, var_6, var_7, var_7)
    assert var_29 == 'from module import A, B, C # NOQA'
    var_30 = 'D'
    var_31 = 'E'
    var_32 = [var_9, var_13, var_14, var_30, var_31]
    var_33 = []
    var_34 = module_0.noqa(var_0, var_32, var_2, var_2, var_23, var_33, var_5, var_6, var_7, var_7)
    assert var_34 == 'from module import A, B, C, D, E # NOQA'



# Parsed testcases at query #24
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'from module import ('
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '    '
    var_6 = 88
    var_7 = []
    var_8 = '\n'
    var_9 = '#'
    var_10 = False
    var_11 = module_0.vertical_grid_grouped_no_comma(var_0, var_4, var_5, var_5, var_6, var_7, var_8, var_9, var_10, var_10)



# Parsed testcases at query #25
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '    '
    var_6 = 88
    var_7 = []
    var_8 = '\n'
    var_9 = '#'
    var_10 = False
    var_11 = module_0.vertical_grid_grouped_no_comma(var_0, var_4, var_5, var_5, var_6, var_7, var_8, var_9, var_10, var_10)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '    '
    var_6 = 80
    var_7 = []
    var_8 = '\n'
    var_9 = '#'
    var_10 = False
    var_11 = module_0.vertical_grid_grouped(var_0, var_4, var_5, var_5, var_6, var_7, var_8, var_9, var_10, var_10)
    var_12 = 'from module import(\n    a, b, c\n)'
    var_13 = [var_1, var_2, var_3]
    var_14 = []
    var_15 = True
    var_16 = module_0.vertical_grid_grouped(var_0, var_13, var_5, var_5, var_6, var_14, var_8, var_9, var_15, var_10)
    var_17 = 'from module import(\n    a, b, c,\n)'
    var_18 = [var_1, var_2, var_3]
    var_19 = '# comment'
    var_20 = [var_19]
    var_21 = module_0.vertical_grid_grouped(var_0, var_18, var_5, var_5, var_6, var_20, var_8, var_9, var_10, var_10)
    var_22 = 'from module import( # comment\n    a, b, c\n)'
    var_23 = 'very_long_import_name'
    var_24 = 'another_long_import'
    var_25 = 'short'
    var_26 = [var_23, var_24, var_25]
    var_27 = 30
    var_28 = []
    var_29 = module_0.vertical_grid_grouped(var_0, var_26, var_5, var_5, var_27, var_28, var_8, var_9, var_10, var_10)
    var_30 = 'from module import(\n    very_long_import_name,\n    another_long_import,\n    short\n)'
    var_31 = []
    var_32 = []
    var_33 = module_0.vertical_grid_grouped(var_0, var_31, var_5, var_5, var_6, var_32, var_8, var_9, var_10, var_10)
    var_34 = ''



# Parsed testcases at query #2
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = []
    var_2 = '    '
    var_3 = 88
    var_4 = []
    var_5 = '\n'
    var_6 = '#'
    var_7 = False
    var_8 = module_0.vertical_prefix_from_module_import(var_0, var_1, var_2, var_2, var_3, var_4, var_5, var_6, var_7, var_7)
    assert var_8 == ''
    var_9 = 'single_import'
    var_10 = [var_9]
    var_11 = []
    var_12 = module_0.vertical_prefix_from_module_import(var_0, var_10, var_2, var_2, var_3, var_11, var_5, var_6, var_7, var_7)
    assert var_12 == 'from module import single_import'
    var_13 = 'first_import'
    var_14 = 'second_import'
    var_15 = 'third_import'
    var_16 = [var_13, var_14, var_15]
    var_17 = []
    var_18 = module_0.vertical_prefix_from_module_import(var_0, var_16, var_2, var_2, var_3, var_17, var_5, var_6, var_7, var_7)
    assert var_18 == 'from module import first_import, second_import, third_import'
    var_19 = [var_13, var_14, var_15]
    var_20 = '# comment1'
    var_21 = '# comment2'
    var_22 = [var_20, var_21]
    var_23 = module_0.vertical_prefix_from_module_import(var_0, var_19, var_2, var_2, var_3, var_22, var_5, var_6, var_7, var_7)
    assert var_23 == 'from module import first_import, second_import, third_import # comment1 # comment2'
    var_24 = [var_13, var_14, var_15]
    var_25 = 30
    var_26 = []
    var_27 = module_0.vertical_prefix_from_module_import(var_0, var_24, var_2, var_2, var_25, var_26, var_5, var_6, var_7, var_7)
    assert var_27 == 'from module import first_import\nfrom module import second_import, third_import'
    var_28 = [var_13, var_14, var_15]
    var_29 = [var_20, var_21]
    var_30 = module_0.vertical_prefix_from_module_import(var_0, var_28, var_2, var_2, var_25, var_29, var_5, var_6, var_7, var_7)
    assert var_30 == 'from module import first_import # comment1 # comment2\nfrom module import second_import, third_import'



# Parsed testcases at query #3
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = []
    var_2 = '    '
    var_3 = 88
    var_4 = []
    var_5 = '\n'
    var_6 = '#'
    var_7 = False
    var_8 = module_0.vertical_hanging_indent(var_0, var_1, var_2, var_2, var_3, var_4, var_5, var_6, var_7, var_7)
    assert var_8 == ''
    var_9 = 'a'
    var_10 = [var_9]
    var_11 = []
    var_12 = module_0.vertical_hanging_indent(var_0, var_10, var_2, var_2, var_3, var_11, var_5, var_6, var_7, var_7)
    assert var_12 == 'from module import(\n    a)'
    var_13 = 'b'
    var_14 = 'c'
    var_15 = [var_9, var_13, var_14]
    var_16 = []
    var_17 = module_0.vertical_hanging_indent(var_0, var_15, var_2, var_2, var_3, var_16, var_5, var_6, var_7, var_7)
    assert var_17 == 'from module import(\n    a,\n    b,\n    c)'
    var_18 = [var_9, var_13, var_14]
    var_19 = []
    var_20 = True
    var_21 = module_0.vertical_hanging_indent(var_0, var_18, var_2, var_2, var_3, var_19, var_5, var_6, var_20, var_7)
    assert var_21 == 'from module import(\n    a,\n    b,\n    c,)'
    var_22 = [var_9, var_13, var_14]
    var_23 = 'comment'
    var_24 = [var_23]
    var_25 = module_0.vertical_hanging_indent(var_0, var_22, var_2, var_2, var_3, var_24, var_5, var_6, var_7, var_7)
    assert var_25 == 'from module import(# comment\n    a,\n    b,\n    c)'
    var_26 = [var_9, var_13, var_14]
    var_27 = 10
    var_28 = []
    var_29 = module_0.vertical_hanging_indent(var_0, var_26, var_2, var_2, var_27, var_28, var_5, var_6, var_7, var_7)
    assert var_29 == 'from module import(\n    a,\n    b,\n    c)'



# Parsed testcases at query #4
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '    '
    var_6 = 79
    var_7 = []
    var_8 = '\n'
    var_9 = '#'
    var_10 = False
    var_11 = module_0.vertical_grid(var_0, var_4, var_5, var_5, var_6, var_7, var_8, var_9, var_10, var_10)
    assert var_11 == 'from module import(\n    a, b, c)'
    var_12 = [var_1, var_2, var_3]
    var_13 = []
    var_14 = True
    var_15 = module_0.vertical_grid(var_0, var_12, var_5, var_5, var_6, var_13, var_8, var_9, var_14, var_10)
    assert var_15 == 'from module import(\n    a, b, c,)'
    var_16 = ','
    var_17 = [var_1, var_2, var_3]
    var_18 = '# comment'
    var_19 = [var_18]
    var_20 = module_0.vertical_grid(var_0, var_17, var_5, var_5, var_6, var_19, var_8, var_9, var_10, var_10)
    assert var_20 == 'from module import( # comment\n    a, b, c)'
    var_21 = 'very_long_name_a'
    var_22 = 'very_long_name_b'
    var_23 = 'very_long_name_c'
    var_24 = [var_21, var_22, var_23]
    var_25 = 30
    var_26 = []
    var_27 = module_0.vertical_grid(var_0, var_24, var_5, var_5, var_25, var_26, var_8, var_9, var_10, var_10)
    assert var_27 == 'from module import(\n    very_long_name_a,\n    very_long_name_b,\n    very_long_name_c)'
    var_28 = [var_1]
    var_29 = []
    var_30 = module_0.vertical_grid(var_0, var_28, var_5, var_5, var_6, var_29, var_8, var_9, var_10, var_10)
    assert var_30 == 'from module import(\n    a)'
    var_31 = []
    var_32 = []
    var_33 = module_0.vertical_grid(var_0, var_31, var_5, var_5, var_6, var_32, var_8, var_9, var_10, var_10)
    assert var_33 == ''



# Parsed testcases at query #5
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'A'
    var_2 = [var_1]
    var_3 = ' '
    var_4 = '    '
    var_5 = 88
    var_6 = []
    var_7 = '\n'
    var_8 = '#'
    var_9 = False
    var_10 = module_0.vertical_prefix_from_module_import(var_0, var_2, var_3, var_4, var_5, var_6, var_7, var_8, var_9, var_9)
    assert var_10 == 'from module import A'
    var_11 = 'B'
    var_12 = 'C'
    var_13 = [var_1, var_11, var_12]
    var_14 = []
    var_15 = module_0.vertical_prefix_from_module_import(var_0, var_13, var_3, var_4, var_5, var_14, var_7, var_8, var_9, var_9)
    assert var_15 == 'from module import A, B, C'
    var_16 = [var_1, var_11, var_12]
    var_17 = 20
    var_18 = 'comment1'
    var_19 = 'comment2'
    var_20 = [var_18, var_19]
    var_21 = module_0.vertical_prefix_from_module_import(var_0, var_16, var_3, var_4, var_17, var_20, var_7, var_8, var_9, var_9)
    assert var_21 == 'from module import A, B, C  # comment1 comment2'
    var_22 = [var_1, var_11, var_12]
    var_23 = 15
    var_24 = [var_18, var_19]
    var_25 = module_0.vertical_prefix_from_module_import(var_0, var_22, var_3, var_4, var_23, var_24, var_7, var_8, var_9, var_9)
    assert var_25 == 'from module import A\nfrom module import B, C  # comment1 comment2'
    var_26 = []
    var_27 = []
    var_28 = module_0.vertical_prefix_from_module_import(var_0, var_26, var_3, var_4, var_5, var_27, var_7, var_8, var_9, var_9)
    assert var_28 == ''
    var_29 = [var_1]
    var_30 = [var_18]
    var_31 = module_0.vertical_prefix_from_module_import(var_0, var_29, var_3, var_4, var_5, var_30, var_7, var_8, var_9, var_9)
    assert var_31 == 'from module import A  # comment1'
    var_32 = [var_1, var_11, var_12]
    var_33 = [var_18, var_19]
    var_34 = True
    var_35 = module_0.vertical_prefix_from_module_import(var_0, var_32, var_3, var_4, var_23, var_33, var_7, var_8, var_9, var_34)
    assert var_35 == 'from module import A\nfrom module import B, C'



# Parsed testcases at query #6
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = []
    var_2 = '    '
    var_3 = 88
    var_4 = []
    var_5 = '\n'
    var_6 = '#'
    var_7 = False
    var_8 = module_0.vertical(var_0, var_1, var_2, var_2, var_3, var_4, var_5, var_6, var_7, var_7)
    assert var_8 == ''
    var_9 = 'a'
    var_10 = [var_9]
    var_11 = []
    var_12 = module_0.vertical(var_0, var_10, var_2, var_2, var_3, var_11, var_5, var_6, var_7, var_7)
    assert var_12 == 'from module import(\n    a)'
    var_13 = 'b'
    var_14 = 'c'
    var_15 = [var_9, var_13, var_14]
    var_16 = []
    var_17 = module_0.vertical(var_0, var_15, var_2, var_2, var_3, var_16, var_5, var_6, var_7, var_7)
    assert var_17 == 'from module import(\n    a,\n    b,\n    c)'
    var_18 = [var_9, var_13, var_14]
    var_19 = []
    var_20 = True
    var_21 = module_0.vertical(var_0, var_18, var_2, var_2, var_3, var_19, var_5, var_6, var_20, var_7)
    assert var_21 == 'from module import(\n    a,\n    b,\n    c,)'



# Parsed testcases at query #7
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '    '
    var_6 = 80
    var_7 = []
    var_8 = '\n'
    var_9 = '#'
    var_10 = False
    var_11 = module_0.hanging_indent_with_parentheses(var_0, var_4, var_5, var_5, var_6, var_7, var_8, var_9, var_10, var_10)
    assert var_11 == 'from module import(a, b, c)'
    var_12 = 'very_long_module_name_a'
    var_13 = 'very_long_module_name_b'
    var_14 = 'very_long_module_name_c'
    var_15 = [var_12, var_13, var_14]
    var_16 = 30
    var_17 = []
    var_18 = module_0.hanging_indent_with_parentheses(var_0, var_15, var_5, var_5, var_16, var_17, var_8, var_9, var_10, var_10)
    var_19 = 'from module import(\n    very_long_module_name_a,\n    very_long_module_name_b,\n    very_long_module_name_c)'
    var_20 = [var_1, var_2, var_3]
    var_21 = []
    var_22 = True
    var_23 = module_0.hanging_indent_with_parentheses(var_0, var_20, var_5, var_5, var_6, var_21, var_8, var_9, var_22, var_10)
    assert var_23 == 'from module import(a, b, c,)'
    var_24 = [var_1, var_2, var_3]
    var_25 = '# comment'
    var_26 = [var_25]
    var_27 = module_0.hanging_indent_with_parentheses(var_0, var_24, var_5, var_5, var_6, var_26, var_8, var_9, var_10, var_10)
    var_28 = 'from module import(a, b, c) # comment'
    var_29 = []
    var_30 = []
    var_31 = module_0.hanging_indent_with_parentheses(var_0, var_29, var_5, var_5, var_6, var_30, var_8, var_9, var_10, var_10)
    assert var_31 == ''
    var_32 = [var_1]
    var_33 = []
    var_34 = module_0.hanging_indent_with_parentheses(var_0, var_32, var_5, var_5, var_6, var_33, var_8, var_9, var_10, var_10)
    assert var_34 == 'from module import(a)'
    var_35 = [var_1, var_13, var_3]
    var_36 = []
    var_37 = module_0.hanging_indent_with_parentheses(var_0, var_35, var_5, var_5, var_16, var_36, var_8, var_9, var_10, var_10)
    var_38 = 'from module import(\n    a, very_long_module_name_b,\n    c)'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'statement'
    var_1 = 'imports'
    var_2 = 'white_space'
    var_3 = 'indent'
    var_4 = 'line_length'
    var_5 = 'comments'
    var_6 = 'line_separator'
    var_7 = 'comment_prefix'
    var_8 = 'include_trailing_comma'
    var_9 = 'remove_comments'
    var_10 = 'from module import'
    var_11 = 'a'
    var_12 = 'b'
    var_13 = 'c'
    var_14 = [var_11, var_12, var_13]
    var_15 = ' '
    var_16 = '    '
    var_17 = 88
    var_18 = []
    var_19 = '\n'
    var_20 = '#'
    var_21 = False
    var_22 = {var_0: var_10, var_1: var_14, var_2: var_15, var_3: var_16, var_4: var_17, var_5: var_18, var_6: var_19, var_7: var_20, var_8: var_21, var_9: var_21}
    var_23 = 'comment'
    var_24 = 'very_long_import_name_a'
    var_25 = 'very_long_import_name_b'
    var_26 = 'very_long_import_name_c'
    var_27 = [var_24, var_25, var_26]
    var_28 = 30
    var_29 = []
    var_30 = {var_0: var_10, var_1: var_27, var_2: var_15, var_3: var_16, var_4: var_28, var_5: var_29, var_6: var_19, var_7: var_20, var_8: var_21, var_9: var_21}
    var_31 = [var_11, var_12]
    var_32 = []
    var_33 = True
    var_34 = {var_0: var_10, var_1: var_31, var_2: var_15, var_3: var_16, var_4: var_17, var_5: var_32, var_6: var_19, var_7: var_20, var_8: var_33, var_9: var_21}
    var_35 = [var_11, var_12]
    var_36 = [var_23]
    var_37 = {var_0: var_10, var_1: var_35, var_2: var_15, var_3: var_16, var_4: var_17, var_5: var_36, var_6: var_19, var_7: var_20, var_8: var_21, var_9: var_33}



# Parsed testcases at query #9
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '    '
    var_6 = 79
    var_7 = []
    var_8 = '\n'
    var_9 = '#'
    var_10 = False
    var_11 = module_0.vertical_prefix_from_module_import(var_0, var_4, var_5, var_5, var_6, var_7, var_8, var_9, var_10, var_10)
    assert var_11 == 'from module import a, b, c'
    var_12 = [var_1, var_2, var_3]
    var_13 = 'comment'
    var_14 = [var_13]
    var_15 = module_0.vertical_prefix_from_module_import(var_0, var_12, var_5, var_5, var_6, var_14, var_8, var_9, var_10, var_10)
    assert var_15 == 'from module import a, b, c # comment'
    var_16 = [var_1, var_2, var_3]
    var_17 = 20
    var_18 = []
    var_19 = module_0.vertical_prefix_from_module_import(var_0, var_16, var_5, var_5, var_17, var_18, var_8, var_9, var_10, var_10)
    assert var_19 == 'from module import a, b\nfrom module import c'
    var_20 = [var_1, var_2, var_3]
    var_21 = [var_13]
    var_22 = module_0.vertical_prefix_from_module_import(var_0, var_20, var_5, var_5, var_17, var_21, var_8, var_9, var_10, var_10)
    assert var_22 == 'from module import a, b # comment\nfrom module import c'
    var_23 = []
    var_24 = []
    var_25 = module_0.vertical_prefix_from_module_import(var_0, var_23, var_5, var_5, var_6, var_24, var_8, var_9, var_10, var_10)
    assert var_25 == ''



# Parsed testcases at query #10
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = [var_1]
    var_3 = '    '
    var_4 = 88
    var_5 = []
    var_6 = '\n'
    var_7 = '#'
    var_8 = False
    var_9 = module_0.vertical_hanging_indent_bracket(var_0, var_2, var_3, var_3, var_4, var_5, var_6, var_7, var_8, var_8)
    assert var_9 == 'from module import(\n    a\n)'
    var_10 = 'b'
    var_11 = 'c'
    var_12 = [var_1, var_10, var_11]
    var_13 = []
    var_14 = module_0.vertical_hanging_indent_bracket(var_0, var_12, var_3, var_3, var_4, var_13, var_6, var_7, var_8, var_8)
    assert var_14 == 'from module import(\n    a,\n    b,\n    c\n)'
    var_15 = [var_1, var_10]
    var_16 = []
    var_17 = True
    var_18 = module_0.vertical_hanging_indent_bracket(var_0, var_15, var_3, var_3, var_4, var_16, var_6, var_7, var_17, var_8)
    assert var_18 == 'from module import(\n    a,\n    b,\n)'
    var_19 = [var_1, var_10]
    var_20 = 'comment'
    var_21 = [var_20]
    var_22 = module_0.vertical_hanging_indent_bracket(var_0, var_19, var_3, var_3, var_4, var_21, var_6, var_7, var_8, var_8)
    assert var_22 == 'from module import(\n# comment\n    a,\n    b\n)'
    var_23 = []
    var_24 = []
    var_25 = module_0.vertical_hanging_indent_bracket(var_0, var_23, var_3, var_3, var_4, var_24, var_6, var_7, var_8, var_8)
    assert var_25 == ''



# Parsed testcases at query #11
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '    '
    var_6 = 79
    var_7 = []
    var_8 = '\n'
    var_9 = '#'
    var_10 = False
    var_11 = module_0.hanging_indent_with_parentheses(var_0, var_4, var_5, var_5, var_6, var_7, var_8, var_9, var_10, var_10)
    assert var_11 == 'from module import(a, b, c)'
    var_12 = 'very_long_import_name'
    var_13 = 'another_long_import'
    var_14 = 'short'
    var_15 = [var_12, var_13, var_14]
    var_16 = 30
    var_17 = []
    var_18 = module_0.hanging_indent_with_parentheses(var_0, var_15, var_5, var_5, var_16, var_17, var_8, var_9, var_10, var_10)
    assert var_18 == 'from module import(\n    very_long_import_name, another_long_import,\n    short)'
    var_19 = [var_1, var_2]
    var_20 = 'comment'
    var_21 = [var_20]
    var_22 = module_0.hanging_indent_with_parentheses(var_0, var_19, var_5, var_5, var_6, var_21, var_8, var_9, var_10, var_10)
    assert var_22 == 'from module import(a, b) # comment'
    var_23 = [var_1, var_2]
    var_24 = []
    var_25 = True
    var_26 = module_0.hanging_indent_with_parentheses(var_0, var_23, var_5, var_5, var_6, var_24, var_8, var_9, var_25, var_10)
    assert var_26 == 'from module import(a, b,)'
    var_27 = []
    var_28 = []
    var_29 = module_0.hanging_indent_with_parentheses(var_0, var_27, var_5, var_5, var_6, var_28, var_8, var_9, var_10, var_10)
    assert var_29 == ''
    var_30 = [var_1]
    var_31 = []
    var_32 = module_0.hanging_indent_with_parentheses(var_0, var_30, var_5, var_5, var_6, var_31, var_8, var_9, var_10, var_10)
    assert var_32 == 'from module import(a)'



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'statement'
    var_1 = 'imports'
    var_2 = 'white_space'
    var_3 = 'indent'
    var_4 = 'line_length'
    var_5 = 'comments'
    var_6 = 'line_separator'
    var_7 = 'comment_prefix'
    var_8 = 'include_trailing_comma'
    var_9 = 'remove_comments'
    var_10 = 'from module import'
    var_11 = 'a'
    var_12 = 'b'
    var_13 = 'c'
    var_14 = [var_11, var_12, var_13]
    var_15 = '    '
    var_16 = 80
    var_17 = []
    var_18 = '\n'
    var_19 = '#'
    var_20 = False
    var_21 = {var_0: var_10, var_1: var_14, var_2: var_15, var_3: var_15, var_4: var_16, var_5: var_17, var_6: var_18, var_7: var_19, var_8: var_20, var_9: var_20}
    var_22 = 'from module import a, b, c'
    var_23 = 'very_long_import_name_1'
    var_24 = 'very_long_import_name_2'
    var_25 = 'very_long_import_name_3'
    var_26 = [var_23, var_24, var_25]
    var_27 = 30
    var_28 = []
    var_29 = {var_0: var_10, var_1: var_26, var_2: var_15, var_3: var_15, var_4: var_27, var_5: var_28, var_6: var_18, var_7: var_19, var_8: var_20, var_9: var_20}
    var_30 = 'from module import very_long_import_name_1, \\\n    very_long_import_name_2, \\\n    very_long_import_name_3'
    var_31 = [var_11, var_12, var_13]
    var_32 = 'comment1'
    var_33 = 'comment2'
    var_34 = [var_32, var_33]
    var_35 = {var_0: var_10, var_1: var_31, var_2: var_15, var_3: var_15, var_4: var_16, var_5: var_34, var_6: var_18, var_7: var_19, var_8: var_20, var_9: var_20}
    var_36 = 'from module import a, b, c  # comment1 comment2'
    var_37 = [var_11, var_12, var_13]
    var_38 = []
    var_39 = True
    var_40 = {var_0: var_10, var_1: var_37, var_2: var_15, var_3: var_15, var_4: var_16, var_5: var_38, var_6: var_18, var_7: var_19, var_8: var_39, var_9: var_20}
    var_41 = 'from module import a, b, c,'
    var_42 = []
    var_43 = []
    var_44 = {var_0: var_10, var_1: var_42, var_2: var_15, var_3: var_15, var_4: var_16, var_5: var_43, var_6: var_18, var_7: var_19, var_8: var_20, var_9: var_20}
    var_45 = ''



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'statement'
    var_1 = 'imports'
    var_2 = 'white_space'
    var_3 = 'indent'
    var_4 = 'line_length'
    var_5 = 'comments'
    var_6 = 'line_separator'
    var_7 = 'comment_prefix'
    var_8 = 'include_trailing_comma'
    var_9 = 'remove_comments'
    var_10 = 'from module import'
    var_11 = 'a'
    var_12 = 'b'
    var_13 = 'c'
    var_14 = [var_11, var_12, var_13]
    var_15 = '    '
    var_16 = 10
    var_17 = []
    var_18 = '\n'
    var_19 = '#'
    var_20 = False
    var_21 = {var_0: var_10, var_1: var_14, var_2: var_15, var_3: var_15, var_4: var_16, var_5: var_17, var_6: var_18, var_7: var_19, var_8: var_20, var_9: var_20}
    var_22 = 'from module import a, b, c'
    var_23 = 'from module import a, \\\n    b, \\\n    c'
    var_24 = 'comment'
    var_25 = 'from module import a, \\\n    b, \\\n    c'
    var_26 = 'from module import a, \\\n    b, \\\n    c,'
    var_27 = ''



# Parsed testcases at query #14
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '    '
    var_6 = 88
    var_7 = []
    var_8 = '\n'
    var_9 = '#'
    var_10 = False
    var_11 = module_0.vertical_grid_grouped_no_comma(var_0, var_4, var_5, var_5, var_6, var_7, var_8, var_9, var_10, var_10)



# Parsed testcases at query #15
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '    '
    var_6 = 79
    var_7 = []
    var_8 = '\n'
    var_9 = '#'
    var_10 = False
    var_11 = module_0.vertical_hanging_indent(var_0, var_4, var_5, var_5, var_6, var_7, var_8, var_9, var_10, var_10)
    var_12 = 'from module import(\n    a,\n    b,\n    c\n)'
    var_13 = [var_1, var_2, var_3]
    var_14 = '# comment'
    var_15 = [var_14]
    var_16 = module_0.vertical_hanging_indent(var_0, var_13, var_5, var_5, var_6, var_15, var_8, var_9, var_10, var_10)
    var_17 = 'from module import(# comment\n    a,\n    b,\n    c\n)'
    var_18 = [var_1, var_2, var_3]
    var_19 = []
    var_20 = True
    var_21 = module_0.vertical_hanging_indent(var_0, var_18, var_5, var_5, var_6, var_19, var_8, var_9, var_20, var_10)
    var_22 = 'from module import(\n    a,\n    b,\n    c,\n)'
    var_23 = []
    var_24 = []
    var_25 = module_0.vertical_hanging_indent(var_0, var_23, var_5, var_5, var_6, var_24, var_8, var_9, var_10, var_10)
    assert var_25 == ''
    var_26 = [var_1, var_2, var_3]
    var_27 = []
    var_28 = '\r\n'
    var_29 = module_0.vertical_hanging_indent(var_0, var_26, var_5, var_5, var_6, var_27, var_28, var_9, var_10, var_10)
    var_30 = 'from module import(\r\n    a,\r\n    b,\r\n    c\r\n)'



# Parsed testcases at query #16
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'GRID'
    var_1 = module_0.from_string(var_0)
    var_2 = 'VERTICAL'
    var_3 = module_0.from_string(var_2)
    var_4 = 'HANGING_INDENT'
    var_5 = module_0.from_string(var_4)
    var_6 = 'VERTICAL_HANGING_INDENT'
    var_7 = module_0.from_string(var_6)
    var_8 = 'VERTICAL_GRID'
    var_9 = module_0.from_string(var_8)
    var_10 = 'VERTICAL_GRID_GROUPED'
    var_11 = module_0.from_string(var_10)
    var_12 = 'VERTICAL_GRID_GROUPED_NO_COMMA'
    var_13 = module_0.from_string(var_12)
    var_14 = 'NOQA'
    var_15 = module_0.from_string(var_14)
    var_16 = 'VERTICAL_HANGING_INDENT_BRACKET'
    var_17 = module_0.from_string(var_16)
    var_18 = 'VERTICAL_PREFIX_FROM_MODULE_IMPORT'
    var_19 = module_0.from_string(var_18)
    var_20 = 'HANGING_INDENT_WITH_PARENTHESES'
    var_21 = module_0.from_string(var_20)
    var_22 = 'BACKSLASH_GRID'
    var_23 = module_0.from_string(var_22)
    var_24 = '0'
    var_25 = module_0.from_string(var_24)
    var_26 = '1'
    var_27 = module_0.from_string(var_26)
    var_28 = '2'
    var_29 = module_0.from_string(var_28)
    var_30 = 'invalid'
    var_31 = module_0.from_string(var_30)
    assert var_31 is None



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'statement'
    var_1 = 'imports'
    var_2 = 'white_space'
    var_3 = 'indent'
    var_4 = 'line_length'
    var_5 = 'comments'
    var_6 = 'line_separator'
    var_7 = 'comment_prefix'
    var_8 = 'include_trailing_comma'
    var_9 = 'remove_comments'
    var_10 = 'from module import'
    var_11 = 'a'
    var_12 = 'b'
    var_13 = 'c'
    var_14 = [var_11, var_12, var_13]
    var_15 = '    '
    var_16 = 80
    var_17 = []
    var_18 = '\n'
    var_19 = '#'
    var_20 = False
    var_21 = {var_0: var_10, var_1: var_14, var_2: var_15, var_3: var_15, var_4: var_16, var_5: var_17, var_6: var_18, var_7: var_19, var_8: var_20, var_9: var_20}
    var_22 = 'from module import(\n    a,\n    b,\n    c\n)'
    var_23 = 'comment1'
    var_24 = 'comment2'
    var_25 = 'from module import(# comment1\n# comment2\n    a,\n    b,\n    c\n)'
    var_26 = 'from module import(# comment1\n# comment2\n    a,\n    b,\n    c,\n)'
    var_27 = ''
    var_28 = 'single_import'
    var_29 = 'from module import(# comment1\n# comment2\n    single_import,\n)'



# Parsed testcases at query #18
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = []
    var_2 = '    '
    var_3 = 88
    var_4 = []
    var_5 = '\n'
    var_6 = '#'
    var_7 = False
    var_8 = module_0.vertical(var_0, var_1, var_2, var_2, var_3, var_4, var_5, var_6, var_7, var_7)
    assert var_8 == ''
    var_9 = 'a'
    var_10 = [var_9]
    var_11 = []
    var_12 = module_0.vertical(var_0, var_10, var_2, var_2, var_3, var_11, var_5, var_6, var_7, var_7)
    assert var_12 == 'from module import(\n    a)'
    var_13 = 'b'
    var_14 = 'c'
    var_15 = [var_9, var_13, var_14]
    var_16 = []
    var_17 = module_0.vertical(var_0, var_15, var_2, var_2, var_3, var_16, var_5, var_6, var_7, var_7)
    assert var_17 == 'from module import(\n    a,\n    b,\n    c)'
    var_18 = [var_9, var_13, var_14]
    var_19 = []
    var_20 = True
    var_21 = module_0.vertical(var_0, var_18, var_2, var_2, var_3, var_19, var_5, var_6, var_20, var_7)
    assert var_21 == 'from module import(\n    a,\n    b,\n    c,)'
    var_22 = [var_9, var_13, var_14]
    var_23 = 'comment'
    var_24 = [var_23]
    var_25 = module_0.vertical(var_0, var_22, var_2, var_2, var_3, var_24, var_5, var_6, var_7, var_7)
    assert var_25 == 'from module import(\n    a, # comment\n    b,\n    c)'
    var_26 = [var_9, var_13, var_14]
    var_27 = [var_23]
    var_28 = module_0.vertical(var_0, var_26, var_2, var_2, var_3, var_27, var_5, var_6, var_7, var_20)
    assert var_28 == 'from module import(\n    a,\n    b,\n    c)'



# Parsed testcases at query #19
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = ' '
    var_6 = '    '
    var_7 = 79
    var_8 = []
    var_9 = '\n'
    var_10 = '#'
    var_11 = False
    var_12 = module_0.noqa(var_0, var_4, var_5, var_6, var_7, var_8, var_9, var_10, var_11, var_11)
    assert var_12 == 'importa, b, c'
    var_13 = [var_1, var_2, var_3]
    var_14 = 'test'
    var_15 = 'comment'
    var_16 = [var_14, var_15]
    var_17 = module_0.noqa(var_0, var_13, var_5, var_6, var_7, var_16, var_9, var_10, var_11, var_11)
    assert var_17 == 'importa, b, c # test comment'
    var_18 = [var_1, var_2, var_3]
    var_19 = 10
    var_20 = [var_14, var_15]
    var_21 = module_0.noqa(var_0, var_18, var_5, var_6, var_19, var_20, var_9, var_10, var_11, var_11)
    assert var_21 == 'importa, b, c # NOQA test comment'
    var_22 = [var_1, var_2, var_3]
    var_23 = 'NOQA'
    var_24 = [var_23, var_14]
    var_25 = module_0.noqa(var_0, var_22, var_5, var_6, var_19, var_24, var_9, var_10, var_11, var_11)
    assert var_25 == 'importa, b, c # NOQA test'
    var_26 = [var_1, var_2, var_3]
    var_27 = 5
    var_28 = []
    var_29 = module_0.noqa(var_0, var_26, var_5, var_6, var_27, var_28, var_9, var_10, var_11, var_11)
    assert var_29 == 'importa, b, c # NOQA'



# Parsed testcases at query #20
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '    '
    var_6 = 80
    var_7 = []
    var_8 = '\n'
    var_9 = '#'
    var_10 = True
    var_11 = False
    var_12 = module_0.vertical_hanging_indent(var_0, var_4, var_5, var_5, var_6, var_7, var_8, var_9, var_10, var_11)
    var_13 = 'from module import(\n    a,\n    b,\n    c,\n)'
    var_14 = [var_1, var_2, var_3]
    var_15 = 'comment'
    var_16 = [var_15]
    var_17 = module_0.vertical_hanging_indent(var_0, var_14, var_5, var_5, var_6, var_16, var_8, var_9, var_10, var_11)
    var_18 = 'from module import(# comment\n    a,\n    b,\n    c,\n)'
    var_19 = [var_1, var_2, var_3]
    var_20 = []
    var_21 = module_0.vertical_hanging_indent(var_0, var_19, var_5, var_5, var_6, var_20, var_8, var_9, var_11, var_11)
    var_22 = 'from module import(\n    a,\n    b,\n    c\n)'
    var_23 = []
    var_24 = []
    var_25 = module_0.vertical_hanging_indent(var_0, var_23, var_5, var_5, var_6, var_24, var_8, var_9, var_10, var_11)
    var_26 = ''



# Parsed testcases at query #21
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '    '
    var_6 = 88
    var_7 = []
    var_8 = '\n'
    var_9 = '#'
    var_10 = False
    var_11 = module_0.vertical_grid_grouped_no_comma(var_0, var_4, var_5, var_5, var_6, var_7, var_8, var_9, var_10, var_10)



# Parsed testcases at query #22
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'import'
    var_1 = []
    var_2 = '    '
    var_3 = 88
    var_4 = []
    var_5 = '\n'
    var_6 = '#'
    var_7 = False
    var_8 = module_0.backslash_grid(var_0, var_1, var_2, var_2, var_3, var_4, var_5, var_6, var_7, var_7)
    assert var_8 == ''
    var_9 = 'from module import'
    var_10 = 'a'
    var_11 = [var_10]
    var_12 = []
    var_13 = module_0.backslash_grid(var_9, var_11, var_2, var_2, var_3, var_12, var_5, var_6, var_7, var_7)
    assert var_13 == 'from module import a'
    var_14 = 'b'
    var_15 = 'c'
    var_16 = [var_10, var_14, var_15]
    var_17 = []
    var_18 = module_0.backslash_grid(var_9, var_16, var_2, var_2, var_3, var_17, var_5, var_6, var_7, var_7)
    assert var_18 == 'from module import a, b, c'
    var_19 = 'very_long_name_a'
    var_20 = 'very_long_name_b'
    var_21 = 'very_long_name_c'
    var_22 = [var_19, var_20, var_21]
    var_23 = 30
    var_24 = []
    var_25 = module_0.backslash_grid(var_9, var_22, var_2, var_2, var_23, var_24, var_5, var_6, var_7, var_7)
    var_26 = 'from module import very_long_name_a, \\\n    very_long_name_b, \\\n    very_long_name_c'
    var_27 = [var_10, var_14]
    var_28 = '# comment'
    var_29 = [var_28]
    var_30 = module_0.backslash_grid(var_9, var_27, var_2, var_2, var_3, var_29, var_5, var_6, var_7, var_7)
    assert var_30 == 'from module import a, b  # comment'
    var_31 = [var_10, var_14]
    var_32 = []
    var_33 = True
    var_34 = module_0.backslash_grid(var_9, var_31, var_2, var_2, var_3, var_32, var_5, var_6, var_33, var_7)
    assert var_34 == 'from module import a, b,'



# Parsed testcases at query #23
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = []
    var_2 = '    '
    var_3 = 88
    var_4 = []
    var_5 = '\n'
    var_6 = '#'
    var_7 = False
    var_8 = module_0.grid(var_0, var_1, var_2, var_2, var_3, var_4, var_5, var_6, var_7, var_7)
    assert var_8 == ''
    var_9 = 'a'
    var_10 = [var_9]
    var_11 = []
    var_12 = module_0.grid(var_0, var_10, var_2, var_2, var_3, var_11, var_5, var_6, var_7, var_7)
    assert var_12 == 'from module import(a)'
    var_13 = 'b'
    var_14 = 'c'
    var_15 = [var_9, var_13, var_14]
    var_16 = []
    var_17 = module_0.grid(var_0, var_15, var_2, var_2, var_3, var_16, var_5, var_6, var_7, var_7)
    assert var_17 == 'from module import(a, b, c)'
    var_18 = 'd'
    var_19 = 'e'
    var_20 = 'f'
    var_21 = [var_9, var_13, var_14, var_18, var_19, var_20]
    var_22 = 30
    var_23 = []
    var_24 = module_0.grid(var_0, var_21, var_2, var_2, var_22, var_23, var_5, var_6, var_7, var_7)
    assert var_24 == 'from module import(a, b, c,\n    d, e, f)'
    var_25 = [var_9, var_13, var_14]
    var_26 = '# Comment'
    var_27 = [var_26]
    var_28 = module_0.grid(var_0, var_25, var_2, var_2, var_3, var_27, var_5, var_6, var_7, var_7)
    assert var_28 == 'from module import(a, b, c) # Comment'
    var_29 = [var_9, var_13, var_14]
    var_30 = []
    var_31 = True
    var_32 = module_0.grid(var_0, var_29, var_2, var_2, var_3, var_30, var_5, var_6, var_31, var_7)
    assert var_32 == 'from module import(a, b, c,)'



# Parsed testcases at query #24
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = []
    var_2 = '    '
    var_3 = 88
    var_4 = []
    var_5 = '\n'
    var_6 = '#'
    var_7 = False
    var_8 = module_0.vertical_grid(var_0, var_1, var_2, var_2, var_3, var_4, var_5, var_6, var_7, var_7)
    assert var_8 == ''
    var_9 = 'a'
    var_10 = [var_9]
    var_11 = []
    var_12 = module_0.vertical_grid(var_0, var_10, var_2, var_2, var_3, var_11, var_5, var_6, var_7, var_7)
    assert var_12 == 'from module import(\n    a)'
    var_13 = 'b'
    var_14 = 'c'
    var_15 = [var_9, var_13, var_14]
    var_16 = []
    var_17 = module_0.vertical_grid(var_0, var_15, var_2, var_2, var_3, var_16, var_5, var_6, var_7, var_7)
    assert var_17 == 'from module import(\n    a, b, c)'
    var_18 = [var_9, var_13, var_14]
    var_19 = []
    var_20 = True
    var_21 = module_0.vertical_grid(var_0, var_18, var_2, var_2, var_3, var_19, var_5, var_6, var_20, var_7)
    assert var_21 == 'from module import(\n    a, b, c,)'



# Parsed testcases at query #25
#--------------------------


import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '    '
    var_6 = 88
    var_7 = []
    var_8 = '\n'
    var_9 = '#'
    var_10 = False
    var_11 = module_0.vertical_grid_grouped_no_comma(var_0, var_4, var_5, var_5, var_6, var_7, var_8, var_9, var_10, var_10)



