####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = 'from module import function'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    var_5 = 'from module import function1, function2, function3, function4'
    var_6 = module_1.line(var_5, var_3, var_1)
    var_7 = 'from module import '
    var_8 = 'from module import function1, function2  # comment'
    var_9 = module_1.line(var_8, var_3, var_1)
    var_10 = 'from module import function1, function2'
    var_11 = 20
    var_12 = module_1.line(var_10, var_3, var_1)
    var_13 = True
    var_14 = module_0.Config()
    var_15 = 'from module import function1, function2'
    var_16 = module_1.line(var_15, var_3, var_14)
    var_17 = module_0.Config()
    var_18 = 'from module import function1, function2'
    var_19 = module_1.line(var_18, var_3, var_17)
    var_20 = ','
    var_21 = 'from module import function1, function2'
    var_22 = module_1.line(var_21, var_3, var_17)
    var_23 = 'from module import function1, function2'
    var_24 = module_1.line(var_23, var_3, var_17)
    var_25 = module_0.Config()
    var_26 = 'from module import function1 as f1, function2 as f2'
    var_27 = module_1.line(var_26, var_3, var_25)
    var_28 = module_0.Config()
    var_29 = 'from module import function1.function2'
    var_30 = module_1.line(var_29, var_3, var_28)
    var_31 = module_0.Config()
    var_32 = 'cimport module.function1, module.function2'
    var_33 = module_1.line(var_32, var_3, var_31)
    var_34 = module_0.Config()
    var_35 = 'from module import function1, function2  # comment'
    var_36 = module_1.line(var_35, var_3, var_34)



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 80
    var_1 = 'from module import very_long_function_name'
    var_2 = '\n'
    var_3 = 'from module import very_long_function_name, another_very_long_function_name, third_very_long_function_name'
    var_4 = 'from module import (\n    very_long_function_name,\n    another_very_long_function_name,\n    third_very_long_function_name,\n)'
    var_5 = 'from module import very_long_function_name  # some comment'
    var_6 = 'from module import (\n    very_long_function_name,  # some comment\n)'
    var_7 = 'from module import very_long_function_name  # NOQA'
    var_8 = 'from module import very_long_function_name  # noqa'
    var_9 = 'from module import (\n    very_long_function_name,  # noqa\n)'
    var_10 = 'from module import very_long_function_name as vlf'
    var_11 = 'from module import very_long_function_name as vlf'
    var_12 = 'from module import very_long_function_name as very_long_alias'
    var_13 = 'from module import (\n    very_long_function_name as very_long_alias,\n)'
    var_14 = 'from module import very_long_function_name, another_very_long_function_name'
    var_15 = 'from module import very_long_function_name,\\n    another_very_long_function_name'
    var_16 = 'from module import very_long_function_name, another_very_long_function_name'
    var_17 = 'from module import (\n    very_long_function_name\n    another_very_long_function_name\n)'
    var_18 = 'from module import very_long_function_name, another_very_long_function_name'
    var_19 = 'from module import very_long_function_name, another_very_long_function_name'
    var_20 = 'from module import very_long_function_name, another_very_long_function_name'
    var_21 = 'from module import very_long_function_name, another_very_long_function_name  # NOQA'



# Parsed testcases at query #3
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'func1'
    var_2 = 'func2'
    var_3 = [var_1, var_2]
    var_4 = module_0.import_statement(var_0, var_3)
    var_5 = [var_1, var_2]
    var_6 = '# Comment'
    var_7 = [var_6]
    var_8 = module_0.import_statement(var_0, var_5, var_7)
    var_9 = [var_1, var_2]
    var_10 = '\r\n'
    var_11 = module_0.import_statement(var_0, var_9, line_separator=var_10)
    var_12 = [var_1, var_2]
    var_13 = True
    var_14 = module_0.import_statement(var_0, var_12, explode=var_13)
    var_15 = '\n'
    var_16 = module_1.Config()
    var_17 = [var_1, var_2]
    var_18 = module_0.import_statement(var_0, var_17, config=var_16)
    var_19 = [var_1, var_2]
    var_20 = module_1.Config()
    var_21 = [var_1, var_2]
    var_22 = module_0.import_statement(var_0, var_21, config=var_20)
    var_23 = ','
    var_24 = module_1.Config()
    var_25 = [var_1, var_2]
    var_26 = [var_6]
    var_27 = module_0.import_statement(var_0, var_25, var_26, config=var_24)
    var_28 = '    '
    var_29 = module_1.Config()
    var_30 = [var_1, var_2]
    var_31 = module_0.import_statement(var_0, var_30, config=var_29)
    var_32 = '# '
    var_33 = module_1.Config()
    var_34 = [var_1, var_2]
    var_35 = 'Comment'
    var_36 = [var_35]
    var_37 = module_0.import_statement(var_0, var_34, var_36, config=var_33)
    var_38 = 50
    var_39 = module_1.Config()
    var_40 = [var_1, var_2]
    var_41 = module_0.import_statement(var_0, var_40, config=var_39)
    var_42 = 0
    var_43 = result.split(var_15)[var_42]
    var_44 = len(var_43)
    var_45 = module_1.Config()
    var_46 = [var_1, var_2]
    var_47 = module_0.import_statement(var_0, var_46, config=var_45)
    var_48 = result.split(var_15)[var_42]
    var_49 = len(var_48)
    var_50 = module_1.Config()
    var_51 = [var_1, var_2]
    var_52 = module_0.import_statement(var_0, var_51, config=var_50)



# Parsed testcases at query #4
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'item1'
    var_2 = 'item2'
    var_3 = 'item3'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    assert var_5 == 'from module import item1, item2, item3'
    var_6 = [var_1, var_2]
    var_7 = '# Comment 1'
    var_8 = '# Comment 2'
    var_9 = [var_7, var_8]
    var_10 = module_0.import_statement(var_0, var_6, var_9)
    var_11 = 'item4'
    var_12 = 'item5'
    var_13 = [var_1, var_2, var_3, var_11, var_12]
    var_14 = [var_1, var_2, var_3]
    var_15 = True
    var_16 = module_0.import_statement(var_0, var_14, explode=var_15)
    var_17 = '\n'
    var_18 = 50
    var_19 = '    '
    var_20 = module_1.Config()
    var_21 = [var_1, var_2, var_3, var_11]
    var_22 = module_0.import_statement(var_0, var_21, config=var_20)
    var_23 = 20
    var_24 = module_1.Config()
    var_25 = [var_1, var_2, var_3]
    var_26 = module_0.import_statement(var_0, var_25, config=var_24)
    var_27 = [var_1, var_2]
    var_28 = '\r\n'
    var_29 = module_0.import_statement(var_0, var_27, line_separator=var_28)



# Parsed testcases at query #5
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = 'from module import very_long_function_name_that_exceeds_line_length'
    var_2 = '\n'
    var_3 = 'import os  # This is a comment'
    var_4 = 'from module import something  # NOQA'
    var_5 = 'from module import something as alias'
    var_6 = 'from module.submodule import something'
    var_7 = 'import os'
    var_8 = True
    var_9 = module_0.Config()
    var_10 = 'from module import func1, func2, func3'
    var_11 = module_1.line(var_10, var_2, var_9)
    var_12 = module_0.Config()
    var_13 = 'from module import func1, func2'
    var_14 = module_1.line(var_13, var_2, var_12)
    var_15 = ','
    var_16 = module_0.Config()
    var_17 = 'from module import func1, func2'
    var_18 = module_1.line(var_17, var_2, var_16)



# Parsed testcases at query #6
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'func1'
    var_2 = 'func2'
    var_3 = 'func3'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    var_6 = [var_1, var_2]
    var_7 = '# Comment 1'
    var_8 = '# Comment 2'
    var_9 = [var_7, var_8]
    var_10 = module_0.import_statement(var_0, var_6, var_9)
    var_11 = [var_1, var_2]
    var_12 = '\r\n'
    var_13 = module_0.import_statement(var_0, var_11, line_separator=var_12)
    var_14 = [var_1, var_2, var_3]
    var_15 = True
    var_16 = module_0.import_statement(var_0, var_14, explode=var_15)
    var_17 = '\n'
    var_18 = [var_1, var_2, var_3]
    var_19 = len(var_18)
    var_20 = 50
    var_21 = 40
    var_22 = '    '
    var_23 = module_1.Config()
    var_24 = [var_1, var_2, var_3]
    var_25 = module_0.import_statement(var_0, var_24, config=var_23)
    var_26 = [var_1, var_2, var_3]
    var_27 = module_1.Config()
    var_28 = [var_1, var_2, var_3]
    var_29 = module_0.import_statement(var_0, var_28, config=var_27)
    var_30 = []
    var_31 = module_0.import_statement(var_0, var_30)
    assert var_31 == 'from module import'
    var_32 = [var_1]
    var_33 = module_0.import_statement(var_0, var_32)
    assert var_33 == 'from module import func1'



# Parsed testcases at query #7
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1
import re as module_2

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'item1'
    var_2 = 'item2'
    var_3 = 'item3'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    var_6 = [var_1, var_2]
    var_7 = '# Comment 1'
    var_8 = '# Comment 2'
    var_9 = [var_7, var_8]
    var_10 = module_0.import_statement(var_0, var_6, var_9)
    var_11 = [var_1, var_2]
    var_12 = '\r\n'
    var_13 = module_0.import_statement(var_0, var_11, line_separator=var_12)
    var_14 = [var_1, var_2, var_3]
    var_15 = True
    var_16 = module_0.import_statement(var_0, var_14, explode=var_15)
    var_17 = '\n'
    var_18 = 20
    var_19 = module_1.Config()
    var_20 = [var_1, var_2, var_3]
    var_21 = module_0.import_statement(var_0, var_20, config=var_19)
    var_22 = module_2.split(var_17)
    var_23 = len(var_22)
    var_24 = -1
    var_25 = var_22[var_24]
    var_26 = len(var_25)
    var_27 = -1
    var_28 = var_22[:var_27]
    var_29 = [var_1, var_2, var_3]
    var_30 = module_1.Config()
    var_31 = [var_1, var_2]
    var_32 = module_0.import_statement(var_0, var_31, config=var_30)
    var_33 = ','
    var_34 = module_1.Config()
    var_35 = [var_1, var_2]
    var_36 = '# Comment'
    var_37 = [var_36]
    var_38 = module_0.import_statement(var_0, var_35, var_37, config=var_34)



# Parsed testcases at query #8
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'func1'
    var_2 = 'func2'
    var_3 = [var_1, var_2]
    var_4 = module_0.import_statement(var_0, var_3)
    var_5 = [var_1, var_2]
    var_6 = '# Comment'
    var_7 = [var_6]
    var_8 = module_0.import_statement(var_0, var_5, var_7)
    var_9 = [var_1, var_2]
    var_10 = '\r\n'
    var_11 = module_0.import_statement(var_0, var_9, line_separator=var_10)
    var_12 = [var_1, var_2]
    var_13 = True
    var_14 = module_0.import_statement(var_0, var_12, explode=var_13)
    var_15 = [var_1, var_2]
    var_16 = 20
    var_17 = module_1.Config()
    var_18 = [var_1, var_2]
    var_19 = module_0.import_statement(var_0, var_18, config=var_17)
    var_20 = 0
    var_21 = '\n'
    var_22 = result.split(var_21)[var_20]
    var_23 = len(var_22)
    var_24 = 'func3'
    var_25 = 'func4'
    var_26 = 'func5'
    var_27 = [var_1, var_2, var_24, var_25, var_26]
    var_28 = module_0.import_statement(var_0, var_27)
    var_29 = []
    var_30 = module_0.import_statement(var_0, var_29)
    var_31 = [var_1]
    var_32 = module_0.import_statement(var_0, var_31)
    var_33 = '    '
    var_34 = module_1.Config()
    var_35 = [var_1, var_2]
    var_36 = module_0.import_statement(var_0, var_35, config=var_34)
    var_37 = module_1.Config()
    var_38 = [var_1, var_2]
    var_39 = module_0.import_statement(var_0, var_38, config=var_37)
    var_40 = ','
    var_41 = module_1.Config()
    var_42 = [var_1, var_2]
    var_43 = [var_6]
    var_44 = module_0.import_statement(var_0, var_42, var_43, config=var_41)
    var_45 = '# '
    var_46 = module_1.Config()
    var_47 = [var_1, var_2]
    var_48 = 'Comment'
    var_49 = [var_48]
    var_50 = module_0.import_statement(var_0, var_47, var_49, config=var_46)



# Parsed testcases at query #9
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1
import re as module_2

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'A'
    var_2 = 'B'
    var_3 = 'C'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    assert var_5 == 'from module import A, B, C'
    var_6 = 'D'
    var_7 = 'E'
    var_8 = [var_1, var_2, var_3, var_6, var_7]
    var_9 = 30
    var_10 = module_1.Config()
    var_11 = module_0.import_statement(var_0, var_8, config=var_10)
    var_12 = [var_1, var_2]
    var_13 = '# Comment 1'
    var_14 = '# Comment 2'
    var_15 = [var_13, var_14]
    var_16 = module_0.import_statement(var_0, var_12, var_15)
    var_17 = [var_1, var_2, var_3]
    var_18 = True
    var_19 = module_0.import_statement(var_0, var_17, explode=var_18)
    var_20 = '\n'
    var_21 = module_2.split(var_20)
    var_22 = len(var_21)
    assert var_22 == 3
    var_23 = [var_1, var_2]
    var_24 = '\r\n'
    var_25 = module_0.import_statement(var_0, var_23, line_separator=var_24)
    var_26 = 20
    var_27 = module_1.Config()
    var_28 = [var_1, var_2, var_3]
    var_29 = module_0.import_statement(var_0, var_28, config=var_27)
    var_30 = module_2.split(var_20)
    var_31 = -1
    var_32 = var_30[var_31]
    var_33 = len(var_32)
    var_34 = -1
    var_35 = var_30[:var_34]
    var_36 = min(var_6)
    var_37 = module_1.Config()
    var_38 = [var_32, var_33, var_34]
    var_39 = module_0.import_statement(var_31, var_38, config=var_37)
    var_40 = ','
    var_41 = module_1.Config()
    var_42 = [var_32, var_33]
    var_43 = '# Comment'
    var_44 = [var_43]
    var_45 = module_0.import_statement(var_31, var_42, var_44, config=var_41)
    var_46 = '    '
    var_47 = module_1.Config()
    var_48 = [var_32, var_33, var_34]
    var_49 = module_0.import_statement(var_31, var_48, config=var_47)
    var_50 = '# '
    var_51 = module_1.Config()
    var_52 = [var_32, var_33]
    var_53 = 'Comment'
    var_54 = [var_53]
    var_55 = module_0.import_statement(var_31, var_52, var_54, config=var_51)



# Parsed testcases at query #10
#--------------------------


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import a, b, c'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'from module import a, b, c'
    var_3 = 20
    var_4 = 'from module import a, b, c, d, e'
    var_5 = 'from module import a, b, c, d, e  # comment'
    var_6 = 'from module import a as b, c as d'
    var_7 = 'cimport module.a, module.b'
    var_8 = 'from module import a.b, c.d'
    var_9 = 'from module import a, b, c, d, e  # noqa'
    var_10 = True
    var_11 = ' # '



# Parsed testcases at query #11
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1
import re as module_2

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'func1'
    var_2 = 'func2'
    var_3 = 'func3'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    var_6 = [var_1, var_2]
    var_7 = '# Comment 1'
    var_8 = '# Comment 2'
    var_9 = [var_7, var_8]
    var_10 = module_0.import_statement(var_0, var_6, var_9)
    var_11 = [var_1, var_2]
    var_12 = '\r\n'
    var_13 = module_0.import_statement(var_0, var_11, line_separator=var_12)
    var_14 = [var_1, var_2, var_3]
    var_15 = True
    var_16 = module_0.import_statement(var_0, var_14, explode=var_15)
    var_17 = '\n'
    var_18 = 50
    var_19 = '    '
    var_20 = module_1.Config()
    var_21 = 'very_long_function_name_1'
    var_22 = 'very_long_function_name_2'
    var_23 = [var_21, var_22]
    var_24 = module_0.import_statement(var_0, var_23, config=var_20)
    var_25 = module_2.split(var_17)
    var_26 = -1
    var_27 = var_25[var_26]
    var_28 = len(var_27)
    var_29 = -1
    var_30 = var_25[:var_29]
    var_31 = len(var_25)
    var_32 = var_31 == var_15
    var_33 = '  '
    var_34 = '# '
    var_35 = module_1.Config()
    var_36 = [var_1, var_2]
    var_37 = module_0.import_statement(var_0, var_36, config=var_35)
    var_38 = ','
    var_39 = [var_1, var_2, var_3]
    var_40 = []
    var_41 = module_0.import_statement(var_0, var_40)
    assert var_41 == 'from module import'
    var_42 = [var_1]
    var_43 = module_0.import_statement(var_0, var_42)
    assert var_43 == 'from module import func1'



# Parsed testcases at query #12
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'function1'
    var_2 = 'function2'
    var_3 = 'function3'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    assert var_5 == 'from module import function1, function2, function3'
    var_6 = [var_1, var_2, var_3]
    var_7 = '# Comment 1'
    var_8 = '# Comment 2'
    var_9 = [var_7, var_8]
    var_10 = module_0.import_statement(var_0, var_6, var_9)
    var_11 = [var_1, var_2, var_3]
    var_12 = '\r\n'
    var_13 = module_0.import_statement(var_0, var_11, line_separator=var_12)
    var_14 = [var_1, var_2, var_3]
    var_15 = True
    var_16 = module_0.import_statement(var_0, var_14, explode=var_15)
    var_17 = 20
    var_18 = module_1.Config()
    var_19 = [var_1, var_2, var_3]
    var_20 = module_0.import_statement(var_0, var_19, config=var_18)
    var_21 = '    '
    var_22 = module_1.Config()
    var_23 = [var_1, var_2, var_3]
    var_24 = module_0.import_statement(var_0, var_23, config=var_22)
    var_25 = module_1.Config()
    var_26 = [var_1, var_2, var_3]
    var_27 = module_0.import_statement(var_0, var_26, config=var_25)
    var_28 = ','
    var_29 = module_1.Config()
    var_30 = [var_1, var_2, var_3]
    var_31 = [var_7, var_8]
    var_32 = module_0.import_statement(var_0, var_30, var_31, config=var_29)
    var_33 = [var_1, var_2, var_3]
    var_34 = module_1.Config()
    var_35 = [var_1, var_2, var_3]
    var_36 = module_0.import_statement(var_0, var_35, config=var_34)



# Parsed testcases at query #13
#--------------------------


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import function'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = 'from module import function1, function2, function3'
    var_4 = 30
    var_5 = 0
    var_6 = result.split(var_1)[var_5]
    var_7 = len(var_6)
    var_8 = 'from module import function  # some comment'
    var_9 = 'from module import function1, function2, function3'
    var_10 = True
    var_11 = ','
    var_12 = 'from module import function as func'
    var_13 = 'from module.submodule import function'
    var_14 = 'import module.function'
    var_15 = 'cimport module.function'
    var_16 = 'import os'
    var_17 = module_0.line(var_16, var_1)
    var_18 = ''
    var_19 = module_0.line(var_18, var_1)
    var_20 = '# This is a comment'
    var_21 = module_0.line(var_20, var_1)



# Parsed testcases at query #14
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = 'from module import function1, function2, function3'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    var_5 = 'from module import function1, function2  # some comment'
    var_6 = module_1.line(var_5, var_3, var_1)
    var_7 = 'from module import function1, function2  # NOQA'
    var_8 = module_1.line(var_7, var_3, var_1)
    var_9 = 'from module import function1, function2, function3, function4, function5'
    var_10 = module_1.line(var_9, var_3, var_1)
    var_11 = True
    var_12 = module_0.Config()
    var_13 = 'from module import function1, function2, function3'
    var_14 = module_1.line(var_13, var_3, var_12)
    var_15 = module_0.Config()
    var_16 = 'from module import function1, function2, function3'
    var_17 = module_1.line(var_16, var_3, var_15)
    var_18 = ','
    var_19 = 'from module import function1, function2, function3'
    var_20 = module_0.Config()
    var_21 = 'from module import function1, function2, function3'
    var_22 = module_1.line(var_21, var_3, var_20)
    var_23 = '\r\n'
    var_24 = module_1.line(var_2, var_23, var_1)
    var_25 = 'from module import function1'
    var_26 = module_1.line(var_25, var_3, var_1)



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 80
    var_1 = 'from module import very_long_function_name_that_exceeds_line_length'
    var_2 = '\n'
    var_3 = 'from module import func  # some comment'
    var_4 = 'from module import very_long_function_name_that_exceeds_line_length  # NOQA'
    var_5 = 'from module import very_long_function_name as alias'
    var_6 = 'from module.submodule import very_long_function_name'
    var_7 = 'from module import func1, func2, func3'
    var_8 = 'from module import func1, func2, func3'
    var_9 = ','
    var_10 = 'from module import func1, func2, func3'
    var_11 = 'from module import func  # some comment'
    var_12 = 'from module import func  # some comment'



# Parsed testcases at query #16
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1
import re as module_2

def test_case_0():
    var_0 = 'from module import something'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = 'from module import something_very_long_that_exceeds_line_length'
    var_4 = 20
    var_5 = module_1.Config()
    var_6 = module_0.line(var_3, var_1, var_5)
    var_7 = 'from module import something  # comment'
    var_8 = module_0.line(var_7, var_1, var_5)
    var_9 = 'from module import something  # NOQA'
    var_10 = module_0.line(var_9, var_1, var_5)
    var_11 = True
    var_12 = module_1.Config()
    var_13 = 'from module import something, another'
    var_14 = module_0.line(var_13, var_1, var_12)
    var_15 = module_1.Config()
    var_16 = 'from module import something, another'
    var_17 = module_0.line(var_16, var_1, var_15)
    var_18 = ','
    var_19 = 'from module.submodule import something'
    var_20 = module_0.line(var_19, var_1, var_5)
    var_21 = 'from module import something as alias'
    var_22 = module_0.line(var_21, var_1, var_5)
    var_23 = 'cimport module.something'
    var_24 = module_0.line(var_23, var_1, var_5)
    var_25 = module_1.Config()
    var_26 = 'from module import something, another, third'
    var_27 = module_0.line(var_26, var_1, var_25)
    var_28 = module_2.split(var_1)
    var_29 = len(var_28)
    var_30 = -1
    var_31 = var_28[:var_30]
    var_32 = -1
    var_33 = var_28[var_32]
    var_34 = len(var_33)
    var_35 = module_1.Config()
    var_36 = 'from module import something  # comment'
    var_37 = module_0.line(var_36, var_1, var_35)



# Parsed testcases at query #17
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1
import re as module_2

def test_case_0():
    var_0 = 'from module import ('
    var_1 = 'item1'
    var_2 = 'item2'
    var_3 = 'item3'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    var_6 = [var_1, var_2]
    var_7 = '# Comment 1'
    var_8 = '# Comment 2'
    var_9 = [var_7, var_8]
    var_10 = module_0.import_statement(var_0, var_6, var_9)
    var_11 = [var_1, var_2]
    var_12 = '\r\n'
    var_13 = module_0.import_statement(var_0, var_11, line_separator=var_12)
    var_14 = [var_1, var_2]
    var_15 = True
    var_16 = module_0.import_statement(var_0, var_14, explode=var_15)
    var_17 = '\n'
    var_18 = 20
    var_19 = module_1.Config()
    var_20 = [var_1, var_2, var_3]
    var_21 = module_0.import_statement(var_0, var_20, config=var_19)
    var_22 = module_2.split(var_17)
    var_23 = len(var_22)
    var_24 = module_1.Config()
    var_25 = [var_1, var_2]
    var_26 = module_0.import_statement(var_0, var_25, config=var_24)
    var_27 = ','
    var_28 = [var_1, var_2]
    var_29 = []
    var_30 = module_0.import_statement(var_0, var_29)



# Parsed testcases at query #18
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = 'from module import function1, function2, function3'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    var_5 = 20
    var_6 = module_0.Config()
    var_7 = 'from module import function1, function2, function3'
    var_8 = 'from module import (\n    function1,\n    function2,\n    function3,\n)'
    var_9 = module_1.line(var_7, var_3, var_6)
    var_10 = '# '
    var_11 = module_0.Config()
    var_12 = 'from module import function1, function2, function3  # comment'
    var_13 = 'from module import (\n    function1,\n    function2,\n    function3,  # comment\n)'
    var_14 = module_1.line(var_12, var_3, var_11)
    var_15 = module_0.Config()
    var_16 = 'from module import function1, function2, function3  # NOQA'
    var_17 = module_1.line(var_16, var_3, var_15)
    var_18 = True
    var_19 = module_0.Config()
    var_20 = 'from module import function1, function2, function3  # noqa'
    var_21 = 'from module import (\n    function1,\n    function2,\n    function3,  # noqa\n)'
    var_22 = module_1.line(var_20, var_3, var_19)
    var_23 = module_0.Config()
    var_24 = 'import module as alias'
    var_25 = 'import module as (\n    alias\n)'
    var_26 = module_1.line(var_24, var_3, var_23)
    var_27 = module_0.Config()
    var_28 = 'cimport module.function1, module.function2'
    var_29 = 'cimport module.function1, (\n    module.function2\n)'
    var_30 = module_1.line(var_28, var_3, var_27)
    var_31 = module_0.Config()
    var_32 = 'from module import function1, function2'
    var_33 = 'from module import (\n    function1,\n    function2,\n)'
    var_34 = module_1.line(var_32, var_3, var_31)
    var_35 = 'from module import function1, function2, function3'
    var_36 = 'from module import function1, function2, function3  # NOQA'
    var_37 = module_1.line(var_35, var_3, var_31)
    var_38 = module_0.Config()
    var_39 = 'import module'
    var_40 = module_1.line(var_39, var_3, var_38)
    var_41 = module_0.Config()
    var_42 = ''
    var_43 = module_1.line(var_42, var_3, var_41)
    var_44 = module_0.Config()
    var_45 = '# comment'
    var_46 = module_1.line(var_45, var_3, var_44)
    var_47 = module_0.Config()
    var_48 = 'from module import function1, function2, function3'
    var_49 = 'from module import (\r\n    function1,\r\n    function2,\r\n    function3,\r\n)'
    var_50 = '\r\n'
    var_51 = module_1.line(var_48, var_50, var_47)



# Parsed testcases at query #19
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import something'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'from module import something'
    var_3 = 'from module import something_very_long, another_thing, third_item'
    var_4 = 30
    var_5 = module_1.Config()
    var_6 = module_0.line(var_3, var_1, var_5)
    var_7 = 'from module import something  # comment'
    var_8 = module_0.line(var_7, var_1, var_5)
    var_9 = 'from module import something_very_long, another_thing, third_item'
    var_10 = '# NOQA'
    var_11 = True
    var_12 = module_1.Config()
    var_13 = module_0.line(var_3, var_1, var_12)
    var_14 = module_1.Config()
    var_15 = module_0.line(var_3, var_1, var_14)
    var_16 = module_1.Config()
    var_17 = module_0.line(var_3, var_1, var_16)



# Parsed testcases at query #20
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import function'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = 'from module import very_long_function_name_that_exceeds_line_length'
    var_4 = 30
    var_5 = module_1.Config()
    var_6 = module_0.line(var_3, var_1, var_5)
    assert var_6 == 'from module import (\n    very_long_function_name_that_exceeds_line_length\n)'
    var_7 = 'from module import function  # some comment'
    var_8 = module_0.line(var_7, var_1, var_5)
    assert var_8 == 'from module import (\n    function,  # some comment\n)'
    var_9 = 'from module import function  # NOQA'
    var_10 = module_0.line(var_9, var_1, var_5)
    var_11 = 'from module import function  # noqa'
    var_12 = module_0.line(var_11, var_1, var_5)
    assert var_12 == 'from module import (\n    function,  # noqa\n)'
    var_13 = 'from module import function as alias'
    var_14 = module_0.line(var_13, var_1, var_5)
    assert var_14 == 'from module import function as (\n    alias\n)'
    var_15 = 'from module.submodule import function'
    var_16 = module_0.line(var_15, var_1, var_5)
    assert var_16 == 'from module.submodule import (\n    function\n)'
    var_17 = 'import module.function'
    var_18 = module_0.line(var_17, var_1, var_5)
    assert var_18 == 'import (\n    module.function\n)'
    var_19 = 'cimport module.function'
    var_20 = module_0.line(var_19, var_1, var_5)
    assert var_20 == 'cimport (\n    module.function\n)'
    var_21 = False
    var_22 = module_1.Config()
    var_23 = module_0.line(var_3, var_1, var_22)
    assert var_23 == 'from module import \\\n    very_long_function_name_that_exceeds_line_length'
    var_24 = module_1.Config()
    var_25 = module_0.line(var_3, var_1, var_24)
    assert var_25 == 'from module import (\n    very_long_function_name_that_exceeds_line_length\n)'
    var_26 = '# '
    var_27 = module_1.Config()
    var_28 = module_0.line(var_7, var_1, var_27)
    assert var_28 == 'from module import (\n    function,  # some comment\n)'
    var_29 = True
    var_30 = module_1.Config()
    var_31 = module_0.line(var_7, var_1, var_30)
    assert var_31 == 'from module import (\n    very_long_function_name_that_exceeds_line_length\n)'
    var_32 = module_1.Config()
    var_33 = module_0.line(var_3, var_1, var_32)
    assert var_33 == 'from module import (\n    very_long_function_name_that_exceeds_line_length\n)'
    var_34 = '\r\n'
    var_35 = module_0.line(var_3, var_34, var_5)
    assert var_35 == 'from module import (\r\n    very_long_function_name_that_exceeds_line_length\r\n)'
    var_36 = 'from module import f'
    var_37 = module_0.line(var_36, var_1, var_5)
    var_38 = 'from module import function'
    var_39 = 28
    var_40 = module_1.Config()
    var_41 = module_0.line(var_38, var_1, var_40)
    var_42 = 'from module import function1'
    var_43 = module_0.line(var_42, var_1, var_40)
    assert var_43 == 'from module import (\n    function1\n)'
    var_44 = ''
    var_45 = module_0.line(var_44, var_1, var_5)
    var_46 = '   '
    var_47 = module_0.line(var_46, var_1, var_5)
    var_48 = 'import module.function'
    var_49 = module_0.line(var_48, var_1, var_5)
    assert var_49 == 'import (\n    module.function\n)'
    var_50 = 'from module.submodule import function as alias'
    var_51 = module_0.line(var_50, var_1, var_5)
    assert var_51 == 'from module.submodule import function as (\n    alias\n)'
    var_52 = 'from module import function  # some comment noqa'
    var_53 = module_0.line(var_52, var_1, var_5)
    assert var_53 == 'from module import (\n    function,  # some comment noqa\n)'
    var_54 = module_0.line(var_52, var_1, var_5)
    assert var_54 == 'from module import (\n    function,  # some comment noqa\n)'
    var_55 = module_0.line(var_52, var_1, var_22)
    assert var_55 == 'from module import \\\n    function  # some comment noqa'
    var_56 = module_0.line(var_52, var_1, var_5)
    assert var_56 == 'from module import (\n    function,  # some comment noqa\n)'
    var_57 = module_0.line(var_52, var_1, var_24)
    assert var_57 == 'from module import (\n    function  # some comment noqa\n)'



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1
import re as module_2

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = 'from module import very_long_function_name_that_exceeds_line_length'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    var_5 = len(var_4)
    var_6 = 'import os  # This is a comment'
    var_7 = module_1.line(var_6, var_3, var_1)
    var_8 = 'import os  # NOQA'
    var_9 = module_1.line(var_8, var_3, var_1)
    var_10 = 'import os  # some comment noqa'
    var_11 = module_1.line(var_10, var_3, var_1)
    var_12 = 'import module as m'
    var_13 = module_1.line(var_12, var_3, var_1)
    var_14 = 'from module import submodule.function'
    var_15 = module_1.line(var_14, var_3, var_1)
    var_16 = 0
    var_17 = result.split(var_3)[var_16]
    var_18 = len(var_17)
    var_19 = 'cimport module'
    var_20 = module_1.line(var_19, var_3, var_1)
    var_21 = 'from module import function1, function2, function3'
    var_22 = module_1.line(var_21, var_3, var_1)
    var_23 = 'from module import function1, function2, function3'
    var_24 = module_1.line(var_23, var_3, var_1)
    var_25 = ','
    var_26 = 'from module import function1, function2, function3'
    var_27 = module_1.line(var_26, var_3, var_1)
    var_28 = module_2.split(var_3)
    var_29 = -1
    var_30 = var_28[var_29]
    var_31 = len(var_30)
    var_32 = -1
    var_33 = var_28[:var_32]
    var_34 = min(var_9)
    var_35 = 'import os  # This comment should be ignored'
    var_36 = module_1.line(var_35, var_30, var_1)
    var_37 = 'from module import function1, function2, function3'
    var_38 = '\n'
    var_39 = module_1.line(var_37, var_38, var_1)



# Parsed testcases at query #2
#--------------------------


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'item1'
    var_2 = 'item2'
    var_3 = 'item3'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    var_6 = [var_1, var_2]
    var_7 = '# comment1'
    var_8 = '# comment2'
    var_9 = [var_7, var_8]
    var_10 = module_0.import_statement(var_0, var_6, var_9)
    var_11 = [var_1, var_2]
    var_12 = '\r\n'
    var_13 = module_0.import_statement(var_0, var_11, line_separator=var_12)
    var_14 = [var_1, var_2, var_3]
    var_15 = True
    var_16 = module_0.import_statement(var_0, var_14, explode=var_15)
    var_17 = 50
    var_18 = [var_1, var_2, var_3]
    var_19 = [var_1, var_2, var_3]
    var_20 = [var_1, var_2, var_3]



# Parsed testcases at query #3
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1
import re as module_2

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'item1'
    var_2 = 'item2'
    var_3 = 'item3'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    var_6 = 'from module import item1, item2, item3'
    var_7 = [var_1, var_2]
    var_8 = '# comment1'
    var_9 = '# comment2'
    var_10 = [var_8, var_9]
    var_11 = module_0.import_statement(var_0, var_7, var_10)
    var_12 = [var_1, var_2]
    var_13 = '\r\n'
    var_14 = module_0.import_statement(var_0, var_12, line_separator=var_13)
    var_15 = [var_1, var_2, var_3]
    var_16 = True
    var_17 = module_0.import_statement(var_0, var_15, explode=var_16)
    var_18 = 50
    var_19 = module_1.Config()
    var_20 = [var_1, var_2, var_3]
    var_21 = module_0.import_statement(var_0, var_20, config=var_19)
    var_22 = '\n'
    var_23 = 30
    var_24 = module_1.Config()
    var_25 = [var_1, var_2, var_3]
    var_26 = module_0.import_statement(var_0, var_25, config=var_24)
    var_27 = module_2.split(var_22)
    var_28 = len(var_27)
    var_29 = -1
    var_30 = var_27[var_29]
    var_31 = len(var_30)
    var_32 = -1
    var_33 = var_27[:var_32]
    var_34 = [var_1, var_2, var_3]
    var_35 = [var_1]
    var_36 = module_0.import_statement(var_0, var_35)
    assert var_36 == 'from module import item1'



# Parsed testcases at query #4
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1
import re as module_2

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'A'
    var_2 = 'B'
    var_3 = 'C'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    var_6 = 'from module import A, B, C'
    var_7 = 'D'
    var_8 = 'E'
    var_9 = [var_1, var_2, var_3, var_7, var_8]
    var_10 = 20
    var_11 = module_1.Config()
    var_12 = module_0.import_statement(var_0, var_9, config=var_11)
    var_13 = [var_1, var_2, var_3]
    var_14 = '# Comment'
    var_15 = [var_14]
    var_16 = module_0.import_statement(var_0, var_13, var_15)
    var_17 = [var_1, var_2, var_3]
    var_18 = '\r\n'
    var_19 = module_0.import_statement(var_0, var_17, line_separator=var_18)
    var_20 = [var_1, var_2, var_3]
    var_21 = True
    var_22 = module_0.import_statement(var_0, var_20, explode=var_21)
    var_23 = '\n'
    var_24 = 30
    var_25 = module_1.Config()
    var_26 = [var_1, var_2, var_3, var_7]
    var_27 = module_0.import_statement(var_0, var_26, config=var_25)
    var_28 = module_2.split(var_23)
    var_29 = -1
    var_30 = var_28[var_29]
    var_31 = len(var_30)
    var_32 = -1
    var_33 = var_28[:var_32]
    var_34 = module_1.Config()
    var_35 = [var_1, var_2, var_3]
    var_36 = module_0.import_statement(var_0, var_35, config=var_34, explode=var_21)
    var_37 = ','
    var_38 = 'from module import'
    var_39 = 'A'
    var_40 = 'B'
    var_41 = 'C'
    var_42 = 'D'
    var_43 = 'E'
    var_44 = [var_39, var_40, var_41, var_42, var_43]
    var_45 = 20
    var_46 = module_1.Config()
    var_47 = module_1.Config()
    var_48 = [var_39, var_40, var_41]
    var_49 = [var_14]
    var_50 = module_0.import_statement(var_38, var_48, var_49, config=var_47)
    var_51 = '    '
    var_52 = module_1.Config()
    var_53 = [var_39, var_40, var_41]
    var_54 = module_0.import_statement(var_38, var_53, config=var_52, explode=var_21)
    var_55 = 'from module import (\n    A,'
    var_56 = '# '
    var_57 = module_1.Config()
    var_58 = [var_39, var_40, var_41]
    var_59 = 'Comment'
    var_60 = [var_59]
    var_61 = module_0.import_statement(var_38, var_58, var_60, config=var_57)



# Parsed testcases at query #5
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1
import re as module_2

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'func1'
    var_2 = 'func2'
    var_3 = 'func3'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    var_6 = [var_1, var_2]
    var_7 = '# Comment 1'
    var_8 = '# Comment 2'
    var_9 = [var_7, var_8]
    var_10 = module_0.import_statement(var_0, var_6, var_9)
    var_11 = [var_1, var_2]
    var_12 = '\r\n'
    var_13 = module_0.import_statement(var_0, var_11, line_separator=var_12)
    var_14 = [var_1, var_2, var_3]
    var_15 = True
    var_16 = module_0.import_statement(var_0, var_14, explode=var_15)
    var_17 = '\n'
    var_18 = [var_1, var_2, var_3]
    var_19 = len(var_18)
    var_20 = 50
    var_21 = module_1.Config()
    var_22 = [var_1, var_2, var_3]
    var_23 = module_0.import_statement(var_0, var_22, config=var_21)
    var_24 = 0
    var_25 = result.split(var_17)[var_24]
    var_26 = len(var_25)
    var_27 = [var_1, var_2, var_3]
    var_28 = [var_1, var_2, var_3]
    var_29 = module_1.Config()
    var_30 = module_0.import_statement(var_0, var_28, config=var_29)
    var_31 = module_2.split(var_17)
    var_32 = -1
    var_33 = var_31[:var_32]
    var_34 = min(var_2)
    var_35 = -1
    var_36 = var_31[var_35]
    var_37 = len(var_36)



# Parsed testcases at query #6
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1
import re as module_2

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'A'
    var_2 = 'B'
    var_3 = [var_1, var_2]
    var_4 = module_0.import_statement(var_0, var_3)
    assert var_4 == 'from module import A, B'
    var_5 = 20
    var_6 = module_1.Config()
    var_7 = 'C'
    var_8 = [var_1, var_2, var_7]
    var_9 = module_0.import_statement(var_0, var_8, config=var_6)
    var_10 = [var_1, var_2, var_7]
    var_11 = True
    var_12 = module_0.import_statement(var_0, var_10, explode=var_11)
    var_13 = '\n'
    var_14 = [var_1, var_2]
    var_15 = '# Comment 1'
    var_16 = '# Comment 2'
    var_17 = [var_15, var_16]
    var_18 = module_0.import_statement(var_0, var_14, var_17)
    var_19 = 30
    var_20 = module_1.Config()
    var_21 = 'D'
    var_22 = [var_1, var_2, var_7, var_21]
    var_23 = module_0.import_statement(var_0, var_22, config=var_20)
    var_24 = module_2.split(var_13)
    var_25 = len(var_24)
    var_26 = -1
    var_27 = var_24[var_26]
    var_28 = len(var_27)
    var_29 = -1
    var_30 = var_24[:var_29]
    var_31 = [var_1, var_2]
    var_32 = '\r\n'
    var_33 = module_0.import_statement(var_0, var_31, line_separator=var_32)
    var_34 = module_1.Config()
    var_35 = [var_1, var_2]
    var_36 = module_0.import_statement(var_0, var_35, config=var_34)
    var_37 = ','
    var_38 = 'from module import'
    var_39 = 'A'
    var_40 = 'B'
    var_41 = 'C'
    var_42 = [var_39, var_40, var_41]
    var_43 = 20
    var_44 = module_1.Config()
    var_45 = []
    var_46 = module_0.import_statement(var_38, var_45)
    assert var_46 == 'from module import'
    var_47 = [var_39]
    var_48 = module_0.import_statement(var_38, var_47)
    assert var_48 == 'from module import A'
    var_49 = 'very_long_module_name_1'
    var_50 = 'very_long_module_name_2'
    var_51 = [var_49, var_50]
    var_52 = module_1.Config()
    var_53 = module_0.import_statement(var_38, var_51, config=var_52)



# Parsed testcases at query #7
#--------------------------


import re as module_0

def test_case_0():
    var_0 = 80
    var_1 = 'from module import very_long_function_name_that_exceeds_line_length'
    var_2 = '\n'
    var_3 = 0
    var_4 = result.split(var_2)[var_3]
    var_5 = len(var_4)
    var_6 = 'from module import func  # some comment'
    var_7 = 'from module import func  # NOQA'
    var_8 = 'from module import func  # some noqa comment'
    var_9 = True
    var_10 = 'from module import very_long_function_name_that_exceeds_line_length, another_func'
    var_11 = ','
    var_12 = 'from module import very_long_function_name as vlf'
    var_13 = 'from module.submodule import func'
    var_14 = 'from module import func'
    var_15 = module_0.split(var_2)
    var_16 = -1
    var_17 = var_15[:var_16]
    var_18 = min(var_2)
    var_19 = -1
    var_20 = var_15[var_19]
    var_21 = len(var_20)



# Parsed testcases at query #8
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1
import re as module_2

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'item1'
    var_2 = 'item2'
    var_3 = 'item3'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    var_6 = [var_1, var_2]
    var_7 = '# comment1'
    var_8 = '# comment2'
    var_9 = [var_7, var_8]
    var_10 = module_0.import_statement(var_0, var_6, var_9)
    var_11 = [var_1, var_2]
    var_12 = '\r\n'
    var_13 = module_0.import_statement(var_0, var_11, line_separator=var_12)
    var_14 = [var_1, var_2]
    var_15 = True
    var_16 = module_0.import_statement(var_0, var_14, explode=var_15)
    var_17 = '\n'
    var_18 = 80
    var_19 = module_1.Config()
    var_20 = [var_1, var_2, var_3]
    var_21 = module_0.import_statement(var_0, var_20, config=var_19)
    var_22 = module_2.split(var_17)
    var_23 = -1
    var_24 = var_22[var_23]
    var_25 = len(var_24)
    var_26 = -1
    var_27 = var_22[:var_26]
    var_28 = module_1.Config()
    var_29 = [var_1, var_2]
    var_30 = module_0.import_statement(var_0, var_29, config=var_28)
    var_31 = ','
    var_32 = module_1.Config()
    var_33 = [var_1, var_2]
    var_34 = '# comment'
    var_35 = [var_34]
    var_36 = module_0.import_statement(var_0, var_33, var_35, config=var_32)
    var_37 = '    '
    var_38 = module_1.Config()
    var_39 = [var_1, var_2]
    var_40 = module_0.import_statement(var_0, var_39, config=var_38)
    var_41 = 'from module import\n    '
    var_42 = [var_1, var_2]
    var_43 = []
    var_44 = module_0.import_statement(var_0, var_43)
    assert var_44 == 'from module import'
    var_45 = [var_1]
    var_46 = module_0.import_statement(var_0, var_45)
    assert var_46 == 'from module import item1'



# Parsed testcases at query #9
#--------------------------


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'func1'
    var_2 = 'func2'
    var_3 = [var_1, var_2]
    var_4 = module_0.import_statement(var_0, var_3)
    assert var_4 == 'from module import func1, func2'
    var_5 = 20
    var_6 = True
    var_7 = 'func3'
    var_8 = [var_1, var_2, var_7]
    var_9 = [var_1, var_2]
    var_10 = '# Comment 1'
    var_11 = '# Comment 2'
    var_12 = [var_10, var_11]
    var_13 = module_0.import_statement(var_0, var_9, var_12)
    var_14 = [var_1, var_2, var_7]
    var_15 = module_0.import_statement(var_0, var_14, explode=var_6)
    var_16 = [var_1, var_2, var_7]
    var_17 = [var_1, var_2]
    var_18 = '\r\n'
    var_19 = module_0.import_statement(var_0, var_17, line_separator=var_18)
    var_20 = []
    var_21 = module_0.import_statement(var_0, var_20)
    assert var_21 == 'from module import'
    var_22 = [var_1]
    var_23 = module_0.import_statement(var_0, var_22)
    assert var_23 == 'from module import func1'



# Parsed testcases at query #10
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = 'from module import function1, function2, function3'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    var_5 = 'from module import function1, function2  # some comment'
    var_6 = module_1.line(var_5, var_3, var_1)
    var_7 = 'from module import function1, function2, function3, function4, function5'
    var_8 = module_1.line(var_7, var_3, var_1)
    var_9 = 'from module import function1, function2, function3  # NOQA'
    var_10 = module_1.line(var_9, var_3, var_1)
    var_11 = 'import module as alias'
    var_12 = module_1.line(var_11, var_3, var_1)
    var_13 = 'from module.submodule import function'
    var_14 = module_1.line(var_13, var_3, var_1)
    var_15 = 'cimport cython_module'
    var_16 = module_1.line(var_15, var_3, var_1)
    var_17 = True
    var_18 = module_0.Config()
    var_19 = 'from module import function1, function2'
    var_20 = module_1.line(var_19, var_3, var_18)
    var_21 = ','
    var_22 = module_0.Config()
    var_23 = 'from module import function1, function2, function3'
    var_24 = module_1.line(var_23, var_3, var_22)
    var_25 = 'from module import function1, function2, function3'
    var_26 = 'from module import function1, function2, function3'
    var_27 = 'from module import function1, function2  # noqa: F401'
    var_28 = module_1.line(var_27, var_3, var_1)



# Parsed testcases at query #11
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import function'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = 'from module import very_long_function_name'
    var_4 = 20
    var_5 = module_1.Config()
    var_6 = module_0.line(var_3, var_1, var_5)
    assert var_6 == 'from module import (\n    very_long_function_name\n)'
    var_7 = 'from module import function  # comment'
    var_8 = module_0.line(var_7, var_1, var_5)
    assert var_8 == 'from module import (\n    function,  # comment\n)'
    var_9 = 'from module import function  # NOQA'
    var_10 = module_0.line(var_9, var_1, var_5)
    var_11 = 'import module as alias'
    var_12 = module_0.line(var_11, var_1, var_5)
    assert var_12 == 'import module\n    as alias'
    var_13 = 'cimport module.function'
    var_14 = module_0.line(var_13, var_1, var_5)
    assert var_14 == 'cimport module\n    .function'
    var_15 = 'from module import function  # noqa: F401'
    var_16 = module_0.line(var_15, var_1, var_5)
    assert var_16 == 'from module import (\n    function,  # noqa: F401\n)'
    var_17 = False
    var_18 = module_1.Config()
    var_19 = module_0.line(var_3, var_1, var_18)
    assert var_19 == 'from module import \\\n    very_long_function_name'
    var_20 = module_1.Config()
    var_21 = module_0.line(var_7, var_1, var_20)
    assert var_21 == 'from module import (\n    function  # comment\n)'
    var_22 = True
    var_23 = module_1.Config()
    var_24 = module_0.line(var_7, var_1, var_23)
    assert var_24 == 'from module import (\n    very_long_function_name\n)'



# Parsed testcases at query #12
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import very_long_function_name'
    var_1 = 20
    var_2 = module_0.Config()
    var_3 = '\n'
    var_4 = module_1.line(var_0, var_3, var_2)
    assert var_4 == 'from module import (\n    very_long_function_name\n)'
    var_5 = 'from module import func  # some comment'
    var_6 = True
    var_7 = module_0.Config()
    var_8 = module_1.line(var_5, var_3, var_7)
    assert var_8 == 'from module import (\n    func,  # some comment\n)'
    var_9 = 'from module import very_long_function_name  # NOQA'
    var_10 = module_1.line(var_9, var_3, var_7)
    var_11 = 'from module import func'
    var_12 = 50
    var_13 = module_0.Config()
    var_14 = module_1.line(var_11, var_3, var_13)
    var_15 = 'from module import very_long_function_name as vlf'
    var_16 = 30
    var_17 = module_0.Config()
    var_18 = module_1.line(var_15, var_3, var_17)
    assert var_18 == 'from module import (\n    very_long_function_name as vlf\n)'
    var_19 = 'cimport module.very_long_function_name'
    var_20 = module_0.Config()
    var_21 = module_1.line(var_19, var_3, var_20)
    assert var_21 == 'cimport (\n    module.very_long_function_name\n)'
    var_22 = 'from module import func1, func2'
    var_23 = module_0.Config()
    var_24 = module_1.line(var_22, var_3, var_23)
    assert var_24 == 'from module import (\n    func1,\n    func2,\n)'
    var_25 = 'from module import func1, func2'
    var_26 = module_1.line(var_25, var_3, var_23)
    assert var_26 == 'from module import (\n    func1,\n    func2,\n)'
    var_27 = 'from module import func1, func2'
    var_28 = module_1.line(var_27, var_3, var_23)
    assert var_28 == 'from module import (\n    func1,\n    func2,\n)'
    var_29 = 'from module import func  # noqa'
    var_30 = module_0.Config()
    var_31 = module_1.line(var_29, var_3, var_30)
    assert var_31 == 'from module import (\n    func,  # noqa\n)'



# Parsed testcases at query #13
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'item1'
    var_2 = 'item2'
    var_3 = 'item3'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    var_6 = [var_1, var_2]
    var_7 = '# Comment 1'
    var_8 = '# Comment 2'
    var_9 = [var_7, var_8]
    var_10 = module_0.import_statement(var_0, var_6, var_9)
    var_11 = [var_1, var_2]
    var_12 = '\r\n'
    var_13 = module_0.import_statement(var_0, var_11, line_separator=var_12)
    var_14 = [var_1, var_2, var_3]
    var_15 = True
    var_16 = module_0.import_statement(var_0, var_14, explode=var_15)
    var_17 = '\n'
    var_18 = [var_1, var_2, var_3]
    var_19 = 50
    var_20 = '    '
    var_21 = module_1.Config()
    var_22 = [var_1, var_2, var_3]
    var_23 = module_0.import_statement(var_0, var_22, config=var_21)
    var_24 = 20
    var_25 = module_1.Config()
    var_26 = [var_1, var_2, var_3]
    var_27 = module_0.import_statement(var_0, var_26, config=var_25)
    var_28 = [var_1]
    var_29 = module_0.import_statement(var_0, var_28)



# Parsed testcases at query #14
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import very_long_function_name'
    var_1 = 20
    var_2 = module_0.Config()
    var_3 = '\n'
    var_4 = module_1.line(var_0, var_3, var_2)
    assert var_4 == 'from module import (\n    very_long_function_name\n)'
    var_5 = 'from module import func  # some comment'
    var_6 = True
    var_7 = module_0.Config()
    var_8 = module_1.line(var_5, var_3, var_7)
    assert var_8 == 'from module import (\n    func,  # some comment\n)'
    var_9 = 'from module import very_long_function_name'
    var_10 = module_1.line(var_9, var_3, var_7)
    assert var_10 == 'from module import very_long_function_name  # NOQA'
    var_11 = 'from module import very_long_function_name as vlf'
    var_12 = 30
    var_13 = module_0.Config()
    var_14 = module_1.line(var_11, var_3, var_13)
    assert var_14 == 'from module import (\n    very_long_function_name as vlf\n)'
    var_15 = 'from module.submodule import function'
    var_16 = module_0.Config()
    var_17 = module_1.line(var_15, var_3, var_16)
    assert var_17 == 'from module.submodule import (\n    function\n)'
    var_18 = 'cimport module.very_long_function_name'
    var_19 = module_0.Config()
    var_20 = module_1.line(var_18, var_3, var_19)
    assert var_20 == 'cimport (\n    module.very_long_function_name\n)'
    var_21 = 'from module import func'
    var_22 = module_0.Config()
    var_23 = module_1.line(var_21, var_3, var_22)
    assert var_23 == 'from module import func'
    var_24 = 'from module import func1, func2'
    var_25 = module_0.Config()
    var_26 = module_1.line(var_24, var_3, var_25)
    assert var_26 == 'from module import (\n    func1,\n    func2,\n)'
    var_27 = 'from module import func  # noqa'
    var_28 = module_0.Config()
    var_29 = module_1.line(var_27, var_3, var_28)
    assert var_29 == 'from module import func  # noqa'
    var_30 = 'from module import func1, func2, func3'
    var_31 = module_0.Config()
    var_32 = module_1.line(var_30, var_3, var_31)
    assert var_32 == 'from module import (\n    func1,\n    func2,\n    func3,\n)'



# Parsed testcases at query #15
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'func1'
    var_2 = 'func2'
    var_3 = [var_1, var_2]
    var_4 = module_0.import_statement(var_0, var_3)
    var_5 = [var_1, var_2]
    var_6 = True
    var_7 = module_0.import_statement(var_0, var_5, explode=var_6)
    var_8 = [var_1, var_2]
    var_9 = '# Comment'
    var_10 = [var_9]
    var_11 = module_0.import_statement(var_0, var_8, var_10)
    var_12 = [var_1, var_2]
    var_13 = '\r\n'
    var_14 = module_0.import_statement(var_0, var_12, line_separator=var_13)
    var_15 = 20
    var_16 = module_1.Config()
    var_17 = [var_1, var_2]
    var_18 = module_0.import_statement(var_0, var_17, config=var_16)
    var_19 = module_1.Config()
    var_20 = [var_1, var_2]
    var_21 = module_0.import_statement(var_0, var_20, config=var_19)
    var_22 = [var_1, var_2]
    var_23 = [var_1]
    var_24 = module_0.import_statement(var_0, var_23)
    var_25 = []
    var_26 = module_0.import_statement(var_0, var_25)
    var_27 = 'func3'
    var_28 = 'func4'
    var_29 = 'func5'
    var_30 = [var_1, var_2, var_27, var_28, var_29]
    var_31 = module_0.import_statement(var_0, var_30)



# Parsed testcases at query #16
#--------------------------


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'item1'
    var_2 = 'item2'
    var_3 = 'item3'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    var_6 = [var_1, var_2]
    var_7 = '# Comment 1'
    var_8 = '# Comment 2'
    var_9 = [var_7, var_8]
    var_10 = module_0.import_statement(var_0, var_6, var_9)
    var_11 = [var_1, var_2]
    var_12 = '\r\n'
    var_13 = module_0.import_statement(var_0, var_11, line_separator=var_12)
    var_14 = [var_1, var_2]
    var_15 = True
    var_16 = module_0.import_statement(var_0, var_14, explode=var_15)
    var_17 = 50
    var_18 = [var_1, var_2]
    var_19 = [var_1, var_2]
    var_20 = [var_1, var_2]



# Parsed testcases at query #17
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = 'from module import function1, function2, function3'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    var_5 = 'from module import function1, function2  # some comment'
    var_6 = module_1.line(var_5, var_3, var_1)
    var_7 = 'from module import function1, function2, function3  # NOQA'
    var_8 = module_1.line(var_7, var_3, var_1)
    var_9 = 'from module import function1, function2, function3, function4, function5'
    var_10 = module_1.line(var_9, var_3, var_1)
    var_11 = True
    var_12 = module_0.Config()
    var_13 = module_1.line(var_9, var_3, var_12)
    var_14 = module_0.Config()
    var_15 = module_1.line(var_9, var_3, var_14)
    var_16 = ','
    var_17 = module_0.Config()
    var_18 = module_1.line(var_9, var_3, var_17)
    var_19 = ' # '
    var_20 = module_0.Config()
    var_21 = 'from module import function1, function2  # some comment'
    var_22 = module_1.line(var_21, var_3, var_20)
    var_23 = module_0.Config()
    var_24 = 'from module import function1, function2  # some comment'
    var_25 = module_1.line(var_24, var_3, var_23)



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'from module import something, something_else, another_thing'
    var_1 = 50
    var_2 = '\n'
    var_3 = 'from module import something  # some comment'
    var_4 = 'from module import something, something_else, another_thing  # NOQA'
    var_5 = 30
    var_6 = 'from module import something as alias'
    var_7 = 'from module.submodule import something'
    var_8 = 'cimport module.something'
    var_9 = True
    var_10 = 'from module import something, something_else, another_thing'
    var_11 = 'from module import something, something_else, another_thing'
    var_12 = ','
    var_13 = 'from module import something'
    var_14 = 'from module import something, something_else, another_thing'
    var_15 = 'from module import something  # some comment'



# Parsed testcases at query #19
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1
import re as module_2

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'A'
    var_2 = 'B'
    var_3 = 'C'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    assert var_5 == 'from module import A, B, C'
    var_6 = [var_1, var_2, var_3]
    var_7 = '# Comment'
    var_8 = [var_7]
    var_9 = module_0.import_statement(var_0, var_6, var_8)
    var_10 = [var_1, var_2, var_3]
    var_11 = 20
    var_12 = module_1.Config()
    var_13 = module_0.import_statement(var_0, var_10, config=var_12)
    var_14 = '\n'
    var_15 = [var_1, var_2, var_3]
    var_16 = True
    var_17 = module_0.import_statement(var_0, var_15, explode=var_16)
    var_18 = [var_1, var_2, var_3]
    var_19 = '\r\n'
    var_20 = module_0.import_statement(var_0, var_18, line_separator=var_19)
    var_21 = 30
    var_22 = module_1.Config()
    var_23 = [var_1, var_2, var_3]
    var_24 = module_0.import_statement(var_0, var_23, config=var_22)
    var_25 = module_2.split(var_14)
    var_26 = -1
    var_27 = var_25[var_26]
    var_28 = len(var_27)
    var_29 = -1
    var_30 = var_25[:var_29]
    var_31 = min(var_6)
    var_32 = module_1.Config()
    var_33 = [var_27, var_28, var_29]
    var_34 = module_0.import_statement(var_26, var_33, config=var_32)
    var_35 = ','
    var_36 = [var_27, var_28, var_29]
    var_37 = []
    var_38 = module_0.import_statement(var_26, var_37)
    assert var_38 == 'from module import'
    var_39 = [var_27]
    var_40 = module_0.import_statement(var_26, var_39)
    assert var_40 == 'from module import A'
    var_41 = 'VERY_LONG_NAME_A'
    var_42 = 'VERY_LONG_NAME_B'
    var_43 = [var_41, var_42]
    var_44 = module_1.Config()
    var_45 = module_0.import_statement(var_26, var_43, config=var_44)



# Parsed testcases at query #20
#--------------------------


import isort.wrap as module_0
import re as module_1
import isort.settings as module_2

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'A'
    var_2 = 'B'
    var_3 = 'C'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    var_6 = 20
    var_7 = True
    var_8 = 'very_long_name_a'
    var_9 = 'very_long_name_b'
    var_10 = 'very_long_name_c'
    var_11 = [var_8, var_9, var_10]
    var_12 = '\n'
    var_13 = [var_1, var_2, var_3]
    var_14 = '# Comment 1'
    var_15 = '# Comment 2'
    var_16 = [var_14, var_15]
    var_17 = module_0.import_statement(var_0, var_13, var_16)
    var_18 = [var_1, var_2, var_3]
    var_19 = module_0.import_statement(var_0, var_18, explode=var_7)
    var_20 = 30
    var_21 = 'short'
    var_22 = 'medium_length'
    var_23 = 'very_long_name'
    var_24 = [var_21, var_22, var_23]
    var_25 = module_1.split(var_12)
    var_26 = len(var_25)
    var_27 = -1
    var_28 = var_25[var_27]
    var_29 = len(var_28)
    var_30 = -1
    var_31 = var_25[:var_30]
    var_32 = [var_1, var_2, var_3]
    var_33 = '\r\n'
    var_34 = module_0.import_statement(var_0, var_32, line_separator=var_33)
    var_35 = module_2.Config()
    var_36 = [var_1, var_2, var_3]
    var_37 = '# This should be ignored'
    var_38 = [var_37]
    var_39 = module_0.import_statement(var_0, var_36, var_38, config=var_35)
    var_40 = '    '
    var_41 = module_2.Config()
    var_42 = [var_1, var_2, var_3]
    var_43 = module_0.import_statement(var_0, var_42, config=var_41)
    var_44 = ' # '
    var_45 = module_2.Config()
    var_46 = [var_1, var_2, var_3]
    var_47 = 'Comment 1'
    var_48 = [var_47]
    var_49 = module_0.import_statement(var_0, var_46, var_48, config=var_45)
    var_50 = False
    var_51 = module_2.Config()
    var_52 = [var_1, var_2, var_3]
    var_53 = module_0.import_statement(var_0, var_52, config=var_51)
    var_54 = ','
    var_55 = [var_8, var_9]
    var_56 = module_0.import_statement(var_0, var_55, config=var_51)
    var_57 = 20
    var_58 = True
    var_59 = 'from module import'
    var_60 = 'A'
    var_61 = 'B'
    var_62 = 'C'
    var_63 = [var_60, var_61, var_62]
    var_64 = module_0.import_statement(var_59, var_63, config=var_51)
    var_65 = []
    var_66 = module_0.import_statement(var_57, var_65)
    var_67 = [var_58]
    var_68 = module_0.import_statement(var_57, var_67)
    var_69 = 1000
    var_70 = module_2.Config()
    var_71 = [var_58, var_59, var_60]
    var_72 = module_0.import_statement(var_57, var_71, config=var_70)



# Parsed testcases at query #21
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1
import re as module_2

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'A'
    var_2 = 'B'
    var_3 = 'C'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    assert var_5 == 'from module import A, B, C'
    var_6 = 'D'
    var_7 = 'E'
    var_8 = [var_1, var_2, var_3, var_6, var_7]
    var_9 = 20
    var_10 = module_1.Config()
    var_11 = module_0.import_statement(var_0, var_8, config=var_10)
    var_12 = [var_1, var_2, var_3]
    var_13 = True
    var_14 = module_0.import_statement(var_0, var_12, explode=var_13)
    assert var_14 == 'from module import (\n    A,\n    B,\n    C,\n)'
    var_15 = [var_1, var_2, var_3]
    var_16 = '# Comment 1'
    var_17 = '# Comment 2'
    var_18 = [var_16, var_17]
    var_19 = module_0.import_statement(var_0, var_15, var_18)
    var_20 = 30
    var_21 = module_1.Config()
    var_22 = [var_1, var_2, var_3, var_6]
    var_23 = module_0.import_statement(var_0, var_22, config=var_21)
    var_24 = '\n'
    var_25 = module_2.split(var_24)
    var_26 = -1
    var_27 = var_25[var_26]
    var_28 = len(var_27)
    var_29 = -1
    var_30 = var_25[:var_29]
    var_31 = [var_1, var_2, var_3]
    var_32 = '\r\n'
    var_33 = module_0.import_statement(var_0, var_31, line_separator=var_32)
    var_34 = module_1.Config()
    var_35 = [var_1, var_2, var_3]
    var_36 = module_0.import_statement(var_0, var_35, config=var_34)
    var_37 = ','
    var_38 = 'from module import'
    var_39 = 'A'
    var_40 = 'B'
    var_41 = 'C'
    var_42 = [var_39, var_40, var_41]
    var_43 = []
    var_44 = module_0.import_statement(var_38, var_43)
    assert var_44 == 'from module import'
    var_45 = 5
    var_46 = range(var_45)
    var_47 = 'very_long_import_name_'
    var_48 = [var_47 + str(i) for i in var_46]
    var_49 = 40
    var_50 = module_1.Config()
    var_51 = module_0.import_statement(var_38, var_48, config=var_50)



# Parsed testcases at query #22
#--------------------------


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import function'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = 'from module import function1, function2, function3, function4'
    var_4 = 30
    var_5 = 'from module import function  # some comment'
    var_6 = 'from module import function1, function2, function3  # NOQA'
    var_7 = 'from module import function1, function2, function3'
    var_8 = True
    var_9 = 'from module import function1, function2, function3'
    var_10 = -1
    var_11 = result.rsplit(var_1, var_8)[var_10]
    var_12 = 'from module import function1 as f1, function2 as f2'
    var_13 = 'cimport module.function1, module.function2'
    var_14 = 'from module import function1, function2, function3.function4'
    var_15 = 'from module import func'
    var_16 = 'from module import function  # some comment'



# Parsed testcases at query #23
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1
import re as module_2

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'function1'
    var_2 = 'function2'
    var_3 = 'function3'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    var_6 = [var_1, var_2]
    var_7 = '# Comment 1'
    var_8 = '# Comment 2'
    var_9 = [var_7, var_8]
    var_10 = module_0.import_statement(var_0, var_6, var_9)
    var_11 = [var_1, var_2]
    var_12 = '\r\n'
    var_13 = module_0.import_statement(var_0, var_11, line_separator=var_12)
    var_14 = [var_1, var_2]
    var_15 = True
    var_16 = module_0.import_statement(var_0, var_14, explode=var_15)
    var_17 = '\n'
    var_18 = [var_1, var_2]
    var_19 = 50
    var_20 = module_1.Config()
    var_21 = 'very_long_function_name1'
    var_22 = 'very_long_function_name2'
    var_23 = [var_21, var_22]
    var_24 = module_0.import_statement(var_0, var_23, config=var_20)
    var_25 = module_2.split(var_17)
    var_26 = len(var_25)
    var_27 = -1
    var_28 = var_25[var_27]
    var_29 = len(var_28)
    var_30 = 0
    var_31 = var_25[var_30]
    var_32 = len(var_31)
    var_33 = module_1.Config()
    var_34 = [var_1, var_2]
    var_35 = module_0.import_statement(var_0, var_34, config=var_33)
    var_36 = ','
    var_37 = module_1.Config()
    var_38 = [var_1, var_2]
    var_39 = [var_7]
    var_40 = module_0.import_statement(var_0, var_38, var_39, config=var_37)



# Parsed testcases at query #24
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import function'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = 'from module import function1, function2, function3, function4'
    var_4 = 30
    var_5 = module_1.Config()
    var_6 = module_0.line(var_3, var_1, var_5)
    var_7 = 0
    var_8 = result.split(var_1)[var_7]
    var_9 = len(var_8)
    var_10 = 'from module import function1, function2, function3  # comment'
    var_11 = module_1.Config()
    var_12 = module_0.line(var_10, var_1, var_11)
    var_13 = result.split(var_1)[var_7]
    var_14 = len(var_13)
    var_15 = 'from module import function1, function2, function3  # NOQA'
    var_16 = module_0.line(var_15, var_1, var_11)
    var_17 = 'from module import function as alias'
    var_18 = 20
    var_19 = module_1.Config()
    var_20 = module_0.line(var_17, var_1, var_19)
    var_21 = result.split(var_1)[var_7]
    var_22 = len(var_21)
    var_23 = 'from module import function1, function2, function3'
    var_24 = True
    var_25 = module_1.Config()
    var_26 = module_0.line(var_23, var_1, var_25)
    var_27 = result.split(var_1)[var_7]
    var_28 = len(var_27)
    var_29 = 'from module import function1, function2, function3'
    var_30 = module_1.Config()
    var_31 = module_0.line(var_29, var_1, var_30)
    var_32 = -2
    var_33 = result.split(var_1)[var_32]



# Parsed testcases at query #25
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import function1, function2, function3'
    var_1 = '\n'
    var_2 = 'from module import function1, function2, function3  # some comment'
    var_3 = 'from module import function1, function2, function3  # NOQA'
    var_4 = 'from module import function1, function2, function3, function4, function5'
    var_5 = True
    var_6 = module_0.Config()
    var_7 = 'from module import function1, function2, function3'
    var_8 = module_1.line(var_7, var_1, var_6)
    var_9 = ','
    var_10 = 'from module import function1, function2, function3'
    var_11 = 'from module import function1 as f1, function2 as f2'
    var_12 = 'from module.submodule import function1, function2'
    var_13 = 'cimport module.function1, module.function2'
    var_14 = module_0.Config()
    var_15 = 'from module import function1, function2, function3'
    var_16 = module_1.line(var_15, var_1, var_14)



