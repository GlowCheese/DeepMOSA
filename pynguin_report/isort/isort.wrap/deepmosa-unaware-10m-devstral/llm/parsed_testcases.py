####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




# Parsed testcases at query #2
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
    var_18 = 20
    var_19 = module_1.Config()
    var_20 = 'very_long_name1'
    var_21 = 'very_long_name2'
    var_22 = [var_20, var_21]
    var_23 = module_0.import_statement(var_0, var_22, config=var_19)
    var_24 = module_2.split(var_17)
    var_25 = len(var_24)
    var_26 = var_25 > var_15
    var_27 = 0
    var_28 = var_24[var_27]
    var_29 = len(var_28)
    var_30 = -1
    var_31 = var_24[var_30]
    var_32 = len(var_31)
    var_33 = var_29 >= var_32
    var_34 = [var_1, var_2, var_3]
    var_35 = module_1.Config()
    var_36 = [var_1, var_2]
    var_37 = module_0.import_statement(var_0, var_36, config=var_35)
    var_38 = ','
    var_39 = module_1.Config()
    var_40 = [var_1, var_2]
    var_41 = '# This should be ignored'
    var_42 = [var_41]
    var_43 = module_0.import_statement(var_0, var_40, var_42, config=var_39)
    var_44 = module_1.Config()
    var_45 = [var_1, var_2, var_3]
    var_46 = module_0.import_statement(var_0, var_45, config=var_44)



# Parsed testcases at query #3
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
    var_21 = 40
    var_22 = '    '
    var_23 = module_1.Config()
    var_24 = [var_1, var_2, var_3]
    var_25 = module_0.import_statement(var_0, var_24, config=var_23)
    var_26 = module_1.Config()
    var_27 = [var_1, var_2, var_3]
    var_28 = module_0.import_statement(var_0, var_27, config=var_26)
    var_29 = module_2.split(var_17)
    var_30 = -1
    var_31 = var_29[:var_30]
    var_32 = min(var_2)
    var_33 = -1
    var_34 = var_29[var_33]
    var_35 = len(var_34)
    var_36 = [var_31, var_2, var_33]
    var_37 = []
    var_38 = module_0.import_statement(var_30, var_37)
    var_39 = [var_31]
    var_40 = module_0.import_statement(var_30, var_39)
    var_41 = 'very_long_function_name_1'
    var_42 = 'very_long_function_name_2'
    var_43 = [var_41, var_42]
    var_44 = 30
    var_45 = module_1.Config()
    var_46 = module_0.import_statement(var_30, var_43, config=var_45)



# Parsed testcases at query #4
#--------------------------


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import function'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = 'from module import very_long_function_name'
    var_4 = 20
    var_5 = 'from module import function  # comment'
    var_6 = 'from module import function  # NOQA'
    var_7 = 'from module import function as alias'
    var_8 = 'from module import (function, other_function)'
    var_9 = True
    var_10 = 'from module import function,'
    var_11 = ','
    var_12 = 'from module import f'
    var_13 = 'from module import function'
    var_14 = '\r\n'
    var_15 = module_0.line(var_13, var_14)
    var_16 = 'from module import very_long_function_name'



# Parsed testcases at query #5
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
    var_7 = 'from module import function  # some comment'
    var_8 = True
    var_9 = module_1.Config()
    var_10 = module_0.line(var_7, var_1, var_9)
    var_11 = 'from module import very_long_function_name'
    var_12 = module_0.line(var_11, var_1, var_9)
    var_13 = 'import module as alias'
    var_14 = 15
    var_15 = module_1.Config()
    var_16 = module_0.line(var_13, var_1, var_15)
    var_17 = 'from module import function  # noqa'
    var_18 = module_1.Config()
    var_19 = module_0.line(var_17, var_1, var_18)
    var_20 = 'from module import function1, function2'
    var_21 = 25
    var_22 = module_1.Config()
    var_23 = module_0.line(var_20, var_1, var_22)
    var_24 = 'from module import function1, function2'
    var_25 = module_0.line(var_24, var_1, var_22)
    var_26 = 'from module import function1, function2'
    var_27 = module_0.line(var_26, var_1, var_22)
    var_28 = 'from module import function  # some comment'
    var_29 = module_1.Config()
    var_30 = module_0.line(var_28, var_1, var_29)



# Parsed testcases at query #6
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import something, something_else, another_thing'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = 'from module import something  # some comment'
    var_4 = module_0.line(var_3, var_1)
    var_5 = 'from module import something, something_else, another_thing  # NOQA'
    var_6 = module_0.line(var_5, var_1)
    var_7 = 'from module import something, something_else, another_thing, yet_another, and_more'
    var_8 = module_0.line(var_7, var_1)
    var_9 = 50
    var_10 = True
    var_11 = module_1.Config()
    var_12 = module_0.line(var_7, var_1, var_11)
    var_13 = 'from module import something as something_else'
    var_14 = module_0.line(var_13, var_1)
    var_15 = 'cimport something, something_else'
    var_16 = module_0.line(var_15, var_1)
    var_17 = 'from module.submodule import something'
    var_18 = module_0.line(var_17, var_1)
    var_19 = 'from module import something, something_else,'
    var_20 = module_0.line(var_19, var_1)
    var_21 = module_1.Config()
    var_22 = module_0.line(var_7, var_1, var_21)
    var_23 = module_1.Config()
    var_24 = module_0.line(var_3, var_1, var_23)
    var_25 = '# '
    var_26 = module_1.Config()
    var_27 = module_0.line(var_3, var_1, var_26)
    var_28 = module_1.Config()
    var_29 = module_0.line(var_7, var_1, var_28)
    var_30 = ','
    var_31 = '\r\n'
    var_32 = module_0.line(var_0, var_31)
    var_33 = 'import something'
    var_34 = module_0.line(var_33, var_1)



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
    var_17 = '\n'
    var_18 = 20
    var_19 = module_1.Config()
    var_20 = 'very_long_item_name1'
    var_21 = 'very_long_item_name2'
    var_22 = [var_20, var_21]
    var_23 = module_0.import_statement(var_0, var_22, config=var_19)
    var_24 = module_2.split(var_17)
    var_25 = 0
    var_26 = var_24[var_25]
    var_27 = len(var_26)
    var_28 = -1
    var_29 = var_24[var_28]
    var_30 = len(var_29)
    var_31 = [var_1, var_2, var_3]
    var_32 = module_1.Config()
    var_33 = [var_1, var_2]
    var_34 = module_0.import_statement(var_0, var_33, config=var_32)
    var_35 = ','
    var_36 = module_1.Config()
    var_37 = [var_1, var_2]
    var_38 = [var_7]
    var_39 = module_0.import_statement(var_0, var_37, var_38, config=var_36)
    var_40 = module_1.Config()
    var_41 = [var_1, var_2]
    var_42 = module_0.import_statement(var_0, var_41, config=var_40)
    var_43 = '# '
    var_44 = module_1.Config()
    var_45 = [var_1, var_2]
    var_46 = 'comment1'
    var_47 = [var_46]
    var_48 = module_0.import_statement(var_0, var_45, var_47, config=var_44)



# Parsed testcases at query #8
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import function'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = 'from module import function # comment'
    var_4 = module_0.line(var_3, var_1)
    var_5 = 'from module import function1, function2, function3'
    var_6 = 30
    var_7 = module_1.Config()
    var_8 = module_0.line(var_5, var_1, var_7)
    assert var_8 == 'from module import (\n    function1,\n    function2,\n    function3,\n)'
    var_9 = 'from module import function1, function2, function3'
    var_10 = module_0.line(var_9, var_1, var_7)
    assert var_10 == 'from module import function1, function2, function3 # NOQA'
    var_11 = 'from module import function1, function2, function3'
    var_12 = True
    var_13 = module_1.Config()
    var_14 = module_0.line(var_11, var_1, var_13)
    assert var_14 == 'from module import (\n    function1,\n    function2,\n    function3,\n)'
    var_15 = 'from module import function1, function2, function3'
    var_16 = module_1.Config()
    var_17 = module_0.line(var_15, var_1, var_16)
    assert var_17 == 'from module import (\n    function1,\n    function2,\n    function3,\n)'
    var_18 = 'from module import function1, function2, function3'
    var_19 = '# '
    var_20 = module_1.Config()
    var_21 = module_0.line(var_18, var_1, var_20)
    assert var_21 == 'from module import (\n    function1,\n    function2,\n    function3,\n)'
    var_22 = 'from module import function1, function2, function3 # comment'
    var_23 = module_1.Config()
    var_24 = module_0.line(var_22, var_1, var_23)
    assert var_24 == 'from module import (\n    function1,\n    function2,\n    function3,\n)'
    var_25 = 'from module import function1, function2, function3'
    var_26 = module_1.Config()
    var_27 = module_0.line(var_25, var_1, var_26)
    assert var_27 == 'from module import (\n    function1,\n    function2,\n    function3,\n)'
    var_28 = 'from module import function1, function2, function3'
    var_29 = module_0.line(var_28, var_1, var_26)
    assert var_29 == 'from module import (\n    function1,\n    function2,\n    function3,\n)'
    var_30 = 'from module import function1, function2, function3'
    var_31 = module_0.line(var_30, var_1, var_26)
    assert var_31 == 'from module import (\n    function1,\n    function2,\n    function3,\n)'
    var_32 = 'from module import function1, function2, function3 # noqa'
    var_33 = module_1.Config()
    var_34 = module_0.line(var_32, var_1, var_33)
    assert var_34 == 'from module import (\n    function1,\n    function2,\n    function3,  # noqa\n)'
    var_35 = 'from module import function1 as f1, function2 as f2'
    var_36 = module_1.Config()
    var_37 = module_0.line(var_35, var_1, var_36)
    assert var_37 == 'from module import (\n    function1 as f1,\n    function2 as f2,\n)'
    var_38 = 'cimport module.function1, module.function2'
    var_39 = module_1.Config()
    var_40 = module_0.line(var_38, var_1, var_39)
    assert var_40 == 'cimport (\n    module.function1,\n    module.function2,\n)'
    var_41 = 'from module import function1, function2, function3'
    var_42 = module_1.Config()
    var_43 = module_0.line(var_41, var_1, var_42)
    assert var_43 == 'from module import (\n    function1,\n    function2,\n    function3,\n)'



# Parsed testcases at query #9
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import function'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'from module import function'
    var_3 = 20
    var_4 = module_1.Config()
    var_5 = 'from module import very_long_function_name'
    var_6 = module_0.line(var_5, var_1, var_4)
    assert var_6 == 'from module import (\n    very_long_function_name\n)'
    var_7 = True
    var_8 = module_1.Config()
    var_9 = 'from module import function  # some comment'
    var_10 = module_0.line(var_9, var_1, var_8)
    assert var_10 == 'from module import (\n    function,  # some comment\n)'
    var_11 = module_0.line(var_0, var_1, var_8)
    assert var_11 == 'from module import function  # NOQA'
    var_12 = module_1.Config()
    var_13 = 'import module as m'
    var_14 = module_0.line(var_13, var_1, var_12)
    assert var_14 == 'import module\n    as m'
    var_15 = module_1.Config()
    var_16 = 'cimport module.function'
    var_17 = module_0.line(var_16, var_1, var_15)
    assert var_17 == 'cimport module\n    .function'
    var_18 = module_1.Config()
    var_19 = 'import module.function'
    var_20 = module_0.line(var_19, var_1, var_18)
    assert var_20 == 'import module\n    .function'
    var_21 = module_1.Config()
    var_22 = module_0.line(var_0, var_1, var_21)
    assert var_22 == 'from module import (\n    function,\n)'
    var_23 = module_1.Config()
    var_24 = 'from module import function  # noqa'
    var_25 = module_0.line(var_24, var_1, var_23)
    assert var_25 == 'from module import (\n    function,  # noqa\n)'
    var_26 = module_0.line(var_0, var_1, var_23)
    assert var_26 == 'from module import (\n    function,\n)'
    var_27 = module_0.line(var_0, var_1, var_23)
    assert var_27 == 'from module import (\n    function,\n)'



# Parsed testcases at query #10
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
    var_7 = 'from module import function  # some comment'
    var_8 = module_0.line(var_7, var_1)
    var_9 = 'from module import very_long_function_name  # NOQA'
    var_10 = module_1.Config()
    var_11 = module_0.line(var_9, var_1, var_10)
    var_12 = 'from module import function as alias'
    var_13 = module_0.line(var_12, var_1)
    var_14 = 'from module.submodule import function'
    var_15 = module_0.line(var_14, var_1)
    var_16 = 'cfrom module import function'
    var_17 = module_0.line(var_16, var_1)
    var_18 = True
    var_19 = module_1.Config()
    var_20 = 'from module import very_long_function_name'
    var_21 = module_0.line(var_20, var_1, var_19)
    var_22 = module_1.Config()
    var_23 = 'from module import very_long_function_name'
    var_24 = module_0.line(var_23, var_1, var_22)
    var_25 = 'from module import very_long_function_name  # some comment'
    var_26 = module_1.Config()
    var_27 = module_0.line(var_25, var_1, var_26)
    var_28 = 'from module import very_long_function_name  # some comment'
    var_29 = module_1.Config()
    var_30 = module_0.line(var_28, var_1, var_29)
    var_31 = 'from module import very_long_function_name  # noqa'
    var_32 = module_1.Config()
    var_33 = module_0.line(var_31, var_1, var_32)
    var_34 = 'from module import very_long_function_name  # noqa'
    var_35 = module_1.Config()
    var_36 = module_0.line(var_34, var_1, var_35)
    var_37 = 'from module import very_long_function_name  # noqa: F401'
    var_38 = module_1.Config()
    var_39 = module_0.line(var_37, var_1, var_38)
    var_40 = 'from module import very_long_function_name  # noqa: F401'
    var_41 = module_1.Config()
    var_42 = module_0.line(var_40, var_1, var_41)
    var_43 = 'from module import very_long_function_name  # noqa: F401'
    var_44 = module_1.Config()
    var_45 = module_0.line(var_43, var_1, var_44)
    var_46 = 'from module import very_long_function_name  # noqa: F401'
    var_47 = module_0.line(var_46, var_1, var_44)
    var_48 = 'from module import very_long_function_name  # noqa: F401'
    var_49 = ' # '
    var_50 = module_0.line(var_48, var_1, var_44)
    var_51 = 'from module import very_long_function_name  # noqa: F401'
    var_52 = module_0.line(var_51, var_1, var_44)



# Parsed testcases at query #11
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import function1, function2, function3'
    var_1 = 50
    var_2 = module_0.Config()
    var_3 = '\n'
    var_4 = module_1.line(var_0, var_3, var_2)
    var_5 = 'from module import function1, function2, function3  # some comment'
    var_6 = module_1.line(var_5, var_3, var_2)
    var_7 = 'from module import function1, function2, function3  # NOQA'
    var_8 = 30
    var_9 = module_1.line(var_7, var_3, var_2)
    var_10 = 'from module import function1 as f1, function2 as f2'
    var_11 = module_1.line(var_10, var_3, var_2)
    var_12 = 'from module.submodule import function1, function2'
    var_13 = module_1.line(var_12, var_3, var_2)
    var_14 = True
    var_15 = module_0.Config()
    var_16 = 'from module import function1, function2, function3'
    var_17 = module_1.line(var_16, var_3, var_15)
    var_18 = module_0.Config()
    var_19 = 'from module import function1, function2, function3'
    var_20 = module_1.line(var_19, var_3, var_18)
    var_21 = ','
    var_22 = 'from module import function1, function2, function3'
    var_23 = module_1.line(var_22, var_3, var_18)
    var_24 = 'from module import function1, function2, function3'
    var_25 = module_1.line(var_24, var_3, var_18)
    var_26 = module_0.Config()
    var_27 = 'from module import function1, function2, function3  # some comment'
    var_28 = module_1.line(var_27, var_3, var_26)
    var_29 = ' # '
    var_30 = module_0.Config()
    var_31 = 'from module import function1, function2, function3  # some comment'
    var_32 = module_1.line(var_31, var_3, var_30)
    var_33 = 'from module import function1, function2, function3'
    var_34 = '\r\n'
    var_35 = module_1.line(var_33, var_34, var_30)
    var_36 = 'from module import function1'
    var_37 = module_0.Config()
    var_38 = module_1.line(var_36, var_3, var_37)
    var_39 = module_0.Config()
    var_40 = 'from module import function1, function2, function3'
    var_41 = module_1.line(var_40, var_3, var_39)



# Parsed testcases at query #12
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import function1, function2, function3'
    var_1 = 30
    var_2 = module_0.Config()
    var_3 = '\n'
    var_4 = module_1.line(var_0, var_3, var_2)
    assert var_4 == 'from module import (\n    function1,\n    function2,\n    function3,\n)'
    var_5 = 'from module import function1, function2  # some comment'
    var_6 = True
    var_7 = module_0.Config()
    var_8 = module_1.line(var_5, var_3, var_7)
    assert var_8 == 'from module import (\n    function1,\n    function2,  # some comment\n)'
    var_9 = 'from module import function1, function2  # NOQA'
    var_10 = module_1.line(var_9, var_3, var_7)
    assert var_10 == 'from module import function1, function2  # NOQA'
    var_11 = 'from module import function1'
    var_12 = 50
    var_13 = module_0.Config()
    var_14 = module_1.line(var_11, var_3, var_13)
    assert var_14 == 'from module import function1'
    var_15 = 'from module import function1 as f1, function2 as f2'
    var_16 = module_0.Config()
    var_17 = module_1.line(var_15, var_3, var_16)
    assert var_17 == 'from module import (\n    function1 as f1,\n    function2 as f2,\n)'
    var_18 = 'cimport module.function1, module.function2'
    var_19 = module_0.Config()
    var_20 = module_1.line(var_18, var_3, var_19)
    assert var_20 == 'cimport (\n    module.function1,\n    module.function2,\n)'
    var_21 = 'from module import function1, function2  # noqa'
    var_22 = module_0.Config()
    var_23 = module_1.line(var_21, var_3, var_22)
    assert var_23 == 'from module import (\n    function1,\n    function2,  # noqa\n)'
    var_24 = 'from module import function1, function2, function3'
    var_25 = module_1.line(var_24, var_3, var_22)
    assert var_25 == 'from module import (\n    function1,\n    function2,\n    function3,\n)'
    var_26 = 'from module import function1, function2, function3'
    var_27 = module_1.line(var_26, var_3, var_22)
    assert var_27 == 'from module import (\n    function1,\n    function2,\n    function3,\n)'



# Parsed testcases at query #13
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
    var_17 = '\n'
    var_18 = 20
    var_19 = module_1.Config()
    var_20 = 'very_long_name1'
    var_21 = 'very_long_name2'
    var_22 = [var_20, var_21]
    var_23 = module_0.import_statement(var_0, var_22, config=var_19)
    var_24 = module_2.split(var_17)
    var_25 = len(var_24)
    var_26 = var_25 > var_15
    var_27 = 0
    var_28 = var_24[var_27]
    var_29 = len(var_28)
    var_30 = -1
    var_31 = var_24[var_30]
    var_32 = len(var_31)
    var_33 = var_29 >= var_32
    var_34 = module_1.Config()
    var_35 = [var_1, var_2]
    var_36 = module_0.import_statement(var_0, var_35, config=var_34)
    var_37 = ','
    var_38 = module_1.Config()
    var_39 = [var_1, var_2]
    var_40 = '# comment'
    var_41 = [var_40]
    var_42 = module_0.import_statement(var_0, var_39, var_41, config=var_38)
    var_43 = 'from module import'
    var_44 = 'func1'
    var_45 = 'func2'
    var_46 = 'func3'
    var_47 = [var_44, var_45, var_46]



# Parsed testcases at query #14
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1
import re as module_2

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = 'from module import function1, function2, function3'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    var_5 = 'from module import function1, function2, function3, function4, function5'
    var_6 = module_1.line(var_5, var_3, var_1)
    var_7 = 'from module import function1, function2  # This is a comment'
    var_8 = module_1.line(var_7, var_3, var_1)
    var_9 = 'from module import function1, function2, function3'
    var_10 = module_1.line(var_9, var_3, var_1)
    var_11 = True
    var_12 = module_0.Config()
    var_13 = 'from module import function1, function2, function3'
    var_14 = module_1.line(var_13, var_3, var_12)
    var_15 = module_0.Config()
    var_16 = 'from module import function1, function2, function3'
    var_17 = module_1.line(var_16, var_3, var_15)
    var_18 = -2
    var_19 = wrapped.split(var_3)[var_18]
    var_20 = 'from module import function1 as f1, function2 as f2'
    var_21 = module_1.line(var_20, var_3, var_15)
    var_22 = 'from module import function1, function2  # noqa'
    var_23 = module_1.line(var_22, var_3, var_15)
    var_24 = module_0.Config()
    var_25 = 'from module import function1, function2, function3'
    var_26 = module_1.line(var_25, var_3, var_24)
    var_27 = module_2.split(var_3)
    var_28 = -1
    var_29 = var_27[var_28]
    var_30 = len(var_29)
    var_31 = -1
    var_32 = var_27[:var_31]
    var_33 = min(var_11)



# Parsed testcases at query #15
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = 'from module import very_long_function_name'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    var_5 = 'from module import very_long_function_name  # some comment'
    var_6 = 'from module import (\n    very_long_function_name  # some comment\n)'
    var_7 = module_1.line(var_5, var_3, var_1)
    var_8 = 'from module import very_long_function_name  # NOQA'
    var_9 = module_1.line(var_8, var_3, var_1)
    var_10 = 'from module import very_long_function_name  # noqa'
    var_11 = 'from module import (\n    very_long_function_name  # noqa\n)'
    var_12 = module_1.line(var_10, var_3, var_1)
    var_13 = 'from module import very_long_function_name as alias'
    var_14 = 'from module import (\n    very_long_function_name as alias\n)'
    var_15 = module_1.line(var_13, var_3, var_1)
    var_16 = 'from module import very_long_function_name'
    var_17 = 'from module import \\\n    very_long_function_name'
    var_18 = module_1.line(var_16, var_3, var_1)
    var_19 = 'from module import very_long_function_name'
    var_20 = 'from module import (\n    very_long_function_name,\n)'
    var_21 = module_1.line(var_19, var_3, var_1)
    var_22 = 'from module import very_long_function_name'
    var_23 = 'from module import very_long_function_name  # NOQA'
    var_24 = module_1.line(var_22, var_3, var_1)
    var_25 = 'from module import very_long_function_name  # NOQA'
    var_26 = module_1.line(var_25, var_3, var_1)
    var_27 = 'from module import very_long_function_name'
    var_28 = 'from module import (\n    very_long_function_name,\n)'
    var_29 = module_1.line(var_27, var_3, var_1)
    var_30 = 'from module import very_long_function_name'
    var_31 = 'from module import (\n    very_long_function_name,\n)'
    var_32 = module_1.line(var_30, var_3, var_1)
    var_33 = 'from module import very_long_function_name  # some comment'
    var_34 = 'from module import (\n    very_long_function_name,  # some comment\n)'
    var_35 = module_1.line(var_33, var_3, var_1)
    var_36 = 'from module import very_long_function_name  # noqa'
    var_37 = 'from module import (\n    very_long_function_name,  # noqa\n)'
    var_38 = module_1.line(var_36, var_3, var_1)
    var_39 = 'from module import very_long_function_name  # NOQA'
    var_40 = 'from module import (\n    very_long_function_name,  # NOQA\n)'
    var_41 = module_1.line(var_39, var_3, var_1)
    var_42 = 'from module import very_long_function_name as alias'
    var_43 = 'from module import (\n    very_long_function_name as alias,\n)'
    var_44 = module_1.line(var_42, var_3, var_1)
    var_45 = 'from module import very_long_function_name'
    var_46 = 'from module import \\\n    very_long_function_name'
    var_47 = module_1.line(var_45, var_3, var_1)
    var_48 = 'from module import very_long_function_name'
    var_49 = 'from module import (\n    very_long_function_name,\n)'
    var_50 = module_1.line(var_48, var_3, var_1)
    var_51 = 'from module import very_long_function_name'
    var_52 = 'from module import (\n    very_long_function_name,\n)'
    var_53 = module_1.line(var_51, var_3, var_1)
    var_54 = 'from module import very_long_function_name  # some comment'
    var_55 = 'from module import (\n    very_long_function_name,  # some comment\n)'
    var_56 = module_1.line(var_54, var_3, var_1)
    var_57 = 'from module import very_long_function_name  # some comment'
    var_58 = 'from module import (\n    very_long_function_name,\n)'
    var_59 = module_1.line(var_57, var_3, var_1)
    var_60 = 'from module import very_long_function_name  # some comment'
    var_61 = 'from module import (\n    very_long_function_name,  # some comment\n)'
    var_62 = module_1.line(var_60, var_3, var_1)
    var_63 = 'from module import very_long_function_name'
    var_64 = 'from module import (\n    very_long_function_name,\n)'
    var_65 = module_1.line(var_63, var_3, var_1)
    var_66 = 'from module import very_long_function_name'
    var_67 = 'from module import (\r\n    very_long_function_name,\r\n)'
    var_68 = '\r\n'
    var_69 = module_1.line(var_66, var_68, var_1)
    var_70 = 'from module import very_long_function_name'
    var_71 = 'from module import (\n    very_long_function_name,\n)'
    var_72 = module_1.line(var_70, var_3, var_1)
    var_73 = 'from module import very_long_function_name'
    var_74 = 'from module import (\n    very_long_function_name\n)'
    var_75 = module_1.line(var_73, var_3, var_1)
    var_76 = 'from module import very_long_function_name'
    var_77 = 'from module import (\n    very_long_function_name,\n)'
    var_78 = module_1.line(var_76, var_3, var_1)
    var_79 = 'from module import very_long_function_name  # some comment'



# Parsed testcases at query #16
#--------------------------


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'thing1'
    var_2 = 'thing2'
    var_3 = 'thing3'
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
    var_21 = [var_1, var_2]



# Parsed testcases at query #17
#--------------------------


import isort.wrap as module_0
import re as module_1
import isort.settings as module_2

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'item1'
    var_2 = 'item2'
    var_3 = [var_1, var_2]
    var_4 = module_0.import_statement(var_0, var_3)
    assert var_4 == 'from module import item1, item2'
    var_5 = 'item3'
    var_6 = 'item4'
    var_7 = [var_1, var_2, var_5, var_6]
    var_8 = module_0.import_statement(var_0, var_7)
    var_9 = [var_1, var_2]
    var_10 = '# Comment 1'
    var_11 = '# Comment 2'
    var_12 = [var_10, var_11]
    var_13 = module_0.import_statement(var_0, var_9, var_12)
    var_14 = [var_1, var_2]
    var_15 = '\r\n'
    var_16 = module_0.import_statement(var_0, var_14, line_separator=var_15)
    var_17 = [var_1, var_2]
    var_18 = True
    var_19 = module_0.import_statement(var_0, var_17, explode=var_18)
    var_20 = '\n'
    var_21 = 20
    var_22 = [var_1, var_2, var_5]
    var_23 = module_1.split(var_20)
    var_24 = -1
    var_25 = var_23[var_24]
    var_26 = len(var_25)
    var_27 = -1
    var_28 = var_23[:var_27]
    var_29 = [var_1, var_2]
    var_30 = module_2.Config()
    var_31 = [var_1, var_2]
    var_32 = module_0.import_statement(var_0, var_31, config=var_30)
    var_33 = ','
    var_34 = module_2.Config()
    var_35 = [var_1, var_2]
    var_36 = '# Comment'
    var_37 = [var_36]
    var_38 = module_0.import_statement(var_0, var_35, var_37, config=var_34)



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
    var_5 = 'from module import function1, function2, function3, function4, function5'
    var_6 = module_1.line(var_5, var_3, var_1)
    var_7 = 'from module import function1, function2, function3  # some comment'
    var_8 = module_1.line(var_7, var_3, var_1)
    var_9 = 'from module import function1, function2, function3'
    var_10 = module_1.line(var_9, var_3, var_1)
    var_11 = True
    var_12 = module_0.Config()
    var_13 = 'from module import function1, function2, function3'
    var_14 = module_1.line(var_13, var_3, var_12)
    var_15 = module_0.Config()
    var_16 = 'from module import function1, function2, function3'
    var_17 = module_1.line(var_16, var_3, var_15)
    var_18 = 'from module import function1, function2, function3'
    var_19 = module_1.line(var_18, var_3, var_15)
    var_20 = 'from module import function1, function2, function3'
    var_21 = module_1.line(var_20, var_3, var_15)
    var_22 = 'import module as alias'
    var_23 = module_1.line(var_22, var_3, var_15)
    var_24 = 'cimport module'
    var_25 = module_1.line(var_24, var_3, var_15)
    var_26 = 'from module.submodule import function'
    var_27 = module_1.line(var_26, var_3, var_15)



# Parsed testcases at query #19
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
    var_19 = 40
    var_20 = module_1.Config()
    var_21 = [var_1, var_2, var_3]
    var_22 = module_0.import_statement(var_0, var_21, config=var_20)
    var_23 = [var_1, var_2, var_3]
    var_24 = 80
    var_25 = module_1.Config()
    var_26 = 'very_long_function_name_1'
    var_27 = 'very_long_function_name_2'
    var_28 = [var_26, var_27]
    var_29 = module_0.import_statement(var_0, var_28, config=var_25)
    var_30 = module_2.split(var_17)
    var_31 = -1
    var_32 = var_30[var_31]
    var_33 = len(var_32)
    var_34 = -1
    var_35 = var_30[:var_34]
    var_36 = min(var_6)
    var_37 = [var_32]
    var_38 = module_0.import_statement(var_31, var_37)



# Parsed testcases at query #20
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
    var_14 = [var_1, var_2, var_3]
    var_15 = True
    var_16 = module_0.import_statement(var_0, var_14, explode=var_15)
    var_17 = '\n'
    var_18 = [var_1, var_2, var_3]
    var_19 = len(var_18)
    var_20 = module_1.Config()
    var_21 = [var_1, var_2, var_3]
    var_22 = module_0.import_statement(var_0, var_21, config=var_20)
    var_23 = module_2.split(var_17)
    var_24 = -1
    var_25 = var_23[var_24]
    var_26 = len(var_25)
    var_27 = -1
    var_28 = var_23[:var_27]
    var_29 = min(var_6)
    var_30 = [var_25, var_26, var_27]
    var_31 = module_1.Config()
    var_32 = [var_25, var_26]
    var_33 = module_0.import_statement(var_24, var_32, config=var_31)
    var_34 = ','
    var_35 = module_1.Config()
    var_36 = [var_25, var_26]
    var_37 = [var_29]
    var_38 = module_0.import_statement(var_24, var_36, var_37, config=var_35)



# Parsed testcases at query #21
#--------------------------


import isort.wrap as module_0

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
    var_17 = 50
    var_18 = [var_1, var_2, var_3]
    var_19 = [var_1, var_2, var_3]
    var_20 = [var_1, var_2, var_3]



# Parsed testcases at query #22
#--------------------------


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import function'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = 'from module import function1, function2, function3'
    var_4 = 30
    var_5 = 'from module import function  # comment'
    var_6 = module_0.line(var_5, var_1)
    var_7 = 'from module import function  # NOQA'
    var_8 = module_0.line(var_7, var_1)
    var_9 = 'from module import function1, function2, function3'
    var_10 = True
    var_11 = 'from module import function1, function2, function3'
    var_12 = 'from module import function as alias'
    var_13 = module_0.line(var_12, var_1)
    var_14 = 'from module import function1 as alias1, function2 as alias2'
    var_15 = 'from module import function'
    var_16 = '\r\n'
    var_17 = module_0.line(var_15, var_16)
    var_18 = 'from module import function1, function2, function3'
    var_19 = 'from module import function  # comment'



# Parsed testcases at query #23
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1
import re as module_2

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'item1'
    var_2 = 'item2'
    var_3 = [var_1, var_2]
    var_4 = module_0.import_statement(var_0, var_3)
    assert var_4 == 'from module import item1, item2'
    var_5 = [var_1, var_2]
    var_6 = '\r\n'
    var_7 = module_0.import_statement(var_0, var_5, line_separator=var_6)
    assert var_7 == 'from module import item1, item2'
    var_8 = [var_1, var_2]
    var_9 = '# comment'
    var_10 = [var_9]
    var_11 = module_0.import_statement(var_0, var_8, var_10)
    var_12 = [var_1, var_2]
    var_13 = True
    var_14 = module_0.import_statement(var_0, var_12, explode=var_13)
    var_15 = 20
    var_16 = 'item3'
    var_17 = [var_1, var_2, var_16]
    var_18 = 30
    var_19 = module_1.Config()
    var_20 = [var_1, var_2, var_16]
    var_21 = module_0.import_statement(var_0, var_20, config=var_19)
    var_22 = '\n'
    var_23 = module_2.split(var_22)
    var_24 = len(var_23)
    var_25 = -1
    var_26 = var_23[var_25]
    var_27 = len(var_26)
    var_28 = -1
    var_29 = var_23[:var_28]
    var_30 = [var_1, var_2, var_16]
    var_31 = 'item4'
    var_32 = 'item5'
    var_33 = [var_1, var_2, var_16, var_31, var_32]
    var_34 = module_1.Config()
    var_35 = module_0.import_statement(var_0, var_33, config=var_34)
    var_36 = [var_1]
    var_37 = module_0.import_statement(var_0, var_36)
    assert var_37 == 'from module import item1'
    var_38 = []
    var_39 = module_0.import_statement(var_0, var_38)
    assert var_39 == 'from module import'
    var_40 = 'item_1'
    var_41 = 'item-2'
    var_42 = [var_40, var_41, var_16]
    var_43 = module_0.import_statement(var_0, var_42)



# Parsed testcases at query #24
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1
import re as module_2

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'function1'
    var_2 = 'function2'
    var_3 = [var_1, var_2]
    var_4 = module_0.import_statement(var_0, var_3)
    var_5 = 'function3'
    var_6 = [var_1, var_2, var_5]
    var_7 = 20
    var_8 = module_1.Config()
    var_9 = module_0.import_statement(var_0, var_6, config=var_8)
    var_10 = '\n'
    var_11 = [var_1, var_2]
    var_12 = '# Comment 1'
    var_13 = '# Comment 2'
    var_14 = [var_12, var_13]
    var_15 = module_0.import_statement(var_0, var_11, var_14)
    var_16 = [var_1, var_2]
    var_17 = True
    var_18 = module_0.import_statement(var_0, var_16, explode=var_17)
    var_19 = 30
    var_20 = module_1.Config()
    var_21 = [var_1, var_2, var_5]
    var_22 = module_0.import_statement(var_0, var_21, config=var_20)
    var_23 = module_2.split(var_10)
    var_24 = len(var_23)
    var_25 = -1
    var_26 = var_23[var_25]
    var_27 = len(var_26)
    var_28 = -1
    var_29 = var_23[:var_28]
    var_30 = module_1.Config()
    var_31 = [var_1, var_2]
    var_32 = module_0.import_statement(var_0, var_31, config=var_30)
    var_33 = ','
    var_34 = [var_1, var_2]
    var_35 = '\r\n'
    var_36 = module_0.import_statement(var_0, var_34, line_separator=var_35)
    var_37 = []
    var_38 = module_0.import_statement(var_0, var_37)
    assert var_38 == 'from module import'
    var_39 = [var_1, var_2, var_5]



# Parsed testcases at query #25
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    assert var_5 == 'from module import a, b, c'
    var_6 = [var_1, var_2, var_3]
    var_7 = '# comment'
    var_8 = [var_7]
    var_9 = module_0.import_statement(var_0, var_6, var_8)
    var_10 = [var_1, var_2, var_3]
    var_11 = '\r\n'
    var_12 = module_0.import_statement(var_0, var_10, line_separator=var_11)
    var_13 = [var_1, var_2, var_3]
    var_14 = True
    var_15 = module_0.import_statement(var_0, var_13, explode=var_14)
    var_16 = '\n'
    var_17 = 20
    var_18 = 'very_long_name_a'
    var_19 = 'very_long_name_b'
    var_20 = [var_18, var_19]
    var_21 = 30
    var_22 = module_1.Config()
    var_23 = 'bb'
    var_24 = 'ccc'
    var_25 = [var_1, var_23, var_24]
    var_26 = module_0.import_statement(var_0, var_25, config=var_22)
    var_27 = -1
    var_28 = result.split(var_16)[var_27]
    var_29 = len(var_28)
    var_30 = 0
    var_31 = result.split(var_16)[var_30]
    var_32 = len(var_31)
    var_33 = [var_1, var_2, var_3]
    var_34 = []
    var_35 = module_0.import_statement(var_0, var_34)
    assert var_35 == 'from module import'
    var_36 = [var_1]
    var_37 = module_0.import_statement(var_0, var_36)
    assert var_37 == 'from module import a'



# Parsed testcases at query #26
#--------------------------


import re as module_0

def test_case_0():
    var_0 = 80
    var_1 = 'from module import very_long_function_name, another_very_long_function_name'
    var_2 = '\n'
    var_3 = 0
    var_4 = result.split(var_2)[var_3]
    var_5 = len(var_4)
    var_6 = 'from module import func  # This is a comment'
    var_7 = 'from module import very_long_function_name  # NOQA'
    var_8 = 'from module import func  # noqa: F401'
    var_9 = 'from module import very_long_function_name as vlf'
    var_10 = 'from module.submodule import very_long_function_name'
    var_11 = True
    var_12 = 'from module import func1, func2, func3'
    var_13 = 'from module import func1, func2, func3'
    var_14 = ','
    var_15 = 'from module import func1, func2, func3'
    var_16 = module_0.split(var_2)
    var_17 = -1
    var_18 = var_16[var_17]
    var_19 = len(var_18)
    var_20 = -1
    var_21 = var_16[:var_20]
    var_22 = min(var_4)
    var_23 = 'from module import func  # This is a comment'
    var_24 = ' '
    var_25 = 'from module import func1, func2'
    var_26 = 'cimport from module import very_long_function_name'
    var_27 = 'from module import very_long_function_name'
    var_28 = 'from module import func1, func2, func3'
    var_29 = ' # '
    var_30 = 'from module import func  # comment'



# Parsed testcases at query #27
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1
import re as module_2

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = 'This is a very long line that should be wrapped if it exceeds the line length limit.'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    var_5 = 0
    var_6 = result.split(var_3)[var_5]
    var_7 = len(var_6)
    var_8 = 'import os# This is a comment'
    var_9 = module_1.line(var_8, var_3, var_1)
    var_10 = 'import os# NOQA'
    var_11 = module_1.line(var_10, var_3, var_1)
    var_12 = 'short'
    var_13 = module_1.line(var_12, var_3, var_1)
    var_14 = 'from module import function, another_function, third_function'
    var_15 = module_1.line(var_14, var_3, var_1)
    var_16 = 'cimport module.function, module.another_function'
    var_17 = module_1.line(var_16, var_3, var_1)
    var_18 = 'import module as alias'
    var_19 = module_1.line(var_18, var_3, var_1)
    var_20 = module_1.line(var_14, var_3, var_1)
    var_21 = ','
    var_22 = ')'
    var_23 = module_1.line(var_14, var_3, var_1)
    var_24 = module_2.split(var_3)
    var_25 = -1
    var_26 = var_24[var_25]
    var_27 = len(var_26)
    var_28 = -1
    var_29 = var_24[:var_28]
    var_30 = min(var_21)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------




# Parsed testcases at query #2
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
    var_15 = 20
    var_16 = module_1.Config()
    var_17 = [var_1, var_2]
    var_18 = module_0.import_statement(var_0, var_17, config=var_16)
    var_19 = 0
    var_20 = '\n'
    var_21 = result.split(var_20)[var_19]
    var_22 = len(var_21)
    var_23 = [var_1, var_2]
    var_24 = module_1.Config()
    var_25 = [var_1, var_2]
    var_26 = module_0.import_statement(var_0, var_25, config=var_24)
    var_27 = ','
    var_28 = module_1.Config()
    var_29 = [var_1, var_2]
    var_30 = [var_6]
    var_31 = module_0.import_statement(var_0, var_29, var_30, config=var_28)
    var_32 = '    '
    var_33 = module_1.Config()
    var_34 = [var_1, var_2]
    var_35 = module_0.import_statement(var_0, var_34, config=var_33)
    var_36 = 10
    var_37 = module_1.Config()
    var_38 = [var_1, var_2]
    var_39 = module_0.import_statement(var_0, var_38, config=var_37)
    var_40 = result.split(var_20)[var_19]
    var_41 = len(var_40)
    var_42 = '# '
    var_43 = module_1.Config()
    var_44 = [var_1, var_2]
    var_45 = 'Comment'
    var_46 = [var_45]
    var_47 = module_0.import_statement(var_0, var_44, var_46, config=var_43)
    var_48 = module_1.Config()
    var_49 = [var_1, var_2]
    var_50 = module_0.import_statement(var_0, var_49, config=var_48)



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
    var_19 = [var_1, var_2, var_3]
    var_20 = ','
    var_21 = 30
    var_22 = module_1.Config()
    var_23 = 'very_long_item_name_1'
    var_24 = 'very_long_item_name_2'
    var_25 = [var_23, var_24]
    var_26 = module_0.import_statement(var_0, var_25, config=var_22)
    var_27 = module_2.split(var_17)
    var_28 = len(var_27)
    var_29 = -1
    var_30 = var_27[var_29]
    var_31 = len(var_30)
    var_32 = -1
    var_33 = var_27[:var_32]
    var_34 = [var_1, var_2, var_3]



# Parsed testcases at query #4
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1
import re as module_2

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = 'from module import function1, function2, function3'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    var_5 = 0
    var_6 = result.split(var_3)[var_5]
    var_7 = len(var_6)
    var_8 = 'from module import func1, func2  # some comment'
    var_9 = module_1.line(var_8, var_3, var_1)
    var_10 = 'from module import func1, func2, func3, func4, func5'
    var_11 = True
    var_12 = module_0.Config()
    var_13 = 'from module import func1, func2, func3, func4'
    var_14 = module_1.line(var_13, var_3, var_12)
    var_15 = module_0.Config()
    var_16 = 'from module import func1, func2, func3'
    var_17 = module_1.line(var_16, var_3, var_15)
    var_18 = ','
    var_19 = 'from module import func1, func2, func3, func4'
    var_20 = module_0.Config()
    var_21 = 'from module import func1, func2, func3, func4, func5'
    var_22 = module_1.line(var_21, var_3, var_20)
    var_23 = module_2.split(var_3)
    var_24 = -1
    var_25 = var_23[var_24]
    var_26 = len(var_25)
    var_27 = -1
    var_28 = var_23[:var_27]
    var_29 = 'from module import func'
    var_30 = module_1.line(var_29, var_25, var_1)
    var_31 = 'from module import func1 as f1, func2 as f2'
    var_32 = module_1.line(var_31, var_25, var_1)
    var_33 = 'cimport module.func1, module.func2, module.func3'
    var_34 = module_1.line(var_33, var_25, var_1)



# Parsed testcases at query #5
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import function'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = 'from module import very_long_function_name_that_exceeds_line_length'
    var_4 = 20
    var_5 = module_1.Config()
    var_6 = module_0.line(var_3, var_1, var_5)
    var_7 = 0
    var_8 = result.split(var_1)[var_7]
    var_9 = len(var_8)
    var_10 = 'from module import function  # some comment'
    var_11 = True
    var_12 = module_1.Config()
    var_13 = module_0.line(var_10, var_1, var_12)
    var_14 = 'from module import function  # NOQA'
    var_15 = module_0.line(var_14, var_1, var_12)
    var_16 = 'from module import function1, function2'
    var_17 = module_1.Config()
    var_18 = module_0.line(var_16, var_1, var_17)
    var_19 = 'from module import function1, function2, function3'
    var_20 = module_0.line(var_19, var_1, var_17)
    var_21 = 'import module as alias'
    var_22 = module_1.Config()
    var_23 = module_0.line(var_21, var_1, var_22)
    var_24 = 'import os'
    var_25 = module_1.Config()
    var_26 = module_0.line(var_24, var_1, var_25)



# Parsed testcases at query #6
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1
import re as module_2

def test_case_0():
    var_0 = 'from module import function'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = 'from module import function1, function2, function3, function4'
    var_4 = 30
    var_5 = module_1.Config()
    var_6 = module_0.line(var_3, var_1, var_5)
    var_7 = 'from module import function  # comment'
    var_8 = module_0.line(var_7, var_1, var_5)
    var_9 = 'from module import function  # NOQA'
    var_10 = module_0.line(var_9, var_1, var_5)
    var_11 = 'from module import function as f'
    var_12 = module_0.line(var_11, var_1, var_5)
    var_13 = 'from module import function.subfunction'
    var_14 = module_0.line(var_13, var_1, var_5)
    var_15 = 'cimport module.function'
    var_16 = module_0.line(var_15, var_1, var_5)
    var_17 = module_0.line(var_3, var_1, var_5)
    var_18 = module_0.line(var_3, var_1, var_5)
    var_19 = module_0.line(var_7, var_1, var_5)
    var_20 = module_0.line(var_7, var_1, var_5)
    var_21 = module_0.line(var_3, var_1, var_5)
    var_22 = module_2.split(var_1)
    var_23 = -1
    var_24 = var_22[var_23]
    var_25 = len(var_24)
    var_26 = -1
    var_27 = var_22[:var_26]
    var_28 = module_0.line(var_3, var_23, var_5)
    var_29 = module_0.line(var_3, var_23, var_5)
    var_30 = '\r\n'
    var_31 = module_0.line(var_3, var_30, var_5)
    var_32 = module_0.line(var_3, var_23, var_5)
    var_33 = '    '



# Parsed testcases at query #7
#--------------------------


import isort.wrap as module_0
import re as module_1

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
    var_19 = [var_1, var_2, var_3]
    var_20 = 0
    var_21 = result.split(var_17)[var_20]
    var_22 = len(var_21)
    var_23 = 30
    var_24 = [var_1, var_2, var_3]
    var_25 = module_1.split(var_17)
    var_26 = -1
    var_27 = var_25[:var_26]
    var_28 = -1
    var_29 = var_25[var_28]
    var_30 = len(var_29)
    var_31 = [var_1, var_2, var_3]



# Parsed testcases at query #8
#--------------------------




# Parsed testcases at query #9
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
    var_19 = module_1.Config()
    var_20 = 'very_long_function_name_1'
    var_21 = 'very_long_function_name_2'
    var_22 = [var_20, var_21]
    var_23 = module_0.import_statement(var_0, var_22, config=var_19)
    var_24 = module_2.split(var_17)
    var_25 = -1
    var_26 = var_24[var_25]
    var_27 = len(var_26)
    var_28 = -1
    var_29 = var_24[:var_28]
    var_30 = len(var_24)
    var_31 = var_30 == var_15
    var_32 = [var_1, var_2, var_3]
    var_33 = module_1.Config()
    var_34 = [var_1, var_2]
    var_35 = module_0.import_statement(var_0, var_34, config=var_33)
    var_36 = ','
    var_37 = module_1.Config()
    var_38 = [var_1, var_2]
    var_39 = [var_7]
    var_40 = module_0.import_statement(var_0, var_38, var_39, config=var_37)
    var_41 = '    '
    var_42 = module_1.Config()
    var_43 = [var_1, var_2]
    var_44 = module_0.import_statement(var_0, var_43, config=var_42)
    var_45 = 40
    var_46 = module_1.Config()
    var_47 = [var_20, var_21]
    var_48 = module_0.import_statement(var_0, var_47, config=var_46)
    var_49 = 0
    var_50 = result.split(var_17)[var_49]
    var_51 = len(var_50)



# Parsed testcases at query #10
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1
import re as module_2

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'foo'
    var_2 = 'bar'
    var_3 = 'baz'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    var_6 = [var_1, var_2]
    var_7 = '\r\n'
    var_8 = module_0.import_statement(var_0, var_6, line_separator=var_7)
    var_9 = [var_1, var_2]
    var_10 = True
    var_11 = module_0.import_statement(var_0, var_9, explode=var_10)
    var_12 = '\n'
    var_13 = [var_1, var_2]
    var_14 = '# comment'
    var_15 = [var_14]
    var_16 = module_0.import_statement(var_0, var_13, var_15)
    var_17 = 50
    var_18 = [var_1, var_2]
    var_19 = 80
    var_20 = module_1.Config()
    var_21 = [var_1, var_2]
    var_22 = module_0.import_statement(var_0, var_21, config=var_20)
    var_23 = module_2.split(var_12)
    var_24 = -1
    var_25 = var_23[var_24]
    var_26 = len(var_25)
    var_27 = -1
    var_28 = var_23[:var_27]
    var_29 = min(var_6)
    var_30 = [var_25, var_26]
    var_31 = [var_25]
    var_32 = module_0.import_statement(var_24, var_31)



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 80
    var_1 = 'from module import very_long_function_name, another_very_long_function_name'
    var_2 = '\n'
    var_3 = 'from module import func # some comment'
    var_4 = 'from module import very_long_function_name, another_very_long_function_name'
    var_5 = True
    var_6 = 'from module import func1, func2, func3'
    var_7 = 'from module import func1, func2, func3'
    var_8 = ','
    var_9 = 'from module import func1, func2, func3'
    var_10 = '\r\n'
    var_11 = 'import os'
    var_12 = 'from module import func as f'
    var_13 = 'cimport module.func'



# Parsed testcases at query #12
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import function'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = 'from module import function1, function2, function3'
    var_4 = 30
    var_5 = module_1.Config()
    var_6 = module_0.line(var_3, var_1, var_5)
    assert var_6 == 'from module import (\n    function1,\n    function2,\n    function3,\n)'
    var_7 = 'from module import function  # some comment'
    var_8 = module_0.line(var_7, var_1, var_5)
    assert var_8 == 'from module import (\n    function1,\n    function2,\n    function3,  # some comment\n)'
    var_9 = 'from module import function1, function2, function3  # NOQA'
    var_10 = module_0.line(var_9, var_1, var_5)
    var_11 = 'from module import function1, function2, function3  # noqa'
    var_12 = module_0.line(var_11, var_1, var_5)
    assert var_12 == 'from module import (\n    function1,\n    function2,\n    function3,  # noqa\n)'
    var_13 = False
    var_14 = module_1.Config()
    var_15 = module_0.line(var_3, var_1, var_14)
    assert var_15 == 'from module import function1,\n    function2,\n    function3'
    var_16 = module_1.Config()
    var_17 = module_0.line(var_3, var_1, var_16)
    assert var_17 == 'from module import (\n    function1\n    function2\n    function3\n)'
    var_18 = True
    var_19 = module_1.Config()
    var_20 = module_0.line(var_3, var_1, var_19)
    assert var_20 == 'from module import (\n    function1,\n    function2,\n    function3,\n)'
    var_21 = module_1.Config()
    var_22 = module_0.line(var_7, var_1, var_21)
    assert var_22 == 'from module import (\n    function1,\n    function2,\n    function3,\n)'
    var_23 = '\r\n'
    var_24 = module_0.line(var_3, var_23, var_5)
    assert var_24 == 'from module import (\r\n    function1,\r\n    function2,\r\n    function3,\r\n)'
    var_25 = 'from module import f'
    var_26 = module_0.line(var_25, var_1, var_5)
    var_27 = ''
    var_28 = module_0.line(var_27, var_1, var_5)
    assert var_28 == ''
    var_29 = '# some comment'
    var_30 = module_0.line(var_29, var_1, var_5)
    var_31 = '   '
    var_32 = module_0.line(var_31, var_1, var_5)



# Parsed testcases at query #13
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1
import re as module_2

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    assert var_5 == 'from module import a, b, c'
    var_6 = [var_1, var_2, var_3]
    var_7 = True
    var_8 = module_0.import_statement(var_0, var_6, explode=var_7)
    assert var_8 == 'from module import (\n    a,\n    b,\n    c,\n)'
    var_9 = [var_1, var_2, var_3]
    var_10 = '# comment'
    var_11 = [var_10]
    var_12 = module_0.import_statement(var_0, var_9, var_11)
    var_13 = [var_1, var_2, var_3]
    var_14 = '\r\n'
    var_15 = module_0.import_statement(var_0, var_13, line_separator=var_14)
    var_16 = 20
    var_17 = module_1.Config()
    var_18 = [var_1, var_2, var_3]
    var_19 = module_0.import_statement(var_0, var_18, config=var_17)
    var_20 = 0
    var_21 = '\n'
    var_22 = result.split(var_21)[var_20]
    var_23 = len(var_22)
    var_24 = [var_1, var_2, var_3]
    var_25 = 30
    var_26 = module_1.Config()
    var_27 = [var_1, var_2, var_3]
    var_28 = module_0.import_statement(var_0, var_27, config=var_26)
    var_29 = module_2.split(var_21)
    var_30 = -1
    var_31 = var_29[var_30]
    var_32 = len(var_31)
    var_33 = -1
    var_34 = var_29[:var_33]
    var_35 = min(var_6)
    var_36 = []
    var_37 = module_0.import_statement(var_30, var_36)
    assert var_37 == 'from module import'
    var_38 = [var_31]
    var_39 = module_0.import_statement(var_30, var_38)
    assert var_39 == 'from module import a'
    var_40 = module_1.Config()
    var_41 = [var_31, var_32]
    var_42 = module_0.import_statement(var_30, var_41, config=var_40)
    var_43 = ','
    var_44 = module_1.Config()
    var_45 = [var_31, var_32]
    var_46 = [var_10]
    var_47 = module_0.import_statement(var_30, var_45, var_46, config=var_44)



# Parsed testcases at query #14
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'func1'
    var_2 = 'func2'
    var_3 = [var_1, var_2]
    var_4 = module_0.import_statement(var_0, var_3)
    assert var_4 == 'from module import func1, func2'
    var_5 = 'func3'
    var_6 = [var_1, var_2, var_5]
    var_7 = module_0.import_statement(var_0, var_6)
    var_8 = [var_1, var_2]
    var_9 = '\r\n'
    var_10 = module_0.import_statement(var_0, var_8, line_separator=var_9)
    var_11 = [var_1, var_2]
    var_12 = '# Comment 1'
    var_13 = '# Comment 2'
    var_14 = [var_12, var_13]
    var_15 = module_0.import_statement(var_0, var_11, var_14)
    var_16 = [var_1, var_2]
    var_17 = True
    var_18 = module_0.import_statement(var_0, var_16, explode=var_17)
    var_19 = '\n'
    var_20 = 50
    var_21 = [var_1, var_2, var_5]
    var_22 = 30
    var_23 = module_1.Config()
    var_24 = [var_1, var_2, var_5]
    var_25 = module_0.import_statement(var_0, var_24, config=var_23)



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
    var_5 = 'very_long_function_name_1'
    var_6 = 'very_long_function_name_2'
    var_7 = 'very_long_function_name_3'
    var_8 = [var_5, var_6, var_7]
    var_9 = module_0.import_statement(var_0, var_8)
    var_10 = '\n'
    var_11 = '# Comment 1'
    var_12 = '# Comment 2'
    var_13 = [var_11, var_12]
    var_14 = [var_1, var_2]
    var_15 = module_0.import_statement(var_0, var_14, var_13)
    var_16 = [var_1, var_2]
    var_17 = True
    var_18 = module_0.import_statement(var_0, var_16, explode=var_17)
    var_19 = [var_1, var_2]
    var_20 = '\r\n'
    var_21 = module_0.import_statement(var_0, var_19, line_separator=var_20)
    var_22 = 20
    var_23 = module_1.Config()
    var_24 = [var_1, var_2]
    var_25 = module_0.import_statement(var_0, var_24, config=var_23)
    var_26 = 0
    var_27 = result.split(var_10)[var_26]
    var_28 = len(var_27)
    var_29 = module_1.Config()
    var_30 = [var_1, var_2]
    var_31 = module_0.import_statement(var_0, var_30, config=var_29)
    var_32 = ','
    var_33 = 'from module import'
    var_34 = 'func1'
    var_35 = 'func2'
    var_36 = [var_34, var_35]
    var_37 = []
    var_38 = module_0.import_statement(var_33, var_37)
    var_39 = 1000
    var_40 = module_1.Config()
    var_41 = [var_34, var_35]
    var_42 = module_0.import_statement(var_33, var_41, config=var_40)
    var_43 = module_1.Config()
    var_44 = [var_34, var_35]
    var_45 = '# Comment'
    var_46 = [var_45]
    var_47 = module_0.import_statement(var_33, var_44, var_46, config=var_43)



# Parsed testcases at query #16
#--------------------------


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'func1'
    var_2 = 'func2'
    var_3 = [var_1, var_2]
    var_4 = module_0.import_statement(var_0, var_3)
    var_5 = [var_1, var_2]
    var_6 = '# Comment 1'
    var_7 = '# Comment 2'
    var_8 = [var_6, var_7]
    var_9 = module_0.import_statement(var_0, var_5, var_8)
    var_10 = [var_1, var_2]
    var_11 = '\r\n'
    var_12 = module_0.import_statement(var_0, var_10, line_separator=var_11)
    var_13 = [var_1, var_2]
    var_14 = True
    var_15 = module_0.import_statement(var_0, var_13, explode=var_14)
    var_16 = 50
    var_17 = [var_1, var_2]
    var_18 = [var_1, var_2]
    var_19 = [var_1, var_2]



# Parsed testcases at query #17
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import function'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = 'from module import function1, function2, function3'
    var_4 = 30
    var_5 = module_1.Config()
    var_6 = module_0.line(var_3, var_1, var_5)
    var_7 = 'from module import function  # comment'
    var_8 = 20
    var_9 = module_1.Config()
    var_10 = module_0.line(var_7, var_1, var_9)
    var_11 = 'from module import function  # NOQA'
    var_12 = module_0.line(var_11, var_1, var_9)
    var_13 = 'from module import function1, function2, function3'
    var_14 = True
    var_15 = module_1.Config()
    var_16 = module_0.line(var_13, var_1, var_15)
    var_17 = 'from module import function1, function2, function3'
    var_18 = module_1.Config()
    var_19 = module_0.line(var_17, var_1, var_18)
    var_20 = -2
    var_21 = result.split(var_1)[var_20]
    var_22 = 'from module import function1, function2, function3'
    var_23 = module_0.line(var_22, var_1, var_18)
    var_24 = 'from module import function1, function2, function3'
    var_25 = module_0.line(var_24, var_1, var_18)
    var_26 = 'from module import function1, function2, function3'
    var_27 = module_1.Config()
    var_28 = module_0.line(var_26, var_1, var_27)
    var_29 = 'from module import function  # comment'
    var_30 = module_1.Config()
    var_31 = module_0.line(var_29, var_1, var_30)
    var_32 = 'from module import function  # comment'
    var_33 = '# '
    var_34 = module_1.Config()
    var_35 = module_0.line(var_32, var_1, var_34)



# Parsed testcases at query #18
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import something'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = 'from module import something_very_long_function_name'
    var_4 = 30
    var_5 = module_1.Config()
    var_6 = 'from module import (\n    something_very_long_function_name\n)'
    var_7 = module_0.line(var_3, var_1, var_5)
    var_8 = 'from module import something  # some comment'
    var_9 = True
    var_10 = module_1.Config()
    var_11 = 'from module import (\n    something,  # some comment\n)'
    var_12 = module_0.line(var_8, var_1, var_10)
    var_13 = 'from module import something  # NOQA'
    var_14 = module_1.Config()
    var_15 = module_0.line(var_13, var_1, var_14)
    var_16 = 'from module import something as alias'
    var_17 = module_1.Config()
    var_18 = 'from module import (\n    something as alias\n)'
    var_19 = module_0.line(var_16, var_1, var_17)
    var_20 = 'cimport module.something'
    var_21 = 20
    var_22 = module_1.Config()
    var_23 = 'cimport (\n    module.something\n)'
    var_24 = module_0.line(var_20, var_1, var_22)
    var_25 = 'from module import something.else'
    var_26 = module_1.Config()
    var_27 = 'from module import (\n    something.else\n)'
    var_28 = module_0.line(var_25, var_1, var_26)
    var_29 = 'from module import something,'
    var_30 = module_1.Config()
    var_31 = 'from module import (\n    something,\n)'
    var_32 = module_0.line(var_29, var_1, var_30)
    var_33 = 'from module import something'
    var_34 = '\r\n'
    var_35 = module_0.line(var_33, var_34)
    var_36 = 'from module import something  # comment'
    var_37 = module_1.Config()
    var_38 = 'from module import (\n    something\n)'
    var_39 = module_0.line(var_36, var_1, var_37)



# Parsed testcases at query #19
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import something'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = 'from module import something_very_long'
    var_4 = 20
    var_5 = module_1.Config()
    var_6 = module_0.line(var_3, var_1, var_5)
    var_7 = 'from module import something  # comment'
    var_8 = module_1.Config()
    var_9 = module_0.line(var_7, var_1, var_8)
    var_10 = 'from module import something_very_long'
    var_11 = module_0.line(var_10, var_1, var_8)
    var_12 = '# NOQA'
    var_13 = 'from module import something_very_long'
    var_14 = True
    var_15 = module_1.Config()
    var_16 = module_0.line(var_13, var_1, var_15)
    var_17 = 'from module import something_very_long'
    var_18 = module_1.Config()
    var_19 = module_0.line(var_17, var_1, var_18)
    var_20 = 'from module import something_very_long'
    var_21 = module_0.line(var_20, var_1, var_18)
    var_22 = 'from module import something_very_long'
    var_23 = module_0.line(var_22, var_1, var_18)
    var_24 = 'from module import something as alias'
    var_25 = module_1.Config()
    var_26 = module_0.line(var_24, var_1, var_25)
    var_27 = 'cimport module.something_very_long'
    var_28 = module_1.Config()
    var_29 = module_0.line(var_27, var_1, var_28)
    var_30 = 'from module import something.very.long'
    var_31 = module_1.Config()
    var_32 = module_0.line(var_30, var_1, var_31)
    var_33 = 'from module import something_very_long  # noqa'
    var_34 = module_1.Config()
    var_35 = module_0.line(var_33, var_1, var_34)



# Parsed testcases at query #20
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
    var_17 = module_1.Config()
    var_18 = [var_1, var_2, var_3]
    var_19 = module_0.import_statement(var_0, var_18, config=var_17)
    var_20 = [var_1, var_2, var_3]



# Parsed testcases at query #21
#--------------------------


import isort.wrap as module_0
import re as module_1
import isort.settings as module_2

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
    var_19 = 'very_long_item_name_1'
    var_20 = 'very_long_item_name_2'
    var_21 = [var_19, var_20]
    var_22 = module_1.split(var_17)
    var_23 = -1
    var_24 = var_22[var_23]
    var_25 = len(var_24)
    var_26 = -1
    var_27 = var_22[:var_26]
    var_28 = len(var_22)
    var_29 = var_28 == var_15
    var_30 = module_2.Config()
    var_31 = [var_1, var_2]
    var_32 = module_0.import_statement(var_0, var_31, config=var_30)
    var_33 = ','
    var_34 = [var_1, var_2, var_3]



# Parsed testcases at query #22
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
    assert var_5 == 'from module import func1, func2, func3'
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
    var_17 = module_1.Config()
    var_18 = [var_1, var_2, var_3]
    var_19 = module_0.import_statement(var_0, var_18, config=var_17)
    assert var_19 == 'from module import func1, func2, func3'
    var_20 = [var_1, var_2, var_3]
    var_21 = module_1.Config()
    var_22 = [var_1, var_2, var_3]
    var_23 = module_0.import_statement(var_0, var_22, config=var_21)
    var_24 = ','
    var_25 = module_1.Config()
    var_26 = [var_1, var_2, var_3]
    var_27 = [var_7, var_8]
    var_28 = module_0.import_statement(var_0, var_26, var_27, config=var_25)
    var_29 = '# '
    var_30 = module_1.Config()
    var_31 = [var_1, var_2, var_3]
    var_32 = [var_7, var_8]
    var_33 = module_0.import_statement(var_0, var_31, var_32, config=var_30)
    var_34 = 20
    var_35 = module_1.Config()
    var_36 = [var_1, var_2, var_3]
    var_37 = module_0.import_statement(var_0, var_36, config=var_35)
    var_38 = 0
    var_39 = '\n'
    var_40 = result.split(var_39)[var_38]
    var_41 = len(var_40)



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'from module import function1, function2, function3'
    var_1 = 50
    var_2 = '\n'
    var_3 = 'from module import function1, function2  # some comment'
    var_4 = 'from module import function1, function2, function3'
    var_5 = 'from module import function1 as f1, function2 as f2'
    var_6 = 40
    var_7 = 'from module.submodule import function1, function2'
    var_8 = 'cimport module.function1, module.function2'
    var_9 = True
    var_10 = 'from module import function1, function2, function3'
    var_11 = 'from module import function1, function2, function3'
    var_12 = 'from module import function1, function2  # noqa'
    var_13 = 'from module import function1, function2, function3'
    var_14 = 'from module import function1, function2, function3'
    var_15 = '\r\n'
    var_16 = 'from module import function1'
    var_17 = ''
    var_18 = '# some comment'



# Parsed testcases at query #24
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = 'from module import function1, function2, function3'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    var_5 = 'from module import function1, function2  # comment'
    var_6 = module_1.line(var_5, var_3, var_1)
    var_7 = 'from module import function1, function2, function3'
    var_8 = True
    var_9 = module_0.Config()
    var_10 = 'from module import function1, function2, function3'
    var_11 = module_1.line(var_10, var_3, var_9)
    var_12 = module_0.Config()
    var_13 = 'from module import function1, function2, function3'
    var_14 = module_1.line(var_13, var_3, var_12)
    var_15 = ','
    var_16 = 'from module import function1, function2, function3'
    var_17 = 'from module import function1 as f1, function2 as f2'
    var_18 = module_1.line(var_17, var_3, var_1)
    var_19 = 'cimport module.function1, module.function2'
    var_20 = module_1.line(var_19, var_3, var_1)
    var_21 = 'from module.submodule import function1, function2'
    var_22 = module_1.line(var_21, var_3, var_1)
    var_23 = module_0.Config()
    var_24 = 'from module import function1, function2, function3'
    var_25 = module_1.line(var_24, var_3, var_23)
    var_26 = module_0.Config()
    var_27 = 'from module import function1, function2  # comment'
    var_28 = module_1.line(var_27, var_3, var_26)
    var_29 = '# '
    var_30 = module_0.Config()
    var_31 = 'from module import function1, function2  # comment'
    var_32 = module_1.line(var_31, var_3, var_30)
    var_33 = '    '
    var_34 = module_0.Config()
    var_35 = 'from module import function1, function2, function3'
    var_36 = module_1.line(var_35, var_3, var_34)
    var_37 = 79
    var_38 = module_0.Config()
    var_39 = 'from module import function1, function2, function3'
    var_40 = module_1.line(var_39, var_3, var_38)



# Parsed testcases at query #25
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
    var_7 = 'from module import function  # some comment'
    var_8 = module_0.line(var_7, var_1, var_5)
    var_9 = 'from module import function  # NOQA'
    var_10 = module_0.line(var_9, var_1, var_5)
    var_11 = 'from module import function  # some comment noqa'
    var_12 = module_0.line(var_11, var_1, var_5)
    var_13 = 'from module import function as alias'
    var_14 = module_0.line(var_13, var_1, var_5)
    var_15 = True
    var_16 = module_1.Config()
    var_17 = 'from module import function1, function2, function3'
    var_18 = module_0.line(var_17, var_1, var_16)
    var_19 = module_1.Config()
    var_20 = 'from module import function1, function2, function3'
    var_21 = module_0.line(var_20, var_1, var_19)
    var_22 = ','
    var_23 = 'from module import function1, function2, function3'
    var_24 = module_1.Config()
    var_25 = 'from module import function1, function2, function3'
    var_26 = module_0.line(var_25, var_1, var_24)



# Parsed testcases at query #26
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1
import re as module_2

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'func1'
    var_2 = 'func2'
    var_3 = [var_1, var_2]
    var_4 = module_0.import_statement(var_0, var_3)
    assert var_4 == 'from module import func1, func2'
    var_5 = 'func3'
    var_6 = 'func4'
    var_7 = [var_1, var_2, var_5, var_6]
    var_8 = 20
    var_9 = module_1.Config()
    var_10 = module_0.import_statement(var_0, var_7, config=var_9)
    var_11 = [var_1, var_2]
    var_12 = '# Comment 1'
    var_13 = '# Comment 2'
    var_14 = [var_12, var_13]
    var_15 = module_0.import_statement(var_0, var_11, var_14)
    var_16 = [var_1, var_2]
    var_17 = True
    var_18 = module_0.import_statement(var_0, var_16, explode=var_17)
    var_19 = 'from module import (\n'
    var_20 = [var_1, var_2]
    var_21 = '\r\n'
    var_22 = module_0.import_statement(var_0, var_20, line_separator=var_21)
    var_23 = 30
    var_24 = module_1.Config()
    var_25 = [var_1, var_2, var_5]
    var_26 = module_0.import_statement(var_0, var_25, config=var_24)
    var_27 = '\n'
    var_28 = module_2.split(var_27)
    var_29 = len(var_28)
    var_30 = -1
    var_31 = var_28[var_30]
    var_32 = len(var_31)
    var_33 = -1
    var_34 = var_28[:var_33]
    var_35 = module_1.Config()
    var_36 = [var_1, var_2]
    var_37 = module_0.import_statement(var_0, var_36, config=var_35)
    var_38 = ','
    var_39 = module_1.Config()
    var_40 = [var_1, var_2]
    var_41 = '# Comment'
    var_42 = [var_41]
    var_43 = module_0.import_statement(var_0, var_40, var_42, config=var_39)
    var_44 = '    '
    var_45 = module_1.Config()
    var_46 = [var_1, var_2]
    var_47 = module_0.import_statement(var_0, var_46, config=var_45)
    var_48 = 'from module import'
    var_49 = 'func1'
    var_50 = 'func2'
    var_51 = 'func3'
    var_52 = [var_49, var_50, var_51]
    var_53 = 20
    var_54 = module_1.Config()



# Parsed testcases at query #27
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import function1, function2, function3'
    var_1 = 50
    var_2 = module_0.Config()
    var_3 = '\n'
    var_4 = module_1.line(var_0, var_3, var_2)
    var_5 = 'from module import function1, function2, function3  # some comment'
    var_6 = module_1.line(var_5, var_3, var_2)
    var_7 = 'from module import function1, function2, function3  # NOQA'
    var_8 = 30
    var_9 = module_1.line(var_7, var_3, var_2)
    var_10 = 'from module import function1, function2, function3, function4, function5'
    var_11 = True
    var_12 = module_0.Config()
    var_13 = module_1.line(var_10, var_3, var_12)
    var_14 = 'from module import function1 as f1, function2 as f2'
    var_15 = module_0.Config()
    var_16 = module_1.line(var_14, var_3, var_15)
    var_17 = 'from module import function'
    var_18 = module_0.Config()
    var_19 = module_1.line(var_17, var_3, var_18)
    var_20 = 'from module import function1, function2, function3'
    var_21 = module_1.line(var_20, var_3, var_18)
    var_22 = 'from module import function1, function2, function3'
    var_23 = module_0.Config()
    var_24 = module_1.line(var_22, var_3, var_23)
    var_25 = 'from module import function1, function2, function3  # comment'
    var_26 = module_0.Config()
    var_27 = module_1.line(var_25, var_3, var_26)
    var_28 = 'from module import function1, function2, function3  # comment'
    var_29 = '# '
    var_30 = module_0.Config()
    var_31 = module_1.line(var_28, var_3, var_30)



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'from module import very_long_function_name'
    var_1 = '\n'
    var_2 = 'from module import func  # some comment'
    var_3 = 'from module import very_long_function_name  # NOQA'
    var_4 = 'from module import very_long_function_name'
    var_5 = 'from module import very_long_function_name'
    var_6 = 'from module import very_long_function_name as alias'
    var_7 = 'from module import very_long_function_name'
    var_8 = 'from module import func  # some comment'
    var_9 = 'from module import very_long_function_name'
    var_10 = 'from module import very_long_function_name'
    var_11 = 'from module import very_long_function_name'
    var_12 = '\r\n'



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 50
    var_1 = 'from module import very_long_function_name_that_exceeds_line_length'
    var_2 = '\n'
    var_3 = 'from module import func  # This is a comment'
    var_4 = 'from module import func  # NOQA'
    var_5 = 'from module import func'
    var_6 = 'from module import func as f'
    var_7 = 'from module.submodule import func'
    var_8 = ','



# Parsed testcases at query #30
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
    var_8 = module_0.line(var_7, var_1)
    var_9 = 'from module import very_long_function_name_that_exceeds_line_length  # NOQA'
    var_10 = module_0.line(var_9, var_1, var_5)
    var_11 = True
    var_12 = module_1.Config()
    var_13 = module_0.line(var_3, var_1, var_12)
    assert var_13 == 'from module import (\n    very_long_function_name_that_exceeds_line_length\n)'
    var_14 = module_1.Config()
    var_15 = module_0.line(var_3, var_1, var_14)
    assert var_15 == 'from module import (\n    very_long_function_name_that_exceeds_line_length,\n)'
    var_16 = 'from module import function as alias'
    var_17 = module_1.Config()
    var_18 = module_0.line(var_16, var_1, var_17)
    assert var_18 == 'from module import function as (\n    alias\n)'
    var_19 = 'cimport module.function'
    var_20 = module_1.Config()
    var_21 = module_0.line(var_19, var_1, var_20)
    assert var_21 == 'cimport module.(\n    function\n)'
    var_22 = 'from module import Class.method'
    var_23 = module_1.Config()
    var_24 = module_0.line(var_22, var_1, var_23)
    assert var_24 == 'from module import Class.(\n    method\n)'
    var_25 = 'from module import function  # some comment'
    var_26 = module_1.Config()
    var_27 = module_0.line(var_25, var_1, var_26)
    assert var_27 == 'from module import function'
    var_28 = 'from module import function  # some comment'
    var_29 = ' # '
    var_30 = module_1.Config()
    var_31 = module_0.line(var_28, var_1, var_30)
    var_32 = 'from module import function1, function2, function3'
    var_33 = module_1.Config()
    var_34 = module_0.line(var_32, var_1, var_33)
    assert var_34 == 'from module import (\n    function1,\n    function2,\n    function3\n)'
    var_35 = 'from module import function1, function2, function3'
    var_36 = module_0.line(var_35, var_1, var_33)
    assert var_36 == 'from module import (\n    function1,\n    function2,\n    function3\n)'
    var_37 = 'from module import function1, function2, function3'
    var_38 = module_0.line(var_37, var_1, var_33)
    assert var_38 == 'from module import (\n    function1,\n    function2,\n    function3\n)'
    var_39 = 'from module import very_long_function_name_that_exceeds_line_length  # noqa'
    var_40 = module_0.line(var_39, var_1, var_33)



# Parsed testcases at query #31
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
    var_16 = module_1.Config()
    var_17 = [var_1, var_2]
    var_18 = module_0.import_statement(var_0, var_17, config=var_16)
    var_19 = 'func3'
    var_20 = 'func4'
    var_21 = 'func5'
    var_22 = [var_1, var_2, var_19, var_20, var_21]
    var_23 = module_0.import_statement(var_0, var_22)
    var_24 = module_1.Config()
    var_25 = [var_1, var_2]
    var_26 = module_0.import_statement(var_0, var_25, config=var_24)
    var_27 = module_1.Config()
    var_28 = [var_1, var_2]
    var_29 = [var_6]
    var_30 = module_0.import_statement(var_0, var_28, var_29, config=var_27)



# Parsed testcases at query #32
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
    var_8 = True
    var_9 = module_1.Config()
    var_10 = module_0.line(var_7, var_1, var_9)
    assert var_10 == 'from module import (\n    function,  # some comment\n)'
    var_11 = 'from module import very_long_function_name_that_exceeds_line_length  # NOQA'
    var_12 = module_0.line(var_11, var_1, var_9)
    var_13 = 'from module import function1, function2'
    var_14 = module_1.Config()
    var_15 = module_0.line(var_13, var_1, var_14)
    assert var_15 == 'from module import (\n    function1,\n    function2,\n)'
    var_16 = 'from module import function as alias'
    var_17 = module_1.Config()
    var_18 = module_0.line(var_16, var_1, var_17)
    assert var_18 == 'from module import function\\\n    as alias'
    var_19 = 'from module import submodule.function'
    var_20 = module_1.Config()
    var_21 = module_0.line(var_19, var_1, var_20)
    assert var_21 == 'from module import submodule\\\n    .function'
    var_22 = 'from module import very_long_function_name_that_exceeds_line_length  # noqa'
    var_23 = module_1.Config()
    var_24 = module_0.line(var_22, var_1, var_23)
    assert var_24 == 'from module import (\n    very_long_function_name_that_exceeds_line_length,  # noqa\n)'
    var_25 = 'from module import function1, function2, function3'
    var_26 = module_1.Config()
    var_27 = module_0.line(var_25, var_1, var_26)
    assert var_27 == 'from module import (\n    function1,\n    function2,\n    function3,\n)'
    var_28 = 'from module import function  # some comment'
    var_29 = module_1.Config()
    var_30 = module_0.line(var_28, var_1, var_29)
    assert var_30 == 'from module import function'



# Parsed testcases at query #33
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 50
    var_1 = False
    var_2 = module_0.Config()
    var_3 = 'from module import very_long_function_name'
    var_4 = '\n'
    var_5 = module_1.line(var_3, var_4, var_2)
    assert var_5 == 'from module import very_long_function_name'
    var_6 = 20
    var_7 = True
    var_8 = module_0.Config()
    var_9 = module_1.line(var_3, var_4, var_8)
    assert var_9 == 'from module import (\n    very_long_function_name\n)'
    var_10 = module_0.Config()
    var_11 = 'from module import func # some comment'
    var_12 = module_1.line(var_11, var_4, var_10)
    assert var_12 == 'from module import (\n    func  # some comment\n)'
    var_13 = module_0.Config()
    var_14 = 'from module import very_long_function_name # NOQA'
    var_15 = module_1.line(var_14, var_4, var_13)
    assert var_15 == 'from module import very_long_function_name # NOQA'
    var_16 = module_0.Config()
    var_17 = 'import module as very_long_alias'
    var_18 = module_1.line(var_17, var_4, var_16)
    assert var_18 == 'import module as (\n    very_long_alias\n)'
    var_19 = module_0.Config()
    var_20 = 'cimport module.very_long_function_name'
    var_21 = module_1.line(var_20, var_4, var_19)
    assert var_21 == 'cimport module.(\n    very_long_function_name\n)'
    var_22 = module_0.Config()
    var_23 = module_1.line(var_3, var_4, var_22)
    assert var_23 == 'from module import (\n    very_long_function_name\n)'
    var_24 = module_0.Config()
    var_25 = 'from module import func1, func2'
    var_26 = module_1.line(var_25, var_4, var_24)
    assert var_26 == 'from module import (\n    func1,\n    func2,\n)'
    var_27 = module_0.Config()
    var_28 = 'from module import func # noqa'
    var_29 = module_1.line(var_28, var_4, var_27)
    assert var_29 == 'from module import (\n    func  # noqa\n)'
    var_30 = module_1.line(var_25, var_4, var_27)
    assert var_30 == 'from module import (\n    func1,\n    func2,\n)'
    var_31 = module_1.line(var_25, var_4, var_27)
    assert var_31 == 'from module import (\n    func1,\n    func2,\n)'
    var_32 = module_0.Config()
    var_33 = module_1.line(var_11, var_4, var_32)
    assert var_33 == 'from module import (\n    func\n)'
    var_34 = module_0.Config()
    var_35 = '\r\n'
    var_36 = module_1.line(var_3, var_35, var_34)
    assert var_36 == 'from module import (\r\n    very_long_function_name\r\n)'
    var_37 = module_0.Config()
    var_38 = 'from module import func'
    var_39 = module_1.line(var_38, var_4, var_37)
    assert var_39 == 'from module import func'
    var_40 = module_1.line(var_3, var_4, var_37)
    assert var_40 == 'from module import very_long_function_name # NOQA'



# Parsed testcases at query #34
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

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
    var_18 = module_1.Config()
    var_19 = [var_1, var_2, var_3]
    var_20 = module_0.import_statement(var_0, var_19, config=var_18)
    var_21 = [var_1, var_2]
    var_22 = module_1.Config()
    var_23 = [var_1, var_2]
    var_24 = module_0.import_statement(var_0, var_23, config=var_22)
    var_25 = ','
    var_26 = module_1.Config()
    var_27 = [var_1, var_2]
    var_28 = '# Comment'
    var_29 = [var_28]
    var_30 = module_0.import_statement(var_0, var_27, var_29, config=var_26)



# Parsed testcases at query #35
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'from module import something, something_else, another_thing'
    var_2 = '\n'
    var_3 = module_1.line(var_1, var_2, var_0)
    var_4 = 'from module import something  # some comment'
    var_5 = module_1.line(var_4, var_2, var_0)
    var_6 = 'from module import something, something_else, another_thing  # NOQA'
    var_7 = module_1.line(var_6, var_2, var_0)
    var_8 = 'from module import something, something_else, another_thing, more_things'
    var_9 = True
    var_10 = module_0.Config()
    var_11 = 'from module import something, something_else, another_thing'
    var_12 = module_1.line(var_11, var_2, var_10)
    var_13 = module_0.Config()
    var_14 = 'from module import something, something_else, another_thing'
    var_15 = module_1.line(var_14, var_2, var_13)
    var_16 = ','
    var_17 = 'from module import something, something_else, another_thing'
    var_18 = module_0.Config()
    var_19 = 'from module import something, something_else, another_thing'
    var_20 = module_1.line(var_19, var_2, var_18)
    var_21 = 'from module import something, something_else, another_thing'
    var_22 = '\r\n'
    var_23 = module_1.line(var_21, var_22, var_0)
    var_24 = 'from module import something'
    var_25 = module_1.line(var_24, var_2, var_0)



# Parsed testcases at query #36
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
    var_8 = module_0.line(var_7, var_1)
    var_9 = 'from module import very_long_function_name_that_exceeds_line_length  # NOQA'
    var_10 = module_0.line(var_9, var_1, var_5)
    var_11 = 'from module import function as alias'
    var_12 = module_0.line(var_11, var_1)
    var_13 = 'from module.submodule import function'
    var_14 = module_0.line(var_13, var_1)
    var_15 = True
    var_16 = module_1.Config()
    var_17 = 'from module import very_long_function_name_that_exceeds_line_length'
    var_18 = module_0.line(var_17, var_1, var_16)
    assert var_18 == 'from module import (\n    very_long_function_name_that_exceeds_line_length\n)'
    var_19 = module_1.Config()
    var_20 = 'from module import very_long_function_name_that_exceeds_line_length'
    var_21 = module_0.line(var_20, var_1, var_19)
    assert var_21 == 'from module import (\n    very_long_function_name_that_exceeds_line_length,\n)'
    var_22 = 'from module import very_long_function_name_that_exceeds_line_length'
    var_23 = module_0.line(var_22, var_1, var_19)
    assert var_23 == 'from module import (\n    very_long_function_name_that_exceeds_line_length,\n)'
    var_24 = 'from module import very_long_function_name_that_exceeds_line_length'
    var_25 = module_0.line(var_24, var_1, var_19)
    assert var_25 == 'from module import (\n    very_long_function_name_that_exceeds_line_length,\n)'



# Parsed testcases at query #37
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'item1'
    var_2 = 'item2'
    var_3 = [var_1, var_2]
    var_4 = module_0.import_statement(var_0, var_3)
    assert var_4 == 'from module import item1, item2'
    var_5 = [var_1, var_2]
    var_6 = '# Comment 1'
    var_7 = '# Comment 2'
    var_8 = [var_6, var_7]
    var_9 = module_0.import_statement(var_0, var_5, var_8)
    var_10 = [var_1, var_2]
    var_11 = '\r\n'
    var_12 = module_0.import_statement(var_0, var_10, line_separator=var_11)
    var_13 = [var_1, var_2]
    var_14 = True
    var_15 = module_0.import_statement(var_0, var_13, explode=var_14)
    var_16 = 'from module import (\n    item1,\n    item2,\n)'
    var_17 = 50
    var_18 = '    '
    var_19 = '# '
    var_20 = module_1.Config()
    var_21 = 'item3'
    var_22 = [var_1, var_2, var_21]
    var_23 = module_0.import_statement(var_0, var_22, config=var_20)
    var_24 = 30
    var_25 = module_1.Config()
    var_26 = [var_1, var_2, var_21]
    var_27 = module_0.import_statement(var_0, var_26, config=var_25)
    var_28 = -1
    var_29 = '\n'
    var_30 = result.split(var_29)[var_28]
    var_31 = len(var_30)
    var_32 = 0
    var_33 = result.split(var_29)[var_32]
    var_34 = len(var_33)
    var_35 = [var_1, var_2, var_21]
    var_36 = module_1.Config()
    var_37 = [var_1, var_2]
    var_38 = [var_6]
    var_39 = module_0.import_statement(var_0, var_37, var_38, config=var_36)
    var_40 = module_1.Config()
    var_41 = [var_1, var_2, var_21]
    var_42 = module_0.import_statement(var_0, var_41, config=var_40)
    var_43 = 'from module import ('
    var_44 = ','
    var_45 = [var_1, var_2]
    var_46 = '# NOQA'
    var_47 = [var_46]
    var_48 = module_0.import_statement(var_0, var_45, var_47, config=var_40)



# Parsed testcases at query #38
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'function1'
    var_2 = 'function2'
    var_3 = [var_1, var_2]
    var_4 = module_0.import_statement(var_0, var_3)
    assert var_4 == 'from module import function1, function2'
    var_5 = 'function3'
    var_6 = [var_1, var_2, var_5]
    var_7 = module_0.import_statement(var_0, var_6)
    var_8 = [var_1, var_2]
    var_9 = '\r\n'
    var_10 = module_0.import_statement(var_0, var_8, line_separator=var_9)
    var_11 = [var_1, var_2]
    var_12 = '# Comment 1'
    var_13 = '# Comment 2'
    var_14 = [var_12, var_13]
    var_15 = module_0.import_statement(var_0, var_11, var_14)
    var_16 = [var_1, var_2]
    var_17 = True
    var_18 = module_0.import_statement(var_0, var_16, explode=var_17)
    var_19 = '\n'
    var_20 = 50
    var_21 = [var_1, var_2, var_5]
    var_22 = 20
    var_23 = module_1.Config()
    var_24 = [var_1, var_2]
    var_25 = module_0.import_statement(var_0, var_24, config=var_23)
    var_26 = 0
    var_27 = result.split(var_19)[var_26]
    var_28 = len(var_27)
    var_29 = [var_1, var_2]
    var_30 = []
    var_31 = module_0.import_statement(var_0, var_30)
    assert var_31 == 'from module import'
    var_32 = 'very_long_function_name_1'
    var_33 = 'very_long_function_name_2'
    var_34 = [var_32, var_33]
    var_35 = module_0.import_statement(var_0, var_34)
    var_36 = 'short'
    var_37 = [var_36]
    var_38 = module_0.import_statement(var_0, var_37)
    assert var_38 == 'from module import short'



# Parsed testcases at query #39
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
    var_18 = 50
    var_19 = module_1.Config()
    var_20 = 'very_long_item_name_1'
    var_21 = 'very_long_item_name_2'
    var_22 = [var_20, var_21]
    var_23 = module_0.import_statement(var_0, var_22, config=var_19)
    var_24 = module_2.split(var_17)
    var_25 = len(var_24)
    var_26 = -1
    var_27 = var_24[var_26]
    var_28 = len(var_27)
    var_29 = -1
    var_30 = var_24[:var_29]
    var_31 = [var_1, var_2, var_3]
    var_32 = module_1.Config()
    var_33 = [var_1, var_2]
    var_34 = module_0.import_statement(var_0, var_33, config=var_32)
    var_35 = ','
    var_36 = module_1.Config()
    var_37 = [var_1, var_2]
    var_38 = '# This should be ignored'
    var_39 = [var_38]
    var_40 = module_0.import_statement(var_0, var_37, var_39, config=var_36)



# Parsed testcases at query #40
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'from module import very_long_function_name_that_exceeds_line_length'
    var_2 = '\n'
    var_3 = module_1.line(var_1, var_2, var_0)
    var_4 = 'from module import something  # some comment'
    var_5 = module_1.line(var_4, var_2, var_0)
    var_6 = 'from module import very_long_function_name_that_exceeds_line_length  # comment'
    var_7 = module_1.line(var_6, var_2, var_0)
    var_8 = 'from module import very_long_function_name_that_exceeds_line_length'
    var_9 = module_1.line(var_8, var_2, var_0)
    var_10 = 'from module import very_long_function_name_that_exceeds_line_length'
    var_11 = module_1.line(var_10, var_2, var_0)
    var_12 = 'from module import very_long_function_name_that_exceeds_line_length as alias'
    var_13 = module_1.line(var_12, var_2, var_0)
    var_14 = 'from module import very_long_function_name_that_exceeds_line_length'
    var_15 = module_1.line(var_14, var_2, var_0)



# Parsed testcases at query #41
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
    var_18 = [var_1, var_2, var_3]
    var_19 = 30
    var_20 = module_1.Config()
    var_21 = [var_1, var_2, var_3]
    var_22 = module_0.import_statement(var_0, var_21, config=var_20)
    var_23 = module_1.Config()
    var_24 = [var_1, var_2]
    var_25 = [var_7, var_8]
    var_26 = module_0.import_statement(var_0, var_24, var_25, config=var_23)
    var_27 = [var_1, var_2]



# Parsed testcases at query #42
#--------------------------


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'item1'
    var_2 = 'item2'
    var_3 = [var_1, var_2]
    var_4 = module_0.import_statement(var_0, var_3)
    assert var_4 == 'from module import item1, item2'
    assert var_4 == 'from module import (\n    item1,\n    item2,\n    item3,\n)'
    var_5 = 'item3'
    var_6 = [var_1, var_2, var_5]
    var_7 = 20
    var_8 = [var_1, var_2]
    var_9 = '# comment1'
    var_10 = '# comment2'
    var_11 = [var_9, var_10]
    var_12 = module_0.import_statement(var_0, var_8, var_11)
    var_13 = [var_1, var_2]
    var_14 = True
    var_15 = module_0.import_statement(var_0, var_13, explode=var_14)
    assert var_15 == 'from module import (\n    item1,\n    item2,\n)'
    assert var_15 == 'from module import (\n    item1,\n    item2,\n    item3,\n)'
    var_16 = [var_1, var_2, var_5]
    var_17 = [var_1, var_2]
    var_18 = '\r\n'
    var_19 = module_0.import_statement(var_0, var_17, line_separator=var_18)
    assert var_19 == 'from module import (\n    item1,\n    item2,\n)'
    var_20 = [var_1, var_2]
    var_21 = 'item2,\n)'
    var_22 = [var_1, var_2]
    var_23 = [var_9]
    var_24 = [var_1, var_2]
    var_25 = '    '
    var_26 = [var_1, var_2]
    var_27 = [var_9]
    var_28 = '  # '



# Parsed testcases at query #43
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import function'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = 'from module import function1, function2, function3'
    var_4 = 30
    var_5 = module_1.Config()
    var_6 = module_0.line(var_3, var_1, var_5)
    assert var_6 == 'from module import (\n    function1,\n    function2,\n    function3,\n)'
    var_7 = 'from module import function  # comment'
    var_8 = module_1.Config()
    var_9 = module_0.line(var_7, var_1, var_8)
    assert var_9 == 'from module import (\n    function,  # comment\n)'
    var_10 = 'from module import function1, function2, function3'
    var_11 = module_0.line(var_10, var_1, var_8)
    assert var_11 == 'from module import function1, function2, function3  # NOQA'
    var_12 = 'from module import function as f'
    var_13 = 20
    var_14 = module_1.Config()
    var_15 = module_0.line(var_12, var_1, var_14)
    assert var_15 == 'from module import (\n    function as f,\n)'
    var_16 = 'cimport module.function'
    var_17 = module_1.Config()
    var_18 = module_0.line(var_16, var_1, var_17)
    assert var_18 == 'cimport (\n    module.function,\n)'
    var_19 = 'from module import function.subfunction'
    var_20 = module_1.Config()
    var_21 = module_0.line(var_19, var_1, var_20)
    assert var_21 == 'from module import (\n    function.subfunction,\n)'
    var_22 = 'from module import function  # noqa'
    var_23 = module_1.Config()
    var_24 = module_0.line(var_22, var_1, var_23)
    assert var_24 == 'from module import (\n    function,  # noqa\n)'
    var_25 = 'from module import function1, function2'
    var_26 = True
    var_27 = module_1.Config()
    var_28 = module_0.line(var_25, var_1, var_27)
    assert var_28 == 'from module import (\n    function1,\n    function2,\n)'
    var_29 = 'from module import function1, function2'
    var_30 = module_1.Config()
    var_31 = module_0.line(var_29, var_1, var_30)
    assert var_31 == 'from module import (\n    function1,\n    function2,\n)'
    var_32 = 'from module import function1, function2'
    var_33 = module_0.line(var_32, var_1, var_30)
    assert var_33 == 'from module import (\n    function1,\n    function2,\n)'
    var_34 = 'from module import function1, function2'
    var_35 = module_0.line(var_34, var_1, var_30)
    assert var_35 == 'from module import (\n    function1,\n    function2,\n)'



# Parsed testcases at query #44
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
    var_18 = module_1.Config()
    var_19 = [var_1, var_2, var_3]
    var_20 = module_0.import_statement(var_0, var_19, config=var_18)
    var_21 = module_2.split(var_17)
    var_22 = -1
    var_23 = var_21[var_22]
    var_24 = len(var_23)
    var_25 = -1
    var_26 = var_21[:var_25]
    var_27 = min(var_6)
    var_28 = module_1.Config()
    var_29 = [var_23, var_24, var_25]
    var_30 = module_0.import_statement(var_22, var_29, config=var_28)
    var_31 = [var_23, var_24, var_25]
    var_32 = module_1.Config()
    var_33 = [var_23, var_24]
    var_34 = module_0.import_statement(var_22, var_33, config=var_32)
    var_35 = ','
    var_36 = module_1.Config()
    var_37 = [var_23, var_24]
    var_38 = [var_27]
    var_39 = module_0.import_statement(var_22, var_37, var_38, config=var_36)



# Parsed testcases at query #45
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1
import re as module_2

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    var_6 = [var_1, var_2, var_3]
    var_7 = '# comment'
    var_8 = [var_7]
    var_9 = module_0.import_statement(var_0, var_6, var_8)
    var_10 = [var_1, var_2, var_3]
    var_11 = '\r\n'
    var_12 = module_0.import_statement(var_0, var_10, line_separator=var_11)
    var_13 = [var_1, var_2, var_3]
    var_14 = True
    var_15 = module_0.import_statement(var_0, var_13, explode=var_14)
    var_16 = 20
    var_17 = module_1.Config()
    var_18 = [var_1, var_2, var_3]
    var_19 = module_0.import_statement(var_0, var_18, config=var_17)
    var_20 = '\n'
    var_21 = [var_1, var_2, var_3]
    var_22 = 30
    var_23 = module_1.Config()
    var_24 = [var_1, var_2, var_3]
    var_25 = module_0.import_statement(var_0, var_24, config=var_23)
    var_26 = module_2.split(var_20)
    var_27 = -1
    var_28 = var_26[var_27]
    var_29 = len(var_28)
    var_30 = -1
    var_31 = var_26[:var_30]
    var_32 = min(var_6)
    var_33 = [var_28]
    var_34 = module_0.import_statement(var_27, var_33)
    assert var_34 == 'from module import a'



# Parsed testcases at query #46
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
    var_14 = [var_1, var_2]
    var_15 = True
    var_16 = module_0.import_statement(var_0, var_14, explode=var_15)
    var_17 = '\n'
    var_18 = module_1.Config()
    var_19 = [var_1, var_2, var_3]
    var_20 = module_0.import_statement(var_0, var_19, config=var_18)
    var_21 = module_2.split(var_17)
    var_22 = len(var_21)
    var_23 = -1
    var_24 = var_21[var_23]
    var_25 = len(var_24)
    var_26 = -1
    var_27 = var_21[:var_26]
    var_28 = [var_1, var_2]
    var_29 = module_1.Config()
    var_30 = [var_1, var_2]
    var_31 = module_0.import_statement(var_0, var_30, config=var_29)
    var_32 = ','
    var_33 = module_1.Config()
    var_34 = [var_1, var_2]
    var_35 = [var_7]
    var_36 = module_0.import_statement(var_0, var_34, var_35, config=var_33)
    var_37 = '    '
    var_38 = module_1.Config()
    var_39 = [var_1, var_2]
    var_40 = module_0.import_statement(var_0, var_39, config=var_38)
    var_41 = '# '
    var_42 = module_1.Config()
    var_43 = [var_1, var_2]
    var_44 = 'Comment 1'
    var_45 = [var_44]
    var_46 = module_0.import_statement(var_0, var_43, var_45, config=var_42)
    var_47 = 20
    var_48 = module_1.Config()
    var_49 = [var_1, var_2]
    var_50 = module_0.import_statement(var_0, var_49, config=var_48)
    var_51 = module_2.split(var_17)
    var_52 = module_1.Config()
    var_53 = [var_1, var_2]
    var_54 = module_0.import_statement(var_0, var_53, config=var_52)



# Parsed testcases at query #47
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
    var_18 = 50
    var_19 = 40
    var_20 = module_1.Config()
    var_21 = [var_1, var_2, var_3]
    var_22 = module_0.import_statement(var_0, var_21, config=var_20)
    var_23 = 0
    var_24 = result.split(var_17)[var_23]
    var_25 = len(var_24)
    var_26 = [var_1, var_2, var_3]
    var_27 = []
    var_28 = module_0.import_statement(var_0, var_27)
    var_29 = [var_1]
    var_30 = module_0.import_statement(var_0, var_29)
    var_31 = module_1.Config()
    var_32 = [var_1, var_2, var_3]
    var_33 = module_0.import_statement(var_0, var_32, config=var_31)
    var_34 = module_2.split(var_17)
    var_35 = -1
    var_36 = var_34[:var_35]
    var_37 = min(var_2)
    var_38 = -1
    var_39 = var_34[var_38]
    var_40 = len(var_39)



# Parsed testcases at query #48
#--------------------------


def test_case_0():
    var_0 = 'from module import function1, function2, function3'
    var_1 = 50
    var_2 = '\n'
    var_3 = 'from module import function1, function2, function3  # some comment'
    var_4 = 'from module import function1, function2, function3  # NOQA'
    var_5 = 30
    var_6 = 'from module import function'
    var_7 = True
    var_8 = 'from module import function1, function2, function3'
    var_9 = 'from module import function1, function2, function3'
    var_10 = ','
    var_11 = 'from module import function1, function2, function3'
    var_12 = 'from module import function1, function2, function3'
    var_13 = '\r\n'
    var_14 = 'from module import function1, function2, function3  # some comment'



# Parsed testcases at query #49
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
    assert var_5 == 'from module import func1, func2, func3'
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
    var_17 = '\n'
    var_18 = module_1.Config()
    var_19 = [var_1, var_2, var_3]
    var_20 = module_0.import_statement(var_0, var_19, config=var_18)
    var_21 = [var_1, var_2, var_3]
    var_22 = module_1.Config()
    var_23 = [var_1, var_2, var_3]
    var_24 = module_0.import_statement(var_0, var_23, config=var_22)
    var_25 = ','
    var_26 = module_1.Config()
    var_27 = [var_1, var_2, var_3]
    var_28 = [var_7, var_8]
    var_29 = module_0.import_statement(var_0, var_27, var_28, config=var_26)
    var_30 = '    '
    var_31 = module_1.Config()
    var_32 = [var_1, var_2, var_3]
    var_33 = module_0.import_statement(var_0, var_32, config=var_31)
    var_34 = '# '
    var_35 = module_1.Config()
    var_36 = [var_1, var_2, var_3]
    var_37 = 'Comment 1'
    var_38 = 'Comment 2'
    var_39 = [var_37, var_38]
    var_40 = module_0.import_statement(var_0, var_36, var_39, config=var_35)



# Parsed testcases at query #50
#--------------------------


import isort.wrap as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'import os'
    var_3 = 'import os, sys, json, ast, re, math, random, datetime, itertools, functools'
    var_4 = 50
    var_5 = 'import os, sys, json, ast, re, math, random, datetime, itertools, functools'
    var_6 = 'import os, sys, json, ast, re, math, random, datetime, itertools, functools # noqa'
    var_7 = 'import os, sys, json, ast, re, math, random, datetime, itertools, functools # noqa'
    var_8 = 'import os, sys, json, ast, re, math, random, datetime, itertools, functools # noqa'
    var_9 = True
    var_10 = 'import os, sys, json, ast, re, math, random, datetime, itertools, functools # noqa'
    var_11 = 'import os, sys, json, ast, re, math, random, datetime, itertools, functools # noqa'
    var_12 = 'import os, sys, json, ast, re, math, random, datetime, itertools, functools # noqa'
    var_13 = 'import os, sys, json, ast, re, math, random, datetime, itertools, functools # noqa'
    var_14 = 'import os, sys, json, ast, re, math, random, datetime, itertools, functools # noqa'
    var_15 = 'import os, sys, json, ast, re, math, random, datetime, itertools, functools # NOQA'
    var_16 = 'import os, sys, json, ast, re, math, random, datetime, itertools, functools # NOQA'
    var_17 = 'import os, sys, json, ast, re, math, random, datetime, itertools, functools # NOQA'
    var_18 = 'import os, sys, json, ast, re, math, random, datetime, itertools, functools'
    var_19 = 'import os, sys, json, ast, re, math, random, datetime, itertools, functools # NOQA'
    var_20 = 'import os, sys, json, ast, re, math, random, datetime, itertools, functools'
    var_21 = 'import os, sys, json, ast, re, math, random, datetime, itertools, functools # NOQA'
    var_22 = '    '
    var_23 = 'import os, sys, json, ast, re, math, random, datetime, itertools, functools'



# Parsed testcases at query #51
#--------------------------


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import something'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = 'from module import something_very_long, another_thing, third_thing'
    var_4 = 50
    var_5 = 'from module import something  # comment'
    var_6 = module_0.line(var_5, var_1)
    var_7 = 'from module import something_very_long, another_thing  # comment'
    var_8 = 'from module import something_very_long, another_thing  # NOQA'
    var_9 = module_0.line(var_8, var_1)
    var_10 = 'from module import something_very_long, another_thing  # noqa'
    var_11 = True
    var_12 = 'from module import something as alias'
    var_13 = module_0.line(var_12, var_1)
    var_14 = 'from module import something_very_long as alias'
    var_15 = 'cimport module.something'
    var_16 = module_0.line(var_15, var_1)
    var_17 = 'cimport module.something_very_long, another_thing'



# Parsed testcases at query #52
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1
import re as module_2

def test_case_0():
    var_0 = 'from module import '
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
    var_17 = 50
    var_18 = module_1.Config()
    var_19 = 'very_long_function_name_1'
    var_20 = 'very_long_function_name_2'
    var_21 = [var_19, var_20]
    var_22 = module_0.import_statement(var_0, var_21, config=var_18)
    var_23 = '\n'
    var_24 = module_2.split(var_23)
    var_25 = -1
    var_26 = var_24[var_25]
    var_27 = len(var_26)
    var_28 = -1
    var_29 = var_24[:var_28]
    var_30 = len(var_24)
    var_31 = var_30 == var_15
    var_32 = 'from module import '
    var_33 = 'function1'
    var_34 = 'function2'
    var_35 = 'function3'
    var_36 = [var_33, var_34, var_35]
    var_37 = module_1.Config()
    var_38 = [var_33, var_34]
    var_39 = module_0.import_statement(var_32, var_38, config=var_37)
    var_40 = ','
    var_41 = module_1.Config()
    var_42 = [var_33, var_34]
    var_43 = '# This should be ignored'
    var_44 = [var_43]
    var_45 = module_0.import_statement(var_32, var_42, var_44, config=var_41)
    var_46 = module_1.Config()
    var_47 = [var_33, var_34]
    var_48 = module_0.import_statement(var_32, var_47, config=var_46)
    var_49 = '    '
    var_50 = module_1.Config()
    var_51 = [var_33, var_34]
    var_52 = module_0.import_statement(var_32, var_51, config=var_50)
    var_53 = 'from module import'
    var_54 = '# '
    var_55 = module_1.Config()
    var_56 = [var_33, var_34]
    var_57 = 'Comment without #'
    var_58 = [var_57]
    var_59 = module_0.import_statement(var_32, var_56, var_58, config=var_55)



# Parsed testcases at query #53
#--------------------------


import isort.wrap as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'import os'
    var_3 = 20
    var_4 = 'import some_very_long_module_name'
    var_5 = 'import os  # some comment'
    var_6 = 'import os  # noqa'
    var_7 = 'import os as operating_system'
    var_8 = 'cimport os'
    var_9 = 'import os.path'
    var_10 = 'import os'
    var_11 = False
    var_12 = '\r\n'



# Parsed testcases at query #54
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
    assert var_10 == 'from module import very_long_function_name  # NOQA'
    var_11 = 'from module import func'
    var_12 = 50
    var_13 = module_0.Config()
    var_14 = module_1.line(var_11, var_3, var_13)
    assert var_14 == 'from module import func'
    var_15 = 'cimport module.very_long_function_name'
    var_16 = module_0.Config()
    var_17 = module_1.line(var_15, var_3, var_16)
    assert var_17 == 'cimport module.(\n    very_long_function_name\n)'
    var_18 = 'import module as very_long_alias'
    var_19 = module_0.Config()
    var_20 = module_1.line(var_18, var_3, var_19)
    assert var_20 == 'import module as (\n    very_long_alias\n)'
    var_21 = 'from module import very_long_module_name.submodule'
    var_22 = 30
    var_23 = module_0.Config()
    var_24 = module_1.line(var_21, var_3, var_23)
    assert var_24 == 'from module import (\n    very_long_module_name.submodule\n)'
    var_25 = 'from module import func  # noqa'
    var_26 = module_0.Config()
    var_27 = module_1.line(var_25, var_3, var_26)
    assert var_27 == 'from module import (\n    func,  # noqa\n)'
    var_28 = 'from module import func1, func2, func3'
    var_29 = module_0.Config()
    var_30 = module_1.line(var_28, var_3, var_29)
    assert var_30 == 'from module import (\n    func1,\n    func2,\n    func3,\n)'
    var_31 = 'from module import func1, func2, func3'
    var_32 = module_1.line(var_31, var_3, var_29)
    assert var_32 == 'from module import (\n    func1,\n    func2,\n    func3,\n)'
    var_33 = 'from module import func1, func2, func3'
    var_34 = module_1.line(var_33, var_3, var_29)
    assert var_34 == 'from module import (\n    func1,\n    func2,\n    func3,\n)'



# Parsed testcases at query #55
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
    var_23 = -1
    var_24 = var_22[var_23]
    var_25 = len(var_24)
    var_26 = -1
    var_27 = var_22[:var_26]
    var_28 = len(var_22)
    var_29 = var_28 == var_15
    var_30 = module_1.Config()
    var_31 = [var_1, var_2]
    var_32 = module_0.import_statement(var_0, var_31, config=var_30)
    var_33 = ','
    var_34 = [var_1, var_2, var_3]
    var_35 = module_1.Config()
    var_36 = [var_1, var_2]
    var_37 = [var_7]
    var_38 = module_0.import_statement(var_0, var_36, var_37, config=var_35)



# Parsed testcases at query #56
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'function1'
    var_2 = 'function2'
    var_3 = [var_1, var_2]
    var_4 = module_0.import_statement(var_0, var_3)
    var_5 = 'func1'
    var_6 = 'func2'
    var_7 = [var_5, var_6]
    var_8 = '# Comment'
    var_9 = [var_8]
    var_10 = module_0.import_statement(var_0, var_7, var_9)
    var_11 = [var_5, var_6]
    var_12 = True
    var_13 = module_0.import_statement(var_0, var_11, explode=var_12)
    var_14 = [var_5, var_6]
    var_15 = '\r\n'
    var_16 = module_0.import_statement(var_0, var_14, line_separator=var_15)
    var_17 = 20
    var_18 = module_1.Config()
    var_19 = [var_5, var_6]
    var_20 = module_0.import_statement(var_0, var_19, config=var_18)
    var_21 = 0
    var_22 = '\n'
    var_23 = result.split(var_22)[var_21]
    var_24 = len(var_23)
    var_25 = module_1.Config()
    var_26 = [var_5, var_6]
    var_27 = module_0.import_statement(var_0, var_26, config=var_25)
    var_28 = ','
    var_29 = [var_5, var_6]
    var_30 = module_1.Config()
    var_31 = [var_5, var_6]
    var_32 = [var_8]
    var_33 = module_0.import_statement(var_0, var_31, var_32, config=var_30)
    var_34 = [var_5]
    var_35 = module_0.import_statement(var_0, var_34)
    assert var_35 == 'from module import func1'



# Parsed testcases at query #57
#--------------------------


def test_case_0():
    var_0 = 'from module import function1, function2, function3'
    var_1 = 30
    var_2 = '\n'
    var_3 = 'from module import function1, function2, function3  # comment'
    var_4 = 'from module import function1, function2, function3  # NOQA'
    var_5 = 'from module import function1'
    var_6 = 50
    var_7 = 'from module import function1, function2, function3'
    var_8 = True
    var_9 = 'from module import function1, function2, function3'
    var_10 = 'from module import function1, function2, function3  # comment'
    var_11 = 'from module import function1, function2, function3'
    var_12 = 'from module import function1, function2, function3'
    var_13 = 'from module import function1, function2, function3'
    var_14 = 'from module import function1, function2, function3'
    var_15 = 'from module import function1, function2, function3'
    var_16 = 'from module import function1, function2, function3'



# Parsed testcases at query #58
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
    var_19 = module_1.Config()
    var_20 = 'very_long_function_name_1'
    var_21 = 'very_long_function_name_2'
    var_22 = [var_20, var_21]
    var_23 = module_0.import_statement(var_0, var_22, config=var_19)
    var_24 = module_2.split(var_17)
    var_25 = -1
    var_26 = var_24[var_25]
    var_27 = len(var_26)
    var_28 = -1
    var_29 = var_24[:var_28]
    var_30 = len(var_24)
    var_31 = var_30 == var_15
    var_32 = [var_1, var_2, var_3]
    var_33 = module_1.Config()
    var_34 = [var_1, var_2]
    var_35 = module_0.import_statement(var_0, var_34, config=var_33)
    var_36 = ','
    var_37 = module_1.Config()
    var_38 = [var_1, var_2]
    var_39 = '# Comment'
    var_40 = [var_39]
    var_41 = module_0.import_statement(var_0, var_38, var_40, config=var_37)
    var_42 = module_1.Config()
    var_43 = [var_1, var_2, var_3]
    var_44 = module_0.import_statement(var_0, var_43, config=var_42)
    var_45 = '# '
    var_46 = module_1.Config()
    var_47 = [var_1, var_2]
    var_48 = 'Comment'
    var_49 = [var_48]
    var_50 = module_0.import_statement(var_0, var_47, var_49, config=var_46)
    var_51 = 40
    var_52 = module_1.Config()
    var_53 = [var_20, var_21]
    var_54 = module_0.import_statement(var_0, var_53, config=var_52)
    var_55 = 0
    var_56 = result.split(var_17)[var_55]
    var_57 = len(var_56)



# Parsed testcases at query #59
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = 'from module import very_long_function_name'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    var_5 = 'from module import very_long_function_name, another_very_long_function_name'
    var_6 = module_1.line(var_5, var_3, var_1)
    var_7 = 'from module import func  # some comment'
    var_8 = module_1.line(var_7, var_3, var_1)
    var_9 = 'from module import very_long_function_name  # NOQA'
    var_10 = module_1.line(var_9, var_3, var_1)
    var_11 = 'from module import very_long_function_name, another_very_long_function_name'
    var_12 = module_1.line(var_11, var_3, var_1)
    var_13 = 'from module import very_long_function_name, another_very_long_function_name'
    var_14 = module_1.line(var_13, var_3, var_1)
    var_15 = -2
    var_16 = wrapped.split(var_3)[var_15]
    var_17 = 'from module import func  # some comment'
    var_18 = module_1.line(var_17, var_3, var_1)
    var_19 = 'from module import func  # some comment'
    var_20 = module_1.line(var_19, var_3, var_1)
    var_21 = 'from module import very_long_function_name, another_very_long_function_name'
    var_22 = '\n'
    var_23 = module_1.line(var_21, var_22, var_1)
    var_24 = 'from module import very_long_function_name, another_very_long_function_name'
    var_25 = module_1.line(var_24, var_3, var_1)
    var_26 = 'from module import very_long_function_name, another_very_long_function_name'
    var_27 = module_1.line(var_26, var_3, var_1)
    var_28 = '    '
    var_29 = 'from module import very_long_function_name, another_very_long_function_name'
    var_30 = '\r\n'
    var_31 = module_1.line(var_29, var_30, var_1)
    var_32 = ''
    var_33 = module_1.line(var_32, var_3, var_1)
    assert var_33 == ''
    var_34 = 'word'
    var_35 = module_1.line(var_34, var_3, var_1)
    assert var_35 == 'word'



# Parsed testcases at query #60
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
    var_7 = 'from module import function  # some comment'
    var_8 = 'from module import function  # NOQA'
    var_9 = 'from module import function as alias'
    var_10 = 'from module.submodule import function'
    var_11 = 'cimport module.function'
    var_12 = True
    var_13 = module_1.Config()
    var_14 = 'from module import function  # some comment'
    var_15 = module_0.line(var_14, var_1, var_13)
    assert var_15 == 'from module import (\n    function\n)'
    var_16 = module_1.Config()
    var_17 = 'from module import function'
    var_18 = module_0.line(var_17, var_1, var_16)
    assert var_18 == 'from module import (\n    function\n)'
    var_19 = module_1.Config()
    var_20 = 'from module import function'
    var_21 = module_0.line(var_20, var_1, var_19)
    assert var_21 == 'from module import (\n    function,\n)'
    var_22 = 'from module import function'
    var_23 = 'from module import function'
    var_24 = 'from module import function'
    var_25 = module_1.Config()
    var_26 = 'from module import function'
    var_27 = module_0.line(var_26, var_1, var_25)
    assert var_27 == 'from module import function'
    var_28 = 'from module import function'
    var_29 = '\r\n'
    var_30 = module_0.line(var_28, var_29)
    var_31 = ''
    var_32 = module_0.line(var_31, var_1)
    var_33 = '   '
    var_34 = module_0.line(var_33, var_1)
    var_35 = 'import os'
    var_36 = module_0.line(var_35, var_1)
    var_37 = 'import module'
    var_38 = 13
    var_39 = module_1.Config()
    var_40 = module_0.line(var_37, var_1, var_39)
    var_41 = 'import module_function'
    var_42 = 10
    var_43 = module_1.Config()
    var_44 = module_0.line(var_41, var_1, var_43)
    assert var_44 == 'import (\n    module_function\n)'
    var_45 = 'from module import function1, function2, function3'
    var_46 = 'from module import function  # NOQA: some comment'
    var_47 = module_1.Config()
    var_48 = 'from module import function  # some comment'
    var_49 = module_0.line(var_48, var_1, var_47)
    assert var_49 == 'from module import (\n    function,  # some comment\n)'
    var_50 = module_1.Config()
    var_51 = 'from module import function  # some comment'
    var_52 = module_0.line(var_51, var_1, var_50)
    assert var_52 == 'from module import (\n    function,  # some comment\n)'
    var_53 = 'from module import function  # some comment'
    var_54 = 'from module import function  # some comment'
    var_55 = 'from module import function  # some comment'
    var_56 = module_1.Config()
    var_57 = 'from module import function  # some comment'
    var_58 = module_0.line(var_57, var_1, var_56)
    assert var_58 == 'from module import function  # some comment'
    var_59 = 'from module import function  # some comment'
    var_60 = module_0.line(var_59, var_29)
    var_61 = ''
    var_62 = module_0.line(var_61, var_1)
    var_63 = '   '
    var_64 = module_0.line(var_63, var_1)
    var_65 = 'import os  # some comment'
    var_66 = module_0.line(var_65, var_1)
    var_67 = 'import module  # some comment'
    var_68 = 23
    var_69 = module_1.Config()
    var_70 = module_0.line(var_67, var_1, var_69)
    var_71 = 'import module_function  # some comment'
    var_72 = module_1.Config()



# Parsed testcases at query #61
#--------------------------


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import something'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'from module import something'
    var_3 = 'from module import something_very_long_function_name'
    var_4 = 30
    var_5 = 'from module import something  # some comment'
    var_6 = 'from module import something_very_long_function_name  # NOQA'
    var_7 = True
    var_8 = 'from module import something  # some comment'
    var_9 = '\r\n'
    var_10 = module_0.line(var_0, var_9)
    assert var_10 == 'from module import something'
    var_11 = 'from module import something  # some comment'
    var_12 = ' # '



# Parsed testcases at query #62
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import function1, function2, function3'
    var_1 = '\n'
    var_2 = 'from module import function1, function2, function3, function4, function5'
    var_3 = 50
    var_4 = module_0.Config()
    var_5 = module_1.line(var_2, var_1, var_4)
    var_6 = 'from module import function1  # some comment'
    var_7 = 'from module import function1, function2, function3  # NOQA'
    var_8 = True
    var_9 = 30
    var_10 = module_0.Config()
    var_11 = 'from module import function1, function2, function3'
    var_12 = module_1.line(var_11, var_1, var_10)
    var_13 = module_0.Config()
    var_14 = 'from module import function1, function2, function3'
    var_15 = module_1.line(var_14, var_1, var_13)
    var_16 = -2
    var_17 = result.split(var_1)[var_16]
    var_18 = 'from module import function1, function2, function3'
    var_19 = 'from module import function1 as f1, function2 as f2'
    var_20 = 'from module.submodule import function1, function2'
    var_21 = 'cimport module.function1, module.function2'
    var_22 = module_0.Config()
    var_23 = 'from module import function1, function2, function3'
    var_24 = module_1.line(var_23, var_1, var_22)



# Parsed testcases at query #63
#--------------------------


import isort.wrap as module_0

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
    var_17 = 50
    var_18 = [var_1, var_2, var_3]
    var_19 = [var_1, var_2, var_3]
    var_20 = [var_1, var_2, var_3]



# Parsed testcases at query #64
#--------------------------


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import function'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = 'from module import function1, function2, function3, function4'
    var_4 = 30
    var_5 = 0
    var_6 = result.split(var_1)[var_5]
    var_7 = len(var_6)
    var_8 = 'from module import function  # comment'
    var_9 = 'from module import function  # NOQA'
    var_10 = 'from module import function1, function2, function3'
    var_11 = True
    var_12 = 'from module import function1, function2, function3'
    var_13 = ','
    var_14 = 'from module import function as alias'
    var_15 = 'from module.submodule import function'
    var_16 = 'from module import function1, function2, function3'



# Parsed testcases at query #65
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
    var_18 = 50
    var_19 = 40
    var_20 = module_1.Config()
    var_21 = [var_1, var_2, var_3]
    var_22 = module_0.import_statement(var_0, var_21, config=var_20)
    var_23 = 0
    var_24 = result.split(var_17)[var_23]
    var_25 = len(var_24)
    var_26 = [var_1, var_2, var_3]
    var_27 = []
    var_28 = module_0.import_statement(var_0, var_27)
    assert var_28 == 'from module import'
    var_29 = [var_1]
    var_30 = module_0.import_statement(var_0, var_29)
    assert var_30 == 'from module import item1'
    var_31 = 20
    var_32 = module_1.Config()
    var_33 = [var_1, var_2, var_3]
    var_34 = module_0.import_statement(var_0, var_33, config=var_32)
    var_35 = module_2.split(var_17)
    var_36 = -1
    var_37 = var_35[:var_36]
    var_38 = min(var_2)
    var_39 = -1
    var_40 = var_35[var_39]
    var_41 = len(var_40)



# Parsed testcases at query #66
#--------------------------


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'func1'
    var_2 = 'func2'
    var_3 = 'func3'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
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
    var_17 = 50
    var_18 = [var_1, var_2, var_3]
    var_19 = [var_1, var_2, var_3]
    var_20 = [var_1, var_2, var_3]
    var_21 = [var_1, var_2, var_3]
    var_22 = [var_7, var_8]
    var_23 = [var_1, var_2, var_3]



# Parsed testcases at query #67
#--------------------------


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import something, another_thing, third_thing'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = 'from module import something  # some comment'
    var_4 = module_0.line(var_3, var_1)
    var_5 = 'from very_long_module_name import very_long_import_name, another_very_long_import_name'
    var_6 = module_0.line(var_5, var_1)
    var_7 = 'from module import something, another_thing, third_thing  # NOQA'
    var_8 = module_0.line(var_7, var_1)
    var_9 = 'from module import something as alias, another_thing as another_alias'
    var_10 = module_0.line(var_9, var_1)
    var_11 = 'from module.submodule import something, another_thing'
    var_12 = module_0.line(var_11, var_1)
    var_13 = 'cimport module.something, module.another_thing'
    var_14 = module_0.line(var_13, var_1)
    var_15 = 'from module import something'
    var_16 = module_0.line(var_15, var_1)
    var_17 = ''
    var_18 = module_0.line(var_17, var_1)
    assert var_18 == ''
    var_19 = '   '
    var_20 = module_0.line(var_19, var_1)



# Parsed testcases at query #68
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    var_6 = [var_1, var_2, var_3]
    var_7 = '# comment'
    var_8 = [var_7]
    var_9 = module_0.import_statement(var_0, var_6, var_8)
    var_10 = [var_1, var_2, var_3]
    var_11 = '\r\n'
    var_12 = module_0.import_statement(var_0, var_10, line_separator=var_11)
    var_13 = [var_1, var_2, var_3]
    var_14 = True
    var_15 = module_0.import_statement(var_0, var_13, explode=var_14)
    var_16 = 20
    var_17 = module_1.Config()
    var_18 = [var_1, var_2, var_3]
    var_19 = module_0.import_statement(var_0, var_18, config=var_17)
    var_20 = 0
    var_21 = '\n'
    var_22 = result.split(var_21)[var_20]
    var_23 = len(var_22)
    var_24 = [var_1, var_2, var_3]
    var_25 = module_1.Config()
    var_26 = [var_1, var_2, var_3]
    var_27 = module_0.import_statement(var_0, var_26, config=var_25)
    var_28 = ','
    var_29 = module_1.Config()
    var_30 = [var_1, var_2, var_3]
    var_31 = [var_7]
    var_32 = module_0.import_statement(var_0, var_30, var_31, config=var_29)
    var_33 = module_1.Config()
    var_34 = [var_1, var_2, var_3]
    var_35 = module_0.import_statement(var_0, var_34, config=var_33)
    var_36 = '# '
    var_37 = module_1.Config()
    var_38 = [var_1, var_2, var_3]
    var_39 = 'comment'
    var_40 = [var_39]
    var_41 = module_0.import_statement(var_0, var_38, var_40, config=var_37)



# Parsed testcases at query #69
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'from module import function'
    var_2 = '\n'
    var_3 = module_1.line(var_1, var_2, var_0)
    assert var_3 == 'from module import function'
    var_4 = 'from module import very_long_function_name'
    var_5 = module_1.line(var_4, var_2, var_0)
    assert var_5 == 'from module import (\n    very_long_function_name\n)'
    var_6 = 'from module import function  # some comment'
    var_7 = module_1.line(var_6, var_2, var_0)
    assert var_7 == 'from module import (\n    function,  # some comment\n)'
    var_8 = 'from module import function  # NOQA'
    var_9 = module_1.line(var_8, var_2, var_0)
    assert var_9 == 'from module import function  # NOQA'
    var_10 = 'from module import function as f'
    var_11 = module_1.line(var_10, var_2, var_0)
    assert var_11 == 'from module import (\n    function as f\n)'
    var_12 = 'from module import function.subfunction'
    var_13 = module_1.line(var_12, var_2, var_0)
    assert var_13 == 'from module import (\n    function.subfunction\n)'
    var_14 = 'cimport module.function'
    var_15 = module_1.line(var_14, var_2, var_0)
    assert var_15 == 'cimport (\n    module.function\n)'
    var_16 = 'from module import function  # noqa'
    var_17 = module_1.line(var_16, var_2, var_0)
    assert var_17 == 'from module import (\n    function,  # noqa\n)'
    var_18 = 'from module import function1, function2'
    var_19 = module_1.line(var_18, var_2, var_0)
    assert var_19 == 'from module import (\n    function1,\n    function2\n)'
    var_20 = module_1.line(var_6, var_2, var_0)
    assert var_20 == 'from module import (\n    function,\n)'



# Parsed testcases at query #70
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = 'from module import function1, function2, function3'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    var_5 = 'from module import function1, function2, function3, function4, function5'
    var_6 = module_1.line(var_5, var_3, var_1)
    var_7 = 'from module import function1, function2, function3  # some comment'
    var_8 = module_1.line(var_7, var_3, var_1)
    var_9 = 'from module import function1, function2, function3  # NOQA'
    var_10 = module_1.line(var_9, var_3, var_1)
    var_11 = 'import module as alias'
    var_12 = module_1.line(var_11, var_3, var_1)
    var_13 = 'import module_with_very_long_name as alias'
    var_14 = module_1.line(var_13, var_3, var_1)
    var_15 = 'from module import function1, function2, function3'
    var_16 = module_1.line(var_15, var_3, var_1)
    var_17 = 'from module import function1, function2, function3'
    var_18 = module_1.line(var_17, var_3, var_1)
    var_19 = ','
    var_20 = 'from module import function1, function2, function3'
    var_21 = module_1.line(var_20, var_3, var_1)
    var_22 = 'from module import function1, function2, function3'
    var_23 = module_1.line(var_22, var_3, var_1)
    var_24 = 'from module import function1, function2, function3'
    var_25 = module_1.line(var_24, var_3, var_1)



# Parsed testcases at query #71
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
    var_14 = [var_1, var_2, var_3]
    var_15 = True
    var_16 = module_0.import_statement(var_0, var_14, explode=var_15)
    var_17 = '\n'
    var_18 = 50
    var_19 = [var_1, var_2, var_3]
    var_20 = [var_1, var_2, var_3]
    var_21 = [var_1]
    var_22 = module_0.import_statement(var_0, var_21)
    var_23 = []
    var_24 = module_0.import_statement(var_0, var_23)



# Parsed testcases at query #72
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
    var_14 = [var_1, var_2]
    var_15 = True
    var_16 = module_0.import_statement(var_0, var_14, explode=var_15)
    var_17 = 50
    var_18 = '# '
    var_19 = False
    var_20 = '    '
    var_21 = [var_1, var_2]
    var_22 = [var_1, var_2]
    var_23 = [var_1, var_2]
    var_24 = module_1.Config()
    var_25 = module_0.import_statement(var_0, var_23, config=var_24)
    var_26 = []
    var_27 = module_0.import_statement(var_0, var_26)
    var_28 = [var_1]
    var_29 = module_0.import_statement(var_0, var_28)



# Parsed testcases at query #73
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
    var_18 = 50
    var_19 = module_1.Config()
    var_20 = [var_1, var_2, var_3]
    var_21 = module_0.import_statement(var_0, var_20, config=var_19)
    var_22 = module_2.split(var_17)
    var_23 = -1
    var_24 = var_22[var_23]
    var_25 = len(var_24)
    var_26 = -1
    var_27 = var_22[:var_26]
    var_28 = len(var_22)
    var_29 = var_28 == var_15
    var_30 = [var_1, var_2, var_3]
    var_31 = module_1.Config()
    var_32 = [var_1, var_2]
    var_33 = module_0.import_statement(var_0, var_32, config=var_31)
    var_34 = ','
    var_35 = module_1.Config()
    var_36 = [var_1, var_2]
    var_37 = '# Comment'
    var_38 = [var_37]
    var_39 = module_0.import_statement(var_0, var_36, var_38, config=var_35)
    var_40 = module_1.Config()
    var_41 = [var_1, var_2]
    var_42 = module_0.import_statement(var_0, var_41, config=var_40)
    var_43 = '# '
    var_44 = module_1.Config()
    var_45 = [var_1, var_2]
    var_46 = 'Comment'
    var_47 = [var_46]
    var_48 = module_0.import_statement(var_0, var_45, var_47, config=var_44)
    var_49 = 40
    var_50 = module_1.Config()
    var_51 = [var_1, var_2, var_3]
    var_52 = module_0.import_statement(var_0, var_51, config=var_50)
    var_53 = 0
    var_54 = result.split(var_17)[var_53]
    var_55 = len(var_54)
    var_56 = '    '
    var_57 = module_1.Config()
    var_58 = [var_1, var_2]
    var_59 = module_0.import_statement(var_0, var_58, config=var_57)



# Parsed testcases at query #74
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
    var_18 = 50
    var_19 = 40
    var_20 = module_1.Config()
    var_21 = [var_1, var_2, var_3]
    var_22 = module_0.import_statement(var_0, var_21, config=var_20)
    var_23 = 0
    var_24 = result.split(var_17)[var_23]
    var_25 = len(var_24)
    var_26 = [var_1, var_2, var_3]
    var_27 = [var_1, var_2, var_3]
    var_28 = module_1.Config()
    var_29 = module_0.import_statement(var_0, var_27, config=var_28)
    var_30 = module_2.split(var_17)
    var_31 = -1
    var_32 = var_30[:var_31]
    var_33 = min(var_2)
    var_34 = -1
    var_35 = var_30[var_34]
    var_36 = len(var_35)
    var_37 = [var_32]
    var_38 = module_0.import_statement(var_31, var_37)



# Parsed testcases at query #75
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
    var_19 = len(var_18)
    var_20 = 50
    var_21 = 40
    var_22 = module_1.Config()
    var_23 = [var_1, var_2, var_3]
    var_24 = module_0.import_statement(var_0, var_23, config=var_22)
    var_25 = [var_1, var_2, var_3]
    var_26 = []
    var_27 = module_0.import_statement(var_0, var_26)
    var_28 = [var_1]
    var_29 = module_0.import_statement(var_0, var_28)



# Parsed testcases at query #76
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import function'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = 'from module import function  # comment'
    var_4 = module_0.line(var_3, var_1)
    var_5 = 'from module import very_long_function_name'
    var_6 = 20
    var_7 = module_1.Config()
    var_8 = module_0.line(var_5, var_1, var_7)
    assert var_8 == 'from module import (\n    very_long_function_name\n)'
    var_9 = 'from module import very_long_function_name  # comment'
    var_10 = True
    var_11 = module_1.Config()
    var_12 = module_0.line(var_9, var_1, var_11)
    assert var_12 == 'from module import (\n    very_long_function_name,  # comment\n)'
    var_13 = 'from module import very_long_function_name  # NOQA'
    var_14 = module_0.line(var_13, var_1, var_11)
    var_15 = 'from module import very_long_function_name'
    var_16 = module_0.line(var_15, var_1, var_11)
    assert var_16 == 'from module import very_long_function_name  # NOQA'
    var_17 = 'from module.cimport very_long_function_name'
    var_18 = module_1.Config()
    var_19 = module_0.line(var_17, var_1, var_18)
    assert var_19 == 'from module.cimport (\n    very_long_function_name\n)'
    var_20 = 'from module import very_long_function_name as alias'
    var_21 = 30
    var_22 = module_1.Config()
    var_23 = module_0.line(var_20, var_1, var_22)
    assert var_23 == 'from module import (\n    very_long_function_name as alias\n)'
    var_24 = 'from module import function1, function2'
    var_25 = module_1.Config()
    var_26 = module_0.line(var_24, var_1, var_25)
    assert var_26 == 'from module import (\n    function1,\n    function2,\n)'
    var_27 = 'from module import function1, function2'
    var_28 = module_0.line(var_27, var_1, var_25)
    assert var_28 == 'from module import (\n    function1,\n    function2,\n)'
    var_29 = 'from module import function1, function2'
    var_30 = module_0.line(var_29, var_1, var_25)
    assert var_30 == 'from module import (\n    function1,\n    function2,\n)'
    var_31 = 'from module import function1, function2  # noqa'
    var_32 = module_1.Config()
    var_33 = module_0.line(var_31, var_1, var_32)
    assert var_33 == 'from module import (\n    function1,\n    function2,  # noqa\n)'
    var_34 = 'from module import function1, function2  # noqa'
    var_35 = False
    var_36 = module_1.Config()
    var_37 = module_0.line(var_34, var_1, var_36)
    assert var_37 == 'from module import function1, function2  # noqa'
    var_38 = 'from module import function1, function2  # noqa'
    var_39 = module_1.Config()
    var_40 = module_0.line(var_38, var_1, var_39)
    assert var_40 == 'from module import (\n    function1,\n    function2,  # noqa\n)'
    var_41 = 'from module import function1, function2  # noqa'
    var_42 = module_1.Config()
    var_43 = module_0.line(var_41, var_1, var_42)
    assert var_43 == 'from module import (\n    function1,\n    function2  # noqa\n)'



# Parsed testcases at query #77
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
    var_19 = 40
    var_20 = '    '
    var_21 = module_1.Config()
    var_22 = [var_1, var_2, var_3]
    var_23 = module_0.import_statement(var_0, var_22, config=var_21)
    var_24 = ','
    var_25 = module_1.Config()
    var_26 = [var_1, var_2, var_3]
    var_27 = module_0.import_statement(var_0, var_26, config=var_25)
    var_28 = module_2.split(var_17)
    var_29 = -1
    var_30 = var_28[:var_29]
    var_31 = min(var_2)
    var_32 = -1
    var_33 = var_28[var_32]
    var_34 = len(var_33)
    var_35 = [var_30, var_2, var_32]



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import re as module_0

def test_case_0():
    var_0 = 'from module import function1, function2, function3'
    var_1 = 50
    var_2 = '\n'
    var_3 = module_0.split(var_2)
    var_4 = len(var_3)
    var_5 = 1
    var_6 = var_4 > var_5
    var_7 = 'from module import func1, func2  # some comment'
    var_8 = 'from module import very_long_function_name_that_exceeds_line_length'
    var_9 = 30
    var_10 = 40
    var_11 = True
    var_12 = 'from module import func1, func2, func3, func4'
    var_13 = True
    var_14 = True
    var_15 = ','
    var_16 = True
    var_17 = module_0.split(var_2)
    var_18 = -1
    var_19 = var_17[:var_18]
    var_20 = min(var_2)
    var_21 = -1
    var_22 = var_17[var_21]
    var_23 = len(var_22)
    var_24 = 'from module import func1, func2, func3'
    var_25 = '\r\n'
    var_26 = 'from module import func'
    var_27 = 'from module import function as alias'
    var_28 = 'cimport module.function1, module.function2'
    var_29 = True
    var_30 = 'from module import func1, func2  # comment'



# Parsed testcases at query #2
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1
import re as module_2

def test_case_0():
    var_0 = 'from module import function'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = 'from module import function1, function2, function3, function4, function5'
    var_4 = 50
    var_5 = module_1.Config()
    var_6 = module_0.line(var_3, var_1, var_5)
    var_7 = 0
    var_8 = wrapped.split(var_1)[var_7]
    var_9 = len(var_8)
    var_10 = 'from module import function  # comment'
    var_11 = module_0.line(var_10, var_1, var_5)
    var_12 = 'from module import function1, function2, function3, function4, function5'
    var_13 = True
    var_14 = module_1.Config()
    var_15 = module_0.line(var_3, var_1, var_14)
    var_16 = module_1.Config()
    var_17 = module_0.line(var_3, var_1, var_16)
    var_18 = ','
    var_19 = 'from module import function as alias'
    var_20 = module_0.line(var_19, var_1, var_5)
    var_21 = 'cimport module.function'
    var_22 = module_0.line(var_21, var_1, var_5)
    var_23 = 'from module.submodule import function'
    var_24 = module_0.line(var_23, var_1, var_5)
    var_25 = module_1.Config()
    var_26 = module_0.line(var_3, var_1, var_25)
    var_27 = module_2.split(var_1)
    var_28 = -1
    var_29 = var_27[:var_28]
    var_30 = min(var_4)
    var_31 = -1
    var_32 = var_27[var_31]
    var_33 = len(var_32)



# Parsed testcases at query #3
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import function1, function2, function3'
    var_1 = 50
    var_2 = module_0.Config()
    var_3 = '\n'
    var_4 = module_1.line(var_0, var_3, var_2)
    var_5 = 'from module import function1, function2  # comment'
    var_6 = True
    var_7 = module_0.Config()
    var_8 = module_1.line(var_5, var_3, var_7)
    var_9 = 'from module import function1, function2, function3'
    var_10 = module_1.line(var_9, var_3, var_7)
    var_11 = 'from module import function1 as f1, function2 as f2'
    var_12 = module_0.Config()
    var_13 = module_1.line(var_11, var_3, var_12)
    var_14 = 'from module.submodule import function1, function2'
    var_15 = module_0.Config()
    var_16 = module_1.line(var_14, var_3, var_15)
    var_17 = 'cimport module.function1, module.function2'
    var_18 = module_0.Config()
    var_19 = module_1.line(var_17, var_3, var_18)
    var_20 = 'from module import function1, function2, function3'
    var_21 = module_1.line(var_20, var_3, var_18)
    var_22 = 'from module import function1, function2, function3'
    var_23 = module_1.line(var_22, var_3, var_18)
    var_24 = 'from module import function1, function2  # comment'
    var_25 = module_0.Config()
    var_26 = module_1.line(var_24, var_3, var_25)
    var_27 = 'from module import function1, function2  # comment'
    var_28 = '// '
    var_29 = module_0.Config()
    var_30 = module_1.line(var_27, var_3, var_29)
    var_31 = 'from module import function1, function2, function3'
    var_32 = module_0.Config()
    var_33 = module_1.line(var_31, var_3, var_32)
    var_34 = ','
    var_35 = 'from module import function1, function2, function3'
    var_36 = module_0.Config()
    var_37 = module_1.line(var_35, var_3, var_36)
    var_38 = 'from module import function1, function2, function3'
    var_39 = module_0.Config()
    var_40 = '\r\n'
    var_41 = module_1.line(var_38, var_40, var_39)
    var_42 = 'from module import function1'
    var_43 = module_0.Config()
    var_44 = module_1.line(var_42, var_3, var_43)
    var_45 = 'from module import function1, function2  # noqa'
    var_46 = module_1.line(var_45, var_3, var_43)



# Parsed testcases at query #4
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
    var_6 = 'from module import func  # some comment'
    var_7 = module_1.line(var_6, var_3, var_1)
    var_8 = 'from module import func  # NOQA'
    var_9 = module_1.line(var_8, var_3, var_1)
    var_10 = 'from module import func  # some noqa comment'
    var_11 = module_1.line(var_10, var_3, var_1)
    var_12 = 'from module import very_long_function_name as alias'
    var_13 = module_1.line(var_12, var_3, var_1)
    var_14 = 'from module import very_long_function_name.submodule'
    var_15 = module_1.line(var_14, var_3, var_1)
    var_16 = 'cimport module.very_long_function_name'
    var_17 = module_1.line(var_16, var_3, var_1)
    var_18 = 'from module import func1, func2, func3'
    var_19 = module_1.line(var_18, var_3, var_1)
    var_20 = 'from module import func1, func2, func3'
    var_21 = module_1.line(var_20, var_3, var_1)
    var_22 = ','
    var_23 = 'from module import func1, func2, func3'
    var_24 = module_1.line(var_23, var_3, var_1)
    var_25 = module_2.split(var_3)
    var_26 = -1
    var_27 = var_25[var_26]
    var_28 = len(var_27)
    var_29 = -1
    var_30 = var_25[:var_29]
    var_31 = min(var_22)
    var_32 = 'from module import func1, func2, func3'
    var_33 = '\n'
    var_34 = module_1.line(var_32, var_33, var_1)
    var_35 = 'from module import func  # comment'
    var_36 = module_1.line(var_35, var_27, var_1)
    var_37 = 'from module import func  # comment'
    var_38 = module_1.line(var_37, var_27, var_1)
    var_39 = 'from module import func'
    var_40 = '\r\n'
    var_41 = module_1.line(var_39, var_40, var_1)
    var_42 = 'from module import func'
    var_43 = module_1.line(var_42, var_27, var_1)



# Parsed testcases at query #5
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import function1, function2, function3'
    var_1 = 50
    var_2 = module_0.Config()
    var_3 = '\n'
    var_4 = module_1.line(var_0, var_3, var_2)
    var_5 = 'from module import function1, function2  # some comment'
    var_6 = True
    var_7 = module_0.Config()
    var_8 = module_1.line(var_5, var_3, var_7)
    var_9 = 'from module import function1, function2, function3'
    var_10 = module_1.line(var_9, var_3, var_7)
    var_11 = 'from module import function1, function2, function3'
    var_12 = module_0.Config()
    var_13 = module_1.line(var_11, var_3, var_12)
    var_14 = 'from module import function1, function2, function3'
    var_15 = module_1.line(var_14, var_3, var_12)
    var_16 = 'from module import function1, function2, function3'
    var_17 = module_1.line(var_16, var_3, var_12)
    var_18 = 'from module import function1, function2, function3'
    var_19 = module_0.Config()
    var_20 = module_1.line(var_18, var_3, var_19)
    var_21 = ','
    var_22 = 'from module import function1, function2  # some comment'
    var_23 = module_0.Config()
    var_24 = module_1.line(var_22, var_3, var_23)
    var_25 = 'from module import function1, function2, function3'
    var_26 = module_0.Config()
    var_27 = '\r\n'
    var_28 = module_1.line(var_25, var_27, var_26)
    var_29 = 'from module import function1'
    var_30 = module_0.Config()
    var_31 = module_1.line(var_29, var_3, var_30)



# Parsed testcases at query #6
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
    var_18 = module_1.Config()
    var_19 = [var_1, var_2, var_3]
    var_20 = module_0.import_statement(var_0, var_19, config=var_18)
    var_21 = module_2.split(var_17)
    var_22 = -1
    var_23 = var_21[var_22]
    var_24 = len(var_23)
    var_25 = -1
    var_26 = var_21[:var_25]
    var_27 = min(var_6)
    var_28 = False
    var_29 = module_1.Config()
    var_30 = [var_23, var_24]
    var_31 = module_0.import_statement(var_22, var_30, config=var_29)
    var_32 = ','
    var_33 = module_1.Config()
    var_34 = [var_23, var_24]
    var_35 = '# Comment'
    var_36 = [var_35]
    var_37 = module_0.import_statement(var_22, var_34, var_36, config=var_33)
    var_38 = '    '
    var_39 = module_1.Config()
    var_40 = [var_23, var_24]
    var_41 = module_0.import_statement(var_22, var_40, config=var_39)
    var_42 = [var_23, var_24, var_25]
    var_43 = 20
    var_44 = module_1.Config()
    var_45 = [var_23, var_24, var_25]
    var_46 = module_0.import_statement(var_22, var_45, config=var_44)
    var_47 = module_2.split(var_17)



# Parsed testcases at query #7
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
    var_17 = '\n'
    var_18 = module_1.Config()
    var_19 = [var_1, var_2, var_3]
    var_20 = module_0.import_statement(var_0, var_19, config=var_18)
    var_21 = module_2.split(var_17)
    var_22 = -1
    var_23 = var_21[:var_22]
    var_24 = min(var_2)
    var_25 = -1
    var_26 = var_21[var_25]
    var_27 = len(var_26)
    var_28 = [var_23, var_2, var_25]
    var_29 = module_1.Config()
    var_30 = [var_23, var_2, var_25]
    var_31 = module_0.import_statement(var_22, var_30, config=var_29)
    var_32 = ','
    var_33 = module_1.Config()
    var_34 = [var_23, var_2, var_25]
    var_35 = [var_7, var_8]
    var_36 = module_0.import_statement(var_22, var_34, var_35, config=var_33)
    var_37 = '    '
    var_38 = module_1.Config()
    var_39 = [var_23, var_2, var_25]
    var_40 = module_0.import_statement(var_22, var_39, config=var_38)
    var_41 = '# '
    var_42 = module_1.Config()
    var_43 = [var_23, var_2, var_25]
    var_44 = 'Comment 1'
    var_45 = 'Comment 2'
    var_46 = [var_44, var_45]
    var_47 = module_0.import_statement(var_22, var_43, var_46, config=var_42)
    var_48 = 50
    var_49 = module_1.Config()
    var_50 = [var_23, var_2, var_25]
    var_51 = module_0.import_statement(var_22, var_50, config=var_49)
    var_52 = module_2.split(var_17)
    var_53 = 60
    var_54 = module_1.Config()
    var_55 = [var_23, var_2, var_25]
    var_56 = module_0.import_statement(var_22, var_55, config=var_54)
    var_57 = module_2.split(var_17)
    var_58 = module_1.Config()
    var_59 = [var_23, var_2, var_25]
    var_60 = module_0.import_statement(var_22, var_59, config=var_58)
    var_61 = [var_23, var_2, var_25]
    var_62 = '# NOQA'
    var_63 = [var_62]
    var_64 = module_0.import_statement(var_22, var_61, var_63)
    var_65 = []
    var_66 = module_0.import_statement(var_22, var_65)
    assert var_66 == 'from module import'
    var_67 = [var_23]
    var_68 = module_0.import_statement(var_22, var_67)
    assert var_68 == 'from module import func1'
    var_69 = 'very_long_function_name_1'
    var_70 = 'very_long_function_name_2'
    var_71 = [var_69, var_70]
    var_72 = module_0.import_statement(var_22, var_71)



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
    var_18 = module_1.Config()
    var_19 = [var_1, var_2, var_3]
    var_20 = module_0.import_statement(var_0, var_19, config=var_18)
    var_21 = module_2.split(var_17)
    var_22 = len(var_21)
    var_23 = -1
    var_24 = var_21[:var_23]
    var_25 = -1
    var_26 = var_21[var_25]
    var_27 = len(var_26)
    var_28 = [var_1, var_2, var_3]
    var_29 = module_1.Config()
    var_30 = [var_1, var_2]
    var_31 = module_0.import_statement(var_0, var_30, config=var_29)
    var_32 = ','
    var_33 = module_1.Config()
    var_34 = [var_1, var_2]
    var_35 = [var_7]
    var_36 = module_0.import_statement(var_0, var_34, var_35, config=var_33)



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'from module import function1, function2, function3'
    var_1 = 50
    var_2 = '\n'
    var_3 = 'from module import function1, function2  # some comment'
    var_4 = 'from module import function1, function2, function3  # NOQA'
    var_5 = 'from module import function1, function2, function3  # noqa'
    var_6 = 'from module import function1, function2, function3'
    var_7 = True
    var_8 = 'from module import function1, function2, function3'
    var_9 = ','
    var_10 = 'from module import function1, function2, function3'
    var_11 = 'from module cimport function1, function2, function3'
    var_12 = 'from module import function1, function2, function3 as f3'
    var_13 = 'from module import function1, function2, function3'
    var_14 = 'from module import function1, function2, function3'
    var_15 = '\r\n'
    var_16 = 'from module import function1, function2, function3  # some comment'



# Parsed testcases at query #10
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
    var_11 = [var_1, var_2, var_3]
    var_12 = True
    var_13 = module_0.import_statement(var_0, var_11, explode=var_12)
    var_14 = '\n'
    var_15 = 50
    var_16 = [var_1, var_2, var_3]
    var_17 = 0
    var_18 = result.split(var_14)[var_17]
    var_19 = len(var_18)
    var_20 = module_1.Config()
    var_21 = [var_1, var_2, var_3]
    var_22 = module_0.import_statement(var_0, var_21, config=var_20)
    var_23 = [var_1, var_2, var_3]
    var_24 = module_1.Config()
    var_25 = [var_1, var_2]
    var_26 = module_0.import_statement(var_0, var_25, config=var_24)
    var_27 = ','
    var_28 = module_1.Config()
    var_29 = [var_1, var_2]
    var_30 = '# Comment'
    var_31 = [var_30]
    var_32 = module_0.import_statement(var_0, var_29, var_31, config=var_28)



# Parsed testcases at query #11
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1
import re as module_2

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'func1'
    var_2 = 'func2'
    var_3 = 'func3'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    assert var_5 == 'from module import func1, func2, func3'
    var_6 = [var_1, var_2]
    var_7 = '# Comment'
    var_8 = [var_7]
    var_9 = module_0.import_statement(var_0, var_6, var_8)
    var_10 = [var_1, var_2]
    var_11 = '\r\n'
    var_12 = module_0.import_statement(var_0, var_10, line_separator=var_11)
    var_13 = [var_1, var_2, var_3]
    var_14 = True
    var_15 = module_0.import_statement(var_0, var_13, explode=var_14)
    var_16 = '\n'
    var_17 = 20
    var_18 = module_1.Config()
    var_19 = 'very_long_name1'
    var_20 = 'very_long_name2'
    var_21 = [var_19, var_20]
    var_22 = module_0.import_statement(var_0, var_21, config=var_18)
    var_23 = module_2.split(var_16)
    var_24 = len(var_23)
    var_25 = var_24 > var_14
    var_26 = 0
    var_27 = var_23[var_26]
    var_28 = len(var_27)
    var_29 = -1
    var_30 = var_23[var_29]
    var_31 = len(var_30)
    var_32 = var_28 >= var_31
    var_33 = [var_1, var_2]
    var_34 = module_1.Config()
    var_35 = [var_1, var_2]
    var_36 = module_0.import_statement(var_0, var_35, config=var_34)
    var_37 = ','
    var_38 = []
    var_39 = module_0.import_statement(var_0, var_38)
    assert var_39 == 'from module import '
    var_40 = [var_1]
    var_41 = module_0.import_statement(var_0, var_40)
    assert var_41 == 'from module import func1'



# Parsed testcases at query #12
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = 'from module import function1, function2, function3'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    var_5 = 'from module import function1, function2, function3, function4, function5'
    var_6 = module_1.line(var_5, var_3, var_1)
    var_7 = 'from module import function1, function2  # some comment'
    var_8 = module_1.line(var_7, var_3, var_1)
    var_9 = 'from module import function1, function2, function3  # NOQA'
    var_10 = module_1.line(var_9, var_3, var_1)
    var_11 = 'from module import function1 as f1, function2 as f2'
    var_12 = module_1.line(var_11, var_3, var_1)
    var_13 = 'from module.submodule import function1, function2'
    var_14 = module_1.line(var_13, var_3, var_1)
    var_15 = 'cimport module.function1, module.function2'
    var_16 = module_1.line(var_15, var_3, var_1)
    var_17 = True
    var_18 = module_0.Config()
    var_19 = 'from module import function1, function2, function3'
    var_20 = module_1.line(var_19, var_3, var_18)
    var_21 = module_0.Config()
    var_22 = 'from module import function1, function2, function3'
    var_23 = module_1.line(var_22, var_3, var_21)
    var_24 = -2
    var_25 = result.split(var_3)[var_24]
    var_26 = 'from module import function1, function2, function3'
    var_27 = module_1.line(var_26, var_3, var_21)
    var_28 = 'from module import function1, function2, function3'
    var_29 = module_1.line(var_28, var_3, var_21)
    var_30 = 'from module import function1, function2, function3'
    var_31 = module_1.line(var_30, var_3, var_21)
    var_32 = 'from module import function1'
    var_33 = module_1.line(var_32, var_3, var_21)
    var_34 = ''
    var_35 = module_1.line(var_34, var_3, var_21)
    var_36 = '# '
    var_37 = module_0.Config()
    var_38 = 'from module import function1, function2  # comment'
    var_39 = module_1.line(var_38, var_3, var_37)
    var_40 = module_0.Config()
    var_41 = 'from module import function1, function2  # comment'
    var_42 = module_1.line(var_41, var_3, var_40)



# Parsed testcases at query #13
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
    var_19 = module_1.Config()
    var_20 = [var_1, var_2, var_3]
    var_21 = module_0.import_statement(var_0, var_20, config=var_19)
    var_22 = 0
    var_23 = result.split(var_17)[var_22]
    var_24 = len(var_23)
    var_25 = [var_1, var_2, var_3]
    var_26 = []
    var_27 = module_0.import_statement(var_0, var_26)
    assert var_27 == 'from module import'
    var_28 = [var_1]
    var_29 = module_0.import_statement(var_0, var_28)
    assert var_29 == 'from module import func1'
    var_30 = 20
    var_31 = module_1.Config()
    var_32 = [var_1, var_2, var_3]
    var_33 = module_0.import_statement(var_0, var_32, config=var_31)
    var_34 = module_2.split(var_17)
    var_35 = -1
    var_36 = var_34[var_35]
    var_37 = len(var_36)
    var_38 = -1
    var_39 = var_34[:var_38]
    var_40 = len(var_34)
    var_41 = var_40 == var_15



# Parsed testcases at query #14
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
    var_6 = 'from module import (\n    very_long_function_name\n)'
    var_7 = module_0.line(var_3, var_1, var_5)
    var_8 = 'from module import function  # some comment'
    var_9 = True
    var_10 = module_1.Config()
    var_11 = 'from module import (\n    function,  # some comment\n)'
    var_12 = module_0.line(var_8, var_1, var_10)
    var_13 = 'from module import function  # NOQA'
    var_14 = module_0.line(var_13, var_1, var_10)
    var_15 = 'from module import function as alias'
    var_16 = module_1.Config()
    var_17 = 'from module import function as (\n    alias\n)'
    var_18 = module_0.line(var_15, var_1, var_16)
    var_19 = 'from module import function  # noqa: F401'
    var_20 = module_1.Config()
    var_21 = 'from module import (\n    function,  # noqa: F401\n)'
    var_22 = module_0.line(var_19, var_1, var_20)
    var_23 = 'from module import function1, function2, function3'
    var_24 = 'from module import (\n    function1,\n    function2,\n    function3,\n)'
    var_25 = module_0.line(var_23, var_1, var_20)
    var_26 = 'from module import function1, function2, function3'
    var_27 = 'from module import (\n    function1,\n    function2,\n    function3,\n)'
    var_28 = module_0.line(var_26, var_1, var_20)
    var_29 = 'import os'
    var_30 = module_0.line(var_29, var_1)
    var_31 = 'from module import function1, function2'
    var_32 = module_1.Config()
    var_33 = 'from module import (\n    function1,\n    function2,\n)'
    var_34 = module_0.line(var_31, var_1, var_32)



# Parsed testcases at query #15
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import something'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'from module import something'
    var_3 = 'from module import something_very_long_function_name'
    var_4 = 20
    var_5 = module_1.Config()
    var_6 = module_0.line(var_3, var_1, var_5)
    assert var_6 == 'from module import (\n    something_very_long_function_name\n)'
    var_7 = 'from module import something  # some comment'
    var_8 = True
    var_9 = module_1.Config()
    var_10 = module_0.line(var_7, var_1, var_9)
    assert var_10 == 'from module import (\n    something,  # some comment\n)'
    var_11 = 'from module import something_very_long_function_name  # NOQA'
    var_12 = module_0.line(var_11, var_1, var_9)
    var_13 = 'from module import something as alias'
    var_14 = module_1.Config()
    var_15 = module_0.line(var_13, var_1, var_14)
    assert var_15 == 'from module import (\n    something as alias\n)'
    var_16 = 'cimport module.something_very_long_function_name'
    var_17 = module_1.Config()
    var_18 = module_0.line(var_16, var_1, var_17)
    assert var_18 == 'cimport (\n    module.something_very_long_function_name\n)'
    var_19 = 'from module import something.very.long.function.name'
    var_20 = module_1.Config()
    var_21 = module_0.line(var_19, var_1, var_20)
    assert var_21 == 'from module import (\n    something.very.long.function.name\n)'
    var_22 = 'from module import something  # noqa'
    var_23 = module_1.Config()
    var_24 = module_0.line(var_22, var_1, var_23)
    assert var_24 == 'from module import (\n    something  # noqa\n)'
    var_25 = 'from module import something, another_thing, third_thing'
    var_26 = module_1.Config()
    var_27 = module_0.line(var_25, var_1, var_26)
    assert var_27 == 'from module import (\n    something,\n    another_thing,\n    third_thing\n)'
    var_28 = 'from module import something  # some comment'
    var_29 = module_1.Config()
    var_30 = module_0.line(var_28, var_1, var_29)
    assert var_30 == 'from module import (\n    something\n)'



# Parsed testcases at query #16
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
    var_16 = module_1.Config()
    var_17 = [var_1, var_2]
    var_18 = module_0.import_statement(var_0, var_17, config=var_16)
    var_19 = 'func3'
    var_20 = 'func4'
    var_21 = 'func5'
    var_22 = [var_1, var_2, var_19, var_20, var_21]
    var_23 = module_0.import_statement(var_0, var_22)
    var_24 = '    '
    var_25 = module_1.Config()
    var_26 = [var_1, var_2]
    var_27 = module_0.import_statement(var_0, var_26, config=var_25)
    var_28 = module_1.Config()
    var_29 = [var_1, var_2]
    var_30 = module_0.import_statement(var_0, var_29, config=var_28)
    var_31 = module_1.Config()
    var_32 = [var_1, var_2]
    var_33 = [var_6]
    var_34 = module_0.import_statement(var_0, var_32, var_33, config=var_31)
    var_35 = '# '
    var_36 = module_1.Config()
    var_37 = [var_1, var_2]
    var_38 = 'Comment'
    var_39 = [var_38]
    var_40 = module_0.import_statement(var_0, var_37, var_39, config=var_36)



# Parsed testcases at query #17
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1
import re as module_2

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
    var_10 = True
    var_11 = module_0.import_statement(var_0, var_9, explode=var_10)
    assert var_11 == 'from module import (\n func1,\n func2,\n)'
    var_12 = [var_1, var_2]
    var_13 = '\r\n'
    var_14 = module_0.import_statement(var_0, var_12, line_separator=var_13)
    var_15 = 20
    var_16 = module_1.Config()
    var_17 = 'func3'
    var_18 = [var_1, var_2, var_17]
    var_19 = module_0.import_statement(var_0, var_18, config=var_16)
    var_20 = '\n'
    var_21 = module_2.split(var_20)
    var_22 = -1
    var_23 = var_21[var_22]
    var_24 = len(var_23)
    var_25 = -1
    var_26 = var_21[:var_25]
    var_27 = len(var_21)
    var_28 = var_27 == var_10
    var_29 = 30
    var_30 = module_1.Config()
    var_31 = [var_1, var_2]
    var_32 = module_0.import_statement(var_0, var_31, config=var_30)
    var_33 = ','
    var_34 = 'from module import'
    var_35 = 'func1'
    var_36 = 'func2'
    var_37 = [var_35, var_36]



# Parsed testcases at query #18
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
    var_18 = 20
    var_19 = module_1.Config()
    var_20 = [var_1, var_2, var_3]
    var_21 = module_0.import_statement(var_0, var_20, config=var_19)
    var_22 = module_2.split(var_17)
    var_23 = -1
    var_24 = var_22[var_23]
    var_25 = len(var_24)
    var_26 = -1
    var_27 = var_22[:var_26]
    var_28 = len(var_22)
    var_29 = var_28 == var_15
    var_30 = [var_1, var_2, var_3]
    var_31 = []
    var_32 = module_0.import_statement(var_0, var_31)
    assert var_32 == 'from module import'
    var_33 = [var_1]
    var_34 = module_0.import_statement(var_0, var_33)
    assert var_34 == 'from module import func1'
    var_35 = module_1.Config()
    var_36 = [var_1, var_2]
    var_37 = module_0.import_statement(var_0, var_36, config=var_35)
    var_38 = ','
    var_39 = module_1.Config()
    var_40 = [var_1, var_2]
    var_41 = '# Comment'
    var_42 = [var_41]
    var_43 = module_0.import_statement(var_0, var_40, var_42, config=var_39)



# Parsed testcases at query #19
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import function1, function2, function3'
    var_1 = 50
    var_2 = module_0.Config()
    var_3 = '\n'
    var_4 = module_1.line(var_0, var_3, var_2)
    var_5 = 'from module import function1, function2, function3  # some comment'
    var_6 = module_0.Config()
    var_7 = module_1.line(var_5, var_3, var_6)
    var_8 = 'from module import function1, function2, function3'
    var_9 = module_1.line(var_8, var_3, var_6)
    var_10 = 'from module import function1, function2, function3'
    var_11 = True
    var_12 = module_0.Config()
    var_13 = module_1.line(var_10, var_3, var_12)
    var_14 = 'from module import function1, function2, function3'
    var_15 = module_0.Config()
    var_16 = module_1.line(var_14, var_3, var_15)
    var_17 = 'from module import function1, function2, function3'
    var_18 = module_0.Config()
    var_19 = module_1.line(var_17, var_3, var_18)
    var_20 = ','
    var_21 = 'from module import function1, function2, function3  # some comment'
    var_22 = module_0.Config()
    var_23 = module_1.line(var_21, var_3, var_22)
    var_24 = 'from module import function1, function2, function3'
    var_25 = 50
    var_26 = '\n'
    var_27 = module_1.line(var_24, var_26, var_22)
    var_28 = 'from module import function1, function2, function3'
    var_29 = module_0.Config()
    var_30 = '\r\n'
    var_31 = module_1.line(var_28, var_30, var_29)
    var_32 = 'from module import function1, function2, function3'
    var_33 = '    '
    var_34 = module_0.Config()
    var_35 = module_1.line(var_32, var_26, var_34)



# Parsed testcases at query #20
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    var_6 = [var_1, var_2, var_3]
    var_7 = '# comment'
    var_8 = [var_7]
    var_9 = module_0.import_statement(var_0, var_6, var_8)
    var_10 = [var_1, var_2, var_3]
    var_11 = True
    var_12 = module_0.import_statement(var_0, var_10, explode=var_11)
    var_13 = '\n'
    var_14 = [var_1, var_2, var_3]
    var_15 = '\r\n'
    var_16 = module_0.import_statement(var_0, var_14, line_separator=var_15)
    var_17 = 20
    var_18 = module_1.Config()
    var_19 = [var_1, var_2, var_3]
    var_20 = module_0.import_statement(var_0, var_19, config=var_18)
    var_21 = [var_1, var_2, var_3]
    var_22 = 30
    var_23 = module_1.Config()
    var_24 = [var_1, var_2, var_3]
    var_25 = module_0.import_statement(var_0, var_24, config=var_23)
    var_26 = -1
    var_27 = result.split(var_13)[var_26]
    var_28 = len(var_27)
    var_29 = 0
    var_30 = result.split(var_13)[var_29]
    var_31 = len(var_30)
    var_32 = []
    var_33 = module_0.import_statement(var_0, var_32)
    assert var_33 == 'from module import'
    var_34 = [var_1]
    var_35 = module_0.import_statement(var_0, var_34)
    assert var_35 == 'from module import a'
    var_36 = 'very_long_name_1'
    var_37 = 'very_long_name_2'
    var_38 = 'very_long_name_3'
    var_39 = [var_36, var_37, var_38]
    var_40 = module_1.Config()
    var_41 = module_0.import_statement(var_0, var_39, config=var_40)
    var_42 = module_1.Config()
    var_43 = [var_1, var_2, var_3]
    var_44 = [var_7]
    var_45 = module_0.import_statement(var_0, var_43, var_44, config=var_42)



# Parsed testcases at query #21
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import function1, function2, function3'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = 'from module import function1, function2  # some comment'
    var_4 = module_0.line(var_3, var_1)
    var_5 = 'from module import function1, function2, function3  # NOQA'
    var_6 = module_0.line(var_5, var_1)
    var_7 = 'from module import function1, function2, function3, function4, function5'
    var_8 = module_0.line(var_7, var_1)
    var_9 = 50
    var_10 = True
    var_11 = module_1.Config()
    var_12 = module_0.line(var_7, var_1, var_11)
    var_13 = 'from module import function1 as f1, function2 as f2'
    var_14 = module_0.line(var_13, var_1)
    var_15 = 'from module.submodule import function1, function2'
    var_16 = module_0.line(var_15, var_1)
    var_17 = 'cimport module.function1, module.function2'
    var_18 = module_0.line(var_17, var_1)
    var_19 = module_1.Config()
    var_20 = module_0.line(var_7, var_1, var_19)
    var_21 = module_1.Config()
    var_22 = 'from module import function1, function2  # some comment'
    var_23 = module_0.line(var_22, var_1, var_21)
    var_24 = 'from module import function1'
    var_25 = module_0.line(var_24, var_1)
    var_26 = ''
    var_27 = module_0.line(var_26, var_1)
    var_28 = '# some comment'
    var_29 = module_0.line(var_28, var_1)



# Parsed testcases at query #22
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
    var_14 = [var_1, var_2, var_3]
    var_15 = True
    var_16 = module_0.import_statement(var_0, var_14, explode=var_15)
    var_17 = '\n'
    var_18 = 50
    var_19 = '    '
    var_20 = '# '
    var_21 = False
    var_22 = [var_1, var_2, var_3]
    var_23 = [var_1, var_2, var_3]
    var_24 = 20
    var_25 = module_1.Config()
    var_26 = [var_1, var_2, var_3]
    var_27 = module_0.import_statement(var_0, var_26, config=var_25)
    var_28 = module_2.split(var_17)
    var_29 = -1
    var_30 = var_28[:var_29]
    var_31 = min(var_2)
    var_32 = -1
    var_33 = var_28[var_32]
    var_34 = len(var_33)
    var_35 = [var_30]
    var_36 = module_0.import_statement(var_29, var_35)
    assert var_36 == 'from module import item1'



# Parsed testcases at query #23
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1
import re as module_2

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
    var_19 = module_2.split(var_15)
    var_20 = -1
    var_21 = var_19[var_20]
    var_22 = len(var_21)
    var_23 = -1
    var_24 = var_19[:var_23]
    var_25 = min(var_6)
    var_26 = [var_21, var_22]
    var_27 = module_1.Config()
    var_28 = [var_21, var_22]
    var_29 = module_0.import_statement(var_20, var_28, config=var_27)
    var_30 = ','
    var_31 = module_1.Config()
    var_32 = [var_21, var_22]
    var_33 = [var_6]
    var_34 = module_0.import_statement(var_20, var_32, var_33, config=var_31)
    var_35 = '    '
    var_36 = module_1.Config()
    var_37 = [var_21, var_22]
    var_38 = module_0.import_statement(var_20, var_37, config=var_36)
    var_39 = '# '
    var_40 = module_1.Config()
    var_41 = [var_21, var_22]
    var_42 = 'Comment'
    var_43 = [var_42]
    var_44 = module_0.import_statement(var_20, var_41, var_43, config=var_40)
    var_45 = 40
    var_46 = module_1.Config()
    var_47 = [var_21, var_22]
    var_48 = module_0.import_statement(var_20, var_47, config=var_46)
    var_49 = 0
    var_50 = result.split(var_15)[var_49]
    var_51 = len(var_50)
    var_52 = module_1.Config()
    var_53 = [var_21, var_22]
    var_54 = module_0.import_statement(var_20, var_53, config=var_52)



# Parsed testcases at query #24
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
    var_7 = 'from module import function  # some comment'
    var_8 = module_0.line(var_7, var_1)
    var_9 = 'from module import very_long_function_name  # some comment'
    var_10 = module_1.Config()
    var_11 = module_0.line(var_9, var_1, var_10)
    assert var_11 == 'from module import (\n    very_long_function_name,  # some comment\n)'
    var_12 = 'from module import function  # NOQA'
    var_13 = module_1.Config()
    var_14 = module_0.line(var_12, var_1, var_13)
    var_15 = 'from module import very_long_function_name  # NOQA'
    var_16 = module_1.Config()
    var_17 = module_0.line(var_15, var_1, var_16)
    var_18 = 'import module as alias'
    var_19 = module_1.Config()
    var_20 = module_0.line(var_18, var_1, var_19)
    assert var_20 == 'import module as (\n    alias\n)'
    var_21 = 'from module import function.subfunction'
    var_22 = module_1.Config()
    var_23 = module_0.line(var_21, var_1, var_22)
    assert var_23 == 'from module import (\n    function.subfunction\n)'
    var_24 = 'cimport module.function'
    var_25 = module_1.Config()
    var_26 = module_0.line(var_24, var_1, var_25)
    assert var_26 == 'cimport (\n    module.function\n)'
    var_27 = 'from module import function'
    var_28 = module_0.line(var_27, var_1, var_25)
    var_29 = 'from module import very_long_function_name'
    var_30 = module_0.line(var_29, var_1, var_25)
    assert var_30 == 'from module import very_long_function_name  # NOQA'



# Parsed testcases at query #25
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    var_6 = [var_1, var_2, var_3]
    var_7 = '# comment'
    var_8 = [var_7]
    var_9 = module_0.import_statement(var_0, var_6, var_8)
    var_10 = [var_1, var_2, var_3]
    var_11 = True
    var_12 = module_0.import_statement(var_0, var_10, explode=var_11)
    var_13 = [var_1, var_2, var_3]
    var_14 = '\r\n'
    var_15 = module_0.import_statement(var_0, var_13, line_separator=var_14)
    var_16 = 20
    var_17 = module_1.Config()
    var_18 = [var_1, var_2, var_3]
    var_19 = module_0.import_statement(var_0, var_18, config=var_17)
    var_20 = 0
    var_21 = '\n'
    var_22 = result.split(var_21)[var_20]
    var_23 = len(var_22)
    var_24 = [var_1, var_2, var_3]
    var_25 = module_1.Config()
    var_26 = [var_1, var_2, var_3]
    var_27 = module_0.import_statement(var_0, var_26, config=var_25)
    var_28 = ','
    var_29 = module_1.Config()
    var_30 = [var_1, var_2, var_3]
    var_31 = [var_7]
    var_32 = module_0.import_statement(var_0, var_30, var_31, config=var_29)



# Parsed testcases at query #26
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1
import re as module_2

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'A'
    var_2 = 'B'
    var_3 = 'C'
    var_4 = [var_1, var_2, var_3]
    var_5 = 80
    var_6 = module_0.Config()
    var_7 = module_1.import_statement(var_0, var_4, config=var_6)
    assert var_7 == 'from module import A, B, C'
    var_8 = 'D'
    var_9 = 'E'
    var_10 = [var_1, var_2, var_3, var_8, var_9]
    var_11 = 20
    var_12 = module_0.Config()
    var_13 = module_1.import_statement(var_0, var_10, config=var_12)
    var_14 = '\n'
    var_15 = module_2.split(var_14)
    var_16 = len(var_15)
    var_17 = [var_1, var_2, var_3]
    var_18 = '# Comment 1'
    var_19 = '# Comment 2'
    var_20 = [var_18, var_19]
    var_21 = module_0.Config()
    var_22 = module_1.import_statement(var_0, var_17, var_20, config=var_21)
    var_23 = [var_1, var_2, var_3]
    var_24 = True
    var_25 = module_1.import_statement(var_0, var_23, explode=var_24)
    var_26 = module_2.split(var_14)
    var_27 = len(var_26)
    assert var_27 == 3
    var_28 = [var_1, var_2, var_3]
    var_29 = '\r\n'
    var_30 = module_0.Config()
    var_31 = module_1.import_statement(var_0, var_28, line_separator=var_29, config=var_30)
    var_32 = module_0.Config()
    var_33 = [var_1, var_2, var_3, var_8]
    var_34 = module_1.import_statement(var_0, var_33, config=var_32)
    var_35 = module_2.split(var_14)
    var_36 = len(var_35)
    var_37 = -1
    var_38 = var_35[var_37]
    var_39 = len(var_38)
    var_40 = -1
    var_41 = var_35[:var_40]
    var_42 = [len(line) for line in var_41]
    var_43 = module_0.Config()
    var_44 = [var_1, var_2, var_3]
    var_45 = module_1.import_statement(var_0, var_44, config=var_43)
    var_46 = ','
    var_47 = '    '
    var_48 = module_0.Config()
    var_49 = [var_1, var_2, var_3]
    var_50 = module_1.import_statement(var_0, var_49, config=var_48)
    var_51 = module_2.split(var_14)
    var_52 = var_51[var_24:]
    var_53 = module_0.Config()
    var_54 = [var_1, var_2, var_3]
    var_55 = '# Comment'
    var_56 = [var_55]
    var_57 = module_1.import_statement(var_0, var_54, var_56, config=var_53)
    var_58 = '// '
    var_59 = module_0.Config()
    var_60 = [var_1, var_2, var_3]
    var_61 = 'Comment'
    var_62 = [var_61]
    var_63 = module_1.import_statement(var_0, var_60, var_62, config=var_59)



# Parsed testcases at query #27
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
    var_23 = -1
    var_24 = var_22[var_23]
    var_25 = len(var_24)
    var_26 = -1
    var_27 = var_22[:var_26]
    var_28 = min(var_6)
    var_29 = [var_24, var_25, var_26]
    var_30 = module_1.Config()
    var_31 = [var_24, var_25]
    var_32 = module_0.import_statement(var_23, var_31, config=var_30)
    var_33 = ','
    var_34 = module_1.Config()
    var_35 = [var_24, var_25]
    var_36 = '# Comment'
    var_37 = [var_36]
    var_38 = module_0.import_statement(var_23, var_35, var_37, config=var_34)



# Parsed testcases at query #28
#--------------------------


import isort.wrap as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'import os'
    var_3 = 'import os # comment'
    var_4 = module_0.line(var_3, var_1)
    assert var_4 == 'import os # comment'
    var_5 = 10
    var_6 = 'import os.path'
    var_7 = ' # '
    var_8 = 'import os.path # comment'
    var_9 = 'import os.path # noqa'
    var_10 = True
    var_11 = '    '
    var_12 = 20
    var_13 = '\r\n'



# Parsed testcases at query #29
#--------------------------


import isort.wrap as module_0
import re as module_1

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'A'
    var_2 = 'B'
    var_3 = 'C'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    assert var_5 == 'from module import A, B, C'
    var_6 = 20
    var_7 = True
    var_8 = 'D'
    var_9 = 'E'
    var_10 = [var_1, var_2, var_3, var_8, var_9]
    var_11 = [var_1, var_2, var_3]
    var_12 = module_0.import_statement(var_0, var_11, explode=var_7)
    assert var_12 == 'from module import (\n    A,\n    B,\n    C,\n)'
    var_13 = [var_1, var_2, var_3]
    var_14 = '# Comment 1'
    var_15 = '# Comment 2'
    var_16 = [var_14, var_15]
    var_17 = module_0.import_statement(var_0, var_13, var_16)
    var_18 = 30
    var_19 = [var_1, var_2, var_3, var_8]
    var_20 = '\n'
    var_21 = module_1.split(var_20)
    var_22 = len(var_21)
    var_23 = -1
    var_24 = var_21[var_23]
    var_25 = len(var_24)
    var_26 = -1
    var_27 = var_21[:var_26]
    var_28 = [var_1, var_2, var_3]
    var_29 = '\r\n'
    var_30 = module_0.import_statement(var_0, var_28, line_separator=var_29)
    var_31 = [var_1, var_2, var_3]



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 80
    var_1 = 'from module import function1, function2, function3'
    var_2 = '\n'
    var_3 = 'from module import function1, function2, function3  # some comment'
    var_4 = 'from module import function1, function2, function3  # NOQA'
    var_5 = 'from module import function1, function2, function3  # noqa'
    var_6 = 'from module import function1 as f1, function2 as f2'
    var_7 = 'from module.submodule import function1, function2'
    var_8 = 'cfrom module import function1, function2'
    var_9 = True
    var_10 = 'from module import function1, function2, function3'
    var_11 = 'from module import function1, function2, function3'
    var_12 = 'from module import function1, function2, function3  # some comment'
    var_13 = 80
    var_14 = 'from module import function1, function2, function3'
    var_15 = '\n'
    var_16 = 'from module import function1, function2, function3'
    var_17 = 'from module import function1, function2, function3'
    var_18 = '\r\n'
    var_19 = 'from module import function1, function2, function3'
    var_20 = ' # '
    var_21 = 'from module import function1, function2, function3  # some comment'
    var_22 = '    '
    var_23 = 'from module import function1, function2, function3'



# Parsed testcases at query #31
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
    var_19 = module_1.Config()
    var_20 = 'very_long_function_name_1'
    var_21 = 'very_long_function_name_2'
    var_22 = [var_20, var_21]
    var_23 = module_0.import_statement(var_0, var_22, config=var_19)
    var_24 = module_2.split(var_17)
    var_25 = -1
    var_26 = var_24[var_25]
    var_27 = len(var_26)
    var_28 = -1
    var_29 = var_24[:var_28]
    var_30 = len(var_24)
    var_31 = var_30 == var_15
    var_32 = [var_1, var_2, var_3]
    var_33 = module_1.Config()
    var_34 = [var_1, var_2]
    var_35 = module_0.import_statement(var_0, var_34, config=var_33)
    var_36 = ','
    var_37 = module_1.Config()
    var_38 = [var_1, var_2]
    var_39 = '# This should be ignored'
    var_40 = [var_39]
    var_41 = module_0.import_statement(var_0, var_38, var_40, config=var_37)
    var_42 = module_1.Config()
    var_43 = [var_1, var_2, var_3]
    var_44 = module_0.import_statement(var_0, var_43, config=var_42)



# Parsed testcases at query #32
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
    var_19 = [var_1, var_2, var_3]
    var_20 = 30
    var_21 = module_1.Config()
    var_22 = [var_1, var_2, var_3]
    var_23 = module_0.import_statement(var_0, var_22, config=var_21)
    var_24 = module_2.split(var_17)
    var_25 = len(var_24)
    var_26 = -1
    var_27 = var_24[var_26]
    var_28 = len(var_27)
    var_29 = -1
    var_30 = var_24[:var_29]
    var_31 = []
    var_32 = module_0.import_statement(var_0, var_31)
    assert var_32 == 'from module import'
    var_33 = [var_1]
    var_34 = module_0.import_statement(var_0, var_33)
    assert var_34 == 'from module import func1'
    var_35 = [var_1, var_2, var_3]



# Parsed testcases at query #33
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1
import re as module_2

def test_case_0():
    var_0 = 'from module import something, something_else, another_thing'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = 0
    var_4 = result.split(var_1)[var_3]
    var_5 = len(var_4)
    var_6 = 'from module import something  # some comment'
    var_7 = module_0.line(var_6, var_1)
    var_8 = 'from module import something_very_long_that_exceeds_line_length  # NOQA'
    var_9 = module_0.line(var_8, var_1)
    var_10 = 'from module import something as alias, another as another_alias'
    var_11 = module_0.line(var_10, var_1)
    var_12 = 'cimport module.something, module.something_else, module.another_thing'
    var_13 = module_0.line(var_12, var_1)
    var_14 = result.split(var_1)[var_3]
    var_15 = len(var_14)
    var_16 = 'from module import something.else, another.thing, third.thing'
    var_17 = module_0.line(var_16, var_1)
    var_18 = result.split(var_1)[var_3]
    var_19 = len(var_18)
    var_20 = True
    var_21 = module_1.Config()
    var_22 = 'from module import something, something_else, another_thing'
    var_23 = module_0.line(var_22, var_1, var_21)
    var_24 = module_1.Config()
    var_25 = 'from module import something, something_else, another_thing'
    var_26 = module_0.line(var_25, var_1, var_24)
    var_27 = ','
    var_28 = module_1.Config()
    var_29 = 'from module import something, something_else, another_thing'
    var_30 = module_0.line(var_29, var_1, var_28)
    var_31 = module_2.split(var_1)
    var_32 = -1
    var_33 = var_31[:var_32]
    var_34 = min(var_4)
    var_35 = -1
    var_36 = var_31[var_35]
    var_37 = len(var_36)
    var_38 = 'from module import something, something_else, another_thing'
    var_39 = 'from module import something, something_else, another_thing'
    var_40 = module_1.Config()
    var_41 = 'from module import something  # some comment'
    var_42 = module_0.line(var_41, var_32, var_40)



# Parsed testcases at query #34
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
    var_14 = [var_1, var_2]
    var_15 = True
    var_16 = module_0.import_statement(var_0, var_14, explode=var_15)
    var_17 = '\n'
    var_18 = [var_1, var_2]
    var_19 = module_1.Config()
    var_20 = [var_1, var_2]
    var_21 = module_0.import_statement(var_0, var_20, config=var_19)
    var_22 = 50
    var_23 = module_1.Config()
    var_24 = [var_1, var_2]
    var_25 = module_0.import_statement(var_0, var_24, config=var_23)



# Parsed testcases at query #35
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
    assert var_5 == 'from module import item1, item2, item3'
    var_6 = [var_1, var_2]
    var_7 = '# comment1'
    var_8 = '# comment2'
    var_9 = [var_7, var_8]
    var_10 = module_0.import_statement(var_0, var_6, var_9)
    var_11 = [var_1, var_2, var_3]
    var_12 = [var_1, var_2]
    var_13 = True
    var_14 = module_0.import_statement(var_0, var_12, explode=var_13)
    var_15 = '\n'
    var_16 = 50
    var_17 = module_1.Config()
    var_18 = [var_1, var_2, var_3]
    var_19 = module_0.import_statement(var_0, var_18, config=var_17)
    var_20 = 0
    var_21 = result.split(var_15)[var_20]
    var_22 = len(var_21)
    var_23 = [var_1, var_2, var_3]
    var_24 = module_1.Config()
    var_25 = module_0.import_statement(var_0, var_23, config=var_24)
    var_26 = module_2.split(var_15)
    var_27 = -1
    var_28 = var_26[var_27]
    var_29 = len(var_28)
    var_30 = -1
    var_31 = var_26[:var_30]
    var_32 = min(var_6)
    var_33 = [var_28, var_29]
    var_34 = '\r\n'
    var_35 = module_0.import_statement(var_27, var_33, line_separator=var_34)
    var_36 = []
    var_37 = module_0.import_statement(var_27, var_36)
    assert var_37 == 'from module import'
    var_38 = [var_28]
    var_39 = module_0.import_statement(var_27, var_38)
    assert var_39 == 'from module import item1'



# Parsed testcases at query #36
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
    var_28 = []
    var_29 = module_0.import_statement(var_0, var_28)
    assert var_29 == 'from module import'
    var_30 = [var_1]
    var_31 = module_0.import_statement(var_0, var_30)
    assert var_31 == 'from module import func1'
    var_32 = 20
    var_33 = module_1.Config()
    var_34 = [var_1, var_2, var_3]
    var_35 = module_0.import_statement(var_0, var_34, config=var_33)
    var_36 = module_2.split(var_17)
    var_37 = -1
    var_38 = var_36[:var_37]
    var_39 = min(var_2)
    var_40 = -1
    var_41 = var_36[var_40]
    var_42 = len(var_41)



# Parsed testcases at query #37
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
    var_20 = 'very_long_item_name_1'
    var_21 = 'very_long_item_name_2'
    var_22 = [var_20, var_21]
    var_23 = module_0.import_statement(var_0, var_22, config=var_19)
    var_24 = module_2.split(var_17)
    var_25 = -1
    var_26 = var_24[var_25]
    var_27 = len(var_26)
    var_28 = -1
    var_29 = var_24[:var_28]
    var_30 = len(var_24)
    var_31 = var_30 == var_15
    var_32 = [var_1, var_2, var_3]
    var_33 = module_1.Config()
    var_34 = [var_1, var_2]
    var_35 = module_0.import_statement(var_0, var_34, config=var_33)
    var_36 = ','
    var_37 = module_1.Config()
    var_38 = [var_1, var_2]
    var_39 = '# This should be ignored'
    var_40 = [var_39]
    var_41 = module_0.import_statement(var_0, var_38, var_40, config=var_37)
    var_42 = module_1.Config()
    var_43 = [var_1, var_2, var_3]
    var_44 = module_0.import_statement(var_0, var_43, config=var_42)
    var_45 = '    '
    var_46 = module_1.Config()
    var_47 = [var_1, var_2]
    var_48 = module_0.import_statement(var_0, var_47, config=var_46)
    var_49 = '# '
    var_50 = module_1.Config()
    var_51 = [var_1]
    var_52 = 'Comment without #'
    var_53 = [var_52]
    var_54 = module_0.import_statement(var_0, var_51, var_53, config=var_50)



# Parsed testcases at query #38
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'foo'
    var_2 = 'bar'
    var_3 = 'baz'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
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
    var_17 = module_1.Config()
    var_18 = [var_1, var_2, var_3]
    var_19 = module_0.import_statement(var_0, var_18, config=var_17)
    var_20 = '    '
    var_21 = module_1.Config()
    var_22 = [var_1, var_2, var_3]
    var_23 = module_0.import_statement(var_0, var_22, config=var_21)
    var_24 = False
    var_25 = module_1.Config()
    var_26 = [var_1, var_2, var_3]
    var_27 = module_0.import_statement(var_0, var_26, config=var_25)
    var_28 = module_1.Config()
    var_29 = [var_1, var_2, var_3]
    var_30 = [var_7, var_8]
    var_31 = module_0.import_statement(var_0, var_29, var_30, config=var_28)
    var_32 = [var_1, var_2, var_3]
    var_33 = [var_1, var_2, var_3]
    var_34 = [var_1, var_2, var_3]



# Parsed testcases at query #39
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
    var_6 = 20
    var_7 = module_1.Config()
    var_8 = [var_1, var_2, var_3]
    var_9 = module_0.import_statement(var_0, var_8, config=var_7)
    var_10 = '\n'
    var_11 = [var_1, var_2, var_3]
    var_12 = '# Comment'
    var_13 = [var_12]
    var_14 = module_0.import_statement(var_0, var_11, var_13)
    var_15 = [var_1, var_2, var_3]
    var_16 = True
    var_17 = module_0.import_statement(var_0, var_15, explode=var_16)
    var_18 = 30
    var_19 = module_1.Config()
    var_20 = [var_1, var_2, var_3]
    var_21 = module_0.import_statement(var_0, var_20, config=var_19)
    var_22 = module_2.split(var_10)
    var_23 = -1
    var_24 = var_22[var_23]
    var_25 = len(var_24)
    var_26 = -1
    var_27 = var_22[:var_26]
    var_28 = min(var_6)
    var_29 = [var_24, var_25, var_26]
    var_30 = '\r\n'
    var_31 = module_0.import_statement(var_23, var_29, line_separator=var_30)
    var_32 = module_1.Config()
    var_33 = [var_24, var_25, var_26]
    var_34 = module_0.import_statement(var_23, var_33, config=var_32)
    var_35 = ','
    var_36 = 'from module import'
    var_37 = 'A'
    var_38 = 'B'
    var_39 = 'C'
    var_40 = [var_37, var_38, var_39]
    var_41 = []
    var_42 = module_0.import_statement(var_36, var_41)
    var_43 = 'very_long_module_name_1'
    var_44 = 'very_long_module_name_2'
    var_45 = 'very_long_module_name_3'
    var_46 = [var_43, var_44, var_45]
    var_47 = module_0.import_statement(var_36, var_46)
    var_48 = '    '
    var_49 = module_1.Config()
    var_50 = [var_37, var_38, var_39]
    var_51 = module_0.import_statement(var_36, var_50, config=var_49)
    var_52 = module_1.Config()
    var_53 = [var_37, var_38, var_39]
    var_54 = [var_12]
    var_55 = module_0.import_statement(var_36, var_53, var_54, config=var_52)



# Parsed testcases at query #40
#--------------------------


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import ('
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    assert var_5 == 'from module import (\n    a,\n    b,\n    c,\n)'
    var_6 = 50
    var_7 = True
    var_8 = [var_1, var_2, var_3]
    var_9 = [var_1, var_2, var_3]
    var_10 = module_0.import_statement(var_0, var_9, explode=var_7)
    assert var_10 == 'from module import (\n    a,\n    b,\n    c,\n)'
    var_11 = [var_1, var_2, var_3]
    var_12 = '# comment 1'
    var_13 = '# comment 2'
    var_14 = [var_12, var_13]
    var_15 = module_0.import_statement(var_0, var_11, var_14)
    assert var_15 == 'from module import (\n    a,\n    b,\n    c,\n)  # comment 1\n# comment 2'
    var_16 = [var_1, var_2, var_3]
    var_17 = '\r\n'
    var_18 = module_0.import_statement(var_0, var_16, line_separator=var_17)
    assert var_18 == 'from module import (\r\n    a,\r\n    b,\r\n    c,\r\n)'
    assert var_18 == 'from module import (\n    a,\n    b,\n    c,\n)'
    var_19 = 20
    var_20 = [var_1, var_2, var_3]
    var_21 = 'from module import'
    var_22 = [var_1]
    var_23 = module_0.import_statement(var_21, var_22)
    assert var_23 == 'from module import a'
    assert var_23 == 'from module import a, b, c  # NOQA'
    var_24 = 10
    var_25 = [var_1, var_2, var_3]



# Parsed testcases at query #41
#--------------------------


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
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
    var_17 = 50
    var_18 = [var_1, var_2, var_3]
    var_19 = 20
    var_20 = [var_1, var_2]
    var_21 = 'short'
    var_22 = [var_21]
    var_23 = module_0.import_statement(var_0, var_22)
    var_24 = []
    var_25 = module_0.import_statement(var_0, var_24)
    var_26 = 'single_function'
    var_27 = [var_26]
    var_28 = module_0.import_statement(var_0, var_27)



# Parsed testcases at query #42
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import a, b, c'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = 0
    var_4 = result.split(var_1)[var_3]
    var_5 = len(var_4)
    var_6 = 'from module import a, b, c  # comment'
    var_7 = module_0.line(var_6, var_1)
    var_8 = 'from module import a, b, c  # NOQA'
    var_9 = module_0.line(var_8, var_1)
    var_10 = 'from module import a, b, c'
    var_11 = True
    var_12 = module_1.Config()
    var_13 = 'from module import a, b, c'
    var_14 = module_0.line(var_13, var_1, var_12)
    var_15 = module_1.Config()
    var_16 = 'from module import a, b, c'
    var_17 = module_0.line(var_16, var_1, var_15)
    var_18 = ','
    var_19 = 'from module import a, b, c'
    var_20 = module_0.line(var_19, var_1, var_15)
    var_21 = 'import a'
    var_22 = module_0.line(var_21, var_1)
    var_23 = 'from module import a as b'
    var_24 = module_0.line(var_23, var_1)
    var_25 = 'cimport module'
    var_26 = module_0.line(var_25, var_1)
    var_27 = 'from module.submodule import a'
    var_28 = module_0.line(var_27, var_1)



# Parsed testcases at query #43
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = 'from module import very_long_function_name, another_long_function_name, third_long_function_name'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    var_5 = 50
    var_6 = module_0.Config()
    var_7 = 'from module import very_long_function_name, another_long_function_name, third_long_function_name'
    var_8 = module_1.line(var_7, var_3, var_6)
    var_9 = True
    var_10 = module_0.Config()
    var_11 = 'from module import func1, func2  # some comment'
    var_12 = module_1.line(var_11, var_3, var_10)
    var_13 = 'from module import func1, func2'
    var_14 = module_1.line(var_13, var_3, var_10)
    var_15 = module_0.Config()
    var_16 = 'from module import func1, func2, func3'
    var_17 = module_1.line(var_16, var_3, var_15)
    var_18 = module_0.Config()
    var_19 = 'from module import func1, func2, func3'
    var_20 = module_1.line(var_19, var_3, var_18)
    var_21 = ','
    var_22 = module_0.Config()
    var_23 = 'import module as alias'
    var_24 = module_1.line(var_23, var_3, var_22)
    var_25 = module_0.Config()
    var_26 = 'cimport module.function'
    var_27 = module_1.line(var_26, var_3, var_25)
    var_28 = module_0.Config()
    var_29 = 'from module import function.subfunction'
    var_30 = module_1.line(var_29, var_3, var_28)
    var_31 = 'from module import func1, func2, func3'
    var_32 = module_1.line(var_31, var_3, var_28)



# Parsed testcases at query #44
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
    var_6 = '# comment'
    var_7 = [var_6]
    var_8 = module_0.import_statement(var_0, var_5, var_7)
    var_9 = [var_1, var_2]
    var_10 = '\r\n'
    var_11 = module_0.import_statement(var_0, var_9, line_separator=var_10)
    var_12 = [var_1, var_2]
    var_13 = True
    var_14 = module_0.import_statement(var_0, var_12, explode=var_13)
    var_15 = 20
    var_16 = module_1.Config()
    var_17 = [var_1, var_2]
    var_18 = module_0.import_statement(var_0, var_17, config=var_16)
    var_19 = 0
    var_20 = '\n'
    var_21 = result.split(var_20)[var_19]
    var_22 = len(var_21)
    var_23 = [var_1, var_2]
    var_24 = module_1.Config()
    var_25 = [var_1, var_2]
    var_26 = module_0.import_statement(var_0, var_25, config=var_24)
    var_27 = ','
    var_28 = module_1.Config()
    var_29 = [var_1, var_2]
    var_30 = [var_6]
    var_31 = module_0.import_statement(var_0, var_29, var_30, config=var_28)
    var_32 = '    '
    var_33 = module_1.Config()
    var_34 = [var_1, var_2]
    var_35 = module_0.import_statement(var_0, var_34, config=var_33)



# Parsed testcases at query #45
#--------------------------


import re as module_0

def test_case_0():
    var_0 = 'from module import function1, function2, function3'
    var_1 = 50
    var_2 = '\n'
    var_3 = 0
    var_4 = result.split(var_2)[var_3]
    var_5 = len(var_4)
    var_6 = 'from module import function1, function2, function3  # some comment'
    var_7 = 'from module import function1, function2, function3'
    var_8 = 'from module import function1 as f1, function2 as f2'
    var_9 = 'from module.submodule import function1, function2'
    var_10 = True
    var_11 = 'from module import function1, function2, function3'
    var_12 = 'from module import function1, function2, function3'
    var_13 = ','
    var_14 = 'from module import function1, function2, function3  # some comment'
    var_15 = ' # '
    var_16 = 'from module import function1, function2, function3  # some comment'
    var_17 = 'from module import function1, function2, function3'
    var_18 = '\r\n'
    var_19 = 'from module import function1'
    var_20 = 'from module import function1, function2, function3'
    var_21 = module_0.split(var_2)
    var_22 = -1
    var_23 = var_21[:var_22]
    var_24 = min(var_2)
    var_25 = -1
    var_26 = var_21[var_25]
    var_27 = len(var_26)



# Parsed testcases at query #46
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
    var_18 = 50
    var_19 = [var_1, var_2, var_3]
    var_20 = 30
    var_21 = module_1.Config()
    var_22 = 'very_long_item_name_1'
    var_23 = 'very_long_item_name_2'
    var_24 = [var_22, var_23]
    var_25 = module_0.import_statement(var_0, var_24, config=var_21)
    var_26 = module_2.split(var_17)
    var_27 = len(var_26)
    var_28 = -1
    var_29 = var_26[var_28]
    var_30 = len(var_29)
    var_31 = -1
    var_32 = var_26[:var_31]
    var_33 = [var_1, var_2, var_3]



# Parsed testcases at query #47
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import something'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'from module import something'
    var_3 = 20
    var_4 = None
    var_5 = 'from module import something_very_long, another_thing'
    var_6 = 'from module import (\n    something_very_long,\n    another_thing,\n)'
    var_7 = 'from module import something  # some comment'
    var_8 = 'from module import (\n    something,  # some comment\n)'
    var_9 = 'from module import something_very_long'
    var_10 = 'from module import something as alias'
    var_11 = 'from module import (\n    something\n) as alias'
    var_12 = 'cimport module.something_very_long'
    var_13 = 'cimport (\n    module.something_very_long,\n)'
    var_14 = 'from module import something.very.long.name'
    var_15 = 'from module import (\n    something.very.long.name,\n)'
    var_16 = True
    var_17 = module_1.Config()
    var_18 = 'from module import (something_very_long, another_thing)'
    var_19 = 'from module import (\n    something_very_long,\n    another_thing,\n)'
    var_20 = module_0.line(var_18, var_1, var_17)
    var_21 = module_1.Config()
    var_22 = 'from module import something, another, thing'
    var_23 = 'from module import (\n    something,\n    another,\n    thing,\n)'
    var_24 = module_0.line(var_22, var_1, var_21)
    var_25 = module_1.Config()
    var_26 = 'from module import something  # some comment'
    var_27 = 'from module import (\n    something,\n)'
    var_28 = module_0.line(var_26, var_1, var_25)



# Parsed testcases at query #48
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    assert var_5 == 'from module import a, b, c'
    var_6 = [var_1, var_2, var_3]
    var_7 = '# Comment'
    var_8 = [var_7]
    var_9 = module_0.import_statement(var_0, var_6, var_8)
    var_10 = [var_1, var_2, var_3]
    var_11 = '\r\n'
    var_12 = module_0.import_statement(var_0, var_10, line_separator=var_11)
    var_13 = [var_1, var_2, var_3]
    var_14 = True
    var_15 = module_0.import_statement(var_0, var_13, explode=var_14)
    var_16 = '\n'
    var_17 = [var_1, var_2, var_3]
    var_18 = 50
    var_19 = '    '
    var_20 = module_1.Config()
    var_21 = [var_1, var_2, var_3]
    var_22 = module_0.import_statement(var_0, var_21, config=var_20)
    var_23 = [var_1, var_2, var_3]
    var_24 = module_1.Config()
    var_25 = module_0.import_statement(var_0, var_23, config=var_24)
    var_26 = []
    var_27 = module_0.import_statement(var_0, var_26)
    assert var_27 == 'from module import'
    var_28 = [var_1]
    var_29 = module_0.import_statement(var_0, var_28)
    assert var_29 == 'from module import a'
    var_30 = 'd'
    var_31 = 'e'
    var_32 = 'f'
    var_33 = [var_1, var_2, var_3, var_30, var_31, var_32]
    var_34 = 20
    var_35 = module_1.Config()
    var_36 = module_0.import_statement(var_0, var_33, config=var_35)



# Parsed testcases at query #49
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import function1, function2, function3'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = 0
    var_4 = result.split(var_1)[var_3]
    var_5 = len(var_4)
    var_6 = 'from module import function1, function2  # some comment'
    var_7 = module_0.line(var_6, var_1)
    var_8 = 'from module import function1, function2, function3, function4, function5'
    var_9 = True
    var_10 = module_1.Config()
    var_11 = 'from module import function1, function2, function3'
    var_12 = module_0.line(var_11, var_1, var_10)
    var_13 = 'import module as alias'
    var_14 = module_0.line(var_13, var_1)
    var_15 = 'from module.submodule import function1, function2'
    var_16 = module_0.line(var_15, var_1)
    var_17 = result.split(var_1)[var_3]
    var_18 = len(var_17)
    var_19 = 'import module'
    var_20 = module_0.line(var_19, var_1)
    var_21 = 'from module import function1, function2, function3'
    var_22 = '\r\n'
    var_23 = module_0.line(var_21, var_22)
    var_24 = module_1.Config()
    var_25 = 'from module import function1, function2, function3'
    var_26 = module_0.line(var_25, var_1, var_24)
    var_27 = result.split(var_1)[var_3]
    var_28 = len(var_27)



# Parsed testcases at query #50
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
    var_18 = 50
    var_19 = [var_1, var_2, var_3]
    var_20 = 0
    var_21 = result.split(var_17)[var_20]
    var_22 = len(var_21)
    var_23 = 30
    var_24 = module_1.Config()
    var_25 = [var_1, var_2, var_3]
    var_26 = module_0.import_statement(var_0, var_25, config=var_24)
    var_27 = module_2.split(var_17)
    var_28 = -1
    var_29 = var_27[:var_28]
    var_30 = min(var_2)
    var_31 = -1
    var_32 = var_27[var_31]
    var_33 = len(var_32)
    var_34 = [var_29, var_2, var_31]



# Parsed testcases at query #51
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1
import re as module_2

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'foo'
    var_2 = 'bar'
    var_3 = 'baz'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    assert var_5 == 'from module import foo, bar, baz'
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
    var_17 = '\n'
    var_18 = 20
    var_19 = module_1.Config()
    var_20 = 'very_long_name_foo'
    var_21 = 'very_long_name_bar'
    var_22 = [var_20, var_21]
    var_23 = module_0.import_statement(var_0, var_22, config=var_19)
    var_24 = module_1.Config()
    var_25 = [var_1, var_2, var_3]
    var_26 = module_0.import_statement(var_0, var_25, config=var_24)
    var_27 = module_2.split(var_17)
    var_28 = -1
    var_29 = var_27[var_28]
    var_30 = len(var_29)
    var_31 = 0
    var_32 = var_27[var_31]
    var_33 = len(var_32)
    var_34 = 5
    var_35 = var_33 - var_34
    var_36 = [var_29, var_30, var_31]
    var_37 = []
    var_38 = module_0.import_statement(var_28, var_37)
    assert var_38 == 'from module import'
    var_39 = [var_29]
    var_40 = module_0.import_statement(var_28, var_39)
    assert var_40 == 'from module import foo'



# Parsed testcases at query #52
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
    var_18 = 20
    var_19 = module_1.Config()
    var_20 = 'very_long_name1'
    var_21 = 'very_long_name2'
    var_22 = [var_20, var_21]
    var_23 = module_0.import_statement(var_0, var_22, config=var_19)
    var_24 = module_2.split(var_17)
    var_25 = len(var_24)
    var_26 = -1
    var_27 = var_24[var_26]
    var_28 = len(var_27)
    var_29 = 0
    var_30 = var_24[var_29]
    var_31 = len(var_30)
    var_32 = [var_1, var_2, var_3]
    var_33 = module_1.Config()
    var_34 = [var_1, var_2]
    var_35 = module_0.import_statement(var_0, var_34, config=var_33)
    var_36 = ','
    var_37 = module_1.Config()
    var_38 = [var_1, var_2]
    var_39 = '# Ignored comment'
    var_40 = [var_39]
    var_41 = module_0.import_statement(var_0, var_38, var_40, config=var_37)



# Parsed testcases at query #53
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1
import re as module_2

def test_case_0():
    var_0 = 'from module import function'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = 'from module import function1, function2, function3, function4, function5'
    var_4 = 50
    var_5 = module_1.Config()
    var_6 = module_0.line(var_3, var_1, var_5)
    var_7 = 0
    var_8 = result.split(var_1)[var_7]
    var_9 = len(var_8)
    var_10 = 'from module import function  # some comment'
    var_11 = 30
    var_12 = True
    var_13 = module_1.Config()
    var_14 = module_0.line(var_10, var_1, var_13)
    var_15 = 'from module import function  # NOQA'
    var_16 = 20
    var_17 = module_0.line(var_15, var_1, var_13)
    var_18 = 'from module import function as alias'
    var_19 = 25
    var_20 = module_1.Config()
    var_21 = module_0.line(var_18, var_1, var_20)
    var_22 = 'from module.submodule import function'
    var_23 = module_1.Config()
    var_24 = module_0.line(var_22, var_1, var_23)
    var_25 = 'from module import function1, function2,'
    var_26 = module_1.Config()
    var_27 = module_0.line(var_25, var_1, var_26)
    var_28 = ','
    var_29 = 'from module import function1, function2, function3'
    var_30 = module_1.Config()
    var_31 = module_0.line(var_29, var_1, var_30)
    var_32 = module_2.split(var_1)
    var_33 = var_32[var_7]
    var_34 = len(var_33)
    var_35 = -1
    var_36 = var_32[var_35]
    var_37 = len(var_36)
    var_38 = 'from module import function1, function2, function3'
    var_39 = module_0.line(var_38, var_1, var_30)
    var_40 = 'from module import ('
    var_41 = 'from module import function1, function2, function3'
    var_42 = module_0.line(var_41, var_1, var_30)
    var_43 = 'from module import function  # some comment'
    var_44 = module_1.Config()
    var_45 = module_0.line(var_43, var_1, var_44)



# Parsed testcases at query #54
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import function'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = 'from module import function1, function2, function3'
    var_4 = 30
    var_5 = module_1.Config()
    var_6 = module_0.line(var_3, var_1, var_5)
    assert var_6 == 'from module import (\n    function1,\n    function2,\n    function3,\n)'
    var_7 = 'from module import function  # some comment'
    var_8 = True
    var_9 = module_1.Config()
    var_10 = module_0.line(var_7, var_1, var_9)
    assert var_10 == 'from module import (\n    function,  # some comment\n)'
    var_11 = 'from module import function  # NOQA'
    var_12 = module_0.line(var_11, var_1, var_9)
    var_13 = 'from module import function as f'
    var_14 = module_1.Config()
    var_15 = module_0.line(var_13, var_1, var_14)
    assert var_15 == 'from module import function as f'
    var_16 = 'from module.submodule import function'
    var_17 = module_1.Config()
    var_18 = module_0.line(var_16, var_1, var_17)
    assert var_18 == 'from module.submodule import (\n    function,\n)'
    var_19 = 'cimport module.function'
    var_20 = module_1.Config()
    var_21 = module_0.line(var_19, var_1, var_20)
    assert var_21 == 'cimport module.function'
    var_22 = 'from module import function,'
    var_23 = module_1.Config()
    var_24 = module_0.line(var_22, var_1, var_23)
    assert var_24 == 'from module import (\n    function,\n)'
    var_25 = 'from module import function  # some comment'
    var_26 = module_1.Config()
    var_27 = module_0.line(var_25, var_1, var_26)
    assert var_27 == 'from module import (\n    function,\n)'
    var_28 = 'from module import function1, function2, function3'
    var_29 = module_0.line(var_28, var_1, var_26)
    assert var_29 == 'from module import (\n    function1,\n    function2,\n    function3,\n)'
    var_30 = 'from module import function1, function2, function3'
    var_31 = module_0.line(var_30, var_1, var_26)
    assert var_31 == 'from module import (\n    function1,\n    function2,\n    function3,\n)'
    var_32 = 'from module import function  # noqa'
    var_33 = module_0.line(var_32, var_1, var_26)



# Parsed testcases at query #55
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = 'from module import function1, function2, function3'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    var_5 = 'from module import func1, func2  # some comment'
    var_6 = module_1.line(var_5, var_3, var_1)
    var_7 = 20
    var_8 = 'from very_long_module_name import very_long_function_name'
    var_9 = 30
    var_10 = True
    var_11 = module_0.Config()
    var_12 = 'from module import func1, func2, func3'
    var_13 = module_1.line(var_12, var_3, var_11)
    var_14 = module_0.Config()
    var_15 = 'from module import func1, func2, func3'
    var_16 = module_1.line(var_15, var_3, var_14)
    var_17 = ','
    var_18 = 'from module import func1, func2, func3'
    var_19 = 'import module as alias'
    var_20 = module_1.line(var_19, var_3, var_1)
    var_21 = 'from module import func1, func2  # noqa'
    var_22 = module_1.line(var_21, var_3, var_1)



# Parsed testcases at query #56
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = 'from module import function'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    var_5 = 'from module import function  # some comment'
    var_6 = module_1.line(var_5, var_3, var_1)
    var_7 = 'from module import function1, function2, function3'
    var_8 = module_1.line(var_7, var_3, var_1)
    var_9 = 'from module import function1, function2, function3  # some comment'
    var_10 = module_1.line(var_9, var_3, var_1)
    var_11 = 'from module import function1, function2, function3  # NOQA'
    var_12 = module_1.line(var_11, var_3, var_1)
    var_13 = 'from module import function1, function2, function3  # NOQA'
    var_14 = module_1.line(var_13, var_3, var_1)
    var_15 = 'from module import function1, function2, function3  # NOQA'
    var_16 = 20
    var_17 = module_1.line(var_15, var_3, var_1)
    var_18 = 'from module import function1, function2, function3'
    var_19 = module_1.line(var_18, var_3, var_1)
    var_20 = 'from module import function1, function2, function3  # some comment'
    var_21 = module_1.line(var_20, var_3, var_1)
    var_22 = 'from module import function1, function2, function3  # some comment'
    var_23 = True
    var_24 = module_1.line(var_22, var_3, var_1)
    var_25 = 'from module import function1, function2, function3  # some comment'
    var_26 = module_1.line(var_25, var_3, var_1)
    var_27 = 'from module import function1, function2, function3  # some comment'
    var_28 = '# '
    var_29 = module_1.line(var_27, var_3, var_1)
    var_30 = 'from module import function1, function2, function3  # some comment'
    var_31 = module_1.line(var_30, var_3, var_1)
    var_32 = 'from module import function1, function2, function3  # some comment'
    var_33 = module_1.line(var_32, var_3, var_1)
    var_34 = 'from module import function1, function2, function3  # some comment'
    var_35 = '    '
    var_36 = module_1.line(var_34, var_3, var_1)
    var_37 = 'from module import function1, function2, function3  # some comment'
    var_38 = module_1.line(var_37, var_3, var_1)
    var_39 = 'from module import function1, function2, function3  # some comment'
    var_40 = module_1.line(var_39, var_3, var_1)
    var_41 = 'from module import function1, function2, function3  # some comment'
    var_42 = module_1.line(var_41, var_3, var_1)
    var_43 = 'from module import function1, function2, function3  # some comment'
    var_44 = False
    var_45 = module_1.line(var_43, var_3, var_1)
    var_46 = 'from module import function1, function2, function3  # some comment'
    var_47 = module_1.line(var_46, var_3, var_1)
    var_48 = 'from module import function1, function2, function3  # some comment'
    var_49 = module_1.line(var_48, var_3, var_1)
    var_50 = 'from module import function1, function2, function3  # some comment'
    var_51 = module_1.line(var_50, var_3, var_1)
    var_52 = 'from module import function1, function2, function3  # some comment'
    var_53 = '#'



# Parsed testcases at query #57
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = 'from module import function1, function2, function3'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    var_5 = 'from module import function1, function2, function3, function4, function5'
    var_6 = module_1.line(var_5, var_3, var_1)
    assert var_6 == 'from module import (\n    function1,\n    function2,\n    function3,\n    function4,\n    function5\n)'
    var_7 = 'from module import function1, function2, function3  # some comment'
    var_8 = module_1.line(var_7, var_3, var_1)
    assert var_8 == 'from module import (\n    function1,\n    function2,\n    function3,  # some comment\n)'
    var_9 = 'from module import function1, function2, function3  # NOQA'
    var_10 = module_1.line(var_9, var_3, var_1)
    var_11 = 'from module import function1, function2, function3  # noqa'
    var_12 = module_1.line(var_11, var_3, var_1)
    assert var_12 == 'from module import (\n    function1,\n    function2,\n    function3,  # noqa\n)'
    var_13 = 'from module import function1 as f1, function2 as f2, function3 as f3'
    var_14 = module_1.line(var_13, var_3, var_1)
    assert var_14 == 'from module import (\n    function1 as f1,\n    function2 as f2,\n    function3 as f3\n)'
    var_15 = 'cimport module.function1, module.function2, module.function3'
    var_16 = module_1.line(var_15, var_3, var_1)
    assert var_16 == 'cimport (\n    module.function1,\n    module.function2,\n    module.function3\n)'
    var_17 = 'from module.submodule import function1, function2, function3'
    var_18 = module_1.line(var_17, var_3, var_1)
    assert var_18 == 'from module.submodule import (\n    function1,\n    function2,\n    function3\n)'
    var_19 = 'from module import function1'
    var_20 = module_1.line(var_19, var_3, var_1)
    var_21 = 'from module import function1, function2, function3'
    var_22 = module_1.line(var_21, var_3, var_1)
    assert var_22 == 'from module import function1, function2, function3  # NOQA'
    var_23 = 'from module import function1, function2, function3'
    var_24 = '\r\n'
    var_25 = module_1.line(var_23, var_24, var_1)
    assert var_25 == 'from module import function1, function2, function3  # NOQA'



# Parsed testcases at query #58
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
    var_38 = var_37 >= var_34
    var_39 = -1
    var_40 = var_31[var_39]
    var_41 = len(var_40)
    var_42 = 0
    var_43 = var_41 == var_42



# Parsed testcases at query #59
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    assert var_5 == 'from module import a, b, c'
    var_6 = [var_1, var_2, var_3]
    var_7 = '# comment'
    var_8 = [var_7]
    var_9 = module_0.import_statement(var_0, var_6, var_8)
    var_10 = [var_1, var_2, var_3]
    var_11 = '\r\n'
    var_12 = module_0.import_statement(var_0, var_10, line_separator=var_11)
    var_13 = [var_1, var_2, var_3]
    var_14 = True
    var_15 = module_0.import_statement(var_0, var_13, explode=var_14)
    assert var_15 == 'from module import (\n    a,\n    b,\n    c,\n)'
    var_16 = module_1.Config()
    var_17 = [var_1, var_2, var_3]
    var_18 = module_0.import_statement(var_0, var_17, config=var_16)
    assert var_18 == 'from module import a, b, c'
    var_19 = [var_1, var_2, var_3]
    var_20 = module_1.Config()
    var_21 = [var_1, var_2, var_3]
    var_22 = module_0.import_statement(var_0, var_21, config=var_20)
    var_23 = ','
    var_24 = module_1.Config()
    var_25 = [var_1, var_2, var_3]
    var_26 = [var_7]
    var_27 = module_0.import_statement(var_0, var_25, var_26, config=var_24)
    var_28 = '    '
    var_29 = module_1.Config()
    var_30 = [var_1, var_2, var_3]
    var_31 = module_0.import_statement(var_0, var_30, config=var_29)
    var_32 = 20
    var_33 = module_1.Config()
    var_34 = [var_1, var_2, var_3]
    var_35 = module_0.import_statement(var_0, var_34, config=var_33)
    var_36 = 0
    var_37 = '\n'
    var_38 = result.split(var_37)[var_36]
    var_39 = len(var_38)



# Parsed testcases at query #60
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
    var_17 = 50
    var_18 = [var_1, var_2, var_3]
    var_19 = module_1.Config()
    var_20 = [var_1, var_2, var_3]
    var_21 = module_0.import_statement(var_0, var_20, config=var_19)
    var_22 = [var_1, var_2, var_3]



# Parsed testcases at query #61
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import function1, function2, function3'
    var_1 = 50
    var_2 = module_0.Config()
    var_3 = '\n'
    var_4 = module_1.line(var_0, var_3, var_2)
    var_5 = 'from module import func1, func2  # some comment'
    var_6 = module_1.line(var_5, var_3, var_2)
    var_7 = 'from module import very_long_function_name_that_exceeds_line_length'
    var_8 = 30
    var_9 = True
    var_10 = module_0.Config()
    var_11 = 'from module import func1, func2, func3'
    var_12 = module_1.line(var_11, var_3, var_10)
    var_13 = module_0.Config()
    var_14 = module_1.line(var_11, var_3, var_13)
    var_15 = ','
    var_16 = module_0.Config()
    var_17 = module_1.line(var_11, var_3, var_16)
    var_18 = '    '
    var_19 = module_0.Config()
    var_20 = module_1.line(var_11, var_3, var_19)
    var_21 = '\r\n'
    var_22 = module_1.line(var_11, var_21, var_19)
    var_23 = module_0.Config()
    var_24 = module_1.line(var_5, var_3, var_23)
    var_25 = '# '
    var_26 = module_0.Config()
    var_27 = module_1.line(var_5, var_3, var_26)
    var_28 = 'from module import func1 as f1, func2 as f2'
    var_29 = module_1.line(var_28, var_3, var_2)
    var_30 = 'cimport module.func1, module.func2'
    var_31 = module_1.line(var_30, var_3, var_2)
    var_32 = 'from module import func1, func2.func3'
    var_33 = module_1.line(var_32, var_3, var_2)
    var_34 = 'import os'
    var_35 = module_1.line(var_34, var_3, var_2)
    var_36 = 'a'
    var_37 = var_2.line_length
    var_38 = var_36 * var_37
    var_39 = module_1.line(var_38, var_3, var_2)
    var_40 = ''
    var_41 = module_1.line(var_40, var_3, var_2)



# Parsed testcases at query #62
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    var_6 = [var_1, var_2]
    var_7 = '# comment'
    var_8 = [var_7]
    var_9 = module_0.import_statement(var_0, var_6, var_8)
    var_10 = [var_1, var_2]
    var_11 = '\r\n'
    var_12 = module_0.import_statement(var_0, var_10, line_separator=var_11)
    var_13 = [var_1, var_2, var_3]
    var_14 = True
    var_15 = module_0.import_statement(var_0, var_13, explode=var_14)
    var_16 = 20
    var_17 = module_1.Config()
    var_18 = [var_1, var_2, var_3]
    var_19 = module_0.import_statement(var_0, var_18, config=var_17)
    var_20 = [var_1, var_2, var_3]
    var_21 = module_1.Config()
    var_22 = [var_1, var_2, var_3]
    var_23 = module_0.import_statement(var_0, var_22, config=var_21)
    var_24 = 'c,\n)'
    var_25 = module_1.Config()
    var_26 = [var_1, var_2]
    var_27 = [var_7]
    var_28 = module_0.import_statement(var_0, var_26, var_27, config=var_25)
    var_29 = ' # '
    var_30 = module_1.Config()
    var_31 = [var_1, var_2]
    var_32 = 'comment'
    var_33 = [var_32]
    var_34 = module_0.import_statement(var_0, var_31, var_33, config=var_30)
    var_35 = module_1.Config()
    var_36 = [var_1, var_2, var_3]
    var_37 = module_0.import_statement(var_0, var_36, config=var_35)
    var_38 = 30
    var_39 = module_1.Config()
    var_40 = [var_1, var_2, var_3]
    var_41 = module_0.import_statement(var_0, var_40, config=var_39)
    var_42 = '    '
    var_43 = module_1.Config()
    var_44 = [var_1, var_2, var_3]
    var_45 = module_0.import_statement(var_0, var_44, config=var_43)



# Parsed testcases at query #63
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
    var_15 = 20
    var_16 = module_1.Config()
    var_17 = [var_1, var_2]
    var_18 = module_0.import_statement(var_0, var_17, config=var_16)
    var_19 = module_1.Config()
    var_20 = [var_1, var_2]
    var_21 = module_0.import_statement(var_0, var_20, config=var_19)
    var_22 = ','
    var_23 = module_1.Config()
    var_24 = [var_1, var_2]
    var_25 = [var_6]
    var_26 = module_0.import_statement(var_0, var_24, var_25, config=var_23)
    var_27 = '    '
    var_28 = module_1.Config()
    var_29 = [var_1, var_2]
    var_30 = module_0.import_statement(var_0, var_29, config=var_28)
    var_31 = [var_1, var_2]
    var_32 = [var_1, var_2]



# Parsed testcases at query #64
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
    var_6 = [var_1, var_2, var_3]
    var_7 = '# comment1'
    var_8 = '# comment2'
    var_9 = [var_7, var_8]
    var_10 = module_0.import_statement(var_0, var_6, var_9)
    var_11 = [var_1, var_2, var_3]
    var_12 = '\r\n'
    var_13 = module_0.import_statement(var_0, var_11, line_separator=var_12)
    var_14 = [var_1, var_2, var_3]
    var_15 = True
    var_16 = module_0.import_statement(var_0, var_14, explode=var_15)
    var_17 = '\n'
    var_18 = 50
    var_19 = '    '
    var_20 = '# '
    var_21 = False
    var_22 = [var_1, var_2, var_3]
    var_23 = [var_1, var_2, var_3]
    var_24 = [var_1, var_2, var_3]
    var_25 = module_1.Config()
    var_26 = module_0.import_statement(var_0, var_24, config=var_25)
    var_27 = []
    var_28 = module_0.import_statement(var_0, var_27)
    assert var_28 == 'from module import'
    var_29 = [var_1]
    var_30 = module_0.import_statement(var_0, var_29)
    assert var_30 == 'from module import item1'



# Parsed testcases at query #65
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    assert var_5 == 'from module import a, b, c'
    var_6 = [var_1, var_2, var_3]
    var_7 = '# comment'
    var_8 = [var_7]
    var_9 = module_0.import_statement(var_0, var_6, var_8)
    var_10 = [var_1, var_2, var_3]
    var_11 = '\r\n'
    var_12 = module_0.import_statement(var_0, var_10, line_separator=var_11)
    var_13 = [var_1, var_2, var_3]
    var_14 = True
    var_15 = module_0.import_statement(var_0, var_13, explode=var_14)
    var_16 = '\n'
    var_17 = [var_1, var_2, var_3]
    var_18 = module_1.Config()
    var_19 = [var_1, var_2, var_3]
    var_20 = module_0.import_statement(var_0, var_19, config=var_18)
    var_21 = 20
    var_22 = module_1.Config()
    var_23 = [var_1, var_2, var_3]
    var_24 = module_0.import_statement(var_0, var_23, config=var_22)
    var_25 = 0
    var_26 = result.split(var_16)[var_25]
    var_27 = len(var_26)
    var_28 = []
    var_29 = module_0.import_statement(var_0, var_28)
    assert var_29 == 'from module import'
    var_30 = [var_1]
    var_31 = module_0.import_statement(var_0, var_30)
    assert var_31 == 'from module import a'
    var_32 = 'very_long_name_1'
    var_33 = 'very_long_name_2'
    var_34 = 'very_long_name_3'
    var_35 = [var_32, var_33, var_34]
    var_36 = 30
    var_37 = module_1.Config()
    var_38 = module_0.import_statement(var_0, var_35, config=var_37)



# Parsed testcases at query #66
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
    var_14 = [var_1, var_2]
    var_15 = True
    var_16 = module_0.import_statement(var_0, var_14, explode=var_15)
    var_17 = 50
    var_18 = '    '
    var_19 = ' # '
    var_20 = module_1.Config()
    var_21 = [var_1, var_2]
    var_22 = module_0.import_statement(var_0, var_21, config=var_20)
    var_23 = [var_1, var_2]
    var_24 = [var_1, var_2, var_3]
    var_25 = module_1.Config()
    var_26 = module_0.import_statement(var_0, var_24, config=var_25)
    var_27 = []
    var_28 = module_0.import_statement(var_0, var_27)



# Parsed testcases at query #67
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1
import re as module_2

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = 'This is a very long line that should be wrapped if it exceeds the line length limit.'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    var_5 = 0
    var_6 = result.split(var_3)[var_5]
    var_7 = len(var_6)
    var_8 = 'import os# This is a comment'
    var_9 = module_1.line(var_8, var_3, var_1)
    var_10 = 'from module import function, another_function, third_function'
    var_11 = module_1.line(var_10, var_3, var_1)
    var_12 = 'cimport cython_module'
    var_13 = module_1.line(var_12, var_3, var_1)
    var_14 = 'module.submodule.function'
    var_15 = module_1.line(var_14, var_3, var_1)
    var_16 = 'import module as alias'
    var_17 = module_1.line(var_16, var_3, var_1)
    var_18 = 'very_long_line_that_should_not_be_wrapped'
    var_19 = 10
    var_20 = 20
    var_21 = True
    var_22 = module_0.Config()
    var_23 = 'from module import function, another_function'
    var_24 = module_1.line(var_23, var_3, var_22)
    var_25 = ','
    var_26 = 30
    var_27 = module_0.Config()
    var_28 = 'from module import function, another_function, third_function'
    var_29 = module_1.line(var_28, var_3, var_27)
    var_30 = module_2.split(var_3)
    var_31 = -1
    var_32 = var_30[:var_31]
    var_33 = min(var_5)
    var_34 = -1
    var_35 = var_30[var_34]
    var_36 = len(var_35)
    var_37 = 'from module import function, another_function'
    var_38 = 'from module import function, another_function'
    var_39 = module_0.Config()
    var_40 = 'import os# This comment should be ignored'
    var_41 = module_1.line(var_40, var_32, var_39)
    var_42 = '# '
    var_43 = module_0.Config()
    var_44 = 'import os#comment'
    var_45 = module_1.line(var_44, var_32, var_43)



# Parsed testcases at query #68
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    var_6 = 20
    var_7 = True
    var_8 = module_1.Config()
    var_9 = [var_1, var_2, var_3]
    var_10 = module_0.import_statement(var_0, var_9, config=var_8)
    var_11 = [var_1, var_2, var_3]
    var_12 = module_0.import_statement(var_0, var_11, explode=var_7)
    var_13 = [var_1, var_2, var_3]
    var_14 = '# Comment'
    var_15 = [var_14]
    var_16 = module_0.import_statement(var_0, var_13, var_15)
    var_17 = [var_1, var_2, var_3]
    var_18 = '\r\n'
    var_19 = module_0.import_statement(var_0, var_17, line_separator=var_18)
    var_20 = 30
    var_21 = module_1.Config()
    var_22 = [var_1, var_2, var_3]
    var_23 = module_0.import_statement(var_0, var_22, config=var_21)
    var_24 = [var_1, var_2, var_3]
    var_25 = [var_1]
    var_26 = module_0.import_statement(var_0, var_25)
    assert var_26 == 'from module import a'
    var_27 = []
    var_28 = module_0.import_statement(var_0, var_27)
    assert var_28 == 'from module import'
    var_29 = 'very_long_name_1'
    var_30 = 'very_long_name_2'
    var_31 = [var_29, var_30]
    var_32 = module_0.import_statement(var_0, var_31)



# Parsed testcases at query #69
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
    var_12 = [var_1, var_2, var_3]
    var_13 = '# Comment 1'
    var_14 = '# Comment 2'
    var_15 = [var_13, var_14]
    var_16 = module_0.import_statement(var_0, var_12, var_15)
    var_17 = [var_1, var_2, var_3]
    var_18 = True
    var_19 = module_0.import_statement(var_0, var_17, explode=var_18)
    var_20 = '\n'
    var_21 = [var_1, var_2, var_3]
    var_22 = '\r\n'
    var_23 = module_0.import_statement(var_0, var_21, line_separator=var_22)
    var_24 = 20
    var_25 = module_1.Config()
    var_26 = [var_1, var_2, var_3, var_6]
    var_27 = module_0.import_statement(var_0, var_26, config=var_25)
    var_28 = module_2.split(var_20)
    var_29 = -1
    var_30 = var_28[var_29]
    var_31 = len(var_30)
    var_32 = -1
    var_33 = var_28[:var_32]
    var_34 = len(var_28)
    var_35 = var_34 == var_18
    var_36 = module_1.Config()
    var_37 = [var_1, var_2, var_3]
    var_38 = module_0.import_statement(var_0, var_37, config=var_36)
    var_39 = ','
    var_40 = [var_1, var_2, var_3]
    var_41 = module_1.Config()
    var_42 = [var_1, var_2, var_3]
    var_43 = '# Comment'
    var_44 = [var_43]
    var_45 = module_0.import_statement(var_0, var_42, var_44, config=var_41)
    var_46 = module_1.Config()
    var_47 = [var_1, var_2, var_3]
    var_48 = module_0.import_statement(var_0, var_47, config=var_46)
    var_49 = '# '
    var_50 = module_1.Config()
    var_51 = [var_1, var_2, var_3]
    var_52 = 'Comment'
    var_53 = [var_52]
    var_54 = module_0.import_statement(var_0, var_51, var_53, config=var_50)



# Parsed testcases at query #70
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = 'from module import function1, function2, function3'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    var_5 = 'from module import function1, function2, function3, function4, function5'
    var_6 = module_1.line(var_5, var_3, var_1)
    var_7 = 'from module import function1, function2, function3  # some comment'
    var_8 = module_1.line(var_7, var_3, var_1)
    var_9 = 'from module import function1, function2, function3'
    var_10 = module_1.line(var_9, var_3, var_1)
    var_11 = True
    var_12 = module_0.Config()
    var_13 = 'from module import function1, function2, function3'
    var_14 = module_1.line(var_13, var_3, var_12)
    var_15 = 'import module as m'
    var_16 = module_1.line(var_15, var_3, var_12)
    var_17 = 'from module.submodule import function'
    var_18 = module_1.line(var_17, var_3, var_12)
    var_19 = 'cimport module.function'
    var_20 = module_1.line(var_19, var_3, var_12)
    var_21 = ''
    var_22 = module_1.line(var_21, var_3, var_12)
    assert var_22 == ''
    var_23 = 'import module'
    var_24 = module_1.line(var_23, var_3, var_12)
    var_25 = 'from module import function1, function2, function3'
    var_26 = '\r\n'
    var_27 = module_1.line(var_25, var_26, var_12)



# Parsed testcases at query #71
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
    assert var_5 == 'from module import func1, func2, func3'
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
    assert var_16 == 'from module import (\n    func1,\n    func2,\n    func3,\n)'
    var_17 = 20
    var_18 = module_1.Config()
    var_19 = [var_1, var_2, var_3]
    var_20 = module_0.import_statement(var_0, var_19, config=var_18)
    var_21 = 0
    var_22 = '\n'
    var_23 = result.split(var_22)[var_21]
    var_24 = len(var_23)
    var_25 = [var_1, var_2, var_3]
    var_26 = module_1.Config()
    var_27 = [var_1, var_2, var_3]
    var_28 = module_0.import_statement(var_0, var_27, config=var_26)
    var_29 = ','
    var_30 = module_1.Config()
    var_31 = [var_1, var_2, var_3]
    var_32 = [var_7]
    var_33 = module_0.import_statement(var_0, var_31, var_32, config=var_30)
    var_34 = '# '
    var_35 = module_1.Config()
    var_36 = [var_1, var_2, var_3]
    var_37 = 'Comment 1'
    var_38 = [var_37]
    var_39 = module_0.import_statement(var_0, var_36, var_38, config=var_35)



# Parsed testcases at query #72
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import something, something_else, another_thing'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = 'from module import something, something_else  # some comment'
    var_4 = module_0.line(var_3, var_1)
    var_5 = 'from module import something, something_else, another_thing  # NOQA'
    var_6 = module_0.line(var_5, var_1)
    var_7 = 'from module import something, something_else, another_thing, yet_another, and_more'
    var_8 = module_0.line(var_7, var_1)
    var_9 = 50
    var_10 = True
    var_11 = module_1.Config()
    var_12 = 'from module import something, something_else, another_thing'
    var_13 = module_0.line(var_12, var_1, var_11)
    var_14 = 'from module import something, something_else'
    var_15 = module_0.line(var_14, var_1)
    var_16 = 'from module import something, something_else'
    var_17 = '\r\n'
    var_18 = module_0.line(var_16, var_17)



# Parsed testcases at query #73
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import very_long_function_name_that_exceeds_line_length'
    var_1 = '\n'
    var_2 = 'from module import func  # some comment'
    var_3 = 'from module import func  # NOQA'
    var_4 = 'from module import func1, func2, func3'
    var_5 = True
    var_6 = module_0.Config()
    var_7 = 'from module import func1, func2, func3'
    var_8 = module_1.line(var_7, var_1, var_6)
    var_9 = module_0.Config()
    var_10 = 'from module import func1, func2, func3'
    var_11 = module_1.line(var_10, var_1, var_9)
    var_12 = ','
    var_13 = 'from module import func1, func2, func3'
    var_14 = module_0.Config()
    var_15 = 'from module import func1, func2, func3'
    var_16 = module_1.line(var_15, var_1, var_14)
    var_17 = ' | '
    var_18 = 'from module import func1, func2, func3'



# Parsed testcases at query #74
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from module import function'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = 'from module import function1, function2, function3'
    var_4 = 30
    var_5 = module_1.Config()
    var_6 = module_0.line(var_3, var_1, var_5)
    assert var_6 == 'from module import (\n    function1,\n    function2,\n    function3\n)'
    var_7 = 'from module import function  # comment'
    var_8 = module_1.Config()
    var_9 = module_0.line(var_7, var_1, var_8)
    assert var_9 == 'from module import (\n    function  # comment\n)'
    var_10 = 'from module import function  # NOQA'
    var_11 = module_1.Config()
    var_12 = module_0.line(var_10, var_1, var_11)
    var_13 = 'from module import function as alias'
    var_14 = module_1.Config()
    var_15 = module_0.line(var_13, var_1, var_14)
    assert var_15 == 'from module import function as (\n    alias\n)'
    var_16 = 'from module.submodule import function'
    var_17 = module_1.Config()
    var_18 = module_0.line(var_16, var_1, var_17)
    assert var_18 == 'from module.submodule import (\n    function\n)'
    var_19 = 'cfrom module import function'
    var_20 = module_1.Config()
    var_21 = module_0.line(var_19, var_1, var_20)
    assert var_21 == 'cfrom module import (\n    function\n)'
    var_22 = 'from module import function  # comment'
    var_23 = True
    var_24 = module_1.Config()
    var_25 = module_0.line(var_22, var_1, var_24)
    assert var_25 == 'from module import (\n    function\n)'
    var_26 = 'from module import function1, function2'
    var_27 = module_1.Config()
    var_28 = module_0.line(var_26, var_1, var_27)
    assert var_28 == 'from module import (\n    function1,\n    function2\n)'
    var_29 = 'from module import function1, function2'
    var_30 = module_1.Config()
    var_31 = module_0.line(var_29, var_1, var_30)
    assert var_31 == 'from module import (\n    function1,\n    function2,\n)'
    var_32 = 'from module import function1, function2'
    var_33 = module_0.line(var_32, var_1, var_30)
    assert var_33 == 'from module import (\n    function1,\n    function2,\n)'
    var_34 = 'from module import function1, function2'
    var_35 = module_0.line(var_34, var_1, var_30)
    assert var_35 == 'from module import (\n    function1,\n    function2,\n)'



# Parsed testcases at query #75
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
    assert var_5 == 'from module import func1, func2, func3'
    var_6 = [var_1, var_2, var_3]
    var_7 = True
    var_8 = module_0.import_statement(var_0, var_6, explode=var_7)
    assert var_8 == 'from module import (\n    func1,\n    func2,\n    func3,\n)'
    var_9 = 20
    var_10 = [var_1, var_2, var_3]
    var_11 = [var_1, var_2, var_3]
    var_12 = '# Comment 1'
    var_13 = '# Comment 2'
    var_14 = [var_12, var_13]
    var_15 = module_0.import_statement(var_0, var_11, var_14)
    var_16 = [var_1, var_2, var_3]
    var_17 = '\r\n'
    var_18 = module_0.import_statement(var_0, var_16, line_separator=var_17)
    assert var_18 == 'from module import (\n    func1,\n    func2,\n    func3,\n)'
    var_19 = 30
    var_20 = [var_1, var_2, var_3]
    var_21 = [var_1, var_2, var_3]
    var_22 = module_1.Config()
    var_23 = [var_1, var_2, var_3]
    var_24 = [var_12, var_13]
    var_25 = module_0.import_statement(var_0, var_23, var_24, config=var_22)



# Parsed testcases at query #76
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
    var_18 = 50
    var_19 = module_1.Config()
    var_20 = [var_1, var_2, var_3]
    var_21 = module_0.import_statement(var_0, var_20, config=var_19)
    var_22 = [var_1, var_2, var_3]
    var_23 = module_1.Config()
    var_24 = [var_1, var_2, var_3]
    var_25 = module_0.import_statement(var_0, var_24, config=var_23)
    var_26 = module_1.Config()
    var_27 = [var_1, var_2]
    var_28 = [var_7, var_8]
    var_29 = module_0.import_statement(var_0, var_27, var_28, config=var_26)
    var_30 = module_1.Config()
    var_31 = [var_1, var_2, var_3]
    var_32 = module_0.import_statement(var_0, var_31, config=var_30)
    var_33 = '# '
    var_34 = module_1.Config()
    var_35 = [var_1, var_2]
    var_36 = 'Comment 1'
    var_37 = 'Comment 2'
    var_38 = [var_36, var_37]
    var_39 = module_0.import_statement(var_0, var_35, var_38, config=var_34)



