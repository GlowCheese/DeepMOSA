####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = '\n'
    var_2 = 80
    var_3 = module_0.Config()
    var_4 = module_1.line(var_0, var_1, var_3)
    assert var_4 == 'import os'
    var_5 = 'from module import very_long_function_name'
    var_6 = 20
    var_7 = module_0.Config()
    var_8 = module_1.line(var_5, var_1, var_7)
    assert var_8 == 'from module import (\n    very_long_function_name)'
    var_9 = 'import os  # comment'
    var_10 = module_0.Config()
    var_11 = module_1.line(var_9, var_1, var_10)
    assert var_11 == 'import os  # comment'
    var_12 = 'from module import function1, function2, function3'
    var_13 = 30
    var_14 = True
    var_15 = module_0.Config()
    var_16 = module_1.line(var_12, var_1, var_15)
    assert var_16 == 'from module import (\n    function1,\n    function2,\n    function3)'
    var_17 = module_0.Config()
    var_18 = module_1.line(var_12, var_1, var_17)
    assert var_18 == 'from module import (\n    function1,\n    function2,\n    function3,\n)'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 80
    var_1 = ' # '
    var_2 = '    '
    var_3 = True
    var_4 = False
    var_5 = 'import os'
    var_6 = '\n'
    var_7 = 'from os import path'
    var_8 = 'from os import path, sys'
    var_9 = 'from os import path, sys # NOQA'
    var_10 = 'from os import path, sys # noqa'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 80
    var_1 = 88
    var_2 = '    '
    var_3 = '# '
    var_4 = True
    var_5 = False
    var_6 = 'from module.submodule import'
    var_7 = 'function1'
    var_8 = 'function2'
    var_9 = 'function3'
    var_10 = [var_7, var_8, var_9]
    var_11 = 'comment1'
    var_12 = 'comment2'
    var_13 = [var_11, var_12]
    var_14 = '\n'
    var_15 = False
    var_16 = 'from module.submodule import (\n    function1,  # comment1\n    function2,  # comment2\n    function3,\n)'



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 80
    var_1 = True
    var_2 = '    '
    var_3 = '# '
    var_4 = False
    var_5 = 'from x import'
    var_6 = 'y'
    var_7 = [var_6]
    var_8 = 'z'
    var_9 = [var_6, var_8]
    var_10 = [var_6, var_8]
    var_11 = 'comment'
    var_12 = [var_11]
    var_13 = [var_6, var_8]
    var_14 = [var_6, var_8]
    var_15 = '\r\n'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 80
    var_1 = ' #'
    var_2 = True
    var_3 = '    '
    var_4 = False
    var_5 = 'import os'
    var_6 = '\n'
    var_7 = 'from module import very_long_function_name_that_exceeds_line_length'
    var_8 = 'from module import function1, function2, function3, function4, function5'
    var_9 = 'from module import (function1, function2, function3, function4, function5)'
    var_10 = 'import os # comment'
    var_11 = 'from module import function # NOQA'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 80
    var_1 = True
    var_2 = '    '
    var_3 = '# '
    var_4 = False
    var_5 = 'from foo import'
    var_6 = 'bar'
    var_7 = 'baz'
    var_8 = [var_6, var_7]
    var_9 = [var_6, var_7]
    var_10 = 'comment'
    var_11 = [var_10]
    var_12 = [var_6, var_7]
    var_13 = [var_6, var_7]
    var_14 = '\r\n'



# Parsed testcases at query #7
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 88
    var_1 = ' '
    var_2 = 4
    var_3 = var_1 * var_2
    var_4 = True
    var_5 = ' #'
    var_6 = False
    var_7 = 'import os'
    var_8 = '\n'
    var_9 = 'from module import function'
    var_10 = 10
    var_11 = module_0.Config()
    var_12 = module_1.line(var_9, var_8, var_11)
    assert var_12 == 'from module \\\n    import function'
    var_13 = 'from module import function # comment'
    var_14 = module_0.Config()
    var_15 = module_1.line(var_13, var_8, var_14)
    assert var_15 == 'from module \\\n    import function # comment'
    var_16 = 'from module import function # noqa'
    var_17 = module_0.Config()
    var_18 = module_1.line(var_16, var_8, var_17)
    assert var_18 == 'from module import function # noqa'
    var_19 = 'from module import function as func'
    var_20 = module_0.Config()
    var_21 = module_1.line(var_19, var_8, var_20)
    assert var_21 == 'from module import function as func'
    var_22 = module_0.Config()
    var_23 = module_1.line(var_19, var_8, var_22)
    assert var_23 == 'from module import function as func'
    var_24 = module_0.Config()
    var_25 = module_1.line(var_19, var_8, var_24)
    assert var_25 == 'from module import function as func'
    var_26 = module_0.Config()
    var_27 = module_1.line(var_9, var_8, var_26)
    assert var_27 == 'from module \\\n    import function,'
    var_28 = module_0.Config()
    var_29 = module_1.line(var_9, var_8, var_28)
    assert var_29 == 'from module \\\n    import function'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 80
    var_1 = '    '
    var_2 = True
    var_3 = '# '
    var_4 = '\n'
    var_5 = 'from module import name'
    var_6 = 'from module import name1, name2, name3, name4, name5, name6, name7, name8, name9, name10'
    var_7 = 'from module import (\n    name1,\n    name2,\n    name3,\n    name4,\n    name5,\n    name6,\n    name7,\n    name8,\n    name9,\n    name10,\n)'
    var_8 = 'from module import name1, name2, name3  # some comment'
    var_9 = 'from module import (\n    name1,\n    name2,\n    name3,  # some comment\n)'
    var_10 = 'from module import name1, name2, name3, name4, name5, name6, name7, name8, name9, name10'
    var_11 = 'from module import name1, name2, name3, name4, name5, name6, name7, name8, name9, name10  # NOQA'
    var_12 = 'from module import very_long_name1 as short1, very_long_name2 as short2'
    var_13 = 'from module import (\n    very_long_name1 as short1,\n    very_long_name2 as short2,\n)'
    var_14 = 'All tests passed!'
    var_15 = print(var_14)



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 80
    var_1 = True
    var_2 = '    '
    var_3 = '# '
    var_4 = False
    var_5 = 'from foo import'
    var_6 = 'bar'
    var_7 = 'baz'
    var_8 = [var_6, var_7]
    var_9 = [var_6, var_7]
    var_10 = 'comment'
    var_11 = [var_10]
    var_12 = [var_6, var_7]
    var_13 = [var_6, var_7]
    var_14 = '\r\n'



# Parsed testcases at query #10
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'module1'
    var_1 = 'module2'
    var_2 = 'module3'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'comment1'
    var_5 = 'comment2'
    var_6 = [var_4, var_5]
    var_7 = module_0.Config()
    var_8 = 'from package'
    var_9 = module_1.import_statement(var_8, var_3, var_6, config=var_7)
    var_10 = True
    var_11 = module_1.import_statement(var_8, var_3, var_6, config=var_7, explode=var_10)



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 88
    var_1 = '    '
    var_2 = True
    var_3 = '# '
    var_4 = False
    var_5 = 'from module'
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 'c'
    var_9 = [var_6, var_7, var_8]
    var_10 = 'comment1'
    var_11 = 'comment2'
    var_12 = [var_10, var_11]
    var_13 = '\n'
    var_14 = 'from module import (\n    a,\n    b,\n    c,\n)  # comment1 # comment2'



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 88
    var_1 = True
    var_2 = '    '
    var_3 = 'from module import thing'
    var_4 = '\n'
    var_5 = 'from module import thing, another_thing, yet_another_thing'
    var_6 = 'from module import thing, another_thing # noqa'
    var_7 = 'from module import thing, another_thing, yet_another_thing # noqa'
    var_8 = False



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'module1'
    var_1 = 'module2'
    var_2 = 'module3'
    var_3 = [var_0, var_1, var_2]
    var_4 = 50
    var_5 = 80
    var_6 = '    '
    var_7 = True
    var_8 = '# '
    var_9 = False
    var_10 = 'from package import'
    var_11 = 'from package import (\n    module1,\n    module2,\n    module3,\n)'
    var_12 = 'module4'
    var_13 = 'module5'
    var_14 = [var_0, var_1, var_2, var_12, var_13]
    var_15 = 'from package import (\n    module1,\n    module2,\n    module3,\n    module4,\n    module5,\n)'
    var_16 = 'from package import (\n    module1, module2,\n    module3, module4,\n    module5,\n)'
    var_17 = 'from package import module1, module2, module3, module4, module5  # NOQA'



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 88
    var_1 = True
    var_2 = '  #'
    var_3 = '    '
    var_4 = False
    var_5 = 'from module import function1, function2, function3'
    var_6 = 'from module import (\n    function1,\n    function2,\n    function3,\n)'
    var_7 = '\n'
    var_8 = 'from module import function1, function2, function3  # NOQA'
    var_9 = 'from module import function1, function2, function3  # NOQA'
    var_10 = 'from module import function1, function2, function3'
    var_11 = 'from module import function1, function2, function3  # NOQA'
    var_12 = 'from module import function1, function2, function3'
    var_13 = 'from module import (\n    function1,\n    function2,\n    function3,\n)'
    var_14 = 'from module import function1, function2, function3'
    var_15 = 'from module import function1,\\\n    function2,\\\n    function3'



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 80
    var_1 = True
    var_2 = '    '
    var_3 = '# '
    var_4 = False
    var_5 = 'from module import function'
    var_6 = 'from module import function'
    var_7 = '\n'
    var_8 = 'from module import function, another_function, yet_another_function'
    var_9 = 'from module import (\n    function,\n    another_function,\n    yet_another_function,\n)'
    var_10 = 'from module import function, another_function, yet_another_function'
    var_11 = 'from module import function, another_function, yet_another_function # NOQA'
    var_12 = 'from module import function # This is a comment'
    var_13 = 'from module import function # This is a comment'
    var_14 = 'from module import function, another_function, yet_another_function # This is a comment'
    var_15 = 'from module import (\n    function,\n    another_function,\n    yet_another_function,  # This is a comment\n)'
    var_16 = 'from module import function, another_function, yet_another_function # NOQA'
    var_17 = 'from module import (\n    function,\n    another_function,\n    yet_another_function,  # NOQA\n)'
    var_18 = 'from module import function, another_function, yet_another_function'
    var_19 = 'from module import (\n    function,\n    another_function,\n    yet_another_function,\n)'
    var_20 = 'from module import function, another_function, yet_another_function'
    var_21 = 'from module import function, \\\n    another_function, \\\n    yet_another_function'
    var_22 = 'from module import function, another_function, yet_another_function'
    var_23 = 'from module import function, another_function, yet_another_function # NOQA'
    var_24 = 'from module import function, another_function, yet_another_function # This is a comment'
    var_25 = 'from module import function, \\\n    another_function, \\\n    yet_another_function # This is a comment'
    var_26 = 'from module import function, another_function, yet_another_function # NOQA'
    var_27 = 'from module import function, \\\n    another_function, \\\n    yet_another_function # NOQA'
    var_28 = 'from module import function, another_function, yet_another_function'
    var_29 = 'from module import function, \\\n    another_function, \\\n    yet_another_function'
    var_30 = 'from module import function, another_function, yet_another_function'
    var_31 = 'from module import function, another_function, yet_another_function # NOQA'
    var_32 = 'from module import function, another_function, yet_another_function # This is a comment'
    var_33 = 'from module import function, \\\n    another_function, \\\n    yet_another_function # This is a comment'
    var_34 = 'from module import function, another_function, yet_another_function # NOQA'
    var_35 = 'from module import function, \\\n    another_function, \\\n    yet_another_function # NOQA'
    var_36 = 'from module import function, another_function, yet_another_function # NOQA'
    var_37 = 'from module import function, \\\n    another_function, \\\n    yet_another_function'
    var_38 = 'from module import function, another_function, yet_another_function # NOQA'
    var_39 = 'from module import function, \\\n    another_function, \\\n    yet_another_function'
    var_40 = 'from module import function, another_function, yet_another_function # NOQA'
    var_41 = 'from module import function, \\\n  another_function, \\\n  yet_another_function'
    var_42 = 'from module import function, another_function, yet_another_function # NOQA'
    var_43 = 'from module import function, \\\n  another_function, \\\n  yet_another_function'



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = 'import os'
    var_2 = '\n'
    var_3 = 'from module import something'
    var_4 = 10
    var_5 = module_0.Config()
    var_6 = module_1.line(var_3, var_2, var_5)
    assert var_6 == 'from module \\\n    import something'
    var_7 = 'from module import something  # comment'
    var_8 = module_0.Config()
    var_9 = module_1.line(var_7, var_2, var_8)
    assert var_9 == 'from module \\\n    import something  # comment'
    var_10 = 'from module import something  # noqa'
    var_11 = module_0.Config()
    var_12 = module_1.line(var_10, var_2, var_11)
    assert var_12 == 'from module import something  # noqa'
    var_13 = True
    var_14 = module_0.Config()
    var_15 = module_1.line(var_3, var_2, var_14)
    assert var_15 == 'from module import (something)'
    var_16 = module_0.Config()
    var_17 = module_1.line(var_3, var_2, var_16)
    assert var_17 == 'from module import (something,)'
    var_18 = module_0.Config()
    var_19 = module_1.line(var_10, var_2, var_18)
    assert var_19 == 'from module import (something)  # noqa'
    var_20 = module_0.Config()
    var_21 = module_1.line(var_10, var_2, var_20)
    assert var_21 == 'from module import (something,)  # noqa'
    var_22 = module_0.Config()
    var_23 = module_1.line(var_7, var_2, var_22)
    assert var_23 == 'from module import (something  # comment)'
    var_24 = module_0.Config()
    var_25 = module_1.line(var_7, var_2, var_24)
    assert var_25 == 'from module import (something,  # comment)'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 80
    var_1 = '    '
    var_2 = '# '
    var_3 = True
    var_4 = False
    var_5 = 'from module'
    var_6 = 'import1'
    var_7 = 'import2'
    var_8 = 'import3'
    var_9 = [var_6, var_7, var_8]
    var_10 = 'comment1'
    var_11 = 'comment2'
    var_12 = [var_10, var_11]
    var_13 = '\n'
    var_14 = 'from module import (\n    import1,  # comment1\n    import2,  # comment2\n    import3\n)'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 50
    var_1 = 80
    var_2 = '    '
    var_3 = True
    var_4 = ' # '
    var_5 = False
    var_6 = 'from module'
    var_7 = 'import1'
    var_8 = 'import2'
    var_9 = 'import3'
    var_10 = [var_7, var_8, var_9]
    var_11 = 'comment1'
    var_12 = 'comment2'
    var_13 = [var_11, var_12]
    var_14 = '\n'
    var_15 = 'from module import (\n    import1,\n    import2,\n    import3,  # comment1\n)  # comment2'
    var_16 = [var_7, var_8, var_9]
    var_17 = [var_11, var_12]
    var_18 = 'from module import (\n    import1,\n    import2,\n    import3,\n)'



# Parsed testcases at query #4
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'from foo'
    var_1 = 'bar'
    var_2 = [var_1]
    var_3 = module_0.import_statement(var_0, var_2)
    assert var_3 == 'from foo import bar'
    var_4 = 'baz'
    var_5 = [var_1, var_4]
    var_6 = module_0.import_statement(var_0, var_5)
    assert var_6 == 'from foo import bar, baz'
    var_7 = [var_1]
    var_8 = 'comment'
    var_9 = [var_8]
    var_10 = module_0.import_statement(var_0, var_7, var_9)
    assert var_10 == 'from foo import bar  # comment'
    var_11 = [var_1, var_4]
    var_12 = '\n'
    var_13 = module_0.import_statement(var_0, var_11, line_separator=var_12)
    assert var_13 == 'from foo import bar, baz'
    var_14 = [var_1, var_4]
    var_15 = [var_1, var_4]
    var_16 = True
    var_17 = module_0.import_statement(var_0, var_15, explode=var_16)
    assert var_17 == 'from foo import (\n    bar,\n    baz,\n)'
    var_18 = 20
    var_19 = module_1.Config()
    var_20 = 'qux'
    var_21 = 'quux'
    var_22 = [var_1, var_4, var_20, var_21]
    var_23 = module_0.import_statement(var_0, var_22, config=var_19)
    assert var_23 == 'from foo import bar, baz,\\\n    qux, quux'



# Parsed testcases at query #5
#--------------------------


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module'
    var_1 = 'import1'
    var_2 = 'import2'
    var_3 = [var_1, var_2]
    var_4 = True
    var_5 = module_0.import_statement(var_0, var_3, explode=var_4)
    assert var_5 == 'from module import (\n    import1,\n    import2,\n)'
    var_6 = [var_1, var_2]
    var_7 = [var_1, var_2]
    var_8 = 'comment1'
    var_9 = 'comment2'
    var_10 = [var_8, var_9]
    var_11 = module_0.import_statement(var_0, var_7, var_10)
    assert var_11 == 'from module import (  # comment1\n    import1,  # comment2\n    import2,\n)'
    var_12 = [var_1, var_2]
    var_13 = '\r\n'
    var_14 = module_0.import_statement(var_0, var_12, line_separator=var_13)
    assert var_14 == 'from module import (\r\n    import1,\r\n    import2,\r\n)'
    var_15 = [var_1, var_2]



# Parsed testcases at query #6
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'import foo'
    var_2 = '\n'
    var_3 = module_1.line(var_1, var_2, var_0)
    assert var_3 == 'import foo'
    var_4 = 'from foo import bar'
    var_5 = module_1.line(var_4, var_2, var_0)
    assert var_5 == 'from foo import bar'
    var_6 = 'from foo import bar, baz'
    var_7 = module_1.line(var_6, var_2, var_0)
    assert var_7 == 'from foo import (\n    bar, baz,)'
    var_8 = 'from foo import bar, baz, qux'
    var_9 = module_1.line(var_8, var_2, var_0)
    assert var_9 == 'from foo import (\n    bar, baz, qux,)'
    var_10 = 'from foo import bar, baz, qux, quux'
    var_11 = module_1.line(var_10, var_2, var_0)
    assert var_11 == 'from foo import (\n    bar, baz, qux, quux,)'
    var_12 = 'from foo import bar, baz, qux, quux, quuz'
    var_13 = module_1.line(var_12, var_2, var_0)
    assert var_13 == 'from foo import (\n    bar, baz, qux, quux, quuz,)'
    var_14 = 'from foo import bar, baz, qux, quux, quuz, corge'
    var_15 = module_1.line(var_14, var_2, var_0)
    assert var_15 == 'from foo import (\n    bar, baz, qux, quux, quuz, corge,)'
    var_16 = 'from foo import bar, baz, qux, quux, quuz, corge, grault'
    var_17 = module_1.line(var_16, var_2, var_0)
    assert var_17 == 'from foo import (\n    bar, baz, qux, quux, quuz, corge, grault,)'
    var_18 = 'from foo import bar, baz, qux, quux, quuz, corge, grault, garply'
    var_19 = module_1.line(var_18, var_2, var_0)
    assert var_19 == 'from foo import (\n    bar, baz, qux, quux, quuz, corge, grault, garply,)'
    var_20 = 'from foo import bar, baz, qux, quux, quuz, corge, grault, garply, waldo'
    var_21 = module_1.line(var_20, var_2, var_0)
    assert var_21 == 'from foo import (\n    bar, baz, qux, quux, quuz, corge, grault, garply, waldo,)'
    var_22 = 'from foo import bar, baz, qux, quux, quuz, corge, grault, garply, waldo, fred'
    var_23 = module_1.line(var_22, var_2, var_0)
    assert var_23 == 'from foo import (\n    bar, baz, qux, quux, quuz, corge, grault, garply, waldo, fred,)'
    var_24 = 'from foo import bar, baz, qux, quux, quuz, corge, grault, garply, waldo, fred, plugh'
    var_25 = module_1.line(var_24, var_2, var_0)
    assert var_25 == 'from foo import (\n    bar, baz, qux, quux, quuz, corge, grault, garply, waldo, fred, plugh,)'
    var_26 = 'from foo import bar, baz, qux, quux, quuz, corge, grault, garply, waldo, fred, plugh, xyzzy'
    var_27 = module_1.line(var_26, var_2, var_0)
    assert var_27 == 'from foo import (\n    bar, baz, qux, quux, quuz, corge, grault, garply, waldo, fred, plugh, xyzzy,)'
    var_28 = 'from foo import bar, baz, qux, quux, quuz, corge, grault, garply, waldo, fred, plugh, xyzzy, thud'
    var_29 = module_1.line(var_28, var_2, var_0)
    assert var_29 == 'from foo import (\n    bar, baz, qux, quux, quuz, corge, grault, garply, waldo, fred, plugh, xyzzy, thud,)'
    var_30 = 'from foo import bar, baz, qux, quux, quuz, corge, grault, garply, waldo, fred, plugh, xyzzy, thud, waldo'
    var_31 = module_1.line(var_30, var_2, var_0)
    assert var_31 == 'from foo import (\n    bar, baz, qux, quux, quuz, corge, grault, garply, waldo, fred, plugh, xyzzy, thud, waldo,)'
    var_32 = 'from foo import bar, baz, qux, quux, quuz, corge, grault, garply, waldo, fred, plugh, xyzzy, thud, waldo, fred'
    var_33 = module_1.line(var_32, var_2, var_0)
    assert var_33 == 'from foo import (\n    bar, baz, qux, quux, quuz, corge, grault, garply, waldo, fred, plugh, xyzzy, thud, waldo, fred,)'
    var_34 = 'from foo import bar, baz, qux, quux, quuz, corge, grault, garply, waldo, fred, plugh, xyzzy, thud, waldo, fred, plugh'
    var_35 = module_1.line(var_34, var_2, var_0)
    assert var_35 == 'from foo import (\n    bar, baz, qux, quux, quuz, corge, grault, garply, waldo, fred, plugh, xyzzy, thud, waldo, fred, plugh,)'
    var_36 = 'from foo import bar, baz, qux, quux, quuz, corge, grault, garply, waldo, fred, plugh, xyzzy, thud, waldo, fred, plugh, xyzzy'
    var_37 = module_1.line(var_36, var_2, var_0)
    assert var_37 == 'from foo import (\n    bar, baz, qux, quux, quuz, corge, grault, garply, waldo, fred, plugh, xyzzy, thud, waldo, fred, plugh, xyzzy,)'
    var_38 = 'from foo import bar, baz, qux, quux, quuz, corge, grault, garply, waldo, fred, plugh, xyzzy, thud, waldo, fred, plugh, xyzzy, thud'
    var_39 = module_1.line(var_38, var_2, var_0)
    assert var_39 == 'from foo import (\n    bar, baz, qux, quux, quuz, corge, grault, garply, waldo, fred, plugh, xyzzy, thud, waldo, fred, plugh, xyzzy, thud,)'
    var_40 = 'from foo import bar, baz, qux, quux, quuz, corge, grault, garply, waldo, fred, plugh, xyzzy, thud, waldo, fred, plugh, xyzzy, thud, waldo'
    var_41 = module_1.line(var_40, var_2, var_0)
    assert var_41 == 'from foo import (\n    bar, baz, qux, quux, quuz, corge, grault, garply, waldo, fred, plugh, xyzzy, thud, waldo, fred, plugh, xyzzy, thud, waldo,)'
    var_42 = 'from foo import bar, baz, qux, quux, quuz, corge, grault, garply, waldo, fred, plugh, xyzzy, thud, waldo, fred, plugh, xyzzy, thud, waldo, fred'
    var_43 = module_1.line(var_42, var_2, var_0)
    assert var_43 == 'from foo import (\n    bar, baz, qux, quux, quuz, corge, grault, garply, waldo, fred, plugh, xyzzy, thud, waldo, fred, plugh, xyzzy, thud, waldo, fred,)'
    var_44 = 'from foo import bar, baz, qux, quux, quuz, corge, grault, garply, waldo, fred, plugh, xyzzy, thud, waldo, fred, plugh, xyzzy, thud, waldo, fred, plugh'
    var_45 = module_1.line(var_44, var_2, var_0)
    assert var_45 == 'from foo import (\n    bar, baz, qux, quux, quuz, corge, grault, garply, waldo, fred, plugh, xyzzy, thud, waldo, fred, plugh, xyzzy, thud, waldo, fred, plugh,)'
    var_46 = 'from foo import bar, baz, qux, quux, quuz, corge, grault, garply, waldo, fred, plugh, xyzzy, thud, waldo, fred, plugh, xyzzy, thud, waldo, fred, plugh, xyzzy'
    var_47 = module_1.line(var_46, var_2, var_0)
    assert var_47 == 'from foo import (\n    bar, baz, qux, quux, quuz, corge, grault, garply, waldo, fred, plugh, xyzzy, thud, waldo, fred, plugh, xyzzy, thud, waldo, fred, plugh, xyzzy,)'



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 88
    var_1 = True
    var_2 = '    '
    var_3 = ' # '
    var_4 = False
    var_5 = 'import os'
    var_6 = '\n'
    var_7 = 'from module import really_long_name_that_exceeds_line_length_by_a_lot'
    var_8 = 'import os  # comment'
    var_9 = 'from module import really_long_name_that_exceeds_line_length_by_a_lot  # comment'
    var_10 = 'from module import really_long_name_that_exceeds_line_length_by_a_lot  # NOQA'



# Parsed testcases at query #8
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'from foo'
    var_2 = 'bar'
    var_3 = 'baz'
    var_4 = [var_2, var_3]
    var_5 = True
    var_6 = module_1.import_statement(var_1, var_4, config=var_0, explode=var_5)
    assert var_6 == 'from foo import \\\n    bar, \\\n    baz'
    var_7 = [var_2, var_3]
    var_8 = [var_2, var_3]
    var_9 = [var_2, var_3]
    var_10 = [var_2, var_3]
    var_11 = [var_2, var_3]
    var_12 = [var_2, var_3]
    var_13 = [var_2, var_3]
    var_14 = [var_2, var_3]



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'import os'
    var_1 = '\n'
    var_2 = 'import os #comment'
    var_3 = 'from os.path import dirname, basename, join'
    var_4 = 'from os.path import (\n    dirname,\n    basename,\n    join\n)'
    var_5 = 'from os.path import dirname, basename, join'
    var_6 = 'from os.path import dirname, basename, join, splitext, isfile'
    var_7 = 'from os.path import (\n    dirname,\n    basename,\n    join,\n    splitext,\n    isfile\n)'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'from module import something'
    var_1 = '\n'
    var_2 = 'from module import something, another_thing, yet_another_thing, and_more'
    var_3 = 'from module import something # NOQA'
    var_4 = 'from module import something, another_thing, yet_another_thing, and_more # NOQA'
    var_5 = 'from module import something, another_thing, yet_another_thing, and_more'
    var_6 = 'from module import something, another_thing, yet_another_thing, and_more'
    var_7 = 'from module import something, another_thing, yet_another_thing, and_more # NOQA'
    var_8 = 'from module import something, another_thing, yet_another_thing, and_more'
    var_9 = 'from module import something, another_thing, yet_another_thing, and_more # NOQA'
    var_10 = 'from module import something, another_thing, yet_another_thing, and_more'
    var_11 = 'from module import something, another_thing, yet_another_thing, and_more # NOQA'
    var_12 = 'from module import something, another_thing, yet_another_thing, and_more'
    var_13 = 'from module import something, another_thing, yet_another_thing, and_more # NOQA'
    var_14 = 'from module import something, another_thing, yet_another_thing, and_more'
    var_15 = 'from module import something, another_thing, yet_another_thing, and_more # NOQA'



# Parsed testcases at query #11
#--------------------------


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module'
    var_1 = 'function1'
    var_2 = 'function2'
    var_3 = [var_1, var_2]
    var_4 = module_0.import_statement(var_0, var_3)
    assert var_4 == 'from module import function1, function2'
    var_5 = 'from module'
    var_6 = 'function3'
    var_7 = 'function4'
    var_8 = [var_1, var_2, var_6, var_7]
    var_9 = 40
    var_10 = 'from module import (\n    function1,\n    function2,\n    function3,\n    function4\n)'
    var_11 = 'from module'
    var_12 = [var_1, var_2]
    var_13 = 'comment1'
    var_14 = 'comment2'
    var_15 = [var_13, var_14]
    var_16 = ' # '
    var_17 = 'from module import (\n    function1,  # comment1\n    function2  # comment2\n)'
    var_18 = 'from module'
    var_19 = [var_1, var_2]
    var_20 = True
    var_21 = module_0.import_statement(var_18, var_19, explode=var_20)
    var_22 = 'from module import (\n    function1,\n    function2,\n)'



# Parsed testcases at query #12
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'Test the import_statement function.'
    var_1 = module_0.Config()
    var_2 = 'from x import '
    var_3 = 'y'
    var_4 = 'z'
    var_5 = [var_3, var_4]
    var_6 = module_1.import_statement(var_2, var_5, config=var_1)
    assert var_6 == 'from x import y, z'
    var_7 = [var_3, var_4]
    var_8 = True
    var_9 = module_1.import_statement(var_2, var_7, config=var_1, explode=var_8)
    assert var_9 == 'from x import (\n    y,\n    z,\n)'
    var_10 = [var_3, var_4]
    var_11 = '# comment'
    var_12 = [var_11]
    var_13 = module_1.import_statement(var_2, var_10, var_12, config=var_1)
    assert var_13 == 'from x import y, z  # comment'
    var_14 = [var_3, var_4]
    var_15 = [var_11]
    var_16 = module_1.import_statement(var_2, var_14, var_15, config=var_1, explode=var_8)
    assert var_16 == 'from x import (\n    y,\n    z,\n)  # comment'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 80
    var_1 = '    '
    var_2 = True
    var_3 = ' # '
    var_4 = False
    var_5 = 'from my_module'
    var_6 = 'import1'
    var_7 = 'import2'
    var_8 = 'import3'
    var_9 = [var_6, var_7, var_8]
    var_10 = [var_6, var_7, var_8]
    var_11 = [var_6, var_7, var_8]



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 20
    var_1 = 'import a_long_module_name'
    var_2 = '\n'
    var_3 = 'import a_long_module_name # comment'
    var_4 = True
    var_5 = 'from module import a_long_function_name'



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'from module import very_long_function_name_that_exceeds_line_length'
    var_1 = '\n'
    var_2 = 50
    var_3 = 'from module import very_long_function_name_that_exceeds_line_length'
    var_4 = '\n'
    var_5 = 'from module import short_name'
    var_6 = '\n'
    var_7 = 'from module import very_long_function_name_that_exceeds_line_length # some comment'
    var_8 = '\n'
    var_9 = 'from module import very_long_function_name_that_exceeds_line_length # existing comment'
    var_10 = '\n'



# Parsed testcases at query #16
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = 'import os'
    var_2 = '\n'
    var_3 = 'from x import y'
    var_4 = 'from x import y  # noqa'
    var_5 = 'from x import y  # comment'
    var_6 = 10
    var_7 = module_0.Config()
    var_8 = module_1.line(var_5, var_2, var_7)
    assert var_8 == 'from x import y  # comment'



# Parsed testcases at query #17
#--------------------------


import isort.wrap as module_0

def test_case_0():
    var_0 = 'Test the import_statement function.'
    var_1 = 80
    var_2 = '    '
    var_3 = True
    var_4 = '# '
    var_5 = False
    var_6 = 'from foo import'
    var_7 = 'bar'
    var_8 = 'baz'
    var_9 = [var_7, var_8]
    var_10 = [var_7]
    var_11 = [var_7, var_8]
    var_12 = module_0.import_statement(var_6, var_11, explode=var_3)
    assert var_12 == 'from foo import \\\n    bar, \\\n    baz'



# Parsed testcases at query #18
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'import os'
    var_3 = 'import os, sys, math'
    var_4 = 50
    var_5 = module_1.Config()
    var_6 = module_0.line(var_3, var_1, var_5)
    assert var_6 == 'import os, sys, math'
    var_7 = 'from module import very_long_name_that_needs_wrapping'
    var_8 = 30
    var_9 = module_1.Config()
    var_10 = module_0.line(var_7, var_1, var_9)
    assert var_10 == 'from module import \\\n    very_long_name_that_needs_wrapping'
    var_11 = 'import os # comment'
    var_12 = module_0.line(var_11, var_1)
    assert var_12 == 'import os # comment'
    assert var_12 == 'import os, sys, math # NOQA'
    var_13 = 'import os, sys, math'
    var_14 = 10
    var_15 = 'from module import (very_long_name_that_needs_wrapping)'
    var_16 = True
    var_17 = module_1.Config()
    var_18 = module_0.line(var_15, var_1, var_17)
    assert var_18 == 'from module import (\n    very_long_name_that_needs_wrapping\n)'
    var_19 = 'from module import very_long_name_that_needs_wrapping'
    var_20 = module_1.Config()
    var_21 = module_0.line(var_19, var_1, var_20)
    assert var_21 == 'from module import (\n    very_long_name_that_needs_wrapping,\n)'



# Parsed testcases at query #19
#--------------------------


import isort.wrap as module_0

def test_case_0():
    var_0 = 88
    var_1 = '    '
    var_2 = True
    var_3 = '# '
    var_4 = False
    var_5 = 'from module'
    var_6 = 'import1'
    var_7 = 'import2'
    var_8 = [var_6, var_7]
    var_9 = [var_6]
    var_10 = 'comment1'
    var_11 = [var_10]
    var_12 = [var_6, var_7]
    var_13 = module_0.import_statement(var_5, var_12, explode=var_2)
    assert var_13 == 'from module import (\n    import1,\n    import2,\n)'
    var_14 = [var_6, var_7]
    var_15 = '\r\n'



# Parsed testcases at query #20
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'import os'
    var_2 = '\n'
    var_3 = module_1.line(var_1, var_2, var_0)
    assert var_3 == 'import os'
    var_4 = 'import os, sys'
    var_5 = module_1.line(var_4, var_2, var_0)
    assert var_5 == 'import os, sys'
    var_6 = 'from module import something, another_thing'
    var_7 = module_1.line(var_6, var_2, var_0)
    assert var_7 == 'from module import something, another_thing'
    var_8 = 'from module import something, another_thing, yet_another_thing, and_another_thing'
    var_9 = 'from module import (something, another_thing, yet_another_thing,\n    and_another_thing)'
    var_10 = module_1.line(var_8, var_2, var_0)
    var_11 = 'import os  # noqa'
    var_12 = module_1.line(var_11, var_2, var_0)
    assert var_12 == 'import os  # noqa'
    var_13 = 'import os  # NOQA'
    var_14 = module_1.line(var_13, var_2, var_0)
    assert var_14 == 'import os  # NOQA'
    var_15 = module_1.line(var_1, var_2, var_0)
    assert var_15 == 'import os'
    var_16 = 'import os  # comment'
    var_17 = module_1.line(var_16, var_2, var_0)
    assert var_17 == 'import os  # comment'
    var_18 = 'from module import something, another_thing  # comment'
    var_19 = module_1.line(var_18, var_2, var_0)
    assert var_19 == 'from module import something, another_thing  # comment'
    var_20 = 'from module import something, another_thing, yet_another_thing, and_another_thing  # comment'
    var_21 = 'from module import (something, another_thing, yet_another_thing,\n    and_another_thing)  # comment'
    var_22 = module_1.line(var_20, var_2, var_0)



# Parsed testcases at query #21
#--------------------------


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'from x import '
    var_2 = 'y'
    var_3 = [var_2]
    var_4 = module_1.import_statement(var_1, var_3, config=var_0)
    assert var_4 == 'from x import y'
    var_5 = 'z'
    var_6 = [var_2, var_5]
    var_7 = module_1.import_statement(var_1, var_6, config=var_0)
    assert var_7 == 'from x import y, z'
    var_8 = [var_2, var_5]
    var_9 = True
    var_10 = module_1.import_statement(var_1, var_8, config=var_0, explode=var_9)
    assert var_10 == 'from x import (\n    y,\n    z,\n)'



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 80
    var_1 = '    '
    var_2 = ' # '
    var_3 = None
    var_4 = True
    var_5 = False
    var_6 = '\n'
    var_7 = 'import os'
    var_8 = 'import os'
    var_9 = 'from very_long_module_name import very_long_function_name, another_long_function_name'
    var_10 = 'from very_long_module_name import (\n    very_long_function_name,\n    another_long_function_name,\n)'
    var_11 = 'import os  # comment'
    var_12 = 'import os  # comment'
    var_13 = 'from module import function1, function2, function3  # comment'
    var_14 = 'from module import (\n    function1,\n    function2,\n    function3,  # comment\n)'
    var_15 = 'from module import function1, function2, function3, function4, function5, function6'
    var_16 = 'from module import function1, function2, function3, function4, function5, function6  # NOQA'
    var_17 = 'All test cases passed!'
    var_18 = print(var_17)



# Parsed testcases at query #23
#--------------------------


import isort.wrap as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'Test the import_statement function.'
    var_1 = 'from foo'
    var_2 = 'bar'
    var_3 = [var_2]
    var_4 = module_0.import_statement(var_1, var_3)
    assert var_4 == 'from foo import bar'
    var_5 = 'baz'
    var_6 = [var_2, var_5]
    var_7 = module_0.import_statement(var_1, var_6)
    assert var_7 == 'from foo import bar, baz'
    var_8 = [var_2]
    var_9 = 'comment'
    var_10 = [var_9]
    var_11 = module_0.import_statement(var_1, var_8, var_10)
    assert var_11 == 'from foo import bar  # comment'
    var_12 = [var_2, var_5]
    var_13 = '\r\n'
    var_14 = module_0.import_statement(var_1, var_12, line_separator=var_13)
    assert var_14 == 'from foo import bar, baz'
    var_15 = [var_2, var_5]
    var_16 = 20
    var_17 = module_1.Config()
    var_18 = [var_2, var_5]
    var_19 = True
    var_20 = module_0.import_statement(var_1, var_18, explode=var_19)
    assert var_20 == 'from foo import (\n    bar,\n    baz,\n)'
    var_21 = module_1.Config()
    var_22 = 'qux'
    var_23 = [var_2, var_5, var_22]
    var_24 = module_0.import_statement(var_1, var_23, config=var_21)
    assert var_24 == 'from foo import (\n    bar,\n    baz,\n    qux,\n)'



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 88
    var_1 = True
    var_2 = '    '
    var_3 = ' # '
    var_4 = False
    var_5 = 'from module'
    var_6 = 'import1'
    var_7 = 'import2'
    var_8 = 'import3'
    var_9 = [var_6, var_7, var_8]
    var_10 = 'comment1'
    var_11 = 'comment2'
    var_12 = [var_10, var_11]
    var_13 = 'from module import (\n    import1,  # comment1\n    import2,  # comment2\n    import3,\n)'
    var_14 = 'from module import (\n    import1,\n    import2,\n    import3,\n)'



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 200
    var_1 = 'import_module'
    var_2 = '\n'
    var_3 = 'import_module # NOQA'
    var_4 = 1
    var_5 = 'from module import something'
    var_6 = 10
    var_7 = True
    var_8 = True
    var_9 = True
    var_10 = True
    var_11 = True
    var_12 = True
    var_13 = True
    var_14 = True
    var_15 = True
    var_16 = True
    var_17 = ' # '
    var_18 = True
    var_19 = True
    var_20 = True
    var_21 = True
    var_22 = True
    var_23 = True



# Parsed testcases at query #26
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'from module'
    var_1 = 'import1'
    var_2 = 'import2'
    var_3 = 'import3'
    var_4 = [var_1, var_2, var_3]
    var_5 = 'comment1'
    var_6 = 'comment2'
    var_7 = [var_5, var_6]
    var_8 = '\n'
    var_9 = module_0.Config()
    var_10 = False



