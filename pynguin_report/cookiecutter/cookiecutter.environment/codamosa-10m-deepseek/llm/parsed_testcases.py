####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'cookiecutter.extensions.TestExtension'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = module_0.ExtensionLoaderMixin(context=var_5)



# Parsed testcases at query #2
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = {}
    var_3 = module_0.ExtensionLoaderMixin(context=var_2)
    var_4 = 'cookiecutter'
    var_5 = '_extensions'
    var_6 = 'test.ext'
    var_7 = [var_6]
    var_8 = {var_5: var_7}
    var_9 = {var_4: var_8}
    var_10 = module_0.ExtensionLoaderMixin(context=var_9)
    var_11 = 'cookiecutter'
    var_12 = '_extensions'
    var_13 = 'nonexistent.extension'
    var_14 = [var_13]
    var_15 = {var_12: var_14}
    var_16 = {var_11: var_15}
    var_17 = module_0.ExtensionLoaderMixin(context=var_16)



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'test_extension'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}



# Parsed testcases at query #4
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = {}
    var_3 = 'cookiecutter'
    var_4 = '_extensions'
    var_5 = 'test.ext'
    var_6 = [var_5]
    var_7 = {var_4: var_6}
    var_8 = {var_3: var_7}
    var_9 = module_0.ExtensionLoaderMixin(context=var_8)
    var_10 = {}
    var_11 = {var_3: var_10}
    var_12 = module_0.ExtensionLoaderMixin(context=var_11)



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'ext1'
    var_3 = 'ext2'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}



# Parsed testcases at query #6
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'cookiecutter.extensions.JsonifyExtension'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = module_0.ExtensionLoaderMixin(context=var_5)



# Parsed testcases at query #7
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = module_0.ExtensionLoaderMixin()
    var_1 = '_read_extensions'
    var_2 = hasattr(var_0, var_1)
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = module_0.ExtensionLoaderMixin(context=var_5)
    var_7 = hasattr(var_6, var_1)
    var_8 = '_extensions'
    var_9 = 'test.ext'
    var_10 = [var_9]
    var_11 = {var_8: var_10}
    var_12 = {var_3: var_11}
    var_13 = module_0.ExtensionLoaderMixin(context=var_12)
    var_14 = hasattr(var_13, var_1)
    var_15 = 'cookiecutter'
    var_16 = '_extensions'
    var_17 = 'nonexistent.extension'
    var_18 = [var_17]
    var_19 = {var_16: var_18}
    var_20 = {var_15: var_19}
    var_21 = module_0.ExtensionLoaderMixin(context=var_20)



# Parsed testcases at query #8
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = {}
    var_3 = module_0.ExtensionLoaderMixin(context=var_2)
    var_4 = 'cookiecutter'
    var_5 = '_extensions'
    var_6 = 'test.ext'
    var_7 = [var_6]
    var_8 = {var_5: var_7}
    var_9 = {var_4: var_8}
    var_10 = module_0.ExtensionLoaderMixin(context=var_9)
    var_11 = 'cookiecutter'
    var_12 = '_extensions'
    var_13 = 'invalid.extension'
    var_14 = [var_13]
    var_15 = {var_12: var_14}
    var_16 = {var_11: var_15}
    var_17 = module_0.ExtensionLoaderMixin(context=var_16)



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 'test.extensions.TestExtension'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = {var_1: var_7}
    var_9 = {}
    var_10 = 'invalid.extension.InvalidExtension'
    var_11 = [var_10]
    var_12 = {var_2: var_11}
    var_13 = {var_1: var_12}



# Parsed testcases at query #10
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'jinja2.ext.loopcontrols'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = module_0.ExtensionLoaderMixin(context=var_5)



# Parsed testcases at query #11
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'cookiecutter.extensions.JsonifyExtension'
    var_3 = 'cookiecutter.extensions.RandomStringExtension'
    var_4 = 'cookiecutter.extensions.SlugifyExtension'
    var_5 = 'cookiecutter.extensions.TimeExtension'
    var_6 = 'cookiecutter.extensions.UUIDExtension'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = {var_1: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.ExtensionLoaderMixin(context=var_9)



# Parsed testcases at query #12
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = {}
    var_3 = 'cookiecutter'
    var_4 = '_extensions'
    var_5 = 'extension1'
    var_6 = 'extension2'
    var_7 = [var_5, var_6]
    var_8 = {var_4: var_7}
    var_9 = {var_3: var_8}
    var_10 = module_0.ExtensionLoaderMixin(context=var_9)
    var_11 = {}
    var_12 = {var_3: var_11}
    var_13 = module_0.ExtensionLoaderMixin(context=var_12)



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 'custom.Extension'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = 'invalid.Extension'
    var_8 = [var_7]
    var_9 = {var_2: var_8}
    var_10 = {var_1: var_9}



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'test_extension'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 'jinja2.ext.i18n'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = 'nonexistent.extension'
    var_8 = [var_7]
    var_9 = {var_2: var_8}
    var_10 = {var_1: var_9}



# Parsed testcases at query #16
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = {}
    var_3 = module_0.ExtensionLoaderMixin(context=var_2)
    var_4 = 'cookiecutter'
    var_5 = '_extensions'
    var_6 = 'test.ext'
    var_7 = [var_6]
    var_8 = {var_5: var_7}
    var_9 = {var_4: var_8}
    var_10 = module_0.ExtensionLoaderMixin(context=var_9)
    var_11 = 'cookiecutter'
    var_12 = '_extensions'
    var_13 = 'nonexistent.extension'
    var_14 = [var_13]
    var_15 = {var_12: var_14}
    var_16 = {var_11: var_15}
    var_17 = module_0.ExtensionLoaderMixin(context=var_16)



# Parsed testcases at query #17
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = {}
    var_3 = module_0.ExtensionLoaderMixin(context=var_2)
    var_4 = 'cookiecutter'
    var_5 = '_extensions'
    var_6 = 'test_ext1'
    var_7 = 'test_ext2'
    var_8 = [var_6, var_7]
    var_9 = {var_5: var_8}
    var_10 = {var_4: var_9}
    var_11 = module_0.ExtensionLoaderMixin(context=var_10)
    var_12 = 'nonexistent.extension'
    var_13 = [var_12]
    var_14 = {var_5: var_13}
    var_15 = {var_4: var_14}
    var_16 = module_0.ExtensionLoaderMixin(context=var_15)



# Parsed testcases at query #18
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'cookiecutter.extensions.JsonifyExtension'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = module_0.ExtensionLoaderMixin(context=var_5)



# Parsed testcases at query #19
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = 'cookiecutter'
    var_3 = '_extensions'
    var_4 = 'some.extension'
    var_5 = [var_4]
    var_6 = {var_3: var_5}
    var_7 = {var_2: var_6}
    var_8 = module_0.ExtensionLoaderMixin(context=var_7)
    var_9 = 'nonexistent.extension'
    var_10 = [var_9]
    var_11 = {var_3: var_10}
    var_12 = {var_2: var_11}
    var_13 = module_0.ExtensionLoaderMixin(context=var_12)



# Parsed testcases at query #20
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = module_0.ExtensionLoaderMixin()
    var_1 = 'cookiecutter'
    var_2 = 'foo'
    var_3 = 'bar'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_0.ExtensionLoaderMixin(context=var_5)
    var_7 = '_extensions'
    var_8 = 'foo.bar'
    var_9 = [var_8]
    var_10 = {var_7: var_9}
    var_11 = {var_1: var_10}
    var_12 = module_0.ExtensionLoaderMixin(context=var_11)
    var_13 = 'cookiecutter'
    var_14 = '_extensions'
    var_15 = 'nonexistent.extension'
    var_16 = [var_15]
    var_17 = {var_14: var_16}
    var_18 = {var_13: var_17}
    var_19 = module_0.ExtensionLoaderMixin(context=var_18)



# Parsed testcases at query #21
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = 'cookiecutter'
    var_3 = '_extensions'
    var_4 = 'cookiecutter.extensions.JsonifyExtension'
    var_5 = [var_4]
    var_6 = {var_3: var_5}
    var_7 = {var_2: var_6}
    var_8 = module_0.ExtensionLoaderMixin(context=var_7)
    var_9 = 'invalid.extension'
    var_10 = [var_9]
    var_11 = {var_3: var_10}
    var_12 = {var_2: var_11}
    var_13 = module_0.ExtensionLoaderMixin(context=var_12)



# Parsed testcases at query #22
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'cookiecutter.extensions.TimeExtension'
    var_3 = 'cookiecutter.extensions.UUIDExtension'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = module_0.ExtensionLoaderMixin(context=var_6)
    var_8 = {}
    var_9 = module_0.ExtensionLoaderMixin(context=var_8)



# Parsed testcases at query #23
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = module_0.ExtensionLoaderMixin()
    var_1 = {}
    var_2 = module_0.ExtensionLoaderMixin(context=var_1)
    var_3 = 'cookiecutter'
    var_4 = '_extensions'
    var_5 = 'jinja2.ext.i18n'
    var_6 = [var_5]
    var_7 = {var_4: var_6}
    var_8 = {var_3: var_7}
    var_9 = module_0.ExtensionLoaderMixin(context=var_8)
    var_10 = 'nonexistent.extension'
    var_11 = [var_10]
    var_12 = {var_4: var_11}
    var_13 = {var_3: var_12}
    var_14 = module_0.ExtensionLoaderMixin(context=var_13)



# Parsed testcases at query #24
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'cookiecutter.extensions.RandomStringExtension'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = module_0.ExtensionLoaderMixin(context=var_5)



# Parsed testcases at query #25
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = module_0.ExtensionLoaderMixin()
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = module_0.ExtensionLoaderMixin(context=var_3)
    var_5 = '_extensions'
    var_6 = 'test_extension'
    var_7 = [var_6]
    var_8 = {var_5: var_7}
    var_9 = {var_1: var_8}
    var_10 = module_0.ExtensionLoaderMixin(context=var_9)



# Parsed testcases at query #26
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = {}
    var_3 = 'cookiecutter'
    var_4 = '_extensions'
    var_5 = 'test_extension1'
    var_6 = 'test_extension2'
    var_7 = [var_5, var_6]
    var_8 = {var_4: var_7}
    var_9 = {var_3: var_8}
    var_10 = module_0.ExtensionLoaderMixin(context=var_9)
    var_11 = 'cookiecutter'
    var_12 = '_extensions'
    var_13 = 'nonexistent_extension'
    var_14 = [var_13]
    var_15 = {var_12: var_14}
    var_16 = {var_11: var_15}
    var_17 = module_0.ExtensionLoaderMixin(context=var_16)



# Parsed testcases at query #27
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = 'cookiecutter'
    var_3 = '_extensions'
    var_4 = 'custom.Extension'
    var_5 = [var_4]
    var_6 = {var_3: var_5}
    var_7 = {var_2: var_6}
    var_8 = module_0.ExtensionLoaderMixin(context=var_7)
    var_9 = module_0.ExtensionLoaderMixin()
    var_10 = {}



# Parsed testcases at query #28
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = module_0.ExtensionLoaderMixin()
    var_1 = {}
    var_2 = module_0.ExtensionLoaderMixin(context=var_1)
    var_3 = 'cookiecutter'
    var_4 = '_extensions'
    var_5 = 'test.ext'
    var_6 = [var_5]
    var_7 = {var_4: var_6}
    var_8 = {var_3: var_7}
    var_9 = module_0.ExtensionLoaderMixin(context=var_8)
    var_10 = 'cookiecutter'
    var_11 = '_extensions'
    var_12 = 'nonexistent.extension'
    var_13 = [var_12]
    var_14 = {var_11: var_13}
    var_15 = {var_10: var_14}
    var_16 = module_0.ExtensionLoaderMixin(context=var_15)



# Parsed testcases at query #29
#--------------------------


import jinja2.environment as module_0
import cookiecutter.environment as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Environment()
    var_2 = module_1.ExtensionLoaderMixin(context=var_0)



# Parsed testcases at query #30
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = {}
    var_3 = module_0.ExtensionLoaderMixin(context=var_2)
    var_4 = 'cookiecutter'
    var_5 = '_extensions'
    var_6 = 'test.ext'
    var_7 = [var_6]
    var_8 = {var_5: var_7}
    var_9 = {var_4: var_8}
    var_10 = module_0.ExtensionLoaderMixin(context=var_9)
    var_11 = 'cookiecutter'
    var_12 = '_extensions'
    var_13 = 'invalid.ext'
    var_14 = [var_13]
    var_15 = {var_12: var_14}
    var_16 = {var_11: var_15}
    var_17 = module_0.ExtensionLoaderMixin(context=var_16)



# Parsed testcases at query #31
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = 'cookiecutter'
    var_3 = '_extensions'
    var_4 = 'some.extension'
    var_5 = [var_4]
    var_6 = {var_3: var_5}
    var_7 = {var_2: var_6}
    var_8 = module_0.ExtensionLoaderMixin(context=var_7)
    var_9 = 'cookiecutter'
    var_10 = '_extensions'
    var_11 = 'nonexistent.extension'
    var_12 = [var_11]
    var_13 = {var_10: var_12}
    var_14 = {var_9: var_13}
    var_15 = module_0.ExtensionLoaderMixin(context=var_14)



# Parsed testcases at query #32
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = '_read_extensions'
    var_3 = hasattr(var_1, var_2)
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = module_0.ExtensionLoaderMixin(context=var_6)
    var_8 = hasattr(var_7, var_2)
    var_9 = '_extensions'
    var_10 = 'test.ext1'
    var_11 = 'test.ext2'
    var_12 = [var_10, var_11]
    var_13 = {var_9: var_12}
    var_14 = {var_4: var_13}
    var_15 = module_0.ExtensionLoaderMixin(context=var_14)
    var_16 = hasattr(var_15, var_2)
    var_17 = 'nonexistent.extension'
    var_18 = [var_17]
    var_19 = {var_9: var_18}
    var_20 = {var_4: var_19}
    var_21 = module_0.ExtensionLoaderMixin(context=var_20)



# Parsed testcases at query #33
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = {}
    var_3 = module_0.ExtensionLoaderMixin(context=var_2)
    var_4 = 'cookiecutter'
    var_5 = '_extensions'
    var_6 = 'test.ext'
    var_7 = [var_6]
    var_8 = {var_5: var_7}
    var_9 = {var_4: var_8}
    var_10 = module_0.ExtensionLoaderMixin(context=var_9)
    var_11 = 'cookiecutter'
    var_12 = '_extensions'
    var_13 = 'invalid.ext'
    var_14 = [var_13]
    var_15 = {var_12: var_14}
    var_16 = {var_11: var_15}
    var_17 = module_0.ExtensionLoaderMixin(context=var_16)



# Parsed testcases at query #34
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = {}
    var_3 = module_0.ExtensionLoaderMixin(context=var_2)
    var_4 = 'cookiecutter'
    var_5 = '_extensions'
    var_6 = 'some.extension'
    var_7 = [var_6]
    var_8 = {var_5: var_7}
    var_9 = {var_4: var_8}
    var_10 = module_0.ExtensionLoaderMixin(context=var_9)
    var_11 = 'nonexistent.extension'
    var_12 = [var_11]
    var_13 = {var_5: var_12}
    var_14 = {var_4: var_13}
    var_15 = module_0.ExtensionLoaderMixin(context=var_14)



# Parsed testcases at query #35
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'some_extension'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = module_0.ExtensionLoaderMixin(context=var_5)



# Parsed testcases at query #36
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'some.extension'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = module_0.ExtensionLoaderMixin(context=var_5)



# Parsed testcases at query #37
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'test_extension'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = module_0.ExtensionLoaderMixin(context=var_5)



# Parsed testcases at query #38
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'cookiecutter.extensions.JsonifyExtension'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = module_0.ExtensionLoaderMixin(context=var_5)



# Parsed testcases at query #39
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = module_0.ExtensionLoaderMixin(context=var_2)
    var_4 = 'cookiecutter.extensions.TestExtension'
    var_5 = [var_4]
    var_6 = '_extensions'
    var_7 = {var_6: var_5}
    var_8 = {var_0: var_7}
    var_9 = module_0.ExtensionLoaderMixin(context=var_8)
    var_10 = 'cookiecutter'
    var_11 = '_extensions'
    var_12 = 'invalid.extension'
    var_13 = [var_12]
    var_14 = {var_11: var_13}
    var_15 = {var_10: var_14}
    var_16 = module_0.ExtensionLoaderMixin(context=var_15)



# Parsed testcases at query #40
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = {}
    var_3 = 'cookiecutter'
    var_4 = '_extensions'
    var_5 = 'ext1'
    var_6 = 'ext2'
    var_7 = [var_5, var_6]
    var_8 = {var_4: var_7}
    var_9 = {var_3: var_8}
    var_10 = module_0.ExtensionLoaderMixin(context=var_9)
    var_11 = {}
    var_12 = {var_3: var_11}
    var_13 = module_0.ExtensionLoaderMixin(context=var_12)



# Parsed testcases at query #41
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = {}
    var_3 = module_0.ExtensionLoaderMixin(context=var_2)
    var_4 = 'cookiecutter'
    var_5 = '_extensions'
    var_6 = 'test.ext'
    var_7 = [var_6]
    var_8 = {var_5: var_7}
    var_9 = {var_4: var_8}
    var_10 = module_0.ExtensionLoaderMixin(context=var_9)
    var_11 = 'cookiecutter'
    var_12 = '_extensions'
    var_13 = 'invalid.extension'
    var_14 = [var_13]
    var_15 = {var_12: var_14}
    var_16 = {var_11: var_15}
    var_17 = module_0.ExtensionLoaderMixin(context=var_16)



# Parsed testcases at query #42
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = {}
    var_3 = module_0.ExtensionLoaderMixin(context=var_2)
    var_4 = 'cookiecutter'
    var_5 = '_extensions'
    var_6 = 'extension1'
    var_7 = 'extension2'
    var_8 = [var_6, var_7]
    var_9 = {var_5: var_8}
    var_10 = {var_4: var_9}
    var_11 = module_0.ExtensionLoaderMixin(context=var_10)
    var_12 = {}
    var_13 = {var_4: var_12}
    var_14 = module_0.ExtensionLoaderMixin(context=var_13)
    var_15 = 'not a dictionary'
    var_16 = module_0.ExtensionLoaderMixin(context=var_15)



# Parsed testcases at query #43
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'cookiecutter.extensions.TestExtension'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = module_0.ExtensionLoaderMixin(context=var_5)



# Parsed testcases at query #44
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = {}
    var_3 = module_0.ExtensionLoaderMixin(context=var_2)
    var_4 = 'cookiecutter'
    var_5 = '_extensions'
    var_6 = 'test_extension1'
    var_7 = 'test_extension2'
    var_8 = [var_6, var_7]
    var_9 = {var_5: var_8}
    var_10 = {var_4: var_9}
    var_11 = module_0.ExtensionLoaderMixin(context=var_10)
    var_12 = 'nonexistent.extension'
    var_13 = [var_12]
    var_14 = {var_5: var_13}
    var_15 = {var_4: var_14}
    var_16 = module_0.ExtensionLoaderMixin(context=var_15)



# Parsed testcases at query #45
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'test_extension1'
    var_3 = 'test_extension2'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'arbitrary_key'
    var_8 = 'arbitrary_value'
    var_9 = {var_7: var_8}
    var_10 = module_0.ExtensionLoaderMixin(context=var_6, **var_9)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = module_0.ExtensionLoaderMixin()
    var_1 = '_read_extensions'
    var_2 = hasattr(var_0, var_1)
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = module_0.ExtensionLoaderMixin(context=var_5)
    var_7 = hasattr(var_6, var_1)
    var_8 = '_extensions'
    var_9 = 'test_extension'
    var_10 = [var_9]
    var_11 = {var_8: var_10}
    var_12 = {var_3: var_11}
    var_13 = module_0.ExtensionLoaderMixin(context=var_12)
    var_14 = hasattr(var_13, var_1)



# Parsed testcases at query #2
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = {}
    var_3 = module_0.ExtensionLoaderMixin(context=var_2)
    var_4 = 'cookiecutter'
    var_5 = '_extensions'
    var_6 = 'test.ext'
    var_7 = [var_6]
    var_8 = {var_5: var_7}
    var_9 = {var_4: var_8}
    var_10 = module_0.ExtensionLoaderMixin(context=var_9)
    var_11 = 'cookiecutter'
    var_12 = '_extensions'
    var_13 = 'invalid.extension'
    var_14 = [var_13]
    var_15 = {var_12: var_14}
    var_16 = {var_11: var_15}
    var_17 = module_0.ExtensionLoaderMixin(context=var_16)



# Parsed testcases at query #3
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'cookiecutter.extensions.JsonifyExtension'
    var_3 = 'cookiecutter.extensions.RandomStringExtension'
    var_4 = 'cookiecutter.extensions.SlugifyExtension'
    var_5 = 'cookiecutter.extensions.TimeExtension'
    var_6 = 'cookiecutter.extensions.UUIDExtension'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = {var_1: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.StrictEnvironment()



# Parsed testcases at query #4
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = '_read_extensions'
    var_3 = hasattr(var_1, var_2)
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = module_0.ExtensionLoaderMixin(context=var_6)
    var_8 = hasattr(var_7, var_2)
    var_9 = '_extensions'
    var_10 = 'test.ext'
    var_11 = [var_10]
    var_12 = {var_9: var_11}
    var_13 = {var_4: var_12}
    var_14 = module_0.ExtensionLoaderMixin(context=var_13)
    var_15 = hasattr(var_14, var_2)
    var_16 = 'cookiecutter'
    var_17 = '_extensions'
    var_18 = 'nonexistent.extension'
    var_19 = [var_18]
    var_20 = {var_17: var_19}
    var_21 = {var_16: var_20}
    var_22 = module_0.ExtensionLoaderMixin(context=var_21)



# Parsed testcases at query #5
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'some.extension'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = module_0.ExtensionLoaderMixin(context=var_5)
    var_7 = {}
    var_8 = {var_0: var_7}
    var_9 = module_0.ExtensionLoaderMixin(context=var_8)
    var_10 = 'invalid.extension'
    var_11 = [var_10]
    var_12 = {var_1: var_11}
    var_13 = {var_0: var_12}
    var_14 = module_0.ExtensionLoaderMixin(context=var_13)



# Parsed testcases at query #6
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'jinja2.ext.i18n'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = module_0.ExtensionLoaderMixin(context=var_5)



# Parsed testcases at query #7
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = {}
    var_3 = module_0.ExtensionLoaderMixin(context=var_2)
    var_4 = 'cookiecutter'
    var_5 = '_extensions'
    var_6 = 'some.extension'
    var_7 = [var_6]
    var_8 = {var_5: var_7}
    var_9 = {var_4: var_8}
    var_10 = module_0.ExtensionLoaderMixin(context=var_9)
    var_11 = 'nonexistent.extension'
    var_12 = [var_11]
    var_13 = {var_5: var_12}
    var_14 = {var_4: var_13}
    var_15 = module_0.ExtensionLoaderMixin(context=var_14)



# Parsed testcases at query #8
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'some_extension'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = module_0.ExtensionLoaderMixin(context=var_5)



# Parsed testcases at query #9
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = {}
    var_3 = module_0.ExtensionLoaderMixin(context=var_2)
    var_4 = 'cookiecutter'
    var_5 = '_extensions'
    var_6 = 'test_ext1'
    var_7 = 'test_ext2'
    var_8 = [var_6, var_7]
    var_9 = {var_5: var_8}
    var_10 = {var_4: var_9}
    var_11 = module_0.ExtensionLoaderMixin(context=var_10)
    var_12 = 'invalid_ext'
    var_13 = [var_12]
    var_14 = {var_5: var_13}
    var_15 = {var_4: var_14}
    var_16 = module_0.ExtensionLoaderMixin(context=var_15)



# Parsed testcases at query #10
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = module_0.ExtensionLoaderMixin()
    var_1 = '_read_extensions'
    var_2 = hasattr(var_0, var_1)
    var_3 = {}
    var_4 = module_0.ExtensionLoaderMixin(context=var_3)
    var_5 = hasattr(var_4, var_1)
    var_6 = 'cookiecutter'
    var_7 = '_extensions'
    var_8 = 'test.ext'
    var_9 = [var_8]
    var_10 = {var_7: var_9}
    var_11 = {var_6: var_10}
    var_12 = module_0.ExtensionLoaderMixin(context=var_11)
    var_13 = hasattr(var_12, var_1)
    var_14 = 'cookiecutter'
    var_15 = '_extensions'
    var_16 = 'invalid.ext'
    var_17 = [var_16]
    var_18 = {var_15: var_17}
    var_19 = {var_14: var_18}
    var_20 = module_0.ExtensionLoaderMixin(context=var_19)



# Parsed testcases at query #11
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = {}
    var_3 = module_0.ExtensionLoaderMixin(context=var_2)
    var_4 = 'cookiecutter'
    var_5 = '_extensions'
    var_6 = 'extension1'
    var_7 = 'extension2'
    var_8 = [var_6, var_7]
    var_9 = {var_5: var_8}
    var_10 = {var_4: var_9}
    var_11 = module_0.ExtensionLoaderMixin(context=var_10)
    var_12 = {}
    var_13 = {var_4: var_12}
    var_14 = module_0.ExtensionLoaderMixin(context=var_13)
    var_15 = 'invalid'
    var_16 = {var_5: var_15}
    var_17 = {var_4: var_16}
    var_18 = module_0.ExtensionLoaderMixin(context=var_17)



# Parsed testcases at query #12
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = '\n    Test that extensions are properly loaded.\n\n    '
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 'jinja2.ext.autoescape'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.ExtensionLoaderMixin(context=var_6)
    var_8 = {}
    var_9 = {var_1: var_8}
    var_10 = module_0.ExtensionLoaderMixin(context=var_9)
    var_11 = 'invalid.extension'
    var_12 = [var_11]
    var_13 = {var_2: var_12}
    var_14 = {var_1: var_13}
    var_15 = module_0.ExtensionLoaderMixin(context=var_14)
    var_16 = module_0.ExtensionLoaderMixin()



# Parsed testcases at query #13
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = 'cookiecutter'
    var_3 = '_extensions'
    var_4 = 'custom.Extension1'
    var_5 = 'custom.Extension2'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = module_0.ExtensionLoaderMixin(context=var_8)
    var_10 = 'invalid.Extension'
    var_11 = [var_10]
    var_12 = {var_3: var_11}
    var_13 = {var_2: var_12}
    var_14 = module_0.ExtensionLoaderMixin(context=var_13)



# Parsed testcases at query #14
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = {}
    var_3 = module_0.ExtensionLoaderMixin(context=var_2)
    var_4 = 'cookiecutter'
    var_5 = '_extensions'
    var_6 = 'test.extensions.TestExtension'
    var_7 = [var_6]
    var_8 = {var_5: var_7}
    var_9 = {var_4: var_8}
    var_10 = module_0.ExtensionLoaderMixin(context=var_9)
    var_11 = 'invalid.extension.InvalidExtension'
    var_12 = [var_11]
    var_13 = {var_5: var_12}
    var_14 = {var_4: var_13}
    var_15 = module_0.ExtensionLoaderMixin(context=var_14)



# Parsed testcases at query #15
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = {}
    var_3 = 'cookiecutter'
    var_4 = '_extensions'
    var_5 = 'test_extension1'
    var_6 = 'test_extension2'
    var_7 = [var_5, var_6]
    var_8 = {var_4: var_7}
    var_9 = {var_3: var_8}
    var_10 = module_0.ExtensionLoaderMixin(context=var_9)
    var_11 = 'other_key'
    var_12 = 'value'
    var_13 = {var_11: var_12}
    var_14 = module_0.ExtensionLoaderMixin(context=var_13)
    var_15 = {var_11: var_12}
    var_16 = {var_3: var_15}
    var_17 = module_0.ExtensionLoaderMixin(context=var_16)



# Parsed testcases at query #16
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = module_0.ExtensionLoaderMixin()
    var_1 = {}
    var_2 = module_0.ExtensionLoaderMixin(context=var_1)
    var_3 = 'cookiecutter'
    var_4 = '_extensions'
    var_5 = 'test.ext'
    var_6 = [var_5]
    var_7 = {var_4: var_6}
    var_8 = {var_3: var_7}
    var_9 = module_0.ExtensionLoaderMixin(context=var_8)
    var_10 = 'cookiecutter'
    var_11 = '_extensions'
    var_12 = 'nonexistent.extension'
    var_13 = [var_12]
    var_14 = {var_11: var_13}
    var_15 = {var_10: var_14}
    var_16 = module_0.ExtensionLoaderMixin(context=var_15)



# Parsed testcases at query #17
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'cookiecutter.extensions.SlugifyExtension'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = module_0.ExtensionLoaderMixin(context=var_5)



# Parsed testcases at query #18
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = module_0.ExtensionLoaderMixin()
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = module_0.ExtensionLoaderMixin(context=var_3)
    var_5 = '_extensions'
    var_6 = 'some.extension'
    var_7 = [var_6]
    var_8 = {var_5: var_7}
    var_9 = {var_1: var_8}
    var_10 = module_0.ExtensionLoaderMixin(context=var_9)
    var_11 = 'nonexistent.extension'
    var_12 = [var_11]
    var_13 = {var_5: var_12}
    var_14 = {var_1: var_13}
    var_15 = module_0.ExtensionLoaderMixin(context=var_14)



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'Unit test for ExtensionLoaderMixin class constructor.'
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 'some_extension'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = 'cookiecutter'
    var_9 = '_extensions'
    var_10 = 'invalid_extension'
    var_11 = [var_10]
    var_12 = {var_9: var_11}
    var_13 = {var_8: var_12}



# Parsed testcases at query #20
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = 'cookiecutter'
    var_3 = '_extensions'
    var_4 = 'jinja2.ext.loopcontrols'
    var_5 = [var_4]
    var_6 = {var_3: var_5}
    var_7 = {var_2: var_6}
    var_8 = module_0.ExtensionLoaderMixin(context=var_7)
    var_9 = 'invalid.extension'
    var_10 = [var_9]
    var_11 = {var_3: var_10}
    var_12 = {var_2: var_11}
    var_13 = module_0.ExtensionLoaderMixin(context=var_12)



# Parsed testcases at query #21
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = module_0.ExtensionLoaderMixin()
    var_1 = {}
    var_2 = 'cookiecutter'
    var_3 = '_extensions'
    var_4 = []
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.ExtensionLoaderMixin(context=var_6)
    var_8 = 'extension1'
    var_9 = 'extension2'
    var_10 = [var_8, var_9]
    var_11 = {var_3: var_10}
    var_12 = {var_2: var_11}
    var_13 = module_0.ExtensionLoaderMixin(context=var_12)
    var_14 = {}
    var_15 = {var_2: var_14}
    var_16 = module_0.ExtensionLoaderMixin(context=var_15)
    var_17 = {}
    var_18 = module_0.ExtensionLoaderMixin(context=var_17)



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'jinja2.ext.i18n'
    var_3 = 'jinja2.ext.do'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'cookiecutter.extensions.JsonifyExtension'
    var_8 = 'cookiecutter.extensions.RandomStringExtension'
    var_9 = 'cookiecutter.extensions.SlugifyExtension'
    var_10 = 'cookiecutter.extensions.TimeExtension'
    var_11 = 'cookiecutter.extensions.UUIDExtension'
    var_12 = [var_7, var_8, var_9, var_10, var_11, var_2, var_3]
    var_13 = {}



# Parsed testcases at query #23
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = {}
    var_3 = module_0.ExtensionLoaderMixin(context=var_2)
    var_4 = 'cookiecutter'
    var_5 = '_extensions'
    var_6 = 'test.ext'
    var_7 = [var_6]
    var_8 = {var_5: var_7}
    var_9 = {var_4: var_8}
    var_10 = module_0.ExtensionLoaderMixin(context=var_9)
    var_11 = 'cookiecutter'
    var_12 = '_extensions'
    var_13 = 'nonexistent.extension'
    var_14 = [var_13]
    var_15 = {var_12: var_14}
    var_16 = {var_11: var_15}
    var_17 = module_0.ExtensionLoaderMixin(context=var_16)



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    pass



# Parsed testcases at query #25
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = {}
    var_3 = module_0.ExtensionLoaderMixin(context=var_2)
    var_4 = 'cookiecutter'
    var_5 = '_extensions'
    var_6 = 'test_extension'
    var_7 = [var_6]
    var_8 = {var_5: var_7}
    var_9 = {var_4: var_8}
    var_10 = module_0.ExtensionLoaderMixin(context=var_9)
    var_11 = 'cookiecutter'
    var_12 = '_extensions'
    var_13 = 'nonexistent_extension'
    var_14 = [var_13]
    var_15 = {var_12: var_14}
    var_16 = {var_11: var_15}
    var_17 = module_0.ExtensionLoaderMixin(context=var_16)



# Parsed testcases at query #26
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = 'cookiecutter'
    var_3 = '_extensions'
    var_4 = 'cookiecutter.extensions.JsonifyExtension'
    var_5 = [var_4]
    var_6 = {var_3: var_5}
    var_7 = {var_2: var_6}
    var_8 = module_0.ExtensionLoaderMixin(context=var_7)
    var_9 = {}
    var_10 = {var_2: var_9}
    var_11 = module_0.ExtensionLoaderMixin(context=var_10)



# Parsed testcases at query #27
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'cookiecutter.extensions.JsonifyExtension'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = module_0.ExtensionLoaderMixin(context=var_5)



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'Test ExtensionLoaderMixin constructor.'
    var_1 = {}



# Parsed testcases at query #29
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'some_extension'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = module_0.ExtensionLoaderMixin(context=var_5)



# Parsed testcases at query #30
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = 'cookiecutter'
    var_3 = '_extensions'
    var_4 = 'some_extension'
    var_5 = [var_4]
    var_6 = {var_3: var_5}
    var_7 = {var_2: var_6}
    var_8 = module_0.ExtensionLoaderMixin(context=var_7)
    var_9 = 'invalid_extension'
    var_10 = [var_9]
    var_11 = {var_3: var_10}
    var_12 = {var_2: var_11}
    var_13 = module_0.ExtensionLoaderMixin(context=var_12)



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'some_extension.Extension'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = 'Unit test for constructor of class ExtensionLoaderMixin.'
    var_1 = {}
    var_2 = 'cookiecutter'
    var_3 = '_extensions'
    var_4 = 'test_ext'
    var_5 = [var_4]
    var_6 = {var_3: var_5}
    var_7 = {var_2: var_6}



# Parsed testcases at query #33
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = {}
    var_3 = {}
    var_4 = module_0.ExtensionLoaderMixin(context=var_3)
    var_5 = {}
    var_6 = 'cookiecutter'
    var_7 = 'other_key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = module_0.ExtensionLoaderMixin(context=var_10)
    var_12 = 'ext1'
    var_13 = 'ext2'
    var_14 = [var_12, var_13]
    var_15 = '_extensions'
    var_16 = {var_15: var_14}
    var_17 = {var_6: var_16}
    var_18 = module_0.ExtensionLoaderMixin(context=var_17)
    var_19 = 'invalid'
    var_20 = {var_6: var_19}
    var_21 = module_0.ExtensionLoaderMixin(context=var_20)



# Parsed testcases at query #34
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = module_0.ExtensionLoaderMixin()
    var_1 = {}
    var_2 = module_0.ExtensionLoaderMixin(context=var_1)
    var_3 = 'cookiecutter'
    var_4 = '_extensions'
    var_5 = 'test.ext'
    var_6 = [var_5]
    var_7 = {var_4: var_6}
    var_8 = {var_3: var_7}
    var_9 = module_0.ExtensionLoaderMixin(context=var_8)
    var_10 = 'cookiecutter'
    var_11 = '_extensions'
    var_12 = 'invalid.extension'
    var_13 = [var_12]
    var_14 = {var_11: var_13}
    var_15 = {var_10: var_14}
    var_16 = module_0.ExtensionLoaderMixin(context=var_15)



# Parsed testcases at query #35
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test constructor of ExtensionLoaderMixin.'
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 'cookiecutter.extensions.JsonifyExtension'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.ExtensionLoaderMixin(context=var_6)



# Parsed testcases at query #36
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = {}
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = module_0.ExtensionLoaderMixin(context=var_5)
    var_7 = {}
    var_8 = {var_3: var_7}
    var_9 = '_extensions'
    var_10 = 'ext1'
    var_11 = 'ext2'
    var_12 = [var_10, var_11]
    var_13 = {var_9: var_12}
    var_14 = {var_3: var_13}
    var_15 = module_0.ExtensionLoaderMixin(context=var_14)
    var_16 = 'cookiecutter'
    var_17 = '_extensions'
    var_18 = 'invalid.ext'
    var_19 = [var_18]
    var_20 = {var_17: var_19}
    var_21 = {var_16: var_20}
    var_22 = module_0.ExtensionLoaderMixin(context=var_21)



# Parsed testcases at query #37
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = {}
    var_3 = module_0.ExtensionLoaderMixin(context=var_2)
    var_4 = 'cookiecutter'
    var_5 = '_extensions'
    var_6 = 'test.extensions.TestExtension'
    var_7 = [var_6]
    var_8 = {var_5: var_7}
    var_9 = {var_4: var_8}
    var_10 = module_0.ExtensionLoaderMixin(context=var_9)
    var_11 = 'nonexistent.extension'
    var_12 = [var_11]
    var_13 = {var_5: var_12}
    var_14 = {var_4: var_13}
    var_15 = module_0.ExtensionLoaderMixin(context=var_14)



# Parsed testcases at query #38
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'some_extension'
    var_3 = 'another_extension'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = module_0.ExtensionLoaderMixin(context=var_6)
    var_8 = {}
    var_9 = module_0.ExtensionLoaderMixin(context=var_8)
    var_10 = {}
    var_11 = {var_0: var_10}
    var_12 = module_0.ExtensionLoaderMixin(context=var_11)
    var_13 = 'test_extension'
    var_14 = [var_13]
    var_15 = {var_1: var_14}
    var_16 = {var_0: var_15}
    var_17 = module_0.ExtensionLoaderMixin(context=var_16)



# Parsed testcases at query #39
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = {}
    var_3 = module_0.ExtensionLoaderMixin(context=var_2)
    var_4 = 'cookiecutter'
    var_5 = '_extensions'
    var_6 = 'test.extensions.TestExtension'
    var_7 = [var_6]
    var_8 = {var_5: var_7}
    var_9 = {var_4: var_8}
    var_10 = module_0.ExtensionLoaderMixin(context=var_9)
    var_11 = 'cookiecutter'
    var_12 = '_extensions'
    var_13 = 'nonexistent.extension'
    var_14 = [var_13]
    var_15 = {var_12: var_14}
    var_16 = {var_11: var_15}
    var_17 = module_0.ExtensionLoaderMixin(context=var_16)



# Parsed testcases at query #40
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter.extensions.JsonifyExtension'
    var_1 = 'cookiecutter.extensions.RandomStringExtension'
    var_2 = 'cookiecutter.extensions.SlugifyExtension'
    var_3 = 'cookiecutter.extensions.TimeExtension'
    var_4 = 'cookiecutter.extensions.UUIDExtension'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = {}
    var_7 = module_0.ExtensionLoaderMixin(context=var_6)
    var_8 = 'custom_extension1'
    var_9 = 'custom_extension2'
    var_10 = [var_8, var_9]
    var_11 = 'cookiecutter'
    var_12 = '_extensions'
    var_13 = {var_12: var_10}
    var_14 = {var_11: var_13}
    var_15 = module_0.ExtensionLoaderMixin(context=var_14)



# Parsed testcases at query #41
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'cookiecutter.extensions.JsonifyExtension'
    var_3 = 'cookiecutter.extensions.RandomStringExtension'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = module_0.ExtensionLoaderMixin(context=var_6)



# Parsed testcases at query #42
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = module_0.ExtensionLoaderMixin()
    var_1 = {}
    var_2 = module_0.ExtensionLoaderMixin(context=var_1)
    var_3 = 'cookiecutter'
    var_4 = '_extensions'
    var_5 = 'test.ext'
    var_6 = [var_5]
    var_7 = {var_4: var_6}
    var_8 = {var_3: var_7}
    var_9 = module_0.ExtensionLoaderMixin(context=var_8)
    var_10 = 'cookiecutter'
    var_11 = '_extensions'
    var_12 = 'nonexistent.extension'
    var_13 = [var_12]
    var_14 = {var_11: var_13}
    var_15 = {var_10: var_14}
    var_16 = module_0.ExtensionLoaderMixin(context=var_15)



# Parsed testcases at query #43
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = 'cookiecutter'
    var_3 = '_extensions'
    var_4 = 'ext1'
    var_5 = 'ext2'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = module_0.ExtensionLoaderMixin(context=var_8)
    var_10 = {}
    var_11 = {var_2: var_10}
    var_12 = module_0.ExtensionLoaderMixin(context=var_11)



# Parsed testcases at query #44
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'cookiecutter.extensions.JsonifyExtension'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = module_0.ExtensionLoaderMixin(context=var_5)



# Parsed testcases at query #45
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.ExtensionLoaderMixin(context=var_0)
    var_2 = 'cookiecutter'
    var_3 = '_extensions'
    var_4 = 'test.ext1'
    var_5 = 'test.ext2'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = module_0.ExtensionLoaderMixin(context=var_8)
    var_10 = 'nonexistent.extension'
    var_11 = [var_10]
    var_12 = {var_3: var_11}
    var_13 = {var_2: var_12}
    var_14 = module_0.ExtensionLoaderMixin(context=var_13)



