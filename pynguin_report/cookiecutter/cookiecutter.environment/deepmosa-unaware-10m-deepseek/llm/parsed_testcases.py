####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'cookiecutter'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '_extensions'
    var_7 = 'my_ext.Extension1'
    var_8 = 'other_ext.Extension2'
    var_9 = [var_7, var_8]
    var_10 = {var_6: var_9}
    var_11 = {var_1: var_10}
    var_12 = 123
    var_13 = 456.789
    var_14 = [var_12, var_13]
    var_15 = {var_6: var_14}
    var_16 = {var_1: var_15}
    var_17 = {}
    var_18 = True
    var_19 = {}



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'cookiecutter'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'my_ext.Extension1'
    var_7 = 'other_ext.Extension2'
    var_8 = [var_6, var_7]
    var_9 = '_extensions'
    var_10 = {var_9: var_8}
    var_11 = {var_1: var_10}
    var_12 = 'ext1'
    var_13 = 123
    var_14 = True
    var_15 = [var_12, var_13, var_14]
    var_16 = {var_9: var_15}
    var_17 = {var_1: var_16}
    var_18 = {}
    var_19 = 42
    var_20 = {}



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = len(var_0)
    assert var_1 == 5
    var_2 = {}
    var_3 = len(var_1)
    assert var_3 == 5
    var_4 = 'cookiecutter'
    var_5 = '_extensions'
    var_6 = 'my_ext.Extension1'
    var_7 = 'other_ext.Extension2'
    var_8 = [var_6, var_7]
    var_9 = {var_5: var_8}
    var_10 = {var_4: var_9}
    var_11 = 'cookiecutter'
    var_12 = '_extensions'
    var_13 = 'my_ext.Extension1'
    var_14 = {var_12: var_13}
    var_15 = {var_11: var_14}
    var_16 = len(var_8)
    assert var_16 == 6
    var_17 = str(var_13)



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = len(var_0)
    assert var_1 == 5
    var_2 = {}
    var_3 = len(var_1)
    assert var_3 == 5
    var_4 = 'cookiecutter'
    var_5 = '_extensions'
    var_6 = 'myextension.Extension1'
    var_7 = 'myextension.Extension2'
    var_8 = [var_6, var_7]
    var_9 = {var_5: var_8}
    var_10 = {var_4: var_9}
    var_11 = 'cookiecutter'
    var_12 = 'other_key'
    var_13 = 'value'
    var_14 = {var_12: var_13}
    assert var_14 is True
    var_15 = {var_11: var_14}
    var_16 = len(var_8)
    assert var_16 == 5
    assert var_16 is True
    var_17 = {}
    var_18 = True
    var_19 = 'trim_blocks'
    var_20 = 'lstrip_blocks'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 'myextension.Extension1'
    var_4 = 'anotherext.Extension2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = 'other_key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = {var_1: var_10}
    var_12 = {}
    var_13 = 123
    var_14 = None
    var_15 = {}



# Parsed testcases at query #6
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'cookiecutter.extensions.JsonifyExtension'
    var_2 = 'cookiecutter.extensions.RandomStringExtension'
    var_3 = 'cookiecutter.extensions.SlugifyExtension'
    var_4 = 'cookiecutter.extensions.TimeExtension'
    var_5 = 'cookiecutter.extensions.UUIDExtension'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = 'myapp.extensions.CustomExtension'
    var_8 = 'another.Extension'
    var_9 = [var_7, var_8]
    var_10 = 'cookiecutter'
    var_11 = '_extensions'
    var_12 = {var_11: var_9}
    var_13 = {var_10: var_12}
    var_14 = []
    var_15 = {var_11: var_14}
    var_16 = {var_10: var_15}
    var_17 = 'other_key'
    var_18 = 'value'
    var_19 = {var_17: var_18}
    var_20 = {var_10: var_19}
    var_21 = {var_17: var_18}
    var_22 = 'ext1'
    var_23 = 123
    var_24 = 'ext2'
    var_25 = [var_22, var_23, var_24]
    var_26 = {var_11: var_25}
    var_27 = {var_10: var_26}
    var_28 = {}
    var_29 = True
    var_30 = None
    var_31 = 'custom.Extension'
    var_32 = [var_31]
    var_33 = {var_11: var_32}
    var_34 = {var_10: var_33}
    var_35 = module_0.StrictEnvironment()



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 'myextension.Extension1'
    var_4 = 'anotherextension.Extension2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = 'other_key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = {var_1: var_10}
    var_12 = {var_8: var_9}
    var_13 = 123



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = 'cookiecutter'
    var_3 = '_extensions'
    var_4 = 'my_ext.Extension1'
    var_5 = 'other_ext.Extension2'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'cookiecutter'
    var_10 = {}
    var_11 = {var_9: var_10}
    var_12 = len(var_4)
    assert var_12 == 5
    var_13 = 'other_key'
    var_14 = 'value'
    var_15 = {var_13: var_14}
    var_16 = len(var_4)
    assert var_16 == 5
    var_17 = {}
    var_18 = True
    var_19 = {}
    var_20 = str(var_16)



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'my_ext.Extension1'
    var_7 = 'other_ext.Extension2'
    var_8 = [var_6, var_7]
    var_9 = '_extensions'
    var_10 = {var_9: var_8}
    var_11 = {var_1: var_10}
    var_12 = 'ext.Ext'
    var_13 = [var_12]
    var_14 = {var_9: var_13}
    var_15 = {var_1: var_14}
    var_16 = 123
    var_17 = True
    var_18 = [var_16, var_17]
    var_19 = {var_9: var_18}
    var_20 = {var_1: var_19}
    var_21 = 'cookiecutter'
    var_22 = '_extensions'
    var_23 = 'bad.ext'
    var_24 = [var_23]
    var_25 = {var_22: var_24}
    var_26 = {var_21: var_25}



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'extensions'
    var_1 = {}
    var_2 = 'cookiecutter'
    var_3 = '_extensions'
    var_4 = 'my_ext.Extension1'
    var_5 = 'other_ext.Extension2'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'other_key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = {var_2: var_11}
    var_13 = {var_9: var_10}
    var_14 = 123
    var_15 = 456.789
    var_16 = [var_14, var_15]
    var_17 = {var_3: var_16}
    var_18 = {var_2: var_17}
    var_19 = {}



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 'my_ext.Extension1'
    var_4 = 'other_ext.Extension2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = 'other_key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = {var_1: var_10}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = True
    var_15 = 'trim_blocks'
    var_16 = 'lstrip_blocks'



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'cookiecutter'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'my_ext.Extension1'
    var_7 = 'my_ext.Extension2'
    var_8 = [var_6, var_7]
    var_9 = '_extensions'
    var_10 = {var_9: var_8}
    var_11 = {var_1: var_10}
    var_12 = 'ext1'
    var_13 = 123
    var_14 = True
    var_15 = [var_12, var_13, var_14]
    var_16 = {var_9: var_15}
    var_17 = {var_1: var_16}
    var_18 = {}
    var_19 = 42
    var_20 = {}



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = 'cookiecutter'
    var_3 = '_extensions'
    var_4 = 'my.custom.Extension1'
    var_5 = 'my.custom.Extension2'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = {}
    var_10 = {var_2: var_9}
    var_11 = 'other_key'
    var_12 = 'value'
    var_13 = {var_11: var_12}
    var_14 = {}
    var_15 = 'extensions'
    var_16 = {}
    var_17 = 'my.custom.Extension'
    var_18 = {var_3: var_17}
    var_19 = {var_2: var_18}
    var_20 = 'ext1'
    var_21 = 123
    var_22 = 'ext2'
    var_23 = [var_20, var_21, var_22]
    var_24 = {var_3: var_23}
    var_25 = {var_2: var_24}



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 'cookiecutter.extensions.JsonifyExtension'
    var_2 = 'cookiecutter.extensions.RandomStringExtension'
    var_3 = 'cookiecutter.extensions.SlugifyExtension'
    var_4 = 'cookiecutter.extensions.TimeExtension'
    var_5 = 'cookiecutter.extensions.UUIDExtension'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = {}
    var_8 = 'cookiecutter'
    var_9 = '_extensions'
    var_10 = 'my.custom.Extension1'
    var_11 = 'my.custom.Extension2'
    var_12 = [var_10, var_11]
    var_13 = {var_9: var_12}
    var_14 = {var_8: var_13}
    var_15 = [var_10, var_11]
    var_16 = var_6 + var_15
    var_17 = {}
    var_18 = 'value'
    var_19 = 123
    var_20 = 'other_key'
    var_21 = 'value'
    var_22 = {var_20: var_21}
    var_23 = 'cookiecutter'
    var_24 = 'other_key'
    var_25 = 'value'
    var_26 = {var_24: var_25}
    var_27 = {var_23: var_26}
    var_28 = str(var_23)



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 'myextension.Extension1'
    var_4 = 'anotherextension.Extension2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {}
    var_9 = {var_1: var_8}
    var_10 = 'other_key'
    var_11 = 'value'
    var_12 = {var_10: var_11}
    var_13 = 123
    var_14 = 'test.Extension'
    var_15 = [var_13, var_14]
    var_16 = {var_2: var_15}
    var_17 = {var_1: var_16}
    var_18 = {}
    var_19 = 42
    var_20 = 'test_arg'
    var_21 = 'another_arg'



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'extensions'
    var_1 = {}
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'cookiecutter'
    var_6 = 'name'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'my_ext.Extension1'
    var_11 = 'my_ext.Extension2'
    var_12 = [var_10, var_11]
    var_13 = '_extensions'
    var_14 = {var_13: var_12, var_6: var_7}
    var_15 = {var_5: var_14}
    var_16 = 'cookiecutter.extensions.JsonifyExtension'
    var_17 = 'cookiecutter.extensions.RandomStringExtension'
    var_18 = 'cookiecutter.extensions.SlugifyExtension'
    var_19 = 'cookiecutter.extensions.TimeExtension'
    var_20 = 'cookiecutter.extensions.UUIDExtension'
    var_21 = [var_16, var_17, var_18, var_19, var_20]
    var_22 = None
    var_23 = 123
    var_24 = 456.789
    var_25 = [var_23, var_24]
    var_26 = {var_13: var_25}
    var_27 = {var_5: var_26}
    var_28 = False
    var_29 = True



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'my_ext.Extension1'
    var_7 = 'other_ext.Extension2'
    var_8 = [var_6, var_7]
    var_9 = '_extensions'
    var_10 = {var_9: var_8}
    var_11 = {var_1: var_10}
    var_12 = 'ext1'
    var_13 = 123
    var_14 = True
    var_15 = [var_12, var_13, var_14]
    var_16 = {var_9: var_15}
    var_17 = {var_1: var_16}
    var_18 = {}
    var_19 = {}



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'other_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = '_extensions'
    var_8 = 'myextension.Extension1'
    var_9 = 'anotherext.Extension2'
    var_10 = [var_8, var_9]
    var_11 = {var_7: var_10}
    var_12 = {var_4: var_11}
    var_13 = 123
    var_14 = 456.789
    var_15 = [var_13, var_14]
    var_16 = {var_7: var_15}
    var_17 = {var_4: var_16}
    var_18 = {}
    var_19 = {}
    var_20 = True



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter.extensions.JsonifyExtension'
    var_1 = 'cookiecutter.extensions.RandomStringExtension'
    var_2 = 'cookiecutter.extensions.SlugifyExtension'
    var_3 = 'cookiecutter.extensions.TimeExtension'
    var_4 = 'cookiecutter.extensions.UUIDExtension'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = 'cookiecutter'
    var_7 = '_extensions'
    var_8 = 'my_ext.Extension1'
    var_9 = 'other_ext.Extension2'
    var_10 = [var_8, var_9]
    var_11 = {var_7: var_10}
    var_12 = {var_6: var_11}
    var_13 = {}
    var_14 = 'other_key'
    var_15 = 'value'
    var_16 = {var_14: var_15}
    var_17 = 'key'
    var_18 = {var_17: var_15}
    var_19 = {var_6: var_18}
    var_20 = True



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'cookiecutter'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'my_ext.Extension1'
    var_7 = 'my_ext.Extension2'
    var_8 = [var_6, var_7]
    var_9 = '_extensions'
    var_10 = {var_9: var_8}
    var_11 = {var_1: var_10}
    var_12 = 'ext1'
    var_13 = 123
    var_14 = True
    var_15 = [var_12, var_13, var_14]
    var_16 = {var_9: var_15}
    var_17 = {var_1: var_16}
    var_18 = {}
    var_19 = {}



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'Test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '_extensions'
    var_7 = 'my_ext.Extension1'
    var_8 = 'other.Extension2'
    var_9 = [var_7, var_8]
    var_10 = {var_6: var_9}
    var_11 = {var_1: var_10}
    var_12 = 123
    var_13 = True
    var_14 = 'my_ext.Extension'
    var_15 = [var_12, var_13, var_14]
    var_16 = {var_6: var_15}
    var_17 = {var_1: var_16}
    var_18 = {}
    var_19 = {}



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = 'cookiecutter.extensions.JsonifyExtension'
    var_2 = 'cookiecutter.extensions.RandomStringExtension'
    var_3 = 'cookiecutter.extensions.SlugifyExtension'
    var_4 = 'cookiecutter.extensions.TimeExtension'
    var_5 = 'cookiecutter.extensions.UUIDExtension'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = 'cookiecutter'
    var_8 = '_extensions'
    var_9 = 'my_ext.Extension1'
    var_10 = 'other_ext.Extension2'
    var_11 = [var_9, var_10]
    var_12 = {var_8: var_11}
    var_13 = {var_7: var_12}
    var_14 = 'value'
    var_15 = 'cookiecutter.extensions.JsonifyExtension'
    var_16 = 'cookiecutter.extensions.RandomStringExtension'
    var_17 = 'cookiecutter.extensions.SlugifyExtension'
    var_18 = 'cookiecutter.extensions.TimeExtension'
    var_19 = 'cookiecutter.extensions.UUIDExtension'
    var_20 = [var_15, var_16, var_17, var_18, var_19, var_9, var_10]
    var_21 = 'cookiecutter'
    var_22 = 'other_key'
    var_23 = 'value'
    var_24 = {var_22: var_23}
    var_25 = {var_21: var_24}
    var_26 = {}
    var_27 = isinstance(var_26, var_22)
    var_28 = str(var_24)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter.extensions.JsonifyExtension'
    var_1 = 'cookiecutter.extensions.RandomStringExtension'
    var_2 = 'cookiecutter.extensions.SlugifyExtension'
    var_3 = 'cookiecutter.extensions.TimeExtension'
    var_4 = 'cookiecutter.extensions.UUIDExtension'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = {}
    var_7 = 'cookiecutter'
    var_8 = '_extensions'
    var_9 = 'myextension.Extension1'
    var_10 = 'anotherextension.Extension2'
    var_11 = [var_9, var_10]
    var_12 = {var_8: var_11}
    var_13 = {var_7: var_12}
    var_14 = [var_9, var_10]
    var_15 = var_5 + var_14
    var_16 = 123
    var_17 = 456.789
    var_18 = [var_16, var_17]
    var_19 = {var_8: var_18}
    var_20 = {var_7: var_19}
    var_21 = '123'
    var_22 = '456.789'
    var_23 = [var_21, var_22]
    var_24 = var_5 + var_23
    var_25 = 'other_key'
    var_26 = 'value'
    var_27 = {var_25: var_26}
    var_28 = {var_7: var_27}
    var_29 = {var_25: var_26}
    var_30 = True
    var_31 = 'trim_blocks'
    var_32 = 'lstrip_blocks'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'cookiecutter.extensions.JsonifyExtension'
    var_2 = 'cookiecutter.extensions.RandomStringExtension'
    var_3 = 'cookiecutter.extensions.SlugifyExtension'
    var_4 = 'cookiecutter.extensions.TimeExtension'
    var_5 = 'cookiecutter.extensions.UUIDExtension'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = 'cookiecutter'
    var_8 = '_extensions'
    var_9 = 'myextension.Extension1'
    var_10 = 'anotherextension.Extension2'
    var_11 = [var_9, var_10]
    var_12 = {var_8: var_11}
    var_13 = {var_7: var_12}
    var_14 = {}
    var_15 = {var_7: var_14}
    var_16 = 'other_key'
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = None
    var_20 = {}
    var_21 = True
    var_22 = {}



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'my_ext.Extension1'
    var_7 = 'my_ext.Extension2'
    var_8 = [var_6, var_7]
    var_9 = '_extensions'
    var_10 = {var_9: var_8}
    var_11 = {var_1: var_10}
    var_12 = 1
    var_13 = 2.5
    var_14 = True
    var_15 = [var_12, var_13, var_14]
    var_16 = {var_9: var_15}
    var_17 = {var_1: var_16}
    var_18 = {}
    var_19 = True
    var_20 = True
    var_21 = 'cookiecutter'
    var_22 = '_extensions'
    var_23 = 'invalid_ext'
    var_24 = [var_23]
    var_25 = {var_22: var_24}
    var_26 = {var_21: var_25}



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'extensions'
    var_1 = {}
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'cookiecutter'
    var_6 = 'name'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = '_extensions'
    var_11 = 'my_ext.Extension1'
    var_12 = 'other_ext.Extension2'
    var_13 = [var_11, var_12]
    var_14 = {var_10: var_13}
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = 123
    var_18 = 'ext1'
    var_19 = True
    var_20 = [var_18, var_17, var_19]
    var_21 = {var_10: var_20}
    var_22 = {var_5: var_21}
    var_23 = False
    var_24 = True



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = 'cookiecutter'
    var_3 = '_extensions'
    var_4 = 'my_ext.Extension1'
    var_5 = 'my_ext.Extension2'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = []
    var_10 = {var_3: var_9}
    var_11 = {var_2: var_10}
    var_12 = 'other_key'
    var_13 = 'value'
    var_14 = {var_12: var_13}
    var_15 = {var_2: var_14}
    var_16 = {var_12: var_13}
    var_17 = {}
    var_18 = 'extra_arg'
    var_19 = 'my_ext.InvalidExtension'
    var_20 = [var_19]
    var_21 = {var_3: var_20}
    var_22 = {var_2: var_21}
    var_23 = 123
    var_24 = 456.789
    var_25 = [var_23, var_24]
    var_26 = {var_3: var_25}
    var_27 = {var_2: var_26}



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = len(var_0)
    assert var_1 == 5
    var_2 = {}
    var_3 = len(var_1)
    assert var_3 == 5
    var_4 = 'cookiecutter'
    var_5 = '_extensions'
    var_6 = 'myextension.Extension1'
    var_7 = 'myextension.Extension2'
    var_8 = [var_6, var_7]
    var_9 = {var_5: var_8}
    var_10 = {var_4: var_9}
    var_11 = 'other_key'
    var_12 = 'value'
    var_13 = {var_11: var_12}
    var_14 = len(var_6)
    assert var_14 == 5
    var_15 = 'cookiecutter'
    var_16 = 'other_key'
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = {var_15: var_18}
    var_20 = len(var_8)
    assert var_20 == 5



# Parsed testcases at query #7
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'my.custom.Extension1'
    var_3 = 'another.Extension2'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = {}
    var_8 = 'other_key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = 'name'
    var_12 = 'test'
    var_13 = {var_11: var_12}
    var_14 = {var_0: var_13}
    var_15 = None
    var_16 = True
    var_17 = 'autoescape'
    var_18 = 'trim_blocks'
    var_19 = {}
    var_20 = module_0.ExtensionLoaderMixin(context=var_19)
    var_21 = {}
    var_22 = {var_11: var_12}
    var_23 = {var_0: var_22}
    var_24 = 'ext1'
    var_25 = 'ext2'
    var_26 = [var_24, var_25]
    var_27 = {var_1: var_26}
    var_28 = {var_0: var_27}
    var_29 = 123
    var_30 = [var_29, var_16]
    var_31 = {var_1: var_30}
    var_32 = {var_0: var_31}



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'custom.Extension1'
    var_3 = 'custom.Extension2'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'cookiecutter.extensions.JsonifyExtension'
    var_8 = 'cookiecutter.extensions.RandomStringExtension'
    var_9 = 'cookiecutter.extensions.SlugifyExtension'
    var_10 = 'cookiecutter.extensions.TimeExtension'
    var_11 = 'cookiecutter.extensions.UUIDExtension'
    var_12 = [var_7, var_8, var_9, var_10, var_11]
    var_13 = 'value'
    var_14 = [var_2, var_3]
    var_15 = var_12 + var_14
    var_16 = {}
    var_17 = {}
    var_18 = {var_0: var_17}
    var_19 = 'not_cookiecutter'
    var_20 = {}
    var_21 = {var_19: var_20}



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 'my_extension.Extension1'
    var_4 = 'other.Extension2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = 'other_key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = {var_1: var_10}
    var_12 = {var_8: var_9}
    var_13 = 'nonexistent.extension'
    var_14 = [var_13]
    var_15 = {var_2: var_14}
    var_16 = {var_1: var_15}
    var_17 = {}
    var_18 = 'test'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = 'cookiecutter'
    var_3 = '_extensions'
    var_4 = 'my.custom.Extension1'
    var_5 = 'another.Extension2'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'cookiecutter'
    var_10 = '_extensions'
    var_11 = []
    var_12 = {var_10: var_11}
    var_13 = {var_9: var_12}
    var_14 = len(var_6)
    assert var_14 == 5
    var_15 = 'other_key'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = len(var_11)
    assert var_18 == 5
    var_19 = {}
    var_20 = None
    var_21 = len(var_16)
    assert var_21 == 5



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'cookiecutter'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'my_ext.Extension1'
    var_7 = 'my_ext.Extension2'
    var_8 = [var_6, var_7]
    var_9 = '_extensions'
    var_10 = {var_9: var_8}
    var_11 = {var_1: var_10}
    var_12 = 123
    var_13 = 456
    var_14 = [var_12, var_13]
    var_15 = {var_9: var_14}
    var_16 = {var_1: var_15}
    var_17 = {}
    var_18 = True
    var_19 = {}



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'my_ext.Extension1'
    var_7 = 'other_ext.Extension2'
    var_8 = [var_6, var_7]
    var_9 = '_extensions'
    var_10 = {var_9: var_8}
    var_11 = {var_1: var_10}
    var_12 = 'ext1'
    var_13 = 123
    var_14 = True
    var_15 = [var_12, var_13, var_14]
    var_16 = {var_9: var_15}
    var_17 = {var_1: var_16}
    var_18 = [var_12]
    var_19 = {var_9: var_18}
    var_20 = {var_1: var_19}
    var_21 = 'cookiecutter'
    var_22 = '_extensions'
    var_23 = 'invalid_ext'
    var_24 = [var_23]
    var_25 = {var_22: var_24}
    var_26 = {var_21: var_25}



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = len(var_0)
    assert var_1 == 5
    var_2 = {}
    var_3 = len(var_1)
    assert var_3 == 5
    var_4 = 'cookiecutter'
    var_5 = '_extensions'
    var_6 = 'my_ext.Extension1'
    var_7 = 'my_ext.Extension2'
    var_8 = [var_6, var_7]
    var_9 = {var_5: var_8}
    var_10 = {var_4: var_9}
    var_11 = 'cookiecutter'
    var_12 = 'other_key'
    var_13 = 'value'
    var_14 = {var_12: var_13}
    var_15 = {var_11: var_14}
    var_16 = len(var_8)
    assert var_16 == 5
    var_17 = {}
    var_18 = 'value'
    var_19 = 123



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter.extensions.JsonifyExtension'
    var_1 = 'cookiecutter.extensions.RandomStringExtension'
    var_2 = 'cookiecutter.extensions.SlugifyExtension'
    var_3 = 'cookiecutter.extensions.TimeExtension'
    var_4 = 'cookiecutter.extensions.UUIDExtension'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = {}
    var_7 = 'cookiecutter'
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = 'my_ext.Extension1'
    var_13 = 'my_ext.Extension2'
    var_14 = [var_12, var_13]
    var_15 = '_extensions'
    var_16 = {var_15: var_14}
    var_17 = {var_7: var_16}
    var_18 = 'ext.Ext'
    var_19 = [var_18]
    var_20 = {var_15: var_19}
    var_21 = {var_7: var_20}
    var_22 = 123
    var_23 = 'ext1'
    var_24 = 'ext2'
    var_25 = [var_23, var_24]
    var_26 = {var_15: var_25}
    var_27 = {var_7: var_26}
    var_28 = True
    var_29 = [var_22, var_28]
    var_30 = {var_15: var_29}
    var_31 = {var_7: var_30}
    var_32 = 'cookiecutter'
    var_33 = '_extensions'
    var_34 = 'invalid_ext'
    var_35 = [var_34]
    var_36 = {var_33: var_35}
    var_37 = {var_32: var_36}



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'my_ext.Extension1'
    var_2 = 'my_ext.Extension2'
    var_3 = [var_1, var_2]
    var_4 = 'cookiecutter'
    var_5 = '_extensions'
    var_6 = {var_5: var_3}
    var_7 = {var_4: var_6}
    var_8 = {}
    var_9 = {var_4: var_8}
    var_10 = 'other_key'
    var_11 = 'value'
    var_12 = {var_10: var_11}
    var_13 = 'trim_blocks'
    var_14 = 'lstrip_blocks'
    var_15 = True
    var_16 = False
    var_17 = {var_13: var_15, var_14: var_16}
    var_18 = {}
    var_19 = {}



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'cookiecutter'
    var_2 = 'other'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'my_ext.Extension1'
    var_7 = 'another.Extension2'
    var_8 = [var_6, var_7]
    var_9 = '_extensions'
    var_10 = {var_9: var_8}
    var_11 = {var_1: var_10}
    var_12 = 123
    var_13 = 456.789
    var_14 = [var_12, var_13]
    var_15 = {var_9: var_14}
    var_16 = {var_1: var_15}
    var_17 = {}
    var_18 = 42
    var_19 = {}



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter.extensions.JsonifyExtension'
    var_1 = 'cookiecutter.extensions.RandomStringExtension'
    var_2 = 'cookiecutter.extensions.SlugifyExtension'
    var_3 = 'cookiecutter.extensions.TimeExtension'
    var_4 = 'cookiecutter.extensions.UUIDExtension'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = 'cookiecutter'
    var_7 = '_extensions'
    var_8 = 'my_ext.Extension1'
    var_9 = 'other_ext.Extension2'
    var_10 = [var_8, var_9]
    var_11 = {var_7: var_10}
    var_12 = {var_6: var_11}
    var_13 = []
    var_14 = {var_7: var_13}
    var_15 = {var_6: var_14}
    var_16 = {}
    var_17 = {var_6: var_16}
    var_18 = {}
    var_19 = 'ext1'
    var_20 = 123
    var_21 = True
    var_22 = [var_19, var_20, var_21]
    var_23 = {var_7: var_22}
    var_24 = {var_6: var_23}
    var_25 = {}
    var_26 = 'test'



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'cookiecutter.extensions.JsonifyExtension'
    var_2 = 'cookiecutter.extensions.RandomStringExtension'
    var_3 = 'cookiecutter.extensions.SlugifyExtension'
    var_4 = 'cookiecutter.extensions.TimeExtension'
    var_5 = 'cookiecutter.extensions.UUIDExtension'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = 'cookiecutter'
    var_8 = '_extensions'
    var_9 = 'myextension.Extension1'
    var_10 = 'anotherextension.Extension2'
    var_11 = [var_9, var_10]
    var_12 = {var_8: var_11}
    var_13 = {var_7: var_12}
    var_14 = [var_9, var_10]
    var_15 = var_6 + var_14
    var_16 = None
    var_17 = 'other_key'
    var_18 = 'value'
    var_19 = {var_17: var_18}
    var_20 = {var_17: var_18}
    var_21 = {var_7: var_20}
    var_22 = 0
    var_23 = ()
    var_24 = 'Test import error'
    var_25 = ImportError(var_24)
    var_26 = {}
    var_27 = {}
    var_28 = True
    var_29 = 'trim_blocks'
    var_30 = 'lstrip_blocks'



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'extensions'
    var_1 = {}
    var_2 = 'cookiecutter'
    var_3 = 'other_key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'my_ext.Extension1'
    var_8 = 'my_ext.Extension2'
    var_9 = [var_7, var_8]
    var_10 = '_extensions'
    var_11 = {var_10: var_9}
    var_12 = {var_2: var_11}
    var_13 = 123
    var_14 = 456
    var_15 = [var_13, var_14]
    var_16 = {var_10: var_15}
    var_17 = {var_2: var_16}
    var_18 = 'cookiecutter'
    var_19 = '_extensions'
    var_20 = 'InvalidExtension'
    var_21 = [var_20]
    var_22 = {var_19: var_21}
    var_23 = {var_18: var_22}
    var_24 = str(var_18)



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'my_ext.Extension1'
    var_7 = 'other_ext.Extension2'
    var_8 = [var_6, var_7]
    var_9 = '_extensions'
    var_10 = {var_9: var_8}
    var_11 = {var_1: var_10}
    var_12 = 'ext.Ext'
    var_13 = [var_12]
    var_14 = {var_9: var_13}
    var_15 = {var_1: var_14}
    var_16 = 123
    var_17 = [var_16, var_12]
    var_18 = {var_9: var_17}
    var_19 = {var_1: var_18}
    var_20 = 'cookiecutter'
    var_21 = '_extensions'
    var_22 = 'invalid_ext'
    var_23 = [var_22]
    var_24 = {var_21: var_23}
    var_25 = {var_20: var_24}



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'my_ext.Extension1'
    var_7 = 'other_ext.Extension2'
    var_8 = [var_6, var_7]
    var_9 = '_extensions'
    var_10 = {var_9: var_8}
    var_11 = {var_1: var_10}
    var_12 = 1
    var_13 = 2.5
    var_14 = True
    var_15 = [var_12, var_13, var_14]
    var_16 = {var_9: var_15}
    var_17 = {var_1: var_16}
    var_18 = {}
    var_19 = True
    var_20 = False
    var_21 = {}



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'extensions'
    var_1 = {}
    var_2 = 'cookiecutter'
    var_3 = 'other_key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '_extensions'
    var_8 = []
    var_9 = {var_7: var_8}
    var_10 = {var_2: var_9}
    var_11 = 'my_ext.Extension1'
    var_12 = 'my_ext.Extension2'
    var_13 = [var_11, var_12]
    var_14 = {var_7: var_13}
    var_15 = {var_2: var_14}
    var_16 = []
    var_17 = {var_7: var_16}
    var_18 = {var_2: var_17}
    var_19 = 123
    var_20 = 456.7
    var_21 = [var_19, var_20]
    var_22 = {var_7: var_21}
    var_23 = {var_2: var_22}



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'cookiecutter'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '_extensions'
    var_7 = 'my_ext.Extension1'
    var_8 = 'other_ext.Extension2'
    var_9 = [var_7, var_8]
    var_10 = {var_6: var_9}
    var_11 = {var_1: var_10}
    var_12 = {}
    var_13 = {var_1: var_12}
    var_14 = 'extra_arg'
    var_15 = 'cookiecutter'
    var_16 = {}
    var_17 = {var_15: var_16}



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'cookiecutter'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'my_ext.Extension1'
    var_7 = 'other_ext.Extension2'
    var_8 = [var_6, var_7]
    var_9 = '_extensions'
    var_10 = {var_9: var_8}
    var_11 = {var_1: var_10}
    var_12 = 123
    var_13 = True
    var_14 = [var_6, var_12, var_13]
    var_15 = {var_9: var_14}
    var_16 = {var_1: var_15}
    var_17 = {}
    var_18 = {}



