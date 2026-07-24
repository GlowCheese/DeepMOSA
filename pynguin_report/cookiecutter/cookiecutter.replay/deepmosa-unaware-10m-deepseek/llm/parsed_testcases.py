####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'my_template'
    var_2 = module_0.get_file_name(var_0, var_1)
    assert var_2 == '/tmp/replay/my_template.json'
    var_3 = 'my_template.json'
    var_4 = module_0.get_file_name(var_0, var_3)
    assert var_4 == '/tmp/replay/my_template.json'
    var_5 = 'my_template.txt'
    var_6 = module_0.get_file_name(var_0, var_5)
    assert var_6 == '/tmp/replay/my_template.txt.json'
    var_7 = ''
    var_8 = module_0.get_file_name(var_0, var_7)
    assert var_8 == '/tmp/replay/.json'
    var_9 = 'nested/template'
    var_10 = module_0.get_file_name(var_0, var_9)
    assert var_10 == '/tmp/replay/nested/template.json'
    var_11 = 'C:\\Users\\test'
    var_12 = module_0.get_file_name(var_11, var_1)



# Parsed testcases at query #2
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'my_template'
    var_2 = module_0.get_file_name(var_0, var_1)
    assert var_2 == '/tmp/replay/my_template.json'
    var_3 = 'my_template.json'
    var_4 = module_0.get_file_name(var_0, var_3)
    assert var_4 == '/tmp/replay/my_template.json'
    var_5 = '/tmp/replay/my_template.json'
    var_6 = 'C:\\replay'
    var_7 = module_0.get_file_name(var_6, var_1)
    assert var_7 == 'C:\\replay\\my_template.json'
    var_8 = './replay'
    var_9 = module_0.get_file_name(var_8, var_1)
    assert var_9 == './replay/my_template.json'
    var_10 = ''
    var_11 = module_0.get_file_name(var_0, var_10)
    assert var_11 == '/tmp/replay/.json'
    var_12 = 'my.template.name'
    var_13 = module_0.get_file_name(var_0, var_12)
    assert var_13 == '/tmp/replay/my.template.name.json'



# Parsed testcases at query #3
#--------------------------




# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'project_name'
    var_2 = 'author'
    var_3 = 'Test Project'
    var_4 = 'Test Author'
    var_5 = {var_1: var_3, var_2: var_4}



# Parsed testcases at query #5
#--------------------------




# Parsed testcases at query #6
#--------------------------




# Parsed testcases at query #7
#--------------------------




# Parsed testcases at query #8
#--------------------------




# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'my_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'replay'
    var_1 = 'template'
    var_2 = 'not_cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'new'
    var_1 = 'nested'
    var_2 = 'replay'
    var_3 = 'template'
    var_4 = 'cookiecutter'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}



# Parsed testcases at query #10
#--------------------------




# Parsed testcases at query #11
#--------------------------




# Parsed testcases at query #12
#--------------------------




# Parsed testcases at query #13
#--------------------------




# Parsed testcases at query #14
#--------------------------




# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'my_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'replay'
    var_1 = 'template'
    var_2 = 'not_cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'new'
    var_1 = 'nested'
    var_2 = 'replay'
    var_3 = 'template'
    var_4 = 'cookiecutter'
    var_5 = 'test'
    var_6 = 'data'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}



# Parsed testcases at query #16
#--------------------------




# Parsed testcases at query #17
#--------------------------




# Parsed testcases at query #18
#--------------------------




# Parsed testcases at query #19
#--------------------------




# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'my_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'replay'
    var_1 = 'template'
    var_2 = 'not_cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'new'
    var_1 = 'nested'
    var_2 = 'replay'
    var_3 = 'template'
    var_4 = 'cookiecutter'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}



# Parsed testcases at query #21
#--------------------------




# Parsed testcases at query #22
#--------------------------




# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'project_name'
    var_2 = 'Test Project'
    var_3 = {var_1: var_2}



# Parsed testcases at query #24
#--------------------------




# Parsed testcases at query #25
#--------------------------




# Parsed testcases at query #26
#--------------------------




# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'my_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'replay'
    var_1 = 'template'
    var_2 = 'not_cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'new_dir'
    var_1 = 'replay'
    var_2 = 'template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}



# Parsed testcases at query #28
#--------------------------




####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------




# Parsed testcases at query #2
#--------------------------




# Parsed testcases at query #3
#--------------------------




# Parsed testcases at query #4
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'version'
    var_4 = 'Test Project'
    var_5 = '1.0'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = 'test_template.json'
    var_9 = 'test_template.json'
    var_10 = 'cookiecutter'
    var_11 = 'key'
    var_12 = 'value'
    var_13 = {var_11: var_12}
    var_14 = {var_10: var_13}
    var_15 = 'test_template.json'
    var_16 = 'another_template'
    var_17 = 'cookiecutter'
    var_18 = 'name'
    var_19 = 'Test'
    var_20 = {var_18: var_19}
    var_21 = {var_17: var_20}
    var_22 = 'another_template.json'
    var_23 = 'bad_template'
    var_24 = 'not_cookiecutter'
    var_25 = 'key'
    var_26 = 'value'
    var_27 = {var_25: var_26}
    var_28 = {var_24: var_27}
    var_29 = 'subdir'
    var_30 = var_24 / var_29
    var_31 = 'nested'
    var_32 = var_30 / var_31
    var_33 = 'nested_template'
    var_34 = 'cookiecutter'
    var_35 = 'test'
    var_36 = 'data'
    var_37 = {var_35: var_36}
    var_38 = {var_34: var_37}
    var_39 = module_0.dump(var_32, var_33, var_38)
    var_40 = 'nested_template.json'
    var_41 = var_32 / var_40



# Parsed testcases at query #5
#--------------------------




# Parsed testcases at query #6
#--------------------------




# Parsed testcases at query #7
#--------------------------




# Parsed testcases at query #8
#--------------------------




# Parsed testcases at query #9
#--------------------------




# Parsed testcases at query #10
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'version'
    var_4 = 'Test Project'
    var_5 = '1.0'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = 'test_template.json'
    var_9 = 'test_template.json'
    var_10 = 'cookiecutter'
    var_11 = 'key'
    var_12 = 'value'
    var_13 = {var_11: var_12}
    var_14 = {var_10: var_13}
    var_15 = 'test_template.json'
    var_16 = 'template'
    var_17 = 'cookiecutter'
    var_18 = 'data'
    var_19 = 'test'
    var_20 = {var_18: var_19}
    var_21 = {var_17: var_20}
    var_22 = 'template.json'
    var_23 = 'subdir'
    var_24 = var_17 / var_23
    var_25 = 'nested'
    var_26 = var_24 / var_25
    var_27 = 'test'
    var_28 = 'cookiecutter'
    var_29 = 'test'
    var_30 = 'value'
    var_31 = {var_29: var_30}
    var_32 = {var_28: var_31}
    var_33 = module_0.dump(var_26, var_27, var_32)
    var_34 = 'test.json'
    var_35 = var_26 / var_34
    var_36 = 'test'
    var_37 = 'not_cookiecutter'
    var_38 = 'key'
    var_39 = 'value'
    var_40 = {var_38: var_39}
    var_41 = {var_37: var_40}
    var_42 = module_0.dump(var_26, var_36, var_41)
    var_43 = 'complex'
    var_44 = 'cookiecutter'
    var_45 = 'project'
    var_46 = 'list_data'
    var_47 = 'null_value'
    var_48 = 'name'
    var_49 = 'settings'
    var_50 = 'Test'
    var_51 = 'debug'
    var_52 = 'port'
    var_53 = True
    var_54 = 8000
    var_55 = {var_51: var_53, var_52: var_54}
    var_56 = {var_48: var_50, var_49: var_55}
    var_57 = 2
    var_58 = 3
    var_59 = [var_53, var_57, var_58]
    var_60 = None
    var_61 = {var_45: var_56, var_46: var_59, var_47: var_60}
    var_62 = {var_44: var_61}
    var_63 = module_0.dump(var_26, var_43, var_62)
    var_64 = 'complex.json'
    var_65 = var_26 / var_64



# Parsed testcases at query #11
#--------------------------




# Parsed testcases at query #12
#--------------------------




# Parsed testcases at query #13
#--------------------------




# Parsed testcases at query #14
#--------------------------




# Parsed testcases at query #15
#--------------------------




# Parsed testcases at query #16
#--------------------------




# Parsed testcases at query #17
#--------------------------




# Parsed testcases at query #18
#--------------------------




# Parsed testcases at query #19
#--------------------------




# Parsed testcases at query #20
#--------------------------




# Parsed testcases at query #21
#--------------------------




# Parsed testcases at query #22
#--------------------------




# Parsed testcases at query #23
#--------------------------




# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'my_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'replay'
    var_1 = 'template'
    var_2 = 'not_cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'new'
    var_1 = 'nested'
    var_2 = 'replay'
    var_3 = 'template'
    var_4 = 'cookiecutter'
    var_5 = 'test'
    var_6 = 'data'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}



# Parsed testcases at query #25
#--------------------------




# Parsed testcases at query #26
#--------------------------




