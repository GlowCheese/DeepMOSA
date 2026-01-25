####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'Test get_file_name function with various inputs.'
    var_1 = '/replay/dir'
    var_2 = 'my_template'
    var_3 = module_0.get_file_name(var_1, var_2)
    var_4 = 'my_template.json'
    var_5 = module_0.get_file_name(var_1, var_4)
    var_6 = 'template_name'
    var_7 = 'template_name.json'
    var_8 = ''
    var_9 = module_0.get_file_name(var_1, var_8)
    var_10 = '.json'
    var_11 = 'my.template.name'
    var_12 = module_0.get_file_name(var_1, var_11)
    var_13 = 'my.template.name.json'
    var_14 = './replay'
    var_15 = 'template'
    var_16 = module_0.get_file_name(var_14, var_15)
    var_17 = 'template.json'



# Parsed testcases at query #2
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'Test get_file_name function.'
    var_1 = '/tmp/replay'
    var_2 = 'my_template'
    var_3 = module_0.get_file_name(var_1, var_2)
    var_4 = 'my_template.json'
    var_5 = module_0.get_file_name(var_1, var_4)
    var_6 = 'another_template'
    var_7 = 'another_template.json'
    var_8 = ''
    var_9 = 'template'
    var_10 = module_0.get_file_name(var_8, var_9)
    var_11 = 'template.json'
    var_12 = 'my.template.name'
    var_13 = module_0.get_file_name(var_1, var_12)
    var_14 = 'my.template.name.json'



# Parsed testcases at query #3
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'Test get_file_name function.'
    var_1 = '/tmp/replay'
    var_2 = 'my-template'
    var_3 = module_0.get_file_name(var_1, var_2)
    var_4 = 'my-template.json'
    var_5 = module_0.get_file_name(var_1, var_4)
    var_6 = ''
    var_7 = module_0.get_file_name(var_6, var_2)
    var_8 = 'my-special-template'
    var_9 = module_0.get_file_name(var_1, var_8)
    var_10 = 'my-special-template.json'



# Parsed testcases at query #4
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'Test get_file_name function.'
    var_1 = '/tmp/replay'
    var_2 = 'my_template'
    var_3 = module_0.get_file_name(var_1, var_2)
    var_4 = 'my_template.json'
    var_5 = module_0.get_file_name(var_1, var_4)
    var_6 = 'another_template'
    var_7 = 'another_template.json'
    var_8 = ''
    var_9 = module_0.get_file_name(var_1, var_8)
    var_10 = '.json'
    var_11 = '/home/user/replay'
    var_12 = 'my-complex-template-name'
    var_13 = module_0.get_file_name(var_11, var_12)
    var_14 = 'my-complex-template-name.json'



# Parsed testcases at query #5
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'Test get_file_name function.'
    var_1 = '/tmp/replay'
    var_2 = 'my_template'
    var_3 = module_0.get_file_name(var_1, var_2)
    var_4 = 'my_template.json'
    var_5 = module_0.get_file_name(var_1, var_4)
    var_6 = 'another_template'
    var_7 = 'another_template.json'
    var_8 = ''
    var_9 = module_0.get_file_name(var_1, var_8)
    var_10 = '.json'
    var_11 = '/home/user/replay'
    var_12 = 'my-complex_template_name'
    var_13 = module_0.get_file_name(var_11, var_12)
    var_14 = 'my-complex_template_name.json'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = "Test dump function doesn't add duplicate .json extension."
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'

def test_case_0():
    var_0 = 'Test dump raises ValueError when context lacks cookiecutter key.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.replay.make_sure_path_exists'



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads and validates json context file.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'test_project'
    var_6 = 'Test Author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = "Test load raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load works when template_name already ends with .json.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load raises JSONDecodeError for invalid json content.'
    var_1 = 'test_template'
    var_2 = 'test_template.json'
    var_3 = 'invalid json content {]'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'Test load works with template_name that already has .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = "Test load raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent_template'



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'Test dump function with template name already containing .json.'
    var_1 = 'replay'
    var_2 = 'my_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'

def test_case_0():
    var_0 = 'Test dump function raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.replay.make_sure_path_exists'

def test_case_0():
    var_0 = 'Test dump function calls make_sure_path_exists.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'Test dump function when template_name already has .json extension.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = 'Test dump function raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = "Test dump function creates replay directory if it doesn't exist."
    var_1 = 'replay'
    var_2 = 'nested'
    var_3 = 'dir'
    var_4 = 'test_template'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'my_project'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'Test load function raises FileNotFoundError when file does not exist.'
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load function works with template_name that already has .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads and returns json context from file.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'test_project'
    var_6 = 'test_author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}

def test_case_0():
    var_0 = 'Test load function handles template names with .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent_template'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'Test dump function handles template names with .json extension.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = 'Test dump function raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads and returns valid context from json file.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function raises FileNotFoundError when file does not exist.'
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load function raises json.JSONDecodeError for invalid json.'
    var_1 = 'test_template'
    var_2 = 'test_template.json'
    var_3 = '{ invalid json }'

def test_case_0():
    var_0 = 'Test load function works when template_name already has .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = "Test load raises FileNotFoundError when replay file doesn't exist."
    var_1 = 'replay'
    var_2 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load works correctly when template_name already has .json extension.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads json data from file correctly.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'project_slug'
    var_5 = 'my_project'
    var_6 = {var_3: var_5, var_4: var_5}
    var_7 = {var_2: var_6}
    var_8 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when replay file doesn't exist."
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load function with template name already having .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'Test load function works when template_name already has .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 2

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when replay file doesn't exist."
    var_1 = 'nonexistent_template'



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads json data from file.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author_name'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function with .json suffix in template name.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json.json'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load function raises json.JSONDecodeError for invalid json.'
    var_1 = 'test_template'
    var_2 = 'test_template.json'
    var_3 = 'invalid json {'



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = "Test load raises FileNotFoundError when replay file doesn't exist."
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load correctly handles template names with .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = "Test load raises FileNotFoundError when replay file doesn't exist."
    var_1 = 'replay'
    var_2 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load works with template name already having .json suffix.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'Test dump raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'project_name'
    var_4 = 'My Project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test dump with template name that already has .json suffix.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'My Project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'test_template.json'



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'Test load function works with template names that already have .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json.json'

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent_template'



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'Test dump function when template_name already has .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'Test dump function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads json data from file correctly.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function works with template_name already having .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent_template'



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'Test dump function when template_name already has .json suffix.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump raises ValueError when context missing cookiecutter key.'
    var_1 = 'cookiecutter.replay.make_sure_path_exists'
    var_2 = 'replay'
    var_3 = 'test_template'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}

def test_case_0():
    var_0 = "Test dump creates replay directory if it doesn't exist."
    var_1 = 'new_replay_dir'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads json data from file correctly.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author_name'
    var_5 = 'test_project'
    var_6 = 'Test Author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function with template name already having .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'some_key'
    var_3 = 'some_value'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when replay file doesn't exist."
    var_1 = 'nonexistent_template'



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'Test the load function.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'
    var_10 = 2
    var_11 = 'project'
    var_12 = 'test'
    var_13 = {var_11: var_12}
    var_14 = 'invalid_template.json'
    var_15 = 2
    var_16 = 'invalid_template'
    var_17 = 'nonexistent_template'



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = "Test load raises FileNotFoundError when replay file doesn't exist."
    var_1 = 'replay'
    var_2 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load handles template names with .json suffix.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = "Test load raises FileNotFoundError when replay file doesn't exist."
    var_1 = 'replay'
    var_2 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load handles template names with .json suffix.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'Test dump function handles template names with .json suffix.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test dump overwrites existing replay file.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'old_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'new_project'
    var_9 = {var_4: var_8}
    var_10 = {var_3: var_9}

def test_case_0():
    var_0 = 'Test dump with nested and complex context data.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'author'
    var_6 = 'nested'
    var_7 = 'list'
    var_8 = 'test_project'
    var_9 = 'test_author'
    var_10 = 'key1'
    var_11 = 'key2'
    var_12 = 'value1'
    var_13 = 'item1'
    var_14 = 'item2'
    var_15 = [var_13, var_14]
    var_16 = {var_10: var_12, var_11: var_15}
    var_17 = 1
    var_18 = 2
    var_19 = 3
    var_20 = [var_17, var_18, var_19]
    var_21 = {var_4: var_8, var_5: var_9, var_6: var_16, var_7: var_20}
    var_22 = {var_3: var_21}



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 'Test load function raises FileNotFoundError when file does not exist.'
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load function works when template_name already has .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = "Test dump creates replay directory if it doesn't exist."
    var_1 = 'nonexistent'
    var_2 = 'replay'
    var_3 = 'test_template'
    var_4 = 'cookiecutter'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}

def test_case_0():
    var_0 = 'Test dump handles template names with .json extension.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'other_key'
    var_4 = 'value'
    var_5 = {var_3: var_4}



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = "Test load raises FileNotFoundError when replay file doesn't exist."
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load works correctly when template_name already has .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads json data from file correctly.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'extra_key'
    var_4 = 'project_name'
    var_5 = 'author'
    var_6 = 'my_project'
    var_7 = 'John Doe'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = 'extra_value'
    var_10 = {var_2: var_8, var_3: var_9}
    var_11 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function with template name already having .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent_template'



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = 'Test load function works with template name ending in .json.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = "Test load raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent_template'



# Parsed testcases at query #36
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads and validates json replay file.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'My Project'
    var_6 = 'Test Author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = "Test load raises FileNotFoundError when replay file doesn't exist."
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load works when template_name already has .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'



# Parsed testcases at query #37
#--------------------------


def test_case_0():
    var_0 = 'Test load function with template name already containing .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent_template'



# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = 'Test load function raises FileNotFoundError when file does not exist.'
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load function works correctly when template_name already has .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #39
#--------------------------


def test_case_0():
    var_0 = 'Test dump function when template_name already has .json suffix.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = 'Test dump function raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}



# Parsed testcases at query #40
#--------------------------


def test_case_0():
    var_0 = 'Test dump function writes context to json file.'
    var_1 = 'replay'
    var_2 = 'my-template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'author'
    var_6 = 'test_project'
    var_7 = 'Test Author'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = 'my-template.json'

def test_case_0():
    var_0 = 'Test dump function handles template names with .json suffix.'
    var_1 = 'replay'
    var_2 = 'my-template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my-template.json'

def test_case_0():
    var_0 = "Test dump function creates replay directory if it doesn't exist."
    var_1 = 'nonexistent'
    var_2 = 'replay'
    var_3 = 'my-template'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'my-template.json'

def test_case_0():
    var_0 = 'Test dump raises ValueError when context lacks cookiecutter key.'
    var_1 = 'replay'
    var_2 = 'my-template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test dump writes valid JSON with proper formatting.'
    var_1 = 'replay'
    var_2 = 'my-template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'author'
    var_6 = 'test_project'
    var_7 = 'Test Author'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = 'my-template.json'
    var_11 = 'utf-8'



# Parsed testcases at query #41
#--------------------------


def test_case_0():
    var_0 = "Test load raises FileNotFoundError when replay file doesn't exist."
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load works correctly when template_name already has .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_slug'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #42
#--------------------------


def test_case_0():
    var_0 = 'Test load function with template name already having .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent_template'



# Parsed testcases at query #43
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads json data from file correctly.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function with template name already having .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'invalid_key'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when replay file doesn't exist."
    var_1 = 'nonexistent_template'



# Parsed testcases at query #44
#--------------------------


def test_case_0():
    var_0 = 'Test load function with template name already having .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent_template'



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'Test get_file_name function.'
    var_1 = '/tmp/replay'
    var_2 = 'my_template'
    var_3 = module_0.get_file_name(var_1, var_2)
    var_4 = 'my_template.json'
    var_5 = module_0.get_file_name(var_1, var_4)
    var_6 = 'another_template'
    var_7 = 'another_template.json'
    var_8 = ''
    var_9 = module_0.get_file_name(var_1, var_8)
    var_10 = '.json'
    var_11 = 'my.template.name'
    var_12 = module_0.get_file_name(var_1, var_11)
    var_13 = 'my.template.name.json'
    var_14 = 'template.json.backup'
    var_15 = module_0.get_file_name(var_1, var_14)
    var_16 = 'template.json.backup.json'



# Parsed testcases at query #2
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'Test get_file_name returns correct file path.'
    var_1 = '/tmp/replay'
    var_2 = 'my_template'
    var_3 = module_0.get_file_name(var_1, var_2)
    var_4 = 'my_template.json'
    var_5 = module_0.get_file_name(var_1, var_4)
    var_6 = 'template_name'
    var_7 = 'template_name.json'
    var_8 = '/home/user/.cookiecutters'
    var_9 = 'my_project'
    var_10 = module_0.get_file_name(var_8, var_9)
    var_11 = 'my_project.json'
    var_12 = '/replay'
    var_13 = 'my.template.name'
    var_14 = module_0.get_file_name(var_12, var_13)
    var_15 = 'my.template.name.json'



# Parsed testcases at query #3
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'Test get_file_name function.'
    var_1 = '/replay'
    var_2 = 'my-template'
    var_3 = module_0.get_file_name(var_1, var_2)
    var_4 = 'my-template.json'
    var_5 = module_0.get_file_name(var_1, var_4)
    var_6 = '/home/user/.cookiecutters'
    var_7 = 'template-name'
    var_8 = module_0.get_file_name(var_6, var_7)
    var_9 = 'template-name.json'
    var_10 = '/tmp'
    var_11 = 'test.json'
    var_12 = module_0.get_file_name(var_10, var_11)
    var_13 = '.json.json'



# Parsed testcases at query #4
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'Test get_file_name function with various inputs.'
    var_1 = '/tmp/replay'
    var_2 = 'my_template'
    var_3 = module_0.get_file_name(var_1, var_2)
    var_4 = 'my_template.json'
    var_5 = module_0.get_file_name(var_1, var_4)
    var_6 = ''
    var_7 = module_0.get_file_name(var_1, var_6)
    var_8 = '.json'
    var_9 = 'my.template.name'
    var_10 = module_0.get_file_name(var_1, var_9)
    var_11 = 'my.template.name.json'
    var_12 = 'my.template.json'
    var_13 = module_0.get_file_name(var_1, var_12)



# Parsed testcases at query #5
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'Test get_file_name returns correct file path.'
    var_1 = '/tmp/replay'
    var_2 = 'my_template'
    var_3 = module_0.get_file_name(var_1, var_2)
    var_4 = 'my_template.json'
    var_5 = module_0.get_file_name(var_1, var_4)
    var_6 = 'another_template'
    var_7 = 'another_template.json'
    var_8 = ''
    var_9 = module_0.get_file_name(var_1, var_8)
    var_10 = '.json'
    var_11 = 'my.template.name'
    var_12 = module_0.get_file_name(var_1, var_11)
    var_13 = 'my.template.name.json'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads json data from file correctly.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function with template name already having .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json.json'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function raises FileNotFoundError when file does not exist.'
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load function raises json.JSONDecodeError for invalid json.'
    var_1 = 'test_template'
    var_2 = 'test_template.json'
    var_3 = '{ invalid json }'



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'Test load function works with template name that already has .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'Test load function raises FileNotFoundError when file does not exist.'
    var_1 = 'nonexistent_template'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads json data from file correctly.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function works when template name already has .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json.json'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent_template'



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'Test dump function with template name already having .json extension.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump function raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads and validates json context from file.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'project_slug'
    var_5 = 'test_project'
    var_6 = {var_3: var_5, var_4: var_5}
    var_7 = {var_2: var_6}
    var_8 = 'test_template.json'
    var_9 = 2

def test_case_0():
    var_0 = 'Test load function works with template name already containing .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'invalid_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when replay file doesn't exist."
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load function works with Path object as replay_dir.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'data'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'Test dump function writes context to json file correctly.'
    var_1 = 'replay'
    var_2 = 'my-template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'project_slug'
    var_6 = 'test_project'
    var_7 = {var_4: var_6, var_5: var_6}
    var_8 = {var_3: var_7}
    var_9 = 'my-template.json'

def test_case_0():
    var_0 = 'Test dump function handles template names ending with .json.'
    var_1 = 'replay'
    var_2 = 'my-template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my-template.json'

def test_case_0():
    var_0 = "Test dump function creates replay directory if it doesn't exist."
    var_1 = 'nonexistent'
    var_2 = 'replay'
    var_3 = 'my-template'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}

def test_case_0():
    var_0 = 'Test dump raises ValueError when context lacks cookiecutter key.'
    var_1 = 'replay'
    var_2 = 'my-template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test dump writes json with proper indentation.'
    var_1 = 'replay'
    var_2 = 'my-template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my-template.json'
    var_9 = 'utf-8'



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'Test dump function writes context to json file correctly.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'author'
    var_6 = 'test_project'
    var_7 = 'test_author'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = 'my_template.json'

def test_case_0():
    var_0 = "Test dump function creates replay directory if it doesn't exist."
    var_1 = 'nonexistent'
    var_2 = 'replay'
    var_3 = 'test_template'
    var_4 = 'cookiecutter'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump function handles template names ending with .json.'
    var_1 = 'replay'
    var_2 = 'my_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'

def test_case_0():
    var_0 = 'Test dump raises ValueError when context lacks cookiecutter key.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'other_key'
    var_4 = 'value'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test dump overwrites existing replay file.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'old_value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'new_value'
    var_9 = {var_4: var_8}
    var_10 = {var_3: var_9}
    var_11 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump handles complex nested context structures.'
    var_1 = 'replay'
    var_2 = 'complex_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'nested'
    var_6 = 'list'
    var_7 = 'bool'
    var_8 = 'null'
    var_9 = 'test'
    var_10 = 'level1'
    var_11 = 'level2'
    var_12 = 'item1'
    var_13 = 'item2'
    var_14 = [var_12, var_13]
    var_15 = {var_11: var_14}
    var_16 = {var_10: var_15}
    var_17 = 1
    var_18 = 2
    var_19 = 3
    var_20 = [var_17, var_18, var_19]
    var_21 = True
    var_22 = None
    var_23 = {var_4: var_9, var_5: var_16, var_6: var_20, var_7: var_21, var_8: var_22}
    var_24 = {var_3: var_23}
    var_25 = 'complex_template.json'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'Test dump raises ValueError when context lacks cookiecutter key.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.replay.make_sure_path_exists'

def test_case_0():
    var_0 = 'Test dump handles template names that already have .json extension.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump calls make_sure_path_exists to create replay directory.'
    var_1 = 'new_replay_dir'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'
    var_9 = 'builtins.open'
    var_10 = 'json.dump'



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'Test dump function with template name already ending in .json.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = 'Test dump function raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = "Test dump function creates replay directory if it doesn't exist."
    var_1 = 'new_replay_dir'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'Test dump function writes context to json file.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'test_project'
    var_6 = 'test_author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}

def test_case_0():
    var_0 = 'Test dump function handles template names with .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = "Test dump function creates replay directory if it doesn't exist."
    var_1 = 'new_dir'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = 'Test dump function writes properly formatted json.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'version'
    var_6 = 'my_project'
    var_7 = 'John Doe'
    var_8 = '1.0.0'
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = {var_2: var_9}



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'Test load function works with template name already having .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'Test load function raises FileNotFoundError when file does not exist.'
    var_1 = 'nonexistent_template'



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = "Test dump function doesn't add duplicate .json extension."
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'test_template.json'
    var_9 = 'test_template.json.json'

def test_case_0():
    var_0 = 'Test dump raises ValueError when context lacks cookiecutter key.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = "Test dump creates replay directory if it doesn't exist."
    var_1 = 'nested'
    var_2 = 'replay'
    var_3 = 'dir'
    var_4 = 'test_template'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'my_project'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}

def test_case_0():
    var_0 = 'Test dump overwrites existing replay file.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'old_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'new_project'
    var_9 = {var_4: var_8}
    var_10 = {var_3: var_9}



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'Test dump function when template_name already ends with .json.'
    var_1 = 'replay'
    var_2 = 'my_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = 'Test dump function raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'Test dump function with template name already having .json extension.'
    var_1 = 'replay'
    var_2 = 'my-template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = 'Test dump function raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'my-template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = "Test dump function doesn't add duplicate .json extension."
    var_1 = 'replay'
    var_2 = 'test-template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'test-template.json'
    var_9 = 'test-template.json.json'

def test_case_0():
    var_0 = 'Test dump function raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'test-template'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test dump function handles complex nested context.'
    var_1 = 'replay'
    var_2 = 'complex-template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'nested'
    var_6 = 'my_project'
    var_7 = 'key1'
    var_8 = 'key2'
    var_9 = 'value1'
    var_10 = 'item1'
    var_11 = 'item2'
    var_12 = [var_10, var_11]
    var_13 = {var_7: var_9, var_8: var_12}
    var_14 = {var_4: var_6, var_5: var_13}
    var_15 = {var_3: var_14}

def test_case_0():
    var_0 = 'Test dump function overwrites existing replay file.'
    var_1 = 'replay'
    var_2 = 'test-template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'old_value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'new_value'
    var_9 = {var_4: var_8}
    var_10 = {var_3: var_9}



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads json data from file correctly.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'test_project'
    var_6 = 'test_author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function works when template_name already has .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load raises ValueError when context lacks cookiecutter key.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = "Test load raises FileNotFoundError when replay file doesn't exist."
    var_1 = 'nonexistent_template'



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads json data from file correctly.'
    var_1 = 'test-template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test-template.json'
    var_10 = 2

def test_case_0():
    var_0 = 'Test load function with .json extension in template name.'
    var_1 = 'test-template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test-template.json'
    var_8 = 2

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test-template'
    var_2 = 'other_key'
    var_3 = 'some_value'
    var_4 = {var_2: var_3}
    var_5 = 'test-template.json'
    var_6 = 2

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent-template'



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'Test load function works with template name already having .json extension.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'Test dump function writes context to json file correctly.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump function with template name already having .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = "Test dump function creates replay directory if it doesn't exist."
    var_1 = 'new_dir'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump function writes json with proper indentation.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'nested'
    var_5 = 'my_project'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = {var_2: var_9}
    var_11 = 'test_template.json'



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load function works when template_name already has .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = "Test load raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load works with template name that already has .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'Test dump function with template name already ending in .json.'
    var_1 = 'replay'
    var_2 = 'my_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = 'Test dump function raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'Test dump function handles template names ending with .json.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump function raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'other_key'
    var_4 = 'value'
    var_5 = {var_3: var_4}



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 'Test load function with template name already having .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent_template'



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads and validates json context from file.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = "Test load function works with template names that don't have .json extension."
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'my_template.json'

def test_case_0():
    var_0 = 'Test load function raises ValueError when context lacks cookiecutter key.'
    var_1 = 'invalid_template'
    var_2 = 'project'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = 'invalid_template.json'

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when replay file doesn't exist."
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load function raises JSONDecodeError for invalid json content.'
    var_1 = 'bad_json'
    var_2 = 'bad_json.json'
    var_3 = 'invalid json content {'



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load function works with template name already having .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = "Test dump function creates replay directory if it doesn't exist."
    var_1 = 'new_replay_dir'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = 'Test dump function handles template names with .json suffix.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump function raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test dump function overwrites existing replay file.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'old_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'new_project'
    var_9 = {var_4: var_8}
    var_10 = {var_3: var_9}

def test_case_0():
    var_0 = 'Test dump function with nested and complex context data.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'nested'
    var_6 = 'number'
    var_7 = 'boolean'
    var_8 = 'my_project'
    var_9 = 'key1'
    var_10 = 'key2'
    var_11 = 'value1'
    var_12 = 'item1'
    var_13 = 'item2'
    var_14 = [var_12, var_13]
    var_15 = {var_9: var_11, var_10: var_14}
    var_16 = 42
    var_17 = True
    var_18 = {var_4: var_8, var_5: var_15, var_6: var_16, var_7: var_17}
    var_19 = {var_3: var_18}



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = 'Test dump function writes context to json file.'
    var_1 = 'replay'
    var_2 = 'my-template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'author'
    var_6 = 'my_project'
    var_7 = 'John Doe'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = 'my-template.json'

def test_case_0():
    var_0 = 'Test dump function adds .json extension if not present.'
    var_1 = 'replay'
    var_2 = 'my-template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my-template.json'

def test_case_0():
    var_0 = "Test dump function doesn't add duplicate .json extension."
    var_1 = 'replay'
    var_2 = 'my-template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my-template.json'

def test_case_0():
    var_0 = 'Test dump raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'my-template'
    var_3 = 'other_key'
    var_4 = 'value'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = "Test dump creates replay directory if it doesn't exist."
    var_1 = 'nonexistent'
    var_2 = 'replay'
    var_3 = 'my-template'
    var_4 = 'cookiecutter'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'my-template.json'

def test_case_0():
    var_0 = 'Test dump works with string path.'
    var_1 = 'replay'
    var_2 = 'my-template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my-template.json'

def test_case_0():
    var_0 = 'Test dump overwrites existing replay file.'
    var_1 = 'replay'
    var_2 = 'my-template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value1'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'value2'
    var_9 = {var_4: var_8}
    var_10 = {var_3: var_9}
    var_11 = 'my-template.json'



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = 'Test load works with template_name that already has .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = "Test load raises FileNotFoundError when replay file doesn't exist."
    var_1 = 'nonexistent_template'



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads json data from file correctly.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load function works with template name that already has .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #36
#--------------------------




# Parsed testcases at query #37
#--------------------------




# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = 'Test dump raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.replay.make_sure_path_exists'

def test_case_0():
    var_0 = 'Test dump handles template name with .json extension.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump creates properly formatted JSON file.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'version'
    var_6 = 'my_project'
    var_7 = '1.0.0'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = 'cookiecutter.replay.make_sure_path_exists'



# Parsed testcases at query #39
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads json data from file correctly.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author_name'
    var_5 = 'test_project'
    var_6 = 'Test Author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}

def test_case_0():
    var_0 = 'Test load function works with template_name that already has .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when replay file doesn't exist."
    var_1 = 'nonexistent_template'



# Parsed testcases at query #40
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads json data from file correctly.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'test_author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load raises FileNotFoundError when replay file does not exist.'
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load function works with template name already having .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'



# Parsed testcases at query #41
#--------------------------


def test_case_0():
    var_0 = 'Test dump function when template_name already has .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'My Project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'Test dump raises ValueError when context missing cookiecutter key.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'My Project'
    var_4 = {var_2: var_3}



# Parsed testcases at query #42
#--------------------------


def test_case_0():
    var_0 = 'Test dump function when template_name already has .json suffix.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump function raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}



# Parsed testcases at query #43
#--------------------------


def test_case_0():
    var_0 = "Test dump doesn't add .json suffix when template_name already has it."
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'

def test_case_0():
    var_0 = 'Test dump raises ValueError when context missing cookiecutter key.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.replay.make_sure_path_exists'

def test_case_0():
    var_0 = 'Test dump calls make_sure_path_exists with replay_dir.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'



# Parsed testcases at query #44
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads json data from file correctly.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 2

def test_case_0():
    var_0 = 'Test load raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = 2

def test_case_0():
    var_0 = 'Test load function handles template names with .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 2

def test_case_0():
    var_0 = "Test load raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent_template'



# Parsed testcases at query #45
#--------------------------


def test_case_0():
    var_0 = "Test dump creates replay directory if it doesn't exist."
    var_1 = 'nonexistent'
    var_2 = 'replay'
    var_3 = 'template'
    var_4 = 'cookiecutter'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}

def test_case_0():
    var_0 = 'Test dump handles template names with .json extension.'
    var_1 = 'replay'
    var_2 = 'my_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'

def test_case_0():
    var_0 = 'Test dump raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'template'
    var_3 = 'other_key'
    var_4 = 'value'
    var_5 = {var_3: var_4}



# Parsed testcases at query #46
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads json data from file correctly.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'project_slug'
    var_5 = 'my_project'
    var_6 = {var_3: var_5, var_4: var_5}
    var_7 = {var_2: var_6}
    var_8 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load function works with template_name already having .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'



# Parsed testcases at query #47
#--------------------------


def test_case_0():
    var_0 = 'Test dump function writes context to json file correctly.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'test_author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump function when template name already has .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = "Test dump function creates replay directory if it doesn't exist."
    var_1 = 'nested'
    var_2 = 'replay'
    var_3 = 'dir'
    var_4 = 'test_template'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'my_project'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'Test dump function works with Path object as replay_dir.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump function overwrites existing replay file.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'project1'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'project2'
    var_8 = {var_3: var_7}
    var_9 = {var_2: var_8}
    var_10 = 'test_template.json'



# Parsed testcases at query #48
#--------------------------


def test_case_0():
    var_0 = 'Test dump function handles template names ending with .json.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump raises ValueError when context lacks cookiecutter key.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}



# Parsed testcases at query #49
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads and returns json context correctly.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'test_author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = "Test load raises FileNotFoundError when replay file doesn't exist."
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load works with template name already having .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json.json'



# Parsed testcases at query #50
#--------------------------


def test_case_0():
    var_0 = "Test load raises FileNotFoundError when replay file doesn't exist."
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load works with template names that already have .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #51
#--------------------------


def test_case_0():
    var_0 = 'Test load with template name already having .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = "Test load raises FileNotFoundError when replay file doesn't exist."
    var_1 = 'nonexistent_template'



# Parsed testcases at query #52
#--------------------------


def test_case_0():
    var_0 = 'Test load raises FileNotFoundError when replay file does not exist.'
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load works correctly when template_name already has .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #53
#--------------------------


def test_case_0():
    var_0 = 'Test dump function with template name already having .json suffix.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = 'Test dump function raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test dump function overwrites existing replay file.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'old_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'new_project'
    var_9 = {var_4: var_8}
    var_10 = {var_3: var_9}



# Parsed testcases at query #54
#--------------------------


def test_case_0():
    var_0 = 'Test load raises FileNotFoundError when replay file does not exist.'
    var_1 = 'replay'
    var_2 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load works when template_name already has .json extension.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}



# Parsed testcases at query #55
#--------------------------


def test_case_0():
    var_0 = 'Test dump function when template_name already has .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'My Project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'Test dump raises ValueError when context lacks cookiecutter key.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'My Project'
    var_4 = {var_2: var_3}



# Parsed testcases at query #56
#--------------------------


def test_case_0():
    var_0 = 'Test dump function when template_name already has .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'Test dump raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}



# Parsed testcases at query #57
#--------------------------


def test_case_0():
    var_0 = 'Test dump function handles template names that already end with .json.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump function raises ValueError when context lacks cookiecutter key.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'other_key'
    var_4 = 'value'
    var_5 = {var_3: var_4}



# Parsed testcases at query #58
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads and validates json context from file.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function with template name already having .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function raises FileNotFoundError when file does not exist.'
    var_1 = 'nonexistent_template'



# Parsed testcases at query #59
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads json data from file correctly.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function works with template name already having .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json.json'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function raises FileNotFoundError when file does not exist.'
    var_1 = 'nonexistent_template'



# Parsed testcases at query #60
#--------------------------


def test_case_0():
    var_0 = "Test load raises FileNotFoundError when file doesn't exist."
    var_1 = 'replay'
    var_2 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load works with template names that already have .json extension.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}



# Parsed testcases at query #61
#--------------------------


def test_case_0():
    var_0 = 'Test dump function writes context to json file.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}

def test_case_0():
    var_0 = 'Test dump function handles template name with .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump raises ValueError when context lacks cookiecutter key.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = "Test dump creates replay directory if it doesn't exist."
    var_1 = 'nonexistent_dir'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'

def test_case_0():
    var_0 = 'Test dump writes properly formatted json with indentation.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'nested'
    var_5 = 'my_project'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = {var_2: var_9}



# Parsed testcases at query #62
#--------------------------


def test_case_0():
    var_0 = 'Test dump function with template name already ending in .json.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump function raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.replay.make_sure_path_exists'

def test_case_0():
    var_0 = 'Test dump function ensures replay directory exists.'
    var_1 = 'new_replay_dir'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'



# Parsed testcases at query #63
#--------------------------


def test_case_0():
    var_0 = 'Test dump function writes context to json file.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}

def test_case_0():
    var_0 = 'Test dump function with template_name already having .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = "Test dump function creates replay directory if it doesn't exist."
    var_1 = 'new_dir'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = 'Test dump function writes json with proper indentation.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #64
#--------------------------


def test_case_0():
    var_0 = 'Test load function handles template names with .json extension.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = "Test load raises FileNotFoundError when replay file doesn't exist."
    var_1 = 'replay'
    var_2 = 'nonexistent_template'



# Parsed testcases at query #65
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads json data from file correctly.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function with template name already containing .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'another_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json.json'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function raises FileNotFoundError when file does not exist.'
    var_1 = 'nonexistent_template'



# Parsed testcases at query #66
#--------------------------


def test_case_0():
    var_0 = 'Test load function raises FileNotFoundError when file does not exist.'
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load function works with template name ending in .json.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #67
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads json data from file correctly.'
    var_1 = 'test-template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author_name'
    var_5 = 'test_project'
    var_6 = 'Test Author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test-template.json'

def test_case_0():
    var_0 = 'Test load function with .json extension in template name.'
    var_1 = 'test-template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test-template.json'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test-template'
    var_2 = 'some_key'
    var_3 = 'some_value'
    var_4 = {var_2: var_3}
    var_5 = 'test-template.json'

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when replay file doesn't exist."
    var_1 = 'nonexistent-template'



# Parsed testcases at query #68
#--------------------------


def test_case_0():
    var_0 = 'Test load raises FileNotFoundError when replay file does not exist.'
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load function works when template_name already has .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'My Project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #69
#--------------------------


def test_case_0():
    var_0 = 'Test dump function when template_name already ends with .json.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump function raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.replay.make_sure_path_exists'



# Parsed testcases at query #70
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads and validates json context from file.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function raises error when file does not exist.'
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'invalid_template'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = 'invalid_template.json'

def test_case_0():
    var_0 = 'Test load function handles template name with .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function raises error for invalid json content.'
    var_1 = 'invalid_json'
    var_2 = 'invalid_json.json'
    var_3 = 'invalid json content {'



# Parsed testcases at query #71
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads and validates json context from file.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load raises ValueError when context missing cookiecutter key.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = "Test load raises FileNotFoundError when replay file doesn't exist."
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load works when template_name already has .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'



# Parsed testcases at query #72
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads json data from file correctly.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function with template_name already containing .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json.json'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function raises FileNotFoundError when file does not exist.'
    var_1 = 'nonexistent_template'



# Parsed testcases at query #73
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads and validates json context from file.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'test_project'
    var_6 = 'test_author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load function raises JSONDecodeError when file contains invalid json.'
    var_1 = 'test_template'
    var_2 = 'invalid json content {]'

def test_case_0():
    var_0 = 'Test load function works with template name that already has .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #74
#--------------------------


def test_case_0():
    var_0 = 'Test dump function when template_name already has .json suffix.'
    var_1 = 'replay'
    var_2 = 'my_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'

def test_case_0():
    var_0 = 'Test dump raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'other_key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.replay.make_sure_path_exists'

def test_case_0():
    var_0 = 'Test dump calls make_sure_path_exists.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.replay.make_sure_path_exists'
    var_7 = 'builtins.open'



# Parsed testcases at query #75
#--------------------------


def test_case_0():
    var_0 = 'Test load works with template name that already has .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = "Test load raises FileNotFoundError when replay file doesn't exist."
    var_1 = 'nonexistent_template'



# Parsed testcases at query #76
#--------------------------


def test_case_0():
    var_0 = 'Test dump function writes context to json file.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'author'
    var_6 = 'test_project'
    var_7 = 'test_author'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = 'my_template.json'

def test_case_0():
    var_0 = 'Test dump function when template name already has .json suffix.'
    var_1 = 'replay'
    var_2 = 'my_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'

def test_case_0():
    var_0 = 'Test dump function raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = "Test dump function creates replay directory if it doesn't exist."
    var_1 = 'replay'
    var_2 = 'nested'
    var_3 = 'my_template'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}

def test_case_0():
    var_0 = 'Test dump function overwrites existing replay file.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'old_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'new_project'
    var_9 = {var_4: var_8}
    var_10 = {var_3: var_9}
    var_11 = 'my_template.json'



# Parsed testcases at query #77
#--------------------------


def test_case_0():
    var_0 = 'Test dump function writes context to json file correctly.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'author'
    var_6 = 'test_project'
    var_7 = 'Test Author'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = 'cookiecutter.replay.make_sure_path_exists'

def test_case_0():
    var_0 = 'Test dump function creates a json file with correct content.'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'test_project'
    var_6 = 'Test Author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'my_template.json'

def test_case_0():
    var_0 = 'Test dump function adds .json extension if not present.'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'my_template.json'

def test_case_0():
    var_0 = "Test dump function doesn't add double .json extension."
    var_1 = 'my_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'my_template.json'
    var_8 = 'my_template.json.json'

def test_case_0():
    var_0 = 'Test dump raises ValueError when cookiecutter key is missing.'
    var_1 = 'my_template'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'Test dump works with Path objects.'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'my_template.json'

def test_case_0():
    var_0 = 'Test dump overwrites existing replay file.'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value1'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'value2'
    var_8 = {var_3: var_7}
    var_9 = {var_2: var_8}
    var_10 = 'my_template.json'



# Parsed testcases at query #78
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads and validates context from json file.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load with template name already containing .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load raises FileNotFoundError when replay file does not exist.'
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load raises JSONDecodeError for invalid json.'
    var_1 = 'test_template'
    var_2 = 'test_template.json'
    var_3 = 'invalid json content {'



# Parsed testcases at query #79
#--------------------------


def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent-template'

def test_case_0():
    var_0 = 'Test load function works when template_name already has .json suffix.'
    var_1 = 'test-template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'My Project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #80
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads json data from file correctly.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'project_slug'
    var_5 = 'my_project'
    var_6 = {var_3: var_5, var_4: var_5}
    var_7 = {var_2: var_6}
    var_8 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load function with template name already having .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'



# Parsed testcases at query #81
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads json data from file correctly.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author_name'
    var_5 = 'test_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function with template name already having .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'some_other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent_template'



# Parsed testcases at query #82
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads and validates json context from file.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author_name'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}

def test_case_0():
    var_0 = 'Test load raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'Test load raises FileNotFoundError when replay file does not exist.'
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load with template name that already has .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #83
#--------------------------


def test_case_0():
    var_0 = "Test dump function doesn't double .json extension."
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'other_key'
    var_4 = 'value'
    var_5 = {var_3: var_4}



# Parsed testcases at query #84
#--------------------------


def test_case_0():
    var_0 = 'Test dump raises ValueError when context lacks cookiecutter key.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'other_key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.replay.make_sure_path_exists'

def test_case_0():
    var_0 = 'Test dump handles template_name that already ends with .json.'
    var_1 = 'replay'
    var_2 = 'my_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'
    var_9 = 'my_template.json'



# Parsed testcases at query #85
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads and validates json context file.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function with template name already having .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when replay file doesn't exist."
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load function raises json.JSONDecodeError for invalid json.'
    var_1 = 'test_template'
    var_2 = 'test_template.json'
    var_3 = 'invalid json content {'



# Parsed testcases at query #86
#--------------------------


def test_case_0():
    var_0 = 'Test dump function writes context to json file.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'test_author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump function with template name already having .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = "Test dump function creates replay directory if it doesn't exist."
    var_1 = 'nonexistent'
    var_2 = 'replay'
    var_3 = 'test_template'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'my_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}

def test_case_0():
    var_0 = 'Test dump raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'Test dump overwrites existing replay file.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'old_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'new_project'
    var_8 = {var_3: var_7}
    var_9 = {var_2: var_8}
    var_10 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump function works with pathlib.Path object.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'



# Parsed testcases at query #87
#--------------------------


def test_case_0():
    var_0 = "Test load raises FileNotFoundError when replay file doesn't exist."
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load works when template_name already has .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #88
#--------------------------


def test_case_0():
    var_0 = 'Test dump function when template_name already has .json suffix.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump function raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.replay.make_sure_path_exists'

def test_case_0():
    var_0 = 'Test dump function calls make_sure_path_exists.'
    var_1 = 'new_replay_dir'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'



# Parsed testcases at query #89
#--------------------------


def test_case_0():
    var_0 = 'Test dump raises ValueError when context lacks cookiecutter key.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'invalid_key'
    var_4 = 'value'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test dump handles template names that already end with .json.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'test_template.json'
    var_9 = 'test_template.json.json'



# Parsed testcases at query #90
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads json data from file.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function with template name already having .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_slug'
    var_4 = 'test_slug'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'invalid_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function raises FileNotFoundError when replay file does not exist.'
    var_1 = 'nonexistent_template'



# Parsed testcases at query #91
#--------------------------


def test_case_0():
    var_0 = 'Test dump function handles template names ending with .json.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump function raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}



# Parsed testcases at query #92
#--------------------------


def test_case_0():
    var_0 = 'Test dump function writes context to json file correctly.'
    var_1 = 'replay'
    var_2 = 'my-template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'author'
    var_6 = 'test_project'
    var_7 = 'Test Author'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = 'my-template.json'

def test_case_0():
    var_0 = "Test dump function creates replay directory if it doesn't exist."
    var_1 = 'nonexistent'
    var_2 = 'replay'
    var_3 = 'my-template'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'my-template.json'

def test_case_0():
    var_0 = 'Test dump function handles template names that already have .json extension.'
    var_1 = 'replay'
    var_2 = 'my-template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my-template.json'

def test_case_0():
    var_0 = 'Test dump function raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'my-template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test dump function overwrites existing replay file.'
    var_1 = 'replay'
    var_2 = 'my-template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'old_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'new_project'
    var_9 = {var_4: var_8}
    var_10 = {var_3: var_9}
    var_11 = 'my-template.json'

def test_case_0():
    var_0 = 'Test dump function handles complex nested context structures.'
    var_1 = 'replay'
    var_2 = 'my-template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'nested'
    var_6 = 'list_of_dicts'
    var_7 = 'test_project'
    var_8 = 'key1'
    var_9 = 'key2'
    var_10 = 'value1'
    var_11 = 'item1'
    var_12 = 'item2'
    var_13 = [var_11, var_12]
    var_14 = {var_8: var_10, var_9: var_13}
    var_15 = 'id'
    var_16 = 'name'
    var_17 = 1
    var_18 = 'first'
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = 2
    var_21 = 'second'
    var_22 = {var_15: var_20, var_16: var_21}
    var_23 = [var_19, var_22]
    var_24 = {var_4: var_7, var_5: var_14, var_6: var_23}
    var_25 = {var_3: var_24}
    var_26 = 'my-template.json'



# Parsed testcases at query #93
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads json data from file correctly.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'
    var_10 = 2

def test_case_0():
    var_0 = 'Test load function handles template names with .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'
    var_8 = 2

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'
    var_6 = 2

def test_case_0():
    var_0 = 'Test load function raises FileNotFoundError when file does not exist.'
    var_1 = 'nonexistent_template'



# Parsed testcases at query #94
#--------------------------


def test_case_0():
    var_0 = 'Test load function works with template_name already having .json extension.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = 'Test load function raises FileNotFoundError when file does not exist.'
    var_1 = 'replay'
    var_2 = 'nonexistent_template'



# Parsed testcases at query #95
#--------------------------


def test_case_0():
    var_0 = 'Test dump function with template name already having .json extension.'
    var_1 = 'replay'
    var_2 = 'my-template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = 'Test dump function raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'my-template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = "Test dump function creates replay directory if it doesn't exist."
    var_1 = 'nonexistent'
    var_2 = 'replay'
    var_3 = 'my-template'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}



# Parsed testcases at query #96
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads json data from file correctly.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function with template name already having .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent_template'



# Parsed testcases at query #97
#--------------------------


def test_case_0():
    var_0 = 'Test dump function with template name already having .json extension.'
    var_1 = 'replay'
    var_2 = 'my_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'
    var_9 = 'my_template.json'

def test_case_0():
    var_0 = 'Test dump raises ValueError when context missing cookiecutter key.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'other_key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.replay.make_sure_path_exists'

def test_case_0():
    var_0 = 'Test dump calls make_sure_path_exists to create replay directory.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'
    var_9 = 'builtins.open'
    var_10 = 'json.dump'



# Parsed testcases at query #98
#--------------------------


def test_case_0():
    var_0 = 'Test load function with template name that already has .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = "Test load raises FileNotFoundError when replay file doesn't exist."
    var_1 = 'nonexistent_template'



# Parsed testcases at query #99
#--------------------------


def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load function works with template name already containing .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #100
#--------------------------


def test_case_0():
    var_0 = 'Test load raises FileNotFoundError when replay file does not exist.'
    var_1 = 'nonexistent-template'

def test_case_0():
    var_0 = 'Test load works with template name that already has .json extension.'
    var_1 = 'test-template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #101
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads and validates JSON context from replay file.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function with template name already having .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function raises FileNotFoundError when replay file does not exist.'
    var_1 = 'nonexistent_template'



# Parsed testcases at query #102
#--------------------------


def test_case_0():
    var_0 = "Test that load raises FileNotFoundError when replay file doesn't exist."
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load works correctly when template_name already has .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #103
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads json data from file correctly.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load function works with template name that already has .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'



# Parsed testcases at query #104
#--------------------------


def test_case_0():
    var_0 = 'Test dump function with template name already ending in .json.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'Test dump raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'Test load function works with template name that already has .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'Test load function raises FileNotFoundError when file does not exist.'
    var_1 = 'nonexistent_template'



# Parsed testcases at query #2
#--------------------------




# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'Test dump function with template name already having .json extension.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump function raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = "Test dump function creates replay directory if it doesn't exist."
    var_1 = 'replay'
    var_2 = 'nested'
    var_3 = 'test_template'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'Test dump raises ValueError when context lacks cookiecutter key.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.replay.make_sure_path_exists'

def test_case_0():
    var_0 = 'Test dump handles template_name that already ends with .json.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'
    var_9 = 'test_template.json'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'Test dump raises ValueError when context lacks cookiecutter key.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'some_key'
    var_4 = 'some_value'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = "Test dump creates replay directory if it doesn't exist."
    var_1 = 'nonexistent'
    var_2 = 'replay'
    var_3 = 'test_template'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'cookiecutter.replay.make_sure_path_exists'

def test_case_0():
    var_0 = "Test dump doesn't add .json suffix if template_name already has it."
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'test_template.json'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'Test dump function writes context to json file.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'My Project'
    var_6 = 'Test Author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump function handles template names with .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'My Project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump raises ValueError when context missing cookiecutter key.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'My Project'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = "Test dump function creates replay directory if it doesn't exist."
    var_1 = 'nonexistent'
    var_2 = 'nested'
    var_3 = 'test_template'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'My Project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump function overwrites existing replay file.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'Project 1'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'Project 2'
    var_8 = {var_3: var_7}
    var_9 = {var_2: var_8}
    var_10 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump function writes properly formatted json with indentation.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'My Project'
    var_6 = 'Test Author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'Test dump raises ValueError when context lacks cookiecutter key.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'other_key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.replay.make_sure_path_exists'

def test_case_0():
    var_0 = 'Test dump handles template names that already end with .json.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'
    var_9 = 'test_template.json'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'Test dump function when template_name already ends with .json.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump function raises ValueError when context lacks cookiecutter key.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.replay.make_sure_path_exists'

def test_case_0():
    var_0 = 'Test dump function calls make_sure_path_exists.'
    var_1 = 'new_replay_dir'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads json data from file correctly.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'project_slug'
    var_5 = 'my_project'
    var_6 = {var_3: var_5, var_4: var_5}
    var_7 = {var_2: var_6}
    var_8 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function works when template_name already has .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'author'
    var_4 = 'Test Author'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent_template'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'Test dump function writes context to json file.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'author'
    var_6 = 'test_project'
    var_7 = 'Test Author'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = 'my_template.json'

def test_case_0():
    var_0 = 'Test dump function with template name already having .json suffix.'
    var_1 = 'replay'
    var_2 = 'my_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'

def test_case_0():
    var_0 = 'Test dump raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = "Test dump creates replay directory if it doesn't exist."
    var_1 = 'new_replay_dir'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = 'Test dump writes valid JSON with proper formatting.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'nested'
    var_6 = 'test_project'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = {var_4: var_6, var_5: var_9}
    var_11 = {var_3: var_10}
    var_12 = 'my_template.json'



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'Test dump function writes context to json file.'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'test_project'
    var_6 = 'Test Author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'my_template.json'

def test_case_0():
    var_0 = 'Test dump function when template_name already has .json suffix.'
    var_1 = 'my_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'my_template.json'

def test_case_0():
    var_0 = 'Test dump function raises ValueError when cookiecutter key is missing.'
    var_1 = 'my_template'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = "Test dump function creates replay directory if it doesn't exist."
    var_1 = 'new_replay_dir'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'

def test_case_0():
    var_0 = 'Test dump function writes properly formatted JSON with indentation.'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'test_project'
    var_6 = 'Test Author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'my_template.json'



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'Test load raises FileNotFoundError when file does not exist.'
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load works correctly when template_name already has .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'Test dump raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.replay.make_sure_path_exists'

def test_case_0():
    var_0 = 'Test dump handles template names with .json suffix.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump calls make_sure_path_exists to create replay directory.'
    var_1 = 'nonexistent_replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'
    var_9 = 'builtins.open'
    var_10 = 'json.dump'



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'Test load raises FileNotFoundError when replay file does not exist.'
    var_1 = 'nonexistent-template'

def test_case_0():
    var_0 = 'Test load works with template name that already has .json extension.'
    var_1 = 'test-template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads json data from file correctly.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}

def test_case_0():
    var_0 = 'Test load function works with template names that already have .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when replay file doesn't exist."
    var_1 = 'nonexistent_template'



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'Test dump function handles template names ending with .json.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'

def test_case_0():
    var_0 = 'Test dump function raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.replay.make_sure_path_exists'

def test_case_0():
    var_0 = 'Test dump function calls make_sure_path_exists.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads json data from file correctly.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'
    var_10 = 2

def test_case_0():
    var_0 = 'Test load function with template_name already having .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'
    var_8 = 2

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'
    var_6 = 2

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent_template'



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads json data from file correctly.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'project_slug'
    var_5 = 'my_project'
    var_6 = {var_3: var_5, var_4: var_5}
    var_7 = {var_2: var_6}
    var_8 = 'test_template.json'
    var_9 = 2

def test_case_0():
    var_0 = 'Test load function with template name already having .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'
    var_8 = 2

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'
    var_6 = 2

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent_template'



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'Test dump function handles template name already ending with .json.'
    var_1 = 'replay'
    var_2 = 'my-template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = 'Test dump raises ValueError when context lacks cookiecutter key.'
    var_1 = 'replay'
    var_2 = 'my-template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = "Test dump creates replay directory if it doesn't exist."
    var_1 = 'replay'
    var_2 = 'nested'
    var_3 = 'my-template'
    var_4 = 'cookiecutter'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'cookiecutter.replay.make_sure_path_exists'



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'Test dump function with template name already having .json suffix.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = "Test dump creates replay directory if it doesn't exist."
    var_1 = 'nonexistent'
    var_2 = 'replay'
    var_3 = 'test_template'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'my_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}



# Parsed testcases at query #21
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test dump function doesn't double .json extension."
    var_1 = 'replay'
    var_2 = 'test-template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'
    var_9 = 'test-template.json'
    var_10 = 'utf-8'
    var_11 = 2
    var_12 = module_0.dumps(var_7, indent=var_11)

def test_case_0():
    var_0 = 'Test dump raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'test-template'
    var_3 = 'other_key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.replay.make_sure_path_exists'

def test_case_0():
    var_0 = 'Test dump calls make_sure_path_exists with correct directory.'
    var_1 = 'replay'
    var_2 = 'test-template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'Test dump function handles template_name ending with .json.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = 'Test dump raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = "Test dump creates replay directory if it doesn't exist."
    var_1 = 'nonexistent'
    var_2 = 'replay'
    var_3 = 'test_template'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}

def test_case_0():
    var_0 = 'Test dump with complex nested context.'
    var_1 = 'replay'
    var_2 = 'complex_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'options'
    var_6 = 'dependencies'
    var_7 = 'my_project'
    var_8 = 'use_docker'
    var_9 = 'python_version'
    var_10 = True
    var_11 = '3.9'
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = 'pytest'
    var_14 = 'black'
    var_15 = [var_13, var_14]
    var_16 = {var_4: var_7, var_5: var_12, var_6: var_15}
    var_17 = {var_3: var_16}



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = "Test dump adds .json extension when template_name doesn't have it."
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'
    var_9 = 'my_template.json'

def test_case_0():
    var_0 = "Test dump doesn't add duplicate .json extension."
    var_1 = 'replay'
    var_2 = 'my_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'
    var_9 = 'my_template.json'

def test_case_0():
    var_0 = 'Test dump raises ValueError when context lacks cookiecutter key.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.replay.make_sure_path_exists'

def test_case_0():
    var_0 = "Test dump creates replay directory if it doesn't exist."
    var_1 = 'new_replay_dir'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump preserves complex context structure.'
    var_1 = 'replay'
    var_2 = 'complex_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'nested'
    var_6 = 'number'
    var_7 = 'my_project'
    var_8 = 'key1'
    var_9 = 'key2'
    var_10 = 'value1'
    var_11 = 'item1'
    var_12 = 'item2'
    var_13 = [var_11, var_12]
    var_14 = {var_8: var_10, var_9: var_13}
    var_15 = 42
    var_16 = {var_4: var_7, var_5: var_14, var_6: var_15}
    var_17 = {var_3: var_16}
    var_18 = 'cookiecutter.replay.make_sure_path_exists'



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'Test dump raises ValueError when context missing cookiecutter key.'
    var_1 = 'replay'
    var_2 = 'test-template'
    var_3 = 'other_key'
    var_4 = 'value'
    var_5 = {var_3: var_4}



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads json data from file correctly.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load function works with template names that already have .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #26
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test load function with template name already having .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dumps(var_6)
    var_8 = 'utf-8'

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent_template'



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads and validates json replay file.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function with .json suffix in template name.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when replay file doesn't exist."
    var_1 = 'nonexistent_template'



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'Test dump function writes context to json file correctly.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump function when template_name already has .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = "Test dump function creates replay directory if it doesn't exist."
    var_1 = 'new_dir'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump function writes valid JSON with proper formatting.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'nested'
    var_5 = 'my_project'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = {var_2: var_9}
    var_11 = 'test_template.json'



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 'Test dump function handles template names with .json suffix.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = 'Test dump function raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}



# Parsed testcases at query #30
#--------------------------




# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 'Test load function with template name already containing .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'Test load raises FileNotFoundError when replay file does not exist.'
    var_1 = 'nonexistent_template'



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = "Test dump function doesn't duplicate .json extension."
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'other_key'
    var_4 = 'value'
    var_5 = {var_3: var_4}



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads json data from file correctly.'
    var_1 = 'test-template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'My Project'
    var_6 = 'Test Author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test-template.json'

def test_case_0():
    var_0 = 'Test load function with .json extension in template name.'
    var_1 = 'test-template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'My Project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test-template.json'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test-template'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'test-template.json'

def test_case_0():
    var_0 = 'Test load function raises FileNotFoundError when replay file does not exist.'
    var_1 = 'non-existent-template'



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = 'Test the load function.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function when template_name already has .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent_template'



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads and validates context from json file.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = "Test load raises FileNotFoundError when replay file doesn't exist."
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load works when template_name already has .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'



# Parsed testcases at query #36
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads json data from file correctly.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function raises FileNotFoundError when file does not exist.'
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load function with template name that already has .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'



# Parsed testcases at query #37
#--------------------------


def test_case_0():
    var_0 = 'Test dump function handles template names ending with .json.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.replay.make_sure_path_exists'

def test_case_0():
    var_0 = 'Test dump calls make_sure_path_exists with correct directory.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'



# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = 'Test load function with template_name already having .json extension.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = 'Test load raises FileNotFoundError when replay file does not exist.'
    var_1 = 'replay'
    var_2 = 'nonexistent_template'



# Parsed testcases at query #39
#--------------------------


def test_case_0():
    var_0 = "Test dump function doesn't add duplicate .json extension."
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump raises ValueError when context lacks cookiecutter key.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'other_key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.replay.make_sure_path_exists'

def test_case_0():
    var_0 = 'Test dump calls make_sure_path_exists for replay directory.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'



# Parsed testcases at query #40
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads json data from file correctly.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function with template name already having .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function raises FileNotFoundError when file does not exist.'
    var_1 = 'nonexistent_template'



# Parsed testcases at query #41
#--------------------------


def test_case_0():
    var_0 = "Test dump function doesn't add duplicate .json extension."
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'

def test_case_0():
    var_0 = 'Test dump raises ValueError when context lacks cookiecutter key.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.replay.make_sure_path_exists'

def test_case_0():
    var_0 = 'Test dump calls make_sure_path_exists with correct replay_dir.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'



# Parsed testcases at query #42
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads json data from file correctly.'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'test_project'
    var_6 = 'test_author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'my_template.json'

def test_case_0():
    var_0 = 'Test load raises ValueError when cookiecutter key is missing.'
    var_1 = 'my_template'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = 'my_template.json'

def test_case_0():
    var_0 = 'Test load function works with template name already having .json suffix.'
    var_1 = 'my_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'my_template.json'

def test_case_0():
    var_0 = "Test load raises FileNotFoundError when replay file doesn't exist."
    var_1 = 'nonexistent_template'



# Parsed testcases at query #43
#--------------------------


def test_case_0():
    var_0 = 'Test dump function adds .json extension if not present.'
    var_1 = 'replay'
    var_2 = 'my-template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'
    var_9 = 'my-template.json'

def test_case_0():
    var_0 = 'Test dump raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'my-template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.replay.make_sure_path_exists'

def test_case_0():
    var_0 = 'Test dump calls make_sure_path_exists.'
    var_1 = 'replay'
    var_2 = 'my-template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'
    var_9 = 'builtins.open'



# Parsed testcases at query #44
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads json file and returns context with cookiecutter key.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when replay file doesn't exist."
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load function with template name already having .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'



# Parsed testcases at query #45
#--------------------------


def test_case_0():
    var_0 = 'Test the load function with .json suffix in template_name.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'another_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 2

def test_case_0():
    var_0 = "Test the load function raises FileNotFoundError when replay file doesn't exist."
    var_1 = 'nonexistent_template'



# Parsed testcases at query #46
#--------------------------


def test_case_0():
    var_0 = "Test load raises FileNotFoundError when replay file doesn't exist."
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load works with template name that already has .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #47
#--------------------------


def test_case_0():
    var_0 = 'Test dump function with template name already ending in .json.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = 'Test dump function raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = "Test dump function creates replay directory if it doesn't exist."
    var_1 = 'nonexistent'
    var_2 = 'replay'
    var_3 = 'test_template'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}



# Parsed testcases at query #48
#--------------------------


def test_case_0():
    var_0 = "Test dump creates replay directory if it doesn't exist."
    var_1 = 'nonexistent'
    var_2 = 'replay'
    var_3 = 'test_template'
    var_4 = 'cookiecutter'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}

def test_case_0():
    var_0 = 'Test dump raises ValueError when context lacks cookiecutter key.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'other_key'
    var_4 = 'value'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test dump handles template_name that already ends with .json.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'test_template.json'



# Parsed testcases at query #49
#--------------------------


def test_case_0():
    var_0 = 'Test dump raises ValueError when context lacks cookiecutter key.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.replay.make_sure_path_exists'

def test_case_0():
    var_0 = 'Test dump handles template names that already have .json extension.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump calls make_sure_path_exists with correct directory.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'



# Parsed testcases at query #50
#--------------------------


def test_case_0():
    var_0 = 'Test load function works with template_name already having .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent_template'



# Parsed testcases at query #51
#--------------------------


def test_case_0():
    var_0 = 'Test load function works with template_name that already has .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json.json'

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent_template'



# Parsed testcases at query #52
#--------------------------


def test_case_0():
    var_0 = 'Test dump function writes context to json file.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'author'
    var_6 = 'test_project'
    var_7 = 'test_author'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = 'my_template.json'

def test_case_0():
    var_0 = 'Test dump function with template name already ending in .json.'
    var_1 = 'replay'
    var_2 = 'my_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'

def test_case_0():
    var_0 = 'Test dump function raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = "Test dump function creates replay directory if it doesn't exist."
    var_1 = 'replay'
    var_2 = 'subdir'
    var_3 = 'my_template'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'cookiecutter.replay.make_sure_path_exists'

def test_case_0():
    var_0 = 'Test dump function writes json with proper indentation.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'nested'
    var_6 = 'test_project'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = {var_4: var_6, var_5: var_9}
    var_11 = {var_3: var_10}
    var_12 = 'my_template.json'
    var_13 = 'utf-8'



# Parsed testcases at query #53
#--------------------------


def test_case_0():
    var_0 = 'Test load function handles template names with .json extension.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when file doesn't exist."
    var_1 = 'replay'
    var_2 = 'nonexistent_template'



# Parsed testcases at query #54
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads json data from file correctly.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function handles template names with .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function raises FileNotFoundError when file does not exist.'
    var_1 = 'nonexistent_template'



# Parsed testcases at query #55
#--------------------------


def test_case_0():
    var_0 = 'Test load works with template name that already has .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'Test load raises FileNotFoundError when file does not exist.'
    var_1 = 'nonexistent_template'



# Parsed testcases at query #56
#--------------------------


def test_case_0():
    var_0 = 'Test load function with template name that already has .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'Test load raises FileNotFoundError when replay file does not exist.'
    var_1 = 'nonexistent_template'



# Parsed testcases at query #57
#--------------------------


def test_case_0():
    var_0 = 'Test dump function handles template names ending with .json.'
    var_1 = 'replay'
    var_2 = 'my_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = 'Test dump raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = "Test dump creates replay directory if it doesn't exist."
    var_1 = 'new_replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}



# Parsed testcases at query #58
#--------------------------


def test_case_0():
    var_0 = 'Test load function reads json data from file correctly.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function with template name already having .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent_template'



# Parsed testcases at query #59
#--------------------------


def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when replay file doesn't exist."
    var_1 = 'nonexistent_template'

def test_case_0():
    var_0 = 'Test load function works with template name that already has .json extension.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #60
#--------------------------


def test_case_0():
    var_0 = 'Test dump function writes context to json file correctly.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump function when template_name already has .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump raises ValueError when context lacks cookiecutter key.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = "Test dump creates replay directory if it doesn't exist."
    var_1 = 'nonexistent'
    var_2 = 'replay'
    var_3 = 'test_template'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'my_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump works with Path object as replay_dir.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump writes json with proper indentation.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'



# Parsed testcases at query #61
#--------------------------


def test_case_0():
    var_0 = 'Test dump function writes context to json file correctly.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}

def test_case_0():
    var_0 = 'Test dump function with template_name already having .json suffix.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'Test dump function raises ValueError when context lacks cookiecutter key.'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = "Test dump function creates replay directory if it doesn't exist."
    var_1 = 'new_dir'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = 'Test dump function writes properly formatted json with indent.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'nested'
    var_5 = 'my_project'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = {var_2: var_9}



# Parsed testcases at query #62
#--------------------------


def test_case_0():
    var_0 = 'Test dump function writes context to JSON file.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'author'
    var_6 = 'test_project'
    var_7 = 'Test Author'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = 'my_template.json'

def test_case_0():
    var_0 = "Test dump function creates replay directory if it doesn't exist."
    var_1 = 'non_existent'
    var_2 = 'replay'
    var_3 = 'template'
    var_4 = 'cookiecutter'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'template.json'

def test_case_0():
    var_0 = 'Test dump function handles template names with .json suffix.'
    var_1 = 'replay'
    var_2 = 'my_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'

def test_case_0():
    var_0 = 'Test dump function raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'template'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test dump function overwrites existing replay file.'
    var_1 = 'replay'
    var_2 = 'template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'old_value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'new_value'
    var_9 = {var_4: var_8}
    var_10 = {var_3: var_9}
    var_11 = 'template.json'

def test_case_0():
    var_0 = 'Test dump function with complex nested context.'
    var_1 = 'replay'
    var_2 = 'complex_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'nested'
    var_6 = 'list'
    var_7 = 'boolean'
    var_8 = 'null_value'
    var_9 = 'my_project'
    var_10 = 'level1'
    var_11 = 'level2'
    var_12 = 'item1'
    var_13 = 'item2'
    var_14 = [var_12, var_13]
    var_15 = {var_11: var_14}
    var_16 = {var_10: var_15}
    var_17 = 1
    var_18 = 2
    var_19 = 3
    var_20 = [var_17, var_18, var_19]
    var_21 = True
    var_22 = None
    var_23 = {var_4: var_9, var_5: var_16, var_6: var_20, var_7: var_21, var_8: var_22}
    var_24 = {var_3: var_23}
    var_25 = 'complex_template.json'



