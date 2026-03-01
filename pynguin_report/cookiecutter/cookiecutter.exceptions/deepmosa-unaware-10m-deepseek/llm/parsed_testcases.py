####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = "Variable 'foo' is undefined"
    var_1 = 'project_name'
    var_2 = 'Test Project'
    var_3 = {var_1: var_2}
    var_4 = 'Template variable error'
    var_5 = "Template variable error. Error message: Variable 'foo' is undefined. Context: {'project_name': 'Test Project'}"
    var_6 = 'Missing variable'
    var_7 = {}
    var_8 = 'Undefined variable'
    var_9 = 'Undefined variable. Error message: Missing variable. Context: {}'
    var_10 = "'bar' not found"
    var_11 = 'name'
    var_12 = 'version'
    var_13 = 'options'
    var_14 = 'test'
    var_15 = 1.0
    var_16 = 'a'
    var_17 = 'b'
    var_18 = [var_16, var_17]
    var_19 = {var_11: var_14, var_12: var_15, var_13: var_18}
    var_20 = 'Rendering failed'
    var_21 = "Rendering failed. Error message: 'bar' not found. Context: {'name': 'test', 'version': 1.0, 'options': ['a', 'b']}"
    var_22 = 'Test error'
    var_23 = 'key'
    var_24 = 'value'
    var_25 = {var_23: var_24}
    var_26 = 'Test'



# Parsed testcases at query #2
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.NonTemplatedInputDirException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Test message'



# Parsed testcases at query #3
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.InvalidZipRepository()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Invalid zip file format'
    var_3 = 'Test error'
    var_4 = module_0.InvalidZipRepository()
    var_5 = type(var_4)
    var_6 = var_5.__name__
    assert var_6 == 'InvalidZipRepository'



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = "Variable 'foo' is undefined"
    var_1 = 'project_name'
    var_2 = 'author'
    var_3 = 'Test Project'
    var_4 = 'John Doe'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'Template variable error occurred'
    var_7 = "Template variable error occurred. Error message: Variable 'foo' is undefined. Context: {'project_name': 'Test Project', 'author': 'John Doe'}"
    var_8 = 'Missing variable'
    var_9 = {}
    var_10 = 'No variables defined'
    var_11 = 'No variables defined. Error message: Missing variable. Context: {}'
    var_12 = "'config.database.host' not found"
    var_13 = 'project'
    var_14 = 'settings'
    var_15 = 'name'
    var_16 = 'version'
    var_17 = 'MyApp'
    var_18 = '1.0'
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = 'debug'
    var_21 = 'production'
    var_22 = [var_20, var_21]
    var_23 = {var_13: var_19, var_14: var_22}
    var_24 = 'Configuration error in template'
    var_25 = "Configuration error in template. Error message: 'config.database.host' not found. Context: {'project': {'name': 'MyApp', 'version': '1.0'}, 'settings': ['debug', 'production']}"
    var_26 = "Variable 'user-name' contains invalid characters"
    var_27 = 'user'
    var_28 = 'test@example.com'
    var_29 = {var_27: var_28}
    var_30 = 'Validation failed: user-name'
    var_31 = "Validation failed: user-name. Error message: Variable 'user-name' contains invalid characters. Context: {'user': 'test@example.com'}"
    var_32 = 'Test error'
    var_33 = 'test'
    var_34 = 'value'
    var_35 = {var_33: var_34}
    var_36 = 'Test message'



# Parsed testcases at query #5
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.InvalidConfiguration()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Invalid configuration file format'
    var_3 = str(var_0)
    var_4 = 'Custom error'
    var_5 = 'Test error'
    var_6 = ''
    var_7 = str(var_0)
    assert var_7 == ''
    var_8 = 'Error: Invalid config @ line 5'
    var_9 = str(var_0)



# Parsed testcases at query #6
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.EmptyDirNameException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Directory name cannot be empty'



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = "'foo' is undefined"
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = 'Variable not defined in template'
    var_5 = "Variable not defined in template. Error message: 'foo' is undefined. Context: {'project_name': 'test_project'}"



# Parsed testcases at query #8
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.ConfigDoesNotExistException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Config file not found at /path/to/config.yaml'
    var_3 = 'Test error'



# Parsed testcases at query #9
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.InvalidModeException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Test custom message'
    var_3 = 'Test raising'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'Hook script failed'



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = "Variable 'foo' is undefined"
    var_1 = 'project_name'
    var_2 = 'author'
    var_3 = 'Test Project'
    var_4 = 'Test Author'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'Template variable error'
    var_7 = "Template variable error. Error message: Variable 'foo' is undefined. Context: {'project_name': 'Test Project', 'author': 'Test Author'}"
    var_8 = 'Missing variable'
    var_9 = 'Another error'
    var_10 = {}
    var_11 = "'undefined' is undefined"
    var_12 = 'config'
    var_13 = 'list'
    var_14 = 'key'
    var_15 = 'value'
    var_16 = {var_14: var_15}
    var_17 = 1
    var_18 = 2
    var_19 = 3
    var_20 = [var_17, var_18, var_19]
    var_21 = {var_12: var_16, var_13: var_20}
    var_22 = 'Complex template error'



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'Config file not found'



# Parsed testcases at query #13
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.ConfigDoesNotExistException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Config file not found at /path/to/config.yaml'
    var_3 = 'Error'
    var_4 = 'additional'
    var_5 = 'info'



# Parsed testcases at query #14
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.ConfigDoesNotExistException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Config file not found at /path/to/config.yaml'
    var_3 = 'Test error'



# Parsed testcases at query #15
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.InvalidZipRepository()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Invalid zip file format'
    var_3 = 'Error'
    var_4 = 'Additional info'



# Parsed testcases at query #16
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.NonTemplatedInputDirException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Test error message'



# Parsed testcases at query #17
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.UnknownTemplateDirException()
    var_1 = 'Custom error message'
    var_2 = 'Test error'



# Parsed testcases at query #18
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.InvalidZipRepository()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Invalid zip repository format'
    var_3 = 'Error'
    var_4 = 'Additional info'



# Parsed testcases at query #19
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.NonTemplatedInputDirException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Input directory should be templated'



# Parsed testcases at query #20
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.InvalidZipRepository()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Invalid zip file format'
    var_3 = str(var_0)
    var_4 = 'Invalid zip'
    var_5 = 'Corrupted archive'
    var_6 = str(var_0)
    assert var_6 == "('Invalid zip', 'Corrupted archive')"



# Parsed testcases at query #21
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test extension error'
    var_1 = module_0.UnknownExtension()
    var_2 = str(var_1)
    assert var_2 == ''
    var_3 = 'Custom error message'
    var_4 = ''
    var_5 = 'Test'



# Parsed testcases at query #22
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.RepositoryNotFound()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Repository not found at specified location'
    var_3 = 'Test error'
    var_4 = 'Another test'



# Parsed testcases at query #23
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.UnknownRepoType()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Repository type could not be determined'
    var_3 = 'Test error'
    var_4 = 'Error'
    var_5 = 'additional'
    var_6 = 'args'



# Parsed testcases at query #24
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.CookiecutterException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Test error message'
    var_3 = 'Test raising'



# Parsed testcases at query #25
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.InvalidZipRepository()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Invalid zip file format'
    var_3 = 'Error'
    var_4 = 'Additional info'



# Parsed testcases at query #26
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.RepositoryNotFound()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Repository not found at specified location'
    var_3 = str(var_0)
    var_4 = 'Test error'



# Parsed testcases at query #27
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test extension error'
    var_1 = module_0.UnknownExtension()
    var_2 = str(var_1)
    assert var_2 == ''
    var_3 = 'Custom error message'



# Parsed testcases at query #28
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.InvalidZipRepository()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Invalid zip file format'



# Parsed testcases at query #29
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.MissingProjectDir()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = "Project directory 'my_project' not found"
    var_3 = 'Test error'
    var_4 = 'Error'
    var_5 = 'additional'
    var_6 = 'args'



# Parsed testcases at query #30
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.ConfigDoesNotExistException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Config file not found at /path/to/config.yaml'
    var_3 = str(var_0)
    var_4 = 'Test error'



# Parsed testcases at query #31
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.UnknownRepoType()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Custom error message'
    var_3 = str(var_0)
    var_4 = 'Test error'
    var_5 = module_0.UnknownRepoType()



# Parsed testcases at query #32
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.OutputDirExistsException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Output directory already exists at /path/to/dir'
    var_3 = 'Test message'



# Parsed testcases at query #33
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.InvalidZipRepository()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Invalid zip repository: corrupted archive'
    var_3 = 'Error'
    var_4 = 'additional'
    var_5 = 'info'



# Parsed testcases at query #34
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.CookiecutterException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Test error message'
    var_3 = 'Error'
    var_4 = 'code'
    var_5 = 500
    var_6 = 'Test exception'



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = 'Failed to clone repository'
    var_1 = 'Cannot clone from https://example.com/template.git'
    var_2 = 'Test error'
    var_3 = ''
    var_4 = "Error: Cannot clone repo with name 'my-template@v1.0'"



# Parsed testcases at query #36
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.EmptyDirNameException()
    var_1 = 'Directory name cannot be empty'
    var_2 = 'Test error'
    var_3 = module_0.EmptyDirNameException()



# Parsed testcases at query #37
#--------------------------


def test_case_0():
    var_0 = 'Hook script failed'



# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = 'Hook script failed'



# Parsed testcases at query #39
#--------------------------


def test_case_0():
    var_0 = 'Config file not found'



# Parsed testcases at query #40
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.UnknownRepoType()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Custom error message'
    var_3 = 'Error'
    var_4 = 'additional'
    var_5 = 'args'



# Parsed testcases at query #41
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.CookiecutterException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Test error message'
    var_3 = 'Test raising'



# Parsed testcases at query #42
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.VCSNotInstalled()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Git is not installed on this system'
    var_3 = 'Test error'



# Parsed testcases at query #43
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test message'
    var_1 = module_0.CookiecutterException()
    var_2 = str(var_1)
    assert var_2 == ''
    var_3 = 'Error occurred'



# Parsed testcases at query #44
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test extension error'
    var_1 = module_0.UnknownExtension()
    var_2 = str(var_1)
    assert var_2 == ''
    var_3 = 'Custom error message'
    var_4 = ''
    var_5 = str(var_1)
    assert var_5 == ''



# Parsed testcases at query #45
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.InvalidConfiguration()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Invalid configuration file format'
    var_3 = 'Test error'



# Parsed testcases at query #46
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.ConfigDoesNotExistException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Config file not found'



# Parsed testcases at query #47
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.ContextDecodingException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Failed to decode JSON context'



# Parsed testcases at query #48
#--------------------------




# Parsed testcases at query #49
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.MissingProjectDir()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Project directory not found'
    var_3 = str(var_0)
    var_4 = 'The generated project directory was not found during cleanup'
    var_5 = str(var_0)
    var_6 = 'Test error'
    var_7 = module_0.MissingProjectDir()



# Parsed testcases at query #50
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.RepositoryNotFound()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Repository not found at specified location'
    var_3 = 'Test error'
    var_4 = 'Parent catch test'
    var_5 = 'Base catch test'



# Parsed testcases at query #51
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.UnknownExtension()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Test extension not found'
    var_3 = str(var_0)
    var_4 = 'Original error'



# Parsed testcases at query #52
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.RepositoryNotFound()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Repository not found at specified location'
    var_3 = str(var_0)
    var_4 = 'Test error'



# Parsed testcases at query #53
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.InvalidConfiguration()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Configuration file is malformed'
    var_3 = 'Invalid YAML at line 5: unexpected indentation'



# Parsed testcases at query #54
#--------------------------


def test_case_0():
    var_0 = 'Test message'



# Parsed testcases at query #55
#--------------------------


def test_case_0():
    var_0 = 'Hook script failed'



# Parsed testcases at query #56
#--------------------------


def test_case_0():
    var_0 = 'JSON decoding failed'
    var_1 = ''
    var_2 = "Failed to decode: {'key': 'value'}"
    var_3 = 'Test error'
    var_4 = 'Test'



# Parsed testcases at query #57
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.MissingProjectDir()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Project directory not found at expected location'
    var_3 = 'Test error'



# Parsed testcases at query #58
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.NonTemplatedInputDirException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Test error message'



# Parsed testcases at query #59
#--------------------------


def test_case_0():
    var_0 = 'Test message'



# Parsed testcases at query #60
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test message'
    var_1 = module_0.CookiecutterException()
    var_2 = str(var_1)
    assert var_2 == ''
    var_3 = 'Custom error'



# Parsed testcases at query #61
#--------------------------


def test_case_0():
    var_0 = "Variable 'foo' is undefined"
    var_1 = 'project_name'
    var_2 = 'author'
    var_3 = 'Test Project'
    var_4 = 'Test Author'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'Template variable error occurred'
    var_7 = "Template variable error occurred. Error message: Variable 'foo' is undefined. Context: {'project_name': 'Test Project', 'author': 'Test Author'}"
    var_8 = 'Missing variable'
    var_9 = {}
    var_10 = 'No variables defined'
    var_11 = 'No variables defined. Error message: Missing variable. Context: {}'
    var_12 = "'user.profile.name' not found"
    var_13 = 'user'
    var_14 = 'settings'
    var_15 = 'id'
    var_16 = 'email'
    var_17 = 123
    var_18 = 'test@example.com'
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = 'theme'
    var_21 = 'notifications'
    var_22 = 'dark'
    var_23 = True
    var_24 = {var_20: var_22, var_21: var_23}
    var_25 = {var_13: var_19, var_14: var_24}
    var_26 = 'Nested variable access failed'
    var_27 = "Nested variable access failed. Error message: 'user.profile.name' not found. Context: {'user': {'id': 123, 'email': 'test@example.com'}, 'settings': {'theme': 'dark', 'notifications': True}}"
    var_28 = "Variable 'test-data' (with-hyphens) is undefined"
    var_29 = 'simple'
    var_30 = 'value'
    var_31 = {var_29: var_30}
    var_32 = 'Special character error'
    var_33 = "Special character error. Error message: Variable 'test-data' (with-hyphens) is undefined. Context: {'simple': 'value'}"
    var_34 = 'Test error'
    var_35 = 'key'
    var_36 = {var_35: var_30}
    var_37 = 'Test message'



# Parsed testcases at query #62
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.InvalidConfiguration()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Invalid configuration file format'
    var_3 = 'Invalid YAML at line 5: unexpected indentation'
    var_4 = 'Test error'



# Parsed testcases at query #63
#--------------------------


def test_case_0():
    var_0 = 'Hook script failed'



# Parsed testcases at query #64
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.MissingProjectDir()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Project directory not found at expected location'
    var_3 = 'Test error'



# Parsed testcases at query #65
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test extension error'
    var_1 = module_0.UnknownExtension()
    var_2 = str(var_1)
    assert var_2 == ''
    var_3 = 'Custom message'



# Parsed testcases at query #66
#--------------------------


def test_case_0():
    var_0 = 'Test message'
    var_1 = 'Template error occurred'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = str(var_4)
    var_6 = 'nested'
    var_7 = 'list'
    var_8 = 'inner'
    var_9 = 'data'
    var_10 = {var_8: var_9}
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = [var_11, var_12, var_13]
    var_15 = {var_6: var_10, var_7: var_14}
    var_16 = 'Another message'



# Parsed testcases at query #67
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.InvalidModeException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Test error message'
    var_3 = 'Test raising'
    var_4 = 'Parent catch test'



# Parsed testcases at query #68
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.UnknownExtension()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Custom extension error message'
    var_3 = 'Test error'



# Parsed testcases at query #69
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.UnknownRepoType()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Custom repository type error'
    var_3 = 'Test error'



# Parsed testcases at query #70
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.RepositoryNotFound()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Repository not found at specified location'
    var_3 = str(var_0)
    var_4 = 'Test error'



# Parsed testcases at query #71
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.InvalidZipRepository()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Invalid zip repository: corrupted archive'
    var_3 = str(var_0)
    var_4 = 'Error'
    var_5 = 'Additional info'
    var_6 = str(var_0)
    assert var_6 == "('Error', 'Additional info')"



# Parsed testcases at query #72
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.VCSNotInstalled()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Git is not installed'



# Parsed testcases at query #73
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.RepositoryCloneFailed()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Failed to clone repository'
    var_3 = str(var_0)
    var_4 = 'Error'
    var_5 = 'Additional info'
    var_6 = str(var_0)
    assert var_6 == "('Error', 'Additional info')"



# Parsed testcases at query #74
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.UnknownRepoType()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Custom error message'
    var_3 = 'Test error'



# Parsed testcases at query #75
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.RepositoryNotFound()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Repository not found at specified location'
    var_3 = 'Test error'
    var_4 = 'Error'
    var_5 = 'additional'
    var_6 = 'args'



# Parsed testcases at query #76
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.ContextDecodingException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Failed to decode JSON context'



# Parsed testcases at query #77
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.ContextDecodingException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Failed to decode JSON context'



# Parsed testcases at query #78
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.InvalidModeException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Cannot use both no_input and replay modes simultaneously'
    var_3 = 'Error'
    var_4 = 'Additional info'



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.EmptyDirNameException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Directory name cannot be empty'
    var_3 = str(var_0)



# Parsed testcases at query #2
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.ContextDecodingException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Failed to decode JSON context'



# Parsed testcases at query #3
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.UnknownRepoType()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Custom error message'
    var_3 = 'arg1'
    var_4 = 'arg2'
    var_5 = 'arg3'
    var_6 = 'Test error'
    var_7 = 'Original error'
    var_8 = ValueError(var_7)



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'Test extension not found'
    var_1 = ''
    var_2 = "Extension 'custom' could not be imported"
    var_3 = 'Test error'
    var_4 = 'Original error'
    var_5 = ImportError(var_4)



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = "Variable 'foo' is undefined"
    var_1 = 'project_name'
    var_2 = 'Test Project'
    var_3 = {var_1: var_2}
    var_4 = 'Template variable error'
    var_5 = "Template variable error. Error message: Variable 'foo' is undefined. Context: {'project_name': 'Test Project'}"
    var_6 = 'Missing variable'
    var_7 = {}
    var_8 = 'Another error'
    var_9 = 'Another error. Error message: Missing variable. Context: {}'
    var_10 = "'bar' not found"
    var_11 = 'nested'
    var_12 = 'list'
    var_13 = 'key'
    var_14 = 'value'
    var_15 = {var_13: var_14}
    var_16 = 1
    var_17 = 2
    var_18 = 3
    var_19 = [var_16, var_17, var_18]
    var_20 = {var_11: var_15, var_12: var_19}
    var_21 = 'Complex error'
    var_22 = "Complex error. Error message: 'bar' not found. Context: {'nested': {'key': 'value'}, 'list': [1, 2, 3]}"



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = "Variable 'foo' is undefined"
    var_1 = 'project_name'
    var_2 = 'version'
    var_3 = 'Test Project'
    var_4 = '1.0'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'Template variable error occurred'
    var_7 = "Template variable error occurred. Error message: Variable 'foo' is undefined. Context: {'project_name': 'Test Project', 'version': '1.0'}"
    var_8 = 'Missing variable'
    var_9 = {}
    var_10 = 'Another error'
    var_11 = 'Another error. Error message: Missing variable. Context: {}'
    var_12 = "'bar' not found"
    var_13 = 'nested'
    var_14 = 'list'
    var_15 = 'number'
    var_16 = 'key'
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = 1
    var_20 = 2
    var_21 = 3
    var_22 = [var_19, var_20, var_21]
    var_23 = 42
    var_24 = {var_13: var_18, var_14: var_22, var_15: var_23}
    var_25 = 'Complex template error'
    var_26 = "Complex template error. Error message: 'bar' not found. Context: {'nested': {'key': 'value'}, 'list': [1, 2, 3], 'number': 42}"
    var_27 = 'Test error message'
    var_28 = 'simple'
    var_29 = 'context'
    var_30 = {var_28: var_29}
    var_31 = 'Test main message'



# Parsed testcases at query #7
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.NonTemplatedInputDirException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Test message'



# Parsed testcases at query #8
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.RepositoryNotFound()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Repository not found at specified location'
    var_3 = 'Test error'
    var_4 = module_0.RepositoryNotFound()



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'Hook execution failed'
    var_1 = ''
    var_2 = "Hook 'pre_gen_project' failed with exit code 1"



# Parsed testcases at query #10
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.FailedHookException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Hook script failed to execute'
    var_3 = str(var_0)
    var_4 = 'Hook failed'
    var_5 = 'pre_gen'
    var_6 = 'script.sh'
    var_7 = var_0.args
    var_8 = len(var_7)
    assert var_8 == 3
    var_9 = 'Test error'
    var_10 = module_0.FailedHookException()



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = "'my_var' is undefined"
    var_1 = 'project_name'
    var_2 = 'author'
    var_3 = 'Test Project'
    var_4 = 'Test Author'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'Variable not found in template'
    var_7 = "Variable not found in template. Error message: 'my_var' is undefined. Context: {'project_name': 'Test Project', 'author': 'Test Author'}"
    var_8 = "'another_var' is not defined"
    var_9 = {}
    var_10 = 'Missing variable'
    var_11 = "Missing variable. Error message: 'another_var' is not defined. Context: {}"
    var_12 = ''
    var_13 = 'key'
    var_14 = 'value'
    var_15 = {var_13: var_14}
    var_16 = 'Template error'
    var_17 = "Template error. Error message: . Context: {'key': 'value'}"



# Parsed testcases at query #12
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test extension error'
    var_1 = module_0.UnknownExtension()
    var_2 = str(var_1)
    assert var_2 == ''
    var_3 = 'Custom message'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = "Variable 'foo' is undefined"
    var_1 = 'project_name'
    var_2 = 'version'
    var_3 = 'Test Project'
    var_4 = '1.0'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'Template variable error occurred'
    var_7 = "Template variable error occurred. Error message: Variable 'foo' is undefined. Context: {'project_name': 'Test Project', 'version': '1.0'}"
    var_8 = 'Missing variable'
    var_9 = {}
    var_10 = 'No variables defined'
    var_11 = 'No variables defined. Error message: Missing variable. Context: {}'
    var_12 = "'user.email' not found"
    var_13 = 'user'
    var_14 = 'settings'
    var_15 = 'name'
    var_16 = 'id'
    var_17 = 'John'
    var_18 = 123
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = 'debug'
    var_21 = 'log_level'
    var_22 = True
    var_23 = 'INFO'
    var_24 = {var_20: var_22, var_21: var_23}
    var_25 = {var_13: var_19, var_14: var_24}
    var_26 = 'Configuration error'
    var_27 = "Configuration error. Error message: 'user.email' not found. Context: {'user': {'name': 'John', 'id': 123}, 'settings': {'debug': True, 'log_level': 'INFO'}}"
    var_28 = "'missing_var' is undefined"
    var_29 = 'available_var'
    var_30 = 'some_value'
    var_31 = {var_29: var_30}



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = "'foo' is undefined"
    var_1 = 'bar'
    var_2 = 'baz'
    var_3 = {var_1: var_2}
    var_4 = "Variable 'foo' is not defined"



# Parsed testcases at query #15
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.UnknownTemplateDirException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Ambiguous template directory found'
    var_3 = 'Error'
    var_4 = 'more info'
    var_5 = 123
    var_6 = 'Test error'



# Parsed testcases at query #16
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.EmptyDirNameException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Directory name cannot be empty'
    var_3 = str(var_0)
    var_4 = 'Test error'
    var_5 = module_0.EmptyDirNameException()



# Parsed testcases at query #17
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Hook script failed to execute'
    var_1 = module_0.FailedHookException()
    var_2 = str(var_1)
    assert var_2 == ''
    var_3 = 'Custom hook failure'
    var_4 = str(var_1)



# Parsed testcases at query #18
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.EmptyDirNameException()
    var_1 = 'Directory name cannot be empty'
    var_2 = 'Test error'
    var_3 = str(var_0)
    assert var_3 == ''



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'Hook script failed'



# Parsed testcases at query #20
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.InvalidModeException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Test error message'
    var_3 = str(var_0)
    var_4 = 'Test raise'



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'Test error message'
    var_1 = ''
    var_2 = "Invalid config at line 5: missing required field 'name'"
    var_3 = 'Configuration error'
    var_4 = 'Error'
    var_5 = 'additional'
    var_6 = 'args'



# Parsed testcases at query #22
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.UnknownTemplateDirException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Test error message'



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'Test message'
    var_1 = ''
    var_2 = "Invalid config: missing required field 'project_name'"



# Parsed testcases at query #24
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.MissingProjectDir()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Project directory not found at expected location'
    var_3 = 'Error'
    var_4 = 'Additional info'
    var_5 = 404



# Parsed testcases at query #25
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.RepositoryNotFound()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Repository not found at specified location'
    var_3 = str(var_0)
    var_4 = 'Test error'



# Parsed testcases at query #26
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.EmptyDirNameException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Directory name cannot be empty'



# Parsed testcases at query #27
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.RepositoryNotFound()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Repository not found at specified location'
    var_3 = str(var_0)
    var_4 = 'Test error'



# Parsed testcases at query #28
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.EmptyDirNameException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Directory name cannot be empty'
    var_3 = str(var_0)
    var_4 = 'Test error'



# Parsed testcases at query #29
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.ContextDecodingException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Failed to decode JSON context'



# Parsed testcases at query #30
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.OutputDirExistsException()
    var_1 = 'Output directory already exists'
    var_2 = 'Test message'
    var_3 = module_0.OutputDirExistsException()



# Parsed testcases at query #31
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.VCSNotInstalled()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Git is not installed'



# Parsed testcases at query #32
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.RepositoryNotFound()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Repository not found at specified location'
    var_3 = 'Test error'
    var_4 = 'Another test'



# Parsed testcases at query #33
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.ContextDecodingException()
    var_1 = 'Test error message'
    var_2 = 'JSON decoding failed'



# Parsed testcases at query #34
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.InvalidModeException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Test error message'
    var_3 = 'Test raising'
    var_4 = 'Another test'
    var_5 = ''



# Parsed testcases at query #35
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.InvalidConfiguration()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Configuration file is malformed'
    var_3 = 'Test error'



# Parsed testcases at query #36
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.RepositoryNotFound()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Repository not found at specified location'
    var_3 = 'Error'
    var_4 = 'additional'
    var_5 = 'info'



# Parsed testcases at query #37
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.InvalidModeException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Test error message'
    var_3 = str(var_0)
    var_4 = 'Test raise'



# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = 'Hook script failed'



# Parsed testcases at query #39
#--------------------------


def test_case_0():
    var_0 = 'Test message'



# Parsed testcases at query #40
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.UnknownRepoType()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Repository type could not be determined'



# Parsed testcases at query #41
#--------------------------


def test_case_0():
    var_0 = 'Clone failed'
    var_1 = 'Custom error message'



# Parsed testcases at query #42
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.InvalidConfiguration()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Invalid configuration file format'
    var_3 = 'Test error'
    var_4 = 'Error'
    var_5 = 'in'
    var_6 = 'config'



# Parsed testcases at query #43
#--------------------------


def test_case_0():
    var_0 = "Variable 'foo' is undefined"
    var_1 = 'bar'
    var_2 = 'num'
    var_3 = 'baz'
    var_4 = 42
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'Template variable error occurred'



# Parsed testcases at query #44
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.ConfigDoesNotExistException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Config file not found at /path/to/config.yaml'



# Parsed testcases at query #45
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.RepositoryCloneFailed()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Failed to clone repository from https://example.com'
    var_3 = 'Clone error'



# Parsed testcases at query #46
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.OutputDirExistsException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Output directory already exists at /path/to/dir'
    var_3 = 'Test message'



# Parsed testcases at query #47
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test message'
    var_1 = module_0.UnknownExtension()
    var_2 = str(var_1)
    assert var_2 == ''
    var_3 = 'Custom error'



# Parsed testcases at query #48
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.RepositoryNotFound()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Repository not found at specified location'
    var_3 = str(var_0)
    var_4 = 'Test error'



# Parsed testcases at query #49
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.EmptyDirNameException()
    var_1 = 'Directory name cannot be empty'
    var_2 = 'Test exception'



# Parsed testcases at query #50
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.OutputDirExistsException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Output directory already exists at /path/to/dir'
    var_3 = str(var_0)
    var_4 = 'Test message'



# Parsed testcases at query #51
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.EmptyDirNameException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Directory name cannot be empty'
    var_3 = str(var_0)
    var_4 = 'Test error'



# Parsed testcases at query #52
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.RepositoryCloneFailed()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Failed to clone repository from https://example.com/template.git'
    var_3 = 'Clone failed'
    var_4 = 'Network error'
    var_5 = 404
    var_6 = 'Test error'



# Parsed testcases at query #53
#--------------------------


def test_case_0():
    var_0 = 'Test message'



# Parsed testcases at query #54
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.UnknownRepoType()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Custom error message'
    var_3 = 'Test error'



# Parsed testcases at query #55
#--------------------------


def test_case_0():
    var_0 = "Variable 'foo' is undefined"
    var_1 = 'project_name'
    var_2 = 'author'
    var_3 = 'Test Project'
    var_4 = 'John Doe'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'Template variable error occurred'
    var_7 = "Template variable error occurred. Error message: Variable 'foo' is undefined. Context: {'project_name': 'Test Project', 'author': 'John Doe'}"
    var_8 = 'Missing variable'
    var_9 = {}
    var_10 = 'Another error'
    var_11 = 'Another error. Error message: Missing variable. Context: {}'
    var_12 = "'bar' not found"
    var_13 = 'nested'
    var_14 = 'list'
    var_15 = 'number'
    var_16 = 'key'
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = 1
    var_20 = 2
    var_21 = 3
    var_22 = [var_19, var_20, var_21]
    var_23 = 42
    var_24 = {var_13: var_18, var_14: var_22, var_15: var_23}
    var_25 = 'Complex template error'
    var_26 = "Complex template error. Error message: 'bar' not found. Context: {'nested': {'key': 'value'}, 'list': [1, 2, 3], 'number': 42}"
    var_27 = "Variable 'user-input' contains & special <chars>"
    var_28 = 'test'
    var_29 = 'data'
    var_30 = {var_28: var_29}
    var_31 = 'Special char error'
    var_32 = "Special char error. Error message: Variable 'user-input' contains & special <chars>. Context: {'test': 'data'}"



# Parsed testcases at query #56
#--------------------------


def test_case_0():
    var_0 = 'Test message'



# Parsed testcases at query #57
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.InvalidZipRepository()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Invalid zip repository format'
    var_3 = 'Bad zip file'
    var_4 = 'extra_info'



# Parsed testcases at query #58
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.CookiecutterException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Test error message'
    var_3 = 'Error'
    var_4 = 'code'
    var_5 = 500
    var_6 = 'Test raising'



# Parsed testcases at query #59
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.VCSNotInstalled()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Git is not installed'



# Parsed testcases at query #60
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.EmptyDirNameException()
    var_1 = 'Directory name cannot be empty'
    var_2 = 'Test exception'
    var_3 = ''



# Parsed testcases at query #61
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.OutputDirExistsException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Output directory already exists'
    var_3 = 'Test message'



# Parsed testcases at query #62
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.ConfigDoesNotExistException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Config file not found at /path/to/config'



# Parsed testcases at query #63
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.UnknownRepoType()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Custom repository type error'
    var_3 = 'Test error'



# Parsed testcases at query #64
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.ConfigDoesNotExistException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Config file not found at /path/to/config.yaml'



# Parsed testcases at query #65
#--------------------------


def test_case_0():
    var_0 = "Variable 'foo' is undefined"
    var_1 = 'project_name'
    var_2 = 'author'
    var_3 = 'test_project'
    var_4 = 'test_author'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'Template variable error'
    var_7 = "Template variable error. Error message: Variable 'foo' is undefined. Context: {'project_name': 'test_project', 'author': 'test_author'}"
    var_8 = 'Another error'
    var_9 = 'Missing variable'
    var_10 = {}
    var_11 = "'bar' is not defined in this context"
    var_12 = 'nested'
    var_13 = 'list'
    var_14 = 'key'
    var_15 = 'value'
    var_16 = {var_14: var_15}
    var_17 = 1
    var_18 = 2
    var_19 = 3
    var_20 = [var_17, var_18, var_19]
    var_21 = {var_12: var_16, var_13: var_20}
    var_22 = 'Complex template error'



# Parsed testcases at query #66
#--------------------------


def test_case_0():
    var_0 = 'Hook execution failed'
    var_1 = ''
    var_2 = 'Pre-gen hook failed with exit code 1'



# Parsed testcases at query #67
#--------------------------


def test_case_0():
    var_0 = 'Test message'



# Parsed testcases at query #68
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.VCSNotInstalled()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Git is not installed on this system'
    var_3 = 'Test message'



# Parsed testcases at query #69
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.NonTemplatedInputDirException()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Input directory should be templated'



# Parsed testcases at query #70
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.MissingProjectDir()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Project directory not found at expected location'
    var_3 = 'Test error'
    var_4 = ''
    var_5 = None



# Parsed testcases at query #71
#--------------------------


def test_case_0():
    var_0 = 'Clone failed due to network error'



# Parsed testcases at query #72
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.InvalidConfiguration()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Configuration file is malformed'
    var_3 = 'Test error'
    var_4 = 'Another test'



# Parsed testcases at query #73
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.InvalidZipRepository()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Invalid zip file format'
    var_3 = 'Test error'



# Parsed testcases at query #74
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.InvalidZipRepository()
    var_1 = str(var_0)
    assert var_1 == ''
    var_2 = 'Invalid zip file format'
    var_3 = 'Test message'
    var_4 = module_0.InvalidZipRepository()



# Parsed testcases at query #75
#--------------------------


def test_case_0():
    var_0 = "Variable 'username' is undefined"
    var_1 = 'project_name'
    var_2 = 'version'
    var_3 = 'Test Project'
    var_4 = '1.0'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'Template variable error occurred'
    var_7 = "Template variable error occurred. Error message: Variable 'username' is undefined. Context: {'project_name': 'Test Project', 'version': '1.0'}"
    var_8 = 'Missing variable'
    var_9 = {}
    var_10 = 'No variables defined'
    var_11 = 'No variables defined. Error message: Missing variable. Context: {}'
    var_12 = "'user.profile.name' not found"
    var_13 = 'settings'
    var_14 = 'users'
    var_15 = 'metadata'
    var_16 = 'debug'
    var_17 = 'port'
    var_18 = True
    var_19 = 8000
    var_20 = {var_16: var_18, var_17: var_19}
    var_21 = 'alice'
    var_22 = 'bob'
    var_23 = [var_21, var_22]
    var_24 = None
    var_25 = {var_13: var_20, var_14: var_23, var_15: var_24}
    var_26 = 'Complex template error'
    var_27 = "Complex template error. Error message: 'user.profile.name' not found. Context: {'settings': {'debug': True, 'port': 8000}, 'users': ['alice', 'bob'], 'metadata': None}"
    var_28 = 'Test error'
    var_29 = 'test'
    var_30 = 'value'
    var_31 = {var_29: var_30}
    var_32 = 'Test message'



# Parsed testcases at query #76
#--------------------------


def test_case_0():
    var_0 = 'Hook execution failed'
    var_1 = ''
    var_2 = "Hook 'pre_gen_project' failed with exit code 1"



