####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test UnknownExtension exception can be instantiated and raised.'
    var_1 = module_0.UnknownExtension()
    var_2 = 'Failed to import extension'
    var_3 = str(var_1)
    var_4 = 'Test extension error'
    var_5 = 'Test extension error'
    var_6 = 'Test extension error'
    var_7 = 'arg1'
    var_8 = 'arg2'



# Parsed testcases at query #2
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test InvalidConfiguration exception can be instantiated and raised.'
    var_1 = module_0.InvalidConfiguration()
    var_2 = 'Configuration file is not valid YAML'
    var_3 = str(var_1)
    var_4 = 'Invalid config'
    var_5 = 'Invalid config'
    var_6 = 'Invalid config'



# Parsed testcases at query #3
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test RepositoryCloneFailed exception initialization and inheritance.'
    var_1 = 'Failed to clone repository'
    var_2 = module_0.RepositoryCloneFailed()
    var_3 = str(var_2)
    assert var_3 == ''
    var_4 = 'Error 1'
    var_5 = 'Error 2'
    var_6 = 'Test error'
    var_7 = 'Test error'
    var_8 = 'Test error'



# Parsed testcases at query #4
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test RepositoryCloneFailed exception can be instantiated and raised.'
    var_1 = 'Test message'
    var_2 = module_0.RepositoryCloneFailed()
    var_3 = 'Clone failed'
    var_4 = 'Clone failed'
    var_5 = 'Clone failed'
    var_6 = 'Error'
    var_7 = 'Extra'
    var_8 = 'Args'



# Parsed testcases at query #5
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test ContextDecodingException can be instantiated and raised.'
    var_1 = module_0.ContextDecodingException()
    var_2 = 'Failed to decode context JSON'
    var_3 = str(var_1)
    var_4 = 'Context decoding failed'
    var_5 = 'Test error'
    var_6 = 'Error'
    var_7 = 'Additional info'



# Parsed testcases at query #6
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test RepositoryNotFound exception can be instantiated and raised.'
    var_1 = module_0.RepositoryNotFound()
    var_2 = 'Repository not found at /path/to/repo'
    var_3 = str(var_1)
    var_4 = 'test repo not found'
    var_5 = 'Custom repository error'
    var_6 = 'test'



# Parsed testcases at query #7
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test ConfigDoesNotExistException can be instantiated and raised.'
    var_1 = 'Config file not found at /path/to/config'
    var_2 = module_0.ConfigDoesNotExistException()
    var_3 = 'Test error message'



# Parsed testcases at query #8
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test OutputDirExistsException can be instantiated and raised.'
    var_1 = module_0.OutputDirExistsException()
    var_2 = 'Output directory already exists'
    var_3 = 'Test output directory exists'
    var_4 = 'Test message'
    var_5 = 'arg1'
    var_6 = 'arg2'



# Parsed testcases at query #9
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test InvalidModeException can be instantiated and raised.'
    var_1 = 'Test message'
    var_2 = 'no_input and replay cannot both be True'
    var_3 = module_0.InvalidModeException()



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'Test NonTemplatedInputDirException can be instantiated and raised.'
    var_1 = 'Test message'
    var_2 = 'Input directory is not templated'
    var_3 = 'Base class catch test'



# Parsed testcases at query #11
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test CookiecutterException can be instantiated and raised.'
    var_1 = 'Test message'
    var_2 = 'Test error'
    var_3 = module_0.CookiecutterException()



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'Test __str__ method of UndefinedVariableInTemplate exception.'
    var_1 = 'name'
    var_2 = 'version'
    var_3 = 'test_project'
    var_4 = '1.0'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'Error rendering template'

def test_case_0():
    var_0 = 'Test __str__ method with empty context.'
    var_1 = 'Template error occurred'
    var_2 = {}

def test_case_0():
    var_0 = 'Test __str__ method with complex nested context.'
    var_1 = 'project'
    var_2 = 'list_var'
    var_3 = 'name'
    var_4 = 'nested'
    var_5 = 'my_project'
    var_6 = 'value'
    var_7 = 42
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = 1
    var_11 = 2
    var_12 = 3
    var_13 = [var_10, var_11, var_12]
    var_14 = {var_1: var_9, var_2: var_13}
    var_15 = 'Complex template rendering failed'



# Parsed testcases at query #13
#--------------------------




# Parsed testcases at query #14
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test ConfigDoesNotExistException constructor and inheritance.'
    var_1 = 'Config file not found at /path/to/config.yaml'
    var_2 = 'Test error message'
    var_3 = module_0.ConfigDoesNotExistException()



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'Test UndefinedVariableInTemplate exception initialization and string representation.'
    var_1 = "Variable 'foo' is undefined"
    var_2 = 'var1'
    var_3 = 'var2'
    var_4 = 'value1'
    var_5 = 'value2'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'Template variable error'



# Parsed testcases at query #16
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test RepositoryNotFound exception can be instantiated and raised.'
    var_1 = module_0.RepositoryNotFound()
    var_2 = 'Repository not found at specified path'
    var_3 = str(var_1)
    var_4 = 'Test repository not found'
    var_5 = 'Test repository not found'
    var_6 = 'Test repository not found'
    var_7 = 'Repository'
    var_8 = 'not'
    var_9 = 'found'



# Parsed testcases at query #17
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test UnknownRepoType exception can be instantiated and raised.'
    var_1 = module_0.UnknownRepoType()
    var_2 = str(var_1)
    assert var_2 == ''
    var_3 = 'Unable to determine repository type'
    var_4 = str(var_1)
    var_5 = 'Repository type is unknown'
    var_6 = 'Repository type is unknown'
    var_7 = 'Repository type is unknown'



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'Test string representation with empty context.'
    var_1 = 'Variable is undefined'
    var_2 = 'undefined error'
    var_3 = {}

def test_case_0():
    var_0 = 'Test string representation with special characters in message.'
    var_1 = "Variable '{{ undefined_var }}' is not defined"
    var_2 = "UndefinedError: '{{ undefined_var }}'"
    var_3 = 'key'
    var_4 = 'value with \'quotes\' and "double quotes"'
    var_5 = {var_3: var_4}



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'Test the __str__ method of UndefinedVariableInTemplate.'
    var_1 = "variable 'foo' is undefined"
    var_2 = 'key1'
    var_3 = 'key2'
    var_4 = 'value1'
    var_5 = 'value2'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'Template variable error'

def test_case_0():
    var_0 = 'Test __str__ method with empty context.'
    var_1 = 'undefined variable'
    var_2 = {}
    var_3 = 'Error'

def test_case_0():
    var_0 = 'Test __str__ method with special characters in message and context.'
    var_1 = 'error with \'quotes\' and "double quotes"'
    var_2 = 'special'
    var_3 = "value with 'special' chars"
    var_4 = {var_2: var_3}
    var_5 = 'Message with special chars: !@#$%'



# Parsed testcases at query #20
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test EmptyDirNameException can be instantiated and raised.'
    var_1 = module_0.EmptyDirNameException()
    var_2 = 'Directory name cannot be empty'
    var_3 = str(var_1)
    var_4 = 'Test error message'
    var_5 = 'Test error message'
    var_6 = 'Test error message'



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'Test FailedHookException can be instantiated and raised.'
    var_1 = 'Test hook failed'
    var_2 = 'Hook script execution failed'
    var_3 = 'Hook failed during execution'
    var_4 = 'Test'
    var_5 = 'Test'



# Parsed testcases at query #22
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test RepositoryCloneFailed exception can be instantiated and raised.'
    var_1 = 'Test message'
    var_2 = module_0.RepositoryCloneFailed()
    var_3 = 'Clone failed'
    var_4 = 'arg1'
    var_5 = 'arg2'



# Parsed testcases at query #23
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test VCSNotInstalled exception can be instantiated and raised.'
    var_1 = module_0.VCSNotInstalled()
    var_2 = 'Git is not installed'
    var_3 = str(var_1)
    var_4 = 'Mercurial is not installed'
    var_5 = 'Version control system not found'
    var_6 = module_0.VCSNotInstalled()



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'Test UnknownTemplateDirException constructor and inheritance.'
    var_1 = 'Test message'

import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test UnknownTemplateDirException with no message.'
    var_1 = module_0.UnknownTemplateDirException()
    var_2 = str(var_1)
    assert var_2 == ''

def test_case_0():
    var_0 = 'Test UnknownTemplateDirException with multiple arguments.'
    var_1 = 'Error'
    var_2 = 'Details'

def test_case_0():
    var_0 = 'Test that UnknownTemplateDirException can be raised and caught.'
    var_1 = 'Template directory is ambiguous'

def test_case_0():
    var_0 = 'Test that UnknownTemplateDirException can be caught as CookiecutterException.'
    var_1 = 'Test error'



# Parsed testcases at query #25
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test UnknownTemplateDirException can be instantiated and raised.'
    var_1 = module_0.UnknownTemplateDirException()
    var_2 = str(var_1)
    assert var_2 == ''
    var_3 = 'Unable to determine which directory is the project template'
    var_4 = str(var_1)
    var_5 = 'Test error'
    var_6 = 'Test error'
    var_7 = 'Test error'



# Parsed testcases at query #26
#--------------------------




# Parsed testcases at query #27
#--------------------------




# Parsed testcases at query #28
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test OutputDirExistsException can be instantiated and raised.'
    var_1 = module_0.OutputDirExistsException()
    var_2 = 'Output directory already exists'
    var_3 = str(var_1)
    var_4 = 'Test error message'
    var_5 = 'Test error message'
    var_6 = 'Test error message'



# Parsed testcases at query #29
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test RepositoryNotFound exception can be instantiated and raised.'
    var_1 = module_0.RepositoryNotFound()
    var_2 = 'Repository not found at specified location'
    var_3 = 'Test repository not found'
    var_4 = 'Repository does not exist'



# Parsed testcases at query #30
#--------------------------




# Parsed testcases at query #31
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test InvalidModeException can be instantiated and raised.'
    var_1 = module_0.InvalidModeException()
    var_2 = 'Cannot use no_input and replay at the same time'
    var_3 = str(var_1)
    var_4 = 'Test error message'
    var_5 = 'Test error message'
    var_6 = 'Test error message'



# Parsed testcases at query #32
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test MissingProjectDir exception initialization and inheritance.'
    var_1 = module_0.MissingProjectDir()
    var_2 = 'Project directory not found'
    var_3 = 'Test error'
    var_4 = 'Test error'
    var_5 = 'Test error'



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = 'Test FailedHookException can be instantiated and raised.'
    var_1 = 'Hook script failed'
    var_2 = 'Hook failed'
    var_3 = 'extra_arg'
    var_4 = 'Test hook failure'



# Parsed testcases at query #34
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test ConfigDoesNotExistException can be instantiated and raised.'
    var_1 = 'Config file not found'
    var_2 = module_0.ConfigDoesNotExistException()
    var_3 = str(var_2)
    assert var_3 == ''
    var_4 = 'Missing config at /path/to/config'



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = 'Test UndefinedVariableInTemplate exception initialization and string representation.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = "Variable 'foo' is not defined"
    var_7 = str(var_5)
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = 'Another error message'



# Parsed testcases at query #36
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test RepositoryNotFound exception can be instantiated and raised.'
    var_1 = module_0.RepositoryNotFound()
    var_2 = 'Repository not found at the specified path'
    var_3 = str(var_1)
    var_4 = 'Test repository not found'
    var_5 = 'Another test message'
    var_6 = 'Generic exception test'



# Parsed testcases at query #37
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test NonTemplatedInputDirException can be instantiated and raised.'
    var_1 = module_0.NonTemplatedInputDirException()
    var_2 = 'Input directory is not templated'
    var_3 = str(var_1)
    var_4 = 'Test error'
    var_5 = 'Test error'
    var_6 = 'Test error'



# Parsed testcases at query #38
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test CookiecutterException can be instantiated and raised.'
    var_1 = 'Test message'
    var_2 = 'Test error'
    var_3 = module_0.CookiecutterException()



# Parsed testcases at query #39
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test UnknownTemplateDirException can be instantiated and raised.'
    var_1 = module_0.UnknownTemplateDirException()
    var_2 = str(var_1)
    assert var_2 == ''
    var_3 = 'Cannot determine template directory'
    var_4 = str(var_1)
    var_5 = 'Error'
    var_6 = 'Additional info'
    var_7 = 'Test error'
    var_8 = 'Test error'
    var_9 = 'Test error'



# Parsed testcases at query #40
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test OutputDirExistsException can be instantiated and raised.'
    var_1 = module_0.OutputDirExistsException()
    var_2 = 'Output directory already exists'
    var_3 = str(var_1)
    var_4 = 'Test error message'
    var_5 = 'Error'
    var_6 = 'Additional info'



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test RepositoryNotFound exception can be instantiated and raised.'
    var_1 = module_0.RepositoryNotFound()
    var_2 = "The specified cookiecutter repository doesn't exist"
    var_3 = str(var_1)
    var_4 = 'Repository not found at path'
    var_5 = 'Repository not found'
    var_6 = 'Custom error message'



# Parsed testcases at query #2
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test OutputDirExistsException can be instantiated and raised.'
    var_1 = module_0.OutputDirExistsException()
    var_2 = 'Output directory already exists'
    var_3 = str(var_1)
    var_4 = 'Directory exists at /path/to/output'
    var_5 = 'Test output directory exists'



# Parsed testcases at query #3
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test InvalidZipRepository exception can be instantiated and raised.'
    var_1 = module_0.InvalidZipRepository()
    var_2 = 'Invalid zip repository'
    var_3 = 'Test error'
    var_4 = 'Test error'
    var_5 = 'Test error'



# Parsed testcases at query #4
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test UnknownTemplateDirException can be instantiated and raised.'
    var_1 = module_0.UnknownTemplateDirException()
    var_2 = str(var_1)
    assert var_2 == ''
    var_3 = 'Ambiguous template directory'
    var_4 = str(var_1)
    var_5 = 'Test error'
    var_6 = 'Test error'
    var_7 = 'Test error'



# Parsed testcases at query #5
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test UnknownRepoType exception can be instantiated and raised.'
    var_1 = 'Test message'
    var_2 = module_0.UnknownRepoType()
    var_3 = 'Repository type unknown'
    var_4 = 'Another test'
    var_5 = 'Generic exception test'



# Parsed testcases at query #6
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test RepositoryCloneFailed exception.'
    var_1 = module_0.RepositoryCloneFailed()
    var_2 = 'Failed to clone the repository'
    var_3 = str(var_1)
    var_4 = 'Error'
    var_5 = 'Additional info'
    var_6 = 'Clone failed'
    var_7 = 'Clone failed'
    var_8 = 'Clone failed'



# Parsed testcases at query #7
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test ConfigDoesNotExistException can be instantiated and raised.'
    var_1 = 'Config file not found'
    var_2 = module_0.ConfigDoesNotExistException()
    var_3 = 'Test error'
    var_4 = 'Test error'
    var_5 = 'Test error'



# Parsed testcases at query #8
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test InvalidConfiguration exception.'
    var_1 = module_0.InvalidConfiguration()
    var_2 = 'Invalid YAML configuration'
    var_3 = str(var_1)
    var_4 = 'Config error'
    var_5 = 'additional info'
    var_6 = str(var_1)
    var_7 = 'Test error message'
    var_8 = 'Test error'
    var_9 = 'Test error'



# Parsed testcases at query #9
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test UnknownTemplateDirException can be instantiated and raised.'
    var_1 = module_0.UnknownTemplateDirException()
    var_2 = 'Ambiguous template directory'
    var_3 = str(var_1)
    var_4 = 'Test error'
    var_5 = 'Test error'
    var_6 = 'Test error'



# Parsed testcases at query #10
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test OutputDirExistsException can be instantiated and raised.'
    var_1 = 'Output directory already exists'
    var_2 = module_0.OutputDirExistsException()
    var_3 = str(var_2)
    assert var_3 == ''
    var_4 = 'Test error'
    var_5 = 'Error'
    var_6 = 'with'
    var_7 = 'multiple'
    var_8 = 'args'



# Parsed testcases at query #11
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test EmptyDirNameException constructor and inheritance.'
    var_1 = 'Directory name is empty'
    var_2 = module_0.EmptyDirNameException()
    var_3 = str(var_2)
    assert var_3 == ''
    var_4 = 'Error'
    var_5 = 'Additional info'



# Parsed testcases at query #12
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test InvalidConfiguration exception can be instantiated and raised.'
    var_1 = module_0.InvalidConfiguration()
    var_2 = 'Configuration file is not valid YAML'
    var_3 = str(var_1)
    var_4 = 'Invalid config'
    var_5 = module_0.InvalidConfiguration()
    var_6 = 'Test message'
    var_7 = 'Test message'



# Parsed testcases at query #13
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test VCSNotInstalled exception can be instantiated and raised.'
    var_1 = module_0.VCSNotInstalled()
    var_2 = 'Git is not installed'
    var_3 = 'Mercurial is not installed'
    var_4 = 'Version control system not found'



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'Test FailedHookException can be instantiated and raised.'
    var_1 = 'Hook script failed'
    var_2 = 'Test hook failure'



# Parsed testcases at query #15
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test ContextDecodingException can be instantiated and raised.'
    var_1 = module_0.ContextDecodingException()
    var_2 = 'Failed to decode context JSON'
    var_3 = str(var_1)
    var_4 = 'Test error message'
    var_5 = 'Context JSON decode error'



# Parsed testcases at query #16
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test VCSNotInstalled exception can be instantiated and raised.'
    var_1 = module_0.VCSNotInstalled()
    var_2 = 'Git is not installed'
    var_3 = 'Mercurial is not installed'
    var_4 = 'VCS not found'
    var_5 = module_0.VCSNotInstalled()



# Parsed testcases at query #17
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test CookiecutterException can be instantiated and raised.'
    var_1 = module_0.CookiecutterException()
    var_2 = str(var_1)
    assert var_2 == ''
    var_3 = 'Test error message'
    var_4 = str(var_1)
    var_5 = 'Test exception'
    var_6 = 'arg1'
    var_7 = 'arg2'



# Parsed testcases at query #18
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test MissingProjectDir exception can be instantiated and raised.'
    var_1 = module_0.MissingProjectDir()
    var_2 = 'Project directory not found'
    var_3 = str(var_1)
    var_4 = 'Test error message'
    var_5 = 'Test error message'



# Parsed testcases at query #19
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test ConfigDoesNotExistException can be instantiated and raised.'
    var_1 = 'Config file not found'
    var_2 = module_0.ConfigDoesNotExistException()
    var_3 = str(var_2)
    assert var_3 == ''
    var_4 = 'Path does not exist: /invalid/path'
    var_5 = 'Config error'
    var_6 = 'Config error'



# Parsed testcases at query #20
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test NonTemplatedInputDirException can be instantiated and raised.'
    var_1 = module_0.NonTemplatedInputDirException()
    var_2 = 'Input directory is not templated'
    var_3 = str(var_1)
    var_4 = 'Test error'
    var_5 = 'Test error'
    var_6 = 'Test error'



# Parsed testcases at query #21
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test ContextDecodingException can be instantiated and raised.'
    var_1 = module_0.ContextDecodingException()
    var_2 = 'Failed to decode context JSON'
    var_3 = str(var_1)
    var_4 = 'Test error message'
    var_5 = 'Test error message'
    var_6 = 'Test error message'



# Parsed testcases at query #22
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test InvalidConfiguration exception initialization and inheritance.'
    var_1 = module_0.InvalidConfiguration()
    var_2 = 'Invalid YAML configuration'
    var_3 = 'Error'
    var_4 = 'Details'
    var_5 = 'Config is not valid YAML'
    var_6 = 'Badly constructed config'
    var_7 = 'Generic error'



# Parsed testcases at query #23
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test EmptyDirNameException can be instantiated and raised.'
    var_1 = module_0.EmptyDirNameException()
    var_2 = 'Directory name cannot be empty'
    var_3 = str(var_1)
    var_4 = 'Test error message'
    var_5 = 'Test error message'
    var_6 = 'Test error message'



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'Test UndefinedVariableInTemplate exception initialization and string representation.'
    var_1 = "variable 'foo' is undefined"
    var_2 = 'Variable is not defined'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = 'value1'
    var_6 = 'value2'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = str(var_7)
    var_9 = {}
    var_10 = 'Different error occurred'



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'Test InvalidModeException can be instantiated and raised.'
    var_1 = 'Test message'
    var_2 = 'no_input and replay cannot both be True'
    var_3 = ''
    var_4 = 'Test'



# Parsed testcases at query #26
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test InvalidConfiguration exception can be instantiated and raised.'
    var_1 = module_0.InvalidConfiguration()
    var_2 = 'Invalid YAML in config file'
    var_3 = str(var_1)
    var_4 = module_0.InvalidConfiguration()
    var_5 = 'Config file is not valid YAML'



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'Test InvalidModeException can be instantiated and raised.'
    var_1 = 'Test message'
    var_2 = 'Both no_input and replay cannot be True'



# Parsed testcases at query #28
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test NonTemplatedInputDirException can be instantiated and raised.'
    var_1 = module_0.NonTemplatedInputDirException()
    var_2 = 'Input directory is not templated'
    var_3 = str(var_1)
    var_4 = 'Test error message'
    var_5 = 'Test error message'
    var_6 = 'Test error message'



# Parsed testcases at query #29
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test UnknownRepoType exception can be instantiated and raised.'
    var_1 = module_0.UnknownRepoType()
    var_2 = 'Unable to determine repository type'
    var_3 = 'Test error message'
    var_4 = 'Test error message'
    var_5 = 'Test error message'



# Parsed testcases at query #30
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test VCSNotInstalled exception can be instantiated and raised.'
    var_1 = module_0.VCSNotInstalled()
    var_2 = 'Git is not installed'
    var_3 = 'Mercurial not found'
    var_4 = 'Version control system unavailable'
    var_5 = module_0.VCSNotInstalled()



# Parsed testcases at query #31
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test RepositoryNotFound exception can be instantiated and raised.'
    var_1 = 'Test repository not found'
    var_2 = module_0.RepositoryNotFound()
    var_3 = 'Repository does not exist'
    var_4 = 'Repository error'
    var_5 = 'Generic error'
    var_6 = 'arg1'
    var_7 = 'arg2'



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = 'Test __str__ method of UndefinedVariableInTemplate.'
    var_1 = 'project_name'
    var_2 = 'author'
    var_3 = 'my_project'
    var_4 = 'John Doe'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'Variable is undefined'

def test_case_0():
    var_0 = 'Test __str__ method with empty context.'
    var_1 = 'Template variable error'
    var_2 = {}

def test_case_0():
    var_0 = 'Test __str__ method with complex context.'
    var_1 = 'simple'
    var_2 = 'nested'
    var_3 = 'list'
    var_4 = 'value'
    var_5 = 'key'
    var_6 = {var_5: var_4}
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = [var_7, var_8, var_9]
    var_11 = {var_1: var_4, var_2: var_6, var_3: var_10}
    var_12 = 'Complex context error'



# Parsed testcases at query #33
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test RepositoryCloneFailed exception can be instantiated and raised.'
    var_1 = module_0.RepositoryCloneFailed()
    var_2 = 'Failed to clone the repository'
    var_3 = str(var_1)
    var_4 = 'Clone operation failed'
    var_5 = 'Clone operation failed'
    var_6 = 'Clone operation failed'



# Parsed testcases at query #34
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test RepositoryNotFound exception can be instantiated and raised.'
    var_1 = module_0.RepositoryNotFound()
    var_2 = 'The repository could not be found'
    var_3 = 'Repository not found at the specified path'
    var_4 = 'Repository error'
    var_5 = 'Generic exception'



# Parsed testcases at query #35
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test InvalidZipRepository exception can be instantiated and raised.'
    var_1 = module_0.InvalidZipRepository()
    var_2 = 'Invalid zip repository'
    var_3 = str(var_1)
    var_4 = 'test message'
    var_5 = 'test message'
    var_6 = 'test message'



# Parsed testcases at query #36
#--------------------------


def test_case_0():
    var_0 = 'Test UndefinedVariableInTemplate exception initialization and string representation.'
    var_1 = "Variable 'foo' is undefined"
    var_2 = 'Error in template rendering'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = 'value1'
    var_6 = 'value2'
    var_7 = {var_3: var_5, var_4: var_6}

def test_case_0():
    var_0 = 'Test UndefinedVariableInTemplate with empty context.'
    var_1 = 'undefined variable'
    var_2 = 'Template error occurred'
    var_3 = {}

def test_case_0():
    var_0 = 'Test UndefinedVariableInTemplate with complex nested context.'
    var_1 = 'Complex error'
    var_2 = 'Complex template rendering failed'
    var_3 = 'nested'
    var_4 = 'list'
    var_5 = 'string'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = [var_9, var_10, var_11]
    var_13 = 'test'
    var_14 = {var_3: var_8, var_4: var_12, var_5: var_13}



# Parsed testcases at query #37
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test VCSNotInstalled exception can be instantiated and raised.'
    var_1 = module_0.VCSNotInstalled()
    var_2 = 'Git is not installed'
    var_3 = 'Mercurial is not installed'
    var_4 = 'VCS not found'
    var_5 = 'arg1'
    var_6 = 'arg2'



# Parsed testcases at query #38
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test OutputDirExistsException constructor and inheritance.'
    var_1 = module_0.OutputDirExistsException()
    var_2 = 'Output directory already exists'
    var_3 = 'Error'
    var_4 = 'Details'
    var_5 = 'Test error'
    var_6 = 'Test error'
    var_7 = 'Test error'



# Parsed testcases at query #39
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test VCSNotInstalled exception can be instantiated and raised.'
    var_1 = module_0.VCSNotInstalled()
    var_2 = 'Git is not installed'
    var_3 = 'Mercurial not found'
    var_4 = 'Version control system unavailable'
    var_5 = module_0.VCSNotInstalled()



# Parsed testcases at query #40
#--------------------------


import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = 'Test VCSNotInstalled exception can be instantiated and raised.'
    var_1 = module_0.VCSNotInstalled()
    var_2 = 'Git is not installed'
    var_3 = str(var_1)
    var_4 = 'Mercurial is not installed'
    var_5 = 'VCS not found'
    var_6 = 'Version control system unavailable'



