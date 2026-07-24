####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_init_with_no_context_and_no_extensions. Retrieved 1/5 statements.
# Partially parsed test_init_with_empty_context. Retrieved 2/6 statements.
# Partially parsed test_init_with_context_containing_extensions. Retrieved 8/12 statements.
# Partially parsed test_init_with_invalid_extension_raises_unknown_extension. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 'Test that the constructor initializes with default extensions when no context is provided.'

def test_case_0():
    var_0 = 'Test that the constructor initializes with default extensions when context is empty.'
    var_1 = {}

def test_case_0():
    var_0 = 'Test that the constructor initializes with default and additional extensions from context.'
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 'custom.extension.One'
    var_4 = 'custom.extension.Two'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}

def test_case_0():
    var_0 = 'Test that the constructor raises UnknownExtension when an invalid extension is provided.'
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 'invalid.extension'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}



# Parsed testcases at query #2
#--------------------------




import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test ExtensionLoaderMixin initialization with no context.'
    var_1 = module_0.ExtensionLoaderMixin()

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test ExtensionLoaderMixin initialization with empty context.'
    var_1 = {}
    var_2 = module_0.ExtensionLoaderMixin(context=var_1)

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test ExtensionLoaderMixin initialization with extensions in context.'
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 'custom.extension.One'
    var_4 = 'custom.extension.Two'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = module_0.ExtensionLoaderMixin(context=var_7)

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test ExtensionLoaderMixin initialization with invalid extension.'
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 'invalid.extension'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.ExtensionLoaderMixin(context=var_6)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_import_error_raises_unknown_extension. Retrieved 5/8 statements.


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.ExtensionLoaderMixin(context=var_0, **var_1)
    var_3 = 'nonexistent.extension'
    var_4 = [var_3]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_import_error_raises_unknown_extension. Retrieved 2/9 statements.


def test_case_0():
    var_0 = {}
    var_1 = {}



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_import_error_raises_unknown_extension. Retrieved 3/5 statements.


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.ExtensionLoaderMixin(context=var_0, **var_1)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_import_error_raises_unknown_extension. Retrieved 6/8 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'nonexistent.extension'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}



# Parsed testcases at query #7
#--------------------------




import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that ExtensionLoaderMixin initializes with default extensions when no context is provided.'
    var_1 = module_0.ExtensionLoaderMixin()

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that ExtensionLoaderMixin initializes with default extensions when context is empty.'
    var_1 = {}
    var_2 = module_0.ExtensionLoaderMixin(context=var_1)

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that ExtensionLoaderMixin initializes with default extensions when context has no _extensions key.'
    var_1 = 'cookiecutter'
    var_2 = 'some_key'
    var_3 = 'some_value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_0.ExtensionLoaderMixin(context=var_5)

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that ExtensionLoaderMixin initializes with default and additional extensions from context.'
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 'custom.extension1'
    var_4 = 'custom.extension2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = module_0.ExtensionLoaderMixin(context=var_7)

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that ExtensionLoaderMixin raises UnknownExtension when an extension cannot be loaded.'
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 'nonexistent.extension'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.ExtensionLoaderMixin(context=var_6)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_unknown_extension_raised_on_import_error. Retrieved 4/10 statements.


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = 'Test error'
    var_3 = module_0.ExtensionLoaderMixin(context=var_0, **var_1)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that the constructor initializes with default extensions when no context is provided.'
    var_1 = module_0.ExtensionLoaderMixin()
    var_2 = 'extensions'
    var_3 = hasattr(var_1, var_2)

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that the constructor initializes with default extensions when context is empty.'
    var_1 = {}
    var_2 = module_0.ExtensionLoaderMixin(context=var_1)
    var_3 = 'extensions'
    var_4 = hasattr(var_2, var_3)

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that the constructor initializes with default and custom extensions when provided in context.'
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 'custom.extension.One'
    var_4 = 'custom.extension.Two'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = module_0.ExtensionLoaderMixin(context=var_7)
    var_9 = 'extensions'
    var_10 = hasattr(var_8, var_9)

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that the constructor raises UnknownExtension when an invalid extension is provided.'
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 'invalid.extension'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.ExtensionLoaderMixin(context=var_6)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_unknown_extension_raised_on_import_error. Retrieved 8/9 statements.


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'nonexistent.extension'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_0.ExtensionLoaderMixin(context=var_5, **var_6)



# Parsed testcases at query #3
#--------------------------




import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that the constructor works with no context provided.'
    var_1 = None
    var_2 = module_0.ExtensionLoaderMixin(context=var_1)

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that the constructor works with an empty context.'
    var_1 = {}
    var_2 = module_0.ExtensionLoaderMixin(context=var_1)

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that the constructor loads extensions from context.'
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 'custom.extension.One'
    var_4 = 'custom.extension.Two'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = module_0.ExtensionLoaderMixin(context=var_7)

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that the constructor raises UnknownExtension for invalid extensions.'
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 'invalid.extension'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.ExtensionLoaderMixin(context=var_6)



