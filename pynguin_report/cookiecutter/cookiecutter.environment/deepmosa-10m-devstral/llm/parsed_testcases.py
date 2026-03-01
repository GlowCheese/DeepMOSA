####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test the constructor with no context provided.'
    var_1 = {}
    var_2 = module_0.ExtensionLoaderMixin(**var_1)
    var_3 = var_2.extensions
    var_4 = bool(var_2.extensions == ['cookiecutter.extensions.JsonifyExtension', 'cookiecutter.extensions.RandomStringExtension', 'cookiecutter.extensions.SlugifyExtension', 'cookiecutter.extensions.TimeExtension', 'cookiecutter.extensions.UUIDExtension'])
    assert var_4 is True

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test the constructor with an empty context.'
    var_1 = {}
    var_2 = {}
    var_3 = module_0.ExtensionLoaderMixin(context=var_1, **var_2)
    var_4 = var_3.extensions
    var_5 = bool(var_3.extensions == ['cookiecutter.extensions.JsonifyExtension', 'cookiecutter.extensions.RandomStringExtension', 'cookiecutter.extensions.SlugifyExtension', 'cookiecutter.extensions.TimeExtension', 'cookiecutter.extensions.UUIDExtension'])
    assert var_5 is True

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test the constructor with extensions in the context.'
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 'custom.extension.One'
    var_4 = 'custom.extension.Two'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {}
    var_9 = module_0.ExtensionLoaderMixin(context=var_7, **var_8)
    var_10 = var_9.extensions
    var_11 = bool(var_9.extensions == ['cookiecutter.extensions.JsonifyExtension', 'cookiecutter.extensions.RandomStringExtension', 'cookiecutter.extensions.SlugifyExtension', 'cookiecutter.extensions.TimeExtension', 'cookiecutter.extensions.UUIDExtension', 'custom.extension.One', 'custom.extension.Two'])
    assert var_11 is True

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test the constructor with an invalid extension in the context.'
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 'invalid.extension'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_0.ExtensionLoaderMixin(context=var_6, **var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_unknown_extension_raised_on_import_error. Retrieved 5/8 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'extensions'
    var_2 = 'nonexistent.extension'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = []
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_import_error_raises_unknown_extension. Retrieved 7/9 statements.


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'extensions'
    var_2 = 'nonexistent.extension'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = 'extensions'
    var_6 = {var_5: var_3}
    var_7 = module_0.ExtensionLoaderMixin(context=var_0, **var_6)
    var_8 = str(var_0)
    assert var_8 == "Unable to load extension: No module named 'nonexistent'"



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_extension_loader_mixin_init_with_no_context.
# Partially parsed test_extension_loader_mixin_init_with_empty_context. Retrieved 1/5 statements.
# Partially parsed test_extension_loader_mixin_init_with_extensions_in_context. Retrieved 7/11 statements.
# Partially parsed test_extension_loader_mixin_init_with_invalid_extension. Retrieved 6/11 statements.


def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'custom.extension1'
    var_3 = 'custom.extension2'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'invalid.extension'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #5
#--------------------------




import cookiecutter.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.ExtensionLoaderMixin(**var_0)
    var_2 = var_1.extensions
    var_3 = bool(var_1.extensions == ['cookiecutter.extensions.JsonifyExtension', 'cookiecutter.extensions.RandomStringExtension', 'cookiecutter.extensions.SlugifyExtension', 'cookiecutter.extensions.TimeExtension', 'cookiecutter.extensions.UUIDExtension'])
    assert var_3 is True

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.ExtensionLoaderMixin(context=var_0, **var_1)
    var_3 = var_2.extensions
    var_4 = bool(var_2.extensions == ['cookiecutter.extensions.JsonifyExtension', 'cookiecutter.extensions.RandomStringExtension', 'cookiecutter.extensions.SlugifyExtension', 'cookiecutter.extensions.TimeExtension', 'cookiecutter.extensions.UUIDExtension'])
    assert var_4 is True

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'custom.extension.One'
    var_3 = 'custom.extension.Two'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = {}
    var_8 = module_0.ExtensionLoaderMixin(context=var_6, **var_7)
    var_9 = var_8.extensions
    var_10 = bool(var_8.extensions == ['cookiecutter.extensions.JsonifyExtension', 'cookiecutter.extensions.RandomStringExtension', 'cookiecutter.extensions.SlugifyExtension', 'cookiecutter.extensions.TimeExtension', 'cookiecutter.extensions.UUIDExtension', 'custom.extension.One', 'custom.extension.Two'])
    assert var_10 is True

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'invalid.extension'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_0.ExtensionLoaderMixin(context=var_5, **var_6)
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_import_error_raises_unknown_extension. Retrieved 3/4 statements.


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = {}
    var_3 = module_0.ExtensionLoaderMixin(context=var_0, **var_2)



# Parsed testcases at query #7
#--------------------------




import cookiecutter.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.ExtensionLoaderMixin(**var_0)
    var_2 = var_1.extensions
    var_3 = bool(var_1.extensions == ['cookiecutter.extensions.JsonifyExtension', 'cookiecutter.extensions.RandomStringExtension', 'cookiecutter.extensions.SlugifyExtension', 'cookiecutter.extensions.TimeExtension', 'cookiecutter.extensions.UUIDExtension'])
    assert var_3 is True

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.ExtensionLoaderMixin(context=var_0, **var_1)
    var_3 = var_2.extensions
    var_4 = bool(var_2.extensions == ['cookiecutter.extensions.JsonifyExtension', 'cookiecutter.extensions.RandomStringExtension', 'cookiecutter.extensions.SlugifyExtension', 'cookiecutter.extensions.TimeExtension', 'cookiecutter.extensions.UUIDExtension'])
    assert var_4 is True

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'custom.extension1'
    var_3 = 'custom.extension2'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = {}
    var_8 = module_0.ExtensionLoaderMixin(context=var_6, **var_7)
    var_9 = var_8.extensions
    var_10 = bool(var_8.extensions == ['cookiecutter.extensions.JsonifyExtension', 'cookiecutter.extensions.RandomStringExtension', 'cookiecutter.extensions.SlugifyExtension', 'cookiecutter.extensions.TimeExtension', 'cookiecutter.extensions.UUIDExtension', 'custom.extension1', 'custom.extension2'])
    assert var_10 is True

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'invalid.extension'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_0.ExtensionLoaderMixin(context=var_5, **var_6)
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_import_error_raises_unknown_extension. Retrieved 7/9 statements.


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



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import cookiecutter.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.ExtensionLoaderMixin(**var_0)
    var_2 = var_1.extensions
    var_3 = bool(var_1.extensions == ['cookiecutter.extensions.JsonifyExtension', 'cookiecutter.extensions.RandomStringExtension', 'cookiecutter.extensions.SlugifyExtension', 'cookiecutter.extensions.TimeExtension', 'cookiecutter.extensions.UUIDExtension'])
    assert var_3 is True

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.ExtensionLoaderMixin(context=var_0, **var_1)
    var_3 = var_2.extensions
    var_4 = bool(var_2.extensions == ['cookiecutter.extensions.JsonifyExtension', 'cookiecutter.extensions.RandomStringExtension', 'cookiecutter.extensions.SlugifyExtension', 'cookiecutter.extensions.TimeExtension', 'cookiecutter.extensions.UUIDExtension'])
    assert var_4 is True

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'custom.extension1'
    var_3 = 'custom.extension2'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = {}
    var_8 = module_0.ExtensionLoaderMixin(context=var_6, **var_7)
    var_9 = var_8.extensions
    var_10 = bool(var_8.extensions == ['cookiecutter.extensions.JsonifyExtension', 'cookiecutter.extensions.RandomStringExtension', 'cookiecutter.extensions.SlugifyExtension', 'cookiecutter.extensions.TimeExtension', 'cookiecutter.extensions.UUIDExtension', 'custom.extension1', 'custom.extension2'])
    assert var_10 is True

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'invalid.extension'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_0.ExtensionLoaderMixin(context=var_5, **var_6)
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_ImportError_raises_UnknownExtension. Retrieved 7/9 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'invalid.extension'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = {}



# Parsed testcases at query #3
#--------------------------




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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_import_error_raises_unknown_extension. Retrieved 7/10 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'invalid.extension'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = []
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #5
#--------------------------




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



# Parsed testcases at query #6
#--------------------------




import cookiecutter.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'extensions'
    var_2 = 'nonexistent_extension'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = 'extensions'
    var_6 = {var_5: var_3}
    var_7 = module_0.ExtensionLoaderMixin(context=var_0, **var_6)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_import_error_raises_unknown_extension. Retrieved 5/9 statements.


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = 'test error'
    var_3 = ImportError(var_2)
    var_4 = {}
    var_5 = module_0.ExtensionLoaderMixin(context=var_0, **var_4)



# Parsed testcases at query #8
#--------------------------




import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that the constructor initializes with default extensions when no context is provided.'
    var_1 = {}
    var_2 = module_0.ExtensionLoaderMixin(**var_1)
    var_3 = var_2.extensions
    var_4 = bool(var_2.extensions == ['cookiecutter.extensions.JsonifyExtension', 'cookiecutter.extensions.RandomStringExtension', 'cookiecutter.extensions.SlugifyExtension', 'cookiecutter.extensions.TimeExtension', 'cookiecutter.extensions.UUIDExtension'])
    assert var_4 is True

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that the constructor initializes with default extensions when context is empty.'
    var_1 = {}
    var_2 = {}
    var_3 = module_0.ExtensionLoaderMixin(context=var_1, **var_2)
    var_4 = var_3.extensions
    var_5 = bool(var_3.extensions == ['cookiecutter.extensions.JsonifyExtension', 'cookiecutter.extensions.RandomStringExtension', 'cookiecutter.extensions.SlugifyExtension', 'cookiecutter.extensions.TimeExtension', 'cookiecutter.extensions.UUIDExtension'])
    assert var_5 is True

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that the constructor initializes with default extensions plus additional extensions from context.'
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 'custom.extension1'
    var_4 = 'custom.extension2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {}
    var_9 = module_0.ExtensionLoaderMixin(context=var_7, **var_8)
    var_10 = var_9.extensions
    var_11 = bool(var_9.extensions == ['cookiecutter.extensions.JsonifyExtension', 'cookiecutter.extensions.RandomStringExtension', 'cookiecutter.extensions.SlugifyExtension', 'cookiecutter.extensions.TimeExtension', 'cookiecutter.extensions.UUIDExtension', 'custom.extension1', 'custom.extension2'])
    assert var_11 is True

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that the constructor raises UnknownExtension when an invalid extension is provided.'
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 'invalid.extension'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_0.ExtensionLoaderMixin(context=var_6, **var_7)
    var_9 = bool(False)
    assert var_9 is True



