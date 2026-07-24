####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_extension_loader_mixin_read_extensions_missing_key. Retrieved 3/4 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_empty_cookiecutter. Retrieved 9/10 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_valid_list. Retrieved 9/10 statements.


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.ExtensionLoaderMixin(context=var_0, **var_1)
    var_3 = var_2.extensions
    var_4 = len(var_3)
    assert var_4 == 5
    var_5 = 'cookiecutter.extensions.JsonifyExtension'
    var_6 = bool('cookiecutter.extensions.JsonifyExtension' in var_2.extensions)
    assert var_6 is True

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'my.custom.Extension'
    var_3 = 123
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = {}
    var_8 = module_0.ExtensionLoaderMixin(context=var_6, **var_7)
    var_9 = 'my.custom.Extension'
    var_10 = bool('my.custom.Extension' in var_8.extensions)
    assert var_10 is True
    var_11 = '123'
    var_12 = bool('123' in var_8.extensions)
    assert var_12 is True
    var_13 = var_8.extensions
    var_14 = len(var_13)
    assert var_14 == 6

def test_case_0():
    pass

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.ExtensionLoaderMixin(context=var_0, **var_1)
    var_3 = {}

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_0.ExtensionLoaderMixin(context=var_2, **var_3)
    var_5 = 'other'
    var_6 = 'data'
    var_7 = {var_5: var_6}
    var_8 = {}
    var_9 = module_0.ExtensionLoaderMixin(context=var_7, **var_8)
    var_10 = {var_5: var_6}

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.ExtensionLoaderMixin(context=var_0, **var_1)
    var_3 = 'cookiecutter'
    var_4 = '_extensions'
    var_5 = 'ext1'
    var_6 = 'ext2'
    var_7 = [var_5, var_6]
    var_8 = {var_4: var_7}
    var_9 = {var_3: var_8}



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_extension_loader_mixin_init_with_no_context. Retrieved 6/12 statements.
# Partially parsed test_extension_loader_mixin_init_with_custom_extensions. Retrieved 13/19 statements.
# Partially parsed test_extension_loader_mixin_init_with_empty_context. Retrieved 7/13 statements.
# Partially parsed test_extension_loader_mixin_init_raises_unknown_extension_on_import_error. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'cookiecutter.extensions.JsonifyExtension'
    var_1 = 'cookiecutter.extensions.RandomStringExtension'
    var_2 = 'cookiecutter.extensions.SlugifyExtension'
    var_3 = 'cookiecutter.extensions.TimeExtension'
    var_4 = 'cookiecutter.extensions.UUIDExtension'
    var_5 = [var_0, var_1, var_2, var_3, var_4]

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'my.custom.Extension'
    var_3 = 123
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'cookiecutter.extensions.JsonifyExtension'
    var_8 = 'cookiecutter.extensions.RandomStringExtension'
    var_9 = 'cookiecutter.extensions.SlugifyExtension'
    var_10 = 'cookiecutter.extensions.TimeExtension'
    var_11 = 'cookiecutter.extensions.UUIDExtension'
    var_12 = [var_7, var_8, var_9, var_10, var_11]

def test_case_0():
    var_0 = {}
    var_1 = 'cookiecutter.extensions.JsonifyExtension'
    var_2 = 'cookiecutter.extensions.RandomStringExtension'
    var_3 = 'cookiecutter.extensions.SlugifyExtension'
    var_4 = 'cookiecutter.extensions.TimeExtension'
    var_5 = 'cookiecutter.extensions.UUIDExtension'
    var_6 = [var_1, var_2, var_3, var_4, var_5]

def test_case_0():
    var_0 = 'Should have raised UnknownExtension'
    var_1 = AssertionError(var_0)
    var_2 = 'Unable to load extension'



# Parsed testcases at query #3
#--------------------------

# Failed to parse test_extension_loader_mixin_init_with_no_context.
# Partially parsed test_extension_loader_mixin_init_with_custom_extensions. Retrieved 7/14 statements.
# Partially parsed test_extension_loader_mixin_init_with_kwargs_passed_to_super. Retrieved 1/8 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_returns_empty_list_on_missing_key. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'custom.ext'
    var_3 = 123
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'custom.ext'
    var_8 = '123'
    var_9 = 'cookiecutter.extensions.JsonifyExtension'

def test_case_0():
    var_0 = 'value'

def test_case_0():
    var_0 = {}
    var_1 = {}



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_init_raises_unknown_extension_on_import_error. Retrieved 1/9 statements.


def test_case_0():
    pass

def test_case_0():
    var_0 = {}



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_extension_loader_mixin_init_no_context.
# Partially parsed test_extension_loader_mixin_init_with_context_extensions. Retrieved 7/13 statements.
# Failed to parse test_extension_loader_mixin_init_raises_unknown_extension.
# Partially parsed test_read_extensions_returns_empty_list_on_missing_key. Retrieved 3/9 statements.
# Partially parsed test_read_extensions_with_valid_context. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'my_custom_ext'
    var_3 = 123
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'my_custom_ext'
    var_8 = '123'
    var_9 = 'cookiecutter.extensions.JsonifyExtension'

def test_case_0():
    var_0 = 'other'
    var_1 = 'data'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'ext1'
    var_3 = 'ext2'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'ext1'
    var_8 = 'ext2'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_init_loads_extensions_successfully. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'custom.Extension'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = 'value'
    var_7 = 'custom.Extension'
    var_8 = 'cookiecutter.extensions.TimeExtension'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_init_raises_unknown_extension_on_import_error. Retrieved 2/12 statements.


def test_case_0():
    var_0 = {}
    var_1 = str(var_0)
    var_2 = 'Unable to load extension: Mock Error'
    var_3 = bool('Unable to load extension: Mock Error' in var_1)
    assert var_3 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_init_handles_import_error. Retrieved 1/18 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_init_handles_import_error. Retrieved 6/20 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = []
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = str(var_0)
    var_6 = 'Unable to load extension'
    var_7 = bool('Unable to load extension' in var_5)
    assert var_7 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_init_handles_import_error. Retrieved 2/12 statements.


def test_case_0():
    var_0 = {}
    var_1 = str(var_0)
    var_2 = 'Unable to load extension: Mock error'
    var_3 = bool('Unable to load extension: Mock error' in var_1)
    assert var_3 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_init_signature_validates_types. Retrieved 9/22 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'ext1'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = 'value'
    var_7 = None
    var_8 = {}



# Parsed testcases at query #8
#--------------------------

# Failed to parse test_init_handles_import_error_by_raising_unknown_extension.




# Parsed testcases at query #5
#--------------------------

# Failed to parse test_extension_loader_mixin_init_with_no_context.
# Partially parsed test_extension_loader_mixin_init_with_custom_extensions. Retrieved 7/16 statements.
# Failed to parse test_extension_loader_mixin_init_raises_unknown_extension_on_import_error.
# Partially parsed test_extension_loader_mixin_read_extensions_empty_context. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'my.custom.Extension'
    var_3 = 123
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'my.custom.Extension'
    var_8 = '123'

def test_case_0():
    var_0 = {}
    var_1 = {}



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_init_triggers_import_error_exception_branch. Retrieved 1/9 statements.


def test_case_0():
    pass

def test_case_0():
    var_0 = {}



# Parsed testcases at query #7
#--------------------------

# Failed to parse test_extension_loader_mixin_init_with_no_context.
# Partially parsed test_extension_loader_mixin_init_with_custom_extensions. Retrieved 7/16 statements.
# Failed to parse test_extension_loader_mixin_init_raises_unknown_extension_on_import_error.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'custom.ext'
    var_3 = 123
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'custom.ext'
    var_8 = '123'



# Parsed testcases at query #8
#--------------------------

# Failed to parse test_init_signature_type_hints.




# Parsed testcases at query #9
#--------------------------

# Partially parsed test_init_context_is_not_none. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = []
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = bool(var_4 is not None)
    assert var_5 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_init_context_not_none. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = []
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_init_context_not_none. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = []
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_extension_loader_mixin_init_with_no_context. Retrieved 6/13 statements.
# Partially parsed test_extension_loader_mixin_init_with_custom_extensions. Retrieved 13/20 statements.
# Failed to parse test_extension_loader_mixin_read_extensions_returns_empty_list_on_missing_key.
# Partially parsed test_extension_loader_mixin_read_extensions_converts_to_string. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 'cookiecutter.extensions.JsonifyExtension'
    var_1 = 'cookiecutter.extensions.RandomStringExtension'
    var_2 = 'cookiecutter.extensions.SlugifyExtension'
    var_3 = 'cookiecutter.extensions.TimeExtension'
    var_4 = 'cookiecutter.extensions.UUIDExtension'
    var_5 = [var_0, var_1, var_2, var_3, var_4]

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'custom.ext.One'
    var_3 = 'custom.ext.Two'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'cookiecutter.extensions.JsonifyExtension'
    var_8 = 'cookiecutter.extensions.RandomStringExtension'
    var_9 = 'cookiecutter.extensions.SlugifyExtension'
    var_10 = 'cookiecutter.extensions.TimeExtension'
    var_11 = 'cookiecutter.extensions.UUIDExtension'
    var_12 = [var_7, var_8, var_9, var_10, var_11, var_2, var_3]

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 123
    var_3 = True
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_init_handles_import_error. Retrieved 2/12 statements.


def test_case_0():
    var_0 = {}
    var_1 = str(var_0)
    var_2 = 'Unable to load extension: Mock error'
    var_3 = bool('Unable to load extension: Mock error' in var_1)
    assert var_3 is True



# Parsed testcases at query #14
#--------------------------

# Failed to parse test_init_raises_unknown_extension_on_import_error.


def test_case_0():
    pass

def test_case_0():
    pass



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_init_handles_import_error_by_raising_unknown_extension. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = []
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = str(var_0)
    var_6 = 'Unable to load extension: Mock error'
    var_7 = bool('Unable to load extension: Mock error' in var_5)
    assert var_7 is True



# Parsed testcases at query #16
#--------------------------

# Failed to parse test_extension_loader_mixin_init_with_no_context.
# Partially parsed test_extension_loader_mixin_init_with_custom_context. Retrieved 7/16 statements.
# Partially parsed test_extension_loader_mixin_init_with_invalid_extension_raises_error. Retrieved 6/17 statements.
# Partially parsed test_read_extensions_method_logic. Retrieved 11/18 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'my_custom_ext'
    var_3 = 123
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'my_custom_ext'
    var_8 = '123'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'non_existent'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = 'Unable to load extension'

def test_case_0():
    var_0 = {}
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = 'other'
    var_9 = {}
    var_10 = {var_8: var_9}



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_extension_loader_mixin_init_with_no_context. Retrieved 6/13 statements.
# Partially parsed test_extension_loader_mixin_init_with_custom_extensions. Retrieved 13/20 statements.
# Partially parsed test_extension_loader_mixin_init_with_kwargs_passed_to_super. Retrieved 1/8 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_empty_context. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'cookiecutter.extensions.JsonifyExtension'
    var_1 = 'cookiecutter.extensions.RandomStringExtension'
    var_2 = 'cookiecutter.extensions.SlugifyExtension'
    var_3 = 'cookiecutter.extensions.TimeExtension'
    var_4 = 'cookiecutter.extensions.UUIDExtension'
    var_5 = [var_0, var_1, var_2, var_3, var_4]

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'my.custom.Extension'
    var_3 = 123
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'cookiecutter.extensions.JsonifyExtension'
    var_8 = 'cookiecutter.extensions.RandomStringExtension'
    var_9 = 'cookiecutter.extensions.SlugifyExtension'
    var_10 = 'cookiecutter.extensions.TimeExtension'
    var_11 = 'cookiecutter.extensions.UUIDExtension'
    var_12 = [var_7, var_8, var_9, var_10, var_11]

def test_case_0():
    var_0 = 'value'

def test_case_0():
    var_0 = {}



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_extension_loader_mixin_init_with_no_context. Retrieved 6/13 statements.
# Partially parsed test_extension_loader_mixin_init_with_context_extensions. Retrieved 13/20 statements.
# Partially parsed test_extension_loader_mixin_init_passes_kwargs. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'cookiecutter.extensions.JsonifyExtension'
    var_1 = 'cookiecutter.extensions.RandomStringExtension'
    var_2 = 'cookiecutter.extensions.SlugifyExtension'
    var_3 = 'cookiecutter.extensions.TimeExtension'
    var_4 = 'cookiecutter.extensions.UUIDExtension'
    var_5 = [var_0, var_1, var_2, var_3, var_4]

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'custom.ext'
    var_3 = 123
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'cookiecutter.extensions.JsonifyExtension'
    var_8 = 'cookiecutter.extensions.RandomStringExtension'
    var_9 = 'cookiecutter.extensions.SlugifyExtension'
    var_10 = 'cookiecutter.extensions.TimeExtension'
    var_11 = 'cookiecutter.extensions.UUIDExtension'
    var_12 = [var_7, var_8, var_9, var_10, var_11]

def test_case_0():
    var_0 = 'value'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_init_context_is_not_none. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'some'
    var_1 = 'data'
    var_2 = {var_0: var_1}
    var_3 = bool(var_2 is not None)
    assert var_3 is True



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_init_handles_import_error.




# Parsed testcases at query #5
#--------------------------

# Failed to parse test_init_raises_unknown_extension_on_import_error.




# Parsed testcases at query #6
#--------------------------

# Partially parsed test_init_handles_import_error_and_raises_unknown_extension. Retrieved 2/12 statements.


def test_case_0():
    var_0 = {}
    var_1 = str(var_0)
    var_2 = 'Unable to load extension: Module not found'
    var_3 = bool('Unable to load extension: Module not found' in var_1)
    assert var_3 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_extension_loader_mixin_init_with_no_context. Retrieved 6/12 statements.
# Partially parsed test_extension_loader_mixin_init_with_custom_extensions. Retrieved 7/15 statements.
# Partially parsed test_extension_loader_mixin_init_with_none_context. Retrieved 1/9 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_logic. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 'cookiecutter.extensions.JsonifyExtension'
    var_1 = 'cookiecutter.extensions.RandomStringExtension'
    var_2 = 'cookiecutter.extensions.SlugifyExtension'
    var_3 = 'cookiecutter.extensions.TimeExtension'
    var_4 = 'cookiecutter.extensions.UUIDExtension'
    var_5 = [var_0, var_1, var_2, var_3, var_4]

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'custom.Ext1'
    var_3 = 'custom.Ext2'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'custom.Ext1'
    var_8 = 'custom.Ext2'

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 123
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = {}



# Parsed testcases at query #8
#--------------------------

# Failed to parse test_extension_loader_mixin_init_with_no_context.
# Partially parsed test_extension_loader_mixin_init_with_custom_extensions. Retrieved 7/14 statements.
# Failed to parse test_extension_loader_mixin_init_raises_unknown_extension_on_import_error.
# Partially parsed test_extension_loader_mixin_read_extensions_empty_context. Retrieved 2/12 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_valid_context. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'custom.Ext'
    var_3 = 123
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'custom.Ext'
    var_8 = '123'
    var_9 = 'cookiecutter.extensions.JsonifyExtension'

def test_case_0():
    var_0 = {}
    var_1 = {}

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'ext1'
    var_3 = 'ext2'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_init_raises_unknown_extension_on_import_error. Retrieved 5/20 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = []
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'Unable to load extension: Failed to load'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_init_handles_import_error. Retrieved 1/30 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #11
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_init_raises_unknown_extension_on_import_error. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = []
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = str(var_0)
    var_6 = 'Unable to load extension: Mock error'
    var_7 = bool('Unable to load extension: Mock error' in var_5)
    assert var_7 is True



