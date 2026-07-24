####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_extension_loader_mixin_init_with_default_extensions. Retrieved 4/21 statements.
# Partially parsed test_extension_loader_mixin_init_with_context. Retrieved 8/12 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_with_valid_context. Retrieved 8/10 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_with_missing_extensions_key. Retrieved 3/5 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_with_missing_cookiecutter_key. Retrieved 1/3 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_with_none_context. Retrieved 3/5 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_converts_to_strings. Retrieved 8/10 statements.


def test_case_0():
    var_0 = 0
    var_1 = None
    var_2 = []
    var_3 = {}

def test_case_0():
    var_0 = 'custom.extension'
    var_1 = [var_0]
    var_2 = 'cookiecutter'
    var_3 = '_extensions'
    var_4 = 'custom.extension'
    var_5 = [var_4]
    var_6 = {var_3: var_5}
    var_7 = {var_2: var_6}

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'ext1'
    var_3 = 'ext2'
    var_4 = 'ext3'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = var_0 or var_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 123
    var_3 = 'string_ext'
    var_4 = 45.6
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_extension_loader_mixin_import_error_handling. Retrieved 4/16 statements.


def test_case_0():
    var_0 = 'Test that ImportError is caught and re-raised as UnknownExtension at line 23.'
    var_1 = 'Module not found'
    var_2 = ImportError(var_1)
    var_3 = {}



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_extension_loader_mixin_init_with_default_extensions. Retrieved 1/6 statements.
# Failed to parse test_extension_loader_mixin_init_with_no_context.
# Partially parsed test_extension_loader_mixin_read_extensions_with_valid_context. Retrieved 7/12 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_with_empty_context. Retrieved 1/6 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_with_no_extensions_key. Retrieved 3/8 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_with_no_cookiecutter_key. Retrieved 3/8 statements.


def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'my.custom.Extension1'
    var_3 = 'my.custom.Extension2'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'other_key'
    var_1 = 'value'
    var_2 = {var_0: var_1}



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_extension_loader_mixin_import_error_handling. Retrieved 3/15 statements.


def test_case_0():
    var_0 = 'Test that ImportError is caught and re-raised as UnknownExtension at line 23.'
    var_1 = 'Module not found'
    var_2 = {}



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_extension_loader_mixin_handles_import_error. Retrieved 3/18 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_extension_loader_mixin_init_with_no_context. Retrieved 1/12 statements.
# Partially parsed test_extension_loader_mixin_init_with_context. Retrieved 8/18 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_with_extensions. Retrieved 9/12 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_without_extensions. Retrieved 4/7 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_empty_context. Retrieved 2/5 statements.
# Partially parsed test_extension_loader_mixin_init_import_error. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'Test ExtensionLoaderMixin initialization with no context.'

def test_case_0():
    var_0 = 'Test ExtensionLoaderMixin initialization with context containing extensions.'
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 'custom.extension.One'
    var_4 = 'custom.extension.Two'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}

def test_case_0():
    var_0 = 'Test _read_extensions method when extensions are present in context.'
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 'ext1'
    var_4 = 'ext2'
    var_5 = 'ext3'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_2: var_6}
    var_8 = {var_1: var_7}

def test_case_0():
    var_0 = 'Test _read_extensions method when extensions are not in context.'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'Test _read_extensions method with empty context.'
    var_1 = {}

def test_case_0():
    var_0 = 'Test ExtensionLoaderMixin initialization raises UnknownExtension on ImportError.'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_extension_loader_mixin_context_not_none. Retrieved 9/23 statements.


def test_case_0():
    var_0 = []
    var_1 = 'TestLoader'
    var_2 = '__init__'
    var_3 = '_read_extensions'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = None
    var_8 = var_6 is var_7
    assert var_8 is False



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_init_with_default_extensions. Retrieved 2/14 statements.
# Partially parsed test_init_with_empty_context. Retrieved 4/12 statements.
# Partially parsed test_read_extensions_with_valid_context. Retrieved 9/13 statements.
# Partially parsed test_read_extensions_with_missing_cookiecutter_key. Retrieved 2/6 statements.
# Partially parsed test_read_extensions_with_missing_extensions_key. Retrieved 4/8 statements.
# Partially parsed test_read_extensions_converts_to_string. Retrieved 9/13 statements.
# Partially parsed test_read_extensions_with_empty_extensions_list. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'Test that __init__ loads default extensions when no context is provided.'
    var_1 = []

def test_case_0():
    var_0 = 'Test that __init__ works with empty context dictionary.'
    var_1 = []
    var_2 = []
    var_3 = {}

def test_case_0():
    var_0 = 'Test _read_extensions returns extensions from context.'
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 'ext1'
    var_4 = 'ext2'
    var_5 = 'ext3'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_2: var_6}
    var_8 = {var_1: var_7}

def test_case_0():
    var_0 = 'Test _read_extensions returns empty list when cookiecutter key is missing.'
    var_1 = {}

def test_case_0():
    var_0 = 'Test _read_extensions returns empty list when _extensions key is missing.'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'Test _read_extensions converts extension items to strings.'
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 1
    var_4 = 2.5
    var_5 = 'text'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_2: var_6}
    var_8 = {var_1: var_7}

def test_case_0():
    var_0 = 'Test _read_extensions returns empty list when _extensions is empty.'
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = []
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_extension_loader_mixin_init_with_none_context. Retrieved 3/15 statements.


def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 'extensions'



# Parsed testcases at query #10
#--------------------------

# Failed to parse test_extension_loader_mixin_init_with_no_context.
# Partially parsed test_extension_loader_mixin_init_with_empty_context. Retrieved 1/7 statements.
# Partially parsed test_extension_loader_mixin_init_with_extensions_in_context. Retrieved 7/13 statements.
# Partially parsed test_extension_loader_mixin_init_includes_default_extensions. Retrieved 1/7 statements.
# Partially parsed test_extension_loader_mixin_init_with_none_context. Retrieved 1/9 statements.
# Partially parsed test_extension_loader_mixin_init_passes_kwargs. Retrieved 2/9 statements.


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
    var_0 = {}

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = {}
    var_1 = True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_extension_loader_mixin_context_not_none. Retrieved 4/16 statements.


def test_case_0():
    var_0 = "Test that the predicate 'context is None' evaluates to False when context is provided."
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_extension_loader_mixin_context_predicate_false. Retrieved 4/16 statements.


def test_case_0():
    var_0 = "Test that the predicate 'context is None' at line 1 evaluates to False when context is provided."
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}



# Parsed testcases at query #13
#--------------------------

# Failed to parse test_extension_loader_mixin_init_with_no_context.
# Partially parsed test_extension_loader_mixin_init_with_empty_context. Retrieved 1/7 statements.
# Partially parsed test_extension_loader_mixin_init_with_extensions_in_context. Retrieved 6/12 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_empty_context. Retrieved 1/5 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_with_extensions. Retrieved 8/12 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_missing_cookiecutter_key. Retrieved 3/7 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_missing_extensions_key. Retrieved 5/9 statements.


def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'jinja2.ext.DebugExtension'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'ext1'
    var_3 = 'ext2'
    var_4 = 'ext3'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}

def test_case_0():
    var_0 = 'other_key'
    var_1 = 'value'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'other_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_import_error_raises_unknown_extension. Retrieved 1/9 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_context_parameter_predicate_evaluates_to_false. Retrieved 6/18 statements.


def test_case_0():
    var_0 = "Test that the predicate 'context is None' at line 1 evaluates to False when context is provided."
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = []
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_extension_loader_mixin_handles_import_error. Retrieved 8/17 statements.


def test_case_0():
    var_0 = 'Test that ImportError is caught and re-raised as UnknownExtension at line 23.'
    var_1 = None
    var_2 = False
    var_3 = ''
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = True
    assert var_7 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_extension_loader_mixin_context_none_evaluates_to_false. Retrieved 2/10 statements.


def test_case_0():
    var_0 = "Test that the predicate 'context' at line 1 evaluates to False when None is passed."
    var_1 = None



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_extension_loader_mixin_init_with_none_context. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'Test that ExtensionLoaderMixin.__init__ handles None context correctly.'
    var_1 = None



# Parsed testcases at query #19
#--------------------------

# Failed to parse test_extension_loader_mixin_init_with_no_context.
# Partially parsed test_extension_loader_mixin_init_with_empty_context. Retrieved 1/14 statements.
# Partially parsed test_extension_loader_mixin_init_with_custom_extensions. Retrieved 7/20 statements.
# Failed to parse test_extension_loader_mixin_init_raises_unknown_extension_on_import_error.
# Partially parsed test_extension_loader_mixin_read_extensions_with_no_extensions_key. Retrieved 1/5 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_with_extensions. Retrieved 8/12 statements.


def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'my.custom.Extension'
    var_3 = 'another.Extension'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'ext1'
    var_3 = 'ext2'
    var_4 = 'ext3'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_extension_loader_mixin_init_with_no_context. Retrieved 1/11 statements.
# Partially parsed test_extension_loader_mixin_init_with_context. Retrieved 7/17 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_with_extensions. Retrieved 9/15 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_without_extensions. Retrieved 4/10 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_empty_context. Retrieved 2/8 statements.
# Partially parsed test_extension_loader_mixin_init_import_error. Retrieved 1/9 statements.
# Partially parsed test_extension_loader_mixin_default_extensions. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'Test ExtensionLoaderMixin initialization with no context.'

def test_case_0():
    var_0 = 'Test ExtensionLoaderMixin initialization with context.'
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 'my.custom.Extension'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}

def test_case_0():
    var_0 = 'Test _read_extensions method when extensions are present.'
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 'ext1'
    var_4 = 'ext2'
    var_5 = 'ext3'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_2: var_6}
    var_8 = {var_1: var_7}

def test_case_0():
    var_0 = 'Test _read_extensions method when extensions are not present.'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'Test _read_extensions method with empty context.'
    var_1 = {}

def test_case_0():
    var_0 = 'Test ExtensionLoaderMixin initialization handles ImportError.'

def test_case_0():
    var_0 = 'Test that default extensions are included.'
    var_1 = {}
    var_2 = 'cookiecutter.extensions.JsonifyExtension'
    var_3 = 'cookiecutter.extensions.RandomStringExtension'
    var_4 = 'cookiecutter.extensions.SlugifyExtension'
    var_5 = 'cookiecutter.extensions.TimeExtension'
    var_6 = 'cookiecutter.extensions.UUIDExtension'
    var_7 = [var_2, var_3, var_4, var_5, var_6]



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_extension_loader_mixin_init_with_none_context. Retrieved 7/18 statements.


def test_case_0():
    var_0 = None
    var_1 = 'cookiecutter.extensions.JsonifyExtension'
    var_2 = 'cookiecutter.extensions.RandomStringExtension'
    var_3 = 'cookiecutter.extensions.SlugifyExtension'
    var_4 = 'cookiecutter.extensions.TimeExtension'
    var_5 = 'cookiecutter.extensions.UUIDExtension'
    var_6 = [var_1, var_2, var_3, var_4, var_5]



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_extension_loader_mixin_handles_import_error. Retrieved 3/14 statements.


def test_case_0():
    var_0 = 'Test that ImportError is caught and re-raised as UnknownExtension at line 23.'
    var_1 = 'test error'
    var_2 = {}



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_extension_loader_mixin_context_none_evaluates_to_false. Retrieved 1/14 statements.


def test_case_0():
    var_0 = "Test that the predicate 'context' at line 1 evaluates to False when None is passed."



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_context_parameter_defaults_to_empty_dict_when_none. Retrieved 8/15 statements.


def test_case_0():
    var_0 = 'Test that context parameter defaults to empty dict when None is passed.'
    var_1 = None
    var_2 = 'cookiecutter.extensions.JsonifyExtension'
    var_3 = 'cookiecutter.extensions.RandomStringExtension'
    var_4 = 'cookiecutter.extensions.SlugifyExtension'
    var_5 = 'cookiecutter.extensions.TimeExtension'
    var_6 = 'cookiecutter.extensions.UUIDExtension'
    var_7 = [var_2, var_3, var_4, var_5, var_6]



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_extension_loader_mixin_init_with_no_context.
# Partially parsed test_extension_loader_mixin_init_with_empty_context. Retrieved 1/6 statements.
# Partially parsed test_extension_loader_mixin_init_with_extensions_in_context. Retrieved 6/11 statements.
# Partially parsed test_extension_loader_mixin_init_with_multiple_extensions. Retrieved 7/12 statements.
# Partially parsed test_extension_loader_mixin_init_calls_read_extensions. Retrieved 5/10 statements.
# Partially parsed test_extension_loader_mixin_init_with_invalid_extension. Retrieved 6/11 statements.


def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'jinja2.ext.DebugExtension'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'jinja2.ext.DebugExtension'
    var_3 = 'jinja2.ext.LoopControlsExtension'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = []
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'nonexistent.extension.that.does.not.exist'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_import_error_raises_unknown_extension. Retrieved 5/17 statements.


def test_case_0():
    var_0 = 'Module not found'
    var_1 = ImportError(var_0)
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_extension_loader_mixin_init_with_no_context. Retrieved 1/6 statements.
# Partially parsed test_extension_loader_mixin_init_with_empty_context. Retrieved 2/7 statements.
# Partially parsed test_extension_loader_mixin_init_with_custom_extensions. Retrieved 7/12 statements.
# Partially parsed test_extension_loader_mixin_init_loads_default_extensions. Retrieved 3/11 statements.
# Partially parsed test_extension_loader_mixin_init_with_invalid_extension. Retrieved 7/13 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_with_no_extensions. Retrieved 2/8 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_with_extensions. Retrieved 8/15 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_missing_cookiecutter_key. Retrieved 4/10 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_missing_extensions_key. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'Test ExtensionLoaderMixin initialization with no context.'

def test_case_0():
    var_0 = 'Test ExtensionLoaderMixin initialization with empty context.'
    var_1 = {}

def test_case_0():
    var_0 = 'Test ExtensionLoaderMixin initialization with custom extensions in context.'
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 'jinja2.ext.DebugExtension'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}

def test_case_0():
    var_0 = 'Test that ExtensionLoaderMixin loads default extensions.'
    var_1 = {}
    var_2 = 0

def test_case_0():
    var_0 = 'Test ExtensionLoaderMixin initialization with invalid extension.'
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 'nonexistent.extension.DoesNotExist'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}

def test_case_0():
    var_0 = 'Test _read_extensions method returns empty list when no extensions in context.'
    var_1 = {}

def test_case_0():
    var_0 = 'Test _read_extensions method returns extensions from context.'
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 'jinja2.ext.DebugExtension'
    var_4 = 'jinja2.ext.LoopControlsExtension'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}

def test_case_0():
    var_0 = 'Test _read_extensions returns empty list when cookiecutter key is missing.'
    var_1 = 'other_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'Test _read_extensions returns empty list when _extensions key is missing.'
    var_1 = 'cookiecutter'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_extension_loader_mixin_context_predicate_false. Retrieved 4/19 statements.


def test_case_0():
    var_0 = "Test that the predicate 'context is None' at line 1 evaluates to False when context is provided."
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_extension_loader_mixin_catches_import_error. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'Test that ImportError at line 23 is caught and re-raised as UnknownExtension.'
    var_1 = {}



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_extension_loader_mixin_init_with_default_extensions. Retrieved 1/8 statements.
# Partially parsed test_extension_loader_mixin_init_with_context_extensions. Retrieved 7/15 statements.
# Partially parsed test_extension_loader_mixin_init_without_context. Retrieved 1/9 statements.
# Partially parsed test_extension_loader_mixin_init_with_import_error. Retrieved 2/11 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_with_valid_context. Retrieved 8/16 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_without_extensions_key. Retrieved 3/11 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_without_cookiecutter_key. Retrieved 1/9 statements.
# Partially parsed test_extension_loader_mixin_init_passes_kwargs. Retrieved 3/11 statements.


def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'my.custom.Extension'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = 1

def test_case_0():
    var_0 = 1

def test_case_0():
    var_0 = 'Module not found'
    var_1 = {}

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'ext1'
    var_3 = 'ext2'
    var_4 = 'ext3'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = {}
    var_1 = 'value'
    var_2 = 1



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_extension_loader_mixin_init_with_no_context. Retrieved 1/8 statements.
# Partially parsed test_extension_loader_mixin_init_with_context. Retrieved 6/12 statements.
# Partially parsed test_extension_loader_mixin_init_with_custom_extensions. Retrieved 7/13 statements.
# Partially parsed test_extension_loader_mixin_init_import_error. Retrieved 2/9 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_with_extensions. Retrieved 7/9 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_without_extensions. Retrieved 3/5 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_empty_context. Retrieved 1/3 statements.


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = module_0.ExtensionLoaderMixin()

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = []
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.ExtensionLoaderMixin(context=var_4)

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'custom.extension'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = module_0.ExtensionLoaderMixin(context=var_5)

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Module not found'
    var_1 = module_0.ExtensionLoaderMixin()

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_extensions'
    var_2 = 'ext1'
    var_3 = 'ext2'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = {}



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_import_error_raises_unknown_extension. Retrieved 3/13 statements.


def test_case_0():
    var_0 = 'test extension not found'
    var_1 = ImportError(var_0)
    var_2 = {}



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_extension_loader_mixin_init_with_default_extensions. Retrieved 2/7 statements.
# Partially parsed test_extension_loader_mixin_init_with_empty_context. Retrieved 2/6 statements.
# Partially parsed test_extension_loader_mixin_init_with_none_context. Retrieved 2/6 statements.
# Partially parsed test_extension_loader_mixin_init_with_custom_extensions. Retrieved 7/11 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_with_valid_context. Retrieved 9/14 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_with_missing_cookiecutter_key. Retrieved 2/7 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_with_missing_extensions_key. Retrieved 4/9 statements.
# Partially parsed test_extension_loader_mixin_read_extensions_converts_to_string. Retrieved 9/14 statements.


def test_case_0():
    var_0 = 'Test that ExtensionLoaderMixin initializes with default extensions.'
    var_1 = {}

def test_case_0():
    var_0 = 'Test that ExtensionLoaderMixin initializes with empty context.'
    var_1 = {}

def test_case_0():
    var_0 = 'Test that ExtensionLoaderMixin initializes with None context.'
    var_1 = None

def test_case_0():
    var_0 = 'Test that ExtensionLoaderMixin initializes with custom extensions.'
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 'jinja2.ext.DebugExtension'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}

def test_case_0():
    var_0 = 'Test _read_extensions returns extensions from context.'
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 'ext1'
    var_4 = 'ext2'
    var_5 = 'ext3'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_2: var_6}
    var_8 = {var_1: var_7}

def test_case_0():
    var_0 = 'Test _read_extensions returns empty list when cookiecutter key missing.'
    var_1 = {}

def test_case_0():
    var_0 = 'Test _read_extensions returns empty list when _extensions key missing.'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'Test _read_extensions converts extension objects to strings.'
    var_1 = 'cookiecutter'
    var_2 = '_extensions'
    var_3 = 123
    var_4 = 456.789
    var_5 = True
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_2: var_6}
    var_8 = {var_1: var_7}



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_extension_loader_mixin_handles_import_error. Retrieved 2/13 statements.


def test_case_0():
    var_0 = "No module named 'fake_extension'"
    var_1 = {}



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_extension_loader_mixin_handles_import_error. Retrieved 3/14 statements.


def test_case_0():
    var_0 = 'Test that ImportError at line 23 is caught and re-raised as UnknownExtension.'
    var_1 = 'test import error'
    var_2 = {}



