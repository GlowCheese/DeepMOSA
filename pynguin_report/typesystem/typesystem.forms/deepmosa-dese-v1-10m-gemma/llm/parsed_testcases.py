####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_load_template_env_directory_only. Retrieved 2/6 statements.
# Partially parsed test_load_template_env_package_only. Retrieved 3/7 statements.
# Partially parsed test_load_template_env_both_directory_and_package. Retrieved 4/15 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/tmp/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'my_package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = 'templates'

import typesystem.forms as module_0

def test_case_0():
    var_0 = '/tmp/templates'
    var_1 = 'my_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = 'templates'

import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Jinja2Forms(directory=var_0, package=var_0)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_jinja2forms_init_with_directory. Retrieved 4/8 statements.
# Partially parsed test_jinja2forms_init_with_package. Retrieved 4/8 statements.
# Partially parsed test_jinja2forms_init_with_both_directory_and_package. Retrieved 5/9 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/tmp/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = var_1.env.loader

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'my_package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env
    var_3 = var_1.env.loader

import typesystem.forms as module_0

def test_case_0():
    var_0 = '/tmp/templates'
    var_1 = 'my_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.env
    var_4 = var_2.env.loader

import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_create_form_returns_form_instance. Retrieved 6/17 statements.
# Partially parsed test_create_form_with_no_values_initializes_empty. Retrieved 4/15 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'test'
    var_5 = {var_0: var_4}

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_load_template_env_predicate_at_line_10_is_false. Retrieved 1/9 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_load_template_env_directory_only. Retrieved 3/7 statements.
# Partially parsed test_load_template_env_package_only. Retrieved 4/8 statements.
# Partially parsed test_load_template_env_both_directory_and_package. Retrieved 5/12 statements.
# Partially parsed test_load_template_env_logic_flow_directory. Retrieved 3/6 statements.
# Partially parsed test_load_template_env_logic_flow_package. Retrieved 4/7 statements.
# Partially parsed test_load_template_env_logic_flow_choice. Retrieved 4/9 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.load_template_env(directory=var_0)

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = 'templates'
    var_3 = var_1.load_template_env(package=var_0)

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = 0
    var_4 = var_2.load_template_env(directory=var_0, package=var_1)

import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Jinja2Forms(directory=var_0, package=var_0)

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'dir'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.load_template_env(directory=var_0)

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'pkg'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.load_template_env(package=var_0)
    var_3 = 'templates'

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'dir'
    var_1 = 'pkg'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.load_template_env(directory=var_0, package=var_1)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_form_html_returns_markup_of_rendered_fields. Retrieved 4/33 statements.
# Partially parsed test_form_html_reflects_rendered_fields_content. Retrieved 4/31 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'val'
    var_2 = {var_0: var_1}
    var_3 = '<forms/input.html>test-field</forms/input.html>'

def test_case_0():
    var_0 = 'my_field'
    var_1 = 'val'
    var_2 = {var_0: var_1}
    var_3 = 'rendered_my_field'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_form_html_returns_markup_instance. Retrieved 4/25 statements.


def test_case_0():
    var_0 = False
    var_1 = 'Test'
    var_2 = False
    var_3 = lambda self: True
    var_4 = 'test_field'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_init_raises_assertion_error_when_jinja2_is_none. Retrieved 2/6 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)



# Parsed testcases at query #9
#--------------------------

# Failed to parse test_input_type_for_field_returns_text_when_no_format_exists.
# Partially parsed test_input_type_for_field_returns_correct_mapped_type. Retrieved 3/19 statements.
# Partially parsed test_input_type_for_field_returns_text_for_unmapped_format. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'email'
    var_1 = 'number'
    var_2 = 'date'

def test_case_0():
    var_0 = 'unknown_type'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_jinja2forms_init_with_directory. Retrieved 4/8 statements.
# Partially parsed test_jinja2forms_init_with_package. Retrieved 4/8 statements.
# Partially parsed test_jinja2forms_init_with_both_directory_and_package. Retrieved 5/9 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/path/to/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = var_1.env.loader

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'my_package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env
    var_3 = var_1.env.loader

import typesystem.forms as module_0

def test_case_0():
    var_0 = '/path/to/templates'
    var_1 = 'my_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.env
    var_4 = var_2.env.loader

import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = 'Did not raise AssertionError'
    var_2 = AssertionError(var_1)

import typesystem.forms as module_0

def test_case_0():
    var_0 = '/path/to/templates'
    var_1 = 'my_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.env.loader.loaders
    var_4 = len(var_3)
    assert var_4 == 2



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_input_type_for_field_text. Retrieved 2/12 statements.
# Partially parsed test_input_type_for_field_email. Retrieved 2/12 statements.
# Partially parsed test_input_type_for_field_number. Retrieved 2/12 statements.
# Partially parsed test_input_type_for_field_no_format. Retrieved 1/11 statements.
# Partially parsed test_input_type_for_field_unknown_format. Retrieved 2/12 statements.
# Partially parsed test_input_type_for_field_date. Retrieved 2/12 statements.
# Partially parsed test_input_type_for_field_password. Retrieved 2/12 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'text'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'email'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'number'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'unknown'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'date'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'password'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_form_html_returns_markup_of_rendered_fields. Retrieved 2/29 statements.
# Partially parsed test_form_html_renders_all_non_readonly_fields. Retrieved 5/35 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'Rendered test_field'

def test_case_0():
    var_0 = 'f1'
    var_1 = 'f2'
    var_2 = False
    var_3 = True
    var_4 = '<f1>'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_load_template_env_predicate_false. Retrieved 1/11 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_form_constructor_initializes_correctly. Retrieved 15/22 statements.
# Partially parsed test_form_constructor_handles_none_values. Retrieved 12/17 statements.


def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = None
    var_4 = lambda self, x: var_3
    var_5 = {var_2: var_4}
    var_6 = 'MockSchema'
    var_7 = ()
    var_8 = 'serialize'
    var_9 = {}
    var_10 = 'name'
    var_11 = 'age'
    var_12 = 'test'
    var_13 = 25
    var_14 = {var_10: var_12, var_11: var_13}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = None
    var_4 = lambda self, x: var_3
    var_5 = {var_2: var_4}
    var_6 = 'MockSchema'
    var_7 = ()
    var_8 = 'serialize'
    var_9 = {}
    var_10 = lambda self, v: var_9
    var_11 = {var_8: var_10}



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_form_constructor_initializes_with_values. Retrieved 17/22 statements.
# Partially parsed test_form_constructor_initializes_with_none_values. Retrieved 12/17 statements.
# Partially parsed test_form_constructor_handles_empty_values. Retrieved 12/17 statements.


def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = None
    var_4 = lambda self, x: var_3
    var_5 = {var_2: var_4}
    var_6 = 'MockSchema'
    var_7 = ()
    var_8 = 'serialize'
    var_9 = 'name'
    var_10 = 'John'
    var_11 = {var_9: var_10}
    var_12 = {var_9: var_10}
    var_13 = {}
    var_14 = lambda self, v: var_12 if v == var_11 else var_13
    var_15 = {var_8: var_14}
    var_16 = {var_9: var_10}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = None
    var_4 = lambda self, x: var_3
    var_5 = {var_2: var_4}
    var_6 = 'MockSchema'
    var_7 = ()
    var_8 = 'serialize'
    var_9 = {}
    var_10 = lambda self, v: var_9
    var_11 = {var_8: var_10}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = None
    var_4 = lambda self, x: var_3
    var_5 = {var_2: var_4}
    var_6 = 'MockSchema'
    var_7 = ()
    var_8 = 'serialize'
    var_9 = {}
    var_10 = lambda self, v: var_9
    var_11 = {var_8: var_10}



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_form_str_renders_fields. Retrieved 4/30 statements.
# Partially parsed test_form_str_with_read_only_field_skips_rendering. Retrieved 4/30 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'val'
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}

def test_case_0():
    var_0 = True
    var_1 = 'readonly_field'
    var_2 = {}
    var_3 = {}



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_render_field_basic_input. Retrieved 11/21 statements.
# Partially parsed test_render_field_password_masks_value. Retrieved 10/18 statements.
# Partially parsed test_render_field_with_error. Retrieved 10/18 statements.
# Partially parsed test_render_field_id_transformation. Retrieved 9/17 statements.
# Partially parsed test_render_field_required_attribute. Retrieved 13/22 statements.


import typesystem.fields as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Environment()
    var_2 = '{{ field_id }}: {{ value }}'
    var_3 = var_1.from_string(var_2)
    var_4 = 'User Name'
    var_5 = module_0.String()
    var_6 = 'name'
    var_7 = 'John'
    var_8 = {var_6: var_7}
    var_9 = None
    var_10 = 'forms/input.html'

import typesystem.fields as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Environment()
    var_2 = '{{ value }}'
    var_3 = var_1.from_string(var_2)
    var_4 = 'password'
    var_5 = module_0.String(format=var_4)
    var_6 = 'pwd'
    var_7 = 'secret'
    var_8 = {var_6: var_7}
    var_9 = None

import typesystem.fields as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Environment()
    var_2 = '{{ error }}'
    var_3 = var_1.from_string(var_2)
    var_4 = 'Email Address'
    var_5 = module_0.String()
    var_6 = {}
    var_7 = 'email'
    var_8 = 'invalid-email'
    var_9 = 'Invalid email format'

import typesystem.fields as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Environment()
    var_2 = '{{ field_id }}'
    var_3 = var_1.from_string(var_2)
    var_4 = module_0.String()
    var_5 = {}
    var_6 = 'user_name'
    var_7 = 'test'
    var_8 = None

import typesystem.fields as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.String()
    var_2 = module_1.Environment()
    var_3 = '{{ required }}'
    var_4 = var_2.from_string(var_3)
    var_5 = {}
    var_6 = module_0.String()
    var_7 = 'req'
    var_8 = 'v'
    var_9 = None
    var_10 = 'something'
    var_11 = module_0.String()
    var_12 = 'opt'



# Parsed testcases at query #18
#--------------------------




def test_case_0():
    pass

def test_case_0():
    pass

def test_case_0():
    pass

def test_case_0():
    pass

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'some_dir'
    var_1 = module_0.Jinja2Forms(directory=var_0)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_template_for_field_choice. Retrieved 4/14 statements.
# Partially parsed test_template_for_field_boolean. Retrieved 2/10 statements.
# Partially parsed test_template_for_field_string_textarea. Retrieved 3/11 statements.
# Partially parsed test_template_for_field_string_input. Retrieved 3/11 statements.
# Partially parsed test_template_for_field_integer_input. Retrieved 2/10 statements.
# Partially parsed test_template_for_field_raises_error_on_object. Retrieved 2/11 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean()

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'text'
    var_2 = module_0.String(format=var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'email'
    var_2 = module_0.String(format=var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer()

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object()



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_form_init_keyword_only_args. Retrieved 6/16 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'test'
    var_5 = {var_0: var_4}



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_validate_updates_data_and_errors. Retrieved 7/12 statements.
# Partially parsed test_validate_sets_errors_on_failure. Retrieved 7/12 statements.
# Partially parsed test_validate_raises_error_if_called_twice. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'Old'
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_1}

def test_case_0():
    var_0 = 'name'
    var_1 = ''
    var_2 = {var_0: var_1}
    var_3 = 'Required'
    var_4 = {var_0: var_3}
    var_5 = {}
    var_6 = {var_0: var_1}

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = {}
    var_5 = 'First'
    var_6 = {var_0: var_5}



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_form_str_calls_render_fields. Retrieved 2/14 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'John'



# Parsed testcases at query #23
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = 'tmp'
    var_1 = 'my_pkg'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_form_constructor_initializes_correctly. Retrieved 17/22 statements.
# Partially parsed test_form_constructor_handles_none_values. Retrieved 12/17 statements.


def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = None
    var_4 = lambda self, x: var_3
    var_5 = {var_2: var_4}
    var_6 = 'MockSchema'
    var_7 = ()
    var_8 = 'serialize'
    var_9 = 'name'
    var_10 = 'John'
    var_11 = {var_9: var_10}
    var_12 = {var_9: var_10}
    var_13 = {}
    var_14 = lambda self, v: var_12 if v == var_11 else var_13
    var_15 = {var_8: var_14}
    var_16 = {var_9: var_10}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = None
    var_4 = lambda self, x: var_3
    var_5 = {var_2: var_4}
    var_6 = 'MockSchema'
    var_7 = ()
    var_8 = 'serialize'
    var_9 = {}
    var_10 = lambda self, v: var_9
    var_11 = {var_8: var_10}



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_form_str_calls_render_fields. Retrieved 2/13 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'test'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_validate_success_on_first_call. Retrieved 4/21 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_load_template_env_directory_only. Retrieved 3/9 statements.
# Partially parsed test_load_template_env_package_only. Retrieved 4/10 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'some/dir'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'my_package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = 'templates'
    var_3 = True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_jinja2forms_init_with_directory. Retrieved 2/4 statements.
# Partially parsed test_jinja2forms_init_with_package. Retrieved 3/5 statements.
# Partially parsed test_jinja2forms_init_with_both. Retrieved 3/5 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'templates_dir'
    var_1 = module_0.Jinja2Forms(directory=var_0)

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'my_package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = 'templates'

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'dir'
    var_1 = 'pkg'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)

import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'dir'
    var_1 = 'pkg'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_init_trigger_assertion_error. Retrieved 5/12 statements.


def test_case_0():
    pass

def test_case_0():
    pass

def test_case_0():
    var_0 = 'sys.modules'
    var_1 = 'jinja2'
    var_2 = None
    var_3 = {var_1: var_2}
    var_4 = 'some_dir'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_form_constructor_initialization. Retrieved 17/22 statements.
# Partially parsed test_form_constructor_with_none_values. Retrieved 12/17 statements.


def test_case_0():
    var_0 = 'Env'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = None
    var_4 = lambda self, x: var_3
    var_5 = {var_2: var_4}
    var_6 = 'Schema'
    var_7 = ()
    var_8 = 'serialize'
    var_9 = 'name'
    var_10 = 'test_value'
    var_11 = {var_9: var_10}
    var_12 = {}
    var_13 = lambda self, v: var_11 if v else var_12
    var_14 = {var_8: var_13}
    var_15 = 'raw_value'
    var_16 = {var_9: var_15}

def test_case_0():
    var_0 = 'Env'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = None
    var_4 = lambda self, x: var_3
    var_5 = {var_2: var_4}
    var_6 = 'Schema'
    var_7 = ()
    var_8 = 'serialize'
    var_9 = {}
    var_10 = lambda self, v: var_9
    var_11 = {var_8: var_10}



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_template_for_field_choice. Retrieved 4/16 statements.
# Partially parsed test_template_for_call_boolean. Retrieved 5/13 statements.
# Partially parsed test_template_for_field_string_textarea. Retrieved 5/13 statements.
# Partially parsed test_template_for_field_string_input. Retrieved 5/13 statements.
# Partially parsed test_template_for_field_default_input. Retrieved 5/13 statements.
# Partially parsed test_template_for_field_raises_on_object. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'choice'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'bool'
    var_1 = module_0.Boolean()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = var_3.fields[var_0]

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'text'
    var_1 = module_0.String(format=var_0)
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = var_3.fields[var_0]

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = module_0.String(format=var_0)
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = var_3.fields[var_0]

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'str'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = var_3.fields[var_0]

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'obj'
    var_1 = module_0.Object()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = var_3.fields[var_0]



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_form_init_keyword_only_arguments. Retrieved 6/11 statements.
# Partially parsed test_form_init_fails_without_env_keyword. Retrieved 4/10 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'test'
    var_5 = {var_0: var_4}

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_render_field_basic_input. Retrieved 10/29 statements.
# Partially parsed test_render_field_password_masks_value. Retrieved 5/23 statements.
# Partially parsed test_render_field_with_error. Retrieved 5/19 statements.
# Partially parsed test_render_field_id_transformation. Retrieved 4/18 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Username'
    var_1 = 'text'
    var_2 = module_0.String(format=var_1)
    var_3 = 'username'
    var_4 = 'john_doe'
    var_5 = None
    var_6 = 'label'
    var_7 = 1
    var_8 = 0
    var_9 = 'value'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'password'
    var_1 = module_0.String(format=var_0)
    var_2 = '<input>'
    var_3 = 'secret123'
    var_4 = None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Email'
    var_1 = 'email'
    var_2 = module_0.String(format=var_1)
    var_3 = 'invalid-email'
    var_4 = 'Invalid email format'

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'user_name_field'
    var_2 = 'val'
    var_3 = None



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_input_type_for_field_text_default. Retrieved 3/12 statements.
# Partially parsed test_input_type_for_field_mapping. Retrieved 3/19 statements.
# Partially parsed test_input_type_for_field_unsupported_format. Retrieved 3/13 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = None
    var_2 = 'test'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = None
    var_2 = 'test'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = None
    var_2 = 'test'

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'email'
    var_2 = 'url'
    var_3 = module_1.String(format=var_1)
    var_4 = module_1.String(format=var_2)
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_2.Schema(var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = var_6.fields[var_1]
    var_9 = var_7.input_type_for_field(var_8)
    assert var_9 == 'email'
    var_10 = var_6.fields[var_2]
    var_11 = var_7.input_type_for_field(var_10)
    assert var_11 == 'url'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_form_init_keyword_only_enforcement. Retrieved 5/11 statements.
# Partially parsed test_form_init_parameters_are_keyword_only. Retrieved 5/13 statements.


import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = module_1.String()
    var_3 = {var_1: var_2}
    var_4 = module_2.Schema(var_3)
    var_5 = module_3.Form(env=var_0, schema=var_4)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = module_1.String()
    var_3 = {var_1: var_2}
    var_4 = module_2.Schema(var_3)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = module_1.String()
    var_3 = {var_1: var_2}
    var_4 = module_2.Schema(var_3)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_input_type_for_field_with_format_present. Retrieved 2/14 statements.


def test_case_0():
    var_0 = 'email'
    var_1 = 'number'
    var_2 = 'email'
    var_3 = 'number'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'email'



# Parsed testcases at query #11
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = 'some_dir'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = None
    var_3 = var_1.load_template_env(directory=var_0, package=var_2)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_form_html_returns_markup_of_rendered_fields. Retrieved 3/34 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'val'
    var_2 = 'rendered-test-field'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_render_fields_renders_all_non_readonly_fields. Retrieved 7/35 statements.
# Partially parsed test_render_fields_uses_data_on_error. Retrieved 5/32 statements.
# Partially parsed test_render_fields_uses_values_when_no_error. Retrieved 5/32 statements.


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'field1'
    var_3 = 'field2'
    var_4 = 'val1'
    var_5 = {var_2: var_4}
    var_6 = {var_2: var_4}

def test_case_0():
    var_0 = 'field1'
    var_1 = 'initial'
    var_2 = {var_0: var_1}
    var_3 = 'new_data'
    var_4 = {var_0: var_3}

def test_case_0():
    var_0 = 'field1'
    var_1 = 'init'
    var_2 = {var_0: var_1}
    var_3 = 'data_val'
    var_4 = {var_0: var_3}



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_template_for_field_returns_select_for_choice. Retrieved 8/21 statements.
# Partially parsed test_template_for_field_returns_checkbox_for_boolean. Retrieved 1/14 statements.
# Partially parsed test_template_for_field_returns_textarea_for_string_text. Retrieved 2/17 statements.
# Partially parsed test_template_for_field_returns_input_for_other_fields. Retrieved 2/15 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.Choice(choices=var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = module_0.String(format=var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = module_0.String(format=var_0)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_init_fails_when_jinja2_is_none. Retrieved 2/8 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_render_field_renders_correctly. Retrieved 6/33 statements.
# Partially parsed test_render_field_password_masks_value. Retrieved 5/32 statements.
# Partially parsed test_render_field_handles_errors. Retrieved 1/25 statements.


def test_case_0():
    var_0 = 'Test Label'
    var_1 = 'email'
    var_2 = 'test_field'
    var_3 = 'test@example.com'
    var_4 = {var_2: var_3}
    var_5 = None

def test_case_0():
    var_0 = 'password'
    var_1 = 'pw'
    var_2 = 'secret123'
    var_3 = {var_1: var_2}
    var_4 = None

def test_case_0():
    var_0 = 'f'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_render_fields_valid_data. Retrieved 7/32 statements.
# Partially parsed test_render_fields_with_errors. Retrieved 6/29 statements.


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'f1'
    var_3 = 'f2'
    var_4 = 'val1'
    var_5 = {var_2: var_4}
    var_6 = {var_2: var_4}

def test_case_0():
    var_0 = False
    var_1 = 'f1'
    var_2 = 'new'
    var_3 = {var_1: var_2}
    var_4 = 'old'
    var_5 = {var_1: var_4}



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_render_fields_with_valid_data. Retrieved 4/32 statements.
# Partially parsed test_render_fields_skips_read_only. Retrieved 13/44 statements.
# Partially parsed test_render_fields_includes_errors. Retrieved 4/32 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}

def test_case_0():
    var_0 = True
    var_1 = 'readonly_field'
    var_2 = 'Env'
    var_3 = ()
    var_4 = 'get_template'
    var_5 = 'T'
    var_6 = ()
    var_7 = 'render'
    var_8 = ''
    var_9 = lambda self, c: var_8
    var_10 = {var_7: var_9}
    var_11 = {}
    var_12 = {}

def test_case_0():
    var_0 = 'name'
    var_1 = 'new_val'
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_render_field_allow_empty_true_via_allow_null. Retrieved 6/22 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test Field'
    var_1 = True
    var_2 = module_0.Field(title=var_0, allow_null=var_1)
    var_3 = 'test_field'
    var_4 = 'val'
    var_5 = None



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_template_for_field_raises_assertion_error_on_object_field. Retrieved 7/24 statements.
# Partially parsed test_template_for_field_reaches_line_10_with_true_condition. Retrieved 8/21 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'Template'
    var_2 = ()
    var_3 = 'render'
    var_4 = ''
    var_5 = lambda args: var_4
    var_6 = {var_3: var_5}

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'Template'
    var_2 = ()
    var_3 = 'render'
    var_4 = ''
    var_5 = lambda args: var_4
    var_6 = {var_3: var_5}
    var_7 = 'text_field'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_render_field_password_masks_value. Retrieved 3/17 statements.
# Partially parsed test_render_field_text_input_with_label. Retrieved 5/18 statements.
# Partially parsed test_render_field_id_transformation. Retrieved 1/9 statements.
# Partially parsed test_render_field_input_type_logic. Retrieved 2/18 statements.
# Partially parsed test_render_field_requires_logic. Retrieved 2/16 statements.
# Partially parsed test_render_field_error_passing. Retrieved 3/17 statements.


def test_case_0():
    var_0 = 'password_field'
    var_1 = 'password_field'
    var_2 = 'password'
    var_3 = 'secret123'

def test_case_0():
    var_0 = {}
    var_1 = 'User Name'
    var_2 = 'user_name'
    var_3 = 'John Doe'
    var_4 = None

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'val'

def test_case_0():
    var_0 = 'test'
    var_1 = 'val'

def test_case_0():
    var_0 = 'test'
    var_1 = 'val'
    var_2 = 'Error Message'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_template_for_field_choice. Retrieved 3/12 statements.
# Partially parsed test_template_for_field_boolean. Retrieved 1/8 statements.
# Partially parsed test_template_for_field_string_text. Retrieved 2/9 statements.
# Partially parsed test_template_for_field_string_other. Retrieved 2/9 statements.
# Partially parsed test_template_for_field_default_input. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = module_0.String(format=var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = module_0.String(format=var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_render_field_password_masks_value. Retrieved 7/21 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'user_password'
    var_2 = 'password'
    var_3 = module_0.String(format=var_2)
    var_4 = 'secret123'
    var_5 = None
    var_6 = 0



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_template_for_field_choice. Retrieved 5/15 statements.
# Partially parsed test_template_for_field_boolean. Retrieved 2/9 statements.
# Partially parsed test_template_for_field_string_textarea. Retrieved 3/10 statements.
# Partially parsed test_template_for_field_string_input. Retrieved 3/10 statements.
# Partially parsed test_template_for_field_default_input. Retrieved 2/9 statements.
# Partially parsed test_template_for_field_raises_on_object. Retrieved 6/16 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'a'
    var_2 = 'A'
    var_3 = {var_1: var_2}
    var_4 = 'Choice'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean()

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'text'
    var_2 = module_0.String(format=var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'email'
    var_2 = module_0.String(format=var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Field()

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'name'
    var_2 = module_0.String()
    var_3 = {var_1: var_2}
    var_4 = 'Should have raised AssertionError'
    var_5 = AssertionError(var_4)



