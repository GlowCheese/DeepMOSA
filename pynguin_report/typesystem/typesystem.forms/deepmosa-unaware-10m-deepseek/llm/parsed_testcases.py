####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = None
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'John'
    var_4 = 25
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'VeryLongNameExceedsLimit'
    var_7 = 200
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = {var_2: var_4}
    var_10 = 'Test'
    var_11 = 30
    var_12 = {var_1: var_10, var_2: var_11}
    var_13 = 'name'
    var_14 = 'age'
    var_15 = 'Another'
    var_16 = 40
    var_17 = {var_13: var_15, var_14: var_16}
    var_18 = 'Initial'
    var_19 = 99
    var_20 = {var_14: var_18, var_15: var_19}
    var_21 = 'Updated'
    var_22 = 100
    var_23 = {var_14: var_21, var_15: var_22}
    var_24 = 42
    var_25 = {var_14: var_10, var_15: var_24}
    var_26 = 'data'



# Parsed testcases at query #2
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/test/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = var_2.loader
    var_4 = 'test_package'
    var_5 = module_0.Jinja2Forms(package=var_4)
    var_6 = var_5.env
    var_7 = var_6.loader
    var_8 = module_0.Jinja2Forms(directory=var_0, package=var_4)
    var_9 = var_8.env
    var_10 = var_9.loader
    var_11 = var_9.loader.loaders
    var_12 = len(var_11)
    assert var_12 == 2
    var_13 = 0
    var_14 = var_9.loader.loaders[var_13]
    var_15 = 1
    var_16 = var_9.loader.loaders[var_15]



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0
import jinja2.loaders as module_1
import jinja2.environment as module_2

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Boolean()
    var_2 = 'forms/input.html'
    var_3 = 'forms/textarea.html'
    var_4 = 'forms/checkbox.html'
    var_5 = 'forms/select.html'
    var_6 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_7 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_8 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>'
    var_9 = '<select name="{{ field_name }}"><option value="{{ value }}">{{ value }}</option></select>'
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_9}
    var_11 = module_1.DictLoader(var_10)
    var_12 = module_2.Environment(loader=var_11)
    var_13 = 'name'
    var_14 = 'email'
    var_15 = 'age'
    var_16 = 'active'
    var_17 = 'role'
    var_18 = 'bio'
    var_19 = 'read_only_field'
    var_20 = 'John Doe'
    var_21 = 'john@example.com'
    var_22 = 30
    var_23 = True
    var_24 = 'admin'
    var_25 = 'Software developer'
    var_26 = 'hidden'
    var_27 = {var_13: var_20, var_14: var_21, var_15: var_22, var_16: var_23, var_17: var_24, var_18: var_25, var_19: var_26}
    var_28 = None



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'field3'
    var_3 = 'Field One'
    var_4 = False
    var_5 = True
    var_6 = 'Field Three'
    var_7 = 'value1'
    var_8 = 'value3'
    var_9 = {var_0: var_7, var_2: var_8}
    var_10 = {var_0: var_7, var_2: var_8}
    var_11 = 'initial'
    var_12 = {var_0: var_11}
    var_13 = 'invalid'
    var_14 = {var_0: var_13}
    var_15 = {}
    var_16 = 'password'
    var_17 = 'secret'
    var_18 = {var_16: var_17}
    var_19 = {var_16: var_17}
    var_20 = 'email'
    var_21 = 'number'
    var_22 = 'test@example.com'
    var_23 = 42
    var_24 = {var_20: var_22, var_21: var_23}
    var_25 = {var_20: var_22, var_21: var_23}



# Parsed testcases at query #5
#--------------------------


import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = '/test/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.loader
    var_3 = 'test_package'
    var_4 = module_0.Jinja2Forms(package=var_3)
    var_5 = var_4.env.loader
    var_6 = module_0.Jinja2Forms(directory=var_0, package=var_3)
    var_7 = var_6.env.loader
    var_8 = module_1.String()



# Parsed testcases at query #6
#--------------------------


import typesystem.fields as module_0
import jinja2.loaders as module_1
import jinja2.environment as module_2

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Boolean()
    var_2 = 'forms/input.html'
    var_3 = 'forms/checkbox.html'
    var_4 = 'forms/select.html'
    var_5 = 'forms/textarea.html'
    var_6 = '\n                <input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}">\n            '
    var_7 = '\n                <input type="checkbox" name="{{ field_name }}" id="{{ field_id }}" {% if value %}checked{% endif %}>\n            '
    var_8 = '\n                <select name="{{ field_name }}" id="{{ field_id }}">\n                    <option value="admin" {% if value == "admin" %}selected{% endif %}>admin</option>\n                    <option value="user" {% if value == "user" %}selected{% endif %}>user</option>\n                </select>\n            '
    var_9 = '\n                <textarea name="{{ field_name }}" id="{{ field_id }}">{{ value }}</textarea>\n            '
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_9}
    var_11 = module_1.DictLoader(var_10)
    var_12 = module_2.Environment(loader=var_11)
    var_13 = 'name'
    var_14 = 'email'
    var_15 = 'age'
    var_16 = 'active'
    var_17 = 'role'
    var_18 = 'bio'
    var_19 = 'John Doe'
    var_20 = 'john@example.com'
    var_21 = 30
    var_22 = True
    var_23 = 'admin'
    var_24 = 'Software developer'
    var_25 = {var_13: var_19, var_14: var_20, var_15: var_21, var_16: var_22, var_17: var_23, var_18: var_24}
    var_26 = None
    var_27 = module_0.String()
    var_28 = 'id'
    var_29 = 'Test'
    var_30 = '123'
    var_31 = {var_13: var_29, var_28: var_30}



# Parsed testcases at query #7
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = '<input>'
    var_1 = None
    var_2 = 'name'
    var_3 = 'age'
    var_4 = 'John'
    var_5 = 25
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'A'
    var_8 = 101
    var_9 = var_7 * var_8
    var_10 = -5
    var_11 = {var_2: var_9, var_3: var_10}
    var_12 = 'Initial'
    var_13 = 30
    var_14 = {var_2: var_12, var_3: var_13}
    var_15 = 'Updated'
    var_16 = 35
    var_17 = {var_2: var_15, var_3: var_16}
    var_18 = 'Test'
    var_19 = 20
    var_20 = {var_2: var_18, var_3: var_19}
    var_21 = 'name'
    var_22 = 'age'
    var_23 = 'Another'
    var_24 = 25
    var_25 = {var_21: var_23, var_22: var_24}
    var_26 = 'id'
    var_27 = '123'
    var_28 = {var_23: var_18, var_26: var_27, var_24: var_5}
    var_29 = module_0.Boolean()
    var_30 = module_0.Float()
    var_31 = 'active'
    var_32 = 'score'
    var_33 = 'true'
    var_34 = '42.5'
    var_35 = {var_31: var_33, var_32: var_34}
    var_36 = {}



# Parsed testcases at query #8
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/textarea.html'
    var_2 = 'forms/select.html'
    var_3 = 'forms/checkbox.html'
    var_4 = '\n                <input type="{{ input_type }}" \n                       id="{{ field_id }}" \n                       name="{{ field_name }}" \n                       value="{{ value }}" \n                       {% if required %}required{% endif %}>\n            '
    var_5 = '\n                <textarea id="{{ field_id }}" \n                          name="{{ field_name }}" \n                          {% if required %}required{% endif %}>{{ value }}</textarea>\n            '
    var_6 = '\n                <select id="{{ field_id }}" \n                        name="{{ field_name }}" \n                        {% if required %}required{% endif %}>\n                </select>\n            '
    var_7 = '\n                <input type="checkbox" \n                       id="{{ field_id }}" \n                       name="{{ field_name }}" \n                       {% if required %}required{% endif %}>\n            '
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'name'
    var_12 = 'email'
    var_13 = 'John'
    var_14 = 'john@example.com'
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = 'invalid-email'
    var_17 = {var_12: var_16}



# Parsed testcases at query #9
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/textarea.html'
    var_2 = 'forms/select.html'
    var_3 = 'forms/checkbox.html'
    var_4 = '\n                <input type="{{ input_type }}" \n                       id="{{ field_id }}" \n                       name="{{ field_name }}" \n                       value="{{ value }}" \n                       {% if required %}required{% endif %}>\n            '
    var_5 = '\n                <textarea id="{{ field_id }}" \n                          name="{{ field_name }}" \n                          {% if required %}required{% endif %}>{{ value }}</textarea>\n            '
    var_6 = '\n                <select id="{{ field_id }}" \n                        name="{{ field_name }}" \n                        {% if required %}required{% endif %}>\n                    <option value="">Select</option>\n                </select>\n            '
    var_7 = '\n                <input type="checkbox" \n                       id="{{ field_id }}" \n                       name="{{ field_name }}" \n                       value="true" \n                       {% if value %}checked{% endif %}>\n            '
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'description'
    var_12 = 'Test description'
    var_13 = 'email'
    var_14 = 'test@example.com'
    var_15 = 'color'
    var_16 = 'active'
    var_17 = True
    var_18 = 'username'
    var_19 = 'ab'
    var_20 = 'Must be at least 3 characters'
    var_21 = 'name'
    var_22 = 'optional'
    var_23 = 'password'
    var_24 = 'secret123'
    var_25 = 'custom'
    var_26 = 'first_name'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'Field One'
    var_3 = True
    var_4 = 'value1'
    var_5 = 'value2'
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = 'new_value1'
    var_8 = 'new_value2'
    var_9 = {var_0: var_7, var_1: var_8}
    var_10 = {}
    var_11 = 'error'
    var_12 = 'test'
    var_13 = {var_11: var_12}
    var_14 = 'initial'
    var_15 = {var_0: var_14}
    var_16 = 'Read Only Field'
    var_17 = {var_0: var_4, var_1: var_5}
    var_18 = 'new_value'
    var_19 = {var_0: var_18}
    var_20 = None
    var_21 = 'test_field'
    var_22 = {}
    var_23 = {var_21: var_12}
    var_24 = 'required_field'
    var_25 = 'optional_field'
    var_26 = False
    var_27 = {}
    var_28 = {}



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0
import jinja2.loaders as module_1
import jinja2.environment as module_2

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = 'forms/input.html'
    var_2 = 'forms/textarea.html'
    var_3 = 'forms/checkbox.html'
    var_4 = 'forms/select.html'
    var_5 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_6 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_7 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>'
    var_8 = '<select name="{{ field_name }}"></select>'
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = module_1.DictLoader(var_9)
    var_11 = module_2.Environment(loader=var_10)
    var_12 = 'name'
    var_13 = 'email'
    var_14 = 'age'
    var_15 = 'active'
    var_16 = 'John Doe'
    var_17 = 'john@example.com'
    var_18 = 30
    var_19 = False
    var_20 = {var_12: var_16, var_13: var_17, var_14: var_18, var_15: var_19}
    var_21 = ''
    var_22 = 'invalid'
    var_23 = 'not-a-number'
    var_24 = {var_12: var_21, var_13: var_22, var_14: var_23}
    var_25 = module_0.String()



# Parsed testcases at query #12
#--------------------------


import typesystem.fields as module_0
import jinja2.loaders as module_1
import jinja2.environment as module_2

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Boolean()
    var_2 = 'forms/input.html'
    var_3 = 'forms/checkbox.html'
    var_4 = 'forms/textarea.html'
    var_5 = 'forms/select.html'
    var_6 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_7 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>'
    var_8 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_9 = '<select name="{{ field_name }}">{{ value }}</select>'
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_9}
    var_11 = module_1.DictLoader(var_10)
    var_12 = module_2.Environment(loader=var_11)
    var_13 = 'name'
    var_14 = 'email'
    var_15 = 'age'
    var_16 = 'active'
    var_17 = 'John'
    var_18 = 'john@test.com'
    var_19 = 25
    var_20 = True
    var_21 = {var_13: var_17, var_14: var_18, var_15: var_19, var_16: var_20}
    var_22 = ''
    var_23 = 'invalid'
    var_24 = {var_13: var_22, var_14: var_23}



# Parsed testcases at query #13
#--------------------------


import typesystem.fields as module_0
import jinja2.loaders as module_1
import jinja2.environment as module_2

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Boolean()
    var_2 = 'forms/input.html'
    var_3 = 'forms/textarea.html'
    var_4 = 'forms/checkbox.html'
    var_5 = 'forms/select.html'
    var_6 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_7 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_8 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>'
    var_9 = '<select name="{{ field_name }}"><option value="{{ value }}">{{ value }}</option></select>'
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_9}
    var_11 = module_1.DictLoader(var_10)
    var_12 = module_2.Environment(loader=var_11)
    var_13 = 'name'
    var_14 = 'age'
    var_15 = 'active'
    var_16 = 'John'
    var_17 = 30
    var_18 = True
    var_19 = {var_13: var_16, var_14: var_17, var_15: var_18}
    var_20 = 'email'
    var_21 = 'Jane'
    var_22 = 'jane@test.com'
    var_23 = 'invalid'
    var_24 = False
    var_25 = {var_13: var_21, var_20: var_22, var_14: var_23, var_15: var_24}
    var_26 = None
    var_27 = 'Original'
    var_28 = {var_13: var_27}



# Parsed testcases at query #14
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import jinja2.utils as module_2
import typesystem.fields as module_3

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/textarea.html'
    var_2 = 'forms/checkbox.html'
    var_3 = 'forms/select.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_5 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_6 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>'
    var_7 = '<select name="{{ field_name }}"><option>{{ value }}</option></select>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'name'
    var_12 = 'email'
    var_13 = 'John'
    var_14 = 'john@example.com'
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = 'test'
    var_17 = module_2.htmlsafe_json_dumps(var_16)
    var_18 = {}
    var_19 = module_2.htmlsafe_json_dumps(var_16)
    var_20 = module_3.Boolean()
    var_21 = 'text'
    var_22 = 'boolean'
    var_23 = 'choice'
    var_24 = 'Hello'
    var_25 = True
    var_26 = 'a'
    var_27 = {var_21: var_24, var_22: var_25, var_23: var_26}
    var_28 = module_2.htmlsafe_json_dumps(var_16)



# Parsed testcases at query #15
#--------------------------


import typesystem.fields as module_0
import jinja2.loaders as module_1
import jinja2.environment as module_2

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Boolean()
    var_2 = 'forms/input.html'
    var_3 = 'forms/checkbox.html'
    var_4 = 'forms/select.html'
    var_5 = 'forms/textarea.html'
    var_6 = '\n        <input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" \n               value="{{ value }}" {% if required %}required{% endif %}>\n        '
    var_7 = '\n        <input type="checkbox" name="{{ field_name }}" id="{{ field_id }}" \n               {% if value %}checked{% endif %} {% if required %}required{% endif %}>\n        '
    var_8 = '\n        <select name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>\n            <option value="">Select</option>\n        </select>\n        '
    var_9 = '\n        <textarea name="{{ field_name }}" id="{{ field_id }}" \n                  {% if required %}required{% endif %}>{{ value }}</textarea>\n        '
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_9}
    var_11 = module_1.DictLoader(var_10)
    var_12 = module_2.Environment(loader=var_11)
    var_13 = 'name'
    var_14 = 'email'
    var_15 = 'age'
    var_16 = 'active'
    var_17 = 'role'
    var_18 = 'bio'
    var_19 = 'John Doe'
    var_20 = 'john@example.com'
    var_21 = 30
    var_22 = True
    var_23 = 'admin'
    var_24 = 'Software developer'
    var_25 = {var_13: var_19, var_14: var_20, var_15: var_21, var_16: var_22, var_17: var_23, var_18: var_24}
    var_26 = None
    var_27 = ''
    var_28 = 'invalid-email'
    var_29 = {var_13: var_27, var_14: var_28}
    var_30 = {var_13: var_27, var_14: var_28}



# Parsed testcases at query #16
#--------------------------


import typesystem.fields as module_0
import jinja2.loaders as module_1
import jinja2.environment as module_2

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = 'forms/input.html'
    var_2 = 'forms/textarea.html'
    var_3 = 'forms/select.html'
    var_4 = 'forms/checkbox.html'
    var_5 = '\n                <input type="{{ input_type }}" id="{{ field_id }}" name="{{ field_name }}" \n                       value="{{ value }}" {% if required %}required{% endif %}>\n            '
    var_6 = '\n                <textarea id="{{ field_id }}" name="{{ field_name }}" \n                          {% if required %}required{% endif %}>{{ value }}</textarea>\n            '
    var_7 = '\n                <select id="{{ field_id }}" name="{{ field_name }}" \n                        {% if required %}required{% endif %}>\n                    <option value="">Select</option>\n                </select>\n            '
    var_8 = '\n                <input type="checkbox" id="{{ field_id }}" name="{{ field_name }}" \n                       value="true" {% if value %}checked{% endif %}>\n            '
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = module_1.DictLoader(var_9)
    var_11 = module_2.Environment(loader=var_10)
    var_12 = 'description'
    var_13 = 'Test description'
    var_14 = 'email'
    var_15 = 'test@example.com'
    var_16 = 'password'
    var_17 = 'secret123'
    var_18 = 'hidden_field'
    var_19 = 'hidden_value'
    var_20 = 'number_field'
    var_21 = '42'
    var_22 = 'country'
    var_23 = 'us'
    var_24 = 'active'
    var_25 = True
    var_26 = False
    var_27 = 'invalid-email'
    var_28 = 'Invalid email format'
    var_29 = 'name'
    var_30 = 'John Doe'
    var_31 = 'test_field'
    var_32 = module_0.String()
    var_33 = 'test'
    var_34 = 'required_field'
    var_35 = module_0.String()
    var_36 = 'nullable_field'
    var_37 = module_0.String()



# Parsed testcases at query #17
#--------------------------


import typesystem.fields as module_0
import jinja2.loaders as module_1
import jinja2.environment as module_2

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Boolean()
    var_2 = 'forms/input.html'
    var_3 = 'forms/textarea.html'
    var_4 = 'forms/checkbox.html'
    var_5 = 'forms/select.html'
    var_6 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_7 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_8 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>'
    var_9 = '<select name="{{ field_name }}"></select>'
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_9}
    var_11 = module_1.DictLoader(var_10)
    var_12 = module_2.Environment(loader=var_11)
    var_13 = {}
    var_14 = 'name'
    var_15 = 'email'
    var_16 = 'age'
    var_17 = 'active'
    var_18 = 'John Doe'
    var_19 = 'john@example.com'
    var_20 = 30
    var_21 = True
    var_22 = {var_14: var_18, var_15: var_19, var_16: var_20, var_17: var_21}
    var_23 = {}
    var_24 = ''
    var_25 = 'invalid'
    var_26 = {var_14: var_24, var_15: var_25}
    var_27 = {var_3: var_7, var_2: var_6}
    var_28 = module_1.DictLoader(var_27)
    var_29 = module_2.Environment(loader=var_28)
    var_30 = {}
    var_31 = {var_5: var_9, var_2: var_6}
    var_32 = module_1.DictLoader(var_31)
    var_33 = module_2.Environment(loader=var_32)
    var_34 = {}



# Parsed testcases at query #18
#--------------------------


import typesystem.fields as module_0
import jinja2.loaders as module_1
import jinja2.environment as module_2

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Boolean()
    var_2 = 'forms/input.html'
    var_3 = 'forms/textarea.html'
    var_4 = 'forms/checkbox.html'
    var_5 = 'forms/select.html'
    var_6 = '\n                <input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">\n            '
    var_7 = '\n                <textarea name="{{ field_name }}">{{ value }}</textarea>\n            '
    var_8 = '\n                <input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>\n            '
    var_9 = '\n                <select name="{{ field_name }}">\n                    <option value="{{ value }}">{{ value }}</option>\n                </select>\n            '
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_9}
    var_11 = module_1.DictLoader(var_10)
    var_12 = module_2.Environment(loader=var_11)
    var_13 = 'name'
    var_14 = 'active'
    var_15 = 'John'
    var_16 = True
    var_17 = {var_13: var_15, var_14: var_16}
    var_18 = 'email'
    var_19 = 'invalid'
    var_20 = {var_18: var_19}



# Parsed testcases at query #19
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/textarea.html'
    var_2 = 'forms/checkbox.html'
    var_3 = 'forms/select.html'
    var_4 = '\n                <input type="{{ input_type }}" \n                       id="{{ field_id }}" \n                       name="{{ field_name }}" \n                       value="{{ value }}">\n            '
    var_5 = '\n                <textarea id="{{ field_id }}" \n                          name="{{ field_name }}">{{ value }}</textarea>\n            '
    var_6 = '\n                <input type="checkbox" \n                       id="{{ field_id }}" \n                       name="{{ field_name }}" \n                       {% if value %}checked{% endif %}>\n            '
    var_7 = '\n                <select id="{{ field_id }}" name="{{ field_name }}">\n                    <option value="{{ value }}">{{ value }}</option>\n                </select>\n            '
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = True
    var_11 = module_1.Environment(autoescape=var_10, loader=var_9)
    var_12 = 'name'
    var_13 = 'email'
    var_14 = 'John'
    var_15 = 'john@example.com'
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = ''
    var_18 = 'invalid'
    var_19 = {var_12: var_17, var_13: var_18}



# Parsed testcases at query #20
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/textarea.html'
    var_2 = 'forms/checkbox.html'
    var_3 = 'forms/select.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_5 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_6 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>'
    var_7 = '<select name="{{ field_name }}"></select>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = True
    var_11 = module_1.Environment(autoescape=var_10, loader=var_9)
    var_12 = 'name'
    var_13 = 'email'
    var_14 = 'John'
    var_15 = 'john@example.com'
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = "<script>alert('xss')</script>"
    var_18 = 'test@example.com'
    var_19 = {var_12: var_17, var_13: var_18}
    var_20 = {}
    var_21 = 'Alice'
    var_22 = 'alice@example.com'
    var_23 = {var_12: var_21, var_13: var_22}
    var_24 = module_2.Boolean()
    var_25 = 'active'
    var_26 = 'role'
    var_27 = 'bio'
    var_28 = 'admin'
    var_29 = 'Test bio'
    var_30 = {var_25: var_10, var_26: var_28, var_27: var_29}



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'field'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'error'
    var_4 = True
    var_5 = {var_0: var_1, var_3: var_4}
    var_6 = {}
    var_7 = {}
    var_8 = None



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = None
    var_4 = lambda x: var_3
    var_5 = {var_2: var_4}
    var_6 = 'MockSchema'
    var_7 = ()
    var_8 = 'fields'
    var_9 = 'serialize'
    var_10 = 'validate_or_error'
    var_11 = {}
    var_12 = lambda x: x
    var_13 = lambda x: (x, var_3)
    var_14 = {var_8: var_11, var_9: var_12, var_10: var_13}
    var_15 = 'MockField'
    var_16 = ()
    var_17 = {}
    var_18 = ()
    var_19 = 'format'
    var_20 = 'unknown'
    var_21 = {var_19: var_20}
    var_22 = 'MockField'
    var_23 = ()
    var_24 = 'format'
    var_25 = ()
    var_26 = ''
    var_27 = {var_19: var_26}
    var_28 = ()
    var_29 = {var_19: var_3}



# Parsed testcases at query #23
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/textarea.html'
    var_2 = 'forms/checkbox.html'
    var_3 = 'forms/select.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_5 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_6 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>'
    var_7 = '<select name="{{ field_name }}"></select>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'name'
    var_12 = 'email'
    var_13 = 'John'
    var_14 = 'john@example.com'
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = None
    var_17 = {}



# Parsed testcases at query #24
#--------------------------


import typesystem.fields as module_0
import jinja2.loaders as module_1
import jinja2.environment as module_2

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Boolean()
    var_2 = 'forms/input.html'
    var_3 = 'forms/textarea.html'
    var_4 = 'forms/checkbox.html'
    var_5 = 'forms/select.html'
    var_6 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_7 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_8 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>'
    var_9 = '<select name="{{ field_name }}"></select>'
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_9}
    var_11 = module_1.DictLoader(var_10)
    var_12 = module_2.Environment(loader=var_11)
    var_13 = 'name'
    var_14 = 'email'
    var_15 = 'John'
    var_16 = 'john@example.com'
    var_17 = {var_13: var_15, var_14: var_16}
    var_18 = ''
    var_19 = 'invalid'
    var_20 = {var_13: var_18, var_14: var_19}
    var_21 = {}
    var_22 = 'active'
    var_23 = True
    var_24 = {var_22: var_23}
    var_25 = 'read_only_field'
    var_26 = 'should not appear'
    var_27 = {var_25: var_26}



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = None
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_2, var_1: var_3}
    var_9 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = ''
    var_3 = -5
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = -5
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = 'This field is required'
    var_8 = 'Must be positive'
    var_9 = {var_0: var_7, var_1: var_8}
    var_10 = (var_6, var_9)
    var_11 = -5
    var_12 = {var_0: var_2, var_1: var_11}
    var_13 = -5
    var_14 = {var_0: var_2, var_1: var_13}

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = {}
    var_5 = {}
    var_6 = {}

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = {}

def test_case_0():
    var_0 = {}
    var_1 = {}

def test_case_0():
    var_0 = 'serialized'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = None
    var_5 = (var_3, var_4)
    var_6 = 'raw'
    var_7 = 'data'
    var_8 = {var_6: var_7}
    var_9 = {var_6: var_7}

def test_case_0():
    var_0 = 'serialized'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = None
    var_5 = (var_3, var_4)
    var_6 = 'raw'
    var_7 = 'data'
    var_8 = {var_6: var_7}
    var_9 = {var_6: var_7}



# Parsed testcases at query #26
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = module_0.String(format=var_0)
    var_2 = {}
    var_3 = 'email'
    var_4 = module_0.String(format=var_3)
    var_5 = module_0.String()
    var_6 = module_0.Boolean()
    var_7 = 'a'
    var_8 = 'A'
    var_9 = (var_7, var_8)
    var_10 = 'b'
    var_11 = 'B'
    var_12 = (var_10, var_11)
    var_13 = [var_9, var_12]
    var_14 = module_0.Choice(choices=var_13)
    var_15 = module_0.Field()
    var_16 = {}
    var_17 = module_0.Object(properties=var_16)



# Parsed testcases at query #27
#--------------------------


import typesystem.fields as module_0
import jinja2.loaders as module_1
import jinja2.environment as module_2

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = 'forms/input.html'
    var_2 = 'forms/textarea.html'
    var_3 = 'forms/checkbox.html'
    var_4 = 'forms/select.html'
    var_5 = '\n            <input type="{{ input_type }}" id="{{ field_id }}" name="{{ field_name }}" \n                   value="{{ value }}" {% if required %}required{% endif %}>\n            '
    var_6 = '\n            <textarea id="{{ field_id }}" name="{{ field_name }}" \n                      {% if required %}required{% endif %}>{{ value }}</textarea>\n            '
    var_7 = '\n            <input type="checkbox" id="{{ field_id }}" name="{{ field_name }}" \n                   value="true" {% if value %}checked{% endif %}>\n            '
    var_8 = '\n            <select id="{{ field_id }}" name="{{ field_name }}" \n                    {% if required %}required{% endif %}>\n                {% for option_value, option_label in field.choices %}\n                <option value="{{ option_value }}" \n                        {% if option_value == value %}selected{% endif %}>\n                    {{ option_label }}\n                </option>\n                {% endfor %}\n            </select>\n            '
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = module_1.DictLoader(var_9)
    var_11 = module_2.Environment(loader=var_10)
    var_12 = 'bio'
    var_13 = 'Test bio'
    var_14 = None
    var_15 = 'email'
    var_16 = 'test@example.com'
    var_17 = 'Invalid email'
    var_18 = 'password'
    var_19 = 'secret123'
    var_20 = 'active'
    var_21 = True
    var_22 = 'role'
    var_23 = 'admin'
    var_24 = 'name'
    var_25 = ''
    var_26 = 'This field is required'
    var_27 = 'first_name'
    var_28 = module_0.String()
    var_29 = 'John'
    var_30 = False
    var_31 = module_0.String()
    var_32 = 'required_field'
    var_33 = module_0.String()
    var_34 = 'optional_field'



# Parsed testcases at query #28
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/test/dir'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = var_2.loader
    var_4 = 'test_package'
    var_5 = module_0.Jinja2Forms(package=var_4)
    var_6 = var_5.env
    var_7 = var_6.loader
    var_8 = module_0.Jinja2Forms(directory=var_0, package=var_4)
    var_9 = var_8.env
    var_10 = var_9.loader
    var_11 = var_9.loader.loaders
    var_12 = len(var_11)
    assert var_12 == 2
    var_13 = 0
    var_14 = var_9.loader.loaders[var_13]
    var_15 = 1
    var_16 = var_9.loader.loaders[var_15]



# Parsed testcases at query #29
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/textarea.html'
    var_2 = 'forms/checkbox.html'
    var_3 = 'forms/select.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_5 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_6 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>'
    var_7 = '<select name="{{ field_name }}">{{ value }}</select>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = True
    var_11 = module_1.Environment(autoescape=var_10, loader=var_9)
    var_12 = 'name'
    var_13 = 'email'
    var_14 = 'John'
    var_15 = 'john@example.com'
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = 'Jane'
    var_18 = 'jane@example.com'
    var_19 = {var_12: var_17, var_13: var_18}
    var_20 = 'comment'
    var_21 = module_2.String()
    var_22 = {var_20: var_21}
    var_23 = module_3.Schema(var_22)
    var_24 = "<script>alert('xss')</script>"
    var_25 = {var_20: var_24}
    var_26 = module_4.Form(env=var_11, schema=var_23, values=var_25)
    var_27 = var_26.__html__()
    var_28 = str(var_27)
    var_29 = str(var_27)
    var_30 = None
    var_31 = 'Test'
    var_32 = 'test@example.com'
    var_33 = {var_12: var_31, var_13: var_32}
    var_34 = module_2.Boolean()
    var_35 = 'active'
    var_36 = 'category'
    var_37 = 'description'
    var_38 = 'a'
    var_39 = 'Some text'
    var_40 = {var_35: var_10, var_36: var_38, var_37: var_39}



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'field'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = None
    var_6 = {}
    var_7 = 'test'
    var_8 = 'data'
    var_9 = {var_7: var_8}
    var_10 = 'another'
    var_11 = 'attempt'
    var_12 = {var_10: var_11}
    var_13 = 'original'
    var_14 = {var_13: var_8}
    var_15 = 'initial'
    var_16 = {var_15: var_12}
    var_17 = 'new'
    var_18 = {var_17: var_8}



# Parsed testcases at query #31
#--------------------------


import typesystem.fields as module_0
import jinja2.loaders as module_1
import jinja2.environment as module_2

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Boolean()
    var_2 = 'forms/input.html'
    var_3 = 'forms/textarea.html'
    var_4 = 'forms/checkbox.html'
    var_5 = 'forms/select.html'
    var_6 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_7 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_8 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>'
    var_9 = '<select name="{{ field_name }}"><option value="{{ value }}">{{ value }}</option></select>'
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_9}
    var_11 = module_1.DictLoader(var_10)
    var_12 = module_2.Environment(loader=var_11)
    var_13 = 'name'
    var_14 = 'email'
    var_15 = 'age'
    var_16 = 'active'
    var_17 = 'role'
    var_18 = 'John'
    var_19 = 'john@example.com'
    var_20 = 30
    var_21 = True
    var_22 = 'admin'
    var_23 = {var_13: var_18, var_14: var_19, var_15: var_20, var_16: var_21, var_17: var_22}
    var_24 = ''
    var_25 = 'invalid'
    var_26 = {var_13: var_24, var_14: var_25}
    var_27 = module_0.String()



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.fields as module_0
import jinja2.loaders as module_1
import jinja2.environment as module_2

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = 'forms/input.html'
    var_2 = 'forms/textarea.html'
    var_3 = 'forms/checkbox.html'
    var_4 = 'forms/select.html'
    var_5 = '\n            <input type="{{ input_type }}" id="{{ field_id }}" name="{{ field_name }}" \n                   value="{{ value }}" {% if required %}required{% endif %}>\n            '
    var_6 = '\n            <textarea id="{{ field_id }}" name="{{ field_name }}" \n                      {% if required %}required{% endif %}>{{ value }}</textarea>\n            '
    var_7 = '\n            <input type="checkbox" id="{{ field_id }}" name="{{ field_name }}" \n                   value="true" {% if value %}checked{% endif %}>\n            '
    var_8 = '\n            <select id="{{ field_id }}" name="{{ field_name }}" \n                    {% if required %}required{% endif %}>\n                <option value="">Select...</option>\n                {% for choice_value, choice_label in field.choices %}\n                <option value="{{ choice_value }}" \n                        {% if choice_value == value %}selected{% endif %}>\n                    {{ choice_label }}\n                </option>\n                {% endfor %}\n            </select>\n            '
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = module_1.DictLoader(var_9)
    var_11 = module_2.Environment(loader=var_10)
    var_12 = 'name'
    var_13 = 'John'
    var_14 = 'email'
    var_15 = 'test@example.com'
    var_16 = 'password'
    var_17 = 'secret'
    var_18 = 'bio'
    var_19 = 'Some bio'
    var_20 = 'active'
    var_21 = True
    var_22 = False
    var_23 = 'status'
    var_24 = 'Invalid name'
    var_25 = None
    var_26 = ''
    var_27 = 'test_field'
    var_28 = module_0.String()
    var_29 = 'test'
    var_30 = module_0.String()
    var_31 = 'required_field'
    var_32 = module_0.String()
    var_33 = 'optional_field'
    var_34 = 'default'
    var_35 = module_0.String()
    var_36 = 'with_default'
    var_37 = module_0.String(allow_blank=var_21)
    var_38 = 'allow_blank'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'unknown_format'
    var_2 = 'color'
    var_3 = (var_2, var_2)
    var_4 = 'datetime'
    var_5 = 'datetime-local'
    var_6 = (var_4, var_5)
    var_7 = 'date'
    var_8 = (var_7, var_7)
    var_9 = 'email'
    var_10 = (var_9, var_9)
    var_11 = 'hidden'
    var_12 = (var_11, var_11)
    var_13 = 'month'
    var_14 = (var_13, var_13)
    var_15 = 'number'
    var_16 = (var_15, var_15)
    var_17 = 'password'
    var_18 = (var_17, var_17)
    var_19 = 'range'
    var_20 = (var_19, var_19)
    var_21 = 'search'
    var_22 = (var_21, var_21)
    var_23 = 'tel'
    var_24 = (var_23, var_23)
    var_25 = 'text'
    var_26 = (var_25, var_25)
    var_27 = 'time'
    var_28 = (var_27, var_27)
    var_29 = 'url'
    var_30 = (var_29, var_29)
    var_31 = 'week'
    var_32 = (var_31, var_31)
    var_33 = [var_3, var_6, var_8, var_10, var_12, var_14, var_16, var_18, var_20, var_22, var_24, var_26, var_28, var_30, var_32]
    var_34 = None
    var_35 = ''



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'MockSchema'
    var_1 = ()
    var_2 = 'fields'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'a'
    var_6 = 'A'
    var_7 = (var_5, var_6)
    var_8 = 'b'
    var_9 = 'B'
    var_10 = (var_8, var_9)
    var_11 = [var_7, var_10]
    var_12 = module_0.Choice(choices=var_11)
    var_13 = module_0.Boolean()
    var_14 = 'text'
    var_15 = module_0.String(format=var_14)
    var_16 = 'email'
    var_17 = module_0.String(format=var_16)
    var_18 = module_0.String()
    var_19 = module_0.Field()
    var_20 = {}
    var_21 = module_0.Object(properties=var_20)



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = 'text'
    var_3 = 'Description'
    var_4 = module_0.String(format=var_2)
    var_5 = 'description'
    var_6 = 'test value'
    var_7 = 'email'
    var_8 = 'Email'
    var_9 = module_0.String(format=var_7)
    var_10 = 'user_email'
    var_11 = 'test@example.com'
    var_12 = 'Agree'
    var_13 = module_0.Boolean()
    var_14 = 'agree_terms'
    var_15 = True
    var_16 = 'a'
    var_17 = 'Option A'
    var_18 = (var_16, var_17)
    var_19 = 'b'
    var_20 = 'Option B'
    var_21 = (var_19, var_20)
    var_22 = [var_18, var_21]
    var_23 = 'Selection'
    var_24 = module_0.Choice(choices=var_22)
    var_25 = 'selection'
    var_26 = 'username'
    var_27 = module_0.String()
    var_28 = ''
    var_29 = 'Required field'
    var_30 = 'password'
    var_31 = module_0.String(format=var_30)
    var_32 = 'secret'
    var_33 = False
    var_34 = module_0.String(allow_blank=var_33)
    var_35 = 'required_field'
    var_36 = module_0.String()
    var_37 = 'optional_field'
    var_38 = 'default'
    var_39 = module_0.String()
    var_40 = 'with_default'
    var_41 = module_0.String()
    var_42 = 'field_without_title'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'field'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'error'
    var_4 = True
    var_5 = {var_0: var_1, var_3: var_4}
    var_6 = 'test'
    var_7 = 'data'
    var_8 = {var_6: var_7}
    var_9 = 'test'
    var_10 = 'data'
    var_11 = {var_9: var_10}
    var_12 = None
    var_13 = 'initial'
    var_14 = {var_13: var_10}



# Parsed testcases at query #6
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/textarea.html'
    var_2 = 'forms/checkbox.html'
    var_3 = 'forms/select.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_5 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_6 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>'
    var_7 = '<select name="{{ field_name }}"></select>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = True
    var_11 = module_1.Environment(autoescape=var_10, loader=var_9)
    var_12 = 'name'
    var_13 = 'email'
    var_14 = 'John'
    var_15 = 'john@example.com'
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = {var_12: var_14, var_13: var_15}
    var_18 = 'test'
    var_19 = ''
    var_20 = 'invalid'
    var_21 = {var_12: var_19, var_13: var_20}



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'field3'
    var_3 = 'Field One'
    var_4 = True
    var_5 = 'test value'
    var_6 = 'default'
    var_7 = {var_0: var_5, var_2: var_6}
    var_8 = {var_0: var_5, var_2: var_6}
    var_9 = 'Field Two'
    var_10 = 'old'
    var_11 = 'old2'
    var_12 = {var_0: var_10, var_1: var_11}
    var_13 = 'invalid'
    var_14 = {var_0: var_13}
    var_15 = 'password_field'
    var_16 = 'text_field'
    var_17 = 'password'
    var_18 = 'text'
    var_19 = 'secret'
    var_20 = 'visible'
    var_21 = {var_15: var_19, var_16: var_20}
    var_22 = {var_15: var_19, var_16: var_20}
    var_23 = 'choice_field'
    var_24 = 'bool_field'
    var_25 = 'text_area'
    var_26 = {var_24: var_4}
    var_27 = {var_24: var_4}
    var_28 = {}



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = None
    var_4 = lambda x: var_3
    var_5 = {var_2: var_4}
    var_6 = 'MockSchema'
    var_7 = ()
    var_8 = 'fields'
    var_9 = 'serialize'
    var_10 = 'validate_or_error'
    var_11 = {}
    var_12 = lambda x: x
    var_13 = lambda x: (x, var_3)
    var_14 = {var_8: var_11, var_9: var_12, var_10: var_13}
    var_15 = 'MockField'
    var_16 = ()
    var_17 = {}
    var_18 = ()
    var_19 = 'format'
    var_20 = ''
    var_21 = {var_19: var_20}
    var_22 = ()
    var_23 = {var_19: var_3}
    var_24 = 'MockField'
    var_25 = ()
    var_26 = 'format'
    var_27 = ()
    var_28 = 'unknown'
    var_29 = {var_19: var_28}
    var_30 = ()
    var_31 = 'checkbox'
    var_32 = {var_19: var_31}



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = None
    var_4 = lambda x: var_3
    var_5 = {var_2: var_4}
    var_6 = 'MockSchema'
    var_7 = ()
    var_8 = 'fields'
    var_9 = 'serialize'
    var_10 = 'validate_or_error'
    var_11 = {}
    var_12 = lambda x: x
    var_13 = lambda x: (x, var_3)
    var_14 = {var_8: var_11, var_9: var_12, var_10: var_13}
    var_15 = 'MockField'
    var_16 = ()
    var_17 = {}
    var_18 = ()
    var_19 = 'format'
    var_20 = 'unknown'
    var_21 = {var_19: var_20}
    var_22 = 'MockField'
    var_23 = ()
    var_24 = 'format'
    var_25 = ()
    var_26 = {var_19: var_3}
    var_27 = ()
    var_28 = ''
    var_29 = {var_19: var_28}



# Parsed testcases at query #10
#--------------------------


import typesystem.fields as module_0
import jinja2.loaders as module_1
import jinja2.environment as module_2

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Boolean()
    var_2 = 'forms/input.html'
    var_3 = 'forms/textarea.html'
    var_4 = 'forms/checkbox.html'
    var_5 = 'forms/select.html'
    var_6 = '\n                <input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" \n                       value="{{ value }}" {% if required %}required{% endif %}>\n            '
    var_7 = '\n                <textarea name="{{ field_name }}" id="{{ field_id }}" \n                          {% if required %}required{% endif %}>{{ value }}</textarea>\n            '
    var_8 = '\n                <input type="checkbox" name="{{ field_name }}" id="{{ field_id }}" \n                       {% if value %}checked{% endif %} {% if required %}required{% endif %}>\n            '
    var_9 = '\n                <select name="{{ field_name }}" id="{{ field_id }}" \n                        {% if required %}required{% endif %}>\n                    <option value="a" {% if value == \'a\' %}selected{% endif %}>Option A</option>\n                    <option value="b" {% if value == \'b\' %}selected{% endif %}>Option B</option>\n                </select>\n            '
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_9}
    var_11 = module_1.DictLoader(var_10)
    var_12 = module_2.Environment(loader=var_11)
    var_13 = 'name'
    var_14 = 'email'
    var_15 = 'age'
    var_16 = 'active'
    var_17 = 'category'
    var_18 = 'John Doe'
    var_19 = 'john@example.com'
    var_20 = 30
    var_21 = True
    var_22 = 'a'
    var_23 = {var_13: var_18, var_14: var_19, var_15: var_20, var_16: var_21, var_17: var_22}
    var_24 = ''
    var_25 = 'invalid'
    var_26 = -5
    var_27 = {var_13: var_24, var_14: var_25, var_15: var_26}



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = module_0.String(format=var_0)
    var_2 = {}
    var_3 = 'email'
    var_4 = module_0.String(format=var_3)
    var_5 = module_0.String()
    var_6 = 'a'
    var_7 = 'A'
    var_8 = (var_6, var_7)
    var_9 = 'b'
    var_10 = 'B'
    var_11 = (var_9, var_10)
    var_12 = [var_8, var_11]
    var_13 = module_0.Choice(choices=var_12)
    var_14 = module_0.Boolean()
    var_15 = {}
    var_16 = module_0.Object(properties=var_15)



# Parsed testcases at query #12
#--------------------------


import typesystem.fields as module_0
import jinja2.loaders as module_1
import jinja2.environment as module_2

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Boolean()
    var_2 = 'forms/input.html'
    var_3 = 'forms/checkbox.html'
    var_4 = 'forms/textarea.html'
    var_5 = 'forms/select.html'
    var_6 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_7 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>'
    var_8 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_9 = '<select name="{{ field_name }}"></select>'
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_9}
    var_11 = module_1.DictLoader(var_10)
    var_12 = module_2.Environment(loader=var_11)
    var_13 = 'name'
    var_14 = 'age'
    var_15 = 'active'
    var_16 = 'John'
    var_17 = 25
    var_18 = True
    var_19 = {var_13: var_16, var_14: var_17, var_15: var_18}
    var_20 = None
    var_21 = 'Test'
    var_22 = {var_13: var_21}
    var_23 = ''
    var_24 = 'invalid'
    var_25 = False
    var_26 = {var_13: var_23, var_14: var_24, var_15: var_25}



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'username'
    var_1 = 'email'
    var_2 = 'Username'
    var_3 = False
    var_4 = 'Email'
    var_5 = True
    var_6 = 'john'
    var_7 = 'john@example.com'
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = {var_0: var_6, var_1: var_7}
    var_10 = 'invalid'
    var_11 = {var_0: var_10}
    var_12 = {var_0: var_10}
    var_13 = 'id'
    var_14 = 'test'
    var_15 = 123
    var_16 = {var_0: var_14, var_13: var_15}
    var_17 = {var_0: var_14}
    var_18 = None
    var_19 = 'password'
    var_20 = 'secret'
    var_21 = {var_19: var_20}
    var_22 = {var_19: var_20}



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'unknown_format'
    var_1 = 'color'
    var_2 = (var_1, var_1)
    var_3 = 'datetime'
    var_4 = 'datetime-local'
    var_5 = (var_3, var_4)
    var_6 = 'date'
    var_7 = (var_6, var_6)
    var_8 = 'email'
    var_9 = (var_8, var_8)
    var_10 = 'hidden'
    var_11 = (var_10, var_10)
    var_12 = 'month'
    var_13 = (var_12, var_12)
    var_14 = 'number'
    var_15 = (var_14, var_14)
    var_16 = 'password'
    var_17 = (var_16, var_16)
    var_18 = 'range'
    var_19 = (var_18, var_18)
    var_20 = 'search'
    var_21 = (var_20, var_20)
    var_22 = 'tel'
    var_23 = (var_22, var_22)
    var_24 = 'text'
    var_25 = (var_24, var_24)
    var_26 = 'time'
    var_27 = (var_26, var_26)
    var_28 = 'url'
    var_29 = (var_28, var_28)
    var_30 = 'week'
    var_31 = (var_30, var_30)
    var_32 = [var_2, var_5, var_7, var_9, var_11, var_13, var_15, var_17, var_19, var_21, var_23, var_25, var_27, var_29, var_31]
    var_33 = None
    var_34 = ''



# Parsed testcases at query #15
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = True
    var_5 = module_1.Environment(autoescape=var_4, loader=var_3)
    var_6 = 'name'
    var_7 = 'email'
    var_8 = 'John'
    var_9 = 'john@example.com'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'Jane'
    var_12 = 'jane@example.com'
    var_13 = {var_6: var_11, var_7: var_12}
    var_14 = {var_0: var_1}
    var_15 = module_0.DictLoader(var_14)
    var_16 = module_1.Environment(autoescape=var_4, loader=var_15)
    var_17 = "<script>alert('xss')</script>"
    var_18 = 'test@example.com'
    var_19 = {var_6: var_17, var_7: var_18}
    var_20 = 'Test'
    var_21 = {var_6: var_20, var_7: var_18}
    var_22 = '<div>'
    var_23 = '</div>'



# Parsed testcases at query #16
#--------------------------


import typesystem.fields as module_0
import jinja2.loaders as module_1
import jinja2.environment as module_2

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Boolean()
    var_2 = 'forms/input.html'
    var_3 = 'forms/checkbox.html'
    var_4 = 'forms/select.html'
    var_5 = 'forms/textarea.html'
    var_6 = '\n        <input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">\n        '
    var_7 = '\n        <input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>\n        '
    var_8 = '\n        <select name="{{ field_name }}">\n            <option value="admin" {% if value == "admin" %}selected{% endif %}>admin</option>\n            <option value="user" {% if value == "user" %}selected{% endif %}>user</option>\n        </select>\n        '
    var_9 = '\n        <textarea name="{{ field_name }}">{{ value }}</textarea>\n        '
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_9}
    var_11 = module_1.DictLoader(var_10)
    var_12 = module_2.Environment(loader=var_11)
    var_13 = 'name'
    var_14 = 'email'
    var_15 = 'age'
    var_16 = 'active'
    var_17 = 'role'
    var_18 = 'bio'
    var_19 = 'John Doe'
    var_20 = 'john@example.com'
    var_21 = 30
    var_22 = True
    var_23 = 'admin'
    var_24 = 'Software developer'
    var_25 = {var_13: var_19, var_14: var_20, var_15: var_21, var_16: var_22, var_17: var_23, var_18: var_24}
    var_26 = None



# Parsed testcases at query #17
#--------------------------


import typesystem.fields as module_0
import jinja2.loaders as module_1
import jinja2.environment as module_2

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Boolean()
    var_2 = 'forms/input.html'
    var_3 = 'forms/textarea.html'
    var_4 = 'forms/checkbox.html'
    var_5 = 'forms/select.html'
    var_6 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_7 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_8 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>'
    var_9 = '<select name="{{ field_name }}"><option value="{{ value }}">{{ value }}</option></select>'
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_9}
    var_11 = module_1.DictLoader(var_10)
    var_12 = module_2.Environment(loader=var_11)
    var_13 = 'name'
    var_14 = 'email'
    var_15 = 'age'
    var_16 = 'active'
    var_17 = 'role'
    var_18 = 'John Doe'
    var_19 = 'john@example.com'
    var_20 = 30
    var_21 = True
    var_22 = 'admin'
    var_23 = {var_13: var_18, var_14: var_19, var_15: var_20, var_16: var_21, var_17: var_22}
    var_24 = ''
    var_25 = 'invalid'
    var_26 = {var_13: var_24, var_14: var_25}



# Parsed testcases at query #18
#--------------------------


import typesystem.fields as module_0
import jinja2.loaders as module_1
import jinja2.environment as module_2

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = 'forms/input.html'
    var_2 = 'forms/textarea.html'
    var_3 = 'forms/checkbox.html'
    var_4 = 'forms/select.html'
    var_5 = '\n        <input type="{{ input_type }}" id="{{ field_id }}" name="{{ field_name }}" \n               value="{{ value }}" {% if required %}required{% endif %}>\n        '
    var_6 = '\n        <textarea id="{{ field_id }}" name="{{ field_name }}" \n                  {% if required %}required{% endif %}>{{ value }}</textarea>\n        '
    var_7 = '\n        <input type="checkbox" id="{{ field_id }}" name="{{ field_name }}" \n               value="true" {% if value %}checked{% endif %}>\n        '
    var_8 = '\n        <select id="{{ field_id }}" name="{{ field_name }}" \n                {% if required %}required{% endif %}>\n            <option value="">Select...</option>\n            {% for choice_value, choice_label in field.choices %}\n            <option value="{{ choice_value }}" \n                    {% if choice_value == value %}selected{% endif %}>\n                {{ choice_label }}\n            </option>\n            {% endfor %}\n        </select>\n        '
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = module_1.DictLoader(var_9)
    var_11 = module_2.Environment(loader=var_10)
    var_12 = 'name'
    var_13 = 'John'
    var_14 = 'email'
    var_15 = 'test@example.com'
    var_16 = 'password'
    var_17 = 'secret123'
    var_18 = 'bio'
    var_19 = 'My bio'
    var_20 = 'active'
    var_21 = True
    var_22 = 'role'
    var_23 = 'admin'
    var_24 = 'Invalid name'
    var_25 = 'test_field'
    var_26 = module_0.String()
    var_27 = 'test'
    var_28 = 'Custom Label'
    var_29 = module_0.String()
    var_30 = 'custom'
    var_31 = module_0.String()
    var_32 = 'untitled'
    var_33 = False
    var_34 = module_0.String()
    var_35 = 'required'
    var_36 = module_0.String()
    var_37 = 'optional'
    var_38 = module_0.String(allow_blank=var_21)
    var_39 = 'blank'
    var_40 = 'default'
    var_41 = module_0.String()
    var_42 = 'with_default'
    var_43 = 'unknown'
    var_44 = module_0.String(format=var_43)
    var_45 = 'test'



# Parsed testcases at query #19
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'email'
    var_2 = 'John'
    var_3 = 'john@example.com'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = ''
    var_6 = 'invalid'
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = None
    var_9 = 0
    var_10 = {}
    var_11 = module_0.String()



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'username'
    var_1 = 'email'
    var_2 = 'Username'
    var_3 = 'Email'
    var_4 = 'john'
    var_5 = 'john@example.com'
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = {var_0: var_4, var_1: var_5}
    var_8 = ''
    var_9 = 'invalid'
    var_10 = {var_0: var_8, var_1: var_9}
    var_11 = 'id'
    var_12 = 'created_at'
    var_13 = True
    var_14 = '2023-01-01'
    var_15 = {var_11: var_13, var_0: var_4, var_12: var_14}
    var_16 = {var_0: var_4}
    var_17 = {}
    var_18 = 'password'
    var_19 = 'secret'
    var_20 = {var_18: var_19}
    var_21 = {var_18: var_19}



# Parsed testcases at query #21
#--------------------------


import typesystem.fields as module_0
import jinja2.loaders as module_1
import jinja2.environment as module_2

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Boolean()
    var_2 = 'forms/input.html'
    var_3 = 'forms/textarea.html'
    var_4 = 'forms/checkbox.html'
    var_5 = 'forms/select.html'
    var_6 = '\n                <input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">\n            '
    var_7 = '\n                <textarea name="{{ field_name }}">{{ value }}</textarea>\n            '
    var_8 = '\n                <input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>\n            '
    var_9 = '\n                <select name="{{ field_name }}">\n                    <option value="{{ value }}">{{ value }}</option>\n                </select>\n            '
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_9}
    var_11 = module_1.DictLoader(var_10)
    var_12 = module_2.Environment(loader=var_11)
    var_13 = 'name'
    var_14 = 'email'
    var_15 = 'active'
    var_16 = 'John'
    var_17 = 'john@example.com'
    var_18 = True
    var_19 = {var_13: var_16, var_14: var_17, var_15: var_18}
    var_20 = 'invalid'
    var_21 = {var_14: var_20}



# Parsed testcases at query #22
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = None
    var_4 = lambda x: var_3
    var_5 = {var_2: var_4}
    var_6 = 'MockSchema'
    var_7 = ()
    var_8 = 'fields'
    var_9 = {}
    var_10 = {var_8: var_9}
    var_11 = 'a'
    var_12 = 'A'
    var_13 = (var_11, var_12)
    var_14 = 'b'
    var_15 = 'B'
    var_16 = (var_14, var_15)
    var_17 = [var_13, var_16]
    var_18 = module_0.Choice(choices=var_17)
    var_19 = module_0.Boolean()
    var_20 = 'text'
    var_21 = module_0.String(format=var_20)
    var_22 = 'email'
    var_23 = module_0.String(format=var_22)
    var_24 = module_0.String()
    var_25 = 'MockField'
    var_26 = ()
    var_27 = {}
    var_28 = {}
    var_29 = module_0.Object(properties=var_28)



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = ''
    var_7 = lambda self, ctx: var_6
    var_8 = {var_5: var_7}
    var_9 = 'MockSchema'
    var_10 = ()
    var_11 = 'fields'
    var_12 = {}
    var_13 = {var_11: var_12}
    var_14 = 'StringField'
    var_15 = ()
    var_16 = 'format'
    var_17 = 'text'
    var_18 = {var_16: var_17}
    var_19 = ()
    var_20 = 'email'
    var_21 = {var_16: var_20}
    var_22 = ()
    var_23 = {}
    var_24 = 'ChoiceField'
    var_25 = ()
    var_26 = {}
    var_27 = 'BooleanField'
    var_28 = ()
    var_29 = {}
    var_30 = 'ObjectField'
    var_31 = ()
    var_32 = {}
    var_33 = 'GenericField'
    var_34 = ()
    var_35 = {}



# Parsed testcases at query #24
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/textarea.html'
    var_2 = 'forms/checkbox.html'
    var_3 = 'forms/select.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}" {% if required %}required{% endif %}>'
    var_5 = '<textarea name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{{ value }}</textarea>'
    var_6 = '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}" {% if value %}checked{% endif %}>'
    var_7 = '<select name="{{ field_name }}" id="{{ field_id }}"><option value="option1">Option 1</option></select>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'name'
    var_12 = 'John'
    var_13 = 'email'
    var_14 = 'test@example.com'
    var_15 = 'password'
    var_16 = 'secret'
    var_17 = 'description'
    var_18 = 'Some text'
    var_19 = module_2.Boolean()
    var_20 = 'active'
    var_21 = True
    var_22 = 'color'
    var_23 = ''
    var_24 = 'This field is required'
    var_25 = 'birthday'
    var_26 = module_2.String()
    var_27 = 'first_name'
    var_28 = 'required_field'
    var_29 = 'optional_field'
    var_30 = 'field_with_title'
    var_31 = 'unknown'



# Parsed testcases at query #25
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = ''
    var_7 = lambda self, ctx: var_6
    var_8 = {var_5: var_7}
    var_9 = 'MockSchema'
    var_10 = ()
    var_11 = 'fields'
    var_12 = {}
    var_13 = {var_11: var_12}
    var_14 = 'a'
    var_15 = 'A'
    var_16 = (var_14, var_15)
    var_17 = 'b'
    var_18 = 'B'
    var_19 = (var_17, var_18)
    var_20 = [var_16, var_19]
    var_21 = module_0.Choice(choices=var_20)
    var_22 = module_0.Boolean()
    var_23 = 'text'
    var_24 = module_0.String(format=var_23)
    var_25 = 'email'
    var_26 = module_0.String(format=var_25)
    var_27 = module_0.Field()
    var_28 = {}
    var_29 = module_0.Object(properties=var_28)



# Parsed testcases at query #26
#--------------------------




# Parsed testcases at query #27
#--------------------------


import typesystem.fields as module_0
import jinja2.loaders as module_1
import jinja2.environment as module_2

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Boolean()
    var_2 = 'forms/input.html'
    var_3 = 'forms/textarea.html'
    var_4 = 'forms/checkbox.html'
    var_5 = 'forms/select.html'
    var_6 = '\n                <input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" \n                       value="{{ value }}" {% if required %}required{% endif %}>\n            '
    var_7 = '\n                <textarea name="{{ field_name }}" id="{{ field_id }}" \n                          {% if required %}required{% endif %}>{{ value }}</textarea>\n            '
    var_8 = '\n                <input type="checkbox" name="{{ field_name }}" id="{{ field_id }}" \n                       {% if value %}checked{% endif %} {% if required %}required{% endif %}>\n            '
    var_9 = '\n                <select name="{{ field_name }}" id="{{ field_id }}" \n                        {% if required %}required{% endif %}>\n                    {% for choice in field.choices %}\n                        <option value="{{ choice.value }}" \n                                {% if choice.value == value %}selected{% endif %}>\n                            {{ choice.text }}\n                        </option>\n                    {% endfor %}\n                </select>\n            '
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_9}
    var_11 = module_1.DictLoader(var_10)
    var_12 = module_2.Environment(loader=var_11)



# Parsed testcases at query #28
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/textarea.html'
    var_2 = 'forms/checkbox.html'
    var_3 = 'forms/select.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_5 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_6 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>'
    var_7 = '<select name="{{ field_name }}"><option value="{{ value }}">{{ value }}</option></select>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'name'
    var_12 = 'email'
    var_13 = 'John'
    var_14 = 'john@example.com'
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = 'Jane'
    var_17 = 'invalid-email'
    var_18 = {var_11: var_16, var_12: var_17}



# Parsed testcases at query #29
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/textarea.html'
    var_2 = 'forms/select.html'
    var_3 = 'forms/checkbox.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_5 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_6 = '<select name="{{ field_name }}"></select>'
    var_7 = '<input type="checkbox" name="{{ field_name }}">'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = True
    var_11 = module_1.Environment(autoescape=var_10, loader=var_9)
    var_12 = 'name'
    var_13 = 'email'
    var_14 = 'John'
    var_15 = 'john@example.com'
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = '<input type="text"'
    var_18 = 'value="John"'
    var_19 = None



# Parsed testcases at query #30
#--------------------------


import typesystem.fields as module_0
import jinja2.loaders as module_1
import jinja2.environment as module_2

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Boolean()
    var_2 = 'forms/input.html'
    var_3 = 'forms/checkbox.html'
    var_4 = 'forms/select.html'
    var_5 = 'forms/textarea.html'
    var_6 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_7 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>'
    var_8 = '<select name="{{ field_name }}"><option value="{{ value }}">{{ value }}</option></select>'
    var_9 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_9}
    var_11 = module_1.DictLoader(var_10)
    var_12 = module_2.Environment(loader=var_11)
    var_13 = 'name'
    var_14 = 'active'
    var_15 = 'John'
    var_16 = True
    var_17 = {var_13: var_15, var_14: var_16}
    var_18 = 'age'
    var_19 = 'status'
    var_20 = 'Jane'
    var_21 = 25
    var_22 = False
    var_23 = {var_13: var_20, var_18: var_21, var_14: var_22, var_19: var_14}
    var_24 = 'invalid'
    var_25 = {var_18: var_24}



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'unknown'
    var_2 = 'email'
    var_3 = ''
    var_4 = None



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = None
    var_4 = lambda x: var_3
    var_5 = {var_2: var_4}
    var_6 = 'MockSchema'
    var_7 = ()
    var_8 = 'fields'
    var_9 = 'serialize'
    var_10 = 'validate_or_error'
    var_11 = {}
    var_12 = lambda x: x
    var_13 = lambda x: (x, var_3)
    var_14 = {var_8: var_11, var_9: var_12, var_10: var_13}
    var_15 = 'MockField'
    var_16 = ()
    var_17 = {}
    var_18 = ()
    var_19 = 'format'
    var_20 = 'unknown'
    var_21 = {var_19: var_20}
    var_22 = 'MockField'
    var_23 = ()
    var_24 = 'format'
    var_25 = ()
    var_26 = ''
    var_27 = {var_19: var_26}
    var_28 = ()
    var_29 = {var_19: var_3}



# Parsed testcases at query #33
#--------------------------


import typesystem.fields as module_0
import jinja2.loaders as module_1
import jinja2.environment as module_2

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Boolean()
    var_2 = 'forms/input.html'
    var_3 = 'forms/textarea.html'
    var_4 = 'forms/checkbox.html'
    var_5 = 'forms/select.html'
    var_6 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_7 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_8 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>'
    var_9 = '<select name="{{ field_name }}"><option value="{{ value }}">{{ value }}</option></select>'
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_9}
    var_11 = module_1.DictLoader(var_10)
    var_12 = module_2.Environment(loader=var_11)
    var_13 = 'name'
    var_14 = 'email'
    var_15 = 'age'
    var_16 = 'active'
    var_17 = 'role'
    var_18 = 'John'
    var_19 = 'john@test.com'
    var_20 = 30
    var_21 = True
    var_22 = 'admin'
    var_23 = {var_13: var_18, var_14: var_19, var_15: var_20, var_16: var_21, var_17: var_22}



