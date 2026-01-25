####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'

def test_case_0():
    var_0 = 'Test work_in restores directory even when exception occurs.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in works with Path objects.'

def test_case_0():
    var_0 = 'Test nested work_in contexts.'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'
    var_1 = 'test_workdir'

def test_case_0():
    var_0 = 'Test work_in context manager with None argument.'

def test_case_0():
    var_0 = 'Test work_in context manager restores directory even on exception.'
    var_1 = 'test_workdir'
    var_2 = 'Test exception'
    var_3 = ValueError(var_2)

def test_case_0():
    var_0 = 'Test work_in context manager with Path object.'
    var_1 = 'test_workdir'

def test_case_0():
    var_0 = 'Test work_in context manager with string path.'
    var_1 = 'test_workdir'



# Parsed testcases at query #3
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter decorator creates a jinja2 extension.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'uppercase'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that multiple simple_filter extensions can coexist.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'lowercase'
    var_3 = var_1.filters[var_2]
    var_4 = 'HELLO'
    var_5 = 'reverse'
    var_6 = var_1.filters[var_5]
    var_7 = 'hello'



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'

def test_case_0():
    var_0 = 'Test work_in context manager with None argument.'

def test_case_0():
    var_0 = 'Test work_in restores directory even when exception occurs.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in context manager with Path object.'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'

def test_case_0():
    var_0 = 'Test work_in restores directory even when exception occurs.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in works with Path objects.'

def test_case_0():
    var_0 = 'Test work_in works with string paths.'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes directory and restores it.'

def test_case_0():
    var_0 = 'Test work_in with None argument stays in current directory.'

def test_case_0():
    var_0 = 'Test work_in restores directory even when exception occurs.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in works with Path objects.'



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'

def test_case_0():
    var_0 = 'Test work_in restores directory even when exception occurs.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in with nested context managers.'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'
    var_1 = 'test_workdir'

def test_case_0():
    var_0 = 'Test work_in context manager with None dirname.'

def test_case_0():
    var_0 = 'Test work_in restores directory even when exception occurs.'
    var_1 = 'test_workdir'
    var_2 = 'Test exception'
    var_3 = ValueError(var_2)

def test_case_0():
    var_0 = 'Test work_in context manager with Path object.'
    var_1 = 'test_workdir'

def test_case_0():
    var_0 = 'Test work_in context manager with string path.'
    var_1 = 'test_workdir'



# Parsed testcases at query #9
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter decorator creates a Jinja2 extension with filter.'
    var_1 = module_0.StrictEnvironment()
    var_2 = '{{ text|uppercase_filter }}'
    var_3 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that multiple simple_filter decorators create independent extensions.'
    var_1 = module_0.StrictEnvironment()
    var_2 = '{{ text|lowercase_filter|reverse_filter }}'
    var_3 = 'HELLO'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter with a filter function that takes arguments.'
    var_1 = module_0.StrictEnvironment()
    var_2 = '{{ text|repeat_filter(3) }}'
    var_3 = 'ab'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'

def test_case_0():
    var_0 = 'Test that work_in restores directory even when exception occurs.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in works with Path objects.'

def test_case_0():
    var_0 = 'Test work_in works with string paths.'



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'Test that simple_filter wraps a function in a jinja2 extension.'
    var_1 = '{{ text|uppercase_filter }}'
    var_2 = 'hello'

def test_case_0():
    var_0 = 'Test simple_filter with a different filter function.'
    var_1 = '{{ text|reverse_filter }}'
    var_2 = 'world'

def test_case_0():
    var_0 = 'Test simple_filter with a filter that takes multiple arguments.'
    var_1 = '{{ text|repeat_filter(3) }}'
    var_2 = 'x'



# Parsed testcases at query #12
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter correctly wraps a function in a Jinja2 extension.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'uppercase_filter'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter with a different filter function.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'reverse_filter'
    var_3 = var_1.filters[var_2]
    var_4 = 'abc'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that multiple simple_filter extensions can coexist.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'add_prefix'
    var_3 = var_1.filters[var_2]
    var_4 = 'test'
    var_5 = 'add_suffix'
    var_6 = var_1.filters[var_5]



# Parsed testcases at query #13
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter wraps a function in a jinja2 extension.'
    var_1 = module_0.StrictEnvironment()
    var_2 = '{{ text|my_custom_filter }}'
    var_3 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter with a different filter function.'
    var_1 = module_0.StrictEnvironment()
    var_2 = '{{ text|reverse_string }}'
    var_3 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter with a filter that takes multiple arguments.'
    var_1 = module_0.StrictEnvironment()
    var_2 = '{{ text|repeat_string(3) }}'
    var_3 = 'x'



# Parsed testcases at query #14
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter correctly wraps a function in a Jinja2 extension.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'uppercase_filter'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter with a different filter function.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'reverse_filter'
    var_3 = var_1.filters[var_2]
    var_4 = 'abc'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that multiple simple_filter extensions can coexist.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'add_prefix'
    var_3 = var_1.filters[var_2]
    var_4 = 'test'
    var_5 = 'add_suffix'
    var_6 = var_1.filters[var_5]



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'

def test_case_0():
    var_0 = 'Test work_in context manager with None argument.'

def test_case_0():
    var_0 = 'Test work_in context manager restores directory even on exception.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in context manager with Path object.'



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes directory and restores it.'

def test_case_0():
    var_0 = 'Test work_in context manager with None dirname.'

def test_case_0():
    var_0 = 'Test work_in context manager restores directory even on exception.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in context manager with Path object.'



# Parsed testcases at query #18
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter creates a Jinja2 extension with the filter registered.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'uppercase'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter with a different filter function.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'reverse_string'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that multiple filters can be created and used together.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'add_prefix'
    var_3 = var_1.filters[var_2]
    var_4 = 'test'
    var_5 = 'add_suffix'
    var_6 = var_1.filters[var_5]



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'

def test_case_0():
    var_0 = 'Test work_in with None argument stays in current directory.'

def test_case_0():
    var_0 = 'Test work_in restores directory even when exception is raised.'
    var_1 = 'test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in works with Path objects.'



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'

def test_case_0():
    var_0 = 'Test work_in with None argument stays in current directory.'

def test_case_0():
    var_0 = 'Test work_in restores directory even when exception occurs.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in works with Path objects.'

def test_case_0():
    var_0 = 'Test nested work_in context managers.'



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'Test simple_filter decorator wraps a function in a Jinja2 extension.'
    var_1 = '{{ text|uppercase }}'
    var_2 = 'hello'

def test_case_0():
    var_0 = 'Test that multiple simple_filter extensions can be used together.'
    var_1 = '{{ text|add_prefix|reverse_string }}'
    var_2 = 'test'

def test_case_0():
    var_0 = 'Test that simple_filter preserves the original function behavior.'
    var_1 = '{{ num|multiply_by_two }}'
    var_2 = 5



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'Test the work_in context manager.'

def test_case_0():
    var_0 = 'Test the work_in context manager with None argument.'

def test_case_0():
    var_0 = 'Test that work_in returns to original directory even on exception.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test the work_in context manager with string path.'



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'
    var_1 = 'test_subdir'

def test_case_0():
    var_0 = 'Test work_in context manager with None argument stays in current directory.'

def test_case_0():
    var_0 = 'Test work_in restores original directory even when exception is raised.'
    var_1 = 'test_subdir'
    var_2 = 'Test exception'
    var_3 = ValueError(var_2)

def test_case_0():
    var_0 = 'Test work_in context manager works with Path objects.'
    var_1 = 'test_subdir'

def test_case_0():
    var_0 = 'Test work_in context manager works with string paths.'
    var_1 = 'test_subdir'



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'

def test_case_0():
    var_0 = 'Test work_in context manager with None argument.'

def test_case_0():
    var_0 = 'Test work_in context manager restores directory even on exception.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in context manager with Path object.'

def test_case_0():
    var_0 = 'Test nested work_in context managers.'



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'

def test_case_0():
    var_0 = 'Test work_in context manager with None argument stays in current directory.'

def test_case_0():
    var_0 = 'Test work_in context manager restores directory even when exception occurs.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in context manager works with Path objects.'

def test_case_0():
    var_0 = 'Test work_in context manager works with string paths.'



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'

def test_case_0():
    var_0 = 'Test work_in context manager with None dirname stays in current directory.'

def test_case_0():
    var_0 = 'Test work_in context manager restores directory even when exception occurs.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in context manager works with Path objects.'



# Parsed testcases at query #29
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter creates a jinja2 extension with a filter.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'uppercase'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'
    var_5 = var_1.filters[var_2]
    var_6 = 'test'

def test_case_0():
    var_0 = 'Test that simple_filter works within a jinja2 template.'
    var_1 = "{{ 'hello' | reverse_string }}"

def test_case_0():
    var_0 = 'Test creating multiple simple_filter extensions.'
    var_1 = 'add_prefix'
    var_2 = 'test'
    var_3 = 'add_suffix'



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'

def test_case_0():
    var_0 = 'Test work_in context manager with None dirname.'

def test_case_0():
    var_0 = 'Test work_in restores directory even when exception occurs.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in context manager with Path object.'



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'

def test_case_0():
    var_0 = 'Test work_in restores directory even when exception occurs.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in works with string paths.'



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'

def test_case_0():
    var_0 = 'Test work_in context manager with Path object.'

def test_case_0():
    var_0 = 'Test that work_in restores directory even when exception occurs.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)



# Parsed testcases at query #33
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Convert value to uppercase.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'test_upper'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Convert value to uppercase.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'test_upper'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'

def test_case_0():
    var_0 = 'Convert value to lowercase.'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Reverse a string.'
    var_1 = -1
    var_2 = module_0.StrictEnvironment()
    var_3 = 'test_lower'
    var_4 = var_2.filters[var_3]
    var_5 = 'HELLO'
    var_6 = 'test_reverse'
    var_7 = var_2.filters[var_6]
    var_8 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Reverse a string.'
    var_1 = -1
    var_2 = module_0.StrictEnvironment()
    var_3 = 'test_lower'
    var_4 = var_2.filters[var_3]
    var_5 = 'HELLO'
    var_6 = 'test_reverse'
    var_7 = var_2.filters[var_6]
    var_8 = 'hello'



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'

def test_case_0():
    var_0 = 'Test work_in context manager with None dirname.'

def test_case_0():
    var_0 = 'Test work_in restores directory even when exception occurs.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in context manager with string path.'



# Parsed testcases at query #35
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter wraps a function in a jinja2 extension.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'uppercase'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that multiple filters can be created independently.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'lowercase'
    var_3 = var_1.filters[var_2]
    var_4 = 'HELLO'
    var_5 = 'reverse'
    var_6 = var_1.filters[var_5]
    var_7 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter with a more complex filter function.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'repeat'
    var_3 = var_1.filters[var_2]
    var_4 = 'ab'
    var_5 = 3



# Parsed testcases at query #36
#--------------------------


def test_case_0():
    var_0 = 'Test the work_in context manager.'

def test_case_0():
    var_0 = 'Test work_in context manager with None argument.'

def test_case_0():
    var_0 = 'Test that work_in restores directory even when exception occurs.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in context manager with string path.'



# Parsed testcases at query #37
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'

def test_case_0():
    var_0 = 'Test work_in restores directory even when exception occurs.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in works with Path objects.'

def test_case_0():
    var_0 = 'Test nested work_in context managers.'



# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)



# Parsed testcases at query #39
#--------------------------


def test_case_0():
    var_0 = 'Test the work_in context manager.'

def test_case_0():
    var_0 = 'Test the work_in context manager with None argument.'

def test_case_0():
    var_0 = 'Test that work_in returns to original directory even on exception.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test the work_in context manager with Path object.'



# Parsed testcases at query #40
#--------------------------


def test_case_0():
    var_0 = 'Test the work_in context manager.'
    var_1 = 'test_directory'

def test_case_0():
    var_0 = 'Test the work_in context manager with None argument.'

def test_case_0():
    var_0 = 'Test that work_in restores original directory even on exception.'
    var_1 = 'test_directory'
    var_2 = 'Test exception'
    var_3 = ValueError(var_2)

def test_case_0():
    var_0 = 'Test the work_in context manager with Path object.'
    var_1 = 'test_directory'

def test_case_0():
    var_0 = 'Test the work_in context manager with string path.'
    var_1 = 'test_directory'



# Parsed testcases at query #41
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'

def test_case_0():
    var_0 = 'Test work_in context manager with None argument stays in current directory.'

def test_case_0():
    var_0 = 'Test work_in context manager restores directory even when exception occurs.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in context manager works with Path objects.'



# Parsed testcases at query #42
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter decorator creates a Jinja2 extension with filter.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'uppercase_filter'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'
    var_5 = var_1.filters[var_2]
    var_6 = 'test'
    var_7 = module_0.StrictEnvironment()
    var_8 = 'reverse_filter'
    var_9 = var_7.filters[var_8]



# Parsed testcases at query #43
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter decorator creates a proper Jinja2 extension.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'uppercase'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter with a custom filter function.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'reverse_string'
    var_3 = var_1.filters[var_2]
    var_4 = 'cookiecutter'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that multiple simple_filter extensions can coexist.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'lowercase'
    var_3 = var_1.filters[var_2]
    var_4 = 'HELLO'
    var_5 = 'add_prefix'
    var_6 = var_1.filters[var_5]
    var_7 = 'test'



# Parsed testcases at query #44
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'
    var_1 = 'test_subdir'

def test_case_0():
    var_0 = 'Test work_in with None stays in current directory.'

def test_case_0():
    var_0 = 'Test work_in restores directory even when exception occurs.'
    var_1 = 'test_subdir'
    var_2 = 'Test exception'
    var_3 = ValueError(var_2)

def test_case_0():
    var_0 = 'Test work_in works with string paths.'
    var_1 = 'test_subdir'
    var_2 = True

def test_case_0():
    var_0 = 'Test work_in works with Path objects.'
    var_1 = 'test_subdir'



# Parsed testcases at query #45
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores directory.'

def test_case_0():
    var_0 = 'Test work_in restores directory even when exception occurs.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in works with Path objects.'

def test_case_0():
    var_0 = 'Test work_in works with string paths.'



# Parsed testcases at query #46
#--------------------------


def test_case_0():
    var_0 = 'Test simple_filter decorator creates a Jinja2 extension.'
    var_1 = 'uppercase'
    var_2 = 'hello'

def test_case_0():
    var_0 = 'Test multiple simple_filter decorators work independently.'
    var_1 = 'lowercase'
    var_2 = 'HELLO'
    var_3 = 'reverse'
    var_4 = 'hello'

def test_case_0():
    var_0 = 'Test simple_filter works within Jinja2 template rendering.'
    var_1 = '{{ text|double }}'
    var_2 = 'ab'



# Parsed testcases at query #47
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter wraps a function in a jinja2 extension.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'uppercase'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter works with different filter functions.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'reverse_string'
    var_3 = var_1.filters[var_2]
    var_4 = 'abc'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter with a more complex filter function.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'multiply_by_two'
    var_3 = var_1.filters[var_2]
    var_4 = 21



# Parsed testcases at query #48
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter wraps a function in a jinja2 extension.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'uppercase'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter with a different custom function.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'double'
    var_3 = var_1.filters[var_2]
    var_4 = 5
    var_5 = var_1.filters[var_2]
    var_6 = 21

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter extension initializes correctly with environment.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'reverse_string'
    var_3 = var_1.filters[var_2]
    var_4 = 'abc'



# Parsed testcases at query #49
#--------------------------


def test_case_0():
    var_0 = 'Convert value to uppercase.'
    var_1 = '{{ "hello" | test_upper }}'

def test_case_0():
    var_0 = 'Convert value to uppercase.'
    var_1 = '{{ "hello" | test_upper }}'

def test_case_0():
    var_0 = 'Test simple_filter with a different filter function.'
    var_1 = '{{ "hello" | reverse_string }}'

def test_case_0():
    var_0 = 'Test that multiple simple_filter extensions can coexist.'
    var_1 = '{{ "Hello" | to_upper }} {{ "World" | to_lower }}'



# Parsed testcases at query #50
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter creates a jinja2 extension from a function.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'uppercase'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that multiple simple_filter extensions can coexist.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'lowercase'
    var_3 = var_1.filters[var_2]
    var_4 = 'HELLO'
    var_5 = 'reverse_string'
    var_6 = var_1.filters[var_5]
    var_7 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter with a more complex filter function.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'custom_join'
    var_3 = var_1.filters[var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = ' | '



# Parsed testcases at query #51
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Convert value to uppercase.'
    var_1 = module_0.StrictEnvironment()
    var_2 = '{{ text|test_upper_filter }}'
    var_3 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Reverse a string.'
    var_1 = -1
    var_2 = module_0.StrictEnvironment()
    var_3 = '{{ text|test_reverse_filter }}'
    var_4 = 'cookiecutter'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Reverse a string.'
    var_1 = -1
    var_2 = module_0.StrictEnvironment()
    var_3 = '{{ text|test_reverse_filter }}'
    var_4 = 'cookiecutter'



# Parsed testcases at query #52
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter wraps a function in a jinja2 extension.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'uppercase'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'
    var_5 = var_1.filters[var_2]
    var_6 = 'test'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that multiple filters can be created independently.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'lowercase'
    var_3 = var_1.filters[var_2]
    var_4 = 'HELLO'
    var_5 = 'reverse'
    var_6 = var_1.filters[var_5]
    var_7 = 'hello'

def test_case_0():
    var_0 = 'Test that simple_filter works within jinja2 templates.'
    var_1 = '{{ name|add_prefix }}'
    var_2 = 'test'



# Parsed testcases at query #53
#--------------------------


def test_case_0():
    var_0 = 'Test that simple_filter decorator creates a jinja2 extension with filters.'
    var_1 = '{{ text|uppercase }}'
    var_2 = 'hello world'

def test_case_0():
    var_0 = 'Test that multiple simple_filter extensions can coexist.'
    var_1 = '{{ text|reverse_string|add_prefix }}'
    var_2 = 'hello'

def test_case_0():
    var_0 = 'Test that simple_filter works with filter functions that take arguments.'
    var_1 = '{{ text|repeat(3) }}'
    var_2 = 'x'



# Parsed testcases at query #54
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter decorator creates a proper Jinja2 extension.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'my_custom_filter'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'
    var_5 = var_1.filters[var_2]
    var_6 = 'test'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter with a different filter function.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'reverse_string'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that multiple simple_filter extensions can coexist.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'filter_one'
    var_3 = var_1.filters[var_2]
    var_4 = 'test'
    var_5 = 'filter_two'
    var_6 = var_1.filters[var_5]



# Parsed testcases at query #55
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter wraps a function in a jinja2 extension.'
    var_1 = module_0.StrictEnvironment()
    var_2 = '{{ name | uppercase }}'
    var_3 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter with a different filter function.'
    var_1 = module_0.StrictEnvironment()
    var_2 = '{{ text | reverse_string }}'
    var_3 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter returns a proper Extension subclass.'
    var_1 = module_0.StrictEnvironment()



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'

def test_case_0():
    var_0 = 'Test work_in restores directory even when exception occurs.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in works with Path objects.'



# Parsed testcases at query #2
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter wraps a function in a jinja2 extension.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'my_custom_filter'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter with a different filter function.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'reverse_string'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter preserves the original function behavior.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'multiply_by_two'
    var_3 = var_1.filters[var_2]
    var_4 = 5
    var_5 = var_1.filters[var_2]
    var_6 = 1
    var_7 = 2
    var_8 = [var_6, var_7]



# Parsed testcases at query #3
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter creates a Jinja2 extension with filter function.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'my_custom_filter'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'
    var_5 = var_1.filters[var_2]
    var_6 = 'test'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter with a different filter function.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'reverse_string'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'
    var_5 = var_1.filters[var_2]
    var_6 = 'abc'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that multiple simple_filter extensions can coexist.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'double'
    var_3 = var_1.filters[var_2]
    var_4 = 5
    var_5 = 'triple'
    var_6 = var_1.filters[var_5]



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores directory.'

def test_case_0():
    var_0 = 'Test work_in context manager with None argument.'

def test_case_0():
    var_0 = 'Test work_in restores directory even when exception occurs.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in context manager with Path object.'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores directory.'
    var_1 = 'test_workdir'

def test_case_0():
    var_0 = "Test work_in with None argument doesn't change directory."

def test_case_0():
    var_0 = 'Test work_in restores directory even when exception is raised.'
    var_1 = 'test_workdir'
    var_2 = 'Test exception'
    var_3 = ValueError(var_2)

def test_case_0():
    var_0 = 'Test work_in works with string path.'
    var_1 = 'test_workdir'

def test_case_0():
    var_0 = 'Test work_in works with Path object.'
    var_1 = 'test_workdir'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'

def test_case_0():
    var_0 = 'Test work_in with None argument keeps current directory.'

def test_case_0():
    var_0 = 'Test work_in restores directory even when exception is raised.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in accepts Path objects.'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'

def test_case_0():
    var_0 = 'Test work_in with None argument stays in current directory.'

def test_case_0():
    var_0 = 'Test work_in restores directory even when exception occurs.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in works with Path objects.'

def test_case_0():
    var_0 = 'Test nested work_in context managers.'



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)



# Parsed testcases at query #10
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter creates a valid Jinja2 extension.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'my_test_filter'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'
    var_5 = '{{ value|my_test_filter }}'
    var_6 = 'test'



# Parsed testcases at query #11
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter correctly wraps a function as a Jinja2 extension.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'uppercase_filter'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'
    var_5 = module_0.StrictEnvironment()
    var_6 = 'reverse_filter'
    var_7 = var_5.filters[var_6]
    var_8 = 'abc'



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'

def test_case_0():
    var_0 = 'Test work_in context manager with None argument stays in current directory.'

def test_case_0():
    var_0 = 'Test work_in restores directory even when exception is raised.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in works with Path objects.'

def test_case_0():
    var_0 = 'Test nested work_in context managers.'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'

def test_case_0():
    var_0 = 'Test work_in context manager with Path object.'

def test_case_0():
    var_0 = 'Test work_in restores directory even when exception occurs.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)



# Parsed testcases at query #14
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter wraps a function in a jinja2 extension.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'uppercase'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that multiple filters can be created independently.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'lowercase'
    var_3 = var_1.filters[var_2]
    var_4 = 'HELLO'
    var_5 = 'reverse_string'
    var_6 = var_1.filters[var_5]
    var_7 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter works with functions that take parameters.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'repeat'
    var_3 = var_1.filters[var_2]
    var_4 = 'x'
    var_5 = 3
    var_6 = var_1.filters[var_2]
    var_7 = 'ab'
    var_8 = 2



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes directory and restores it.'
    var_1 = 'test_subdir'

def test_case_0():
    var_0 = 'Test work_in context manager with None dirname stays in current directory.'

def test_case_0():
    var_0 = 'Test work_in restores original directory even when exception occurs.'
    var_1 = 'test_subdir'
    var_2 = 'Test exception'
    var_3 = ValueError(var_2)

def test_case_0():
    var_0 = 'Test work_in context manager with Path object.'
    var_1 = 'test_subdir'

def test_case_0():
    var_0 = 'Test work_in context manager with string path.'
    var_1 = 'test_subdir'



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'
    var_1 = 'test_subdir'

def test_case_0():
    var_0 = "Test work_in with None argument doesn't change directory."

def test_case_0():
    var_0 = 'Test work_in restores directory even when exception is raised.'
    var_1 = 'test_subdir'
    var_2 = 'Test exception'
    var_3 = ValueError(var_2)

def test_case_0():
    var_0 = 'Test work_in works with string path argument.'
    var_1 = 'test_subdir'

def test_case_0():
    var_0 = 'Test work_in works with Path object argument.'
    var_1 = 'test_subdir'



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'Test the work_in context manager.'

def test_case_0():
    var_0 = 'Test the work_in context manager with None (no directory change).'

def test_case_0():
    var_0 = 'Test that work_in restores original directory even when exception occurs.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test the work_in context manager with Path object.'



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes directory and restores it.'

def test_case_0():
    var_0 = 'Test work_in with None argument stays in current directory.'

def test_case_0():
    var_0 = 'Test work_in restores directory even when exception occurs.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in works with Path objects.'



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'Test the work_in context manager.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'
    var_1 = True

def test_case_0():
    var_0 = 'Test work_in context manager with None argument stays in current directory.'

def test_case_0():
    var_0 = 'Test work_in context manager restores directory even when exception occurs.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)
    var_3 = True

def test_case_0():
    var_0 = 'Test work_in context manager works with Path objects.'
    var_1 = True



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'
    var_1 = 'test_subdir'

def test_case_0():
    var_0 = 'Test work_in with None argument stays in current directory.'

def test_case_0():
    var_0 = 'Test work_in restores directory even when exception is raised.'
    var_1 = 'test_subdir'
    var_2 = 'Test exception'
    var_3 = ValueError(var_2)

def test_case_0():
    var_0 = 'Test work_in works with string path argument.'
    var_1 = 'test_subdir'

def test_case_0():
    var_0 = 'Test work_in works with Path object argument.'
    var_1 = 'test_subdir'



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'

def test_case_0():
    var_0 = 'Test work_in context manager with None argument stays in current directory.'

def test_case_0():
    var_0 = 'Test work_in context manager restores directory even when exception occurs.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in context manager works with Path objects.'



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in works with Path objects.'

def test_case_0():
    var_0 = 'Test work_in works with string paths.'



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'

def test_case_0():
    var_0 = 'Test work_in context manager with None argument.'

def test_case_0():
    var_0 = 'Test work_in restores directory even when exception occurs.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in context manager with Path object.'



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores directory.'
    var_1 = 'test_subdir'

def test_case_0():
    var_0 = 'Test work_in with None argument stays in current directory.'

def test_case_0():
    var_0 = 'Test work_in restores directory even when exception occurs.'
    var_1 = 'test_subdir'
    var_2 = 'Test exception'
    var_3 = ValueError(var_2)

def test_case_0():
    var_0 = 'Test work_in works with Path objects.'
    var_1 = 'test_subdir'

def test_case_0():
    var_0 = 'Test work_in works with string paths.'
    var_1 = 'test_subdir'



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'

def test_case_0():
    var_0 = 'Test work_in context manager with None argument.'

def test_case_0():
    var_0 = 'Test work_in restores directory even when exception occurs.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in context manager with string path.'



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'
    var_1 = 'test_subdir'

def test_case_0():
    var_0 = 'Test work_in context manager with None argument.'

def test_case_0():
    var_0 = 'Test work_in restores directory even when exception occurs.'
    var_1 = 'test_subdir'
    var_2 = 'Test exception'
    var_3 = ValueError(var_2)

def test_case_0():
    var_0 = 'Test work_in works with Path objects.'
    var_1 = 'test_subdir'

def test_case_0():
    var_0 = 'Test work_in works with string paths.'
    var_1 = 'test_subdir'
    var_2 = True



# Parsed testcases at query #29
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter wraps a function in a jinja2 extension.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'uppercase_filter'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that multiple simple_filter decorators create independent extensions.'
    var_1 = module_0.StrictEnvironment()
    var_2 = module_0.StrictEnvironment()
    var_3 = 'lowercase_filter'
    var_4 = var_1.filters[var_3]
    var_5 = 'HELLO'
    var_6 = 'reverse_filter'
    var_7 = var_2.filters[var_6]
    var_8 = 'hello'



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'

def test_case_0():
    var_0 = 'Test that work_in restores directory even when exception occurs.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in works with string paths.'

def test_case_0():
    var_0 = 'Test nested work_in context managers.'



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 'Test the work_in context manager.'

def test_case_0():
    var_0 = 'Test the work_in context manager with Path object.'

def test_case_0():
    var_0 = 'Test that work_in restores directory even when exception occurs.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)



# Parsed testcases at query #32
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter creates a jinja2 extension with filter.'
    var_1 = module_0.StrictEnvironment()
    var_2 = '{{ value | my_custom_filter }}'
    var_3 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that multiple simple_filter decorators create independent extensions.'
    var_1 = module_0.StrictEnvironment()
    var_2 = '{{ value | filter_one | filter_two }}'
    var_3 = 'test'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter with a more complex filter function.'
    var_1 = module_0.StrictEnvironment()
    var_2 = '{{ text | reverse_string }}'
    var_3 = 'cookiecutter'



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores directory.'

def test_case_0():
    var_0 = 'Test work_in context manager with Path object.'

def test_case_0():
    var_0 = 'Test work_in restores directory even when exception occurs.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores directory.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = 'Test that simple_filter wraps a function in a Jinja2 extension.'
    var_1 = '{{ text | uppercase }}'
    var_2 = 'hello'

def test_case_0():
    var_0 = 'Test that multiple filters can be created independently.'
    var_1 = '{{ text | lowercase | reverse }}'
    var_2 = 'HELLO'

def test_case_0():
    var_0 = 'Test that simple_filter works with filter functions that take arguments.'
    var_1 = '{{ text | repeat(3) }}'
    var_2 = 'ab'



# Parsed testcases at query #36
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter creates a jinja2 extension with filter.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'my_custom_filter'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'
    var_5 = var_1.filters[var_2]
    var_6 = 'test'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that multiple simple filters can be created independently.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'filter_one'
    var_3 = var_1.filters[var_2]
    var_4 = 'HELLO'
    var_5 = 'filter_two'
    var_6 = var_1.filters[var_5]
    var_7 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter with a more complex filter function.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'word_count'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello world test'
    var_5 = var_1.filters[var_2]
    var_6 = 'single'
    var_7 = var_1.filters[var_2]
    var_8 = ''



# Parsed testcases at query #37
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter creates a jinja2 extension from a filter function.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'uppercase'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that multiple simple_filter decorators work independently.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'lowercase'
    var_3 = var_1.filters[var_2]
    var_4 = 'HELLO'
    var_5 = 'reverse'
    var_6 = var_1.filters[var_5]
    var_7 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter works with filters that take arguments.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'repeat'
    var_3 = var_1.filters[var_2]
    var_4 = 'x'
    var_5 = var_1.filters[var_2]
    var_6 = 3



# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'

def test_case_0():
    var_0 = "Test work_in with None argument doesn't change directory."

def test_case_0():
    var_0 = 'Test work_in restores directory even if exception is raised.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in works with Path objects.'



# Parsed testcases at query #39
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores directory.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)



# Parsed testcases at query #40
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter decorator creates a jinja2 extension with filter.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'uppercase_filter'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter with a different filter function.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'reverse_filter'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = module_0.StrictEnvironment()



# Parsed testcases at query #41
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter decorator creates a jinja2 extension.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'my_custom_filter'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter with a different filter function.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'reverse_string'
    var_3 = var_1.filters[var_2]
    var_4 = 'abc'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test creating multiple simple_filter extensions.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'filter_one'
    var_3 = var_1.filters[var_2]
    var_4 = 'test'
    var_5 = 'filter_two'
    var_6 = var_1.filters[var_5]



# Parsed testcases at query #42
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter decorator creates a jinja2 extension.'
    var_1 = module_0.StrictEnvironment()
    var_2 = "{{ 'hello' | my_test_filter }}"

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter with a different filter function.'
    var_1 = module_0.StrictEnvironment()
    var_2 = "{{ 'test' | reverse_string }}"

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter preserves the original filter function.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'multiply_by_two'
    var_3 = var_1.filters[var_2]
    var_4 = 5
    var_5 = 3



# Parsed testcases at query #43
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'

def test_case_0():
    var_0 = 'Test work_in context manager with None argument.'

def test_case_0():
    var_0 = 'Test work_in restores directory even when exception is raised.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in context manager with Path object.'

def test_case_0():
    var_0 = 'Test work_in context manager with default None parameter.'



# Parsed testcases at query #44
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes directory and restores it.'

def test_case_0():
    var_0 = 'Test work_in context manager with None argument.'

def test_case_0():
    var_0 = 'Test work_in restores directory even when exception occurs.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in context manager with Path object.'

def test_case_0():
    var_0 = 'Test nested work_in context managers.'



# Parsed testcases at query #45
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'

def test_case_0():
    var_0 = 'Test that work_in restores directory even when exception occurs.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test that work_in works with Path objects.'

def test_case_0():
    var_0 = 'Test that work_in works with string paths.'



# Parsed testcases at query #46
#--------------------------


def test_case_0():
    var_0 = 'Test the work_in context manager.'

def test_case_0():
    var_0 = 'Test the work_in context manager with None argument.'

def test_case_0():
    var_0 = 'Test that work_in returns to original directory even on exception.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test the work_in context manager with Path object.'



# Parsed testcases at query #47
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'

def test_case_0():
    var_0 = 'Test work_in context manager with None argument.'

def test_case_0():
    var_0 = 'Test work_in context manager restores directory even on exception.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in context manager with Path object.'



# Parsed testcases at query #48
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'
    var_1 = 'test_directory'

def test_case_0():
    var_0 = "Test work_in with None argument doesn't change directory."

def test_case_0():
    var_0 = 'Test work_in restores original directory even when exception occurs.'
    var_1 = 'test_directory'
    var_2 = 'Test exception'
    var_3 = ValueError(var_2)

def test_case_0():
    var_0 = 'Test work_in works with string path argument.'
    var_1 = 'test_directory'

def test_case_0():
    var_0 = 'Test work_in works with Path object argument.'
    var_1 = 'test_directory'



# Parsed testcases at query #49
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'

def test_case_0():
    var_0 = 'Test work_in restores directory even when exception is raised.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in works with string paths.'



# Parsed testcases at query #50
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores working directory.'

def test_case_0():
    var_0 = 'Test work_in context manager with None argument.'

def test_case_0():
    var_0 = 'Test work_in context manager restores directory even when exception occurs.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in context manager with Path object.'



# Parsed testcases at query #51
#--------------------------


def test_case_0():
    var_0 = 'Test work_in context manager changes and restores directory.'

def test_case_0():
    var_0 = 'Test work_in restores directory even when exception occurs.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in works with Path objects.'

def test_case_0():
    var_0 = 'Test nested work_in context managers.'



# Parsed testcases at query #52
#--------------------------


def test_case_0():
    var_0 = 'Test the work_in context manager.'

def test_case_0():
    var_0 = 'Test work_in context manager with None argument.'

def test_case_0():
    var_0 = 'Test that work_in restores directory even when exception occurs.'
    var_1 = 'Test exception'
    var_2 = ValueError(var_1)

def test_case_0():
    var_0 = 'Test work_in with Path object.'



# Parsed testcases at query #53
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter decorator creates a Jinja2 extension with filter.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'uppercase_filter'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter works with multiple different filters.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'lowercase_filter'
    var_3 = var_1.filters[var_2]
    var_4 = 'HELLO'
    var_5 = 'reverse_filter'
    var_6 = var_1.filters[var_5]
    var_7 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter works with functions that take multiple arguments.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'repeat_filter'
    var_3 = var_1.filters[var_2]
    var_4 = 'a'
    var_5 = var_1.filters[var_2]
    var_6 = 3



# Parsed testcases at query #54
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter creates a jinja2 extension with filter.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'my_test_filter'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'
    var_5 = var_1.filters[var_2]
    var_6 = 'world'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter with a different filter function.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'reverse_string'
    var_3 = var_1.filters[var_2]
    var_4 = 'abc'
    var_5 = var_1.filters[var_2]
    var_6 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that multiple simple_filter extensions can coexist.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'filter1'
    var_3 = var_1.filters[var_2]
    var_4 = 'Hello'
    var_5 = 'filter2'
    var_6 = var_1.filters[var_5]



# Parsed testcases at query #55
#--------------------------


def test_case_0():
    var_0 = 'Test that work_in context manager changes and restores directory.'
    var_1 = 'test_subdir'

def test_case_0():
    var_0 = "Test that work_in with None doesn't change directory."

def test_case_0():
    var_0 = 'Test that work_in restores directory even when exception is raised.'
    var_1 = 'test_subdir'
    var_2 = 'Test exception'
    var_3 = ValueError(var_2)

def test_case_0():
    var_0 = 'Test that work_in works with Path objects.'
    var_1 = 'test_subdir'

def test_case_0():
    var_0 = 'Test that work_in works with string paths.'
    var_1 = 'test_subdir'



# Parsed testcases at query #56
#--------------------------


def test_case_0():
    var_0 = 'Test simple_filter decorator creates a Jinja2 extension with filter.'
    var_1 = '{{ text | my_custom_filter }}'
    var_2 = 'hello'

def test_case_0():
    var_0 = 'Test simple_filter with different filter functions.'
    var_1 = '{{ text | uppercase_filter }}'
    var_2 = 'hello'

def test_case_0():
    var_0 = 'Test registering multiple simple_filter extensions.'
    var_1 = '{{ text | filter_one | filter_two }}'
    var_2 = 'test'



# Parsed testcases at query #57
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter wraps a function in a jinja2 extension.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'uppercase'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'
    var_5 = var_1.filters[var_2]
    var_6 = 'test123'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that multiple simple_filter extensions can coexist.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'lowercase'
    var_3 = var_1.filters[var_2]
    var_4 = 'HELLO'
    var_5 = 'reverse'
    var_6 = var_1.filters[var_5]
    var_7 = 'hello'



# Parsed testcases at query #58
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter wraps a function in a jinja2 extension.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'uppercase'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that multiple simple_filter extensions can coexist.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'lowercase'
    var_3 = var_1.filters[var_2]
    var_4 = 'HELLO'
    var_5 = 'reverse'
    var_6 = var_1.filters[var_5]
    var_7 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter preserves the original function behavior.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'add_prefix'
    var_3 = var_1.filters[var_2]
    var_4 = 'test'
    var_5 = var_1.filters[var_2]
    var_6 = 'CUSTOM_'



# Parsed testcases at query #59
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Convert value to uppercase.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'test_upper'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Convert value to uppercase.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'test_upper'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter with a different filter function.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'reverse_string'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'

import cookiecutter.utils as module_0
import cookiecutter.environment as module_1

def test_case_0():
    var_0 = 'Test simple_filter with a lambda function.'
    var_1 = 2
    var_2 = lambda x: x * var_1
    var_3 = module_0.simple_filter(var_2)
    var_4 = module_1.StrictEnvironment()
    var_5 = 'double'
    var_6 = var_4.filters[var_5]
    var_7 = 5



# Parsed testcases at query #60
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter wraps a function in a jinja2 extension.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'my_custom_filter'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter with a different filter function.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'reverse_string'
    var_3 = var_1.filters[var_2]
    var_4 = 'abc'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that multiple simple_filter extensions can coexist.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'add_prefix'
    var_3 = var_1.filters[var_2]
    var_4 = 'test'
    var_5 = 'add_suffix'
    var_6 = var_1.filters[var_5]



# Parsed testcases at query #61
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter correctly wraps a function as a Jinja2 extension.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'uppercase_filter'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'
    var_5 = var_1.filters[var_2]
    var_6 = 'test'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that multiple simple_filter extensions can coexist.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'reverse_filter'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'
    var_5 = 'double_filter'
    var_6 = var_1.filters[var_5]
    var_7 = 5

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter preserves the original function behavior.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'custom_filter'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'
    var_5 = var_1.filters[var_2]
    var_6 = 'test'
    var_7 = '?'



# Parsed testcases at query #62
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test filter function that doubles a value.'
    var_1 = 2
    var_2 = module_0.StrictEnvironment()
    var_3 = 'test_func'
    var_4 = var_2.filters[var_3]
    var_5 = 5
    var_6 = var_2.filters[var_3]
    var_7 = 'ab'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test filter function that doubles a value.'
    var_1 = 2
    var_2 = module_0.StrictEnvironment()
    var_3 = 'test_func'
    var_4 = var_2.filters[var_3]
    var_5 = 5
    var_6 = var_2.filters[var_3]
    var_7 = 'ab'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that multiple simple_filter decorators create independent extensions.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'filter_one'
    var_3 = var_1.filters[var_2]
    var_4 = 5
    var_5 = 'filter_two'
    var_6 = var_1.filters[var_5]



# Parsed testcases at query #63
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = module_0.StrictEnvironment()
    var_1 = 'test_uppercase'
    var_2 = var_0.filters[var_1]
    var_3 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter with a different filter function.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'reverse_string'
    var_3 = var_1.filters[var_2]
    var_4 = 'cookiecutter'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test creating multiple simple_filter extensions.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'add_prefix'
    var_3 = var_1.filters[var_2]
    var_4 = 'test'
    var_5 = 'add_suffix'
    var_6 = var_1.filters[var_5]



# Parsed testcases at query #64
#--------------------------


def test_case_0():
    var_0 = 'Test that simple_filter wraps a function in a jinja2 extension.'
    var_1 = 'uppercase'
    var_2 = 'hello'

def test_case_0():
    var_0 = 'Test simple_filter with a different filter function.'
    var_1 = 'reverse_string'
    var_2 = 'hello'



# Parsed testcases at query #65
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter wraps a function in a jinja2 extension.'
    var_1 = module_0.StrictEnvironment()
    var_2 = '{{ text|my_custom_filter }}'
    var_3 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that multiple simple_filter extensions can coexist.'
    var_1 = module_0.StrictEnvironment()
    var_2 = '{{ text|filter_one|filter_two }}'
    var_3 = 'test'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter with a numeric operation.'
    var_1 = module_0.StrictEnvironment()
    var_2 = '{{ num|double }}'
    var_3 = 5



# Parsed testcases at query #66
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter wraps a function in a jinja2 extension.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'uppercase_filter'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that multiple simple_filter extensions can coexist.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'lowercase_filter'
    var_3 = var_1.filters[var_2]
    var_4 = 'HELLO'
    var_5 = 'reverse_filter'
    var_6 = var_1.filters[var_5]
    var_7 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that simple_filter works with filter functions that take arguments.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'replace_filter'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello world'
    var_5 = 'world'
    var_6 = 'there'



# Parsed testcases at query #67
#--------------------------


def test_case_0():
    var_0 = 'Convert value to uppercase.'
    var_1 = 'test_uppercase'
    var_2 = 'hello'
    var_3 = 'world'

def test_case_0():
    var_0 = 'Convert value to uppercase.'
    var_1 = 'test_uppercase'
    var_2 = 'hello'
    var_3 = 'world'

def test_case_0():
    var_0 = 'Test that multiple simple_filter extensions can be created.'
    var_1 = 'lowercase_filter'
    var_2 = 'HELLO'
    var_3 = 'reverse_filter'
    var_4 = 'hello'

def test_case_0():
    var_0 = 'Test that simple_filter works within actual jinja2 templates.'
    var_1 = '{{ name | add_prefix }}'
    var_2 = 'test'



# Parsed testcases at query #68
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test the simple_filter decorator function.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'my_test_filter'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'
    var_5 = var_1.filters[var_2]
    var_6 = 'test'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that multiple simple_filter decorators create independent filters.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'filter_one'
    var_3 = var_1.filters[var_2]
    var_4 = 'test'
    var_5 = 'filter_two'
    var_6 = var_1.filters[var_5]

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter with a more complex filter function.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'reverse_string'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'
    var_5 = var_1.filters[var_2]
    var_6 = 'world'



# Parsed testcases at query #69
#--------------------------


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter decorator creates a proper jinja2 extension.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'uppercase'
    var_3 = var_1.filters[var_2]
    var_4 = 'hello'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test simple_filter with a more complex filter function.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'reverse_string'
    var_3 = var_1.filters[var_2]
    var_4 = 'abc'

import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that multiple simple_filter extensions can coexist.'
    var_1 = module_0.StrictEnvironment()
    var_2 = 'add_prefix'
    var_3 = var_1.filters[var_2]
    var_4 = 'test'
    var_5 = 'add_suffix'
    var_6 = var_1.filters[var_5]



