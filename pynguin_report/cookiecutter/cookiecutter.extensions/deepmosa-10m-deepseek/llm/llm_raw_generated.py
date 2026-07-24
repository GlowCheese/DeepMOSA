####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_slugify_extension_constructor_adds_filter():
    from jinja2 import Environment
    from jinja2.ext import Extension
    import re
    from typing import Iterable
    DEFAULT_SEPARATOR = '-'
    def pyslugify(value, entities, decimal, hexadecimal, max_length, word_boundary, separator, save_order, stopwords, regex_pattern, lowercase, replacements, allow_unicode):
        return 'mocked-slug'
    class SlugifyExtension(Extension):
        def __init__(self, environment):
            super().__init__(environment)
            def slugify(value, entities=True, decimal=True, hexadecimal=True, max_length=0, word_boundary=False, separator=DEFAULT_SEPARATOR, save_order=False, stopwords=(), regex_pattern=None, lowercase=True, replacements=(), allow_unicode=False):
                return pyslugify(value, entities, decimal, hexadecimal, max_length, word_boundary, separator, save_order, stopwords, regex_pattern, lowercase, replacements, allow_unicode)
            environment.filters['slugify'] = slugify
    env = Environment()
    extension = SlugifyExtension(env)
    result = env.filters['slugify']('Test String')
    assert result == 'mocked-slug'
    assert 'slugify' in env.filters


# LLM-generated content at query #2
#--------------------------

def test_uuid_extension_constructor_adds_uuid4_to_globals():
    from jinja2 import Environment
    import uuid
    from module_under_test import UUIDExtension
    env = Environment()
    extension = UUIDExtension(env)
    assert 'uuid4' in env.globals
    assert callable(env.globals['uuid4'])
    generated_uuid = env.globals['uuid4']()
    try:
        uuid_obj = uuid.UUID(generated_uuid, version=4)
        assert str(uuid_obj) == generated_uuid
    except ValueError:
        assert False, "Generated string is not a valid UUID4"


# LLM-generated content at query #3
#--------------------------

def test_constructor_initializes_environment_with_default_format():
    mock_environment = Mock()
    extension = TimeExtension(mock_environment)
    assert mock_environment.datetime_format == '%Y-%m-%d'


# LLM-generated content at query #4
#--------------------------

def test_random_string_extension_constructor_adds_function_to_globals():
    from jinja2 import Environment
    from string import ascii_letters, punctuation
    environment = Environment()
    extension = RandomStringExtension(environment)
    assert 'random_ascii_string' in environment.globals
    func = environment.globals['random_ascii_string']
    result_no_punct = func(10, False)
    assert len(result_no_punct) == 10
    assert all(c in ascii_letters for c in result_no_punct)
    result_with_punct = func(15, True)
    assert len(result_with_punct) == 15
    assert all(c in ascii_letters + punctuation for c in result_with_punct)


# LLM-generated content at query #5
#--------------------------

def test_jsonify_extension_constructor_adds_filter():
    environment = Environment()
    extension = JsonifyExtension(environment)
    assert 'jsonify' in environment.filters
    assert callable(environment.filters['jsonify'])


# LLM-generated content at query #6
#--------------------------

def test_environment_extend_datetime_format_not_called():
    mock_environment = Mock()
    extension = TimeExtension(mock_environment)
    assert mock_environment.extend.call_count == 0


# LLM-generated content at query #7
#--------------------------

def test_constructor_extends_environment_with_default_datetime_format():
    mock_environment = Mock()
    extension = TimeExtension(mock_environment)
    mock_environment.extend.assert_called_once_with(datetime_format='%Y-%m-%d')


# LLM-generated content at query #8
#--------------------------

def test_environment_extended_with_datetime_format():
    mock_environment = Mock()
    extension = TimeExtension(mock_environment)
    mock_environment.extend.assert_called_once_with(datetime_format='%Y-%m-%d')


# LLM-generated content at query #9
#--------------------------

def test_constructor_extends_environment_with_default_datetime_format():
    mock_environment = Mock()
    extension = TimeExtension(mock_environment)
    mock_environment.extend.assert_called_once_with(datetime_format='%Y-%m-%d')


# LLM-generated content at query #10
#--------------------------

def test_jsonify_extension_constructor_adds_filter():
    environment = Environment()
    extension = JsonifyExtension(environment)
    assert 'jsonify' in environment.filters
    assert callable(environment.filters['jsonify'])


# LLM-generated content at query #11
#--------------------------

def test_environment_extended_with_datetime_format():
    mock_environment = Mock()
    extension = TimeExtension(mock_environment)
    mock_environment.extend.assert_called_once_with(datetime_format='%Y-%m-%d')


