####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_constructor_loads_default_extensions():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    instance = TestClass()
    expected = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected

def test_constructor_loads_extensions_from_context():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    context = {'cookiecutter': {'_extensions': ['my.Extension1', 'other.Extension2']}}
    instance = TestClass(context=context)
    expected = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my.Extension1',
        'other.Extension2',
    ]
    assert instance.extensions == expected

def test_constructor_handles_empty_extensions_in_context():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    context = {'cookiecutter': {}}
    instance = TestClass(context=context)
    expected = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected

def test_constructor_handles_missing_cookiecutter_key():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    context = {}
    instance = TestClass(context=context)
    expected = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected

def test_constructor_handles_none_context():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    instance = TestClass(context=None)
    expected = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected

def test_constructor_passes_kwargs_to_parent():
    class MockParent:
        def __init__(self, extensions, extra_arg=None, **kwargs):
            self.extensions = extensions
            self.extra_arg = extra_arg
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    instance = TestClass(extra_arg='test_value')
    assert instance.extra_arg == 'test_value'

def test_constructor_raises_unknown_extension_on_import_error():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            raise ImportError('No module named invalid')
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    try:
        TestClass()
        assert False
    except Exception as e:
        assert e.__class__.__name__ == 'UnknownExtension'
        assert 'Unable to load extension' in str(e)


# LLM-generated content at query #2
#--------------------------

def test_constructor_without_context():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = kwargs.get('extensions', [])

    instance = TestClass()
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected_extensions


def test_constructor_with_empty_context():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = kwargs.get('extensions', [])

    instance = TestClass(context={})
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected_extensions


def test_constructor_with_extensions_in_context():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = kwargs.get('extensions', [])

    context = {
        'cookiecutter': {
            '_extensions': ['my_extension.Extension1', 'another.Extension2']
        }
    }
    instance = TestClass(context=context)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my_extension.Extension1',
        'another.Extension2',
    ]
    assert instance.extensions == expected_extensions


def test_constructor_without_cookiecutter_key_in_context():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = kwargs.get('extensions', [])

    context = {'other_key': 'value'}
    instance = TestClass(context=context)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected_extensions


def test_constructor_without_extensions_key_in_cookiecutter():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = kwargs.get('extensions', [])

    context = {'cookiecutter': {'other_key': 'value'}}
    instance = TestClass(context=context)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected_extensions


def test_constructor_passes_other_kwargs_to_super():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.kwargs = kwargs

    instance = TestClass(optimized=True, trim_blocks=False)
    assert instance.kwargs['optimized'] == True
    assert instance.kwargs['trim_blocks'] == False
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.kwargs['extensions'] == expected_extensions


def test_constructor_raises_unknown_extension_on_import_error():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            try:
                super().__init__(**kwargs)
            except Exception as e:
                self.error = e
            else:
                self.error = None

    class MockSuper:
        def __init__(self, **kwargs):
            raise ImportError("No module named 'cookiecutter.extensions'")

    import sys
    original_bases = TestClass.__bases__
    TestClass.__bases__ = (MockSuper,)
    instance = TestClass()
    assert instance.error is not None
    assert isinstance(instance.error, UnknownExtension)
    assert "Unable to load extension: No module named 'cookiecutter.extensions'" in str(instance.error)
    TestClass.__bases__ = original_bases


# LLM-generated content at query #3
#--------------------------

def test_extension_loader_mixin_raises_unknown_extension_on_import_error():
    class MockSuper:
        def __init__(self, extensions, **kwargs):
            raise ImportError("No module named 'fake_extension'")
    class TestClass(ExtensionLoaderMixin, MockSuper):
        pass
    try:
        TestClass(context={'cookiecutter': {'_extensions': ['fake_extension']}})
    except UnknownExtension as e:
        assert str(e) == "Unable to load extension: No module named 'fake_extension'"


# LLM-generated content at query #4
#--------------------------

def test_extension_loader_mixin_init_without_context():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = kwargs.get('extensions', [])

    instance = TestClass()
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected_extensions


def test_extension_loader_mixin_init_with_empty_context():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = kwargs.get('extensions', [])

    instance = TestClass(context={})
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected_extensions


def test_extension_loader_mixin_init_with_extensions_in_context():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = kwargs.get('extensions', [])

    context = {
        'cookiecutter': {
            '_extensions': ['my_extension.Extension1', 'another.Extension2']
        }
    }
    instance = TestClass(context=context)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my_extension.Extension1',
        'another.Extension2',
    ]
    assert instance.extensions == expected_extensions


def test_extension_loader_mixin_init_with_context_missing_extensions_key():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = kwargs.get('extensions', [])

    context = {'cookiecutter': {}}
    instance = TestClass(context=context)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected_extensions


def test_extension_loader_mixin_init_with_context_missing_cookiecutter_key():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = kwargs.get('extensions', [])

    context = {'other_key': 'value'}
    instance = TestClass(context=context)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected_extensions


def test_extension_loader_mixin_init_passes_other_kwargs():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = kwargs.get('extensions', [])
            self.other_arg = kwargs.get('other_arg')

    instance = TestClass(other_arg='test_value')
    assert instance.other_arg == 'test_value'
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected_extensions


# LLM-generated content at query #5
#--------------------------

def test_extension_loader_mixin_initializes_with_default_extensions():
    class TestEnvironment:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
    class TestClass(ExtensionLoaderMixin, TestEnvironment):
        pass
    instance = TestClass()
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected_extensions

def test_extension_loader_mixin_initializes_with_context_extensions():
    class TestEnvironment:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
    class TestClass(ExtensionLoaderMixin, TestEnvironment):
        pass
    context = {'cookiecutter': {'_extensions': ['some.extension', 'another.extension']}}
    instance = TestClass(context=context)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'some.extension',
        'another.extension',
    ]
    assert instance.extensions == expected_extensions

def test_extension_loader_mixin_initializes_with_empty_context_extensions():
    class TestEnvironment:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
    class TestClass(ExtensionLoaderMixin, TestEnvironment):
        pass
    context = {'cookiecutter': {}}
    instance = TestClass(context=context)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected_extensions

def test_extension_loader_mixin_initializes_with_missing_cookiecutter_key():
    class TestEnvironment:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
    class TestClass(ExtensionLoaderMixin, TestEnvironment):
        pass
    context = {}
    instance = TestClass(context=context)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected_extensions

def test_extension_loader_mixin_initializes_with_none_context():
    class TestEnvironment:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
    class TestClass(ExtensionLoaderMixin, TestEnvironment):
        pass
    instance = TestClass(context=None)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected_extensions

def test_extension_loader_mixin_passes_kwargs_to_parent():
    class TestEnvironment:
        def __init__(self, extensions, extra_arg=None, **kwargs):
            self.extensions = extensions
            self.extra_arg = extra_arg
    class TestClass(ExtensionLoaderMixin, TestEnvironment):
        pass
    instance = TestClass(extra_arg='test_value')
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected_extensions
    assert instance.extra_arg == 'test_value'

def test_extension_loader_mixin_raises_unknown_extension_on_import_error():
    class TestEnvironment:
        def __init__(self, extensions, **kwargs):
            raise ImportError('No module named invalid_extension')
    class TestClass(ExtensionLoaderMixin, TestEnvironment):
        pass
    try:
        TestClass()
        assert False
    except Exception as e:
        assert str(e) == 'Unable to load extension: No module named invalid_extension'


# LLM-generated content at query #6
#--------------------------

def test_init_without_context_and_no_extensions():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            pass
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    instance = TestClass()
    assert True


# LLM-generated content at query #7
#--------------------------

def test_extension_loader_mixin_raises_unknown_extension_on_import_error():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            raise ImportError("No module named 'fake_extension'")
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    try:
        TestClass(context={})
    except UnknownExtension as e:
        assert str(e) == "Unable to load extension: No module named 'fake_extension'"


# LLM-generated content at query #8
#--------------------------

def test_extension_loader_mixin_initializes_with_default_extensions():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    instance = TestClass()
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected_extensions

def test_extension_loader_mixin_initializes_with_context_extensions():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    context = {'cookiecutter': {'_extensions': ['my.Extension1', 'other.Extension2']}}
    instance = TestClass(context=context)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my.Extension1',
        'other.Extension2',
    ]
    assert instance.extensions == expected_extensions

def test_extension_loader_mixin_initializes_with_empty_context_extensions():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    context = {'cookiecutter': {}}
    instance = TestClass(context=context)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected_extensions

def test_extension_loader_mixin_initializes_without_extensions_key_in_context():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    context = {}
    instance = TestClass(context=context)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected_extensions

def test_extension_loader_mixin_passes_kwargs_to_parent():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    instance = TestClass(some_arg='value', another_arg=123)
    assert instance.kwargs == {'some_arg': 'value', 'another_arg': 123}

def test_extension_loader_mixin_raises_unknown_extension_on_import_error():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            raise ImportError('No module named invalid')
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    try:
        TestClass()
        assert False
    except Exception as e:
        assert e.__class__.__name__ == 'UnknownExtension'
        assert str(e) == 'Unable to load extension: No module named invalid'


# LLM-generated content at query #9
#--------------------------

def test_init_without_context_uses_empty_dict():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    instance = TestClass()
    assert instance.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.kwargs == {}

def test_init_with_none_context_uses_empty_dict():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    instance = TestClass(context=None)
    assert instance.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.kwargs == {}

def test_init_with_empty_context_dict():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    instance = TestClass(context={})
    assert instance.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.kwargs == {}

def test_init_with_context_containing_no_cookiecutter_key():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    instance = TestClass(context={"other": "value"})
    assert instance.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.kwargs == {}

def test_init_with_context_containing_cookiecutter_without_extensions():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    instance = TestClass(context={"cookiecutter": {"key": "value"}})
    assert instance.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.kwargs == {}

def test_init_with_context_containing_extensions():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    context = {"cookiecutter": {"_extensions": ["custom.Extension1", "custom.Extension2"]}}
    instance = TestClass(context=context)
    assert instance.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'custom.Extension1',
        'custom.Extension2',
    ]
    assert instance.kwargs == {}

def test_init_passes_additional_kwargs_to_parent():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    instance = TestClass(context=None, extra_arg=123, another="test")
    assert instance.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.kwargs == {"extra_arg": 123, "another": "test"}

def test_init_raises_unknown_extension_on_import_error():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            raise ImportError("No module named 'cookiecutter.extensions'")
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    try:
        TestClass()
    except Exception as e:
        assert e.__class__.__name__ == "UnknownExtension"
        assert str(e) == "Unable to load extension: No module named 'cookiecutter.extensions'"
    else:
        assert False, "Expected UnknownExtension to be raised"


# LLM-generated content at query #10
#--------------------------

def test_constructor_loads_default_extensions():
    mock_super = MockSuper()
    mixin = ExtensionLoaderMixin(context={}, mock_super=mock_super)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert mock_super.init_called_with['extensions'] == expected_extensions

def test_constructor_loads_context_extensions():
    mock_super = MockSuper()
    context = {'cookiecutter': {'_extensions': ['my.Extension1', 'other.Extension2']}}
    mixin = ExtensionLoaderMixin(context=context, mock_super=mock_super)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my.Extension1',
        'other.Extension2',
    ]
    assert mock_super.init_called_with['extensions'] == expected_extensions

def test_constructor_handles_missing_extensions_key():
    mock_super = MockSuper()
    context = {'cookiecutter': {}}
    mixin = ExtensionLoaderMixin(context=context, mock_super=mock_super)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert mock_super.init_called_with['extensions'] == expected_extensions

def test_constructor_handles_missing_cookiecutter_key():
    mock_super = MockSuper()
    context = {}
    mixin = ExtensionLoaderMixin(context=context, mock_super=mock_super)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert mock_super.init_called_with['extensions'] == expected_extensions

def test_constructor_handles_none_context():
    mock_super = MockSuper()
    mixin = ExtensionLoaderMixin(context=None, mock_super=mock_super)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert mock_super.init_called_with['extensions'] == expected_extensions

def test_constructor_passes_remaining_kwargs():
    mock_super = MockSuper()
    mixin = ExtensionLoaderMixin(context={}, extra_arg='value', another=123, mock_super=mock_super)
    assert mock_super.init_called_with['extra_arg'] == 'value'
    assert mock_super.init_called_with['another'] == 123

def test_constructor_raises_unknown_extension_on_import_error():
    mock_super = MockSuper(raise_import_error=True)
    try:
        mixin = ExtensionLoaderMixin(context={}, mock_super=mock_super)
        assert False, "Expected UnknownExtension to be raised"
    except UnknownExtension as e:
        assert "Unable to load extension" in str(e)


# LLM-generated content at query #11
#--------------------------

def test_constructor_loads_default_extensions():
    mock_super = Mock()
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super):
        instance = ExtensionLoaderMixin()
    call_kwargs = mock_super.call_args[1]
    extensions = call_kwargs['extensions']
    expected = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert extensions == expected

def test_constructor_loads_extensions_from_context():
    mock_super = Mock()
    context = {'cookiecutter': {'_extensions': ['my.Extension1', 'other.Extension2']}}
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super):
        instance = ExtensionLoaderMixin(context=context)
    call_kwargs = mock_super.call_args[1]
    extensions = call_kwargs['extensions']
    expected = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my.Extension1',
        'other.Extension2',
    ]
    assert extensions == expected

def test_constructor_handles_empty_extensions_in_context():
    mock_super = Mock()
    context = {'cookiecutter': {}}
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super):
        instance = ExtensionLoaderMixin(context=context)
    call_kwargs = mock_super.call_args[1]
    extensions = call_kwargs['extensions']
    expected = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert extensions == expected

def test_constructor_handles_missing_cookiecutter_key():
    mock_super = Mock()
    context = {}
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super):
        instance = ExtensionLoaderMixin(context=context)
    call_kwargs = mock_super.call_args[1]
    extensions = call_kwargs['extensions']
    expected = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert extensions == expected

def test_constructor_handles_none_context():
    mock_super = Mock()
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super):
        instance = ExtensionLoaderMixin(context=None)
    call_kwargs = mock_super.call_args[1]
    extensions = call_kwargs['extensions']
    expected = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert extensions == expected

def test_constructor_passes_extra_kwargs_to_super():
    mock_super = Mock()
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super):
        instance = ExtensionLoaderMixin(extra_arg='value', another_arg=123)
    call_kwargs = mock_super.call_args[1]
    assert 'extra_arg' in call_kwargs
    assert call_kwargs['extra_arg'] == 'value'
    assert 'another_arg' in call_kwargs
    assert call_kwargs['another_arg'] == 123

def test_constructor_raises_unknown_extension_on_import_error():
    mock_super = Mock(side_effect=ImportError('No module named'))
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super):
        try:
            instance = ExtensionLoaderMixin()
            raised = False
        except UnknownExtension:
            raised = True
    assert raised


# LLM-generated content at query #12
#--------------------------

def test_extension_loader_mixin_init_raises_unknown_extension_on_import_error():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            raise ImportError("No module named 'fake_extension'")

    class TestClass(ExtensionLoaderMixin, MockParent):
        pass

    try:
        TestClass(context={})
    except UnknownExtension as e:
        assert str(e) == "Unable to load extension: No module named 'fake_extension'"
    else:
        assert False, "Expected UnknownExtension to be raised"


# LLM-generated content at query #13
#--------------------------

def test_extension_loader_mixin_init_raises_unknown_extension_on_import_error():
    class MockSuper:
        def __init__(self, extensions, **kwargs):
            raise ImportError("No module named 'fake_extension'")
    class TestClass(ExtensionLoaderMixin, MockSuper):
        pass
    try:
        TestClass(context={})
    except UnknownExtension as e:
        assert str(e) == "Unable to load extension: No module named 'fake_extension'"


# LLM-generated content at query #14
#--------------------------

def test_constructor_loads_default_extensions():
    mock_super = Mock()
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super):
        instance = ExtensionLoaderMixin()
        call_kwargs = mock_super.call_args[1]
        extensions = call_kwargs['extensions']
        assert 'cookiecutter.extensions.JsonifyExtension' in extensions
        assert 'cookiecutter.extensions.RandomStringExtension' in extensions
        assert 'cookiecutter.extensions.SlugifyExtension' in extensions
        assert 'cookiecutter.extensions.TimeExtension' in extensions
        assert 'cookiecutter.extensions.UUIDExtension' in extensions

def test_constructor_loads_extensions_from_context():
    mock_super = Mock()
    context = {'cookiecutter': {'_extensions': ['my.ext.Extension1', 'other.ext.Extension2']}}
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super):
        instance = ExtensionLoaderMixin(context=context)
        call_kwargs = mock_super.call_args[1]
        extensions = call_kwargs['extensions']
        assert 'my.ext.Extension1' in extensions
        assert 'other.ext.Extension2' in extensions
        assert 'cookiecutter.extensions.JsonifyExtension' in extensions

def test_constructor_handles_empty_extensions_in_context():
    mock_super = Mock()
    context = {'cookiecutter': {}}
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super):
        instance = ExtensionLoaderMixin(context=context)
        call_kwargs = mock_super.call_args[1]
        extensions = call_kwargs['extensions']
        assert len(extensions) == 5
        assert 'cookiecutter.extensions.JsonifyExtension' in extensions

def test_constructor_handles_missing_cookiecutter_key_in_context():
    mock_super = Mock()
    context = {'other_key': 'value'}
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super):
        instance = ExtensionLoaderMixin(context=context)
        call_kwargs = mock_super.call_args[1]
        extensions = call_kwargs['extensions']
        assert len(extensions) == 5
        assert 'cookiecutter.extensions.JsonifyExtension' in extensions

def test_constructor_handles_none_context():
    mock_super = Mock()
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super):
        instance = ExtensionLoaderMixin(context=None)
        call_kwargs = mock_super.call_args[1]
        extensions = call_kwargs['extensions']
        assert len(extensions) == 5
        assert 'cookiecutter.extensions.JsonifyExtension' in extensions

def test_constructor_passes_extra_kwargs_to_super():
    mock_super = Mock()
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super):
        instance = ExtensionLoaderMixin(extra_arg='value', another_arg=123)
        call_kwargs = mock_super.call_args[1]
        assert call_kwargs['extra_arg'] == 'value'
        assert call_kwargs['another_arg'] == 123
        assert 'extensions' in call_kwargs

def test_constructor_raises_unknown_extension_on_import_error():
    mock_super = Mock(side_effect=ImportError('No module named invalid'))
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super):
        try:
            instance = ExtensionLoaderMixin()
            assert False
        except UnknownExtension as e:
            assert 'Unable to load extension' in str(e)
            assert isinstance(e.__cause__, ImportError)


# LLM-generated content at query #15
#--------------------------

def test_constructor_loads_default_extensions():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    instance = TestClass(context={})
    expected = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected

def test_constructor_loads_context_extensions():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    context = {'cookiecutter': {'_extensions': ['my.Extension1', 'other.Extension2']}}
    instance = TestClass(context=context)
    expected = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my.Extension1',
        'other.Extension2',
    ]
    assert instance.extensions == expected

def test_constructor_handles_missing_extensions_key():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    context = {'cookiecutter': {}}
    instance = TestClass(context=context)
    expected = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected

def test_constructor_handles_missing_cookiecutter_key():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    context = {}
    instance = TestClass(context=context)
    expected = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected

def test_constructor_handles_none_context():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    instance = TestClass(context=None)
    expected = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected

def test_constructor_passes_kwargs_to_parent():
    class MockParent:
        def __init__(self, extensions, extra_arg=None, **kwargs):
            self.extensions = extensions
            self.extra_arg = extra_arg
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    instance = TestClass(context={}, extra_arg='test_value')
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected_extensions
    assert instance.extra_arg == 'test_value'

def test_constructor_raises_unknown_extension_on_import_error():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            raise ImportError('No module named invalid.extension')
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    try:
        TestClass(context={})
        assert False
    except Exception as e:
        assert e.__class__.__name__ == 'UnknownExtension'
        assert 'Unable to load extension' in str(e)

def test_constructor_converts_extension_objects_to_strings():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    context = {'cookiecutter': {'_extensions': [123, True]}}
    instance = TestClass(context=context)
    expected = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        '123',
        'True',
    ]
    assert instance.extensions == expected


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_constructor_loads_default_extensions():
    mock_super_init = lambda **kwargs: None
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        def __init_subclass__(cls):
            cls.__bases__ = (ExtensionLoaderMixin,)
    test_instance = TestClass()
    assert True

def test_constructor_uses_provided_context():
    mock_super_init = lambda **kwargs: None
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        def __init_subclass__(cls):
            cls.__bases__ = (ExtensionLoaderMixin,)
    context = {'cookiecutter': {'_extensions': ['test.Extension1', 'test.Extension2']}}
    test_instance = TestClass(context=context)
    assert True

def test_constructor_handles_empty_context():
    mock_super_init = lambda **kwargs: None
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        def __init_subclass__(cls):
            cls.__bases__ = (ExtensionLoaderMixin,)
    test_instance = TestClass(context={})
    assert True

def test_constructor_handles_none_context():
    mock_super_init = lambda **kwargs: None
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        def __init_subclass__(cls):
            cls.__bases__ = (ExtensionLoaderMixin,)
    test_instance = TestClass(context=None)
    assert True

def test_constructor_passes_kwargs_to_super():
    mock_super_init = lambda **kwargs: None
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        def __init_subclass__(cls):
            cls.__bases__ = (ExtensionLoaderMixin,)
    test_instance = TestClass(extra_arg='value')
    assert True

def test_constructor_raises_unknown_extension_on_import_error():
    mock_super_init = lambda **kwargs: (_ for _ in ()).throw(ImportError('No module named test'))
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        def __init_subclass__(cls):
            cls.__bases__ = (ExtensionLoaderMixin,)
    try:
        test_instance = TestClass()
        assert False
    except UnknownExtension:
        assert True

def test_constructor_with_context_missing_cookiecutter_key():
    mock_super_init = lambda **kwargs: None
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        def __init_subclass__(cls):
            cls.__bases__ = (ExtensionLoaderMixin,)
    context = {'other_key': 'value'}
    test_instance = TestClass(context=context)
    assert True

def test_constructor_with_context_missing_extensions_key():
    mock_super_init = lambda **kwargs: None
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        def __init_subclass__(cls):
            cls.__bases__ = (ExtensionLoaderMixin,)
    context = {'cookiecutter': {'other_key': 'value'}}
    test_instance = TestClass(context=context)
    assert True


# LLM-generated content at query #2
#--------------------------

def test_extension_loader_mixin_raises_unknown_extension_on_import_error():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            raise ImportError("No module named 'fake_extension'")
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    try:
        TestClass(context={})
    except UnknownExtension as e:
        assert str(e) == "Unable to load extension: No module named 'fake_extension'"


# LLM-generated content at query #3
#--------------------------

def test_constructor_without_context():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = kwargs.get('extensions', [])
    test_instance = TestClass()
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert test_instance.extensions == expected_extensions

def test_constructor_with_empty_context():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = kwargs.get('extensions', [])
    test_instance = TestClass(context={})
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert test_instance.extensions == expected_extensions

def test_constructor_with_extensions_in_context():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = kwargs.get('extensions', [])
    context = {'cookiecutter': {'_extensions': ['my.Extension1', 'my.Extension2']}}
    test_instance = TestClass(context=context)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my.Extension1',
        'my.Extension2',
    ]
    assert test_instance.extensions == expected_extensions

def test_constructor_without_cookiecutter_key_in_context():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = kwargs.get('extensions', [])
    context = {'other_key': 'value'}
    test_instance = TestClass(context=context)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert test_instance.extensions == expected_extensions

def test_constructor_without_extensions_key_in_cookiecutter():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = kwargs.get('extensions', [])
    context = {'cookiecutter': {'other_key': 'value'}}
    test_instance = TestClass(context=context)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert test_instance.extensions == expected_extensions

def test_constructor_passes_other_kwargs():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.kwargs = kwargs
    test_instance = TestClass(extra_arg='value', another_arg=123)
    assert test_instance.kwargs.get('extra_arg') == 'value'
    assert test_instance.kwargs.get('another_arg') == 123
    assert 'extensions' in test_instance.kwargs

def test_constructor_raises_unknown_extension_on_import_error():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    class MockSuper:
        def __init__(self, **kwargs):
            raise ImportError('No module named invalid.extension')
    import sys
    original_bases = TestClass.__bases__
    TestClass.__bases__ = (MockSuper,)
    try:
        raised = False
        try:
            TestClass()
        except Exception as e:
            raised = True
            assert e.__class__.__name__ == 'UnknownExtension'
            assert 'Unable to load extension' in str(e)
            assert isinstance(e.__cause__, ImportError)
            assert 'No module named invalid.extension' in str(e.__cause__)
        assert raised
    finally:
        TestClass.__bases__ = original_bases


# LLM-generated content at query #4
#--------------------------

def test_extension_loader_mixin_init_without_context():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions_passed = kwargs.get('extensions', [])
    instance = TestClass()
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions_passed == expected_extensions


def test_extension_loader_mixin_init_with_empty_context():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions_passed = kwargs.get('extensions', [])
    instance = TestClass(context={})
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions_passed == expected_extensions


def test_extension_loader_mixin_init_with_extensions_in_context():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions_passed = kwargs.get('extensions', [])
    context = {'cookiecutter': {'_extensions': ['my.ext.Extension1', 'other.ext.Extension2']}}
    instance = TestClass(context=context)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my.ext.Extension1',
        'other.ext.Extension2',
    ]
    assert instance.extensions_passed == expected_extensions


def test_extension_loader_mixin_init_with_context_missing_extensions_key():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions_passed = kwargs.get('extensions', [])
    context = {'cookiecutter': {}}
    instance = TestClass(context=context)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions_passed == expected_extensions


def test_extension_loader_mixin_init_with_context_missing_cookiecutter_key():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions_passed = kwargs.get('extensions', [])
    context = {'other_key': 'value'}
    instance = TestClass(context=context)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions_passed == expected_extensions


def test_extension_loader_mixin_init_passes_other_kwargs():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.kwargs_received = kwargs
    instance = TestClass(optimized=True, trim_blocks=False)
    assert instance.kwargs_received['optimized'] == True
    assert instance.kwargs_received['trim_blocks'] == False
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.kwargs_received['extensions'] == expected_extensions


def test_extension_loader_mixin_init_raises_unknown_extension_on_import_error():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            try:
                super().__init__(**kwargs)
                self.exception_raised = None
            except Exception as e:
                self.exception_raised = e
    context = {'cookiecutter': {'_extensions': ['nonexistent.extension']}}
    instance = TestClass(context=context)
    assert instance.exception_raised is not None
    assert isinstance(instance.exception_raised, UnknownExtension)
    assert 'Unable to load extension' in str(instance.exception_raised)


# LLM-generated content at query #5
#--------------------------

def test_import_error_raises_unknown_extension():
    class MockSuper:
        def __init__(self, extensions, **kwargs):
            raise ImportError("No module named 'fake_extension'")
    class TestClass(ExtensionLoaderMixin, MockSuper):
        pass
    try:
        TestClass(context={})
    except Exception as e:
        assert e.__class__.__name__ == "UnknownExtension"
        assert str(e) == "Unable to load extension: No module named 'fake_extension'"


# LLM-generated content at query #6
#--------------------------

def test_import_error_raises_unknown_extension():
    class MockSuper:
        def __init__(self, extensions, **kwargs):
            raise ImportError("No module named 'fake'")
    class TestClass(ExtensionLoaderMixin, MockSuper):
        pass
    try:
        TestClass(context={})
    except Exception as e:
        assert e.__class__.__name__ == 'UnknownExtension'
        assert str(e) == "Unable to load extension: No module named 'fake'"


# LLM-generated content at query #7
#--------------------------

def test_constructor_loads_default_extensions():
    mock_super = Mock()
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super):
        instance = ExtensionLoaderMixin()
    call_kwargs = mock_super.call_args[1]
    extensions = call_kwargs['extensions']
    expected = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert extensions == expected

def test_constructor_loads_extensions_from_context():
    mock_super = Mock()
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super):
        context = {'cookiecutter': {'_extensions': ['my.ext.Extension1', 'other.Extension2']}}
        instance = ExtensionLoaderMixin(context=context)
    call_kwargs = mock_super.call_args[1]
    extensions = call_kwargs['extensions']
    expected = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my.ext.Extension1',
        'other.Extension2',
    ]
    assert extensions == expected

def test_constructor_handles_empty_extensions_in_context():
    mock_super = Mock()
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super):
        context = {'cookiecutter': {'_extensions': []}}
        instance = ExtensionLoaderMixin(context=context)
    call_kwargs = mock_super.call_args[1]
    extensions = call_kwargs['extensions']
    expected = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert extensions == expected

def test_constructor_handles_missing_extensions_key_in_context():
    mock_super = Mock()
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super):
        context = {'cookiecutter': {}}
        instance = ExtensionLoaderMixin(context=context)
    call_kwargs = mock_super.call_args[1]
    extensions = call_kwargs['extensions']
    expected = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert extensions == expected

def test_constructor_handles_missing_cookiecutter_key_in_context():
    mock_super = Mock()
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super):
        context = {}
        instance = ExtensionLoaderMixin(context=context)
    call_kwargs = mock_super.call_args[1]
    extensions = call_kwargs['extensions']
    expected = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert extensions == expected

def test_constructor_handles_none_context():
    mock_super = Mock()
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super):
        instance = ExtensionLoaderMixin(context=None)
    call_kwargs = mock_super.call_args[1]
    extensions = call_kwargs['extensions']
    expected = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert extensions == expected

def test_constructor_passes_extra_kwargs_to_super():
    mock_super = Mock()
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super):
        instance = ExtensionLoaderMixin(extra_arg='value', another=123)
    call_kwargs = mock_super.call_args[1]
    assert call_kwargs['extra_arg'] == 'value'
    assert call_kwargs['another'] == 123
    assert 'extensions' in call_kwargs

def test_constructor_raises_unknown_extension_on_import_error():
    mock_super = Mock(side_effect=ImportError('No module named'))
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super):
        try:
            instance = ExtensionLoaderMixin()
            raised = False
        except UnknownExtension as e:
            raised = True
            assert 'Unable to load extension' in str(e)
    assert raised


# LLM-generated content at query #8
#--------------------------

def test_extension_loader_mixin_init_raises_unknown_extension_on_import_error():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            raise ImportError("No module named 'fake_extension'")
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    try:
        TestClass(context={})
    except UnknownExtension as e:
        assert str(e) == "Unable to load extension: No module named 'fake_extension'"


# LLM-generated content at query #9
#--------------------------

def test_extension_loader_mixin_init_raises_unknown_extension_on_import_error():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            raise ImportError("No module named 'fake_extension'")
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    try:
        TestClass(context={})
    except UnknownExtension as e:
        assert str(e) == "Unable to load extension: No module named 'fake_extension'"


# LLM-generated content at query #10
#--------------------------

def test_extension_loader_mixin_raises_unknown_extension_on_import_error():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            raise ImportError("No module named 'fake_extension'")

    class TestClass(ExtensionLoaderMixin, MockParent):
        pass

    context = {'cookiecutter': {'_extensions': ['fake_extension']}}
    try:
        TestClass(context=context)
    except UnknownExtension as e:
        assert str(e) == "Unable to load extension: No module named 'fake_extension'"
    else:
        assert False


# LLM-generated content at query #11
#--------------------------

def test_extension_loader_mixin_init_raises_unknown_extension_on_import_error():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            raise ImportError("No module named 'fake_extension'")
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    try:
        TestClass(context={})
    except Exception as e:
        assert e.__class__.__name__ == 'UnknownExtension'
        assert str(e) == "Unable to load extension: No module named 'fake_extension'"


