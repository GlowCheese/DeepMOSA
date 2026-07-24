####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_constructor_loads_default_extensions():
    mock_super = Mock()
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super.__init__):
        instance = ExtensionLoaderMixin()
    call_kwargs = mock_super.call_args[1]
    extensions = call_kwargs['extensions']
    assert 'cookiecutter.extensions.JsonifyExtension' in extensions
    assert 'cookiecutter.extensions.RandomStringExtension' in extensions
    assert 'cookiecutter.extensions.SlugifyExtension' in extensions
    assert 'cookiecutter.extensions.TimeExtension' in extensions
    assert 'cookiecutter.extensions.UUIDExtension' in extensions

def test_constructor_reads_extensions_from_context():
    mock_super = Mock()
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super.__init__):
        context = {'cookiecutter': {'_extensions': ['my.Extension', 'other.Ext']}}
        instance = ExtensionLoaderMixin(context=context)
    call_kwargs = mock_super.call_args[1]
    extensions = call_kwargs['extensions']
    assert 'my.Extension' in extensions
    assert 'other.Ext' in extensions
    assert 'cookiecutter.extensions.JsonifyExtension' in extensions

def test_constructor_handles_empty_extensions_in_context():
    mock_super = Mock()
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super.__init__):
        context = {'cookiecutter': {}}
        instance = ExtensionLoaderMixin(context=context)
    call_kwargs = mock_super.call_args[1]
    extensions = call_kwargs['extensions']
    assert len(extensions) == 5
    assert 'cookiecutter.extensions.JsonifyExtension' in extensions

def test_constructor_handles_missing_cookiecutter_key():
    mock_super = Mock()
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super.__init__):
        context = {}
        instance = ExtensionLoaderMixin(context=context)
    call_kwargs = mock_super.call_args[1]
    extensions = call_kwargs['extensions']
    assert len(extensions) == 5
    assert 'cookiecutter.extensions.JsonifyExtension' in extensions

def test_constructor_handles_none_context():
    mock_super = Mock()
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super.__init__):
        instance = ExtensionLoaderMixin(context=None)
    call_kwargs = mock_super.call_args[1]
    extensions = call_kwargs['extensions']
    assert len(extensions) == 5
    assert 'cookiecutter.extensions.JsonifyExtension' in extensions

def test_constructor_passes_extra_kwargs_to_super():
    mock_super = Mock()
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super.__init__):
        instance = ExtensionLoaderMixin(context=None, extra_arg='value', another=123)
    call_kwargs = mock_super.call_args[1]
    assert call_kwargs['extra_arg'] == 'value'
    assert call_kwargs['another'] == 123
    assert 'extensions' in call_kwargs

def test_constructor_raises_unknown_extension_on_import_error():
    mock_super = Mock()
    mock_super.side_effect = ImportError('No module named fake')
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super.__init__):
        try:
            instance = ExtensionLoaderMixin()
            raised = False
        except UnknownExtension as e:
            raised = True
            assert 'Unable to load extension' in str(e)
    assert raised

def test_constructor_uses_str_conversion_for_extensions():
    mock_super = Mock()
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super.__init__):
        context = {'cookiecutter': {'_extensions': [Mock(__str__=Mock(return_value='mock.Extension'))]}}
        instance = ExtensionLoaderMixin(context=context)
    call_kwargs = mock_super.call_args[1]
    extensions = call_kwargs['extensions']
    assert 'mock.Extension' in extensions


# LLM-generated content at query #2
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


# LLM-generated content at query #3
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
    else:
        assert False, "Expected UnknownExtension to be raised"


# LLM-generated content at query #4
#--------------------------

def test_extension_loader_mixin_raises_unknown_extension_on_import_error():
    class MockSuper:
        def __init__(self, extensions, **kwargs):
            raise ImportError("No module named 'fake'")
    class TestClass(ExtensionLoaderMixin, MockSuper):
        pass
    try:
        TestClass(context={})
    except UnknownExtension as e:
        assert str(e) == "Unable to load extension: No module named 'fake'"


# LLM-generated content at query #5
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


# LLM-generated content at query #6
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

def test_init_raises_unknown_extension_on_import_error():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            raise ImportError("No module named 'fake_extension'")
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    try:
        TestClass(context={})
    except UnknownExtension as e:
        assert str(e) == "Unable to load extension: No module named 'fake_extension'"


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_extension_loader_mixin_initializes_with_default_extensions():
    class TestEnvironment:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
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
    assert instance.kwargs == {}

def test_extension_loader_mixin_initializes_with_context_extensions():
    class TestEnvironment:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, TestEnvironment):
        pass
    context = {'cookiecutter': {'_extensions': ['some.Extension', 'another.Extension']}}
    instance = TestClass(context=context)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'some.Extension',
        'another.Extension',
    ]
    assert instance.extensions == expected_extensions

def test_extension_loader_mixin_initializes_with_empty_context_extensions():
    class TestEnvironment:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
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
            self.kwargs = kwargs
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

def test_extension_loader_mixin_initializes_with_additional_kwargs():
    class TestEnvironment:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, TestEnvironment):
        pass
    instance = TestClass(some_arg='value', another_arg=123)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected_extensions
    assert instance.kwargs == {'some_arg': 'value', 'another_arg': 123}

def test_extension_loader_mixin_raises_unknown_extension_on_import_error():
    class TestEnvironment:
        def __init__(self, extensions, **kwargs):
            raise ImportError('No module named some_module')
    class TestClass(ExtensionLoaderMixin, TestEnvironment):
        pass
    try:
        TestClass()
        assert False, 'Expected UnknownExtension to be raised'
    except UnknownExtension as e:
        assert str(e) == 'Unable to load extension: No module named some_module'
        assert isinstance(e.__cause__, ImportError)
        assert str(e.__cause__) == 'No module named some_module'


# LLM-generated content at query #2
#--------------------------

def test_init_without_context():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    instance = TestClass()
    expected_defaults = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected_defaults
    assert instance.kwargs == {}

def test_init_with_empty_context():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    instance = TestClass(context={})
    expected_defaults = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected_defaults
    assert instance.kwargs == {}

def test_init_with_context_without_extensions():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    context = {'cookiecutter': {'other_key': 'value'}}
    instance = TestClass(context=context)
    expected_defaults = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected_defaults
    assert instance.kwargs == {}

def test_init_with_context_with_extensions():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    context = {'cookiecutter': {'_extensions': ['my.extension.Extension', 'another.extension.Extension']}}
    instance = TestClass(context=context)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my.extension.Extension',
        'another.extension.Extension',
    ]
    assert instance.extensions == expected_extensions
    assert instance.kwargs == {}

def test_init_with_additional_kwargs():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    instance = TestClass(extra_arg='value', another_arg=123)
    expected_defaults = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected_defaults
    assert instance.kwargs == {'extra_arg': 'value', 'another_arg': 123}

def test_init_raises_unknown_extension_on_import_error():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            raise ImportError('No module named cookiecutter.extensions')
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    try:
        TestClass()
        assert False, "Expected UnknownExtension to be raised"
    except UnknownExtension as e:
        assert str(e) == "Unable to load extension: No module named cookiecutter.extensions"


# LLM-generated content at query #3
#--------------------------

def test_import_error_raises_unknown_extension():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            raise ImportError("No module named 'fake'")
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    try:
        TestClass(context={})
    except UnknownExtension as e:
        assert str(e) == "Unable to load extension: No module named 'fake'"


# LLM-generated content at query #4
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


# LLM-generated content at query #5
#--------------------------

def test_init_without_context():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    obj = TestClass()
    expected_defaults = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert obj.extensions == expected_defaults
    assert obj.kwargs == {}

def test_init_with_empty_context():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    obj = TestClass(context={})
    expected_defaults = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert obj.extensions == expected_defaults
    assert obj.kwargs == {}

def test_init_with_context_containing_extensions():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    context = {'cookiecutter': {'_extensions': ['my.extension.Extension', 'another.extension.Foo']}}
    obj = TestClass(context=context)
    expected = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my.extension.Extension',
        'another.extension.Foo',
    ]
    assert obj.extensions == expected
    assert obj.kwargs == {}

def test_init_with_context_missing_extensions_key():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    context = {'cookiecutter': {}}
    obj = TestClass(context=context)
    expected_defaults = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert obj.extensions == expected_defaults
    assert obj.kwargs == {}

def test_init_with_context_missing_cookiecutter_key():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    context = {'other': 'value'}
    obj = TestClass(context=context)
    expected_defaults = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert obj.extensions == expected_defaults
    assert obj.kwargs == {}

def test_init_with_additional_kwargs():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    obj = TestClass(foo='bar', baz=42)
    expected_defaults = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert obj.extensions == expected_defaults
    assert obj.kwargs == {'foo': 'bar', 'baz': 42}

def test_init_with_context_and_additional_kwargs():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    context = {'cookiecutter': {'_extensions': ['custom.Extension']}}
    obj = TestClass(context=context, test='value')
    expected = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'custom.Extension',
    ]
    assert obj.extensions == expected
    assert obj.kwargs == {'test': 'value'}

def test_init_raises_unknown_extension_on_import_error():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            raise ImportError('No module named cookiecutter.extensions')
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    try:
        TestClass()
        assert False, "Expected UnknownExtension to be raised"
    except Exception as e:
        assert e.__class__.__name__ == 'UnknownExtension'
        assert str(e) == 'Unable to load extension: No module named cookiecutter.extensions'


# LLM-generated content at query #6
#--------------------------

def test_init_raises_unknown_extension_on_import_error():
    class MockSuper:
        def __init__(self, extensions, **kwargs):
            raise ImportError("No module named 'fake'")
    class TestClass(ExtensionLoaderMixin, MockSuper):
        pass
    try:
        TestClass(context={})
    except UnknownExtension as e:
        assert str(e) == "Unable to load extension: No module named 'fake'"


# LLM-generated content at query #7
#--------------------------

def test_init_without_context_and_no_extensions():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            pass
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    obj = TestClass()
    assert True

def test_init_with_empty_context_and_no_extensions():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            pass
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    obj = TestClass(context={})
    assert True

def test_init_with_context_missing_cookiecutter_key():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            pass
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    obj = TestClass(context={"other_key": "value"})
    assert True

def test_init_with_context_cookiecutter_missing_extensions_key():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            pass
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    obj = TestClass(context={"cookiecutter": {"key": "value"}})
    assert True

def test_init_with_context_and_empty_extensions_list():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            pass
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    obj = TestClass(context={"cookiecutter": {"_extensions": []}})
    assert True

def test_init_with_context_and_valid_extensions_list():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            pass
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    obj = TestClass(context={"cookiecutter": {"_extensions": ["some.extension"]}})
    assert True

def test_init_with_additional_kwargs():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            pass
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    obj = TestClass(some_arg="value")
    assert True

def test_init_with_context_and_additional_kwargs():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            pass
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    obj = TestClass(context={"cookiecutter": {"_extensions": ["ext"]}}, other_arg=123)
    assert True


# LLM-generated content at query #8
#--------------------------

def test_extension_loader_mixin_init_without_context():
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


def test_extension_loader_mixin_init_with_empty_context():
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


def test_extension_loader_mixin_init_with_extensions_in_context():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = kwargs.get('extensions', [])
    context = {'cookiecutter': {'_extensions': ['my.ext.Extension1', 'other.ext.Extension2']}}
    test_instance = TestClass(context=context)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my.ext.Extension1',
        'other.ext.Extension2',
    ]
    assert test_instance.extensions == expected_extensions


def test_extension_loader_mixin_init_with_context_missing_extensions_key():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = kwargs.get('extensions', [])
    context = {'cookiecutter': {}}
    test_instance = TestClass(context=context)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert test_instance.extensions == expected_extensions


def test_extension_loader_mixin_init_with_context_missing_cookiecutter_key():
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


def test_extension_loader_mixin_init_passes_other_kwargs():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.kwargs = kwargs
    test_instance = TestClass(extra_arg='value', another_arg=123)
    assert test_instance.kwargs.get('extra_arg') == 'value'
    assert test_instance.kwargs.get('another_arg') == 123
    assert 'extensions' in test_instance.kwargs


# LLM-generated content at query #9
#--------------------------

def test_context_is_none():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            pass
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    obj = TestClass(context=None)


# LLM-generated content at query #10
#--------------------------

def test_init_without_context():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    obj = TestClass()
    assert True

def test_init_with_none_context():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    obj = TestClass(context=None)
    assert True

def test_init_with_empty_context():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    obj = TestClass(context={})
    assert True

def test_init_with_context():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    obj = TestClass(context={"key": "value"})
    assert True

def test_init_with_additional_kwargs():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    obj = TestClass(context={"key": "value"}, extra_arg=123)
    assert True


# LLM-generated content at query #11
#--------------------------

def test_init_without_context():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            pass
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    obj = TestClass()
    assert obj._read_extensions({}) == []

def test_init_with_empty_context():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            pass
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    obj = TestClass(context={})
    assert obj._read_extensions({}) == []

def test_init_with_context_missing_cookiecutter():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            pass
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    obj = TestClass(context={'other_key': 'value'})
    assert obj._read_extensions({'other_key': 'value'}) == []

def test_init_with_context_missing_extensions():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            pass
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    obj = TestClass(context={'cookiecutter': {'key': 'value'}})
    assert obj._read_extensions({'cookiecutter': {'key': 'value'}}) == []


# LLM-generated content at query #12
#--------------------------

def test_init_without_context_uses_empty_dict():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
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
    assert instance.kwargs == {}

def test_init_with_empty_context_uses_empty_dict():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    instance = TestClass(context={})
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected_extensions
    assert instance.kwargs == {}

def test_init_with_context_without_extensions_key():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
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
    assert instance.kwargs == {}

def test_init_with_context_with_empty_extensions_list():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    context = {'cookiecutter': {'_extensions': []}}
    instance = TestClass(context=context)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected_extensions
    assert instance.kwargs == {}

def test_init_with_context_with_additional_extensions():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    context = {'cookiecutter': {'_extensions': ['my.extension.Extension1', 'my.extension.Extension2']}}
    instance = TestClass(context=context)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my.extension.Extension1',
        'my.extension.Extension2',
    ]
    assert instance.extensions == expected_extensions
    assert instance.kwargs == {}

def test_init_passes_additional_kwargs_to_parent():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    instance = TestClass(context=None, extra_arg=123, another='value')
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected_extensions
    assert instance.kwargs == {'extra_arg': 123, 'another': 'value'}

def test_init_raises_unknown_extension_on_import_error():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            raise ImportError('No module named cookiecutter.extensions')
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    try:
        TestClass()
        assert False, "Expected UnknownExtension to be raised"
    except Exception as e:
        assert e.__class__.__name__ == 'UnknownExtension'
        assert str(e) == 'Unable to load extension: No module named cookiecutter.extensions'


# LLM-generated content at query #13
#--------------------------

def test_import_error_raises_unknown_extension():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            raise ImportError("No module named 'fake'")
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    try:
        TestClass(context={})
    except UnknownExtension as e:
        assert str(e) == "Unable to load extension: No module named 'fake'"


# LLM-generated content at query #14
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


def test_constructor_passes_other_kwargs():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = kwargs.get('extensions', [])
            self.other_arg = kwargs.get('other_arg', None)

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


# LLM-generated content at query #15
#--------------------------

def test_init_without_context():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    obj = TestClass()
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert obj.kwargs == {}

def test_init_with_empty_context():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    obj = TestClass(context={})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert obj.kwargs == {}

def test_init_with_context_without_extensions():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    obj = TestClass(context={'cookiecutter': {}})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert obj.kwargs == {}

def test_init_with_context_with_extensions():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    obj = TestClass(context={'cookiecutter': {'_extensions': ['my.Extension1', 'my.Extension2']}})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my.Extension1',
        'my.Extension2',
    ]
    assert obj.kwargs == {}

def test_init_with_additional_kwargs():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    obj = TestClass(context={'cookiecutter': {'_extensions': ['my.Extension']}}, extra_arg=123)
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my.Extension',
    ]
    assert obj.kwargs == {'extra_arg': 123}

def test_init_raises_unknown_extension_on_import_error():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            raise ImportError('No module named cookiecutter.extensions')
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    try:
        TestClass()
    except UnknownExtension as e:
        assert str(e) == 'Unable to load extension: No module named cookiecutter.extensions'
    else:
        assert False, 'Expected UnknownExtension to be raised'


# LLM-generated content at query #16
#--------------------------

def test_constructor_without_context():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = []
        def __init__(self, extensions=None, **kwargs):
            self.extensions = extensions if extensions is not None else []
    instance = TestClass()
    assert len(instance.extensions) == 5
    assert 'cookiecutter.extensions.JsonifyExtension' in instance.extensions
    assert 'cookiecutter.extensions.RandomStringExtension' in instance.extensions
    assert 'cookiecutter.extensions.SlugifyExtension' in instance.extensions
    assert 'cookiecutter.extensions.TimeExtension' in instance.extensions
    assert 'cookiecutter.extensions.UUIDExtension' in instance.extensions

def test_constructor_with_empty_context():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = []
        def __init__(self, extensions=None, **kwargs):
            self.extensions = extensions if extensions is not None else []
    instance = TestClass(context={})
    assert len(instance.extensions) == 5
    assert 'cookiecutter.extensions.JsonifyExtension' in instance.extensions
    assert 'cookiecutter.extensions.RandomStringExtension' in instance.extensions
    assert 'cookiecutter.extensions.SlugifyExtension' in instance.extensions
    assert 'cookiecutter.extensions.TimeExtension' in instance.extensions
    assert 'cookiecutter.extensions.UUIDExtension' in instance.extensions

def test_constructor_with_extensions_in_context():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = []
        def __init__(self, extensions=None, **kwargs):
            self.extensions = extensions if extensions is not None else []
    context = {'cookiecutter': {'_extensions': ['my.extension.Extension1', 'another.extension.Extension2']}}
    instance = TestClass(context=context)
    assert len(instance.extensions) == 7
    assert 'cookiecutter.extensions.JsonifyExtension' in instance.extensions
    assert 'cookiecutter.extensions.RandomStringExtension' in instance.extensions
    assert 'cookiecutter.extensions.SlugifyExtension' in instance.extensions
    assert 'cookiecutter.extensions.TimeExtension' in instance.extensions
    assert 'cookiecutter.extensions.UUIDExtension' in instance.extensions
    assert 'my.extension.Extension1' in instance.extensions
    assert 'another.extension.Extension2' in instance.extensions

def test_constructor_with_context_missing_cookiecutter_key():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = []
        def __init__(self, extensions=None, **kwargs):
            self.extensions = extensions if extensions is not None else []
    context = {'other_key': 'value'}
    instance = TestClass(context=context)
    assert len(instance.extensions) == 5
    assert 'cookiecutter.extensions.JsonifyExtension' in instance.extensions
    assert 'cookiecutter.extensions.RandomStringExtension' in instance.extensions
    assert 'cookiecutter.extensions.SlugifyExtension' in instance.extensions
    assert 'cookiecutter.extensions.TimeExtension' in instance.extensions
    assert 'cookiecutter.extensions.UUIDExtension' in instance.extensions

def test_constructor_with_context_missing_extensions_key():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = []
        def __init__(self, extensions=None, **kwargs):
            self.extensions = extensions if extensions is not None else []
    context = {'cookiecutter': {'other_key': 'value'}}
    instance = TestClass(context=context)
    assert len(instance.extensions) == 5
    assert 'cookiecutter.extensions.JsonifyExtension' in instance.extensions
    assert 'cookiecutter.extensions.RandomStringExtension' in instance.extensions
    assert 'cookiecutter.extensions.SlugifyExtension' in instance.extensions
    assert 'cookiecutter.extensions.TimeExtension' in instance.extensions
    assert 'cookiecutter.extensions.UUIDExtension' in instance.extensions

def test_constructor_passes_other_keyword_arguments():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = []
            self.other_arg = None
        def __init__(self, extensions=None, other_arg=None, **kwargs):
            self.extensions = extensions if extensions is not None else []
            self.other_arg = other_arg
    instance = TestClass(other_arg='test_value')
    assert instance.other_arg == 'test_value'
    assert len(instance.extensions) == 5

def test_constructor_raises_unknown_extension_on_import_error():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    class MockSuper:
        def __init__(self, extensions, **kwargs):
            raise ImportError('No module named')
    import sys
    original_bases = TestClass.__bases__
    TestClass.__bases__ = (MockSuper,)
    try:
        exception_raised = False
        try:
            instance = TestClass()
        except Exception as e:
            exception_raised = True
            assert e.__class__.__name__ == 'UnknownExtension'
            assert 'Unable to load extension' in str(e)
        assert exception_raised
    finally:
        TestClass.__bases__ = original_bases


# LLM-generated content at query #17
#--------------------------

def test_constructor_loads_default_extensions():
    mock_super = Mock()
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super):
        instance = ExtensionLoaderMixin()
        call_args = mock_super.call_args
        passed_extensions = call_args[1]['extensions']
        expected = [
            'cookiecutter.extensions.JsonifyExtension',
            'cookiecutter.extensions.RandomStringExtension',
            'cookiecutter.extensions.SlugifyExtension',
            'cookiecutter.extensions.TimeExtension',
            'cookiecutter.extensions.UUIDExtension',
        ]
        assert passed_extensions == expected

def test_constructor_loads_context_extensions():
    mock_super = Mock()
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super):
        context = {'cookiecutter': {'_extensions': ['my.ext.Extension1', 'other.Extension2']}}
        instance = ExtensionLoaderMixin(context=context)
        call_args = mock_super.call_args
        passed_extensions = call_args[1]['extensions']
        expected = [
            'cookiecutter.extensions.JsonifyExtension',
            'cookiecutter.extensions.RandomStringExtension',
            'cookiecutter.extensions.SlugifyExtension',
            'cookiecutter.extensions.TimeExtension',
            'cookiecutter.extensions.UUIDExtension',
            'my.ext.Extension1',
            'other.Extension2',
        ]
        assert passed_extensions == expected

def test_constructor_handles_missing_extensions_key():
    mock_super = Mock()
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super):
        context = {'cookiecutter': {}}
        instance = ExtensionLoaderMixin(context=context)
        call_args = mock_super.call_args
        passed_extensions = call_args[1]['extensions']
        expected = [
            'cookiecutter.extensions.JsonifyExtension',
            'cookiecutter.extensions.RandomStringExtension',
            'cookiecutter.extensions.SlugifyExtension',
            'cookiecutter.extensions.TimeExtension',
            'cookiecutter.extensions.UUIDExtension',
        ]
        assert passed_extensions == expected

def test_constructor_handles_missing_cookiecutter_key():
    mock_super = Mock()
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super):
        context = {}
        instance = ExtensionLoaderMixin(context=context)
        call_args = mock_super.call_args
        passed_extensions = call_args[1]['extensions']
        expected = [
            'cookiecutter.extensions.JsonifyExtension',
            'cookiecutter.extensions.RandomStringExtension',
            'cookiecutter.extensions.SlugifyExtension',
            'cookiecutter.extensions.TimeExtension',
            'cookiecutter.extensions.UUIDExtension',
        ]
        assert passed_extensions == expected

def test_constructor_handles_none_context():
    mock_super = Mock()
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super):
        instance = ExtensionLoaderMixin(context=None)
        call_args = mock_super.call_args
        passed_extensions = call_args[1]['extensions']
        expected = [
            'cookiecutter.extensions.JsonifyExtension',
            'cookiecutter.extensions.RandomStringExtension',
            'cookiecutter.extensions.SlugifyExtension',
            'cookiecutter.extensions.TimeExtension',
            'cookiecutter.extensions.UUIDExtension',
        ]
        assert passed_extensions == expected

def test_constructor_passes_extra_kwargs():
    mock_super = Mock()
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super):
        instance = ExtensionLoaderMixin(extra_arg='value', another=123)
        call_args = mock_super.call_args
        assert call_args[1]['extra_arg'] == 'value'
        assert call_args[1]['another'] == 123

def test_constructor_raises_unknown_extension_on_import_error():
    mock_super = Mock(side_effect=ImportError('No module named'))
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super):
        try:
            instance = ExtensionLoaderMixin()
            assert False, "Expected UnknownExtension to be raised"
        except UnknownExtension as e:
            assert 'Unable to load extension' in str(e)


# LLM-generated content at query #18
#--------------------------

def test_init_raises_unknown_extension_on_import_error():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            raise ImportError("No module named 'fake'")
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    try:
        TestClass(context={})
    except Exception as e:
        assert e.__class__.__name__ == "UnknownExtension"
        assert str(e) == "Unable to load extension: No module named 'fake'"


# LLM-generated content at query #19
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

def test_constructor_with_context_without_extensions():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = kwargs.get('extensions', [])
    test_instance = TestClass(context={'cookiecutter': {}})
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert test_instance.extensions == expected_extensions

def test_constructor_with_context_with_extensions():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = kwargs.get('extensions', [])
    context = {
        'cookiecutter': {
            '_extensions': ['my_extension.Extension1', 'another.Extension2']
        }
    }
    test_instance = TestClass(context=context)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my_extension.Extension1',
        'another.Extension2',
    ]
    assert test_instance.extensions == expected_extensions

def test_constructor_passes_other_kwargs():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.kwargs = kwargs
    test_instance = TestClass(some_arg='value', another_arg=123)
    assert test_instance.kwargs.get('some_arg') == 'value'
    assert test_instance.kwargs.get('another_arg') == 123
    assert 'extensions' in test_instance.kwargs

def test_constructor_raises_unknown_extension_on_import_error():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    class MockSuper:
        def __init__(self, **kwargs):
            raise ImportError('No module named')
    import sys
    original_super = __builtins__['super']
    __builtins__['super'] = lambda *args, **kwargs: MockSuper()
    try:
        raised = False
        try:
            TestClass()
        except Exception as e:
            raised = True
            assert e.__class__.__name__ == 'UnknownExtension'
            assert 'Unable to load extension' in str(e)
        assert raised
    finally:
        __builtins__['super'] = original_super


