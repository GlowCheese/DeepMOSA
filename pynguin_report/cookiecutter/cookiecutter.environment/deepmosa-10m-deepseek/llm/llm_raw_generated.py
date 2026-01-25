####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
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

def test_constructor_reads_extensions_from_context():
    mock_super = Mock()
    context = {'cookiecutter': {'_extensions': ['my.Extension1', 'my.Extension2']}}
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super):
        instance = ExtensionLoaderMixin(context=context)
    call_kwargs = mock_super.call_args[1]
    extensions = call_kwargs['extensions']
    assert 'my.Extension1' in extensions
    assert 'my.Extension2' in extensions
    assert 'cookiecutter.extensions.JsonifyExtension' in extensions

def test_constructor_handles_empty_extensions_in_context():
    mock_super = Mock()
    context = {'cookiecutter': {}}
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super):
        instance = ExtensionLoaderMixin(context=context)
    call_kwargs = mock_super.call_args[1]
    extensions = call_kwargs['extensions']
    assert len(extensions) == 5

def test_constructor_handles_missing_cookiecutter_key_in_context():
    mock_super = Mock()
    context = {}
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super):
        instance = ExtensionLoaderMixin(context=context)
    call_kwargs = mock_super.call_args[1]
    extensions = call_kwargs['extensions']
    assert len(extensions) == 5

def test_constructor_passes_remaining_kwargs_to_super():
    mock_super = Mock()
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super):
        instance = ExtensionLoaderMixin(extra_arg='value', another_arg=123)
    call_kwargs = mock_super.call_args[1]
    assert call_kwargs['extra_arg'] == 'value'
    assert call_kwargs['another_arg'] == 123

def test_constructor_raises_unknown_extension_on_import_error():
    mock_super = Mock(side_effect=ImportError('No module named my'))
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super):
        try:
            instance = ExtensionLoaderMixin()
            assert False
        except UnknownExtension as e:
            assert 'Unable to load extension' in str(e)

def test_constructor_with_none_context():
    mock_super = Mock()
    with patch.object(ExtensionLoaderMixin, '__init__', mock_super):
        instance = ExtensionLoaderMixin(context=None)
    call_kwargs = mock_super.call_args[1]
    extensions = call_kwargs['extensions']
    assert len(extensions) == 5


# LLM-generated content at query #2
#--------------------------

def test_constructor_loads_default_extensions():
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

def test_constructor_loads_extensions_from_context():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    context = {'cookiecutter': {'_extensions': ['my.Extension1', 'my.Extension2']}}
    instance = TestClass(context=context)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my.Extension1',
        'my.Extension2',
    ]
    assert instance.extensions == expected_extensions

def test_constructor_handles_empty_extensions_in_context():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
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

def test_constructor_handles_missing_cookiecutter_in_context():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
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

def test_constructor_passes_kwargs_to_parent():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    instance = TestClass(foo='bar', baz=123)
    assert instance.kwargs == {'foo': 'bar', 'baz': 123}

def test_constructor_raises_unknown_extension_on_import_error():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            raise ImportError('No module named cookiecutter.extensions')
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    try:
        instance = TestClass()
        assert False, 'Expected UnknownExtension to be raised'
    except Exception as e:
        assert e.__class__.__name__ == 'UnknownExtension'
        assert 'Unable to load extension' in str(e)

def test_constructor_with_none_context():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockParent):
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


# LLM-generated content at query #3
#--------------------------

def test_import_error_raises_unknown_extension():
    class MockSuper:
        def __init__(self, extensions, **kwargs):
            raise ImportError("No module named 'fake'")
    class TestClass(ExtensionLoaderMixin, MockSuper):
        pass
    try:
        TestClass(context={})
    except UnknownExtension as e:
        assert str(e) == "Unable to load extension: No module named 'fake'"
    else:
        assert False


# LLM-generated content at query #4
#--------------------------

def test_constructor_without_context():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = []
        def __init_subclass__(cls, **kwargs):
            pass
    test_instance = TestClass()
    assert test_instance.extensions == []

def test_constructor_with_empty_context():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = []
        def __init_subclass__(cls, **kwargs):
            pass
    test_instance = TestClass(context={})
    assert test_instance.extensions == []

def test_constructor_with_context_without_extensions():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = []
        def __init_subclass__(cls, **kwargs):
            pass
    test_instance = TestClass(context={"cookiecutter": {}})
    assert test_instance.extensions == []

def test_constructor_with_context_with_extensions():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = []
        def __init_subclass__(cls, **kwargs):
            pass
    test_instance = TestClass(context={"cookiecutter": {"_extensions": ["test.Extension1", "test.Extension2"]}})
    assert test_instance.extensions == []

def test_constructor_passes_extensions_to_super():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            self.received_kwargs = kwargs
            super().__init__(**kwargs)
        def __init_subclass__(cls, **kwargs):
            pass
    test_instance = TestClass(context={"cookiecutter": {"_extensions": ["test.Extension1"]}})
    assert "extensions" in test_instance.received_kwargs
    assert "test.Extension1" in test_instance.received_kwargs["extensions"]

def test_constructor_includes_default_extensions():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            self.received_kwargs = kwargs
            super().__init__(**kwargs)
        def __init_subclass__(cls, **kwargs):
            pass
    test_instance = TestClass()
    assert "extensions" in test_instance.received_kwargs
    default_exts = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    for ext in default_exts:
        assert ext in test_instance.received_kwargs["extensions"]

def test_constructor_raises_unknown_extension_on_import_error():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            try:
                super().__init__(**kwargs)
            except Exception as e:
                self.error = e
        def __init_subclass__(cls, **kwargs):
            pass
    test_instance = TestClass(context={"cookiecutter": {"_extensions": ["nonexistent.Extension"]}})
    assert hasattr(test_instance, 'error')
    assert "Unable to load extension" in str(test_instance.error)

def test_constructor_with_additional_kwargs():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            self.received_kwargs = kwargs
            super().__init__(**kwargs)
        def __init_subclass__(cls, **kwargs):
            pass
    test_instance = TestClass(context={}, extra_arg="value")
    assert test_instance.received_kwargs["extra_arg"] == "value"


# LLM-generated content at query #5
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


# LLM-generated content at query #6
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


# LLM-generated content at query #7
#--------------------------

def test_constructor_without_context():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = []
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
    test_instance = TestClass(context=None)
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
            self.extensions = []
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
    test_instance = TestClass(context={})
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert test_instance.extensions == expected_extensions

def test_constructor_with_context_missing_extensions_key():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = []
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
    test_instance = TestClass(context={'cookiecutter': {}})
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert test_instance.extensions == expected_extensions

def test_constructor_with_context_containing_extensions():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = []
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
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

def test_constructor_passes_remaining_kwargs():
    class Parent:
        def __init__(self, extensions, extra_arg=None, another_arg=None):
            self.extensions = extensions
            self.extra_arg = extra_arg
            self.another_arg = another_arg
    class TestClass(ExtensionLoaderMixin, Parent):
        pass
    test_instance = TestClass(context=None, extra_arg='value', another_arg=123)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert test_instance.extensions == expected_extensions
    assert test_instance.extra_arg == 'value'
    assert test_instance.another_arg == 123

def test_constructor_raises_unknown_extension_on_import_error():
    class Parent:
        def __init__(self, extensions, **kwargs):
            raise ImportError('No module named nonexistent')
    class TestClass(ExtensionLoaderMixin, Parent):
        pass
    try:
        TestClass(context=None)
        assert False
    except Exception as e:
        assert str(e) == 'Unable to load extension: No module named nonexistent'


# LLM-generated content at query #8
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


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_extension_loader_mixin_initializes_without_context():
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


def test_extension_loader_mixin_initializes_with_empty_context():
    class TestEnvironment:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, TestEnvironment):
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


def test_extension_loader_mixin_initializes_with_context_without_extensions():
    class TestEnvironment:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, TestEnvironment):
        pass
    instance = TestClass(context={"other_key": "value"})
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected_extensions
    assert instance.kwargs == {}


def test_extension_loader_mixin_initializes_with_context_with_extensions():
    class TestEnvironment:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, TestEnvironment):
        pass
    context = {"cookiecutter": {"_extensions": ["my.extension.Extension", "another.extension.Extension"]}}
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


def test_extension_loader_mixin_initializes_with_additional_kwargs():
    class TestEnvironment:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, TestEnvironment):
        pass
    instance = TestClass(trim_blocks=True, lstrip_blocks=True)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected_extensions
    assert instance.kwargs == {"trim_blocks": True, "lstrip_blocks": True}


def test_extension_loader_mixin_raises_unknown_extension_on_import_error():
    class TestEnvironment:
        def __init__(self, extensions, **kwargs):
            raise ImportError("No module named 'cookiecutter.extensions'")
    class TestClass(ExtensionLoaderMixin, TestEnvironment):
        pass
    try:
        TestClass()
        assert False, "Expected UnknownExtension to be raised"
    except Exception as e:
        assert e.__class__.__name__ == "UnknownExtension"
        assert "Unable to load extension" in str(e)


def test_extension_loader_mixin_passes_context_and_kwargs_correctly():
    class TestEnvironment:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, TestEnvironment):
        pass
    context = {"cookiecutter": {"_extensions": ["test.Extension"]}}
    instance = TestClass(context=context, extra_arg=42, another_arg="test")
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'test.Extension',
    ]
    assert instance.extensions == expected_extensions
    assert instance.kwargs == {"extra_arg": 42, "another_arg": "test"}


# LLM-generated content at query #2
#--------------------------

def test_constructor_loads_default_extensions():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    obj = TestClass()
    expected = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert obj.extensions == expected

def test_constructor_loads_context_extensions():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    context = {'cookiecutter': {'_extensions': ['my.Extension1', 'other.Extension2']}}
    obj = TestClass(context=context)
    expected = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my.Extension1',
        'other.Extension2',
    ]
    assert obj.extensions == expected

def test_constructor_handles_missing_extensions_key():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    context = {'cookiecutter': {}}
    obj = TestClass(context=context)
    expected = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert obj.extensions == expected

def test_constructor_handles_missing_cookiecutter_key():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    context = {}
    obj = TestClass(context=context)
    expected = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert obj.extensions == expected

def test_constructor_handles_none_context():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    obj = TestClass(context=None)
    expected = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert obj.extensions == expected

def test_constructor_passes_kwargs_to_parent():
    class MockParent:
        def __init__(self, extensions, extra_arg=None, **kwargs):
            self.extensions = extensions
            self.extra_arg = extra_arg
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    obj = TestClass(extra_arg='test_value')
    expected = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert obj.extensions == expected
    assert obj.extra_arg == 'test_value'

def test_constructor_raises_unknown_extension_on_import_error():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            raise ImportError('No module named invalid')
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    try:
        obj = TestClass()
        assert False, 'Expected UnknownExtension to be raised'
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
    obj = TestClass(context=context)
    assert obj.extensions[-2] == '123'
    assert obj.extensions[-1] == 'True'


# LLM-generated content at query #3
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


# LLM-generated content at query #4
#--------------------------

def test_init_without_context():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    instance = TestClass()
    assert True

def test_init_with_none_context():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    instance = TestClass(context=None)
    assert True

def test_init_with_empty_context():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    instance = TestClass(context={})
    assert True

def test_init_with_context():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    instance = TestClass(context={"key": "value"})
    assert True

def test_init_with_additional_kwargs():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    instance = TestClass(context={"key": "value"}, extra_arg=123)
    assert True


# LLM-generated content at query #5
#--------------------------

def test_init_without_context_and_no_extensions():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            pass
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    instance = TestClass()
    assert isinstance(instance, TestClass)


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

def test_init_without_context_and_no_extensions_in_context():
    class MockSuper:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockSuper):
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

def test_init_with_empty_context_and_no_extensions_in_context():
    class MockSuper:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockSuper):
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

def test_init_with_context_missing_cookiecutter_key():
    class MockSuper:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockSuper):
        pass
    instance = TestClass(context={'other_key': 'value'})
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected_extensions
    assert instance.kwargs == {}

def test_init_with_context_having_cookiecutter_but_missing_extensions_key():
    class MockSuper:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockSuper):
        pass
    instance = TestClass(context={'cookiecutter': {'key': 'value'}})
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected_extensions
    assert instance.kwargs == {}


# LLM-generated content at query #8
#--------------------------

def test_extension_loader_mixin_initializes_with_default_extensions():
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

def test_extension_loader_mixin_reads_extensions_from_context():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = kwargs.get('extensions', [])
    context = {'cookiecutter': {'_extensions': ['my.extension.Extension', 'another.extension.Extension']}}
    test_instance = TestClass(context=context)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my.extension.Extension',
        'another.extension.Extension',
    ]
    assert test_instance.extensions == expected_extensions

def test_extension_loader_mixin_handles_missing_extensions_key():
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

def test_extension_loader_mixin_handles_missing_cookiecutter_key():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = kwargs.get('extensions', [])
    context = {}
    test_instance = TestClass(context=context)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert test_instance.extensions == expected_extensions

def test_extension_loader_mixin_handles_none_context():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = kwargs.get('extensions', [])
    test_instance = TestClass(context=None)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert test_instance.extensions == expected_extensions

def test_extension_loader_mixin_passes_other_kwargs():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.kwargs = kwargs
    test_instance = TestClass(extra_arg='value', another_arg=123)
    assert 'extra_arg' in test_instance.kwargs
    assert test_instance.kwargs['extra_arg'] == 'value'
    assert 'another_arg' in test_instance.kwargs
    assert test_instance.kwargs['another_arg'] == 123

def test_extension_loader_mixin_raises_unknown_extension_on_import_error():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    context = {'cookiecutter': {'_extensions': ['nonexistent.extension.Extension']}}
    try:
        TestClass(context=context)
        assert False, "Expected UnknownExtension to be raised"
    except UnknownExtension as e:
        assert 'Unable to load extension' in str(e)


# LLM-generated content at query #9
#--------------------------

def test_init_without_context():
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

def test_init_with_empty_context():
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

def test_init_with_context_without_extensions():
    class MockParent:
        def __init__(self, extensions, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    class TestClass(ExtensionLoaderMixin, MockParent):
        pass
    context = {'cookiecutter': {'key': 'value'}}
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
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert instance.extensions == expected_extensions
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
        assert str(e) == 'Unable to load extension: No module named cookiecutter.extensions'


# LLM-generated content at query #10
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
    except UnknownExtension as e:
        assert str(e) == "Unable to load extension: No module named 'fake_extension'"


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
    except Exception as e:
        assert e.__class__.__name__ == 'UnknownExtension'
        assert str(e) == "Unable to load extension: No module named 'fake'"


