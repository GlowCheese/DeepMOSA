####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_init_with_no_context_and_no_extensions():
    """Test that the constructor initializes with default extensions when no context is provided."""
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    loader = TestExtensionLoader()
    assert loader.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

def test_init_with_empty_context():
    """Test that the constructor initializes with default extensions when context is empty."""
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    loader = TestExtensionLoader(context={})
    assert loader.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

def test_init_with_context_containing_extensions():
    """Test that the constructor initializes with default and additional extensions from context."""
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    context = {
        'cookiecutter': {
            '_extensions': [
                'custom.extension.One',
                'custom.extension.Two',
            ]
        }
    }
    loader = TestExtensionLoader(context=context)
    assert loader.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'custom.extension.One',
        'custom.extension.Two',
    ]

def test_init_with_invalid_extension_raises_unknown_extension():
    """Test that the constructor raises UnknownExtension when an invalid extension is provided."""
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    context = {
        'cookiecutter': {
            '_extensions': [
                'invalid.extension',
            ]
        }
    }
    try:
        TestExtensionLoader(context=context)
        assert False, "Expected UnknownExtension to be raised"
    except UnknownExtension as e:
        assert str(e) == "Unable to load extension: No module named 'invalid'"


# LLM-generated content at query #2
#--------------------------

```python
def test_extension_loader_mixin_init_with_no_context():
    """Test ExtensionLoaderMixin initialization with no context."""
    loader = ExtensionLoaderMixin()
    assert loader.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

def test_extension_loader_mixin_init_with_empty_context():
    """Test ExtensionLoaderMixin initialization with empty context."""
    loader = ExtensionLoaderMixin(context={})
    assert loader.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

def test_extension_loader_mixin_init_with_extensions_in_context():
    """Test ExtensionLoaderMixin initialization with extensions in context."""
    context = {
        'cookiecutter': {
            '_extensions': [
                'custom.extension.One',
                'custom.extension.Two',
            ]
        }
    }
    loader = ExtensionLoaderMixin(context=context)
    assert loader.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'custom.extension.One',
        'custom.extension.Two',
    ]

def test_extension_loader_mixin_init_with_invalid_extension():
    """Test ExtensionLoaderMixin initialization with invalid extension."""
    context = {
        'cookiecutter': {
            '_extensions': [
                'invalid.extension',
            ]
        }
    }
    try:
        ExtensionLoaderMixin(context=context)
        assert False, "Expected UnknownExtension to be raised"
    except UnknownExtension as e:
        assert str(e) == "Unable to load extension: No module named 'invalid'"


# LLM-generated content at query #3
#--------------------------

```python
def test_import_error_raises_unknown_extension():
    context = {}
    kwargs = {}

    loader = ExtensionLoaderMixin(context=context, **kwargs)

    try:
        super(ExtensionLoaderMixin, loader).__init__(extensions=['nonexistent.extension'], **kwargs)
    except ImportError as err:
        assert isinstance(err, UnknownExtension)
        assert str(err) == 'Unable to load extension: No module named \'nonexistent\''


# LLM-generated content at query #4
#--------------------------

```python
def test_import_error_raises_unknown_extension():
    context = {}
    kwargs = {}

    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            raise ImportError("Test error")

    with pytest.raises(UnknownExtension) as exc_info:
        TestClass(context=context, **kwargs)

    assert str(exc_info.value) == "Unable to load extension: Test error"


# LLM-generated content at query #5
#--------------------------

```python
def test_import_error_raises_unknown_extension():
    context = {}
    kwargs = {}
    loader = ExtensionLoaderMixin(context=context, **kwargs)
    with pytest.raises(UnknownExtension):
        loader.__init__(context=context, **kwargs)


# LLM-generated content at query #6
#--------------------------

```python
def test_import_error_raises_unknown_extension():
    context = {'cookiecutter': {'_extensions': ['nonexistent.extension']}}
    with pytest.raises(UnknownExtension):
        ExtensionLoaderMixin.__init__(context=context)


# LLM-generated content at query #7
#--------------------------

```python
def test_extension_loader_mixin_init_with_no_context():
    """Test that ExtensionLoaderMixin initializes with default extensions when no context is provided."""
    mixin = ExtensionLoaderMixin()
    assert mixin.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

def test_extension_loader_mixin_init_with_empty_context():
    """Test that ExtensionLoaderMixin initializes with default extensions when context is empty."""
    mixin = ExtensionLoaderMixin(context={})
    assert mixin.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

def test_extension_loader_mixin_init_with_context_no_extensions():
    """Test that ExtensionLoaderMixin initializes with default extensions when context has no _extensions key."""
    context = {'cookiecutter': {'some_key': 'some_value'}}
    mixin = ExtensionLoaderMixin(context=context)
    assert mixin.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

def test_extension_loader_mixin_init_with_context_with_extensions():
    """Test that ExtensionLoaderMixin initializes with default and additional extensions from context."""
    context = {'cookiecutter': {'_extensions': ['custom.extension1', 'custom.extension2']}}
    mixin = ExtensionLoaderMixin(context=context)
    assert mixin.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'custom.extension1',
        'custom.extension2',
    ]

def test_extension_loader_mixin_init_raises_unknown_extension():
    """Test that ExtensionLoaderMixin raises UnknownExtension when an extension cannot be loaded."""
    context = {'cookiecutter': {'_extensions': ['nonexistent.extension']}}
    try:
        ExtensionLoaderMixin(context=context)
        assert False, "Expected UnknownExtension to be raised"
    except UnknownExtension as e:
        assert str(e) == "Unable to load extension: No module named 'nonexistent'"


# LLM-generated content at query #8
#--------------------------

```python
def test_unknown_extension_raised_on_import_error():
    context = {}
    kwargs = {}

    with patch.object(ExtensionLoaderMixin, '_read_extensions', return_value=[]):
        with patch('builtins.super') as mock_super:
            mock_super().__init__.side_effect = ImportError("Test error")

            with pytest.raises(UnknownExtension) as exc_info:
                ExtensionLoaderMixin(context=context, **kwargs)

            assert str(exc_info.value) == "Unable to load extension: Test error"


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_extension_loader_mixin_init_with_no_context():
    """Test that the constructor initializes with default extensions when no context is provided."""
    instance = ExtensionLoaderMixin()
    assert hasattr(instance, 'extensions')
    assert 'cookiecutter.extensions.JsonifyExtension' in instance.extensions
    assert 'cookiecutter.extensions.RandomStringExtension' in instance.extensions
    assert 'cookiecutter.extensions.SlugifyExtension' in instance.extensions
    assert 'cookiecutter.extensions.TimeExtension' in instance.extensions
    assert 'cookiecutter.extensions.UUIDExtension' in instance.extensions

def test_extension_loader_mixin_init_with_empty_context():
    """Test that the constructor initializes with default extensions when context is empty."""
    instance = ExtensionLoaderMixin(context={})
    assert hasattr(instance, 'extensions')
    assert 'cookiecutter.extensions.JsonifyExtension' in instance.extensions
    assert 'cookiecutter.extensions.RandomStringExtension' in instance.extensions
    assert 'cookiecutter.extensions.SlugifyExtension' in instance.extensions
    assert 'cookiecutter.extensions.TimeExtension' in instance.extensions
    assert 'cookiecutter.extensions.UUIDExtension' in instance.extensions

def test_extension_loader_mixin_init_with_custom_extensions():
    """Test that the constructor initializes with default and custom extensions when provided in context."""
    context = {
        'cookiecutter': {
            '_extensions': [
                'custom.extension.One',
                'custom.extension.Two'
            ]
        }
    }
    instance = ExtensionLoaderMixin(context=context)
    assert hasattr(instance, 'extensions')
    assert 'cookiecutter.extensions.JsonifyExtension' in instance.extensions
    assert 'cookiecutter.extensions.RandomStringExtension' in instance.extensions
    assert 'cookiecutter.extensions.SlugifyExtension' in instance.extensions
    assert 'cookiecutter.extensions.TimeExtension' in instance.extensions
    assert 'cookiecutter.extensions.UUIDExtension' in instance.extensions
    assert 'custom.extension.One' in instance.extensions
    assert 'custom.extension.Two' in instance.extensions

def test_extension_loader_mixin_init_with_invalid_extension():
    """Test that the constructor raises UnknownExtension when an invalid extension is provided."""
    context = {
        'cookiecutter': {
            '_extensions': [
                'invalid.extension'
            ]
        }
    }
    try:
        ExtensionLoaderMixin(context=context)
        assert False, "Expected UnknownExtension to be raised"
    except UnknownExtension as e:
        assert "Unable to load extension" in str(e)


# LLM-generated content at query #2
#--------------------------

```python
def test_unknown_extension_raised_on_import_error():
    context = {'cookiecutter': {'_extensions': ['nonexistent.extension']}}
    kwargs = {}
    loader = ExtensionLoaderMixin(context=context, **kwargs)
    assert isinstance(loader, ExtensionLoaderMixin)


# LLM-generated content at query #3
#--------------------------

```python
def test_extension_loader_mixin_init_with_no_context():
    """Test that the constructor works with no context provided."""
    loader = ExtensionLoaderMixin(context=None)
    assert loader.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

def test_extension_loader_mixin_init_with_empty_context():
    """Test that the constructor works with an empty context."""
    loader = ExtensionLoaderMixin(context={})
    assert loader.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

def test_extension_loader_mixin_init_with_extensions_in_context():
    """Test that the constructor loads extensions from context."""
    context = {
        'cookiecutter': {
            '_extensions': [
                'custom.extension.One',
                'custom.extension.Two',
            ]
        }
    }
    loader = ExtensionLoaderMixin(context=context)
    assert loader.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'custom.extension.One',
        'custom.extension.Two',
    ]

def test_extension_loader_mixin_init_with_invalid_extension():
    """Test that the constructor raises UnknownExtension for invalid extensions."""
    context = {
        'cookiecutter': {
            '_extensions': ['invalid.extension']
        }
    }
    try:
        ExtensionLoaderMixin(context=context)
    except UnknownExtension as e:
        assert str(e) == 'Unable to load extension: No module named \'invalid\''
    else:
        assert False, "Expected UnknownExtension to be raised"


