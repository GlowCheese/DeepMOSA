####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_extension_loader_mixin_init_with_no_context():
    """Test the constructor with no context provided."""
    mixin = ExtensionLoaderMixin()
    assert mixin.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

def test_extension_loader_mixin_init_with_empty_context():
    """Test the constructor with an empty context."""
    mixin = ExtensionLoaderMixin(context={})
    assert mixin.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

def test_extension_loader_mixin_init_with_extensions_in_context():
    """Test the constructor with extensions in the context."""
    context = {
        'cookiecutter': {
            '_extensions': [
                'custom.extension.One',
                'custom.extension.Two',
            ]
        }
    }
    mixin = ExtensionLoaderMixin(context=context)
    assert mixin.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'custom.extension.One',
        'custom.extension.Two',
    ]

def test_extension_loader_mixin_init_with_invalid_extension():
    """Test the constructor with an invalid extension in the context."""
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
        assert str(e) == 'Unable to load extension: No module named \'invalid\''


# LLM-generated content at query #2
#--------------------------

```python
def test_unknown_extension_raised_on_import_error():
    context = {}
    kwargs = {'extensions': ['nonexistent.extension']}

    try:
        ExtensionLoaderMixin.__init__(self=object(), context=context, **kwargs)
    except UnknownExtension:
        pass
    else:
        assert False, "Expected UnknownExtension to be raised"


# LLM-generated content at query #3
#--------------------------

```python
def test_import_error_raises_unknown_extension():
    with pytest.raises(UnknownExtension) as exc_info:
        ExtensionLoaderMixin(context={}, **{'extensions': ['nonexistent.extension']})
    assert str(exc_info.value) == 'Unable to load extension: No module named \'nonexistent\''


# LLM-generated content at query #4
#--------------------------

```python
def test_extension_loader_mixin_init_with_no_context():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    instance = TestClass()
    assert instance.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

def test_extension_loader_mixin_init_with_empty_context():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    instance = TestClass(context={})
    assert instance.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

def test_extension_loader_mixin_init_with_extensions_in_context():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    context = {
        'cookiecutter': {
            '_extensions': [
                'custom.extension1',
                'custom.extension2',
            ]
        }
    }
    instance = TestClass(context=context)
    assert instance.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'custom.extension1',
        'custom.extension2',
    ]

def test_extension_loader_mixin_init_with_invalid_extension():
    class TestClass(ExtensionLoaderMixin):
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
        TestClass(context=context)
        assert False, "Expected UnknownExtension to be raised"
    except UnknownExtension as e:
        assert str(e) == "Unable to load extension: No module named 'invalid'"


# LLM-generated content at query #5
#--------------------------

```python
def test_extension_loader_mixin_init_with_no_context():
    loader = ExtensionLoaderMixin()
    assert loader.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

def test_extension_loader_mixin_init_with_empty_context():
    loader = ExtensionLoaderMixin(context={})
    assert loader.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

def test_extension_loader_mixin_init_with_extensions_in_context():
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


# LLM-generated content at query #6
#--------------------------

```python
def test_import_error_raises_unknown_extension():
    context = {}
    kwargs = {}
    loader = ExtensionLoaderMixin(context=context, **kwargs)
    assert isinstance(loader, ExtensionLoaderMixin)


# LLM-generated content at query #7
#--------------------------

```python
def test_extension_loader_mixin_init_with_no_context():
    loader = ExtensionLoaderMixin()
    assert loader.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

def test_extension_loader_mixin_init_with_empty_context():
    loader = ExtensionLoaderMixin(context={})
    assert loader.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

def test_extension_loader_mixin_init_with_extensions_in_context():
    context = {
        'cookiecutter': {
            '_extensions': [
                'custom.extension1',
                'custom.extension2',
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
        'custom.extension1',
        'custom.extension2',
    ]

def test_extension_loader_mixin_init_with_invalid_extension():
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


# LLM-generated content at query #8
#--------------------------

```python
def test_import_error_raises_unknown_extension():
    context = {'cookiecutter': {'_extensions': ['nonexistent.extension']}}
    loader = ExtensionLoaderMixin(context=context)
    try:
        loader.__init__(context=context)
    except UnknownExtension as e:
        assert str(e) == 'Unable to load extension: No module named \'nonexistent\''


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_constructor_with_no_context():
    loader = ExtensionLoaderMixin()
    assert loader.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

def test_constructor_with_empty_context():
    loader = ExtensionLoaderMixin(context={})
    assert loader.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

def test_constructor_with_extensions_in_context():
    context = {
        'cookiecutter': {
            '_extensions': [
                'custom.extension1',
                'custom.extension2',
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
        'custom.extension1',
        'custom.extension2',
    ]

def test_constructor_with_invalid_extension():
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


# LLM-generated content at query #2
#--------------------------

```python
def test_ImportError_raises_UnknownExtension():
    with raises(ImportError):
        ExtensionLoaderMixin.__init__(
            context={'cookiecutter': {'_extensions': ['invalid.extension']}},
            kwargs={}
        )


# LLM-generated content at query #3
#--------------------------

```python
def test_extension_loader_mixin_init_catches_import_error():
    context = {'cookiecutter': {'_extensions': ['nonexistent.extension']}}
    with pytest.raises(UnknownExtension):
        ExtensionLoaderMixin(context=context)


# LLM-generated content at query #4
#--------------------------

```python
def test_import_error_raises_unknown_extension():
    context = {'cookiecutter': {'_extensions': ['invalid.extension']}}
    kwargs = {}

    try:
        ExtensionLoaderMixin.__init__(self=object(), context=context, **kwargs)
    except UnknownExtension as e:
        assert str(e) == 'Unable to load extension: No module named \'invalid\''
    else:
        assert False, "Expected UnknownExtension to be raised"


# LLM-generated content at query #5
#--------------------------

```python
def test_import_error_raises_unknown_extension():
    context = {'cookiecutter': {'_extensions': ['nonexistent.extension']}}
    with pytest.raises(UnknownExtension):
        ExtensionLoaderMixin(context=context)


# LLM-generated content at query #6
#--------------------------

```python
def test_unknown_extension_raised_when_import_error():
    with pytest.raises(UnknownExtension):
        ExtensionLoaderMixin(context={}, **{'extensions': ['nonexistent_extension']})


# LLM-generated content at query #7
#--------------------------

```python
def test_import_error_raises_unknown_extension():
    context = {}
    kwargs = {}

    with patch.object(ExtensionLoaderMixin, '_read_extensions', return_value=[]):
        with patch('builtins.super') as mock_super:
            mock_super().__init__ = Mock(side_effect=ImportError("test error"))
            try:
                ExtensionLoaderMixin(context=context, **kwargs)
            except UnknownExtension as e:
                assert str(e) == "Unable to load extension: test error"


# LLM-generated content at query #8
#--------------------------

```python
def test_extension_loader_mixin_init_with_no_context():
    """Test that the constructor initializes with default extensions when no context is provided."""
    loader = ExtensionLoaderMixin()
    assert loader.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

def test_extension_loader_mixin_init_with_empty_context():
    """Test that the constructor initializes with default extensions when context is empty."""
    loader = ExtensionLoaderMixin(context={})
    assert loader.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

def test_extension_loader_mixin_init_with_extensions_in_context():
    """Test that the constructor initializes with default extensions plus additional extensions from context."""
    context = {
        'cookiecutter': {
            '_extensions': [
                'custom.extension1',
                'custom.extension2',
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
        'custom.extension1',
        'custom.extension2',
    ]

def test_extension_loader_mixin_init_with_invalid_extension():
    """Test that the constructor raises UnknownExtension when an invalid extension is provided."""
    context = {
        'cookiecutter': {
            '_extensions': [
                'invalid.extension',
            ]
        }
    }
    try:
        ExtensionLoaderMixin(context=context)
    except UnknownExtension as e:
        assert str(e) == 'Unable to load extension: No module named \'invalid\''
    else:
        assert False, "Expected UnknownExtension to be raised"


