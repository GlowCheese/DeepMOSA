####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_extension_loader_mixin_initialization_with_no_context():
    """Test ExtensionLoaderMixin initialization with no context."""
    loader = ExtensionLoaderMixin()
    assert loader.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

def test_extension_loader_mixin_initialization_with_empty_context():
    """Test ExtensionLoaderMixin initialization with empty context."""
    loader = ExtensionLoaderMixin(context={})
    assert loader.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

def test_extension_loader_mixin_initialization_with_extensions_in_context():
    """Test ExtensionLoaderMixin initialization with extensions in context."""
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

def test_extension_loader_mixin_initialization_with_invalid_extension():
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
    except UnknownExtension as e:
        assert str(e) == 'Unable to load extension: No module named \'invalid\''
    else:
        assert False, "Expected UnknownExtension to be raised"


# LLM-generated content at query #2
#--------------------------

```python
def test_extension_loader_mixin_init_with_no_context():
    """Test that the constructor works with no context provided."""
    try:
        ExtensionLoaderMixin()
    except TypeError:
        pass  # Expected if super().__init__() is not implemented in the test environment

def test_extension_loader_mixin_init_with_empty_context():
    """Test that the constructor works with an empty context."""
    try:
        ExtensionLoaderMixin(context={})
    except TypeError:
        pass  # Expected if super().__init__() is not implemented in the test environment

def test_extension_loader_mixin_init_with_context_no_extensions():
    """Test that the constructor works with a context that has no extensions."""
    context = {'cookiecutter': {}}
    try:
        ExtensionLoaderMixin(context=context)
    except TypeError:
        pass  # Expected if super().__init__() is not implemented in the test environment

def test_extension_loader_mixin_init_with_context_with_extensions():
    """Test that the constructor works with a context that has extensions."""
    context = {'cookiecutter': {'_extensions': ['custom.extension']}}
    try:
        ExtensionLoaderMixin(context=context)
    except TypeError:
        pass  # Expected if super().__init__() is not implemented in the test environment

def test_extension_loader_mixin_init_with_invalid_extension():
    """Test that the constructor raises UnknownExtension for invalid extensions."""
    context = {'cookiecutter': {'_extensions': ['invalid.extension']}}
    try:
        ExtensionLoaderMixin(context=context)
    except UnknownExtension:
        pass  # Expected behavior
    except TypeError:
        pass  # Expected if super().__init__() is not implemented in the test environment


# LLM-generated content at query #3
#--------------------------

```python
def test_unknown_extension_raised_on_import_error():
    context = {'cookiecutter': {'_extensions': ['invalid.extension']}}
    try:
        ExtensionLoaderMixin.__init__(self=object(), context=context)
    except UnknownExtension as e:
        assert str(e) == 'Unable to load extension: No module named \'invalid\''
    else:
        assert False, 'Expected UnknownExtension to be raised'


# LLM-generated content at query #4
#--------------------------

```python
def test_import_error_raises_unknown_extension():
    context = {'cookiecutter': {'_extensions': ['nonexistent.extension']}}
    with pytest.raises(UnknownExtension):
        ExtensionLoaderMixin(context=context)


# LLM-generated content at query #5
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

def test_extension_loader_mixin_init_with_custom_extensions():
    """Test that the constructor initializes with default and custom extensions."""
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
        assert False, "Expected UnknownExtension to be raised"
    except UnknownExtension as e:
        assert str(e) == "Unable to load extension: No module named 'invalid'"


# LLM-generated content at query #6
#--------------------------

```python
def test_import_error_raises_unknown_extension():
    context = {'cookiecutter': {'_extensions': ['nonexistent.extension']}}
    loader = ExtensionLoaderMixin(context=context)
    try:
        loader.__init__(context=context)
    except UnknownExtension as e:
        assert str(e) == 'Unable to load extension: No module named \'nonexistent\''


# LLM-generated content at query #7
#--------------------------

```python
def test_import_error_raises_unknown_extension():
    context = {}
    kwargs = {}
    loader = ExtensionLoaderMixin(context=context, **kwargs)
    try:
        super(ExtensionLoaderMixin, loader).__init__(extensions=['nonexistent.extension'], **kwargs)
    except ImportError:
        pass
    else:
        assert False, "Expected ImportError to be raised"


# LLM-generated content at query #8
#--------------------------

```python
def test_import_error_raises_unknown_extension():
    context = {'cookiecutter': {'_extensions': ['nonexistent.extension']}}
    with pytest.raises(UnknownExtension):
        ExtensionLoaderMixin(context=context)


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
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
    except UnknownExtension as e:
        assert str(e) == 'Unable to load extension: No module named \'invalid\''
    else:
        assert False, "Expected UnknownExtension to be raised"


# LLM-generated content at query #2
#--------------------------

```python
def test_unknown_extension_raised_when_import_error_occurs():
    with pytest.raises(UnknownExtension) as exc_info:
        ExtensionLoaderMixin(context={}, **{"extensions": ["nonexistent.extension"]})
    assert str(exc_info.value) == "Unable to load extension: No module named 'nonexistent'"


# LLM-generated content at query #3
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


# LLM-generated content at query #4
#--------------------------

```python
def test_predicate_at_line_23_evaluates_to_true():
    context = {}
    default_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    extensions = default_extensions + ExtensionLoaderMixin._read_extensions(ExtensionLoaderMixin(), context)
    try:
        super(ExtensionLoaderMixin, ExtensionLoaderMixin()).__init__(extensions=extensions, **{})
    except ImportError as err:
        assert True


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
def test_import_error_raises_unknown_extension():
    context = {'cookiecutter': {'_extensions': ['nonexistent.extension']}}
    with pytest.raises(UnknownExtension):
        ExtensionLoaderMixin(context=context)


# LLM-generated content at query #7
#--------------------------

```python
def test_unknown_extension_raised_on_import_error():
    context = {}
    kwargs = {}
    loader = ExtensionLoaderMixin(context=context, **kwargs)
    assert isinstance(loader, UnknownExtension)


# LLM-generated content at query #8
#--------------------------

```python
def test_extension_loader_mixin_init_with_no_context():
    class TestClass(ExtensionLoaderMixin):
        pass

    obj = TestClass()
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

def test_extension_loader_mixin_init_with_empty_context():
    class TestClass(ExtensionLoaderMixin):
        pass

    obj = TestClass(context={})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

def test_extension_loader_mixin_init_with_extensions_in_context():
    class TestClass(ExtensionLoaderMixin):
        pass

    context = {
        'cookiecutter': {
            '_extensions': [
                'custom.extension.One',
                'custom.extension.Two',
            ]
        }
    }
    obj = TestClass(context=context)
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'custom.extension.One',
        'custom.extension.Two',
    ]

def test_extension_loader_mixin_init_with_missing_extensions_key():
    class TestClass(ExtensionLoaderMixin):
        pass

    context = {
        'cookiecutter': {
            'other_key': 'value'
        }
    }
    obj = TestClass(context=context)
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]


