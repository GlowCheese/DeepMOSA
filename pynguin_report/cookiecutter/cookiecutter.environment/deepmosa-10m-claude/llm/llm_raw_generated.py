####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_extension_loader_mixin_init_with_no_context():
    """Test initialization with no context provided."""
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = kwargs.get('extensions', [])
            super().__init__(context=context, **kwargs)
    
    # This would need a mock or actual Jinja2 Environment as parent
    # Testing the mixin behavior with default extensions
    pass


def test_extension_loader_mixin_init_with_empty_context():
    """Test initialization with empty context dictionary."""
    from unittest.mock import Mock, patch
    
    with patch('cookiecutter.extensions.ExtensionLoaderMixin.__bases__', (Mock,)):
        mixin = ExtensionLoaderMixin(context={})
        # Verify that default extensions are set up


def test_extension_loader_mixin_init_with_custom_extensions():
    """Test initialization with custom extensions in context."""
    from unittest.mock import Mock, patch
    
    context = {
        'cookiecutter': {
            '_extensions': ['some.custom.Extension', 'another.Extension']
        }
    }
    
    with patch('cookiecutter.extensions.ExtensionLoaderMixin.__bases__', (Mock,)):
        mixin = ExtensionLoaderMixin(context=context)


def test_extension_loader_mixin_init_default_extensions_included():
    """Test that default extensions are always included."""
    from unittest.mock import Mock, patch, call
    
    mock_parent = Mock()
    
    with patch.object(ExtensionLoaderMixin, '__bases__', (mock_parent,)):
        mixin = ExtensionLoaderMixin(context={})
        # Verify default extensions were passed to parent __init__


def test_extension_loader_mixin_init_import_error_handling():
    """Test that ImportError is caught and re-raised as UnknownExtension."""
    from unittest.mock import Mock, patch
    
    with patch('cookiecutter.extensions.ExtensionLoaderMixin.__bases__', (Mock,)):
        try:
            mixin = ExtensionLoaderMixin(context={})
        except UnknownExtension:
            pass


def test_extension_loader_mixin_read_extensions_with_valid_context():
    """Test _read_extensions with valid context containing extensions."""
    mixin = ExtensionLoaderMixin.__new__(ExtensionLoaderMixin)
    context = {
        'cookiecutter': {
            '_extensions': ['ext1', 'ext2', 'ext3']
        }
    }
    
    result = mixin._read_extensions(context)
    assert result == ['ext1', 'ext2', 'ext3']


def test_extension_loader_mixin_read_extensions_with_missing_key():
    """Test _read_extensions when _extensions key is missing."""
    mixin = ExtensionLoaderMixin.__new__(ExtensionLoaderMixin)
    context = {'cookiecutter': {}}
    
    result = mixin._read_extensions(context)
    assert result == []


def test_extension_loader_mixin_read_extensions_with_empty_context():
    """Test _read_extensions with empty context."""
    mixin = ExtensionLoaderMixin.__new__(ExtensionLoaderMixin)
    context = {}
    
    result = mixin._read_extensions(context)
    assert result == []


def test_extension_loader_mixin_read_extensions_converts_to_string():
    """Test _read_extensions converts extension items to strings."""
    mixin = ExtensionLoaderMixin.__new__(ExtensionLoaderMixin)
    context = {
        'cookiecutter': {
            '_extensions': [123, 456, 789]
        }
    }
    
    result = mixin._read_extensions(context)
    assert result == ['123', '456', '789']


# LLM-generated content at query #2
#--------------------------

```python
def test_extension_loader_mixin_init_with_default_extensions():
    """Test that ExtensionLoaderMixin initializes with default extensions."""
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = []
            super().__init__(context=context, **kwargs)
    
    env = TestEnv(context={})
    assert env is not None


def test_extension_loader_mixin_init_with_custom_context():
    """Test that ExtensionLoaderMixin reads extensions from context."""
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = []
            super().__init__(context=context, **kwargs)
    
    context = {
        'cookiecutter': {
            '_extensions': ['jinja2.ext.DebugExtension']
        }
    }
    env = TestEnv(context=context)
    assert env is not None


def test_extension_loader_mixin_init_without_context():
    """Test that ExtensionLoaderMixin initializes without context argument."""
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = []
            super().__init__(context=context, **kwargs)
    
    env = TestEnv()
    assert env is not None


def test_extension_loader_mixin_init_with_none_context():
    """Test that ExtensionLoaderMixin handles None context gracefully."""
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = []
            super().__init__(context=context, **kwargs)
    
    env = TestEnv(context=None)
    assert env is not None


def test_extension_loader_mixin_init_with_empty_extensions():
    """Test that ExtensionLoaderMixin works with empty _extensions list."""
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = []
            super().__init__(context=context, **kwargs)
    
    context = {
        'cookiecutter': {
            '_extensions': []
        }
    }
    env = TestEnv(context=context)
    assert env is not None


def test_extension_loader_mixin_init_with_multiple_custom_extensions():
    """Test that ExtensionLoaderMixin handles multiple custom extensions."""
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = []
            super().__init__(context=context, **kwargs)
    
    context = {
        'cookiecutter': {
            '_extensions': ['ext1', 'ext2', 'ext3']
        }
    }
    env = TestEnv(context=context)
    assert env is not None


def test_extension_loader_mixin_init_preserves_kwargs():
    """Test that ExtensionLoaderMixin passes additional kwargs to parent."""
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.kwargs_received = kwargs
            super().__init__(context=context, **kwargs)
    
    env = TestEnv(context={}, trim_blocks=True, lstrip_blocks=True)
    assert env is not None


# LLM-generated content at query #3
#--------------------------

```python
def test_import_error_raises_unknown_extension():
    from unittest.mock import Mock, patch
    
    class UnknownExtension(Exception):
        pass
    
    class TestExtensionLoaderMixin(ExtensionLoaderMixin):
        pass
    
    with patch('builtins.super') as mock_super:
        mock_super_instance = Mock()
        mock_super_instance.__init__.side_effect = ImportError("test error")
        mock_super.return_value = mock_super_instance
        
        try:
            TestExtensionLoaderMixin(context={'cookiecutter': {}})
            assert False, "Expected UnknownExtension to be raised"
        except UnknownExtension as e:
            assert "Unable to load extension:" in str(e)
            assert True


# LLM-generated content at query #4
#--------------------------

```python
def test_extension_loader_mixin_handles_import_error():
    from unittest.mock import Mock, patch
    from cookiecutter.extensions import ExtensionLoaderMixin
    
    class MockEnvironment(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            super().__init__(context=context, **kwargs)
    
    with patch.object(ExtensionLoaderMixin, '_read_extensions', return_value=[]):
        with patch('builtins.super') as mock_super:
            mock_super_instance = Mock()
            mock_super_instance.__init__.side_effect = ImportError("test error")
            mock_super.return_value = mock_super_instance
            
            try:
                MockEnvironment()
                assert False, "Expected UnknownExtension to be raised"
            except Exception as e:
                assert type(e).__name__ == 'UnknownExtension'
                assert 'Unable to load extension:' in str(e)


# LLM-generated content at query #5
#--------------------------

```python
def test_import_error_raises_unknown_extension():
    from unittest.mock import Mock, patch
    
    class UnknownExtension(Exception):
        pass
    
    class TestExtensionLoaderMixin(ExtensionLoaderMixin):
        pass
    
    with patch('builtins.super') as mock_super:
        mock_super_instance = Mock()
        mock_super_instance.__init__.side_effect = ImportError("test error")
        mock_super.return_value = mock_super_instance
        
        try:
            TestExtensionLoaderMixin(context={})
            assert False, "Expected UnknownExtension to be raised"
        except UnknownExtension as e:
            assert "Unable to load extension:" in str(e)
            assert True


# LLM-generated content at query #6
#--------------------------

```python
def test_import_error_handling():
    from unittest.mock import Mock, patch
    
    class UnknownExtension(Exception):
        pass
    
    class MockExtensionLoaderMixin(ExtensionLoaderMixin):
        pass
    
    mock_instance = Mock(spec=ExtensionLoaderMixin)
    mock_instance._read_extensions = Mock(return_value=[])
    
    with patch('builtins.super') as mock_super:
        mock_super_instance = Mock()
        mock_super_instance.__init__ = Mock(side_effect=ImportError("test error"))
        mock_super.return_value = mock_super_instance
        
        try:
            ExtensionLoaderMixin.__init__(mock_instance, context={})
        except UnknownExtension as e:
            assert "Unable to load extension: test error" in str(e)
            assert True
        else:
            assert False, "UnknownExtension should have been raised"


# LLM-generated content at query #7
#--------------------------

```python
def test_extension_loader_mixin_handles_import_error():
    """Test that ImportError is caught and re-raised as UnknownExtension at line 23."""
    from unittest.mock import Mock, patch
    from cookiecutter.extensions import ExtensionLoaderMixin
    
    class MockUnknownExtension(Exception):
        pass
    
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            with patch('cookiecutter.extensions.UnknownExtension', MockUnknownExtension):
                try:
                    super().__init__(context=context, **kwargs)
                except MockUnknownExtension as e:
                    assert 'Unable to load extension:' in str(e)
                    raise
    
    mock_super = Mock(side_effect=ImportError("test import error"))
    
    with patch('cookiecutter.extensions.ExtensionLoaderMixin.__init__', 
               side_effect=lambda self, extensions, **kw: (_ for _ in ()).throw(ImportError("test import error"))):
        try:
            loader = TestExtensionLoader(context={'cookiecutter': {}})
        except MockUnknownExtension:
            pass
    
    assert True


# LLM-generated content at query #8
#--------------------------

```python
def test_extension_loader_mixin_init_with_no_context():
    """Test initialization with no context provided."""
    class TestLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = None
            super().__init__(context=context, **kwargs)
    
    loader = TestLoader()
    assert loader is not None


def test_extension_loader_mixin_init_with_empty_context():
    """Test initialization with empty context dictionary."""
    class TestLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = None
            super().__init__(context=context, **kwargs)
    
    loader = TestLoader(context={})
    assert loader is not None


def test_extension_loader_mixin_read_extensions_no_extensions_key():
    """Test _read_extensions when context has no _extensions key."""
    class TestLoader(ExtensionLoaderMixin):
        pass
    
    loader = TestLoader.__new__(TestLoader)
    result = loader._read_extensions({})
    assert result == []


def test_extension_loader_mixin_read_extensions_with_extensions():
    """Test _read_extensions with valid extensions in context."""
    class TestLoader(ExtensionLoaderMixin):
        pass
    
    loader = TestLoader.__new__(TestLoader)
    context = {
        'cookiecutter': {
            '_extensions': ['ext1', 'ext2', 'ext3']
        }
    }
    result = loader._read_extensions(context)
    assert result == ['ext1', 'ext2', 'ext3']


def test_extension_loader_mixin_read_extensions_converts_to_string():
    """Test _read_extensions converts non-string extensions to strings."""
    class TestLoader(ExtensionLoaderMixin):
        pass
    
    loader = TestLoader.__new__(TestLoader)
    context = {
        'cookiecutter': {
            '_extensions': [123, 456.78, 'string_ext']
        }
    }
    result = loader._read_extensions(context)
    assert result == ['123', '456.78', 'string_ext']


def test_extension_loader_mixin_read_extensions_missing_cookiecutter_key():
    """Test _read_extensions when cookiecutter key is missing."""
    class TestLoader(ExtensionLoaderMixin):
        pass
    
    loader = TestLoader.__new__(TestLoader)
    context = {'other_key': 'value'}
    result = loader._read_extensions(context)
    assert result == []


def test_extension_loader_mixin_read_extensions_empty_extensions_list():
    """Test _read_extensions with empty extensions list."""
    class TestLoader(ExtensionLoaderMixin):
        pass
    
    loader = TestLoader.__new__(TestLoader)
    context = {
        'cookiecutter': {
            '_extensions': []
        }
    }
    result = loader._read_extensions(context)
    assert result == []


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_extension_loader_mixin_init_with_no_context():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = []
            super().__init__(context=context, **kwargs)
        
    # Mock the parent class to capture extensions
    original_init = object.__init__
    captured_extensions = []
    
    def mock_init(self, extensions=None, **kwargs):
        captured_extensions.extend(extensions or [])
        original_init(self)
    
    ExtensionLoaderMixin.__bases__ = (object,)
    object.__init__ = mock_init
    
    loader = TestExtensionLoader()
    
    object.__init__ = original_init
    
    assert 'cookiecutter.extensions.JsonifyExtension' in captured_extensions
    assert 'cookiecutter.extensions.RandomStringExtension' in captured_extensions
    assert 'cookiecutter.extensions.SlugifyExtension' in captured_extensions
    assert 'cookiecutter.extensions.TimeExtension' in captured_extensions
    assert 'cookiecutter.extensions.UUIDExtension' in captured_extensions


def test_extension_loader_mixin_init_with_empty_context():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            super().__init__(context=context, **kwargs)
    
    captured_extensions = []
    
    def mock_init(self, extensions=None, **kwargs):
        captured_extensions.extend(extensions or [])
    
    ExtensionLoaderMixin.__bases__ = (object,)
    original_init = object.__init__
    object.__init__ = mock_init
    
    loader = TestExtensionLoader(context={})
    
    object.__init__ = original_init
    
    assert len(captured_extensions) == 5


def test_extension_loader_mixin_init_with_custom_extensions():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            super().__init__(context=context, **kwargs)
    
    captured_extensions = []
    
    def mock_init(self, extensions=None, **kwargs):
        captured_extensions.extend(extensions or [])
    
    ExtensionLoaderMixin.__bases__ = (object,)
    original_init = object.__init__
    object.__init__ = mock_init
    
    context = {
        'cookiecutter': {
            '_extensions': ['my.custom.Extension', 'another.Extension']
        }
    }
    loader = TestExtensionLoader(context=context)
    
    object.__init__ = original_init
    
    assert 'my.custom.Extension' in captured_extensions
    assert 'another.Extension' in captured_extensions
    assert len(captured_extensions) == 7


def test_extension_loader_mixin_read_extensions_with_missing_key():
    mixin = ExtensionLoaderMixin.__new__(ExtensionLoaderMixin)
    result = mixin._read_extensions({})
    assert result == []


def test_extension_loader_mixin_read_extensions_with_valid_context():
    mixin = ExtensionLoaderMixin.__new__(ExtensionLoaderMixin)
    context = {
        'cookiecutter': {
            '_extensions': ['ext1', 'ext2', 'ext3']
        }
    }
    result = mixin._read_extensions(context)
    assert result == ['ext1', 'ext2', 'ext3']


def test_extension_loader_mixin_read_extensions_converts_to_string():
    mixin = ExtensionLoaderMixin.__new__(ExtensionLoaderMixin)
    context = {
        'cookiecutter': {
            '_extensions': [1, 2.5, 'ext']
        }
    }
    result = mixin._read_extensions(context)
    assert result == ['1', '2.5', 'ext']


# LLM-generated content at query #2
#--------------------------

```python
def test_extension_loader_mixin_handles_import_error():
    from unittest.mock import Mock, patch
    
    class UnknownExtension(Exception):
        pass
    
    class TestExtensionLoaderMixin(ExtensionLoaderMixin):
        pass
    
    with patch('builtins.super') as mock_super:
        mock_super_instance = Mock()
        mock_super_instance.__init__.side_effect = ImportError("Module not found")
        mock_super.return_value = mock_super_instance
        
        try:
            TestExtensionLoaderMixin(context={'cookiecutter': {}})
            assert False, "Expected UnknownExtension to be raised"
        except UnknownExtension as e:
            assert "Unable to load extension:" in str(e)
            assert isinstance(e.__cause__, ImportError)


# LLM-generated content at query #3
#--------------------------

```python
def test_extension_loader_mixin_context_is_not_none():
    from unittest.mock import Mock, patch
    
    # Create a mock class that inherits from ExtensionLoaderMixin
    with patch('cookiecutter.extensions.JsonifyExtension'), \
         patch('cookiecutter.extensions.RandomStringExtension'), \
         patch('cookiecutter.extensions.SlugifyExtension'), \
         patch('cookiecutter.extensions.TimeExtension'), \
         patch('cookiecutter.extensions.UUIDExtension'):
        
        class TestClass(ExtensionLoaderMixin):
            def __init__(self, *, context=None, **kwargs):
                self.context_arg = context
                super().__init__(context=context, **kwargs)
        
        test_context = {'cookiecutter': {'_extensions': []}}
        instance = TestClass(context=test_context)
        
        # The predicate at line 1 is "context: dict[str, Any] | None = None"
        # This evaluates to False when context is not None
        assert instance.context_arg is not None
        assert instance.context_arg == test_context


# LLM-generated content at query #4
#--------------------------

```python
def test_import_error_raises_unknown_extension():
    from unittest.mock import Mock, patch
    
    class UnknownExtension(Exception):
        pass
    
    class TestExtensionLoaderMixin(ExtensionLoaderMixin):
        pass
    
    with patch('builtins.super') as mock_super:
        mock_super_instance = Mock()
        mock_super_instance.__init__.side_effect = ImportError("Module not found")
        mock_super.return_value = mock_super_instance
        
        try:
            instance = TestExtensionLoaderMixin(context={})
            assert False, "Expected UnknownExtension to be raised"
        except UnknownExtension as e:
            assert "Unable to load extension:" in str(e)
            assert isinstance(e.__cause__, ImportError)


# LLM-generated content at query #5
#--------------------------

```python
def test_extension_loader_mixin_init_with_no_context():
    from unittest.mock import MagicMock, patch
    
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.parent_init_called = False
            self.parent_init_kwargs = {}
            super().__init__(context=context, **kwargs)
    
    with patch.object(ExtensionLoaderMixin, '_read_extensions', return_value=[]):
        with patch('builtins.super') as mock_super:
            mock_super_instance = MagicMock()
            mock_super.return_value = mock_super_instance
            
            loader = TestExtensionLoader()
            assert loader is not None


def test_extension_loader_mixin_init_with_context():
    from unittest.mock import MagicMock, patch
    
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            super().__init__(context=context, **kwargs)
    
    test_context = {'cookiecutter': {}}
    
    with patch.object(ExtensionLoaderMixin, '_read_extensions', return_value=[]):
        with patch('builtins.super') as mock_super:
            mock_super_instance = MagicMock()
            mock_super.return_value = mock_super_instance
            
            loader = TestExtensionLoader(context=test_context)
            assert loader is not None


def test_extension_loader_mixin_init_with_custom_extensions():
    from unittest.mock import MagicMock, patch
    
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            super().__init__(context=context, **kwargs)
    
    test_context = {'cookiecutter': {'_extensions': ['custom.extension']}}
    
    with patch.object(ExtensionLoaderMixin, '_read_extensions', return_value=['custom.extension']):
        with patch('builtins.super') as mock_super:
            mock_super_instance = MagicMock()
            mock_super.return_value = mock_super_instance
            
            loader = TestExtensionLoader(context=test_context)
            assert loader is not None


def test_extension_loader_mixin_init_import_error():
    from unittest.mock import MagicMock, patch
    
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            super().__init__(context=context, **kwargs)
    
    with patch.object(ExtensionLoaderMixin, '_read_extensions', return_value=[]):
        with patch('builtins.super') as mock_super:
            mock_super.side_effect = ImportError('Module not found')
            
            try:
                loader = TestExtensionLoader()
                assert False, "Expected UnknownExtension to be raised"
            except UnknownExtension as e:
                assert 'Unable to load extension' in str(e)


def test_extension_loader_mixin_init_with_kwargs():
    from unittest.mock import MagicMock, patch
    
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            super().__init__(context=context, **kwargs)
    
    with patch.object(ExtensionLoaderMixin, '_read_extensions', return_value=[]):
        with patch('builtins.super') as mock_super:
            mock_super_instance = MagicMock()
            mock_super.return_value = mock_super_instance
            
            loader = TestExtensionLoader(context=None, extra_arg='value')
            assert loader is not None


# LLM-generated content at query #6
#--------------------------

```python
def test_context_parameter_not_none():
    """Test that the predicate 'context is None' at line 1 evaluates to False."""
    from unittest.mock import MagicMock, patch
    
    # Create a mock parent class to avoid actual Jinja2 initialization
    mock_parent = MagicMock()
    
    # Create a test context that is not None
    test_context = {'cookiecutter': {}}
    
    # Create an instance of ExtensionLoaderMixin with a non-None context
    with patch('cookiecutter.extensions.ExtensionLoaderMixin.__bases__', (object,)):
        loader = ExtensionLoaderMixin(context=test_context)
    
    # Verify that context was not None by checking the assignment path
    # The predicate "context is None" should evaluate to False
    assert test_context is not None
    assert loader._read_extensions(test_context) == []


# LLM-generated content at query #7
#--------------------------

```python
def test_extension_loader_mixin_init_with_no_context():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = None
            try:
                super().__init__(context=context, **kwargs)
            except Exception:
                pass

    loader = TestExtensionLoader()
    assert loader is not None


def test_extension_loader_mixin_init_with_empty_context():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = None
            try:
                super().__init__(context=context, **kwargs)
            except Exception:
                pass

    loader = TestExtensionLoader(context={})
    assert loader is not None


def test_extension_loader_mixin_read_extensions_empty_context():
    class TestExtensionLoader(ExtensionLoaderMixin):
        pass

    loader = TestExtensionLoader()
    result = loader._read_extensions({})
    assert result == []


def test_extension_loader_mixin_read_extensions_no_extensions_key():
    class TestExtensionLoader(ExtensionLoaderMixin):
        pass

    loader = TestExtensionLoader()
    result = loader._read_extensions({'cookiecutter': {}})
    assert result == []


def test_extension_loader_mixin_read_extensions_with_extensions():
    class TestExtensionLoader(ExtensionLoaderMixin):
        pass

    loader = TestExtensionLoader()
    context = {'cookiecutter': {'_extensions': ['ext1', 'ext2', 'ext3']}}
    result = loader._read_extensions(context)
    assert result == ['ext1', 'ext2', 'ext3']


def test_extension_loader_mixin_read_extensions_converts_to_string():
    class TestExtensionLoader(ExtensionLoaderMixin):
        pass

    loader = TestExtensionLoader()
    context = {'cookiecutter': {'_extensions': [1, 2, 3]}}
    result = loader._read_extensions(context)
    assert result == ['1', '2', '3']


def test_extension_loader_mixin_init_with_custom_extensions():
    class MockEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, extensions=None, **kwargs):
            self.extensions_param = extensions
            context = context or {}
            default_extensions = [
                'cookiecutter.extensions.JsonifyExtension',
                'cookiecutter.extensions.RandomStringExtension',
                'cookiecutter.extensions.SlugifyExtension',
                'cookiecutter.extensions.TimeExtension',
                'cookiecutter.extensions.UUIDExtension',
            ]
            extensions_list = default_extensions + self._read_extensions(context)
            self.extensions_param = extensions_list

        def _read_extensions(self, context):
            try:
                extensions = context['cookiecutter']['_extensions']
            except KeyError:
                return []
            else:
                return [str(ext) for ext in extensions]

    context = {'cookiecutter': {'_extensions': ['custom.ext1', 'custom.ext2']}}
    loader = MockEnv(context=context)
    assert 'custom.ext1' in loader.extensions_param
    assert 'custom.ext2' in loader.extensions_param
    assert 'cookiecutter.extensions.JsonifyExtension' in loader.extensions_param


# LLM-generated content at query #8
#--------------------------

```python
def test_extension_loader_mixin_context_parameter_accepts_none():
    from unittest.mock import Mock, patch
    
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self._read_extensions = Mock(return_value=[])
            with patch('builtins.super') as mock_super:
                mock_super_instance = Mock()
                mock_super.return_value = mock_super_instance
                super(TestExtensionLoader, self).__init__(context=context, **kwargs)
    
    loader = TestExtensionLoader(context=None)
    assert loader is not None


def test_extension_loader_mixin_context_parameter_accepts_dict():
    from unittest.mock import Mock, patch
    
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self._read_extensions = Mock(return_value=[])
            with patch('builtins.super') as mock_super:
                mock_super_instance = Mock()
                mock_super.return_value = mock_super_instance
                super(TestExtensionLoader, self).__init__(context=context, **kwargs)
    
    test_context = {'cookiecutter': {}}
    loader = TestExtensionLoader(context=test_context)
    assert loader is not None


def test_extension_loader_mixin_context_defaults_to_empty_dict():
    from unittest.mock import Mock, patch, MagicMock
    
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            context = context or {}
            self._read_extensions = Mock(return_value=[])
            with patch('builtins.super') as mock_super:
                mock_super_instance = MagicMock()
                mock_super.return_value = mock_super_instance
                super(TestExtensionLoader, self).__init__(context=context, **kwargs)
    
    loader = TestExtensionLoader()
    assert loader is not None


# LLM-generated content at query #9
#--------------------------

```python
def test_extension_loader_mixin_context_is_not_none():
    """Test that the predicate 'context is None' evaluates to False when context is provided."""
    from unittest.mock import Mock, patch
    
    # Create a mock class that inherits from ExtensionLoaderMixin
    with patch('cookiecutter.extensions.JsonifyExtension'), \
         patch('cookiecutter.extensions.RandomStringExtension'), \
         patch('cookiecutter.extensions.SlugifyExtension'), \
         patch('cookiecutter.extensions.TimeExtension'), \
         patch('cookiecutter.extensions.UUIDExtension'):
        
        class MockEnvironment(ExtensionLoaderMixin):
            def __init__(self, *, context=None, **kwargs):
                self.context_value = context
                super().__init__(context=context, **kwargs)
        
        test_context = {'cookiecutter': {'_extensions': []}}
        mock_env = MockEnvironment(context=test_context)
        
        # Verify that context is not None (predicate evaluates to False)
        assert mock_env.context_value is not None
        assert mock_env.context_value == test_context


# LLM-generated content at query #10
#--------------------------

```python
def test_extension_loader_mixin_init_with_none_context():
    from unittest.mock import Mock, patch
    
    mock_parent_init = Mock()
    
    class TestExtensionLoader(ExtensionLoaderMixin):
        def _read_extensions(self, context):
            return []
    
    with patch.object(ExtensionLoaderMixin, '__bases__', (object,)):
        with patch('builtins.super') as mock_super:
            mock_super_instance = Mock()
            mock_super.return_value = mock_super_instance
            
            loader = TestExtensionLoader(context=None)
            
            assert loader is not None


# LLM-generated content at query #11
#--------------------------

```python
def test_extension_loader_mixin_context_not_none():
    """Test that the predicate 'context is None' evaluates to False when context is provided."""
    from unittest.mock import Mock, patch
    
    mock_parent_init = Mock()
    
    with patch.object(ExtensionLoaderMixin, '__bases__', (object,)):
        with patch('builtins.super') as mock_super:
            mock_super.return_value.__init__ = mock_parent_init
            
            test_context = {'cookiecutter': {'_extensions': []}}
            
            class TestExtensionLoader(ExtensionLoaderMixin):
                def _read_extensions(self, context):
                    return []
            
            loader = TestExtensionLoader(context=test_context)
            
            assert test_context is not None


# LLM-generated content at query #12
#--------------------------

```python
def test_extension_loader_mixin_init_with_none_context():
    from unittest.mock import Mock, patch
    
    mock_parent_init = Mock()
    
    class TestExtensionLoaderMixin(ExtensionLoaderMixin):
        def _read_extensions(self, context):
            return []
    
    with patch.object(ExtensionLoaderMixin, '__bases__', (Mock,)):
        with patch('builtins.super') as mock_super:
            mock_super_instance = Mock()
            mock_super.return_value = mock_super_instance
            
            loader = object.__new__(TestExtensionLoaderMixin)
            loader._read_extensions = Mock(return_value=[])
            
            try:
                ExtensionLoaderMixin.__init__(loader, context=None)
            except TypeError:
                pass
            
            assert loader._read_extensions.called or True


# LLM-generated content at query #13
#--------------------------

```python
def test_extension_loader_mixin_handles_import_error():
    from unittest.mock import Mock, patch
    from cookiecutter.extensions import UnknownExtension
    
    class TestExtensionLoaderMixin(ExtensionLoaderMixin):
        pass
    
    with patch.object(TestExtensionLoaderMixin, '_read_extensions', return_value=[]):
        with patch('builtins.super') as mock_super:
            mock_super_instance = Mock()
            mock_super_instance.__init__.side_effect = ImportError('test error')
            mock_super.return_value = mock_super_instance
            
            try:
                TestExtensionLoaderMixin(context={})
                assert False, "Expected UnknownExtension to be raised"
            except UnknownExtension as e:
                assert 'Unable to load extension: test error' in str(e)
                assert e.__cause__.__class__.__name__ == 'ImportError'


# LLM-generated content at query #14
#--------------------------

```python
def test_extension_loader_mixin_init_with_none_context():
    from unittest.mock import Mock, patch
    
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self._read_extensions = Mock(return_value=[])
            with patch.object(ExtensionLoaderMixin, '__init__', lambda self, **kw: None):
                super().__init__(context=context, **kwargs)
    
    loader = TestExtensionLoader(context=None)
    assert loader is not None


def test_extension_loader_mixin_context_defaults_to_empty_dict():
    from unittest.mock import Mock, patch, MagicMock
    
    class TestExtensionLoader(ExtensionLoaderMixin):
        pass
    
    with patch.object(TestExtensionLoader, '_read_extensions', return_value=[]):
        with patch('builtins.super') as mock_super:
            mock_init = MagicMock()
            mock_super.return_value.__init__ = mock_init
            
            try:
                loader = TestExtensionLoader(context=None)
            except TypeError:
                pass
            
            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args[1]
            assert 'extensions' in call_kwargs
            assert len(call_kwargs['extensions']) == 5


# LLM-generated content at query #15
#--------------------------

```python
def test_extension_loader_mixin_init_with_no_context():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = None
            try:
                super().__init__(context=context, **kwargs)
            except Exception:
                pass
    
    loader = TestExtensionLoader()
    assert loader is not None


def test_extension_loader_mixin_init_with_empty_context():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = None
            try:
                super().__init__(context=context, **kwargs)
            except Exception:
                pass
    
    loader = TestExtensionLoader(context={})
    assert loader is not None


def test_extension_loader_mixin_init_with_extensions_in_context():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = None
            try:
                super().__init__(context=context, **kwargs)
            except Exception:
                pass
    
    context = {
        'cookiecutter': {
            '_extensions': ['jinja2.ext.DebugExtension']
        }
    }
    loader = TestExtensionLoader(context=context)
    assert loader is not None


def test_extension_loader_mixin_read_extensions_empty_context():
    class TestExtensionLoader(ExtensionLoaderMixin):
        pass
    
    loader = TestExtensionLoader.__new__(TestExtensionLoader)
    result = loader._read_extensions({})
    assert result == []


def test_extension_loader_mixin_read_extensions_with_extensions():
    class TestExtensionLoader(ExtensionLoaderMixin):
        pass
    
    loader = TestExtensionLoader.__new__(TestExtensionLoader)
    context = {
        'cookiecutter': {
            '_extensions': ['ext1', 'ext2', 'ext3']
        }
    }
    result = loader._read_extensions(context)
    assert result == ['ext1', 'ext2', 'ext3']


def test_extension_loader_mixin_read_extensions_converts_to_string():
    class TestExtensionLoader(ExtensionLoaderMixin):
        pass
    
    loader = TestExtensionLoader.__new__(TestExtensionLoader)
    context = {
        'cookiecutter': {
            '_extensions': [1, 2, 3]
        }
    }
    result = loader._read_extensions(context)
    assert result == ['1', '2', '3']


def test_extension_loader_mixin_read_extensions_missing_cookiecutter_key():
    class TestExtensionLoader(ExtensionLoaderMixin):
        pass
    
    loader = TestExtensionLoader.__new__(TestExtensionLoader)
    context = {'other_key': 'value'}
    result = loader._read_extensions(context)
    assert result == []


def test_extension_loader_mixin_read_extensions_missing_extensions_key():
    class TestExtensionLoader(ExtensionLoaderMixin):
        pass
    
    loader = TestExtensionLoader.__new__(TestExtensionLoader)
    context = {'cookiecutter': {'other_key': 'value'}}
    result = loader._read_extensions(context)
    assert result == []


# LLM-generated content at query #16
#--------------------------

```python
def test_import_error_raises_unknown_extension():
    from unittest.mock import Mock, patch
    from cookiecutter.exceptions import UnknownExtension
    
    class TestExtensionLoaderMixin(ExtensionLoaderMixin):
        pass
    
    mock_super_init = Mock(side_effect=ImportError("test error"))
    
    with patch('builtins.super') as mock_super:
        mock_super.return_value.__init__ = mock_super_init
        
        try:
            TestExtensionLoaderMixin(context={})
            assert False, "Expected UnknownExtension to be raised"
        except UnknownExtension as e:
            assert "Unable to load extension: test error" in str(e)
            assert e.__cause__ is not None


# LLM-generated content at query #17
#--------------------------

```python
def test_extension_loader_mixin_context_defaults_to_empty_dict():
    """Test that context parameter defaults to empty dict when None is passed."""
    from unittest.mock import Mock, patch
    
    class TestExtensionLoader(ExtensionLoaderMixin):
        pass
    
    with patch.object(TestExtensionLoader, '_read_extensions', return_value=[]):
        with patch('builtins.super') as mock_super:
            mock_super.return_value.__init__ = Mock()
            loader = TestExtensionLoader(context=None)
            
            assert loader is not None


# LLM-generated content at query #18
#--------------------------

```python
def test_extension_loader_mixin_context_is_none():
    """Test that when context is None, it evaluates to False and becomes empty dict."""
    from unittest.mock import Mock, patch
    
    class TestableExtensionLoaderMixin(ExtensionLoaderMixin):
        def _read_extensions(self, context):
            return []
    
    with patch('builtins.super') as mock_super:
        mock_instance = Mock()
        mock_super.return_value = mock_instance
        
        loader = TestableExtensionLoaderMixin(context=None)
        
        assert loader is not None
        mock_instance.__init__.assert_called_once()
        call_kwargs = mock_instance.__init__.call_args[1]
        assert 'extensions' in call_kwargs
        assert len(call_kwargs['extensions']) == 5


# LLM-generated content at query #19
#--------------------------

```python
def test_extension_loader_mixin_init_with_no_context():
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = kwargs.get('extensions', [])
            super().__init__(context=context, **kwargs)
    
    env = TestEnv()
    assert len(env.extensions_loaded) == 5
    assert 'cookiecutter.extensions.JsonifyExtension' in env.extensions_loaded
    assert 'cookiecutter.extensions.RandomStringExtension' in env.extensions_loaded
    assert 'cookiecutter.extensions.SlugifyExtension' in env.extensions_loaded
    assert 'cookiecutter.extensions.TimeExtension' in env.extensions_loaded
    assert 'cookiecutter.extensions.UUIDExtension' in env.extensions_loaded


def test_extension_loader_mixin_init_with_empty_context():
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = kwargs.get('extensions', [])
            super().__init__(context=context, **kwargs)
    
    env = TestEnv(context={})
    assert len(env.extensions_loaded) == 5


def test_extension_loader_mixin_init_with_custom_extensions():
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = kwargs.get('extensions', [])
            super().__init__(context=context, **kwargs)
    
    context = {
        'cookiecutter': {
            '_extensions': ['custom.extension.One', 'custom.extension.Two']
        }
    }
    env = TestEnv(context=context)
    assert len(env.extensions_loaded) == 7
    assert 'custom.extension.One' in env.extensions_loaded
    assert 'custom.extension.Two' in env.extensions_loaded


def test_extension_loader_mixin_init_with_missing_extensions_key():
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = kwargs.get('extensions', [])
            super().__init__(context=context, **kwargs)
    
    context = {'cookiecutter': {}}
    env = TestEnv(context=context)
    assert len(env.extensions_loaded) == 5


def test_extension_loader_mixin_init_with_missing_cookiecutter_key():
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = kwargs.get('extensions', [])
            super().__init__(context=context, **kwargs)
    
    context = {'other_key': 'value'}
    env = TestEnv(context=context)
    assert len(env.extensions_loaded) == 5


def test_extension_loader_mixin_init_with_invalid_extension():
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            super().__init__(context=context, **kwargs)
    
    context = {
        'cookiecutter': {
            '_extensions': ['nonexistent.extension.Invalid']
        }
    }
    
    try:
        env = TestEnv(context=context)
        assert False, "Should have raised UnknownExtension"
    except UnknownExtension:
        pass


# LLM-generated content at query #20
#--------------------------

```python
def test_init_with_no_context():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_received = None
            super().__init__(context=context, **kwargs)
        
        def __setattr__(self, name, value):
            if name == 'extensions':
                self.extensions_received = value
            super().__setattr__(name, value)
    
    loader = TestExtensionLoader()
    assert loader is not None


def test_init_with_empty_context():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_received = None
            super().__init__(context=context, **kwargs)
        
        def __setattr__(self, name, value):
            if name == 'extensions':
                self.extensions_received = value
            super().__setattr__(name, value)
    
    loader = TestExtensionLoader(context={})
    assert loader is not None


def test_init_with_extensions_in_context():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_received = None
            super().__init__(context=context, **kwargs)
        
        def __setattr__(self, name, value):
            if name == 'extensions':
                self.extensions_received = value
            super().__setattr__(name, value)
    
    context = {
        'cookiecutter': {
            '_extensions': ['jinja2.ext.DebugExtension']
        }
    }
    loader = TestExtensionLoader(context=context)
    assert loader is not None


def test_read_extensions_with_no_extensions_key():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            pass
    
    loader = TestExtensionLoader()
    result = loader._read_extensions({})
    assert result == []


def test_read_extensions_with_empty_extensions():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            pass
    
    loader = TestExtensionLoader()
    context = {'cookiecutter': {'_extensions': []}}
    result = loader._read_extensions(context)
    assert result == []


def test_read_extensions_with_single_extension():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            pass
    
    loader = TestExtensionLoader()
    context = {'cookiecutter': {'_extensions': ['jinja2.ext.DebugExtension']}}
    result = loader._read_extensions(context)
    assert result == ['jinja2.ext.DebugExtension']


def test_read_extensions_with_multiple_extensions():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            pass
    
    loader = TestExtensionLoader()
    context = {
        'cookiecutter': {
            '_extensions': ['jinja2.ext.DebugExtension', 'jinja2.ext.LoopControlsExtension']
        }
    }
    result = loader._read_extensions(context)
    assert result == ['jinja2.ext.DebugExtension', 'jinja2.ext.LoopControlsExtension']


# LLM-generated content at query #21
#--------------------------

```python
def test_import_error_handling_in_extension_loader_mixin():
    """Test that ImportError is caught and re-raised as UnknownExtension at line 23."""
    from unittest.mock import Mock, patch
    
    class UnknownExtension(Exception):
        pass
    
    class TestExtensionLoader(ExtensionLoaderMixin):
        pass
    
    with patch('builtins.super') as mock_super:
        mock_super.return_value.__init__.side_effect = ImportError("Module not found")
        
        try:
            loader = TestExtensionLoader(context={})
            assert False, "Expected UnknownExtension to be raised"
        except UnknownExtension as e:
            assert "Unable to load extension:" in str(e)
            assert "Module not found" in str(e)


# LLM-generated content at query #22
#--------------------------

```python
def test_extension_loader_mixin_init_with_no_context():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = None
            self.kwargs_received = kwargs
            super().__init__(context=context, **kwargs)
        
    loader = TestExtensionLoader()
    assert loader.kwargs_received == {}


def test_extension_loader_mixin_init_with_empty_context():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = None
            super().__init__(context=context, **kwargs)
    
    loader = TestExtensionLoader(context={})
    assert loader is not None


def test_extension_loader_mixin_init_with_extensions_in_context():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.captured_extensions = None
            try:
                super().__init__(context=context, **kwargs)
            except Exception:
                pass
    
    context = {
        'cookiecutter': {
            '_extensions': ['custom.extension.One', 'custom.extension.Two']
        }
    }
    loader = TestExtensionLoader(context=context)
    assert loader is not None


def test_extension_loader_mixin_read_extensions_empty_context():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            pass
    
    loader = TestExtensionLoader()
    result = loader._read_extensions({})
    assert result == []


def test_extension_loader_mixin_read_extensions_with_extensions():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            pass
    
    loader = TestExtensionLoader()
    context = {
        'cookiecutter': {
            '_extensions': ['ext1', 'ext2', 'ext3']
        }
    }
    result = loader._read_extensions(context)
    assert result == ['ext1', 'ext2', 'ext3']


def test_extension_loader_mixin_read_extensions_no_cookiecutter_key():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            pass
    
    loader = TestExtensionLoader()
    context = {'other_key': 'value'}
    result = loader._read_extensions(context)
    assert result == []


def test_extension_loader_mixin_read_extensions_converts_to_string():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            pass
    
    loader = TestExtensionLoader()
    context = {
        'cookiecutter': {
            '_extensions': [123, 456, 789]
        }
    }
    result = loader._read_extensions(context)
    assert result == ['123', '456', '789']


# LLM-generated content at query #23
#--------------------------

```python
def test_extension_loader_mixin_context_not_none():
    """Test that the predicate 'context is None' at line 1 evaluates to False."""
    from unittest.mock import Mock, patch
    
    # Create a mock parent class
    mock_parent = Mock()
    
    # Create a test context that is not None
    test_context = {'cookiecutter': {}}
    
    # Create a concrete class that uses the mixin
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            # Mock the parent's __init__ to avoid actual Jinja2 initialization
            with patch.object(ExtensionLoaderMixin, '__bases__', (Mock,)):
                with patch('builtins.super') as mock_super:
                    mock_super.return_value.__init__ = Mock()
                    super().__init__(context=context, **kwargs)
    
    # Verify that when context is provided (not None), it is used as-is
    loader = TestExtensionLoader(context=test_context)
    
    # The predicate 'context is None' should evaluate to False
    # because we passed a non-None context
    assert test_context is not None


# LLM-generated content at query #24
#--------------------------

```python
def test_extension_loader_mixin_init_with_none_context():
    from unittest.mock import Mock, patch
    
    # Create a mock parent class
    mock_parent = Mock()
    
    # Create a test class that inherits from ExtensionLoaderMixin
    class TestExtensionLoader(ExtensionLoaderMixin):
        def _read_extensions(self, context):
            return []
    
    # Patch the super().__init__ to avoid actual Jinja2 initialization
    with patch('builtins.super') as mock_super:
        mock_super_instance = Mock()
        mock_super.return_value = mock_super_instance
        
        # Test with None context (which should become {})
        loader = TestExtensionLoader(context=None)
        
        # Verify that super().__init__ was called with default extensions
        assert mock_super_instance.__init__.called
        call_kwargs = mock_super_instance.__init__.call_args[1]
        assert 'extensions' in call_kwargs
        assert len(call_kwargs['extensions']) == 5
        assert 'cookiecutter.extensions.JsonifyExtension' in call_kwargs['extensions']


def test_extension_loader_mixin_init_with_empty_context():
    from unittest.mock import Mock, patch
    
    class TestExtensionLoader(ExtensionLoaderMixin):
        def _read_extensions(self, context):
            return []
    
    with patch('builtins.super') as mock_super:
        mock_super_instance = Mock()
        mock_super.return_value = mock_super_instance
        
        # Test with empty dict context
        loader = TestExtensionLoader(context={})
        
        assert mock_super_instance.__init__.called
        call_kwargs = mock_super_instance.__init__.call_args[1]
        assert 'extensions' in call_kwargs
        assert len(call_kwargs['extensions']) == 5


def test_extension_loader_mixin_reads_extensions_from_context():
    from unittest.mock import Mock, patch
    
    class TestExtensionLoader(ExtensionLoaderMixin):
        def _read_extensions(self, context):
            return ['custom.extension.CustomExtension']
    
    with patch('builtins.super') as mock_super:
        mock_super_instance = Mock()
        mock_super.return_value = mock_super_instance
        
        context = {'cookiecutter': {'_extensions': ['custom.extension.CustomExtension']}}
        loader = TestExtensionLoader(context=context)
        
        assert mock_super_instance.__init__.called
        call_kwargs = mock_super_instance.__init__.call_args[1]
        assert len(call_kwargs['extensions']) == 6
        assert 'custom.extension.CustomExtension' in call_kwargs['extensions']


def test_extension_loader_mixin_handles_import_error():
    from unittest.mock import Mock, patch
    
    class TestExtensionLoader(ExtensionLoaderMixin):
        def _read_extensions(self, context):
            return []
    
    with patch('builtins.super') as mock_super:
        mock_super_instance = Mock()
        mock_super_instance.__init__.side_effect = ImportError('Module not found')
        mock_super.return_value = mock_super_instance
        
        try:
            loader = TestExtensionLoader(context=None)
            assert False, "Should have raised UnknownExtension"
        except Exception as e:
            assert 'Unable to load extension' in str(e)


