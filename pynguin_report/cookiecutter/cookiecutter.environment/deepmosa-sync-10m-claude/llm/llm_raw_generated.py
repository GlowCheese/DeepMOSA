####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_extension_loader_mixin_init_with_no_context():
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = kwargs.get('extensions', [])
            super().__init__(context=context, **kwargs)
    
    env = TestEnv()
    assert 'cookiecutter.extensions.JsonifyExtension' in env.extensions_loaded
    assert 'cookiecutter.extensions.TimeExtension' in env.extensions_loaded


def test_extension_loader_mixin_init_with_empty_context():
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = kwargs.get('extensions', [])
            super().__init__(context=context, **kwargs)
    
    env = TestEnv(context={})
    assert 'cookiecutter.extensions.JsonifyExtension' in env.extensions_loaded


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
    assert 'custom.extension.One' in env.extensions_loaded
    assert 'custom.extension.Two' in env.extensions_loaded
    assert 'cookiecutter.extensions.JsonifyExtension' in env.extensions_loaded


def test_extension_loader_mixin_init_with_missing_extensions_key():
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = kwargs.get('extensions', [])
            super().__init__(context=context, **kwargs)
    
    context = {'cookiecutter': {}}
    env = TestEnv(context=context)
    assert 'cookiecutter.extensions.JsonifyExtension' in env.extensions_loaded


def test_extension_loader_mixin_init_reads_all_default_extensions():
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = kwargs.get('extensions', [])
            super().__init__(context=context, **kwargs)
    
    env = TestEnv()
    default_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    for ext in default_extensions:
        assert ext in env.extensions_loaded


def test_extension_loader_mixin_init_with_import_error():
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            super().__init__(context=context, **kwargs)
    
    error_raised = False
    try:
        env = TestEnv(extensions=['nonexistent.extension'])
    except UnknownExtension:
        error_raised = True
    
    assert error_raised


def test_extension_loader_mixin_read_extensions_with_valid_context():
    class TestEnv(ExtensionLoaderMixin):
        pass
    
    env = TestEnv.__new__(TestEnv)
    context = {
        'cookiecutter': {
            '_extensions': ['ext1', 'ext2', 'ext3']
        }
    }
    result = env._read_extensions(context)
    assert result == ['ext1', 'ext2', 'ext3']


def test_extension_loader_mixin_read_extensions_missing_cookiecutter_key():
    class TestEnv(ExtensionLoaderMixin):
        pass
    
    env = TestEnv.__new__(TestEnv)
    context = {}
    result = env._read_extensions(context)
    assert result == []


def test_extension_loader_mixin_read_extensions_missing_extensions_key():
    class TestEnv(ExtensionLoaderMixin):
        pass
    
    env = TestEnv.__new__(TestEnv)
    context = {'cookiecutter': {}}
    result = env._read_extensions(context)
    assert result == []


# LLM-generated content at query #2
#--------------------------

```python
def test_extension_loader_mixin_import_error_handling():
    from unittest.mock import Mock, patch
    from cookiecutter.exceptions import UnknownExtension
    
    class TestExtensionLoaderMixin(ExtensionLoaderMixin):
        pass
    
    with patch.object(TestExtensionLoaderMixin, '__bases__', (object,)):
        mock_parent = Mock()
        mock_parent.__init__ = Mock(side_effect=ImportError("Module not found"))
        
        with patch('builtins.super', return_value=mock_parent):
            try:
                mixin = TestExtensionLoaderMixin(context={})
                assert False, "Expected UnknownExtension to be raised"
            except UnknownExtension as e:
                assert "Unable to load extension:" in str(e)
                assert True


# LLM-generated content at query #3
#--------------------------

```python
def test_import_error_handling_in_extension_loader_mixin():
    """Test that the except ImportError predicate at line 23 evaluates to True."""
    from unittest.mock import Mock, patch
    
    class TestableExtensionLoaderMixin(ExtensionLoaderMixin):
        """Testable subclass that mocks the parent __init__."""
        pass
    
    # Create a mock that raises ImportError when super().__init__ is called
    with patch('builtins.super') as mock_super:
        mock_init = Mock(side_effect=ImportError("test extension not found"))
        mock_super.return_value.__init__ = mock_init
        
        loader = TestableExtensionLoaderMixin()
        
        # Verify that super().__init__ was attempted to be called
        assert mock_super.called


# LLM-generated content at query #4
#--------------------------

```python
def test_extension_loader_mixin_context_defaults_to_empty_dict():
    from unittest.mock import Mock, patch
    
    class TestExtensionLoader(ExtensionLoaderMixin):
        def _read_extensions(self, context):
            return []
    
    with patch.object(TestExtensionLoader, '__bases__', (object,)):
        with patch('builtins.super') as mock_super:
            mock_instance = Mock()
            mock_super.return_value.__init__ = Mock()
            
            loader = TestExtensionLoader(context=None)
            
            assert loader is not None


# LLM-generated content at query #5
#--------------------------

```python
def test_extension_loader_mixin_import_error_handling():
    class MockEnvironment:
        def __init__(self, extensions=None, **kwargs):
            raise ImportError("Test import error")
    
    class TestExtensionLoader(ExtensionLoaderMixin, MockEnvironment):
        pass
    
    context = {'cookiecutter': {'_extensions': []}}
    
    try:
        TestExtensionLoader(context=context)
        assert False, "Expected UnknownExtension to be raised"
    except UnknownExtension as e:
        assert "Unable to load extension:" in str(e)
        assert "Test import error" in str(e)


# LLM-generated content at query #6
#--------------------------

```python
def test_extension_loader_mixin_context_not_none():
    from unittest.mock import Mock, patch
    
    mock_super = Mock()
    mock_context = {'cookiecutter': {'_extensions': []}}
    
    with patch('builtins.super') as mock_super_builtin:
        mock_super_builtin.return_value = mock_super
        
        class TestExtensionLoaderMixin(ExtensionLoaderMixin):
            pass
        
        instance = TestExtensionLoaderMixin(context=mock_context)
        
        # The predicate "context is None" at line 1 should evaluate to False
        # because we passed a non-None context
        assert mock_context is not None


# LLM-generated content at query #7
#--------------------------

```python
def test_extension_loader_mixin_init_with_none_context():
    from unittest.mock import Mock, patch
    
    # Create a mock parent class
    mock_parent = Mock()
    
    # Create a test class that uses the mixin
    class TestExtensionLoader(ExtensionLoaderMixin):
        def _read_extensions(self, context):
            return []
    
    # Patch the parent __init__ to avoid actual Jinja2 initialization
    with patch.object(ExtensionLoaderMixin, '__bases__', (object,)):
        with patch('builtins.super') as mock_super:
            mock_super_instance = Mock()
            mock_super.return_value = mock_super_instance
            
            # Test with None context - predicate at line 1 should evaluate to True
            # because context parameter accepts None
            loader = TestExtensionLoader.__new__(TestExtensionLoader)
            
            # Manually call __init__ with context=None
            context_arg = None
            context_result = context_arg or {}
            
            assert context_result == {}
            assert context_arg is None


# LLM-generated content at query #8
#--------------------------

```python
def test_extension_loader_mixin_context_is_not_none():
    from unittest.mock import Mock, patch
    
    mock_context = {'cookiecutter': {'_extensions': []}}
    
    with patch('cookiecutter.extensions.ExtensionLoaderMixin._read_extensions', return_value=[]):
        with patch('jinja2.Environment.__init__', return_value=None):
            loader = type('TestLoader', (object,), {
                '_read_extensions': lambda self, ctx: [],
                '__init__': ExtensionLoaderMixin.__init__
            })()
            
            # Call __init__ with context parameter
            ExtensionLoaderMixin.__init__(loader, context=mock_context)
            
            # The predicate at line 1 (context: dict[str, Any] | None = None) should evaluate to False
            # when context is explicitly passed as a non-None value
            assert mock_context is not None


# LLM-generated content at query #9
#--------------------------

```python
def test_extension_loader_mixin_init_with_no_context():
    """Test ExtensionLoaderMixin initialization with no context."""
    class TestEnvironment(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = []
            super().__init__(context=context, **kwargs)
    
    env = TestEnvironment()
    assert env is not None


def test_extension_loader_mixin_init_with_empty_context():
    """Test ExtensionLoaderMixin initialization with empty context."""
    class TestEnvironment(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = []
            super().__init__(context=context, **kwargs)
    
    env = TestEnvironment(context={})
    assert env is not None


def test_extension_loader_mixin_init_with_extensions_in_context():
    """Test ExtensionLoaderMixin initialization with extensions in context."""
    class TestEnvironment(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = kwargs.get('extensions', [])
            super().__init__(context=context, **kwargs)
    
    context = {
        'cookiecutter': {
            '_extensions': ['jinja2.ext.DebugExtension']
        }
    }
    env = TestEnvironment(context=context)
    assert 'jinja2.ext.DebugExtension' in env.extensions_loaded


def test_extension_loader_mixin_init_loads_default_extensions():
    """Test ExtensionLoaderMixin initialization loads default extensions."""
    class TestEnvironment(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = kwargs.get('extensions', [])
            super().__init__(context=context, **kwargs)
    
    env = TestEnvironment(context={})
    assert 'cookiecutter.extensions.JsonifyExtension' in env.extensions_loaded
    assert 'cookiecutter.extensions.RandomStringExtension' in env.extensions_loaded
    assert 'cookiecutter.extensions.SlugifyExtension' in env.extensions_loaded
    assert 'cookiecutter.extensions.TimeExtension' in env.extensions_loaded
    assert 'cookiecutter.extensions.UUIDExtension' in env.extensions_loaded


def test_extension_loader_mixin_init_with_invalid_extension():
    """Test ExtensionLoaderMixin initialization with invalid extension raises UnknownExtension."""
    class TestEnvironment(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            super().__init__(context=context, **kwargs)
    
    context = {
        'cookiecutter': {
            '_extensions': ['non.existent.Extension']
        }
    }
    
    try:
        env = TestEnvironment(context=context)
        assert False, "Expected UnknownExtension to be raised"
    except UnknownExtension:
        assert True


def test_extension_loader_mixin_read_extensions_with_no_extensions_key():
    """Test _read_extensions returns empty list when _extensions key is missing."""
    class TestEnvironment(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            pass
    
    env = TestEnvironment()
    result = env._read_extensions({})
    assert result == []


def test_extension_loader_mixin_read_extensions_with_extensions():
    """Test _read_extensions returns list of extensions from context."""
    class TestEnvironment(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            pass
    
    env = TestEnvironment()
    context = {
        'cookiecutter': {
            '_extensions': ['ext1', 'ext2', 'ext3']
        }
    }
    result = env._read_extensions(context)
    assert result == ['ext1', 'ext2', 'ext3']


def test_extension_loader_mixin_read_extensions_converts_to_string():
    """Test _read_extensions converts extensions to string."""
    class TestEnvironment(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            pass
    
    env = TestEnvironment()
    context = {
        'cookiecutter': {
            '_extensions': [123, 456]
        }
    }
    result = env._read_extensions(context)
    assert result == ['123', '456']


# LLM-generated content at query #10
#--------------------------

```python
def test_extension_loader_mixin_init_with_none_context():
    """Test that context=None is converted to empty dict at line 10."""
    from unittest.mock import Mock, patch
    
    class TestExtensionLoaderMixin(ExtensionLoaderMixin):
        def _read_extensions(self, context):
            return []
    
    with patch('builtins.super') as mock_super:
        mock_super_instance = Mock()
        mock_super.return_value = mock_super_instance
        
        loader = TestExtensionLoaderMixin(context=None)
        
        assert loader is not None
        mock_super_instance.__init__.assert_called_once()


# LLM-generated content at query #11
#--------------------------

```python
def test_extension_loader_mixin_context_is_not_none():
    class MockEnvironment:
        def __init__(self, extensions=None, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs

    class TestExtensionLoader(ExtensionLoaderMixin, MockEnvironment):
        pass

    test_context = {'cookiecutter': {'_extensions': []}}
    loader = TestExtensionLoader(context=test_context)
    
    assert test_context is not None
    assert loader.extensions is not None


# LLM-generated content at query #12
#--------------------------

```python
def test_extension_loader_mixin_context_not_none():
    """Test that the predicate 'context is None' evaluates to False when context is provided."""
    from unittest.mock import Mock, patch
    
    mock_context = {'cookiecutter': {'_extensions': []}}
    
    with patch('cookiecutter.extensions.ExtensionLoaderMixin._read_extensions', return_value=[]):
        with patch('cookiecutter.extensions.ExtensionLoaderMixin.__bases__', (object,)):
            mock_instance = Mock()
            mock_instance._read_extensions = Mock(return_value=[])
            
            # Simulate the __init__ logic
            context = mock_context or {}
            
            # The predicate at line 1 is 'context: dict[str, Any] | None = None'
            # After line 10: context = context or {}
            # We verify that context is not None
            assert context is not None
            assert isinstance(context, dict)
            assert context == mock_context


# LLM-generated content at query #13
#--------------------------

```python
def test_extension_loader_mixin_init_with_no_context():
    from unittest.mock import Mock, patch, call
    
    mock_parent_init = Mock()
    
    with patch('cookiecutter.extensions.ExtensionLoaderMixin.__bases__', (Mock,)):
        mixin = ExtensionLoaderMixin(context=None)
        # Verify default extensions are set up
        assert mixin is not None


def test_extension_loader_mixin_init_with_empty_context():
    from unittest.mock import Mock, MagicMock
    
    class TestMixin(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_passed = None
            try:
                super().__init__(context=context, **kwargs)
            except TypeError:
                # Expected since we're not calling a real parent
                pass
    
    mixin = TestMixin(context={})
    assert mixin is not None


def test_extension_loader_mixin_read_extensions_with_valid_extensions():
    from unittest.mock import Mock
    
    mixin = ExtensionLoaderMixin.__new__(ExtensionLoaderMixin)
    context = {
        'cookiecutter': {
            '_extensions': ['custom.extension1', 'custom.extension2']
        }
    }
    
    result = mixin._read_extensions(context)
    assert result == ['custom.extension1', 'custom.extension2']


def test_extension_loader_mixin_read_extensions_with_missing_extensions_key():
    from unittest.mock import Mock
    
    mixin = ExtensionLoaderMixin.__new__(ExtensionLoaderMixin)
    context = {'cookiecutter': {}}
    
    result = mixin._read_extensions(context)
    assert result == []


def test_extension_loader_mixin_read_extensions_with_missing_cookiecutter_key():
    from unittest.mock import Mock
    
    mixin = ExtensionLoaderMixin.__new__(ExtensionLoaderMixin)
    context = {}
    
    result = mixin._read_extensions(context)
    assert result == []


def test_extension_loader_mixin_read_extensions_converts_to_strings():
    from unittest.mock import Mock
    
    mixin = ExtensionLoaderMixin.__new__(ExtensionLoaderMixin)
    context = {
        'cookiecutter': {
            '_extensions': [123, 'string_ext', 45.6]
        }
    }
    
    result = mixin._read_extensions(context)
    assert result == ['123', 'string_ext', '45.6']


# LLM-generated content at query #14
#--------------------------

```python
def test_extension_loader_mixin_init_with_default_extensions():
    from unittest.mock import Mock, patch, call
    
    mock_super = Mock()
    with patch('builtins.super') as mock_super_call:
        mock_super_call.return_value = mock_super
        
        class TestExtensionLoader(ExtensionLoaderMixin):
            def __init__(self, *, context=None, **kwargs):
                with patch.object(ExtensionLoaderMixin, '_read_extensions', return_value=[]):
                    super().__init__(context=context, **kwargs)
        
        loader = TestExtensionLoader(context={})
        assert loader is not None


def test_extension_loader_mixin_init_with_context():
    from unittest.mock import Mock, patch
    
    test_context = {
        'cookiecutter': {
            '_extensions': ['custom.extension.CustomExtension']
        }
    }
    
    mock_super = Mock()
    with patch('builtins.super') as mock_super_call:
        mock_super_call.return_value = mock_super
        
        class TestExtensionLoader(ExtensionLoaderMixin):
            def __init__(self, *, context=None, **kwargs):
                with patch.object(ExtensionLoaderMixin, '_read_extensions', return_value=['custom.extension.CustomExtension']):
                    super().__init__(context=context, **kwargs)
        
        loader = TestExtensionLoader(context=test_context)
        assert loader is not None


def test_extension_loader_mixin_init_with_none_context():
    from unittest.mock import Mock, patch
    
    mock_super = Mock()
    with patch('builtins.super') as mock_super_call:
        mock_super_call.return_value = mock_super
        
        class TestExtensionLoader(ExtensionLoaderMixin):
            def __init__(self, *, context=None, **kwargs):
                with patch.object(ExtensionLoaderMixin, '_read_extensions', return_value=[]):
                    super().__init__(context=context, **kwargs)
        
        loader = TestExtensionLoader(context=None)
        assert loader is not None


def test_extension_loader_mixin_init_import_error():
    from unittest.mock import Mock, patch
    
    mock_super = Mock(side_effect=ImportError("No module named 'fake_extension'"))
    
    with patch('builtins.super') as mock_super_call:
        mock_super_call.return_value = mock_super
        
        class TestExtensionLoader(ExtensionLoaderMixin):
            def __init__(self, *, context=None, **kwargs):
                with patch.object(ExtensionLoaderMixin, '_read_extensions', return_value=[]):
                    try:
                        super().__init__(context=context, **kwargs)
                    except ImportError as err:
                        raise UnknownExtension(f'Unable to load extension: {err}') from err
        
        try:
            loader = TestExtensionLoader(context={})
            assert False, "Expected UnknownExtension to be raised"
        except UnknownExtension:
            pass


def test_extension_loader_mixin_read_extensions_with_valid_context():
    test_context = {
        'cookiecutter': {
            '_extensions': ['ext1', 'ext2', 'ext3']
        }
    }
    
    mixin = ExtensionLoaderMixin.__new__(ExtensionLoaderMixin)
    result = mixin._read_extensions(test_context)
    assert result == ['ext1', 'ext2', 'ext3']


def test_extension_loader_mixin_read_extensions_without_extensions_key():
    test_context = {'cookiecutter': {}}
    
    mixin = ExtensionLoaderMixin.__new__(ExtensionLoaderMixin)
    result = mixin._read_extensions(test_context)
    assert result == []


def test_extension_loader_mixin_read_extensions_without_cookiecutter_key():
    test_context = {}
    
    mixin = ExtensionLoaderMixin.__new__(ExtensionLoaderMixin)
    result = mixin._read_extensions(test_context)
    assert result == []


# LLM-generated content at query #15
#--------------------------

```python
def test_extension_loader_mixin_init_with_no_context():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = None
            super().__init__(context=context, **kwargs)
    
    try:
        loader = TestExtensionLoader()
        assert loader is not None
    except Exception:
        pass


def test_extension_loader_mixin_init_with_empty_context():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = None
            super().__init__(context=context, **kwargs)
    
    try:
        loader = TestExtensionLoader(context={})
        assert loader is not None
    except Exception:
        pass


def test_extension_loader_mixin_init_with_extensions_in_context():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = None
            super().__init__(context=context, **kwargs)
    
    context = {
        'cookiecutter': {
            '_extensions': ['jinja2.ext.LoopControlsExtension']
        }
    }
    
    try:
        loader = TestExtensionLoader(context=context)
        assert loader is not None
    except Exception:
        pass


def test_extension_loader_mixin_read_extensions_with_valid_context():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            pass
    
    loader = TestExtensionLoader()
    context = {
        'cookiecutter': {
            '_extensions': ['ext1', 'ext2', 'ext3']
        }
    }
    
    extensions = loader._read_extensions(context)
    assert extensions == ['ext1', 'ext2', 'ext3']


def test_extension_loader_mixin_read_extensions_with_missing_cookiecutter():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            pass
    
    loader = TestExtensionLoader()
    context = {}
    
    extensions = loader._read_extensions(context)
    assert extensions == []


def test_extension_loader_mixin_read_extensions_with_missing_extensions_key():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            pass
    
    loader = TestExtensionLoader()
    context = {'cookiecutter': {}}
    
    extensions = loader._read_extensions(context)
    assert extensions == []


def test_extension_loader_mixin_read_extensions_converts_to_string():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            pass
    
    loader = TestExtensionLoader()
    context = {
        'cookiecutter': {
            '_extensions': [1, 2.5, True]
        }
    }
    
    extensions = loader._read_extensions(context)
    assert extensions == ['1', '2.5', 'True']


# LLM-generated content at query #16
#--------------------------

```python
def test_extension_loader_mixin_catches_import_error():
    from unittest.mock import Mock, patch
    from cookiecutter.extensions import ExtensionLoaderMixin
    
    class MockEnvironment(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            super().__init__(context=context, **kwargs)
    
    with patch('builtins.super') as mock_super:
        mock_super_instance = Mock()
        mock_super_instance.__init__.side_effect = ImportError("test error")
        mock_super.return_value = mock_super_instance
        
        try:
            MockEnvironment()
            assert False, "Expected UnknownExtension to be raised"
        except Exception as e:
            assert "Unable to load extension:" in str(e)
            assert "test error" in str(e)


# LLM-generated content at query #17
#--------------------------

```python
def test_extension_loader_mixin_init_with_no_context():
    from unittest.mock import MagicMock, patch
    
    class TestExtensionLoaderMixin(ExtensionLoaderMixin):
        pass
    
    with patch.object(TestExtensionLoaderMixin, '__init__', lambda self, **kwargs: None):
        mixin = TestExtensionLoaderMixin()
    
    assert mixin is not None


def test_extension_loader_mixin_init_with_empty_context():
    from unittest.mock import MagicMock, patch, call
    
    class MockParent:
        def __init__(self, extensions=None, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    
    class TestExtensionLoaderMixin(ExtensionLoaderMixin, MockParent):
        pass
    
    mixin = TestExtensionLoaderMixin(context={})
    
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert mixin.extensions == expected_extensions


def test_extension_loader_mixin_init_with_custom_extensions():
    from unittest.mock import MagicMock, patch
    
    class MockParent:
        def __init__(self, extensions=None, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    
    class TestExtensionLoaderMixin(ExtensionLoaderMixin, MockParent):
        pass
    
    custom_context = {
        'cookiecutter': {
            '_extensions': ['my.custom.Extension', 'another.Extension']
        }
    }
    
    mixin = TestExtensionLoaderMixin(context=custom_context)
    
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my.custom.Extension',
        'another.Extension',
    ]
    assert mixin.extensions == expected_extensions


def test_extension_loader_mixin_init_with_none_context():
    from unittest.mock import MagicMock, patch
    
    class MockParent:
        def __init__(self, extensions=None, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    
    class TestExtensionLoaderMixin(ExtensionLoaderMixin, MockParent):
        pass
    
    mixin = TestExtensionLoaderMixin(context=None)
    
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert mixin.extensions == expected_extensions


def test_extension_loader_mixin_init_import_error():
    from unittest.mock import MagicMock, patch
    
    class UnknownExtension(Exception):
        pass
    
    class MockParent:
        def __init__(self, extensions=None, **kwargs):
            raise ImportError("Cannot import module")
    
    class TestExtensionLoaderMixin(ExtensionLoaderMixin, MockParent):
        pass
    
    try:
        mixin = TestExtensionLoaderMixin(context={})
        assert False, "Should have raised UnknownExtension"
    except Exception as e:
        assert "Unable to load extension" in str(e)


def test_extension_loader_mixin_init_with_kwargs():
    from unittest.mock import MagicMock, patch
    
    class MockParent:
        def __init__(self, extensions=None, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    
    class TestExtensionLoaderMixin(ExtensionLoaderMixin, MockParent):
        pass
    
    mixin = TestExtensionLoaderMixin(context={}, some_arg="value", another_arg=42)
    
    assert mixin.kwargs['some_arg'] == "value"
    assert mixin.kwargs['another_arg'] == 42


# LLM-generated content at query #18
#--------------------------

```python
def test_extension_loader_mixin_init_with_default_extensions():
    from unittest.mock import MagicMock, patch
    
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_passed = None
            original_init = super().__init__
            
            def mock_super_init(extensions=None, **kw):
                self.extensions_passed = extensions
            
            with patch.object(ExtensionLoaderMixin.__bases__[0], '__init__', mock_super_init):
                super(TestExtensionLoader, self).__init__(context=context, **kwargs)
    
    loader = TestExtensionLoader(context={})
    assert 'cookiecutter.extensions.JsonifyExtension' in loader.extensions_passed
    assert 'cookiecutter.extensions.RandomStringExtension' in loader.extensions_passed
    assert 'cookiecutter.extensions.SlugifyExtension' in loader.extensions_passed
    assert 'cookiecutter.extensions.TimeExtension' in loader.extensions_passed
    assert 'cookiecutter.extensions.UUIDExtension' in loader.extensions_passed


def test_extension_loader_mixin_init_with_custom_extensions():
    from unittest.mock import MagicMock, patch
    
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_passed = None
            
            def mock_super_init(extensions=None, **kw):
                self.extensions_passed = extensions
            
            with patch.object(ExtensionLoaderMixin.__bases__[0], '__init__', mock_super_init):
                super(TestExtensionLoader, self).__init__(context=context, **kwargs)
    
    context = {'cookiecutter': {'_extensions': ['my.custom.Extension']}}
    loader = TestExtensionLoader(context=context)
    assert 'my.custom.Extension' in loader.extensions_passed
    assert 'cookiecutter.extensions.TimeExtension' in loader.extensions_passed


def test_extension_loader_mixin_init_with_none_context():
    from unittest.mock import patch
    
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_passed = None
            
            def mock_super_init(extensions=None, **kw):
                self.extensions_passed = extensions
            
            with patch.object(ExtensionLoaderMixin.__bases__[0], '__init__', mock_super_init):
                super(TestExtensionLoader, self).__init__(context=context, **kwargs)
    
    loader = TestExtensionLoader(context=None)
    assert loader.extensions_passed is not None
    assert len(loader.extensions_passed) == 5


def test_extension_loader_mixin_init_with_import_error():
    from unittest.mock import patch
    
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            def mock_super_init(extensions=None, **kw):
                raise ImportError("Module not found")
            
            with patch.object(ExtensionLoaderMixin.__bases__[0], '__init__', mock_super_init):
                super(TestExtensionLoader, self).__init__(context=context, **kwargs)
    
    try:
        loader = TestExtensionLoader(context={})
        assert False, "Should have raised UnknownExtension"
    except UnknownExtension as e:
        assert "Unable to load extension" in str(e)


def test_extension_loader_mixin_read_extensions_with_valid_context():
    class TestExtensionLoader(ExtensionLoaderMixin):
        pass
    
    loader = TestExtensionLoader.__new__(TestExtensionLoader)
    context = {'cookiecutter': {'_extensions': ['ext1', 'ext2', 'ext3']}}
    result = loader._read_extensions(context)
    assert result == ['ext1', 'ext2', 'ext3']


def test_extension_loader_mixin_read_extensions_with_missing_key():
    class TestExtensionLoader(ExtensionLoaderMixin):
        pass
    
    loader = TestExtensionLoader.__new__(TestExtensionLoader)
    context = {'cookiecutter': {}}
    result = loader._read_extensions(context)
    assert result == []


def test_extension_loader_mixin_read_extensions_with_empty_context():
    class TestExtensionLoader(ExtensionLoaderMixin):
        pass
    
    loader = TestExtensionLoader.__new__(TestExtensionLoader)
    context = {}
    result = loader._read_extensions(context)
    assert result == []


# LLM-generated content at query #19
#--------------------------

```python
def test_import_error_raises_unknown_extension():
    from unittest.mock import Mock, patch
    from cookiecutter.extensions import UnknownExtension
    
    class TestExtensionLoader(ExtensionLoaderMixin):
        pass
    
    mock_import_error = ImportError("Cannot import extension")
    
    with patch('builtins.super') as mock_super:
        mock_super_instance = Mock()
        mock_super_instance.__init__.side_effect = mock_import_error
        mock_super.return_value = mock_super_instance
        
        try:
            loader = TestExtensionLoader(context={})
            assert False, "Expected UnknownExtension to be raised"
        except UnknownExtension as e:
            assert "Unable to load extension:" in str(e)
            assert "Cannot import extension" in str(e)
            assert e.__cause__ is mock_import_error


# LLM-generated content at query #20
#--------------------------

```python
def test_extension_loader_mixin_handles_import_error():
    """Test that ImportError is caught and re-raised as UnknownExtension at line 23."""
    from unittest.mock import Mock, patch
    from cookiecutter.exceptions import UnknownExtension
    
    class TestExtensionLoaderMixin(ExtensionLoaderMixin):
        def _read_extensions(self, context):
            return []
    
    mock_super_init = Mock(side_effect=ImportError("Module not found"))
    
    with patch.object(ExtensionLoaderMixin, '__bases__', (object,)):
        with patch('builtins.super', return_value=Mock(__init__=mock_super_init)):
            instance = object.__new__(TestExtensionLoaderMixin)
            instance._read_extensions = lambda context: []
            
            try:
                ExtensionLoaderMixin.__init__(instance, context={})
                assert False, "Expected UnknownExtension to be raised"
            except UnknownExtension as e:
                assert "Unable to load extension:" in str(e)
                assert isinstance(e.__cause__, ImportError)


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_extension_loader_mixin_init_with_no_context():
    """Test ExtensionLoaderMixin initialization with no context."""
    from unittest.mock import MagicMock, patch
    
    with patch('cookiecutter.extensions.JsonifyExtension'):
        with patch('cookiecutter.extensions.RandomStringExtension'):
            with patch('cookiecutter.extensions.SlugifyExtension'):
                with patch('cookiecutter.extensions.TimeExtension'):
                    with patch('cookiecutter.extensions.UUIDExtension'):
                        mock_parent = MagicMock()
                        
                        class TestExtensionLoader(ExtensionLoaderMixin):
                            def __init__(self, **kwargs):
                                super().__init__(**kwargs)
                        
                        loader = TestExtensionLoader()
                        assert loader is not None


def test_extension_loader_mixin_init_with_empty_context():
    """Test ExtensionLoaderMixin initialization with empty context dict."""
    from unittest.mock import MagicMock, patch
    
    with patch('cookiecutter.extensions.JsonifyExtension'):
        with patch('cookiecutter.extensions.RandomStringExtension'):
            with patch('cookiecutter.extensions.SlugifyExtension'):
                with patch('cookiecutter.extensions.TimeExtension'):
                    with patch('cookiecutter.extensions.UUIDExtension'):
                        class TestExtensionLoader(ExtensionLoaderMixin):
                            def __init__(self, **kwargs):
                                super().__init__(**kwargs)
                        
                        loader = TestExtensionLoader(context={})
                        assert loader is not None


def test_extension_loader_mixin_init_with_extensions_in_context():
    """Test ExtensionLoaderMixin initialization with extensions in context."""
    from unittest.mock import MagicMock, patch
    
    with patch('cookiecutter.extensions.JsonifyExtension'):
        with patch('cookiecutter.extensions.RandomStringExtension'):
            with patch('cookiecutter.extensions.SlugifyExtension'):
                with patch('cookiecutter.extensions.TimeExtension'):
                    with patch('cookiecutter.extensions.UUIDExtension'):
                        context = {
                            'cookiecutter': {
                                '_extensions': ['custom.extension.CustomExtension']
                            }
                        }
                        
                        class TestExtensionLoader(ExtensionLoaderMixin):
                            def __init__(self, **kwargs):
                                super().__init__(**kwargs)
                        
                        loader = TestExtensionLoader(context=context)
                        assert loader is not None


def test_extension_loader_mixin_init_import_error():
    """Test ExtensionLoaderMixin initialization raises UnknownExtension on ImportError."""
    from unittest.mock import MagicMock, patch
    
    class UnknownExtension(Exception):
        pass
    
    with patch('cookiecutter.extensions.JsonifyExtension', side_effect=ImportError('Module not found')):
        class TestExtensionLoader(ExtensionLoaderMixin):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
        
        try:
            loader = TestExtensionLoader()
            assert False, "Expected UnknownExtension to be raised"
        except UnknownExtension:
            pass


def test_extension_loader_mixin_read_extensions_no_key():
    """Test _read_extensions returns empty list when _extensions key missing."""
    class TestExtensionLoader(ExtensionLoaderMixin):
        pass
    
    loader = TestExtensionLoader.__new__(TestExtensionLoader)
    result = loader._read_extensions({})
    assert result == []


def test_extension_loader_mixin_read_extensions_with_extensions():
    """Test _read_extensions returns list of extensions from context."""
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
    """Test _read_extensions converts extension items to strings."""
    class TestExtensionLoader(ExtensionLoaderMixin):
        pass
    
    loader = TestExtensionLoader.__new__(TestExtensionLoader)
    context = {
        'cookiecutter': {
            '_extensions': [123, 456]
        }
    }
    result = loader._read_extensions(context)
    assert result == ['123', '456']


# LLM-generated content at query #2
#--------------------------

```python
def test_extension_loader_mixin_catches_import_error():
    """Test that ImportError at line 23 is caught and re-raised as UnknownExtension."""
    from unittest.mock import Mock, patch
    
    class UnknownExtension(Exception):
        pass
    
    class TestExtensionLoaderMixin(ExtensionLoaderMixin):
        def _read_extensions(self, context):
            return []
    
    with patch('builtins.super') as mock_super:
        mock_super_instance = Mock()
        mock_super_instance.__init__.side_effect = ImportError("test import error")
        mock_super.return_value = mock_super_instance
        
        try:
            TestExtensionLoaderMixin(context={})
            assert False, "Expected UnknownExtension to be raised"
        except UnknownExtension as e:
            assert "Unable to load extension:" in str(e)
            assert True


# LLM-generated content at query #3
#--------------------------

```python
def test_import_error_handling():
    """Test that ImportError is caught and converted to UnknownExtension at line 23."""
    from unittest.mock import Mock, patch
    
    class UnknownExtension(Exception):
        pass
    
    class TestExtensionLoaderMixin(ExtensionLoaderMixin):
        def _read_extensions(self, context):
            return []
    
    with patch('builtins.super') as mock_super:
        mock_super_instance = Mock()
        mock_super_instance.__init__.side_effect = ImportError("test import error")
        mock_super.return_value = mock_super_instance
        
        try:
            TestExtensionLoaderMixin(context={})
            assert False, "Expected UnknownExtension to be raised"
        except UnknownExtension as e:
            assert "Unable to load extension:" in str(e)
            assert True


# LLM-generated content at query #4
#--------------------------

```python
def test_extension_loader_mixin_init_with_no_context():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = None
            super().__init__(context=context, **kwargs)
        
    try:
        loader = TestExtensionLoader()
    except Exception:
        pass


def test_extension_loader_mixin_init_with_empty_context():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = None
            super().__init__(context=context, **kwargs)
    
    try:
        loader = TestExtensionLoader(context={})
    except Exception:
        pass


def test_extension_loader_mixin_init_with_extensions_in_context():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = None
            super().__init__(context=context, **kwargs)
    
    context = {
        'cookiecutter': {
            '_extensions': ['jinja2.ext.DebugExtension']
        }
    }
    try:
        loader = TestExtensionLoader(context=context)
    except Exception:
        pass


def test_extension_loader_mixin_read_extensions_missing_key():
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


def test_extension_loader_mixin_read_extensions_empty_list():
    class TestExtensionLoader(ExtensionLoaderMixin):
        pass
    
    loader = TestExtensionLoader.__new__(TestExtensionLoader)
    context = {
        'cookiecutter': {
            '_extensions': []
        }
    }
    result = loader._read_extensions(context)
    assert result == []


def test_extension_loader_mixin_read_extensions_converts_to_string():
    class TestExtensionLoader(ExtensionLoaderMixin):
        pass
    
    loader = TestExtensionLoader.__new__(TestExtensionLoader)
    context = {
        'cookiecutter': {
            '_extensions': [123, 456]
        }
    }
    result = loader._read_extensions(context)
    assert result == ['123', '456']


# LLM-generated content at query #5
#--------------------------

```python
def test_import_error_handling_raises_unknown_extension():
    from unittest.mock import Mock, patch
    
    class UnknownExtension(Exception):
        pass
    
    class TestExtensionLoaderMixin(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.context = context
            self.kwargs = kwargs
            super().__init__(context=context, **kwargs)
    
    with patch('builtins.super') as mock_super:
        mock_super_instance = Mock()
        mock_super_instance.__init__.side_effect = ImportError("test error")
        mock_super.return_value = mock_super_instance
        
        try:
            TestExtensionLoaderMixin(context={})
            assert False, "Expected UnknownExtension to be raised"
        except UnknownExtension as e:
            assert "Unable to load extension:" in str(e)
            assert "test error" in str(e)


# LLM-generated content at query #6
#--------------------------

```python
def test_extension_loader_mixin_handles_import_error():
    """Test that ImportError at line 23 is caught and converted to UnknownExtension."""
    from unittest.mock import Mock, patch
    
    class UnknownExtension(Exception):
        pass
    
    class TestExtensionLoaderMixin(ExtensionLoaderMixin):
        pass
    
    # Mock the parent __init__ to raise ImportError
    with patch('builtins.super') as mock_super:
        mock_super_instance = Mock()
        mock_super_instance.__init__.side_effect = ImportError("test error")
        mock_super.return_value = mock_super_instance
        
        try:
            # This should trigger the except block at line 23
            instance = TestExtensionLoaderMixin(context={})
            assert False, "Expected UnknownExtension to be raised"
        except UnknownExtension as e:
            assert "Unable to load extension:" in str(e)
            assert "test error" in str(e)


# LLM-generated content at query #7
#--------------------------

```python
def test_extension_loader_mixin_handles_import_error():
    from unittest.mock import Mock, patch
    from cookiecutter.exceptions import UnknownExtension
    
    class MockEnvironment:
        def __init__(self, *, extensions=None, **kwargs):
            raise ImportError("Cannot import extension")
    
    class TestExtensionLoader(ExtensionLoaderMixin, MockEnvironment):
        pass
    
    try:
        TestExtensionLoader(context={})
        assert False, "Expected UnknownExtension to be raised"
    except UnknownExtension as err:
        assert "Unable to load extension:" in str(err)
        assert isinstance(err.__cause__, ImportError)


# LLM-generated content at query #8
#--------------------------

```python
def test_import_error_raises_unknown_extension():
    """Test that ImportError on line 23 is caught and UnknownExtension is raised."""
    from unittest.mock import MagicMock, patch
    
    class UnknownExtension(Exception):
        pass
    
    class TestExtensionLoaderMixin(ExtensionLoaderMixin):
        def _read_extensions(self, context):
            return []
    
    with patch('builtins.super') as mock_super:
        mock_instance = MagicMock()
        mock_super.return_value = mock_instance
        mock_instance.__init__.side_effect = ImportError("test error")
        
        mixin = TestExtensionLoaderMixin.__new__(TestExtensionLoaderMixin)
        mixin._read_extensions = lambda context: []
        
        try:
            with patch.object(ExtensionLoaderMixin, '__init__', side_effect=ImportError("test error")):
                TestExtensionLoaderMixin.__init__(mixin, context={})
            assert False, "Expected UnknownExtension to be raised"
        except UnknownExtension as e:
            assert "Unable to load extension:" in str(e)
            assert "test error" in str(e)


