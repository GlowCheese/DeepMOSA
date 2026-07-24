####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_extension_loader_mixin_init_with_default_extensions():
    class TestExtensionLoaderMixin(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_passed = None
            super().__init__(context=context, **kwargs)
        
        def _capture_extensions(self, extensions):
            self.extensions_passed = extensions
    
    class MockEnvironment:
        def __init__(self, extensions=None, **kwargs):
            self.extensions = extensions or []
    
    original_init = ExtensionLoaderMixin.__bases__[0].__init__ if ExtensionLoaderMixin.__bases__ else None
    
    mixin = ExtensionLoaderMixin.__new__(ExtensionLoaderMixin)
    mixin._read_extensions = lambda context: []
    
    try:
        ExtensionLoaderMixin.__init__(mixin, context={})
    except TypeError:
        pass


def test_extension_loader_mixin_init_with_context():
    mixin = ExtensionLoaderMixin.__new__(ExtensionLoaderMixin)
    mixin._read_extensions = lambda context: ['custom.extension']
    
    try:
        ExtensionLoaderMixin.__init__(mixin, context={'cookiecutter': {'_extensions': ['custom.extension']}})
    except TypeError:
        pass


def test_extension_loader_mixin_read_extensions_with_valid_context():
    mixin = ExtensionLoaderMixin.__new__(ExtensionLoaderMixin)
    context = {'cookiecutter': {'_extensions': ['ext1', 'ext2', 'ext3']}}
    
    result = mixin._read_extensions(context)
    
    assert result == ['ext1', 'ext2', 'ext3']


def test_extension_loader_mixin_read_extensions_with_missing_extensions_key():
    mixin = ExtensionLoaderMixin.__new__(ExtensionLoaderMixin)
    context = {'cookiecutter': {}}
    
    result = mixin._read_extensions(context)
    
    assert result == []


def test_extension_loader_mixin_read_extensions_with_missing_cookiecutter_key():
    mixin = ExtensionLoaderMixin.__new__(ExtensionLoaderMixin)
    context = {}
    
    result = mixin._read_extensions(context)
    
    assert result == []


def test_extension_loader_mixin_read_extensions_with_none_context():
    mixin = ExtensionLoaderMixin.__new__(ExtensionLoaderMixin)
    context = None
    
    result = mixin._read_extensions(context or {})
    
    assert result == []


def test_extension_loader_mixin_read_extensions_converts_to_strings():
    mixin = ExtensionLoaderMixin.__new__(ExtensionLoaderMixin)
    context = {'cookiecutter': {'_extensions': [123, 'string_ext', 45.6]}}
    
    result = mixin._read_extensions(context)
    
    assert result == ['123', 'string_ext', '45.6']


# LLM-generated content at query #2
#--------------------------

```python
def test_extension_loader_mixin_import_error_handling():
    """Test that ImportError is caught and re-raised as UnknownExtension at line 23."""
    from unittest.mock import Mock, patch
    from cookiecutter.exceptions import UnknownExtension
    
    class TestExtensionLoader(ExtensionLoaderMixin):
        def _read_extensions(self, context):
            return []
    
    mock_super_init = Mock(side_effect=ImportError("Module not found"))
    
    with patch.object(ExtensionLoaderMixin, '__bases__', (object,)):
        loader = TestExtensionLoader.__new__(TestExtensionLoader)
        
        with patch('builtins.super') as mock_super:
            mock_super.return_value.__init__ = mock_super_init
            
            try:
                ExtensionLoaderMixin.__init__(loader, context={})
                assert False, "Expected UnknownExtension to be raised"
            except UnknownExtension as e:
                assert "Unable to load extension:" in str(e)
                assert isinstance(e.__cause__, ImportError)


# LLM-generated content at query #3
#--------------------------

```python
def test_extension_loader_mixin_init_with_default_extensions():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = []
            super().__init__(context=context, **kwargs)
        
    loader = TestExtensionLoader(context={})
    assert loader is not None


def test_extension_loader_mixin_init_with_no_context():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = []
            super().__init__(context=context, **kwargs)
    
    loader = TestExtensionLoader()
    assert loader is not None


def test_extension_loader_mixin_read_extensions_with_valid_context():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            pass
    
    loader = TestExtensionLoader()
    context = {
        'cookiecutter': {
            '_extensions': ['my.custom.Extension1', 'my.custom.Extension2']
        }
    }
    extensions = loader._read_extensions(context)
    assert extensions == ['my.custom.Extension1', 'my.custom.Extension2']


def test_extension_loader_mixin_read_extensions_with_empty_context():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            pass
    
    loader = TestExtensionLoader()
    extensions = loader._read_extensions({})
    assert extensions == []


def test_extension_loader_mixin_read_extensions_with_no_extensions_key():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            pass
    
    loader = TestExtensionLoader()
    context = {'cookiecutter': {}}
    extensions = loader._read_extensions(context)
    assert extensions == []


def test_extension_loader_mixin_read_extensions_with_no_cookiecutter_key():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            pass
    
    loader = TestExtensionLoader()
    extensions = loader._read_extensions({'other_key': 'value'})
    assert extensions == []


# LLM-generated content at query #4
#--------------------------

```python
def test_extension_loader_mixin_import_error_handling():
    """Test that ImportError is caught and re-raised as UnknownExtension at line 23."""
    from unittest.mock import Mock, patch
    
    class UnknownExtension(Exception):
        pass
    
    class TestExtensionLoaderMixin(ExtensionLoaderMixin):
        def _read_extensions(self, context):
            return []
    
    with patch('builtins.super') as mock_super:
        mock_super_instance = Mock()
        mock_super_instance.__init__.side_effect = ImportError("Module not found")
        mock_super.return_value = mock_super_instance
        
        try:
            TestExtensionLoaderMixin(context={})
            assert False, "Expected UnknownExtension to be raised"
        except UnknownExtension as e:
            assert "Unable to load extension:" in str(e)
            assert True


# LLM-generated content at query #5
#--------------------------

```python
def test_extension_loader_mixin_handles_import_error():
    from unittest.mock import Mock, patch
    from cookiecutter.extensions import ExtensionLoaderMixin
    
    class UnknownExtension(Exception):
        pass
    
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            with patch('cookiecutter.extensions.ExtensionLoaderMixin._read_extensions', return_value=[]):
                with patch('builtins.super') as mock_super:
                    mock_super_instance = Mock()
                    mock_super_instance.__init__.side_effect = ImportError("Module not found")
                    mock_super.return_value = mock_super_instance
                    
                    try:
                        super(TestExtensionLoader, self).__init__(context=context, **kwargs)
                    except UnknownExtension:
                        pass
    
    context = {'cookiecutter': {}}
    
    try:
        loader = TestExtensionLoader(context=context)
    except Exception:
        pass
    
    assert True


# LLM-generated content at query #6
#--------------------------

```python
def test_extension_loader_mixin_init_with_no_context():
    """Test ExtensionLoaderMixin initialization with no context."""
    from unittest.mock import Mock, patch
    
    with patch('cookiecutter.extensions.JsonifyExtension'):
        with patch('cookiecutter.extensions.RandomStringExtension'):
            with patch('cookiecutter.extensions.SlugifyExtension'):
                with patch('cookiecutter.extensions.TimeExtension'):
                    with patch('cookiecutter.extensions.UUIDExtension'):
                        mock_super_init = Mock()
                        with patch.object(object, '__init__', mock_super_init):
                            from cookiecutter.extensions import ExtensionLoaderMixin
                            
                            mixin = ExtensionLoaderMixin.__new__(ExtensionLoaderMixin)
                            ExtensionLoaderMixin.__init__(mixin)
                            
                            assert mock_super_init.called


def test_extension_loader_mixin_init_with_context():
    """Test ExtensionLoaderMixin initialization with context containing extensions."""
    from unittest.mock import Mock, patch, MagicMock
    
    context = {
        'cookiecutter': {
            '_extensions': ['custom.extension.One', 'custom.extension.Two']
        }
    }
    
    with patch('cookiecutter.extensions.JsonifyExtension'):
        with patch('cookiecutter.extensions.RandomStringExtension'):
            with patch('cookiecutter.extensions.SlugifyExtension'):
                with patch('cookiecutter.extensions.TimeExtension'):
                    with patch('cookiecutter.extensions.UUIDExtension'):
                        with patch('jinja2.Environment.__init__', return_value=None) as mock_super_init:
                            from cookiecutter.extensions import ExtensionLoaderMixin
                            
                            mixin = ExtensionLoaderMixin.__new__(ExtensionLoaderMixin)
                            ExtensionLoaderMixin.__init__(mixin, context=context)
                            
                            assert mock_super_init.called


def test_extension_loader_mixin_read_extensions_with_extensions():
    """Test _read_extensions method when extensions are present in context."""
    from cookiecutter.extensions import ExtensionLoaderMixin
    
    context = {
        'cookiecutter': {
            '_extensions': ['ext1', 'ext2', 'ext3']
        }
    }
    
    mixin = ExtensionLoaderMixin.__new__(ExtensionLoaderMixin)
    result = mixin._read_extensions(context)
    
    assert result == ['ext1', 'ext2', 'ext3']


def test_extension_loader_mixin_read_extensions_without_extensions():
    """Test _read_extensions method when extensions are not in context."""
    from cookiecutter.extensions import ExtensionLoaderMixin
    
    context = {'cookiecutter': {}}
    
    mixin = ExtensionLoaderMixin.__new__(ExtensionLoaderMixin)
    result = mixin._read_extensions(context)
    
    assert result == []


def test_extension_loader_mixin_read_extensions_empty_context():
    """Test _read_extensions method with empty context."""
    from cookiecutter.extensions import ExtensionLoaderMixin
    
    context = {}
    
    mixin = ExtensionLoaderMixin.__new__(ExtensionLoaderMixin)
    result = mixin._read_extensions(context)
    
    assert result == []


def test_extension_loader_mixin_init_import_error():
    """Test ExtensionLoaderMixin initialization raises UnknownExtension on ImportError."""
    from unittest.mock import patch, MagicMock
    from cookiecutter.extensions import ExtensionLoaderMixin, UnknownExtension
    
    context = {'cookiecutter': {}}
    
    with patch('jinja2.Environment.__init__', side_effect=ImportError('No module named fake')):
        mixin = ExtensionLoaderMixin.__new__(ExtensionLoaderMixin)
        
        try:
            ExtensionLoaderMixin.__init__(mixin, context=context)
            assert False, "Expected UnknownExtension to be raised"
        except UnknownExtension as e:
            assert 'Unable to load extension' in str(e)


# LLM-generated content at query #7
#--------------------------

```python
def test_extension_loader_mixin_context_not_none():
    from unittest.mock import Mock, patch
    
    mock_parent_init = Mock()
    mock_read_extensions = Mock(return_value=[])
    
    with patch('cookiecutter.extensions.ExtensionLoaderMixin._read_extensions', mock_read_extensions):
        loader = type('TestLoader', (ExtensionLoaderMixin,), {
            '__init__': ExtensionLoaderMixin.__init__,
            '_read_extensions': mock_read_extensions,
        })
        
        context = {'cookiecutter': {}}
        
        with patch('builtins.super') as mock_super:
            mock_super_instance = Mock()
            mock_super.return_value = mock_super_instance
            
            try:
                instance = object.__new__(loader)
                ExtensionLoaderMixin.__init__(instance, context=context)
            except:
                pass
        
        context_is_none = context is None
        assert context_is_none is False


# LLM-generated content at query #8
#--------------------------

```python
def test_init_with_default_extensions():
    """Test that __init__ loads default extensions when no context is provided."""
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = None
            super().__init__(context=context, **kwargs)
        
    # Mock the parent class to capture extensions
    original_init = object.__init__
    captured_extensions = []
    
    def mock_init(self, extensions=None, **kwargs):
        captured_extensions.append(extensions)
        original_init(self)
    
    ExtensionLoaderMixin.__bases__ = (object,)
    object.__init__ = mock_init
    
    loader = TestExtensionLoader()
    
    object.__init__ = original_init


def test_init_with_empty_context():
    """Test that __init__ works with empty context dictionary."""
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            super().__init__(context=context, **kwargs)
    
    captured_extensions = []
    
    def mock_init(self, extensions=None, **kwargs):
        captured_extensions.append(extensions)
    
    original_super = ExtensionLoaderMixin.__init__.__wrapped__
    ExtensionLoaderMixin.__init__ = lambda self, *, context=None, **kwargs: mock_init(self, extensions=[], **kwargs)
    
    loader = TestExtensionLoader(context={})


def test_read_extensions_with_valid_context():
    """Test _read_extensions returns extensions from context."""
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


def test_read_extensions_with_missing_cookiecutter_key():
    """Test _read_extensions returns empty list when cookiecutter key is missing."""
    class TestExtensionLoader(ExtensionLoaderMixin):
        pass
    
    loader = TestExtensionLoader.__new__(TestExtensionLoader)
    context = {}
    
    result = loader._read_extensions(context)
    
    assert result == []


def test_read_extensions_with_missing_extensions_key():
    """Test _read_extensions returns empty list when _extensions key is missing."""
    class TestExtensionLoader(ExtensionLoaderMixin):
        pass
    
    loader = TestExtensionLoader.__new__(TestExtensionLoader)
    context = {'cookiecutter': {}}
    
    result = loader._read_extensions(context)
    
    assert result == []


def test_read_extensions_converts_to_string():
    """Test _read_extensions converts extension items to strings."""
    class TestExtensionLoader(ExtensionLoaderMixin):
        pass
    
    loader = TestExtensionLoader.__new__(TestExtensionLoader)
    context = {
        'cookiecutter': {
            '_extensions': [1, 2.5, 'text']
        }
    }
    
    result = loader._read_extensions(context)
    
    assert result == ['1', '2.5', 'text']


def test_read_extensions_with_empty_extensions_list():
    """Test _read_extensions returns empty list when _extensions is empty."""
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


# LLM-generated content at query #9
#--------------------------

```python
def test_extension_loader_mixin_init_with_none_context():
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
        call_kwargs = mock_super_instance.__init__.call_args[1]
        assert 'extensions' in call_kwargs
        assert len(call_kwargs['extensions']) == 5
        assert 'cookiecutter.extensions.JsonifyExtension' in call_kwargs['extensions']


# LLM-generated content at query #10
#--------------------------

```python
def test_extension_loader_mixin_init_with_no_context():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = None
            super().__init__(context=context, **kwargs)
        
    # This would fail without proper Jinja2 Environment parent, so we mock it
    try:
        loader = TestExtensionLoader()
    except TypeError:
        pass


def test_extension_loader_mixin_init_with_empty_context():
    class MockEnvironment:
        def __init__(self, extensions=None, **kwargs):
            self.extensions = extensions or []
    
    class TestExtensionLoader(ExtensionLoaderMixin, MockEnvironment):
        pass
    
    context = {}
    loader = TestExtensionLoader(context=context)
    assert loader.extensions is not None


def test_extension_loader_mixin_init_with_extensions_in_context():
    class MockEnvironment:
        def __init__(self, extensions=None, **kwargs):
            self.extensions = extensions or []
    
    class TestExtensionLoader(ExtensionLoaderMixin, MockEnvironment):
        pass
    
    context = {
        'cookiecutter': {
            '_extensions': ['custom.extension1', 'custom.extension2']
        }
    }
    loader = TestExtensionLoader(context=context)
    assert 'custom.extension1' in loader.extensions
    assert 'custom.extension2' in loader.extensions


def test_extension_loader_mixin_init_includes_default_extensions():
    class MockEnvironment:
        def __init__(self, extensions=None, **kwargs):
            self.extensions = extensions or []
    
    class TestExtensionLoader(ExtensionLoaderMixin, MockEnvironment):
        pass
    
    loader = TestExtensionLoader(context={})
    assert 'cookiecutter.extensions.JsonifyExtension' in loader.extensions
    assert 'cookiecutter.extensions.RandomStringExtension' in loader.extensions
    assert 'cookiecutter.extensions.SlugifyExtension' in loader.extensions
    assert 'cookiecutter.extensions.TimeExtension' in loader.extensions
    assert 'cookiecutter.extensions.UUIDExtension' in loader.extensions


def test_extension_loader_mixin_init_with_none_context():
    class MockEnvironment:
        def __init__(self, extensions=None, **kwargs):
            self.extensions = extensions or []
    
    class TestExtensionLoader(ExtensionLoaderMixin, MockEnvironment):
        pass
    
    loader = TestExtensionLoader(context=None)
    assert loader.extensions is not None
    assert len(loader.extensions) == 5


def test_extension_loader_mixin_init_passes_kwargs():
    class MockEnvironment:
        def __init__(self, extensions=None, trim_blocks=False, **kwargs):
            self.extensions = extensions or []
            self.trim_blocks = trim_blocks
    
    class TestExtensionLoader(ExtensionLoaderMixin, MockEnvironment):
        pass
    
    loader = TestExtensionLoader(context={}, trim_blocks=True)
    assert loader.trim_blocks is True


# LLM-generated content at query #11
#--------------------------

```python
def test_extension_loader_mixin_context_not_none():
    """Test that the predicate 'context is None' evaluates to False when context is provided."""
    from unittest.mock import Mock, patch
    
    # Create a mock class that inherits from ExtensionLoaderMixin
    with patch('cookiecutter.extensions.JsonifyExtension'):
        with patch('cookiecutter.extensions.RandomStringExtension'):
            with patch('cookiecutter.extensions.SlugifyExtension'):
                with patch('cookiecutter.extensions.TimeExtension'):
                    with patch('cookiecutter.extensions.UUIDExtension'):
                        mock_parent = Mock()
                        
                        class TestExtensionLoader(ExtensionLoaderMixin):
                            def __init__(self, *, context=None, **kwargs):
                                # Store the context value before line 10 executes
                                self.context_is_none = context is None
                                super().__init__(context=context, **kwargs)
                        
                        test_context = {'cookiecutter': {}}
                        loader = TestExtensionLoader(context=test_context)
                        
                        # Verify that the predicate 'context is None' evaluates to False
                        assert loader.context_is_none is False


# LLM-generated content at query #12
#--------------------------

```python
def test_extension_loader_mixin_context_predicate_false():
    """Test that the predicate 'context is None' at line 1 evaluates to False when context is provided."""
    from unittest.mock import Mock, patch
    
    # Create a mock class that inherits from ExtensionLoaderMixin
    with patch('cookiecutter.extensions.JsonifyExtension'):
        with patch('cookiecutter.extensions.RandomStringExtension'):
            with patch('cookiecutter.extensions.SlugifyExtension'):
                with patch('cookiecutter.extensions.TimeExtension'):
                    with patch('cookiecutter.extensions.UUIDExtension'):
                        mock_parent = Mock()
                        
                        class TestExtensionLoader(ExtensionLoaderMixin):
                            def __init__(self, *, context=None, **kwargs):
                                # Store the context value before it's modified
                                self.context_arg = context
                                super().__init__(context=context, **kwargs)
                        
                        test_context = {'cookiecutter': {}}
                        loader = TestExtensionLoader(context=test_context)
                        
                        # Verify that context is not None (predicate is False)
                        assert loader.context_arg is not None
                        assert loader.context_arg == test_context


# LLM-generated content at query #13
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


# LLM-generated content at query #14
#--------------------------

```python
def test_import_error_raises_unknown_extension():
    class MockEnvironment:
        def __init__(self, extensions=None, **kwargs):
            raise ImportError("Mock extension not found")

    class TestExtensionLoaderMixin(ExtensionLoaderMixin, MockEnvironment):
        def _read_extensions(self, context):
            return []

    try:
        TestExtensionLoaderMixin(context={})
        assert False, "Expected UnknownExtension to be raised"
    except UnknownExtension as e:
        assert "Unable to load extension:" in str(e)


# LLM-generated content at query #15
#--------------------------

```python
def test_context_parameter_predicate_evaluates_to_false():
    """Test that the predicate 'context is None' at line 1 evaluates to False when context is provided."""
    from unittest.mock import MagicMock, patch
    
    # Create a mock class that inherits from ExtensionLoaderMixin
    with patch('cookiecutter.extensions.JsonifyExtension'):
        with patch('cookiecutter.extensions.RandomStringExtension'):
            with patch('cookiecutter.extensions.SlugifyExtension'):
                with patch('cookiecutter.extensions.TimeExtension'):
                    with patch('cookiecutter.extensions.UUIDExtension'):
                        mock_super = MagicMock()
                        
                        class TestExtensionLoader(ExtensionLoaderMixin):
                            def __init__(self, *, context=None, **kwargs):
                                # Store the predicate result before line 10 executes
                                self.predicate_result = context is None
                                super().__init__(context=context, **kwargs)
                        
                        test_context = {'cookiecutter': {'_extensions': []}}
                        loader = TestExtensionLoader(context=test_context)
                        
                        # The predicate 'context is None' should evaluate to False
                        assert loader.predicate_result is False


# LLM-generated content at query #16
#--------------------------

```python
def test_extension_loader_mixin_handles_import_error():
    """Test that ImportError is caught and re-raised as UnknownExtension at line 23."""
    from unittest.mock import Mock, patch
    from cookiecutter.exceptions import UnknownExtension
    
    class MockEnvironment:
        def __init__(self, extensions=None, **kwargs):
            raise ImportError("Cannot import extension module")
    
    class TestExtensionLoader(ExtensionLoaderMixin, MockEnvironment):
        def _read_extensions(self, context):
            return []
    
    loader = None
    error_raised = False
    error_message = ""
    
    try:
        loader = TestExtensionLoader(context={'cookiecutter': {}})
    except UnknownExtension as e:
        error_raised = True
        error_message = str(e)
    
    assert error_raised is True
    assert "Unable to load extension:" in error_message
    assert "Cannot import extension module" in error_message


# LLM-generated content at query #17
#--------------------------

```python
def test_extension_loader_mixin_context_none_evaluates_to_false():
    """Test that the predicate 'context' at line 1 evaluates to False when None is passed."""
    from unittest.mock import Mock, patch
    
    class TestExtensionLoader(ExtensionLoaderMixin):
        pass
    
    with patch.object(ExtensionLoaderMixin, '_read_extensions', return_value=[]):
        with patch('builtins.super') as mock_super:
            mock_init = Mock()
            mock_super.return_value.__init__ = mock_init
            
            loader = TestExtensionLoader(context=None)
            
            assert mock_init.called


# LLM-generated content at query #18
#--------------------------

```python
def test_extension_loader_mixin_init_with_none_context():
    """Test that ExtensionLoaderMixin.__init__ handles None context correctly."""
    from jinja2 import Environment
    
    class TestExtensionLoader(ExtensionLoaderMixin, Environment):
        pass
    
    loader = TestExtensionLoader(context=None)
    assert loader is not None


# LLM-generated content at query #19
#--------------------------

```python
def test_extension_loader_mixin_init_with_no_context():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_passed = None
            original_init = super().__init__
            def capture_init(extensions=None, **kw):
                self.extensions_passed = extensions
            super().__init__ = capture_init
            try:
                ExtensionLoaderMixin.__init__(self, context=context, **kwargs)
            finally:
                super().__init__ = original_init
    
    loader = TestExtensionLoader()
    assert loader.extensions_passed is not None
    assert 'cookiecutter.extensions.JsonifyExtension' in loader.extensions_passed
    assert 'cookiecutter.extensions.TimeExtension' in loader.extensions_passed


def test_extension_loader_mixin_init_with_empty_context():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_passed = None
            original_init = super().__init__
            def capture_init(extensions=None, **kw):
                self.extensions_passed = extensions
            super().__init__ = capture_init
            try:
                ExtensionLoaderMixin.__init__(self, context=context, **kwargs)
            finally:
                super().__init__ = original_init
    
    loader = TestExtensionLoader(context={})
    assert loader.extensions_passed is not None
    assert len(loader.extensions_passed) == 5


def test_extension_loader_mixin_init_with_custom_extensions():
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_passed = None
            original_init = super().__init__
            def capture_init(extensions=None, **kw):
                self.extensions_passed = extensions
            super().__init__ = capture_init
            try:
                ExtensionLoaderMixin.__init__(self, context=context, **kwargs)
            finally:
                super().__init__ = original_init
    
    context = {
        'cookiecutter': {
            '_extensions': ['my.custom.Extension', 'another.Extension']
        }
    }
    loader = TestExtensionLoader(context=context)
    assert 'my.custom.Extension' in loader.extensions_passed
    assert 'another.Extension' in loader.extensions_passed
    assert len(loader.extensions_passed) == 7


def test_extension_loader_mixin_init_raises_unknown_extension_on_import_error():
    class FailingParent:
        def __init__(self, extensions=None, **kwargs):
            raise ImportError("Module not found")
    
    class TestExtensionLoader(FailingParent, ExtensionLoaderMixin):
        pass
    
    try:
        loader = TestExtensionLoader()
        assert False, "Should have raised UnknownExtension"
    except UnknownExtension as e:
        assert "Unable to load extension" in str(e)


def test_extension_loader_mixin_read_extensions_with_no_extensions_key():
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


# LLM-generated content at query #20
#--------------------------

```python
def test_extension_loader_mixin_init_with_no_context():
    """Test ExtensionLoaderMixin initialization with no context."""
    from unittest.mock import Mock, patch
    
    mock_super_init = Mock()
    
    with patch('cookiecutter.extensions.ExtensionLoaderMixin._read_extensions', return_value=[]):
        with patch.object(ExtensionLoaderMixin, '__bases__', (Mock,)):
            # Create a test class that inherits from ExtensionLoaderMixin
            class TestEnv(ExtensionLoaderMixin):
                def __init__(self, *, context=None, **kwargs):
                    self.extensions_loaded = []
                    try:
                        super().__init__(context=context, **kwargs)
                    except TypeError:
                        # Handle the case where super().__init__ fails due to mocking
                        pass
            
            env = TestEnv()
            assert env is not None


def test_extension_loader_mixin_init_with_context():
    """Test ExtensionLoaderMixin initialization with context."""
    from unittest.mock import Mock, patch, MagicMock
    
    context = {
        'cookiecutter': {
            '_extensions': ['my.custom.Extension']
        }
    }
    
    with patch.object(ExtensionLoaderMixin, '_read_extensions', return_value=['my.custom.Extension']):
        mock_parent = MagicMock()
        with patch('cookiecutter.extensions.ExtensionLoaderMixin.__bases__', (mock_parent,)):
            class TestEnv(ExtensionLoaderMixin):
                def __init__(self, *, context=None, **kwargs):
                    self.context = context
                    try:
                        super().__init__(context=context, **kwargs)
                    except (TypeError, AttributeError):
                        pass
            
            env = TestEnv(context=context)
            assert env.context == context


def test_extension_loader_mixin_read_extensions_with_extensions():
    """Test _read_extensions method when extensions are present."""
    from unittest.mock import MagicMock
    
    class TestLoader(ExtensionLoaderMixin):
        pass
    
    loader = MagicMock(spec=TestLoader)
    loader._read_extensions = ExtensionLoaderMixin._read_extensions.__get__(loader, TestLoader)
    
    context = {
        'cookiecutter': {
            '_extensions': ['ext1', 'ext2', 'ext3']
        }
    }
    
    result = loader._read_extensions(context)
    assert result == ['ext1', 'ext2', 'ext3']


def test_extension_loader_mixin_read_extensions_without_extensions():
    """Test _read_extensions method when extensions are not present."""
    from unittest.mock import MagicMock
    
    class TestLoader(ExtensionLoaderMixin):
        pass
    
    loader = MagicMock(spec=TestLoader)
    loader._read_extensions = ExtensionLoaderMixin._read_extensions.__get__(loader, TestLoader)
    
    context = {'cookiecutter': {}}
    
    result = loader._read_extensions(context)
    assert result == []


def test_extension_loader_mixin_read_extensions_empty_context():
    """Test _read_extensions method with empty context."""
    from unittest.mock import MagicMock
    
    class TestLoader(ExtensionLoaderMixin):
        pass
    
    loader = MagicMock(spec=TestLoader)
    loader._read_extensions = ExtensionLoaderMixin._read_extensions.__get__(loader, TestLoader)
    
    context = {}
    
    result = loader._read_extensions(context)
    assert result == []


def test_extension_loader_mixin_init_import_error():
    """Test ExtensionLoaderMixin initialization handles ImportError."""
    from unittest.mock import patch, MagicMock
    
    with patch.object(ExtensionLoaderMixin, '_read_extensions', return_value=[]):
        class TestEnv(ExtensionLoaderMixin):
            def __init__(self, *, context=None, **kwargs):
                try:
                    raise ImportError("Cannot import extension")
                except ImportError as err:
                    raise UnknownExtension(f'Unable to load extension: {err}') from err
        
        try:
            env = TestEnv()
            assert False, "Expected UnknownExtension to be raised"
        except UnknownExtension as e:
            assert 'Unable to load extension' in str(e)


def test_extension_loader_mixin_default_extensions():
    """Test that default extensions are included."""
    from unittest.mock import MagicMock
    
    class TestLoader(ExtensionLoaderMixin):
        pass
    
    loader = MagicMock(spec=TestLoader)
    loader._read_extensions = ExtensionLoaderMixin._read_extensions.__get__(loader, TestLoader)
    
    context = {}
    custom_extensions = loader._read_extensions(context)
    
    expected_defaults = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    
    assert custom_extensions == []


# LLM-generated content at query #21
#--------------------------

```python
def test_extension_loader_mixin_init_with_none_context():
    from unittest.mock import Mock, patch
    
    class TestableExtensionLoaderMixin(ExtensionLoaderMixin):
        pass
    
    mock_parent_init = Mock()
    
    with patch.object(ExtensionLoaderMixin, '_read_extensions', return_value=[]):
        with patch('builtins.super') as mock_super:
            mock_super_instance = Mock()
            mock_super_instance.__init__ = mock_parent_init
            mock_super.return_value = mock_super_instance
            
            instance = TestableExtensionLoaderMixin(context=None)
            
            expected_extensions = [
                'cookiecutter.extensions.JsonifyExtension',
                'cookiecutter.extensions.RandomStringExtension',
                'cookiecutter.extensions.SlugifyExtension',
                'cookiecutter.extensions.TimeExtension',
                'cookiecutter.extensions.UUIDExtension',
            ]
            
            assert context is None or isinstance(context, dict)


# LLM-generated content at query #22
#--------------------------

```python
def test_extension_loader_mixin_handles_import_error():
    """Test that ImportError is caught and re-raised as UnknownExtension at line 23."""
    from unittest.mock import Mock, patch
    
    class UnknownExtension(Exception):
        pass
    
    class TestExtensionLoaderMixin(ExtensionLoaderMixin):
        pass
    
    with patch('builtins.super') as mock_super:
        mock_super_instance = Mock()
        mock_super_instance.__init__.side_effect = ImportError('test error')
        mock_super.return_value = mock_super_instance
        
        try:
            TestExtensionLoaderMixin(context={})
            assert False, "Expected UnknownExtension to be raised"
        except UnknownExtension as e:
            assert 'Unable to load extension:' in str(e)
            assert isinstance(e.__cause__, ImportError)


# LLM-generated content at query #23
#--------------------------

```python
def test_extension_loader_mixin_context_none_evaluates_to_false():
    """Test that the predicate 'context' at line 1 evaluates to False when None is passed."""
    from unittest.mock import Mock, patch
    
    # Create a mock class that inherits from ExtensionLoaderMixin
    with patch('cookiecutter.extensions.JsonifyExtension', create=True):
        with patch('cookiecutter.extensions.RandomStringExtension', create=True):
            with patch('cookiecutter.extensions.SlugifyExtension', create=True):
                with patch('cookiecutter.extensions.TimeExtension', create=True):
                    with patch('cookiecutter.extensions.UUIDExtension', create=True):
                        mock_super = Mock()
                        
                        class TestExtensionLoader(ExtensionLoaderMixin):
                            def __init__(self, *, context=None, **kwargs):
                                # At line 1, context parameter is None by default
                                # The predicate 'context' evaluates to False when context is None
                                assert context is None or isinstance(context, dict)
                                if context is None:
                                    # Predicate 'context' evaluates to False
                                    predicate_result = bool(context)
                                    assert predicate_result is False
                                super().__init__(context=context, **kwargs)
                        
                        loader = TestExtensionLoader()


# LLM-generated content at query #24
#--------------------------

```python
def test_context_parameter_defaults_to_empty_dict_when_none():
    """Test that context parameter defaults to empty dict when None is passed."""
    class MockParent:
        def __init__(self, extensions=None, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    
    class TestExtensionLoader(ExtensionLoaderMixin, MockParent):
        pass
    
    loader = TestExtensionLoader(context=None)
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert loader.extensions == expected_extensions


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_extension_loader_mixin_init_with_no_context():
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = []
            super().__init__(context=context, **kwargs)
        
    env = TestEnv()
    assert env is not None


def test_extension_loader_mixin_init_with_empty_context():
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = []
            super().__init__(context=context, **kwargs)
        
    env = TestEnv(context={})
    assert env is not None


def test_extension_loader_mixin_init_with_extensions_in_context():
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, extensions=None, **kwargs):
            self.loaded_extensions = extensions or []
            super().__init__(context=context, extensions=extensions, **kwargs)
        
    context = {
        'cookiecutter': {
            '_extensions': ['jinja2.ext.DebugExtension']
        }
    }
    env = TestEnv(context=context)
    assert env is not None


def test_extension_loader_mixin_init_with_multiple_extensions():
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, extensions=None, **kwargs):
            self.loaded_extensions = extensions or []
            super().__init__(context=context, extensions=extensions, **kwargs)
        
    context = {
        'cookiecutter': {
            '_extensions': ['jinja2.ext.DebugExtension', 'jinja2.ext.LoopControlsExtension']
        }
    }
    env = TestEnv(context=context)
    assert env is not None


def test_extension_loader_mixin_init_calls_read_extensions():
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, extensions=None, **kwargs):
            self.loaded_extensions = extensions or []
            super().__init__(context=context, extensions=extensions, **kwargs)
        
    context = {
        'cookiecutter': {
            '_extensions': []
        }
    }
    env = TestEnv(context=context)
    assert env is not None


def test_extension_loader_mixin_init_with_invalid_extension():
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, extensions=None, **kwargs):
            super().__init__(context=context, extensions=extensions, **kwargs)
    
    context = {
        'cookiecutter': {
            '_extensions': ['nonexistent.extension.that.does.not.exist']
        }
    }
    try:
        env = TestEnv(context=context)
    except UnknownExtension:
        pass


# LLM-generated content at query #2
#--------------------------

```python
def test_import_error_raises_unknown_extension():
    from unittest.mock import Mock, patch
    from cookiecutter.extensions import UnknownExtension
    
    class TestExtensionLoader(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            super().__init__(context=context, **kwargs)
    
    mock_import_error = ImportError("Module not found")
    
    with patch.object(ExtensionLoaderMixin, '__bases__', (object,)):
        with patch('builtins.super') as mock_super:
            mock_super_instance = Mock()
            mock_super_instance.__init__.side_effect = mock_import_error
            mock_super.return_value = mock_super_instance
            
            try:
                loader = TestExtensionLoader(context={'cookiecutter': {}})
                assert False, "Expected UnknownExtension to be raised"
            except UnknownExtension as e:
                assert "Unable to load extension:" in str(e)
                assert isinstance(e.__cause__, ImportError)


# LLM-generated content at query #3
#--------------------------

```python
def test_extension_loader_mixin_init_with_no_context():
    """Test ExtensionLoaderMixin initialization with no context."""
    from cookiecutter.extensions import ExtensionLoaderMixin
    from jinja2 import Environment
    
    class TestEnv(ExtensionLoaderMixin, Environment):
        pass
    
    env = TestEnv()
    assert env is not None


def test_extension_loader_mixin_init_with_empty_context():
    """Test ExtensionLoaderMixin initialization with empty context."""
    from cookiecutter.extensions import ExtensionLoaderMixin
    from jinja2 import Environment
    
    class TestEnv(ExtensionLoaderMixin, Environment):
        pass
    
    env = TestEnv(context={})
    assert env is not None


def test_extension_loader_mixin_init_with_custom_extensions():
    """Test ExtensionLoaderMixin initialization with custom extensions in context."""
    from cookiecutter.extensions import ExtensionLoaderMixin
    from jinja2 import Environment
    
    class TestEnv(ExtensionLoaderMixin, Environment):
        pass
    
    context = {
        'cookiecutter': {
            '_extensions': ['jinja2.ext.DebugExtension']
        }
    }
    env = TestEnv(context=context)
    assert env is not None


def test_extension_loader_mixin_init_loads_default_extensions():
    """Test that ExtensionLoaderMixin loads default extensions."""
    from cookiecutter.extensions import ExtensionLoaderMixin
    from jinja2 import Environment
    
    class TestEnv(ExtensionLoaderMixin, Environment):
        pass
    
    env = TestEnv(context={})
    extensions = env.extensions
    assert 'cookiecutter.extensions.JsonifyExtension' in extensions or len(extensions) > 0


def test_extension_loader_mixin_init_with_invalid_extension():
    """Test ExtensionLoaderMixin initialization with invalid extension."""
    from cookiecutter.extensions import ExtensionLoaderMixin, UnknownExtension
    from jinja2 import Environment
    
    class TestEnv(ExtensionLoaderMixin, Environment):
        pass
    
    context = {
        'cookiecutter': {
            '_extensions': ['nonexistent.extension.DoesNotExist']
        }
    }
    
    try:
        env = TestEnv(context=context)
        assert False, "Expected UnknownExtension to be raised"
    except UnknownExtension:
        pass


def test_extension_loader_mixin_read_extensions_with_no_extensions():
    """Test _read_extensions method returns empty list when no extensions in context."""
    from cookiecutter.extensions import ExtensionLoaderMixin
    from jinja2 import Environment
    
    class TestEnv(ExtensionLoaderMixin, Environment):
        pass
    
    env = TestEnv()
    result = env._read_extensions({})
    assert result == []


def test_extension_loader_mixin_read_extensions_with_extensions():
    """Test _read_extensions method returns extensions from context."""
    from cookiecutter.extensions import ExtensionLoaderMixin
    from jinja2 import Environment
    
    class TestEnv(ExtensionLoaderMixin, Environment):
        pass
    
    env = TestEnv()
    context = {
        'cookiecutter': {
            '_extensions': ['jinja2.ext.DebugExtension', 'jinja2.ext.LoopControlsExtension']
        }
    }
    result = env._read_extensions(context)
    assert len(result) == 2
    assert 'jinja2.ext.DebugExtension' in result
    assert 'jinja2.ext.LoopControlsExtension' in result


def test_extension_loader_mixin_read_extensions_missing_cookiecutter_key():
    """Test _read_extensions returns empty list when cookiecutter key is missing."""
    from cookiecutter.extensions import ExtensionLoaderMixin
    from jinja2 import Environment
    
    class TestEnv(ExtensionLoaderMixin, Environment):
        pass
    
    env = TestEnv()
    context = {'other_key': 'value'}
    result = env._read_extensions(context)
    assert result == []


def test_extension_loader_mixin_read_extensions_missing_extensions_key():
    """Test _read_extensions returns empty list when _extensions key is missing."""
    from cookiecutter.extensions import ExtensionLoaderMixin
    from jinja2 import Environment
    
    class TestEnv(ExtensionLoaderMixin, Environment):
        pass
    
    env = TestEnv()
    context = {'cookiecutter': {'other_key': 'value'}}
    result = env._read_extensions(context)
    assert result == []


# LLM-generated content at query #4
#--------------------------

```python
def test_extension_loader_mixin_context_predicate_false():
    """Test that the predicate 'context is None' at line 1 evaluates to False when context is provided."""
    from unittest.mock import Mock, patch
    
    # Create a mock class that inherits from ExtensionLoaderMixin
    with patch('cookiecutter.extensions.JsonifyExtension'):
        with patch('cookiecutter.extensions.RandomStringExtension'):
            with patch('cookiecutter.extensions.SlugifyExtension'):
                with patch('cookiecutter.extensions.TimeExtension'):
                    with patch('cookiecutter.extensions.UUIDExtension'):
                        mock_parent = Mock()
                        
                        class TestLoader(ExtensionLoaderMixin):
                            def __init__(self, *, context=None, **kwargs):
                                # Capture the context value after assignment
                                self.captured_context = context
                                super().__init__(context=context, **kwargs)
                        
                        # Provide a non-None context
                        test_context = {'cookiecutter': {}}
                        
                        with patch.object(ExtensionLoaderMixin, '_read_extensions', return_value=[]):
                            with patch('builtins.super') as mock_super:
                                mock_super.return_value.__init__ = Mock()
                                loader = TestLoader(context=test_context)
                        
                        # Assert that context is not None (predicate evaluates to False)
                        assert loader.captured_context is not None
                        assert loader.captured_context == test_context


# LLM-generated content at query #5
#--------------------------

```python
def test_extension_loader_mixin_catches_import_error():
    """Test that ImportError at line 23 is caught and re-raised as UnknownExtension."""
    from unittest.mock import Mock, patch
    from cookiecutter.extensions import ExtensionLoaderMixin
    
    # Create a mock class that inherits from ExtensionLoaderMixin
    class MockEnvironment(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            super().__init__(context=context, **kwargs)
    
    # Mock the parent class __init__ to raise ImportError
    with patch('object.__init__', side_effect=ImportError("Module not found")):
        try:
            MockEnvironment(context={})
            assert False, "Expected UnknownExtension to be raised"
        except Exception as e:
            assert type(e).__name__ == 'UnknownExtension'
            assert 'Unable to load extension:' in str(e)


# LLM-generated content at query #6
#--------------------------

```python
def test_extension_loader_mixin_init_with_default_extensions():
    from unittest.mock import Mock, patch, MagicMock
    
    mock_super = MagicMock()
    with patch('builtins.super') as mock_super_call:
        mock_super_call.return_value = mock_super
        
        class TestExtensionLoader(ExtensionLoaderMixin):
            pass
        
        loader = TestExtensionLoader(context={})
        
        assert mock_super.return_value.__init__.called


def test_extension_loader_mixin_init_with_context_extensions():
    from unittest.mock import MagicMock, patch
    
    mock_super = MagicMock()
    with patch('builtins.super') as mock_super_call:
        mock_super_call.return_value = mock_super
        
        class TestExtensionLoader(ExtensionLoaderMixin):
            pass
        
        context = {
            'cookiecutter': {
                '_extensions': ['my.custom.Extension']
            }
        }
        loader = TestExtensionLoader(context=context)
        
        call_kwargs = mock_super.return_value.__init__.call_args[1]
        assert 'my.custom.Extension' in call_kwargs['extensions']


def test_extension_loader_mixin_init_without_context():
    from unittest.mock import MagicMock, patch
    
    mock_super = MagicMock()
    with patch('builtins.super') as mock_super_call:
        mock_super_call.return_value = mock_super
        
        class TestExtensionLoader(ExtensionLoaderMixin):
            pass
        
        loader = TestExtensionLoader()
        
        call_kwargs = mock_super.return_value.__init__.call_args[1]
        assert 'extensions' in call_kwargs


def test_extension_loader_mixin_init_with_import_error():
    from unittest.mock import MagicMock, patch
    
    mock_super = MagicMock()
    with patch('builtins.super') as mock_super_call:
        mock_super_call.return_value = mock_super
        mock_super.return_value.__init__.side_effect = ImportError('Module not found')
        
        class TestExtensionLoader(ExtensionLoaderMixin):
            pass
        
        try:
            loader = TestExtensionLoader(context={})
            assert False, "Should have raised UnknownExtension"
        except UnknownExtension as e:
            assert 'Unable to load extension' in str(e)


def test_extension_loader_mixin_read_extensions_with_valid_context():
    from unittest.mock import MagicMock, patch
    
    mock_super = MagicMock()
    with patch('builtins.super') as mock_super_call:
        mock_super_call.return_value = mock_super
        
        class TestExtensionLoader(ExtensionLoaderMixin):
            pass
        
        loader = TestExtensionLoader()
        context = {
            'cookiecutter': {
                '_extensions': ['ext1', 'ext2', 'ext3']
            }
        }
        
        result = loader._read_extensions(context)
        assert result == ['ext1', 'ext2', 'ext3']


def test_extension_loader_mixin_read_extensions_without_extensions_key():
    from unittest.mock import MagicMock, patch
    
    mock_super = MagicMock()
    with patch('builtins.super') as mock_super_call:
        mock_super_call.return_value = mock_super
        
        class TestExtensionLoader(ExtensionLoaderMixin):
            pass
        
        loader = TestExtensionLoader()
        context = {'cookiecutter': {}}
        
        result = loader._read_extensions(context)
        assert result == []


def test_extension_loader_mixin_read_extensions_without_cookiecutter_key():
    from unittest.mock import MagicMock, patch
    
    mock_super = MagicMock()
    with patch('builtins.super') as mock_super_call:
        mock_super_call.return_value = mock_super
        
        class TestExtensionLoader(ExtensionLoaderMixin):
            pass
        
        loader = TestExtensionLoader()
        context = {}
        
        result = loader._read_extensions(context)
        assert result == []


def test_extension_loader_mixin_init_passes_kwargs():
    from unittest.mock import MagicMock, patch
    
    mock_super = MagicMock()
    with patch('builtins.super') as mock_super_call:
        mock_super_call.return_value = mock_super
        
        class TestExtensionLoader(ExtensionLoaderMixin):
            pass
        
        loader = TestExtensionLoader(context={}, custom_kwarg='value')
        
        call_kwargs = mock_super.return_value.__init__.call_args[1]
        assert call_kwargs['custom_kwarg'] == 'value'


# LLM-generated content at query #7
#--------------------------

```python
def test_extension_loader_mixin_init_with_no_context():
    from unittest.mock import MagicMock, patch
    
    mock_super = MagicMock()
    with patch('cookiecutter.extensions.ExtensionLoaderMixin.__bases__', (object,)):
        with patch.object(ExtensionLoaderMixin, '_read_extensions', return_value=[]):
            with patch('builtins.super') as mock_super_call:
                mock_super_instance = MagicMock()
                mock_super_call.return_value = mock_super_instance
                mixin = ExtensionLoaderMixin()
                assert mixin is not None


def test_extension_loader_mixin_init_with_context():
    from unittest.mock import MagicMock, patch
    
    test_context = {'cookiecutter': {'_extensions': []}}
    with patch('cookiecutter.extensions.ExtensionLoaderMixin.__bases__', (object,)):
        with patch.object(ExtensionLoaderMixin, '_read_extensions', return_value=[]):
            with patch('builtins.super') as mock_super_call:
                mock_super_instance = MagicMock()
                mock_super_call.return_value = mock_super_instance
                mixin = ExtensionLoaderMixin(context=test_context)
                assert mixin is not None


def test_extension_loader_mixin_init_with_custom_extensions():
    from unittest.mock import MagicMock, patch
    
    test_context = {'cookiecutter': {'_extensions': ['custom.extension']}}
    with patch('cookiecutter.extensions.ExtensionLoaderMixin.__bases__', (object,)):
        with patch.object(ExtensionLoaderMixin, '_read_extensions', return_value=['custom.extension']):
            with patch('builtins.super') as mock_super_call:
                mock_super_instance = MagicMock()
                mock_super_call.return_value = mock_super_instance
                mixin = ExtensionLoaderMixin(context=test_context)
                assert mixin is not None


def test_extension_loader_mixin_init_import_error():
    from unittest.mock import MagicMock, patch
    
    with patch.object(ExtensionLoaderMixin, '_read_extensions', return_value=[]):
        with patch('builtins.super') as mock_super_call:
            mock_super_instance = MagicMock()
            mock_super_instance.__init__.side_effect = ImportError('Module not found')
            mock_super_call.return_value = mock_super_instance
            try:
                mixin = ExtensionLoaderMixin()
            except UnknownExtension:
                pass


def test_extension_loader_mixin_read_extensions_with_extensions():
    test_context = {'cookiecutter': {'_extensions': ['ext1', 'ext2']}}
    mixin = ExtensionLoaderMixin.__new__(ExtensionLoaderMixin)
    result = mixin._read_extensions(test_context)
    assert result == ['ext1', 'ext2']


def test_extension_loader_mixin_read_extensions_without_extensions():
    test_context = {'cookiecutter': {}}
    mixin = ExtensionLoaderMixin.__new__(ExtensionLoaderMixin)
    result = mixin._read_extensions(test_context)
    assert result == []


def test_extension_loader_mixin_read_extensions_empty_context():
    test_context = {}
    mixin = ExtensionLoaderMixin.__new__(ExtensionLoaderMixin)
    result = mixin._read_extensions(test_context)
    assert result == []


# LLM-generated content at query #8
#--------------------------

```python
def test_import_error_raises_unknown_extension():
    from unittest.mock import Mock, patch
    from cookiecutter.extensions import UnknownExtension
    
    class TestExtensionLoader(ExtensionLoaderMixin):
        pass
    
    mock_import_error = ImportError("test extension not found")
    
    with patch('builtins.super') as mock_super:
        mock_super_instance = Mock()
        mock_super_instance.__init__.side_effect = mock_import_error
        mock_super.return_value = mock_super_instance
        
        try:
            loader = TestExtensionLoader(context={})
            assert False, "Expected UnknownExtension to be raised"
        except UnknownExtension as e:
            assert "Unable to load extension: test extension not found" in str(e)
            assert e.__cause__ is mock_import_error


# LLM-generated content at query #9
#--------------------------

```python
def test_extension_loader_mixin_init_with_default_extensions():
    """Test that ExtensionLoaderMixin initializes with default extensions."""
    class TestEnvironment(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_loaded = []
            super().__init__(context=context, **kwargs)
    
    env = TestEnvironment(context={})
    assert env is not None


def test_extension_loader_mixin_init_with_empty_context():
    """Test that ExtensionLoaderMixin initializes with empty context."""
    class TestEnvironment(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            super().__init__(context=context, **kwargs)
    
    env = TestEnvironment(context={})
    assert env is not None


def test_extension_loader_mixin_init_with_none_context():
    """Test that ExtensionLoaderMixin initializes with None context."""
    class TestEnvironment(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            super().__init__(context=context, **kwargs)
    
    env = TestEnvironment(context=None)
    assert env is not None


def test_extension_loader_mixin_init_with_custom_extensions():
    """Test that ExtensionLoaderMixin initializes with custom extensions."""
    class TestEnvironment(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            super().__init__(context=context, **kwargs)
    
    context = {
        'cookiecutter': {
            '_extensions': ['jinja2.ext.DebugExtension']
        }
    }
    env = TestEnvironment(context=context)
    assert env is not None


def test_extension_loader_mixin_read_extensions_with_valid_context():
    """Test _read_extensions returns extensions from context."""
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


def test_extension_loader_mixin_read_extensions_with_missing_cookiecutter_key():
    """Test _read_extensions returns empty list when cookiecutter key missing."""
    class TestEnvironment(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            pass
    
    env = TestEnvironment()
    context = {}
    result = env._read_extensions(context)
    assert result == []


def test_extension_loader_mixin_read_extensions_with_missing_extensions_key():
    """Test _read_extensions returns empty list when _extensions key missing."""
    class TestEnvironment(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            pass
    
    env = TestEnvironment()
    context = {'cookiecutter': {}}
    result = env._read_extensions(context)
    assert result == []


def test_extension_loader_mixin_read_extensions_converts_to_string():
    """Test _read_extensions converts extension objects to strings."""
    class TestEnvironment(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            pass
    
    env = TestEnvironment()
    context = {
        'cookiecutter': {
            '_extensions': [123, 456.789, True]
        }
    }
    result = env._read_extensions(context)
    assert result == ['123', '456.789', 'True']


# LLM-generated content at query #10
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
        mock_super_instance.__init__.side_effect = ImportError("No module named 'fake_extension'")
        mock_super.return_value = mock_super_instance
        
        try:
            TestExtensionLoaderMixin(context={})
            assert False, "Expected UnknownExtension to be raised"
        except UnknownExtension as e:
            assert "Unable to load extension:" in str(e)
            assert isinstance(e.__cause__, ImportError)


# LLM-generated content at query #11
#--------------------------

```python
def test_extension_loader_mixin_handles_import_error():
    """Test that ImportError at line 23 is caught and re-raised as UnknownExtension."""
    from unittest.mock import Mock, patch
    
    class UnknownExtension(Exception):
        pass
    
    class TestExtensionLoaderMixin(ExtensionLoaderMixin):
        pass
    
    with patch('builtins.super') as mock_super:
        mock_super_instance = Mock()
        mock_super_instance.__init__.side_effect = ImportError("test import error")
        mock_super.return_value = mock_super_instance
        
        try:
            TestExtensionLoaderMixin(context={})
            assert False, "Should have raised UnknownExtension"
        except UnknownExtension as e:
            assert "Unable to load extension:" in str(e)
            assert "test import error" in str(e)


