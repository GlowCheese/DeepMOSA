####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_ExtensionLoaderMixin():
    class TestEnvironment:
        def __init__(self, extensions=None, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs

    class TestExtensionLoaderMixin(ExtensionLoaderMixin, TestEnvironment):
        pass

    # Test with no context
    env = TestExtensionLoaderMixin()
    assert env.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with empty context
    env = TestExtensionLoaderMixin(context={})
    assert env.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with context containing no extensions
    env = TestExtensionLoaderMixin(context={'cookiecutter': {'other_key': 'value'}})
    assert env.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with context containing extensions
    env = TestExtensionLoaderMixin(context={
        'cookiecutter': {
            '_extensions': ['my_ext.Extension1', 'other_ext.Extension2']
        }
    })
    assert env.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my_ext.Extension1',
        'other_ext.Extension2'
    ]

    # Test with context containing extensions as non-strings
    env = TestExtensionLoaderMixin(context={
        'cookiecutter': {
            '_extensions': [123, 456.789]
        }
    })
    assert env.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        '123',
        '456.789'
    ]

    # Test that kwargs are passed through
    env = TestExtensionLoaderMixin(context={}, trim_blocks=True, lstrip_blocks=True)
    assert env.kwargs == {'trim_blocks': True, 'lstrip_blocks': True}
    assert env.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test ImportError handling
    class FailingEnvironment:
        def __init__(self, extensions=None, **kwargs):
            raise ImportError("No module named 'invalid_ext'")

    class FailingExtensionLoaderMixin(ExtensionLoaderMixin, FailingEnvironment):
        pass

    try:
        FailingExtensionLoaderMixin(context={})
        assert False, "Should have raised UnknownExtension"
    except UnknownExtension as e:
        assert "Unable to load extension: No module named 'invalid_ext'" in str(e)


# LLM-generated content at query #2
#--------------------------

```python
def test_ExtensionLoaderMixin():
    class TestEnvironment:
        def __init__(self, extensions=None, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs

    class TestClass(ExtensionLoaderMixin, TestEnvironment):
        pass

    # Test with no context
    obj = TestClass()
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with empty context
    obj = TestClass(context={})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with context but no extensions
    obj = TestClass(context={'cookiecutter': {'other_key': 'value'}})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with custom extensions
    custom_exts = ['my_ext.Extension1', 'other_ext.Extension2']
    obj = TestClass(context={'cookiecutter': {'_extensions': custom_exts}})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my_ext.Extension1',
        'other_ext.Extension2',
    ]

    # Test with non-string extensions (should be converted to string)
    custom_exts = ['ext1', 123, True]
    obj = TestClass(context={'cookiecutter': {'_extensions': custom_exts}})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'ext1',
        '123',
        'True',
    ]

    # Test that kwargs are passed to parent
    obj = TestClass(context={}, some_arg='value', another_arg=42)
    assert obj.kwargs == {'some_arg': 'value', 'another_arg': 42}

    # Test ImportError handling
    class FailingEnvironment:
        def __init__(self, extensions=None, **kwargs):
            raise ImportError("No module named 'invalid_ext'")

    class FailingTestClass(ExtensionLoaderMixin, FailingEnvironment):
        pass

    try:
        FailingTestClass(context={})
        assert False, "Should have raised UnknownExtension"
    except UnknownExtension as e:
        assert "Unable to load extension: No module named 'invalid_ext'" in str(e)


# LLM-generated content at query #3
#--------------------------

```python
def test_ExtensionLoaderMixin():
    # Test 1: Initialize with no context
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.called_super = False
            self.extensions_passed = None
            self.kwargs_passed = None
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, extensions, **kwargs):
            self.called_super = True
            self.extensions_passed = extensions
            self.kwargs_passed = kwargs
    
    # Replace the parent's __init__ with our mock
    import cookiecutter.environment
    original_super = cookiecutter.environment.ExtensionLoaderMixin.__init__.__closure__[0].cell_contents
    cookiecutter.environment.ExtensionLoaderMixin.__init__.__closure__[0].cell_contents = TestEnv.__init__mock
    
    try:
        env = TestEnv()
        assert env.called_super
        assert len(env.extensions_passed) == 5
        assert 'cookiecutter.extensions.JsonifyExtension' in env.extensions_passed
        assert 'cookiecutter.extensions.RandomStringExtension' in env.extensions_passed
        assert 'cookiecutter.extensions.SlugifyExtension' in env.extensions_passed
        assert 'cookiecutter.extensions.TimeExtension' in env.extensions_passed
        assert 'cookiecutter.extensions.UUIDExtension' in env.extensions_passed
    finally:
        # Restore original
        cookiecutter.environment.ExtensionLoaderMixin.__init__.__closure__[0].cell_contents = original_super
    
    # Test 2: Initialize with empty context
    class TestEnv2(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.called_super = False
            self.extensions_passed = None
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, extensions, **kwargs):
            self.called_super = True
            self.extensions_passed = extensions
    
    cookiecutter.environment.ExtensionLoaderMixin.__init__.__closure__[0].cell_contents = TestEnv2.__init__mock
    
    try:
        env = TestEnv2(context={})
        assert env.called_super
        assert len(env.extensions_passed) == 5
    finally:
        cookiecutter.environment.ExtensionLoaderMixin.__init__.__closure__[0].cell_contents = original_super
    
    # Test 3: Initialize with extensions in context
    class TestEnv3(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.called_super = False
            self.extensions_passed = None
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, extensions, **kwargs):
            self.called_super = True
            self.extensions_passed = extensions
    
    cookiecutter.environment.ExtensionLoaderMixin.__init__.__closure__[0].cell_contents = TestEnv3.__init__mock
    
    try:
        context = {
            'cookiecutter': {
                '_extensions': ['my_ext.Extension1', 'other_ext.Extension2']
            }
        }
        env = TestEnv3(context=context)
        assert env.called_super
        assert len(env.extensions_passed) == 7
        assert 'cookiecutter.extensions.JsonifyExtension' in env.extensions_passed
        assert 'my_ext.Extension1' in env.extensions_passed
        assert 'other_ext.Extension2' in env.extensions_passed
    finally:
        cookiecutter.environment.ExtensionLoaderMixin.__init__.__closure__[0].cell_contents = original_super
    
    # Test 4: Initialize with non-list extensions in context
    class TestEnv4(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.called_super = False
            self.extensions_passed = None
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, extensions, **kwargs):
            self.called_super = True
            self.extensions_passed = extensions
    
    cookiecutter.environment.ExtensionLoaderMixin.__init__.__closure__[0].cell_contents = TestEnv4.__init__mock
    
    try:
        context = {
            'cookiecutter': {
                '_extensions': 'my_ext.Extension1'  # String instead of list
            }
        }
        env = TestEnv4(context=context)
        assert env.called_super
        assert len(env.extensions_passed) == 6
        assert 'my_ext.Extension1' in env.extensions_passed
    finally:
        cookiecutter.environment.ExtensionLoaderMixin.__init__.__closure__[0].cell_contents = original_super
    
    # Test 5: Test ImportError handling
    class TestEnv5(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.exception_raised = None
            try:
                super().__init__(context=context, **kwargs)
            except Exception as e:
                self.exception_raised = e
        
        def __init__mock(self, extensions, **kwargs):
            raise ImportError("No module named 'invalid_ext'")
    
    cookiecutter.environment.ExtensionLoaderMixin.__init__.__closure__[0].cell_contents = TestEnv5.__init__mock
    
    try:
        env = TestEnv5()
        assert env.exception_raised is not None
        assert isinstance(env.exception_raised, UnknownExtension)
        assert "Unable to load extension" in str(env.exception_raised)
    finally:
        cookiecutter.environment.ExtensionLoaderMixin.__init__.__closure__[0].cell_contents = original_super


# LLM-generated content at query #4
#--------------------------

```python
def test_ExtensionLoaderMixin():
    # Test 1: Initialize with no context
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.called_super = False
            self.extensions_passed = None
            self.kwargs_passed = None
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, extensions=None, **kwargs):
            self.called_super = True
            self.extensions_passed = extensions
            self.kwargs_passed = kwargs
    
    # Replace the super().__init__ call
    import cookiecutter.environment
    original_init = TestEnv.__mro__[1].__init__
    TestEnv.__mro__[1].__init__ = TestEnv.__init__mock
    
    try:
        env = TestEnv()
        assert env.called_super is True
        assert len(env.extensions_passed) == 5
        assert 'cookiecutter.extensions.JsonifyExtension' in env.extensions_passed
        assert 'cookiecutter.extensions.RandomStringExtension' in env.extensions_passed
        assert 'cookiecutter.extensions.SlugifyExtension' in env.extensions_passed
        assert 'cookiecutter.extensions.TimeExtension' in env.extensions_passed
        assert 'cookiecutter.extensions.UUIDExtension' in env.extensions_passed
    finally:
        TestEnv.__mro__[1].__init__ = original_init
    
    # Test 2: Initialize with empty context
    class TestEnv2(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.called_super = False
            self.extensions_passed = None
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, extensions=None, **kwargs):
            self.called_super = True
            self.extensions_passed = extensions
    
    TestEnv2.__mro__[1].__init__ = TestEnv2.__init__mock
    
    try:
        env = TestEnv2(context={})
        assert env.called_super is True
        assert len(env.extensions_passed) == 5
    finally:
        TestEnv2.__mro__[1].__init__ = original_init
    
    # Test 3: Initialize with context containing extensions
    class TestEnv3(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.called_super = False
            self.extensions_passed = None
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, extensions=None, **kwargs):
            self.called_super = True
            self.extensions_passed = extensions
    
    TestEnv3.__mro__[1].__init__ = TestEnv3.__init__mock
    
    try:
        context = {
            'cookiecutter': {
                '_extensions': ['myextension.Extension1', 'myextension.Extension2']
            }
        }
        env = TestEnv3(context=context)
        assert env.called_super is True
        assert len(env.extensions_passed) == 7
        assert 'cookiecutter.extensions.JsonifyExtension' in env.extensions_passed
        assert 'myextension.Extension1' in env.extensions_passed
        assert 'myextension.Extension2' in env.extensions_passed
    finally:
        TestEnv3.__mro__[1].__init__ = original_init
    
    # Test 4: Initialize with context but no _extensions key
    class TestEnv4(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.called_super = False
            self.extensions_passed = None
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, extensions=None, **kwargs):
            self.called_super = True
            self.extensions_passed = extensions
    
    TestEnv4.__mro__[1].__init__ = TestEnv4.__init__mock
    
    try:
        context = {'cookiecutter': {'other_key': 'value'}}
        env = TestEnv4(context=context)
        assert env.called_super is True
        assert len(env.extensions_passed) == 5
    finally:
        TestEnv4.__mro__[1].__init__ = original_init
    
    # Test 5: Test that additional kwargs are passed through
    class TestEnv5(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.called_super = False
            self.extensions_passed = None
            self.kwargs_passed = None
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, extensions=None, **kwargs):
            self.called_super = True
            self.extensions_passed = extensions
            self.kwargs_passed = kwargs
    
    TestEnv5.__mro__[1].__init__ = TestEnv5.__init__mock
    
    try:
        env = TestEnv5(context={}, trim_blocks=True, lstrip_blocks=True)
        assert env.called_super is True
        assert env.kwargs_passed.get('trim_blocks') is True
        assert env.kwargs_passed.get('lstrip_blocks') is True
    finally:
        TestEnv5.__mro__[1].__init__ = original_init


# LLM-generated content at query #5
#--------------------------

```python
def test_ExtensionLoaderMixin():
    class TestEnvironment:
        def __init__(self, extensions=None, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    
    class TestClass(ExtensionLoaderMixin, TestEnvironment):
        pass
    
    # Test with empty context
    obj = TestClass(context={})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    
    # Test with context containing extensions
    context = {
        'cookiecutter': {
            '_extensions': ['myextension.Extension1', 'anotherext.Extension2']
        }
    }
    obj = TestClass(context=context)
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'myextension.Extension1',
        'anotherext.Extension2',
    ]
    
    # Test with context but no extensions key
    context = {'cookiecutter': {'other_key': 'value'}}
    obj = TestClass(context=context)
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    
    # Test with no context parameter
    obj = TestClass()
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    
    # Test that other kwargs are passed through
    obj = TestClass(context={}, some_arg='value', another_arg=123)
    assert obj.kwargs == {'some_arg': 'value', 'another_arg': 123}
    
    # Test with context=None
    obj = TestClass(context=None)
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    
    # Test error handling when extension import fails
    class FailingEnvironment:
        def __init__(self, extensions=None, **kwargs):
            raise ImportError("No module named 'myextension'")
    
    class FailingTestClass(ExtensionLoaderMixin, FailingEnvironment):
        pass
    
    try:
        FailingTestClass(context={})
        assert False, "Should have raised UnknownExtension"
    except UnknownExtension as e:
        assert "Unable to load extension" in str(e)


# LLM-generated content at query #6
#--------------------------

```python
def test_ExtensionLoaderMixin():
    # Test 1: Default extensions are loaded when no context is provided
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions = kwargs.get('extensions', [])
            self.kwargs = kwargs
            # Don't call super().__init__ to avoid actual Jinja2 initialization
    
    env = TestEnv(context={})
    expected_default = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert env.extensions == expected_default
    
    # Test 2: Custom extensions from context are added to default ones
    custom_extensions = ['myapp.extensions.CustomExtension', 'another.Extension']
    env = TestEnv(context={
        'cookiecutter': {
            '_extensions': custom_extensions
        }
    })
    assert env.extensions == expected_default + custom_extensions
    
    # Test 3: Empty list when _extensions key exists but is empty
    env = TestEnv(context={
        'cookiecutter': {
            '_extensions': []
        }
    })
    assert env.extensions == expected_default
    
    # Test 4: No custom extensions when _extensions key doesn't exist
    env = TestEnv(context={
        'cookiecutter': {
            'other_key': 'value'
        }
    })
    assert env.extensions == expected_default
    
    # Test 5: No custom extensions when cookiecutter key doesn't exist
    env = TestEnv(context={'other_key': 'value'})
    assert env.extensions == expected_default
    
    # Test 6: Extensions are converted to strings
    env = TestEnv(context={
        'cookiecutter': {
            '_extensions': ['ext1', 123, 'ext2']
        }
    })
    assert all(isinstance(ext, str) for ext in env.extensions)
    assert '123' in env.extensions
    
    # Test 7: Additional kwargs are passed through
    env = TestEnv(context={}, trim_blocks=True, lstrip_blocks=True)
    assert env.kwargs['trim_blocks'] is True
    assert env.kwargs['lstrip_blocks'] is True
    
    # Test 8: None context is handled
    env = TestEnv(context=None)
    assert env.extensions == expected_default
    
    # Test 9: Test with actual StrictEnvironment to verify integration
    # This tests that the mixin properly integrates with Environment
    env = StrictEnvironment(context={
        'cookiecutter': {
            '_extensions': ['custom.Extension']
        }
    })
    assert env.undefined == StrictUndefined
    # The extensions would be loaded by Jinja2 at this point


# LLM-generated content at query #7
#--------------------------

```python
def test_ExtensionLoaderMixin():
    class TestEnvironment:
        def __init__(self, extensions=None, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs

    class TestExtensionLoaderMixin(ExtensionLoaderMixin, TestEnvironment):
        pass

    # Test with no context and no extensions
    env = TestExtensionLoaderMixin()
    assert env.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with empty context
    env = TestExtensionLoaderMixin(context={})
    assert env.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with context containing extensions
    context = {
        'cookiecutter': {
            '_extensions': ['myextension.Extension1', 'anotherextension.Extension2']
        }
    }
    env = TestExtensionLoaderMixin(context=context)
    assert env.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'myextension.Extension1',
        'anotherextension.Extension2',
    ]

    # Test with context but no _extensions key
    context = {'cookiecutter': {'other_key': 'value'}}
    env = TestExtensionLoaderMixin(context=context)
    assert env.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with context but no cookiecutter key
    context = {'other_key': 'value'}
    env = TestExtensionLoaderMixin(context=context)
    assert env.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test that kwargs are passed to parent
    env = TestExtensionLoaderMixin(context=context, test_arg='value', another_arg=123)
    assert env.kwargs['test_arg'] == 'value'
    assert env.kwargs['another_arg'] == 123

    # Test with extension that causes ImportError
    class FailingEnvironment:
        def __init__(self, extensions=None, **kwargs):
            raise ImportError("No module named 'nonexistent'")

    class FailingExtensionLoaderMixin(ExtensionLoaderMixin, FailingEnvironment):
        pass

    try:
        FailingExtensionLoaderMixin(context=context)
        assert False, "Should have raised UnknownExtension"
    except UnknownExtension as e:
        assert "Unable to load extension" in str(e)


# LLM-generated content at query #8
#--------------------------

```python
def test_ExtensionLoaderMixin():
    # Test 1: Default extensions are loaded when context is empty
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.mock_super_called = False
            self.mock_extensions = None
            self.mock_kwargs = None
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, extensions, **kwargs):
            self.mock_super_called = True
            self.mock_extensions = extensions
            self.mock_kwargs = kwargs
    
    # Replace the parent's __init__ with our mock
    original_super = TestEnv.__mro__[1].__init__
    TestEnv.__mro__[1].__init__ = TestEnv.__init__mock
    
    try:
        env = TestEnv(context={})
        assert env.mock_super_called is True
        assert len(env.mock_extensions) == 5
        assert 'cookiecutter.extensions.JsonifyExtension' in env.mock_extensions
        assert 'cookiecutter.extensions.RandomStringExtension' in env.mock_extensions
        assert 'cookiecutter.extensions.SlugifyExtension' in env.mock_extensions
        assert 'cookiecutter.extensions.TimeExtension' in env.mock_extensions
        assert 'cookiecutter.extensions.UUIDExtension' in env.mock_extensions
    finally:
        TestEnv.__mro__[1].__init__ = original_super
    
    # Test 2: Custom extensions are added from context
    class TestEnv2(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.mock_super_called = False
            self.mock_extensions = None
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, extensions, **kwargs):
            self.mock_super_called = True
            self.mock_extensions = extensions
    
    TestEnv2.__mro__[1].__init__ = TestEnv2.__init__mock
    
    try:
        context = {
            'cookiecutter': {
                '_extensions': ['my_ext.Extension1', 'other_ext.Extension2']
            }
        }
        env = TestEnv2(context=context)
        assert env.mock_super_called is True
        assert len(env.mock_extensions) == 7
        assert 'my_ext.Extension1' in env.mock_extensions
        assert 'other_ext.Extension2' in env.mock_extensions
    finally:
        TestEnv2.__mro__[1].__init__ = original_super
    
    # Test 3: No _extensions key returns only default extensions
    class TestEnv3(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.mock_super_called = False
            self.mock_extensions = None
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, extensions, **kwargs):
            self.mock_super_called = True
            self.mock_extensions = extensions
    
    TestEnv3.__mro__[1].__init__ = TestEnv3.__init__mock
    
    try:
        context = {'cookiecutter': {}}
        env = TestEnv3(context=context)
        assert env.mock_super_called is True
        assert len(env.mock_extensions) == 5
    finally:
        TestEnv3.__mro__[1].__init__ = original_super
    
    # Test 4: No cookiecutter key returns only default extensions
    class TestEnv4(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.mock_super_called = False
            self.mock_extensions = None
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, extensions, **kwargs):
            self.mock_super_called = True
            self.mock_extensions = extensions
    
    TestEnv4.__mro__[1].__init__ = TestEnv4.__init__mock
    
    try:
        context = {'other_key': 'value'}
        env = TestEnv4(context=context)
        assert env.mock_super_called is True
        assert len(env.mock_extensions) == 5
    finally:
        TestEnv4.__mro__[1].__init__ = original_super
    
    # Test 5: Additional kwargs are passed to parent
    class TestEnv5(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.mock_super_called = False
            self.mock_kwargs = None
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, extensions, **kwargs):
            self.mock_super_called = True
            self.mock_kwargs = kwargs
    
    TestEnv5.__mro__[1].__init__ = TestEnv5.__init__mock
    
    try:
        env = TestEnv5(context={}, trim_blocks=True, lstrip_blocks=True)
        assert env.mock_super_called is True
        assert env.mock_kwargs['trim_blocks'] is True
        assert env.mock_kwargs['lstrip_blocks'] is True
    finally:
        TestEnv5.__mro__[1].__init__ = original_super
    
    # Test 6: ImportError is converted to UnknownExtension
    class TestEnv6(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.exception_raised = None
            try:
                super().__init__(context=context, **kwargs)
            except Exception as e:
                self.exception_raised = e
        
        def __init__mock(self, extensions, **kwargs):
            raise ImportError("No module named 'invalid_extension'")
    
    TestEnv6.__mro__[1].__init__ = TestEnv6.__init__mock
    
    try:
        env = TestEnv6(context={})
        assert env.exception_raised is not None
        assert isinstance(env.exception_raised, UnknownExtension)
        assert "Unable to load extension" in str(env.exception_raised)
    finally:
        TestEnv6.__mro__[1].__init__ = original_super


# LLM-generated content at query #9
#--------------------------

```python
def test_ExtensionLoaderMixin():
    class TestEnvironment:
        def __init__(self, extensions=None, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs

    class TestClass(ExtensionLoaderMixin, TestEnvironment):
        pass

    # Test with no context
    obj = TestClass()
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with empty context
    obj = TestClass(context={})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with context but no extensions
    obj = TestClass(context={'cookiecutter': {'key': 'value'}})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with custom extensions
    custom_exts = ['my_ext.Extension1', 'other_ext.Extension2']
    obj = TestClass(context={'cookiecutter': {'_extensions': custom_exts}})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my_ext.Extension1',
        'other_ext.Extension2',
    ]

    # Test with additional kwargs
    obj = TestClass(context={'cookiecutter': {'_extensions': ['ext.Ext']}}, extra_arg='value')
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'ext.Ext',
    ]
    assert obj.kwargs == {'extra_arg': 'value'}

    # Test with non-string extensions (should be converted to string)
    obj = TestClass(context={'cookiecutter': {'_extensions': [123, True]}})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        '123',
        'True',
    ]

    # Test ImportError handling
    class FailingEnvironment:
        def __init__(self, extensions=None, **kwargs):
            raise ImportError("Cannot import extension")

    class FailingTestClass(ExtensionLoaderMixin, FailingEnvironment):
        pass

    try:
        FailingTestClass(context={'cookiecutter': {'_extensions': ['bad.ext']}})
        assert False, "Should have raised UnknownExtension"
    except UnknownExtension as e:
        assert "Unable to load extension" in str(e)


# LLM-generated content at query #10
#--------------------------

```python
def test_ExtensionLoaderMixin():
    # Test 1: Basic initialization with no context
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.super_called = False
            self.super_kwargs = {}
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, extensions, **kwargs):
            self.super_called = True
            self.super_kwargs = {'extensions': extensions, **kwargs}
    
    # Monkey patch the super call
    test_obj = TestClass()
    assert test_obj.super_called is True
    assert 'extensions' in test_obj.super_kwargs
    assert len(test_obj.super_kwargs['extensions']) == 5
    
    # Test 2: Initialization with empty context
    test_obj = TestClass(context={})
    assert test_obj.super_called is True
    assert len(test_obj.super_kwargs['extensions']) == 5
    
    # Test 3: Initialization with context containing extensions
    context = {
        'cookiecutter': {
            '_extensions': ['my_ext.Extension1', 'other_ext.Extension2']
        }
    }
    test_obj = TestClass(context=context)
    assert test_obj.super_called is True
    assert len(test_obj.super_kwargs['extensions']) == 7
    assert 'my_ext.Extension1' in test_obj.super_kwargs['extensions']
    assert 'other_ext.Extension2' in test_obj.super_kwargs['extensions']
    
    # Test 4: Initialization with context but no extensions key
    context = {'cookiecutter': {'other_key': 'value'}}
    test_obj = TestClass(context=context)
    assert test_obj.super_called is True
    assert len(test_obj.super_kwargs['extensions']) == 5
    
    # Test 5: Initialization with context but no cookiecutter key
    context = {'other_key': 'value'}
    test_obj = TestClass(context=context)
    assert test_obj.super_called is True
    assert len(test_obj.super_kwargs['extensions']) == 5
    
    # Test 6: Test that extensions are properly converted to strings
    context = {
        'cookiecutter': {
            '_extensions': [123, 456.789]  # Non-string extensions
        }
    }
    test_obj = TestClass(context=context)
    assert test_obj.super_called is True
    assert '123' in test_obj.super_kwargs['extensions']
    assert '456.789' in test_obj.super_kwargs['extensions']
    
    # Test 7: Test additional kwargs are passed through
    test_obj = TestClass(context={}, extra_arg='value', another_arg=123)
    assert test_obj.super_called is True
    assert test_obj.super_kwargs['extra_arg'] == 'value'
    assert test_obj.super_kwargs['another_arg'] == 123
    
    # Test 8: Test ImportError handling
    class FailingTestClass(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, extensions, **kwargs):
            raise ImportError("Test import error")
    
    # Monkey patch to simulate ImportError
    import cookiecutter.exceptions
    try:
        test_obj = FailingTestClass()
        assert False, "Should have raised UnknownExtension"
    except cookiecutter.exceptions.UnknownExtension as e:
        assert "Unable to load extension" in str(e)
        assert "Test import error" in str(e)


# LLM-generated content at query #11
#--------------------------

```python
def test_ExtensionLoaderMixin():
    # Test 1: Initialize with no context
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.called_super = False
            self.extensions_passed = None
            self.kwargs_passed = None
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, extensions, **kwargs):
            self.called_super = True
            self.extensions_passed = extensions
            self.kwargs_passed = kwargs
    
    # Replace the parent's __init__ with our mock
    import cookiecutter.environment
    original_super = cookiecutter.environment.ExtensionLoaderMixin.__init__.__code__
    
    test_env = TestEnv()
    assert test_env.called_super
    assert len(test_env.extensions_passed) == 5
    assert 'cookiecutter.extensions.JsonifyExtension' in test_env.extensions_passed
    assert 'cookiecutter.extensions.RandomStringExtension' in test_env.extensions_passed
    assert 'cookiecutter.extensions.SlugifyExtension' in test_env.extensions_passed
    assert 'cookiecutter.extensions.TimeExtension' in test_env.extensions_passed
    assert 'cookiecutter.extensions.UUIDExtension' in test_env.extensions_passed
    
    # Test 2: Initialize with empty context
    test_env = TestEnv(context={})
    assert test_env.called_super
    assert len(test_env.extensions_passed) == 5
    
    # Test 3: Initialize with context containing extensions
    context = {
        'cookiecutter': {
            '_extensions': ['my_ext.Extension1', 'other_ext.Extension2']
        }
    }
    test_env = TestEnv(context=context)
    assert test_env.called_super
    assert len(test_env.extensions_passed) == 7
    assert 'cookiecutter.extensions.JsonifyExtension' in test_env.extensions_passed
    assert 'cookiecutter.extensions.RandomStringExtension' in test_env.extensions_passed
    assert 'my_ext.Extension1' in test_env.extensions_passed
    assert 'other_ext.Extension2' in test_env.extensions_passed
    
    # Test 4: Initialize with context but no extensions key
    context = {'cookiecutter': {'other_key': 'value'}}
    test_env = TestEnv(context=context)
    assert test_env.called_super
    assert len(test_env.extensions_passed) == 5
    
    # Test 5: Initialize with context but empty cookiecutter
    context = {'cookiecutter': {}}
    test_env = TestEnv(context=context)
    assert test_env.called_super
    assert len(test_env.extensions_passed) == 5
    
    # Test 6: Test that additional kwargs are passed through
    test_env = TestEnv(context=context, trim_blocks=True, lstrip_blocks=True)
    assert test_env.called_super
    assert test_env.kwargs_passed.get('trim_blocks') is True
    assert test_env.kwargs_passed.get('lstrip_blocks') is True
    
    # Test 7: Test ImportError handling
    class FailingTestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.called_super = False
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, extensions, **kwargs):
            self.called_super = True
            raise ImportError("Test import error")
    
    # Restore original __init__ code
    cookiecutter.environment.ExtensionLoaderMixin.__init__.__code__ = original_super


# LLM-generated content at query #12
#--------------------------

```python
def test_ExtensionLoaderMixin():
    class TestEnvironment:
        def __init__(self, extensions=None, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs

    class TestClass(ExtensionLoaderMixin, TestEnvironment):
        pass

    # Test with no context
    obj = TestClass()
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with empty context
    obj = TestClass(context={})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with context but no extensions
    obj = TestClass(context={'cookiecutter': {'other_key': 'value'}})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with custom extensions
    custom_extensions = ['my_ext.Extension1', 'my_ext.Extension2']
    obj = TestClass(context={'cookiecutter': {'_extensions': custom_extensions}})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my_ext.Extension1',
        'my_ext.Extension2',
    ]

    # Test with custom extensions as non-strings
    custom_extensions = ['ext1', 123, True]
    obj = TestClass(context={'cookiecutter': {'_extensions': custom_extensions}})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'ext1',
        '123',
        'True',
    ]

    # Test that kwargs are passed through
    obj = TestClass(context={}, some_arg='value', another_arg=42)
    assert obj.kwargs == {'some_arg': 'value', 'another_arg': 42}
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test ImportError handling
    class FailingEnvironment:
        def __init__(self, extensions=None, **kwargs):
            raise ImportError("No module named 'invalid_extension'")

    class FailingClass(ExtensionLoaderMixin, FailingEnvironment):
        pass

    try:
        FailingClass(context={})
        assert False, "Should have raised UnknownExtension"
    except UnknownExtension as e:
        assert "Unable to load extension" in str(e)


# LLM-generated content at query #13
#--------------------------

```python
def test_ExtensionLoaderMixin():
    # Test 1: Initialize with empty context
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions = kwargs.get('extensions', [])
            self.kwargs = kwargs
            super().__init__(context=context, **kwargs)
    
    env = TestEnv(context={})
    assert len(env.extensions) == 5
    assert 'cookiecutter.extensions.JsonifyExtension' in env.extensions
    assert 'cookiecutter.extensions.RandomStringExtension' in env.extensions
    assert 'cookiecutter.extensions.SlugifyExtension' in env.extensions
    assert 'cookiecutter.extensions.TimeExtension' in env.extensions
    assert 'cookiecutter.extensions.UUIDExtension' in env.extensions
    
    # Test 2: Initialize with None context
    env = TestEnv(context=None)
    assert len(env.extensions) == 5
    
    # Test 3: Initialize with custom extensions in context
    context = {
        'cookiecutter': {
            '_extensions': [
                'my.custom.Extension1',
                'my.custom.Extension2'
            ]
        }
    }
    env = TestEnv(context=context)
    assert len(env.extensions) == 7
    assert 'my.custom.Extension1' in env.extensions
    assert 'my.custom.Extension2' in env.extensions
    
    # Test 4: Initialize with empty cookiecutter dict
    context = {'cookiecutter': {}}
    env = TestEnv(context=context)
    assert len(env.extensions) == 5
    
    # Test 5: Initialize without cookiecutter key
    context = {'other_key': 'value'}
    env = TestEnv(context=context)
    assert len(env.extensions) == 5
    
    # Test 6: Test that kwargs are passed to parent
    class TestEnvWithSuper(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.super_called = False
            self.super_kwargs = None
            super().__init__(context=context, **kwargs)
        
        def __init_subclass__(cls):
            pass
        
        def super_init(self, **kwargs):
            self.super_called = True
            self.super_kwargs = kwargs
    
    # Mock the super call
    env = TestEnvWithSuper(context={}, test_arg='value')
    assert env.super_called
    assert 'extensions' in env.super_kwargs
    assert len(env.super_kwargs['extensions']) == 5
    
    # Test 7: Test ImportError handling
    class TestEnvWithImportError(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.super_called = False
            super().__init__(context=context, **kwargs)
        
        def __init_subclass__(cls):
            pass
        
        def super_init(self, **kwargs):
            self.super_called = True
            raise ImportError("Test import error")
    
    # Test that ImportError is converted to UnknownExtension
    try:
        env = TestEnvWithImportError(context={})
        assert False, "Should have raised UnknownExtension"
    except UnknownExtension as e:
        assert "Unable to load extension: Test import error" in str(e)
    
    # Test 8: Test with string extensions in context
    context = {
        'cookiecutter': {
            '_extensions': 'my.custom.Extension'
        }
    }
    env = TestEnv(context=context)
    assert len(env.extensions) == 6
    assert 'my.custom.Extension' in env.extensions
    
    # Test 9: Test with mixed type extensions
    context = {
        'cookiecutter': {
            '_extensions': ['ext1', 123, 'ext2']
        }
    }
    env = TestEnv(context=context)
    assert len(env.extensions) == 8
    assert 'ext1' in env.extensions
    assert '123' in env.extensions
    assert 'ext2' in env.extensions


# LLM-generated content at query #14
#--------------------------

```python
def test_ExtensionLoaderMixin():
    # Test 1: Basic initialization with no context
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.called_super = False
            self.extensions_passed = None
            self.kwargs_passed = None
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, extensions=None, **kwargs):
            self.called_super = True
            self.extensions_passed = extensions
            self.kwargs_passed = kwargs
    
    # Replace the parent's __init__ with our mock
    original_super = TestClass.__mro__[1].__init__
    TestClass.__mro__[1].__init__ = TestClass.__init__mock
    
    try:
        obj = TestClass()
        assert obj.called_super is True
        default_extensions = [
            'cookiecutter.extensions.JsonifyExtension',
            'cookiecutter.extensions.RandomStringExtension',
            'cookiecutter.extensions.SlugifyExtension',
            'cookiecutter.extensions.TimeExtension',
            'cookiecutter.extensions.UUIDExtension',
        ]
        assert obj.extensions_passed == default_extensions
        assert obj.kwargs_passed == {}
    finally:
        # Restore original __init__
        TestClass.__mro__[1].__init__ = original_super
    
    # Test 2: Initialization with empty context
    TestClass.__mro__[1].__init__ = TestClass.__init__mock
    try:
        obj = TestClass(context={})
        assert obj.called_super is True
        assert obj.extensions_passed == default_extensions
        assert obj.kwargs_passed == {}
    finally:
        TestClass.__mro__[1].__init__ = original_super
    
    # Test 3: Initialization with context containing _extensions
    TestClass.__mro__[1].__init__ = TestClass.__init__mock
    try:
        context = {
            'cookiecutter': {
                '_extensions': ['my.custom.Extension1', 'my.custom.Extension2']
            }
        }
        obj = TestClass(context=context)
        assert obj.called_super is True
        expected_extensions = default_extensions + ['my.custom.Extension1', 'my.custom.Extension2']
        assert obj.extensions_passed == expected_extensions
        assert obj.kwargs_passed == {}
    finally:
        TestClass.__mro__[1].__init__ = original_super
    
    # Test 4: Initialization with additional kwargs
    TestClass.__mro__[1].__init__ = TestClass.__init__mock
    try:
        obj = TestClass(context={}, extra_arg='value', another_arg=123)
        assert obj.called_super is True
        assert obj.extensions_passed == default_extensions
        assert obj.kwargs_passed == {'extra_arg': 'value', 'another_arg': 123}
    finally:
        TestClass.__mro__[1].__init__ = original_super
    
    # Test 5: Initialization with context missing cookiecutter key
    TestClass.__mro__[1].__init__ = TestClass.__init__mock
    try:
        context = {'other_key': 'value'}
        obj = TestClass(context=context)
        assert obj.called_super is True
        assert obj.extensions_passed == default_extensions
        assert obj.kwargs_passed == {}
    finally:
        TestClass.__mro__[1].__init__ = original_super
    
    # Test 6: Initialization with context missing _extensions key
    TestClass.__mro__[1].__init__ = TestClass.__init__mock
    try:
        context = {'cookiecutter': {'other_key': 'value'}}
        obj = TestClass(context=context)
        assert obj.called_super is True
        assert obj.extensions_passed == default_extensions
        assert obj.kwargs_passed == {}
    finally:
        TestClass.__mro__[1].__init__ = original_super
    
    # Test 7: Test ImportError handling
    class TestClassWithImportError(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, extensions=None, **kwargs):
            raise ImportError("No module named 'nonexistent'")
    
    TestClassWithImportError.__mro__[1].__init__ = TestClassWithImportError.__init__mock
    try:
        import pytest
        with pytest.raises(UnknownExtension) as exc_info:
            TestClassWithImportError()
        assert "Unable to load extension: No module named 'nonexistent'" in str(exc_info.value)
    finally:
        TestClassWithImportError.__mro__[1].__init__ = original_super


# LLM-generated content at query #15
#--------------------------

```python
def test_ExtensionLoaderMixin():
    # Test 1: Initialize with no context
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions = kwargs.get('extensions', [])
            self.other_args = {k: v for k, v in kwargs.items() if k != 'extensions'}
            super().__init__(context=context, **kwargs)

    env = TestEnv()
    assert 'cookiecutter.extensions.JsonifyExtension' in env.extensions
    assert 'cookiecutter.extensions.TimeExtension' in env.extensions
    assert len(env.extensions) == 5

    # Test 2: Initialize with empty context
    env = TestEnv(context={})
    assert len(env.extensions) == 5

    # Test 3: Initialize with context containing extensions
    context = {
        'cookiecutter': {
            '_extensions': ['myextension.Extension1', 'anotherextension.Extension2']
        }
    }
    env = TestEnv(context=context)
    assert 'cookiecutter.extensions.JsonifyExtension' in env.extensions
    assert 'myextension.Extension1' in env.extensions
    assert 'anotherextension.Extension2' in env.extensions
    assert len(env.extensions) == 7

    # Test 4: Initialize with context missing _extensions key
    context = {'cookiecutter': {}}
    env = TestEnv(context=context)
    assert len(env.extensions) == 5

    # Test 5: Initialize with context missing cookiecutter key
    context = {'other_key': 'value'}
    env = TestEnv(context=context)
    assert len(env.extensions) == 5

    # Test 6: Test that extensions are properly converted to strings
    context = {
        'cookiecutter': {
            '_extensions': [123, 'test.Extension']  # Mixed types
        }
    }
    env = TestEnv(context=context)
    assert all(isinstance(ext, str) for ext in env.extensions)
    assert '123' in env.extensions
    assert 'test.Extension' in env.extensions

    # Test 7: Test that other kwargs are passed through
    env = TestEnv(context={}, test_arg='value', another_arg=42)
    assert env.other_args.get('test_arg') == 'value'
    assert env.other_args.get('another_arg') == 42

    # Test 8: Test ImportError handling
    class FailingTestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            # Simulate ImportError by overriding super().__init__
            raise ImportError("No module named 'nonexistent'")

    import pytest
    from cookiecutter.exceptions import UnknownExtension
    
    with pytest.raises(UnknownExtension) as exc_info:
        FailingTestEnv()
    assert "Unable to load extension" in str(exc_info.value)


# LLM-generated content at query #16
#--------------------------

```python
def test_ExtensionLoaderMixin():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.super_called = False
            self.super_kwargs = None
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, *, context=None, **kwargs):
            self.super_called = True
            self.super_kwargs = kwargs

    # Test 1: Default extensions are loaded when no context provided
    test_obj = TestClass()
    assert test_obj.super_called
    assert 'extensions' in test_obj.super_kwargs
    extensions = test_obj.super_kwargs['extensions']
    assert len(extensions) == 5
    assert 'cookiecutter.extensions.JsonifyExtension' in extensions
    assert 'cookiecutter.extensions.RandomStringExtension' in extensions
    assert 'cookiecutter.extensions.SlugifyExtension' in extensions
    assert 'cookiecutter.extensions.TimeExtension' in extensions
    assert 'cookiecutter.extensions.UUIDExtension' in extensions

    # Test 2: Empty context results in only default extensions
    test_obj = TestClass(context={})
    assert test_obj.super_called
    extensions = test_obj.super_kwargs['extensions']
    assert len(extensions) == 5

    # Test 3: Context without cookiecutter key results in only default extensions
    test_obj = TestClass(context={'other_key': 'value'})
    assert test_obj.super_called
    extensions = test_obj.super_kwargs['extensions']
    assert len(extensions) == 5

    # Test 4: Context with cookiecutter but no _extensions results in only default extensions
    test_obj = TestClass(context={'cookiecutter': {'name': 'test'}})
    assert test_obj.super_called
    extensions = test_obj.super_kwargs['extensions']
    assert len(extensions) == 5

    # Test 5: Context with _extensions adds them to default extensions
    custom_extensions = ['my_ext.Extension1', 'my_ext.Extension2']
    test_obj = TestClass(context={
        'cookiecutter': {
            '_extensions': custom_extensions,
            'name': 'test'
        }
    })
    assert test_obj.super_called
    extensions = test_obj.super_kwargs['extensions']
    assert len(extensions) == 7
    assert all(default_ext in extensions for default_ext in [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension'
    ])
    assert 'my_ext.Extension1' in extensions
    assert 'my_ext.Extension2' in extensions

    # Test 6: Additional kwargs are passed to parent
    test_obj = TestClass(context=None, extra_arg='value', another_arg=123)
    assert test_obj.super_called
    assert test_obj.super_kwargs['extra_arg'] == 'value'
    assert test_obj.super_kwargs['another_arg'] == 123
    assert 'extensions' in test_obj.super_kwargs

    # Test 7: _extensions with non-string values are converted to strings
    test_obj = TestClass(context={
        'cookiecutter': {
            '_extensions': [123, 456.789]
        }
    })
    assert test_obj.super_called
    extensions = test_obj.super_kwargs['extensions']
    assert len(extensions) == 7
    assert '123' in extensions
    assert '456.789' in extensions

    # Test 8: ImportError is caught and converted to UnknownExtension
    class FailingTestClass(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, *, context=None, **kwargs):
            raise ImportError("No module named 'invalid_extension'")
    
    import_error_raised = False
    try:
        FailingTestClass()
    except Exception as e:
        import_error_raised = True
        assert "Unable to load extension" in str(e)
    assert import_error_raised


# LLM-generated content at query #17
#--------------------------

```python
def test_ExtensionLoaderMixin():
    class TestEnvironment:
        def __init__(self, extensions=None, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs

    class TestExtensionLoaderMixin(ExtensionLoaderMixin, TestEnvironment):
        pass

    # Test with no context
    env = TestExtensionLoaderMixin()
    assert env.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with empty context
    env = TestExtensionLoaderMixin(context={})
    assert env.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with context but no extensions
    env = TestExtensionLoaderMixin(context={'cookiecutter': {'key': 'value'}})
    assert env.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with custom extensions
    custom_exts = ['my_ext.Extension1', 'other_ext.Extension2']
    env = TestExtensionLoaderMixin(
        context={'cookiecutter': {'_extensions': custom_exts}}
    )
    assert env.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my_ext.Extension1',
        'other_ext.Extension2',
    ]

    # Test with non-string extensions (should be converted to strings)
    custom_exts = ['ext1', 123, True]
    env = TestExtensionLoaderMixin(
        context={'cookiecutter': {'_extensions': custom_exts}}
    )
    assert env.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'ext1',
        '123',
        'True',
    ]

    # Test that kwargs are passed through
    env = TestExtensionLoaderMixin(context={}, trim_blocks=True, lstrip_blocks=True)
    assert env.kwargs == {'trim_blocks': True, 'lstrip_blocks': True}
    assert env.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test ImportError handling
    class FailingEnvironment:
        def __init__(self, extensions=None, **kwargs):
            raise ImportError("No module named 'invalid_ext'")

    class FailingExtensionLoaderMixin(ExtensionLoaderMixin, FailingEnvironment):
        pass

    try:
        FailingExtensionLoaderMixin(context={})
        assert False, "Should have raised UnknownExtension"
    except UnknownExtension as e:
        assert "Unable to load extension: No module named 'invalid_ext'" in str(e)


# LLM-generated content at query #18
#--------------------------

```python
def test_ExtensionLoaderMixin():
    # Test 1: Default extensions are loaded when no context provided
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions = kwargs.get('extensions', [])
            super().__init__(context=context, **kwargs)

    env = TestEnv()
    assert len(env.extensions) == 5
    assert 'cookiecutter.extensions.JsonifyExtension' in env.extensions
    assert 'cookiecutter.extensions.RandomStringExtension' in env.extensions
    assert 'cookiecutter.extensions.SlugifyExtension' in env.extensions
    assert 'cookiecutter.extensions.TimeExtension' in env.extensions
    assert 'cookiecutter.extensions.UUIDExtension' in env.extensions

    # Test 2: Empty context returns only default extensions
    env = TestEnv(context={})
    assert len(env.extensions) == 5

    # Test 3: Context without cookiecutter key returns only default extensions
    env = TestEnv(context={'other_key': 'value'})
    assert len(env.extensions) == 5

    # Test 4: Context with empty cookiecutter returns only default extensions
    env = TestEnv(context={'cookiecutter': {}})
    assert len(env.extensions) == 5

    # Test 5: Context with extensions adds them to default extensions
    context = {
        'cookiecutter': {
            '_extensions': ['myextension.Extension1', 'anotherext.Extension2']
        }
    }
    env = TestEnv(context=context)
    assert len(env.extensions) == 7
    assert 'myextension.Extension1' in env.extensions
    assert 'anotherext.Extension2' in env.extensions
    assert 'cookiecutter.extensions.JsonifyExtension' in env.extensions

    # Test 6: Extensions are converted to strings
    context = {
        'cookiecutter': {
            '_extensions': [123, 456.789]  # Non-string extensions
        }
    }
    env = TestEnv(context=context)
    assert len(env.extensions) == 7
    assert '123' in env.extensions
    assert '456.789' in env.extensions

    # Test 7: ImportError is converted to UnknownExtension
    class FailingEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.kwargs = kwargs
            # Simulate ImportError from parent's __init__
            raise ImportError("No module named 'nonexistent'")

    try:
        FailingEnv(context={})
        assert False, "Should have raised UnknownExtension"
    except UnknownExtension as e:
        assert "Unable to load extension" in str(e)
        assert "No module named 'nonexistent'" in str(e)

    # Test 8: Additional kwargs are passed to parent
    class KwargsEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.received_kwargs = kwargs
            super().__init__(context=context, **kwargs)

    env = KwargsEnv(context={}, trim_blocks=True, lstrip_blocks=True)
    assert env.received_kwargs['trim_blocks'] is True
    assert env.received_kwargs['lstrip_blocks'] is True
    assert 'extensions' in env.received_kwargs


# LLM-generated content at query #19
#--------------------------

```python
def test_ExtensionLoaderMixin():
    # Test 1: Default extensions are loaded when no context provided
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions = kwargs.get('extensions', [])
            super().__init__(context=context, **kwargs)

    env = TestEnv()
    expected_defaults = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert env.extensions == expected_defaults

    # Test 2: Extensions from context are added to default extensions
    context = {
        'cookiecutter': {
            '_extensions': ['my_ext.Extension1', 'other_ext.Extension2']
        }
    }
    env = TestEnv(context=context)
    assert env.extensions == expected_defaults + ['my_ext.Extension1', 'other_ext.Extension2']

    # Test 3: Empty context returns only default extensions
    env = TestEnv(context={})
    assert env.extensions == expected_defaults

    # Test 4: Context without cookiecutter key returns only default extensions
    env = TestEnv(context={'other_key': 'value'})
    assert env.extensions == expected_defaults

    # Test 5: Context with cookiecutter but no _extensions returns only default extensions
    env = TestEnv(context={'cookiecutter': {'key': 'value'}})
    assert env.extensions == expected_defaults

    # Test 6: ImportError is caught and re-raised as UnknownExtension
    class FailingEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.kwargs = kwargs
            super().__init__(context=context, **kwargs)
        
        def __init__(self, *, context=None, **kwargs):
            raise ImportError("No module named 'nonexistent'")

    try:
        FailingEnv()
        assert False, "Should have raised UnknownExtension"
    except UnknownExtension as e:
        assert "Unable to load extension" in str(e)

    # Test 7: Additional kwargs are passed to parent
    class KwargsEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.kwargs = kwargs
            super().__init__(context=context, **kwargs)

    env = KwargsEnv(context=context, trim_blocks=True, lstrip_blocks=True)
    assert env.kwargs['trim_blocks'] is True
    assert env.kwargs['lstrip_blocks'] is True
    assert 'extensions' in env.kwargs


# LLM-generated content at query #20
#--------------------------

```python
def test_ExtensionLoaderMixin():
    class TestEnvironment:
        def __init__(self, extensions=None, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs

    class TestClass(ExtensionLoaderMixin, TestEnvironment):
        pass

    # Test with no context
    obj = TestClass()
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with empty context
    obj = TestClass(context={})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with context but no _extensions
    obj = TestClass(context={'cookiecutter': {'other_key': 'value'}})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with custom extensions
    custom_extensions = ['my_ext.Extension1', 'my_ext.Extension2']
    obj = TestClass(context={'cookiecutter': {'_extensions': custom_extensions}})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my_ext.Extension1',
        'my_ext.Extension2',
    ]

    # Test with custom extensions as non-strings
    custom_extensions = ['ext1', 123, True]
    obj = TestClass(context={'cookiecutter': {'_extensions': custom_extensions}})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'ext1',
        '123',
        'True',
    ]

    # Test that kwargs are passed through
    obj = TestClass(context={}, test_kwarg='value', another_kwarg=123)
    assert obj.kwargs == {'test_kwarg': 'value', 'another_kwarg': 123}

    # Test ImportError handling
    class FailingEnvironment:
        def __init__(self, extensions=None, **kwargs):
            raise ImportError("No module named 'invalid_extension'")

    class FailingClass(ExtensionLoaderMixin, FailingEnvironment):
        pass

    try:
        FailingClass(context={})
        assert False, "Should have raised UnknownExtension"
    except UnknownExtension as e:
        assert "Unable to load extension: No module named 'invalid_extension'" in str(e)


# LLM-generated content at query #21
#--------------------------

```python
def test_ExtensionLoaderMixin():
    class TestEnvironment:
        def __init__(self, extensions=None, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs

    class TestExtensionLoaderMixin(ExtensionLoaderMixin, TestEnvironment):
        pass

    # Test with no context
    env = TestExtensionLoaderMixin()
    assert env.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with empty context
    env = TestExtensionLoaderMixin(context={})
    assert env.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with context but no _extensions
    context = {'cookiecutter': {'project_name': 'Test'}}
    env = TestExtensionLoaderMixin(context=context)
    assert env.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with custom extensions
    context = {
        'cookiecutter': {
            '_extensions': ['my_ext.Extension1', 'other.Extension2']
        }
    }
    env = TestExtensionLoaderMixin(context=context)
    assert env.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my_ext.Extension1',
        'other.Extension2',
    ]

    # Test with non-string extensions (should be converted to string)
    context = {
        'cookiecutter': {
            '_extensions': [123, True, 'my_ext.Extension']
        }
    }
    env = TestExtensionLoaderMixin(context=context)
    assert '123' in env.extensions
    assert 'True' in env.extensions
    assert 'my_ext.Extension' in env.extensions

    # Test that other kwargs are passed through
    env = TestExtensionLoaderMixin(context={}, trim_blocks=True, lstrip_blocks=True)
    assert env.kwargs['trim_blocks'] is True
    assert env.kwargs['lstrip_blocks'] is True

    # Test ImportError handling
    class FailingEnvironment:
        def __init__(self, extensions=None, **kwargs):
            raise ImportError("No module named 'invalid_extension'")

    class FailingExtensionLoaderMixin(ExtensionLoaderMixin, FailingEnvironment):
        pass

    try:
        FailingExtensionLoaderMixin(context={})
        assert False, "Should have raised UnknownExtension"
    except UnknownExtension as e:
        assert "Unable to load extension" in str(e)
        assert isinstance(e.__cause__, ImportError)


# LLM-generated content at query #22
#--------------------------

```python
def test_ExtensionLoaderMixin():
    # Test 1: Initialize with no context
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.called_super = False
            self.extensions_passed = None
            self.kwargs_passed = None
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, extensions, **kwargs):
            self.called_super = True
            self.extensions_passed = extensions
            self.kwargs_passed = kwargs
    
    # Replace the parent's __init__ with our mock
    import cookiecutter.environment
    original_super = TestClass.__bases__[0].__init__
    TestClass.__bases__[0].__init__ = TestClass.__init__mock
    
    try:
        obj = TestClass()
        assert obj.called_super is True
        default_extensions = [
            'cookiecutter.extensions.JsonifyExtension',
            'cookiecutter.extensions.RandomStringExtension',
            'cookiecutter.extensions.SlugifyExtension',
            'cookiecutter.extensions.TimeExtension',
            'cookiecutter.extensions.UUIDExtension',
        ]
        assert obj.extensions_passed == default_extensions
        assert obj.kwargs_passed == {}
    finally:
        TestClass.__bases__[0].__init__ = original_super
    
    # Test 2: Initialize with context containing extensions
    class TestClass2(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.called_super = False
            self.extensions_passed = None
            self.kwargs_passed = None
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, extensions, **kwargs):
            self.called_super = True
            self.extensions_passed = extensions
            self.kwargs_passed = kwargs
    
    TestClass2.__bases__[0].__init__ = TestClass2.__init__mock
    
    try:
        context = {
            'cookiecutter': {
                '_extensions': ['my_ext.Extension1', 'other_ext.Extension2']
            }
        }
        obj = TestClass2(context=context, some_arg='value')
        assert obj.called_super is True
        expected_extensions = [
            'cookiecutter.extensions.JsonifyExtension',
            'cookiecutter.extensions.RandomStringExtension',
            'cookiecutter.extensions.SlugifyExtension',
            'cookiecutter.extensions.TimeExtension',
            'cookiecutter.extensions.UUIDExtension',
            'my_ext.Extension1',
            'other_ext.Extension2',
        ]
        assert obj.extensions_passed == expected_extensions
        assert obj.kwargs_passed == {'some_arg': 'value'}
    finally:
        TestClass2.__bases__[0].__init__ = original_super
    
    # Test 3: Initialize with context without _extensions key
    class TestClass3(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.called_super = False
            self.extensions_passed = None
            self.kwargs_passed = None
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, extensions, **kwargs):
            self.called_super = True
            self.extensions_passed = extensions
            self.kwargs_passed = kwargs
    
    TestClass3.__bases__[0].__init__ = TestClass3.__init__mock
    
    try:
        context = {'cookiecutter': {'other_key': 'value'}}
        obj = TestClass3(context=context)
        assert obj.called_super is True
        assert obj.extensions_passed == default_extensions
        assert obj.kwargs_passed == {}
    finally:
        TestClass3.__bases__[0].__init__ = original_super
    
    # Test 4: Initialize with empty context
    class TestClass4(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.called_super = False
            self.extensions_passed = None
            self.kwargs_passed = None
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, extensions, **kwargs):
            self.called_super = True
            self.extensions_passed = extensions
            self.kwargs_passed = kwargs
    
    TestClass4.__bases__[0].__init__ = TestClass4.__init__mock
    
    try:
        obj = TestClass4(context={})
        assert obj.called_super is True
        assert obj.extensions_passed == default_extensions
        assert obj.kwargs_passed == {}
    finally:
        TestClass4.__bases__[0].__init__ = original_super
    
    # Test 5: Test ImportError handling
    class TestClass5(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.exception_raised = None
            try:
                super().__init__(context=context, **kwargs)
            except Exception as e:
                self.exception_raised = e
        
        def __init__mock(self, extensions, **kwargs):
            raise ImportError("Cannot import module")
    
    TestClass5.__bases__[0].__init__ = TestClass5.__init__mock
    
    try:
        obj = TestClass5()
        assert obj.exception_raised is not None
        assert isinstance(obj.exception_raised, cookiecutter.exceptions.UnknownExtension)
        assert "Unable to load extension" in str(obj.exception_raised)
    finally:
        TestClass5.__bases__[0].__init__ = original_super


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_ExtensionLoaderMixin():
    # Test 1: Initialize with no context
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            self.super_called = False
            self.extensions_passed = None
            super().__init__(**kwargs)
        
        def __init__mock(self, extensions, **kwargs):
            self.super_called = True
            self.extensions_passed = extensions
            self.kwargs_passed = kwargs
    
    # Monkey patch the parent's __init__ to capture arguments
    import cookiecutter.environment
    original_init = cookiecutter.environment.ExtensionLoaderMixin.__init__
    
    def mock_init(self, **kwargs):
        # Call the actual ExtensionLoaderMixin.__init__ but intercept super().__init__
        class MockSuper:
            def __init__(self, extensions, **kwargs):
                self.extensions = extensions
                self.kwargs = kwargs
            
            def __call__(self, extensions, **kwargs):
                self.extensions = extensions
                self.kwargs = kwargs
                return self
        
        mock_super = MockSuper()
        
        def mock_super_init(extensions, **kwargs):
            mock_super(extensions, **kwargs)
            return mock_super
        
        import builtins
        original_super = builtins.super
        
        def mock_super(cls, obj=None):
            if obj is not None and isinstance(obj, TestClass):
                class MockSuperClass:
                    def __init__(self):
                        pass
                    
                    def __init__(self_, extensions, **kwargs):
                        obj.super_called = True
                        obj.extensions_passed = extensions
                        obj.kwargs_passed = kwargs
                
                return MockSuperClass()
            return original_super(cls, obj)
        
        builtins.super = mock_super
        
        try:
            original_init(self, **kwargs)
        finally:
            builtins.super = original_super
    
    cookiecutter.environment.ExtensionLoaderMixin.__init__ = mock_init
    
    try:
        # Test with no context
        obj = TestClass()
        assert obj.super_called is True
        default_extensions = [
            'cookiecutter.extensions.JsonifyExtension',
            'cookiecutter.extensions.RandomStringExtension',
            'cookiecutter.extensions.SlugifyExtension',
            'cookiecutter.extensions.TimeExtension',
            'cookiecutter.extensions.UUIDExtension',
        ]
        assert obj.extensions_passed == default_extensions
        
        # Test with empty context
        obj = TestClass(context={})
        assert obj.super_called is True
        assert obj.extensions_passed == default_extensions
        
        # Test with context containing _extensions
        context = {
            'cookiecutter': {
                '_extensions': ['myextension.Extension1', 'anotherextension.Extension2']
            }
        }
        obj = TestClass(context=context)
        assert obj.super_called is True
        expected_extensions = default_extensions + ['myextension.Extension1', 'anotherextension.Extension2']
        assert obj.extensions_passed == expected_extensions
        
        # Test with context containing _extensions with non-string values
        context = {
            'cookiecutter': {
                '_extensions': [123, 456.789]  # These should be converted to strings
            }
        }
        obj = TestClass(context=context)
        assert obj.super_called is True
        expected_extensions = default_extensions + ['123', '456.789']
        assert obj.extensions_passed == expected_extensions
        
        # Test with context but no _extensions key
        context = {
            'cookiecutter': {
                'other_key': 'value'
            }
        }
        obj = TestClass(context=context)
        assert obj.super_called is True
        assert obj.extensions_passed == default_extensions
        
        # Test with context but no cookiecutter key
        context = {
            'other_key': 'value'
        }
        obj = TestClass(context=context)
        assert obj.super_called is True
        assert obj.extensions_passed == default_extensions
        
        # Test passing additional kwargs
        obj = TestClass(context=context, trim_blocks=True, lstrip_blocks=True)
        assert obj.super_called is True
        assert obj.extensions_passed == default_extensions
        assert obj.kwargs_passed.get('trim_blocks') is True
        assert obj.kwargs_passed.get('lstrip_blocks') is True
        
    finally:
        # Restore original __init__
        cookiecutter.environment.ExtensionLoaderMixin.__init__ = original_init


# LLM-generated content at query #2
#--------------------------

```python
def test_ExtensionLoaderMixin():
    class TestEnvironment:
        def __init__(self, extensions=None, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs

    class TestClass(ExtensionLoaderMixin, TestEnvironment):
        pass

    # Test with empty context
    obj = TestClass(context={})
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert obj.extensions == expected_extensions

    # Test with context containing extensions
    context = {
        'cookiecutter': {
            '_extensions': ['myextension.Extension1', 'anotherextension.Extension2']
        }
    }
    obj = TestClass(context=context)
    assert obj.extensions == expected_extensions + ['myextension.Extension1', 'anotherextension.Extension2']

    # Test with context but no extensions key
    context = {'cookiecutter': {}}
    obj = TestClass(context=context)
    assert obj.extensions == expected_extensions

    # Test with context missing cookiecutter key
    context = {'other_key': 'value'}
    obj = TestClass(context=context)
    assert obj.extensions == expected_extensions

    # Test with None context
    obj = TestClass(context=None)
    assert obj.extensions == expected_extensions

    # Test additional kwargs are passed through
    obj = TestClass(context={}, trim_blocks=True, lstrip_blocks=True)
    assert obj.kwargs['trim_blocks'] is True
    assert obj.kwargs['lstrip_blocks'] is True

    # Test ImportError handling
    class FailingEnvironment:
        def __init__(self, extensions=None, **kwargs):
            raise ImportError("No module named 'invalid'")

    class FailingTestClass(ExtensionLoaderMixin, FailingEnvironment):
        pass

    try:
        FailingTestClass(context={})
        assert False, "Should have raised UnknownExtension"
    except UnknownExtension as e:
        assert "Unable to load extension" in str(e)


# LLM-generated content at query #3
#--------------------------

```python
def test_ExtensionLoaderMixin():
    class TestEnvironment:
        def __init__(self, extensions=None, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs
    
    class TestClass(ExtensionLoaderMixin, TestEnvironment):
        pass
    
    # Test with no context
    obj = TestClass()
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    
    # Test with empty context
    obj = TestClass(context={})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    
    # Test with context but no extensions
    obj = TestClass(context={'cookiecutter': {'key': 'value'}})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    
    # Test with custom extensions
    custom_extensions = ['my_ext.Extension1', 'my_ext.Extension2']
    obj = TestClass(context={'cookiecutter': {'_extensions': custom_extensions}})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my_ext.Extension1',
        'my_ext.Extension2',
    ]
    
    # Test with string conversion of extensions
    obj = TestClass(context={'cookiecutter': {'_extensions': [1, 2.5, True]}})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        '1',
        '2.5',
        'True',
    ]
    
    # Test additional kwargs are passed through
    obj = TestClass(context={}, trim_blocks=True, lstrip_blocks=True)
    assert obj.kwargs == {'trim_blocks': True, 'lstrip_blocks': True}
    
    # Test ImportError handling
    class FailingEnvironment:
        def __init__(self, extensions=None, **kwargs):
            raise ImportError("No module named 'invalid_ext'")
    
    class FailingClass(ExtensionLoaderMixin, FailingEnvironment):
        pass
    
    try:
        FailingClass(context={'cookiecutter': {'_extensions': ['invalid_ext']}})
        assert False, "Should have raised UnknownExtension"
    except UnknownExtension as e:
        assert "Unable to load extension: No module named 'invalid_ext'" in str(e)


# LLM-generated content at query #4
#--------------------------

```python
def test_ExtensionLoaderMixin():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.super_called = False
            self.super_kwargs = None
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, *, context=None, **kwargs):
            self.super_called = True
            self.super_kwargs = kwargs

    # Test 1: Default extensions are loaded when no context provided
    test_obj = TestClass()
    assert test_obj.super_called
    assert 'extensions' in test_obj.super_kwargs
    extensions = test_obj.super_kwargs['extensions']
    assert len(extensions) == 5
    assert 'cookiecutter.extensions.JsonifyExtension' in extensions
    assert 'cookiecutter.extensions.RandomStringExtension' in extensions
    assert 'cookiecutter.extensions.SlugifyExtension' in extensions
    assert 'cookiecutter.extensions.TimeExtension' in extensions
    assert 'cookiecutter.extensions.UUIDExtension' in extensions

    # Test 2: Empty context results in only default extensions
    test_obj = TestClass(context={})
    assert test_obj.super_called
    extensions = test_obj.super_kwargs['extensions']
    assert len(extensions) == 5

    # Test 3: Context without cookiecutter key results in only default extensions
    test_obj = TestClass(context={'other_key': 'value'})
    assert test_obj.super_called
    extensions = test_obj.super_kwargs['extensions']
    assert len(extensions) == 5

    # Test 4: Context with cookiecutter but no _extensions results in only default extensions
    test_obj = TestClass(context={'cookiecutter': {'name': 'test'}})
    assert test_obj.super_called
    extensions = test_obj.super_kwargs['extensions']
    assert len(extensions) == 5

    # Test 5: Context with extensions adds them to default extensions
    test_obj = TestClass(context={
        'cookiecutter': {
            '_extensions': ['my_ext.Extension1', 'other_ext.Extension2']
        }
    })
    assert test_obj.super_called
    extensions = test_obj.super_kwargs['extensions']
    assert len(extensions) == 7
    assert 'cookiecutter.extensions.JsonifyExtension' in extensions
    assert 'cookiecutter.extensions.RandomStringExtension' in extensions
    assert 'cookiecutter.extensions.SlugifyExtension' in extensions
    assert 'cookiecutter.extensions.TimeExtension' in extensions
    assert 'cookiecutter.extensions.UUIDExtension' in extensions
    assert 'my_ext.Extension1' in extensions
    assert 'other_ext.Extension2' in extensions

    # Test 6: Additional kwargs are passed to parent
    test_obj = TestClass(context={}, some_arg='value', another_arg=123)
    assert test_obj.super_called
    assert test_obj.super_kwargs['some_arg'] == 'value'
    assert test_obj.super_kwargs['another_arg'] == 123
    assert 'extensions' in test_obj.super_kwargs

    # Test 7: Extensions are converted to strings
    test_obj = TestClass(context={
        'cookiecutter': {
            '_extensions': ['ext1', 123, True]
        }
    })
    extensions = test_obj.super_kwargs['extensions']
    assert all(isinstance(ext, str) for ext in extensions)
    assert 'ext1' in extensions
    assert '123' in extensions
    assert 'True' in extensions

    # Test 8: ImportError is converted to UnknownExtension
    class FailingClass(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, *, context=None, **kwargs):
            raise ImportError("No module named 'invalid_extension'")
    
    import_error_raised = False
    try:
        FailingClass()
    except Exception as e:
        import_error_raised = True
        assert "Unable to load extension" in str(e)
    assert import_error_raised


# LLM-generated content at query #5
#--------------------------

```python
def test_ExtensionLoaderMixin():
    # Test 1: Initialize with empty context
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            self.extensions = kwargs.get('extensions', [])
            self.other_args = {k: v for k, v in kwargs.items() if k != 'extensions'}
    
    env = TestEnv(context={})
    assert env.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    
    # Test 2: Initialize with None context
    env = TestEnv(context=None)
    assert env.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    
    # Test 3: Initialize with custom extensions in context
    context = {
        'cookiecutter': {
            '_extensions': ['my_ext.Extension1', 'my_ext.Extension2']
        }
    }
    env = TestEnv(context=context)
    assert env.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my_ext.Extension1',
        'my_ext.Extension2'
    ]
    
    # Test 4: Initialize with empty extensions in context
    context = {'cookiecutter': {'_extensions': []}}
    env = TestEnv(context=context)
    assert env.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    
    # Test 5: Initialize without _extensions key in context
    context = {'cookiecutter': {'other_key': 'value'}}
    env = TestEnv(context=context)
    assert env.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    
    # Test 6: Initialize without cookiecutter key in context
    context = {'other_key': 'value'}
    env = TestEnv(context=context)
    assert env.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    
    # Test 7: Pass additional kwargs
    env = TestEnv(context={}, extra_arg='value')
    assert env.other_args.get('extra_arg') == 'value'
    
    # Test 8: Test that ImportError is converted to UnknownExtension
    class FailingEnv(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            extensions = kwargs.get('extensions', [])
            if extensions:
                raise ImportError("No module named 'my_ext'")
    
    context = {
        'cookiecutter': {
            '_extensions': ['my_ext.InvalidExtension']
        }
    }
    try:
        FailingEnv(context=context)
        assert False, "Should have raised UnknownExtension"
    except UnknownExtension as e:
        assert "Unable to load extension" in str(e)
    
    # Test 9: Test extension conversion to string
    context = {
        'cookiecutter': {
            '_extensions': [123, 456.789]  # Non-string extensions
        }
    }
    env = TestEnv(context=context)
    assert '123' in env.extensions
    assert '456.789' in env.extensions


# LLM-generated content at query #6
#--------------------------

```python
def test_ExtensionLoaderMixin():
    # Test 1: Basic initialization with no context
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            self.super_called = False
            self.extensions_passed = None
            super().__init__(**kwargs)
        
        def __init__mock(self, extensions, **kwargs):
            self.super_called = True
            self.extensions_passed = extensions
    
    # Monkey patch the parent's __init__ to capture arguments
    import cookiecutter.environment
    original_super = TestClass.__mro__[1].__init__
    TestClass.__mro__[1].__init__ = TestClass.__init__mock
    
    try:
        obj = TestClass()
        assert obj.super_called is True
        assert len(obj.extensions_passed) == 5
        assert 'cookiecutter.extensions.JsonifyExtension' in obj.extensions_passed
        assert 'cookiecutter.extensions.RandomStringExtension' in obj.extensions_passed
        assert 'cookiecutter.extensions.SlugifyExtension' in obj.extensions_passed
        assert 'cookiecutter.extensions.TimeExtension' in obj.extensions_passed
        assert 'cookiecutter.extensions.UUIDExtension' in obj.extensions_passed
    finally:
        TestClass.__mro__[1].__init__ = original_super
    
    # Test 2: Initialization with empty context
    class TestClass2(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            self.super_called = False
            self.extensions_passed = None
            super().__init__(**kwargs)
        
        def __init__mock(self, extensions, **kwargs):
            self.super_called = True
            self.extensions_passed = extensions
    
    TestClass2.__mro__[1].__init__ = TestClass2.__init__mock
    
    try:
        obj = TestClass2(context={})
        assert obj.super_called is True
        assert len(obj.extensions_passed) == 5
    finally:
        TestClass2.__mro__[1].__init__ = original_super
    
    # Test 3: Initialization with context containing _extensions
    class TestClass3(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            self.super_called = False
            self.extensions_passed = None
            super().__init__(**kwargs)
        
        def __init__mock(self, extensions, **kwargs):
            self.super_called = True
            self.extensions_passed = extensions
    
    TestClass3.__mro__[1].__init__ = TestClass3.__init__mock
    
    try:
        context = {
            'cookiecutter': {
                '_extensions': ['myextension.Extension1', 'myextension.Extension2']
            }
        }
        obj = TestClass3(context=context)
        assert obj.super_called is True
        assert len(obj.extensions_passed) == 7
        assert 'cookiecutter.extensions.JsonifyExtension' in obj.extensions_passed
        assert 'myextension.Extension1' in obj.extensions_passed
        assert 'myextension.Extension2' in obj.extensions_passed
    finally:
        TestClass3.__mro__[1].__init__ = original_super
    
    # Test 4: Initialization with context missing cookiecutter key
    class TestClass4(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            self.super_called = False
            self.extensions_passed = None
            super().__init__(**kwargs)
        
        def __init__mock(self, extensions, **kwargs):
            self.super_called = True
            self.extensions_passed = extensions
    
    TestClass4.__mro__[1].__init__ = TestClass4.__init__mock
    
    try:
        context = {'other_key': 'value'}
        obj = TestClass4(context=context)
        assert obj.super_called is True
        assert len(obj.extensions_passed) == 5
    finally:
        TestClass4.__mro__[1].__init__ = original_super
    
    # Test 5: Initialization with context containing cookiecutter but missing _extensions
    class TestClass5(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            self.super_called = False
            self.extensions_passed = None
            super().__init__(**kwargs)
        
        def __init__mock(self, extensions, **kwargs):
            self.super_called = True
            self.extensions_passed = extensions
    
    TestClass5.__mro__[1].__init__ = TestClass5.__init__mock
    
    try:
        context = {'cookiecutter': {'other_key': 'value'}}
        obj = TestClass5(context=context)
        assert obj.super_called is True
        assert len(obj.extensions_passed) == 5
    finally:
        TestClass5.__mro__[1].__init__ = original_super
    
    # Test 6: Test ImportError handling
    class TestClass6(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            try:
                super().__init__(**kwargs)
            except Exception as e:
                self.error = e
                raise
        
        def __init__mock(self, extensions, **kwargs):
            raise ImportError("Test import error")
    
    TestClass6.__mro__[1].__init__ = TestClass6.__init__mock
    
    try:
        try:
            obj = TestClass6()
            assert False, "Should have raised an exception"
        except Exception as e:
            assert isinstance(e, UnknownExtension)
            assert "Unable to load extension: Test import error" in str(e)
    finally:
        TestClass6.__mro__[1].__init__ = original_super


# LLM-generated content at query #7
#--------------------------

```python
def test_ExtensionLoaderMixin():
    # Test 1: Default extensions are loaded when no context is provided
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions_received = None
            self.kwargs_received = kwargs
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, extensions, **kwargs):
            self.extensions_received = extensions
            self.kwargs_received = kwargs
    
    # Mock the parent class __init__ call
    test_env = TestEnv()
    assert test_env.extensions_received is not None
    assert len(test_env.extensions_received) == 5
    assert 'cookiecutter.extensions.JsonifyExtension' in test_env.extensions_received
    assert 'cookiecutter.extensions.RandomStringExtension' in test_env.extensions_received
    assert 'cookiecutter.extensions.SlugifyExtension' in test_env.extensions_received
    assert 'cookiecutter.extensions.TimeExtension' in test_env.extensions_received
    assert 'cookiecutter.extensions.UUIDExtension' in test_env.extensions_received

    # Test 2: Additional extensions from context are loaded
    context_with_extensions = {
        'cookiecutter': {
            '_extensions': ['my.custom.Extension1', 'another.Extension2']
        }
    }
    
    test_env = TestEnv(context=context_with_extensions)
    assert len(test_env.extensions_received) == 7
    assert 'my.custom.Extension1' in test_env.extensions_received
    assert 'another.Extension2' in test_env.extensions_received

    # Test 3: Empty context returns only default extensions
    test_env = TestEnv(context={})
    assert len(test_env.extensions_received) == 5

    # Test 4: Context without cookiecutter key returns only default extensions
    test_env = TestEnv(context={'other_key': 'value'})
    assert len(test_env.extensions_received) == 5

    # Test 5: Context with cookiecutter but no _extensions returns only default extensions
    test_env = TestEnv(context={'cookiecutter': {'name': 'test'}})
    assert len(test_env.extensions_received) == 5

    # Test 6: Additional kwargs are passed through
    test_env = TestEnv(context=None, autoescape=True, trim_blocks=True)
    assert test_env.kwargs_received.get('autoescape') is True
    assert test_env.kwargs_received.get('trim_blocks') is True

    # Test 7: ImportError is converted to UnknownExtension
    class FailingEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.kwargs = kwargs
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, extensions, **kwargs):
            raise ImportError("No module named 'invalid.extension'")
    
    import pytest
    from cookiecutter.exceptions import UnknownExtension
    
    with pytest.raises(UnknownExtension) as exc_info:
        FailingEnv()
    assert "Unable to load extension" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ImportError)

    # Test 8: _read_extensions method works correctly
    loader = ExtensionLoaderMixin(context={})
    
    # Test with no extensions
    result = loader._read_extensions({})
    assert result == []
    
    # Test with cookiecutter but no _extensions
    result = loader._read_extensions({'cookiecutter': {'name': 'test'}})
    assert result == []
    
    # Test with extensions
    context = {'cookiecutter': {'_extensions': ['ext1', 'ext2']}}
    result = loader._read_extensions(context)
    assert result == ['ext1', 'ext2']
    
    # Test with non-string extensions (should be converted to string)
    context = {'cookiecutter': {'_extensions': [123, True]}}
    result = loader._read_extensions(context)
    assert result == ['123', 'True']


# LLM-generated content at query #8
#--------------------------

```python
def test_ExtensionLoaderMixin():
    class TestEnvironment:
        def __init__(self, extensions=None, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs

    class TestClass(ExtensionLoaderMixin, TestEnvironment):
        pass

    context_with_extensions = {
        'cookiecutter': {
            '_extensions': [
                'custom.Extension1',
                'custom.Extension2'
            ]
        }
    }

    default_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    obj = TestClass(context=context_with_extensions, test_arg='value')
    expected_extensions = default_extensions + ['custom.Extension1', 'custom.Extension2']
    assert obj.extensions == expected_extensions
    assert obj.kwargs == {'test_arg': 'value'}

    obj_no_context = TestClass(test_arg='value')
    assert obj_no_context.extensions == default_extensions
    assert obj_no_context.kwargs == {'test_arg': 'value'}

    context_empty = {}
    obj_empty = TestClass(context=context_empty)
    assert obj_empty.extensions == default_extensions

    context_no_extensions = {'cookiecutter': {}}
    obj_no_extensions = TestClass(context=context_no_extensions)
    assert obj_no_extensions.extensions == default_extensions

    context_wrong_structure = {'not_cookiecutter': {}}
    obj_wrong = TestClass(context=context_wrong_structure)
    assert obj_wrong.extensions == default_extensions


# LLM-generated content at query #9
#--------------------------

```python
def test_ExtensionLoaderMixin():
    # Test 1: Basic initialization with no context
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extensions = kwargs.get('extensions', [])
    
    obj = TestClass()
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    
    # Test 2: Initialization with empty context
    obj = TestClass(context={})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    
    # Test 3: Initialization with context containing _extensions
    context = {
        'cookiecutter': {
            '_extensions': ['my_extension.Extension1', 'other.Extension2']
        }
    }
    obj = TestClass(context=context)
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my_extension.Extension1',
        'other.Extension2',
    ]
    
    # Test 4: Initialization with context but no _extensions key
    context = {'cookiecutter': {'other_key': 'value'}}
    obj = TestClass(context=context)
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    
    # Test 5: Initialization with context but no cookiecutter key
    context = {'other_key': 'value'}
    obj = TestClass(context=context)
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    
    # Test 6: Test that ImportError is converted to UnknownExtension
    class FailingTestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            try:
                super().__init__(**kwargs)
            except Exception as e:
                self.error = e
    
    # Simulate an ImportError by passing an invalid extension
    context = {
        'cookiecutter': {
            '_extensions': ['nonexistent.extension']
        }
    }
    obj = FailingTestClass(context=context)
    assert isinstance(obj.error, UnknownExtension)
    assert 'Unable to load extension' in str(obj.error)
    
    # Test 7: Additional kwargs are passed through
    class KwargsTestClass(ExtensionLoaderMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.kwargs = kwargs
    
    obj = KwargsTestClass(context={}, extra_param='test')
    assert 'extra_param' in obj.kwargs
    assert obj.kwargs['extra_param'] == 'test'


# LLM-generated content at query #10
#--------------------------

```python
def test_ExtensionLoaderMixin():
    # Test 1: Basic initialization with no context
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.called_super = False
            self.extensions_passed = None
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, extensions, **kwargs):
            self.called_super = True
            self.extensions_passed = extensions
    
    # Mock the parent class __init__ call
    import cookiecutter.extensions
    original_init = TestClass.__mro__[1].__init__
    TestClass.__mro__[1].__init__ = TestClass.__init__mock
    
    try:
        obj = TestClass(context={})
        assert obj.called_super is True
        assert len(obj.extensions_passed) == 5
        assert 'cookiecutter.extensions.JsonifyExtension' in obj.extensions_passed
        assert 'cookiecutter.extensions.RandomStringExtension' in obj.extensions_passed
        assert 'cookiecutter.extensions.SlugifyExtension' in obj.extensions_passed
        assert 'cookiecutter.extensions.TimeExtension' in obj.extensions_passed
        assert 'cookiecutter.extensions.UUIDExtension' in obj.extensions_passed
    finally:
        TestClass.__mro__[1].__init__ = original_init
    
    # Test 2: Initialization with custom extensions in context
    class TestClass2(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.called_super = False
            self.extensions_passed = None
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, extensions, **kwargs):
            self.called_super = True
            self.extensions_passed = extensions
    
    TestClass2.__mro__[1].__init__ = TestClass2.__init__mock
    
    try:
        context = {
            'cookiecutter': {
                '_extensions': ['my.custom.Extension1', 'another.Extension2']
            }
        }
        obj = TestClass2(context=context)
        assert obj.called_super is True
        assert len(obj.extensions_passed) == 7
        assert 'my.custom.Extension1' in obj.extensions_passed
        assert 'another.Extension2' in obj.extensions_passed
        assert 'cookiecutter.extensions.JsonifyExtension' in obj.extensions_passed
    finally:
        TestClass2.__mro__[1].__init__ = original_init
    
    # Test 3: Initialization with empty extensions list in context
    class TestClass3(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.called_super = False
            self.extensions_passed = None
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, extensions, **kwargs):
            self.called_super = True
            self.extensions_passed = extensions
    
    TestClass3.__mro__[1].__init__ = TestClass3.__init__mock
    
    try:
        context = {
            'cookiecutter': {
                '_extensions': []
            }
        }
        obj = TestClass3(context=context)
        assert obj.called_super is True
        assert len(obj.extensions_passed) == 5
    finally:
        TestClass3.__mro__[1].__init__ = original_init
    
    # Test 4: Initialization with no cookiecutter key in context
    class TestClass4(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.called_super = False
            self.extensions_passed = None
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, extensions, **kwargs):
            self.called_super = True
            self.extensions_passed = extensions
    
    TestClass4.__mro__[1].__init__ = TestClass4.__init__mock
    
    try:
        context = {'other_key': 'value'}
        obj = TestClass4(context=context)
        assert obj.called_super is True
        assert len(obj.extensions_passed) == 5
    finally:
        TestClass4.__mro__[1].__init__ = original_init
    
    # Test 5: Test ImportError handling
    class TestClass5(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            try:
                super().__init__(context=context, **kwargs)
            except Exception as e:
                self.raised_error = e
                raise
    
    def raising_init(extensions, **kwargs):
        raise ImportError("No module named 'invalid.extension'")
    
    TestClass5.__mro__[1].__init__ = raising_init
    
    try:
        try:
            obj = TestClass5(context={})
            assert False, "Should have raised UnknownExtension"
        except Exception as e:
            assert isinstance(e, UnknownExtension)
            assert "Unable to load extension" in str(e)
            assert isinstance(e.__cause__, ImportError)
    finally:
        TestClass5.__mro__[1].__init__ = original_init
    
    # Test 6: Test with None context
    class TestClass6(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.called_super = False
            self.extensions_passed = None
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, extensions, **kwargs):
            self.called_super = True
            self.extensions_passed = extensions
    
    TestClass6.__mro__[1].__init__ = TestClass6.__init__mock
    
    try:
        obj = TestClass6(context=None)
        assert obj.called_super is True
        assert len(obj.extensions_passed) == 5
    finally:
        TestClass6.__mro__[1].__init__ = original_init


# LLM-generated content at query #11
#--------------------------

```python
def test_ExtensionLoaderMixin():
    class TestEnvironment:
        def __init__(self, extensions=None, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs

    class TestClass(ExtensionLoaderMixin, TestEnvironment):
        pass

    # Test with no context
    obj = TestClass()
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with empty context
    obj = TestClass(context={})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with context but no extensions
    obj = TestClass(context={'cookiecutter': {'other_key': 'value'}})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with custom extensions
    custom_extensions = ['my_ext.Extension1', 'my_ext.Extension2']
    obj = TestClass(context={'cookiecutter': {'_extensions': custom_extensions}})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my_ext.Extension1',
        'my_ext.Extension2',
    ]

    # Test with custom extensions as non-strings
    custom_extensions = [123, 456]
    obj = TestClass(context={'cookiecutter': {'_extensions': custom_extensions}})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        '123',
        '456',
    ]

    # Test that kwargs are passed to parent
    obj = TestClass(context={}, trim_blocks=True, lstrip_blocks=True)
    assert obj.kwargs == {'trim_blocks': True, 'lstrip_blocks': True}

    # Test ImportError handling
    class FailingEnvironment:
        def __init__(self, extensions=None, **kwargs):
            raise ImportError("No module named 'invalid_ext'")

    class FailingClass(ExtensionLoaderMixin, FailingEnvironment):
        pass

    try:
        FailingClass(context={})
        assert False, "Should have raised UnknownExtension"
    except UnknownExtension as e:
        assert "Unable to load extension" in str(e)
        assert isinstance(e.__cause__, ImportError)


# LLM-generated content at query #12
#--------------------------

```python
def test_ExtensionLoaderMixin():
    class TestEnvironment:
        def __init__(self, extensions=None, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs

    class TestClass(ExtensionLoaderMixin, TestEnvironment):
        pass

    # Test with no context
    obj = TestClass()
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with empty context
    obj = TestClass(context={})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with context but no extensions
    obj = TestClass(context={'cookiecutter': {'key': 'value'}})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with custom extensions
    custom_exts = ['my_ext.Extension1', 'other_ext.Extension2']
    obj = TestClass(context={'cookiecutter': {'_extensions': custom_exts}})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my_ext.Extension1',
        'other_ext.Extension2',
    ]

    # Test with custom extensions as non-string types
    custom_exts = ['ext1', 123, True]
    obj = TestClass(context={'cookiecutter': {'_extensions': custom_exts}})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'ext1',
        '123',
        'True',
    ]

    # Test passing additional kwargs
    obj = TestClass(context={'cookiecutter': {'_extensions': ['ext1']}}, extra_arg='value')
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'ext1',
    ]
    assert obj.kwargs == {'extra_arg': 'value'}

    # Test ImportError handling
    class FailingEnvironment:
        def __init__(self, extensions=None, **kwargs):
            raise ImportError("No module named 'invalid_ext'")

    class FailingClass(ExtensionLoaderMixin, FailingEnvironment):
        pass

    try:
        FailingClass(context={'cookiecutter': {'_extensions': ['invalid_ext']}})
        assert False, "Should have raised UnknownExtension"
    except UnknownExtension as e:
        assert "Unable to load extension" in str(e)


# LLM-generated content at query #13
#--------------------------

```python
def test_ExtensionLoaderMixin():
    # Test 1: Basic initialization with no context
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.called_super = False
            self.extensions_passed = None
            self.kwargs_passed = None
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, extensions, **kwargs):
            self.called_super = True
            self.extensions_passed = extensions
            self.kwargs_passed = kwargs
    
    # Replace the parent's __init__ with our mock
    original_super = TestClass.__mro__[1].__init__
    TestClass.__mro__[1].__init__ = TestClass.__init__mock
    
    try:
        obj = TestClass()
        assert obj.called_super is True
        assert len(obj.extensions_passed) == 5
        assert 'cookiecutter.extensions.JsonifyExtension' in obj.extensions_passed
        assert 'cookiecutter.extensions.RandomStringExtension' in obj.extensions_passed
        assert 'cookiecutter.extensions.SlugifyExtension' in obj.extensions_passed
        assert 'cookiecutter.extensions.TimeExtension' in obj.extensions_passed
        assert 'cookiecutter.extensions.UUIDExtension' in obj.extensions_passed
    finally:
        # Restore original __init__
        TestClass.__mro__[1].__init__ = original_super
    
    # Test 2: Initialization with empty context
    TestClass.__mro__[1].__init__ = TestClass.__init__mock
    try:
        obj = TestClass(context={})
        assert obj.called_super is True
        assert len(obj.extensions_passed) == 5
    finally:
        TestClass.__mro__[1].__init__ = original_super
    
    # Test 3: Initialization with context containing extensions
    TestClass.__mro__[1].__init__ = TestClass.__init__mock
    try:
        context = {
            'cookiecutter': {
                '_extensions': ['my_ext.Extension1', 'my_ext.Extension2']
            }
        }
        obj = TestClass(context=context)
        assert obj.called_super is True
        assert len(obj.extensions_passed) == 7
        assert 'my_ext.Extension1' in obj.extensions_passed
        assert 'my_ext.Extension2' in obj.extensions_passed
        assert 'cookiecutter.extensions.JsonifyExtension' in obj.extensions_passed
    finally:
        TestClass.__mro__[1].__init__ = original_super
    
    # Test 4: Initialization with context but no _extensions key
    TestClass.__mro__[1].__init__ = TestClass.__init__mock
    try:
        context = {'cookiecutter': {'other_key': 'value'}}
        obj = TestClass(context=context)
        assert obj.called_super is True
        assert len(obj.extensions_passed) == 5
    finally:
        TestClass.__mro__[1].__init__ = original_super
    
    # Test 5: Initialization with additional kwargs
    TestClass.__mro__[1].__init__ = TestClass.__init__mock
    try:
        obj = TestClass(context={}, extra_arg='value', another_arg=123)
        assert obj.called_super is True
        assert obj.kwargs_passed['extra_arg'] == 'value'
        assert obj.kwargs_passed['another_arg'] == 123
    finally:
        TestClass.__mro__[1].__init__ = original_super
    
    # Test 6: Test ImportError handling
    class TestClassWithError(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, extensions, **kwargs):
            raise ImportError("No module named 'invalid_extension'")
    
    TestClassWithError.__mro__[1].__init__ = TestClassWithError.__init__mock
    try:
        try:
            obj = TestClassWithError()
            assert False, "Should have raised UnknownExtension"
        except UnknownExtension as e:
            assert "Unable to load extension" in str(e)
            assert isinstance(e.__cause__, ImportError)
    finally:
        TestClassWithError.__mro__[1].__init__ = original_super


# LLM-generated content at query #14
#--------------------------

```python
def test_ExtensionLoaderMixin():
    class TestEnvironment:
        def __init__(self, extensions=None, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs

    class TestClass(ExtensionLoaderMixin, TestEnvironment):
        pass

    # Test with no context
    obj = TestClass()
    expected_extensions = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert obj.extensions == expected_extensions

    # Test with empty context
    obj = TestClass(context={})
    assert obj.extensions == expected_extensions

    # Test with context containing no extensions
    obj = TestClass(context={'cookiecutter': {'key': 'value'}})
    assert obj.extensions == expected_extensions

    # Test with context containing extensions
    custom_extensions = ['my_ext.Extension1', 'my_ext.Extension2']
    obj = TestClass(context={'cookiecutter': {'_extensions': custom_extensions}})
    assert obj.extensions == expected_extensions + custom_extensions

    # Test with additional kwargs
    obj = TestClass(context={'cookiecutter': {'_extensions': ['ext.Ext']}}, other_arg=123)
    assert obj.extensions == expected_extensions + ['ext.Ext']
    assert obj.kwargs['other_arg'] == 123

    # Test with string extensions in context
    obj = TestClass(context={'cookiecutter': {'_extensions': ['ext1', 'ext2']}})
    assert obj.extensions == expected_extensions + ['ext1', 'ext2']

    # Test with non-string extensions in context (should be converted to string)
    obj = TestClass(context={'cookiecutter': {'_extensions': [123, True]}})
    assert obj.extensions == expected_extensions + ['123', 'True']

    # Test error handling when extension fails to load
    class FailingEnvironment:
        def __init__(self, extensions=None, **kwargs):
            raise ImportError("No module named 'invalid_ext'")

    class FailingClass(ExtensionLoaderMixin, FailingEnvironment):
        pass

    try:
        FailingClass(context={'cookiecutter': {'_extensions': ['invalid_ext']}})
        assert False, "Should have raised UnknownExtension"
    except UnknownExtension as e:
        assert "Unable to load extension" in str(e)


# LLM-generated content at query #15
#--------------------------

```python
def test_ExtensionLoaderMixin():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.called_super = False
            self.extensions_passed = None
            self.kwargs_passed = None
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, *, context=None, **kwargs):
            self.called_super = True
            self.extensions_passed = kwargs.get('extensions')
            self.kwargs_passed = kwargs
    
    test_obj = TestClass(context={})
    assert test_obj.called_super
    assert len(test_obj.extensions_passed) == 5
    assert 'cookiecutter.extensions.JsonifyExtension' in test_obj.extensions_passed
    assert 'cookiecutter.extensions.RandomStringExtension' in test_obj.extensions_passed
    assert 'cookiecutter.extensions.SlugifyExtension' in test_obj.extensions_passed
    assert 'cookiecutter.extensions.TimeExtension' in test_obj.extensions_passed
    assert 'cookiecutter.extensions.UUIDExtension' in test_obj.extensions_passed
    
    custom_extensions = ['my_ext.Extension1', 'my_ext.Extension2']
    context_with_extensions = {
        'cookiecutter': {
            '_extensions': custom_extensions
        }
    }
    test_obj2 = TestClass(context=context_with_extensions)
    assert test_obj2.called_super
    assert len(test_obj2.extensions_passed) == 7
    assert all(ext in test_obj2.extensions_passed for ext in custom_extensions)
    
    context_without_extensions_key = {'cookiecutter': {}}
    test_obj3 = TestClass(context=context_without_extensions_key)
    assert test_obj3.called_super
    assert len(test_obj3.extensions_passed) == 5
    
    context_without_cookiecutter = {'other_key': 'value'}
    test_obj4 = TestClass(context=context_without_cookiecutter)
    assert test_obj4.called_super
    assert len(test_obj4.extensions_passed) == 5
    
    test_obj5 = TestClass()
    assert test_obj5.called_super
    assert len(test_obj5.extensions_passed) == 5
    
    additional_kwargs = {'trim_blocks': True, 'lstrip_blocks': False}
    test_obj6 = TestClass(context={}, **additional_kwargs)
    assert test_obj6.called_super
    assert test_obj6.kwargs_passed['trim_blocks'] == True
    assert test_obj6.kwargs_passed['lstrip_blocks'] == False
    assert 'extensions' in test_obj6.kwargs_passed
    
    class FailingClass(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.mock_super_called = False
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, *, context=None, **kwargs):
            self.mock_super_called = True
            raise ImportError("Test import error")
    
    import pytest
    from cookiecutter.exceptions import UnknownExtension
    
    with pytest.raises(UnknownExtension) as exc_info:
        FailingClass(context={})
    assert "Unable to load extension" in str(exc_info.value)


# LLM-generated content at query #16
#--------------------------

```python
def test_ExtensionLoaderMixin():
    class TestEnvironment:
        def __init__(self, extensions=None, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs

    class TestExtensionLoader(ExtensionLoaderMixin, TestEnvironment):
        pass

    # Test with no context
    env = TestExtensionLoader()
    assert env.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with empty context
    env = TestExtensionLoader(context={})
    assert env.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with context but no extensions
    env = TestExtensionLoader(context={'cookiecutter': {'other': 'value'}})
    assert env.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with custom extensions
    custom_extensions = ['my_ext.Extension1', 'another.Extension2']
    env = TestExtensionLoader(
        context={'cookiecutter': {'_extensions': custom_extensions}}
    )
    assert env.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my_ext.Extension1',
        'another.Extension2',
    ]

    # Test with custom extensions as non-strings
    custom_extensions = [123, 456.789]
    env = TestExtensionLoader(
        context={'cookiecutter': {'_extensions': custom_extensions}}
    )
    assert env.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        '123',
        '456.789',
    ]

    # Test that kwargs are passed through
    env = TestExtensionLoader(context={}, test_param='value', another=42)
    assert env.kwargs == {'test_param': 'value', 'another': 42}
    assert env.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test ImportError handling
    class FailingEnvironment:
        def __init__(self, extensions=None, **kwargs):
            raise ImportError("No module named 'invalid_extension'")

    class FailingExtensionLoader(ExtensionLoaderMixin, FailingEnvironment):
        pass

    try:
        FailingExtensionLoader(context={})
        assert False, "Should have raised UnknownExtension"
    except UnknownExtension as e:
        assert "Unable to load extension" in str(e)
        assert isinstance(e.__cause__, ImportError)


# LLM-generated content at query #17
#--------------------------

```python
def test_ExtensionLoaderMixin():
    # Test 1: Default extensions are loaded when no context provided
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.extensions = kwargs.get('extensions', [])
            super().__init__(context=context, **kwargs)
    
    env = TestEnv()
    expected_defaults = [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]
    assert env.extensions == expected_defaults
    
    # Test 2: Extensions from context are added to default extensions
    context = {
        'cookiecutter': {
            '_extensions': ['my_ext.Extension1', 'other_ext.Extension2']
        }
    }
    env = TestEnv(context=context)
    assert env.extensions == expected_defaults + ['my_ext.Extension1', 'other_ext.Extension2']
    
    # Test 3: Empty extensions list in context
    context = {'cookiecutter': {'_extensions': []}}
    env = TestEnv(context=context)
    assert env.extensions == expected_defaults
    
    # Test 4: No _extensions key in context
    context = {'cookiecutter': {}}
    env = TestEnv(context=context)
    assert env.extensions == expected_defaults
    
    # Test 5: No cookiecutter key in context
    context = {}
    env = TestEnv(context=context)
    assert env.extensions == expected_defaults
    
    # Test 6: Non-string extensions in context are converted to strings
    context = {
        'cookiecutter': {
            '_extensions': ['ext1', 123, True]
        }
    }
    env = TestEnv(context=context)
    assert env.extensions == expected_defaults + ['ext1', '123', 'True']
    
    # Test 7: ImportError is caught and converted to UnknownExtension
    class FailingEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.kwargs = kwargs
            # Simulate ImportError from parent's __init__
            raise ImportError("No module named 'nonexistent'")
    
    try:
        FailingEnv()
        assert False, "Should have raised UnknownExtension"
    except UnknownExtension as e:
        assert "Unable to load extension" in str(e)
    
    # Test 8: Additional kwargs are passed to parent
    class KwargsEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.kwargs = kwargs
            super().__init__(context=context, **kwargs)
    
    env = KwargsEnv(context={}, extra_param='test', another_param=123)
    assert env.kwargs['extra_param'] == 'test'
    assert env.kwargs['another_param'] == 123


# LLM-generated content at query #18
#--------------------------

```python
def test_ExtensionLoaderMixin():
    # Test 1: Initialize with empty context
    class TestEnv(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.called_super = False
            self.extensions_passed = None
            super().__init__(context=context, **kwargs)
        
        def __init__mock(self, extensions, **kwargs):
            self.called_super = True
            self.extensions_passed = extensions
            self.kwargs_passed = kwargs
    
    # Replace the parent's __init__ with our mock
    import cookiecutter.environment
    original_init = cookiecutter.environment.ExtensionLoaderMixin.__init__
    
    def mock_init(self, *, context=None, **kwargs):
        self.called_super = False
        self.extensions_passed = None
        self.kwargs_passed = None
        # Call the actual ExtensionLoaderMixin.__init__ but intercept the super().__init__ call
        self.__class__.__bases__[0].__init__ = lambda self_, extensions, **kw: (
            setattr(self_, 'called_super', True) or 
            setattr(self_, 'extensions_passed', extensions) or
            setattr(self_, 'kwargs_passed', kw)
        )
        original_init(self, context=context, **kwargs)
    
    cookiecutter.environment.ExtensionLoaderMixin.__init__ = mock_init
    
    try:
        # Test with empty context
        env = TestEnv(context={})
        assert env.called_super
        default_extensions = [
            'cookiecutter.extensions.JsonifyExtension',
            'cookiecutter.extensions.RandomStringExtension',
            'cookiecutter.extensions.SlugifyExtension',
            'cookiecutter.extensions.TimeExtension',
            'cookiecutter.extensions.UUIDExtension',
        ]
        assert env.extensions_passed == default_extensions
        
        # Test 2: Initialize with extensions in context
        context_with_extensions = {
            'cookiecutter': {
                '_extensions': ['myextension.Extension1', 'anotherextension.Extension2']
            }
        }
        env2 = TestEnv(context=context_with_extensions)
        assert env2.called_super
        expected_extensions = default_extensions + ['myextension.Extension1', 'anotherextension.Extension2']
        assert env2.extensions_passed == expected_extensions
        
        # Test 3: Initialize with None context
        env3 = TestEnv(context=None)
        assert env3.called_super
        assert env3.extensions_passed == default_extensions
        
        # Test 4: Initialize with context missing cookiecutter key
        env4 = TestEnv(context={'other_key': 'value'})
        assert env4.called_super
        assert env4.extensions_passed == default_extensions
        
        # Test 5: Initialize with context missing _extensions key
        env5 = TestEnv(context={'cookiecutter': {'other_key': 'value'}})
        assert env5.called_super
        assert env5.extensions_passed == default_extensions
        
        # Test 6: Test that ImportError is converted to UnknownExtension
        class FailingEnv(ExtensionLoaderMixin):
            def __init__(self, *, context=None, **kwargs):
                self.called_super = False
                super().__init__(context=context, **kwargs)
            
            def __init__mock(self, extensions, **kwargs):
                raise ImportError("Test import error")
        
        # Monkey patch to force ImportError
        import cookiecutter.exceptions
        original_super_init = FailingEnv.__bases__[0].__init__
        FailingEnv.__bases__[0].__init__ = lambda self, extensions, **kwargs: (_ for _ in ()).throw(ImportError("Test import error"))
        
        try:
            env6 = FailingEnv(context={})
            assert False, "Should have raised UnknownExtension"
        except cookiecutter.exceptions.UnknownExtension as e:
            assert "Unable to load extension: Test import error" in str(e)
        finally:
            FailingEnv.__bases__[0].__init__ = original_super_init
        
        # Test 7: Test with additional kwargs passed through
        env7 = TestEnv(context={}, trim_blocks=True, lstrip_blocks=True)
        assert env7.called_super
        assert env7.extensions_passed == default_extensions
        assert env7.kwargs_passed.get('trim_blocks') == True
        assert env7.kwargs_passed.get('lstrip_blocks') == True
        
    finally:
        # Restore original __init__
        cookiecutter.environment.ExtensionLoaderMixin.__init__ = original_init


# LLM-generated content at query #19
#--------------------------

```python
def test_ExtensionLoaderMixin():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.super_called = False
            self.super_kwargs = None
            super().__init__(context=context, **kwargs)

        def __init_subclass__(cls):
            pass

    # Test with no context
    obj = TestClass()
    assert obj.super_called
    assert 'extensions' in obj.super_kwargs
    assert len(obj.super_kwargs['extensions']) == 5
    assert 'cookiecutter.extensions.JsonifyExtension' in obj.super_kwargs['extensions']
    assert 'cookiecutter.extensions.RandomStringExtension' in obj.super_kwargs['extensions']
    assert 'cookiecutter.extensions.SlugifyExtension' in obj.super_kwargs['extensions']
    assert 'cookiecutter.extensions.TimeExtension' in obj.super_kwargs['extensions']
    assert 'cookiecutter.extensions.UUIDExtension' in obj.super_kwargs['extensions']

    # Test with empty context
    obj = TestClass(context={})
    assert obj.super_called
    assert len(obj.super_kwargs['extensions']) == 5

    # Test with context but no extensions
    obj = TestClass(context={'cookiecutter': {'other_key': 'value'}})
    assert obj.super_called
    assert len(obj.super_kwargs['extensions']) == 5

    # Test with custom extensions
    custom_extensions = ['my_ext.Extension1', 'my_ext.Extension2']
    obj = TestClass(context={'cookiecutter': {'_extensions': custom_extensions}})
    assert obj.super_called
    assert len(obj.super_kwargs['extensions']) == 7
    assert all(ext in obj.super_kwargs['extensions'] for ext in custom_extensions)
    assert 'cookiecutter.extensions.JsonifyExtension' in obj.super_kwargs['extensions']

    # Test with string conversion of extensions
    obj = TestClass(context={'cookiecutter': {'_extensions': [123, 456]}})
    assert obj.super_called
    assert '123' in obj.super_kwargs['extensions']
    assert '456' in obj.super_kwargs['extensions']

    # Test ImportError handling
    class FailingTestClass(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.super_called = False
            super().__init__(context=context, **kwargs)

        def __init_subclass__(cls):
            pass

    import builtins
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if 'InvalidExtension' in name:
            raise ImportError(f"No module named '{name}'")
        return original_import(name, *args, **kwargs)

    builtins.__import__ = mock_import
    try:
        import pytest
        with pytest.raises(UnknownExtension) as exc_info:
            FailingTestClass(context={'cookiecutter': {'_extensions': ['InvalidExtension']}})
        assert "Unable to load extension" in str(exc_info.value)
    finally:
        builtins.__import__ = original_import


# LLM-generated content at query #20
#--------------------------

```python
def test_ExtensionLoaderMixin():
    class TestEnvironment:
        def __init__(self, extensions=None, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs

    class TestClass(ExtensionLoaderMixin, TestEnvironment):
        pass

    # Test with no context
    obj = TestClass()
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with empty context
    obj = TestClass(context={})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with context but no extensions
    obj = TestClass(context={'cookiecutter': {'key': 'value'}})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with custom extensions
    custom_exts = ['my_ext.Extension1', 'other_ext.Extension2']
    obj = TestClass(context={'cookiecutter': {'_extensions': custom_exts}})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my_ext.Extension1',
        'other_ext.Extension2',
    ]

    # Test with additional kwargs
    obj = TestClass(context={'cookiecutter': {'_extensions': ['ext.Ext']}}, extra_arg='value')
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'ext.Ext',
    ]
    assert obj.kwargs['extra_arg'] == 'value'

    # Test with non-string extensions in context
    obj = TestClass(context={'cookiecutter': {'_extensions': [123, 'ext.Ext']}})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        '123',
        'ext.Ext',
    ]

    # Test ImportError handling
    class FailingEnvironment:
        def __init__(self, extensions=None, **kwargs):
            raise ImportError("No module named 'invalid_ext'")

    class FailingClass(ExtensionLoaderMixin, FailingEnvironment):
        pass

    try:
        FailingClass(context={'cookiecutter': {'_extensions': ['invalid_ext']}})
        assert False, "Should have raised UnknownExtension"
    except UnknownExtension as e:
        assert "Unable to load extension" in str(e)


# LLM-generated content at query #21
#--------------------------

```python
def test_ExtensionLoaderMixin():
    class TestEnvironment:
        def __init__(self, extensions=None, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs

    class TestClass(ExtensionLoaderMixin, TestEnvironment):
        pass

    # Test with no context
    obj = TestClass()
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with empty context
    obj = TestClass(context={})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with context but no _extensions
    obj = TestClass(context={'cookiecutter': {'key': 'value'}})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with custom extensions
    custom_exts = ['my_ext.Extension1', 'other_ext.Extension2']
    obj = TestClass(context={'cookiecutter': {'_extensions': custom_exts}})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my_ext.Extension1',
        'other_ext.Extension2',
    ]

    # Test with string conversion of extensions
    obj = TestClass(context={'cookiecutter': {'_extensions': [1, 2.5, True]}})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        '1',
        '2.5',
        'True',
    ]

    # Test that kwargs are passed to parent
    obj = TestClass(context={}, autoescape=True, trim_blocks=False)
    assert obj.kwargs['autoescape'] is True
    assert obj.kwargs['trim_blocks'] is False

    # Test ImportError handling
    class FailingEnvironment:
        def __init__(self, extensions=None, **kwargs):
            raise ImportError("No module named 'invalid_ext'")

    class FailingTestClass(ExtensionLoaderMixin, FailingEnvironment):
        pass

    try:
        FailingTestClass(context={})
        assert False, "Should have raised UnknownExtension"
    except UnknownExtension as e:
        assert "Unable to load extension" in str(e)
        assert isinstance(e.__cause__, ImportError)


# LLM-generated content at query #22
#--------------------------

```python
def test_ExtensionLoaderMixin():
    class TestClass(ExtensionLoaderMixin):
        def __init__(self, *, context=None, **kwargs):
            self.super_called_with = kwargs
            super().__init__(context=context, **kwargs)

    # Test with no context
    obj = TestClass()
    assert 'extensions' in obj.super_called_with
    assert len(obj.super_called_with['extensions']) == 5
    assert 'cookiecutter.extensions.JsonifyExtension' in obj.super_called_with['extensions']
    assert 'cookiecutter.extensions.RandomStringExtension' in obj.super_called_with['extensions']
    assert 'cookiecutter.extensions.SlugifyExtension' in obj.super_called_with['extensions']
    assert 'cookiecutter.extensions.TimeExtension' in obj.super_called_with['extensions']
    assert 'cookiecutter.extensions.UUIDExtension' in obj.super_called_with['extensions']

    # Test with empty context
    obj = TestClass(context={})
    assert len(obj.super_called_with['extensions']) == 5

    # Test with context containing no _extensions
    obj = TestClass(context={'cookiecutter': {'other_key': 'value'}})
    assert len(obj.super_called_with['extensions']) == 5

    # Test with context containing empty _extensions
    obj = TestClass(context={'cookiecutter': {'_extensions': []}})
    assert len(obj.super_called_with['extensions']) == 5

    # Test with context containing additional extensions
    extensions = ['my_ext.Extension1', 'my_ext.Extension2']
    obj = TestClass(context={'cookiecutter': {'_extensions': extensions}})
    assert len(obj.super_called_with['extensions']) == 7
    assert 'my_ext.Extension1' in obj.super_called_with['extensions']
    assert 'my_ext.Extension2' in obj.super_called_with['extensions']
    assert obj.super_called_with['extensions'][:5] == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test that other kwargs are passed through
    obj = TestClass(context={'cookiecutter': {'_extensions': []}}, other_arg='value')
    assert obj.super_called_with['other_arg'] == 'value'
    assert 'extensions' in obj.super_called_with

    # Test with non-string extensions in context (should be converted to string)
    obj = TestClass(context={'cookiecutter': {'_extensions': [123, 456.7]}})
    assert len(obj.super_called_with['extensions']) == 7
    assert '123' in obj.super_called_with['extensions']
    assert '456.7' in obj.super_called_with['extensions']


# LLM-generated content at query #23
#--------------------------

```python
def test_ExtensionLoaderMixin():
    class TestEnvironment:
        def __init__(self, extensions=None, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs

    class TestClass(ExtensionLoaderMixin, TestEnvironment):
        pass

    # Test with no context
    obj = TestClass()
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with empty context
    obj = TestClass(context={})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with context containing no extensions
    obj = TestClass(context={'cookiecutter': {'other_key': 'value'}})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with context containing extensions
    obj = TestClass(context={
        'cookiecutter': {
            '_extensions': ['my_ext.Extension1', 'other_ext.Extension2']
        }
    })
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my_ext.Extension1',
        'other_ext.Extension2'
    ]

    # Test with additional kwargs passed through
    obj = TestClass(context={'cookiecutter': {}}, extra_arg='value')
    assert obj.kwargs.get('extra_arg') == 'value'

    # Test that ImportError is converted to UnknownExtension
    class FailingEnvironment:
        def __init__(self, extensions=None, **kwargs):
            raise ImportError("No module named 'bad_extension'")

    class FailingTestClass(ExtensionLoaderMixin, FailingEnvironment):
        pass

    try:
        FailingTestClass(context={'cookiecutter': {}})
        assert False, "Should have raised UnknownExtension"
    except UnknownExtension as e:
        assert "Unable to load extension" in str(e)
        assert isinstance(e.__cause__, ImportError)


# LLM-generated content at query #24
#--------------------------

```python
def test_ExtensionLoaderMixin():
    class TestEnvironment:
        def __init__(self, extensions=None, **kwargs):
            self.extensions = extensions
            self.kwargs = kwargs

    class TestClass(ExtensionLoaderMixin, TestEnvironment):
        pass

    # Test with no context
    obj = TestClass()
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with empty context
    obj = TestClass(context={})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with context but no _extensions
    obj = TestClass(context={'cookiecutter': {'other_key': 'value'}})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
    ]

    # Test with custom extensions
    custom_exts = ['my_ext.Extension1', 'other_ext.Extension2']
    obj = TestClass(context={'cookiecutter': {'_extensions': custom_exts}})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my_ext.Extension1',
        'other_ext.Extension2',
    ]

    # Test with custom extensions as non-strings
    custom_exts = ['my_ext.Extension1', 123, True]
    obj = TestClass(context={'cookiecutter': {'_extensions': custom_exts}})
    assert obj.extensions == [
        'cookiecutter.extensions.JsonifyExtension',
        'cookiecutter.extensions.RandomStringExtension',
        'cookiecutter.extensions.SlugifyExtension',
        'cookiecutter.extensions.TimeExtension',
        'cookiecutter.extensions.UUIDExtension',
        'my_ext.Extension1',
        '123',
        'True',
    ]

    # Test that kwargs are passed through
    obj = TestClass(context={}, trim_blocks=True, lstrip_blocks=True)
    assert obj.kwargs == {'trim_blocks': True, 'lstrip_blocks': True}

    # Test ImportError handling
    class FailingEnvironment:
        def __init__(self, extensions=None, **kwargs):
            raise ImportError("No module named 'invalid_ext'")

    class FailingClass(ExtensionLoaderMixin, FailingEnvironment):
        pass

    try:
        FailingClass(context={})
        assert False, "Should have raised UnknownExtension"
    except UnknownExtension as e:
        assert "Unable to load extension: No module named 'invalid_ext'" in str(e)


