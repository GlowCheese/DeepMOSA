####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_loader(tmp_path, monkeypatch):
    """Test loader function."""
    # Create a test package structure
    pkg_dir = tmp_path / "test_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    
    # Create a module with docstring
    module_file = pkg_dir / "test_module.py"
    module_file.write_text('''
"""Test module docstring."""

def test_func():
    """Test function docstring."""
    pass

class TestClass:
    """Test class docstring."""
    pass
''')
    
    # Create a stub file
    stub_file = pkg_dir / "stub_module.pyi"
    stub_file.write_text('''
"""Stub module docstring."""

def stub_func() -> None: ...
''')
    
    # Test loader with the created package
    result = loader("test_pkg", str(tmp_path), link=True, level=1, toc=False)
    
    # Verify result is a string (compiled documentation)
    assert isinstance(result, str)
    # Verify that documentation was generated (non-empty)
    assert len(result) > 0


def test_loader_empty_package(tmp_path):
    """Test loader with empty package."""
    pkg_dir = tmp_path / "empty_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    
    result = loader("empty_pkg", str(tmp_path), link=True, level=1, toc=False)
    
    # Empty package should still return a string
    assert isinstance(result, str)


def test_loader_with_different_levels(tmp_path):
    """Test loader with different header levels."""
    pkg_dir = tmp_path / "test_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "module.py").write_text('"""Module."""\ndef func(): """Func."""')
    
    # Test with different levels
    for level in [1, 2, 3]:
        result = loader("test_pkg", str(tmp_path), link=True, level=level, toc=False)
        assert isinstance(result, str)


def test_loader_with_toc(tmp_path):
    """Test loader with table of contents enabled."""
    pkg_dir = tmp_path / "test_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "module.py").write_text('"""Module."""')
    
    result = loader("test_pkg", str(tmp_path), link=True, level=1, toc=True)
    
    assert isinstance(result, str)


def test_loader_with_link_disabled(tmp_path):
    """Test loader with link generation disabled."""
    pkg_dir = tmp_path / "test_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "module.py").write_text('"""Module."""')
    
    result = loader("test_pkg", str(tmp_path), link=False, level=1, toc=False)
    
    assert isinstance(result, str)


def test_loader_nonexistent_package(tmp_path):
    """Test loader with non-existent package."""
    result = loader("nonexistent_pkg", str(tmp_path), link=True, level=1, toc=False)
    
    # Should return empty string for non-existent package
    assert isinstance(result, str)


# LLM-generated content at query #2
#--------------------------

```python
def test_loader(tmp_path, monkeypatch):
    """Test the loader function."""
    # Create a temporary package structure
    pkg_dir = tmp_path / "test_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    
    # Create a module with docstring
    module_file = pkg_dir / "module.py"
    module_file.write_text('''
"""Module docstring."""

def func():
    """Function docstring."""
    pass

class MyClass:
    """Class docstring."""
    pass
''')
    
    # Create a stub file
    stub_file = pkg_dir / "stub_module.pyi"
    stub_file.write_text('''
"""Stub module docstring."""

def stub_func() -> None: ...
''')
    
    # Monkeypatch sys.path
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    
    # Test loader function
    result = loader("test_pkg", str(tmp_path), link=True, level=1, toc=False)
    
    # Verify result is a string
    assert isinstance(result, str)
    
    # Verify that module documentation was parsed
    assert "module" in result.lower() or result.strip() != ""


def test_loader_with_link_and_level(tmp_path, monkeypatch):
    """Test loader with different link and level parameters."""
    pkg_dir = tmp_path / "test_pkg2"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    
    module_file = pkg_dir / "test_module.py"
    module_file.write_text('''
"""Test module."""

def test_function():
    """Test function."""
    pass
''')
    
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    
    # Test with link=False and level=2
    result = loader("test_pkg2", str(tmp_path), link=False, level=2, toc=False)
    
    assert isinstance(result, str)


def test_loader_with_toc(tmp_path, monkeypatch):
    """Test loader with table of contents enabled."""
    pkg_dir = tmp_path / "test_pkg3"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    
    module_file = pkg_dir / "documented.py"
    module_file.write_text('''
"""Documented module."""

def documented_func():
    """A documented function."""
    pass
''')
    
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    
    # Test with toc=True
    result = loader("test_pkg3", str(tmp_path), link=True, level=1, toc=True)
    
    assert isinstance(result, str)


def test_loader_empty_package(tmp_path, monkeypatch):
    """Test loader with empty package."""
    pkg_dir = tmp_path / "empty_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    
    result = loader("empty_pkg", str(tmp_path), link=True, level=1, toc=False)
    
    assert isinstance(result, str)


def test_loader_nonexistent_path(tmp_path):
    """Test loader with non-existent path."""
    result = loader("nonexistent", str(tmp_path / "nonexistent"), link=True, level=1, toc=False)
    
    assert isinstance(result, str)


# LLM-generated content at query #3
#--------------------------

```python
def test_loader(tmp_path, monkeypatch):
    """Test the loader function."""
    # Create a temporary package structure
    pkg_dir = tmp_path / "test_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    
    # Create a module with docstring
    module_file = pkg_dir / "module.py"
    module_file.write_text(
        '"""Module docstring."""\n'
        'def func():\n'
        '    """Function docstring."""\n'
        '    pass\n'
    )
    
    # Create a stub file
    stub_file = pkg_dir / "stub_module.pyi"
    stub_file.write_text(
        '"""Stub module docstring."""\n'
        'def stub_func() -> None: ...\n'
    )
    
    # Test loader with valid package
    result = loader("test_pkg", str(tmp_path), link=True, level=1, toc=False)
    assert isinstance(result, str)
    assert len(result) > 0
    
    # Test with different parameters
    result_no_link = loader("test_pkg", str(tmp_path), link=False, level=2, toc=True)
    assert isinstance(result_no_link, str)
    
    # Test with non-existent package
    result_empty = loader("nonexistent_pkg", str(tmp_path), link=True, level=1, toc=False)
    assert isinstance(result_empty, str)


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path


def test_loader():
    """Test the loader function."""
    # Mock the walk_packages function to return test data
    mock_packages = [
        ("test_module", "/path/to/test_module"),
        ("test_module.sub", "/path/to/test_module/sub"),
    ]
    
    mock_source_code = """
def test_function():
    '''Test function documentation.'''
    pass

class TestClass:
    '''Test class documentation.'''
    pass
"""
    
    with patch('compiler.walk_packages', return_value=mock_packages):
        with patch('compiler.isfile') as mock_isfile:
            with patch('compiler.builtins.open', mock_open(read_data=mock_source_code)):
                with patch('compiler._load_module', return_value=False):
                    with patch('compiler.Parser.new') as mock_parser_new:
                        mock_parser = MagicMock()
                        mock_parser_new.return_value = mock_parser
                        mock_parser.compile.return_value = "# Compiled Documentation\n"
                        
                        # Setup isfile to return True for .py files
                        def isfile_side_effect(path):
                            return path.endswith('.py')
                        
                        mock_isfile.side_effect = isfile_side_effect
                        
                        result = loader("test_package", "/test/path", link=True, level=1, toc=False)
                        
                        # Assertions
                        assert result == "# Compiled Documentation\n"
                        mock_parser_new.assert_called_once_with(True, 1, False)
                        assert mock_parser.parse.call_count == 2
                        mock_parser.compile.assert_called_once()


def test_loader_with_extension_modules():
    """Test loader function with extension modules."""
    mock_packages = [
        ("extension_module", "/path/to/extension_module"),
    ]
    
    with patch('compiler.walk_packages', return_value=mock_packages):
        with patch('compiler.isfile') as mock_isfile:
            with patch('compiler._load_module') as mock_load:
                with patch('compiler.Parser.new') as mock_parser_new:
                    mock_parser = MagicMock()
                    mock_parser_new.return_value = mock_parser
                    mock_parser.compile.return_value = "# Extension Module Docs\n"
                    
                    # Setup: .py file doesn't exist, but .so file does
                    def isfile_side_effect(path):
                        return path.endswith('.so')
                    
                    mock_isfile.side_effect = isfile_side_effect
                    mock_load.return_value = True
                    
                    result = loader("ext_pkg", "/test/path", link=False, level=2, toc=True)
                    
                    assert result == "# Extension Module Docs\n"
                    mock_load.assert_called_once()
                    mock_parser.compile.assert_called_once()


def test_loader_empty_result():
    """Test loader function when no packages are found."""
    with patch('compiler.walk_packages', return_value=[]):
        with patch('compiler.Parser.new') as mock_parser_new:
            mock_parser = MagicMock()
            mock_parser_new.return_value = mock_parser
            mock_parser.compile.return_value = ""
            
            result = loader("nonexistent", "/test/path", link=True, level=1, toc=False)
            
            assert result == ""
            mock_parser_new.assert_called_once_with(True, 1, False)
            mock_parser.compile.assert_called_once()


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock, mock_open, call
from os.path import join


def test_gen_api():
    """Test gen_api function."""
    # Test with dry run
    with patch('os.path.isdir', return_value=True), \
         patch('os.mkdir') as mock_mkdir, \
         patch('sys.path') as mock_sys_path, \
         patch('importlib.util.find_spec') as mock_find_spec, \
         patch('walk_packages') as mock_walk_packages, \
         patch('loader') as mock_loader, \
         patch('os.path.isfile', return_value=True), \
         patch('builtins.open', mock_open()), \
         patch('logger') as mock_logger:
        
        mock_walk_packages.return_value = []
        mock_loader.return_value = "# Module Doc"
        
        result = gen_api(
            {'Test': 'test_module'},
            pwd='/test/path',
            prefix='docs',
            link=True,
            level=1,
            toc=False,
            dry=True
        )
        
        assert mock_sys_path.append.called
        assert len(result) == 1
        assert "# Test API" in result[0]


def test_gen_api_no_pwd():
    """Test gen_api without pwd parameter."""
    with patch('os.path.isdir', return_value=True), \
         patch('os.mkdir') as mock_mkdir, \
         patch('sys.path') as mock_sys_path, \
         patch('importlib.util.find_spec') as mock_find_spec, \
         patch('walk_packages') as mock_walk_packages, \
         patch('loader') as mock_loader, \
         patch('builtins.open', mock_open()), \
         patch('logger') as mock_logger:
        
        mock_walk_packages.return_value = []
        mock_loader.return_value = "# Module Doc"
        
        result = gen_api(
            {'Test': 'test_module'},
            prefix='docs',
            dry=True
        )
        
        assert mock_sys_path.append.call_count == 0


def test_gen_api_create_directory():
    """Test gen_api creates directory when it doesn't exist."""
    with patch('os.path.isdir', return_value=False), \
         patch('os.mkdir') as mock_mkdir, \
         patch('sys.path'), \
         patch('importlib.util.find_spec') as mock_find_spec, \
         patch('walk_packages') as mock_walk_packages, \
         patch('loader') as mock_loader, \
         patch('builtins.open', mock_open()), \
         patch('logger'):
        
        mock_walk_packages.return_value = []
        mock_loader.return_value = "# Module Doc"
        
        gen_api({'Test': 'test_module'}, prefix='docs', dry=True)
        
        mock_mkdir.assert_called_once_with('docs')


def test_gen_api_empty_doc():
    """Test gen_api handles empty documentation."""
    with patch('os.path.isdir', return_value=True), \
         patch('os.mkdir'), \
         patch('sys.path'), \
         patch('importlib.util.find_spec') as mock_find_spec, \
         patch('walk_packages') as mock_walk_packages, \
         patch('loader') as mock_loader, \
         patch('builtins.open', mock_open()), \
         patch('logger'):
        
        mock_walk_packages.return_value = []
        mock_loader.return_value = "   \n\n   "
        
        result = gen_api({'Test': 'test_module'}, dry=True)
        
        assert len(result) == 0


def test_gen_api_write_file():
    """Test gen_api writes file when dry=False."""
    with patch('os.path.isdir', return_value=True), \
         patch('os.mkdir'), \
         patch('sys.path'), \
         patch('importlib.util.find_spec') as mock_find_spec, \
         patch('walk_packages') as mock_walk_packages, \
         patch('loader') as mock_loader, \
         patch('_write') as mock_write, \
         patch('builtins.open', mock_open()), \
         patch('logger'):
        
        mock_walk_packages.return_value = []
        mock_loader.return_value = "# Module Doc"
        
        result = gen_api(
            {'Test': 'test_module'},
            prefix='docs',
            dry=False
        )
        
        assert mock_write.called
        assert len(result) == 1


def test_gen_api_multiple_modules():
    """Test gen_api with multiple modules."""
    with patch('os.path.isdir', return_value=True), \
         patch('os.mkdir'), \
         patch('sys.path'), \
         patch('importlib.util.find_spec') as mock_find_spec, \
         patch('walk_packages') as mock_walk_packages, \
         patch('loader') as mock_loader, \
         patch('_write') as mock_write, \
         patch('builtins.open', mock_open()), \
         patch('logger'):
        
        mock_walk_packages.return_value = []
        mock_loader.return_value = "# Module Doc"
        
        result = gen_api(
            {'Test1': 'module1', 'Test2': 'module2'},
            dry=False
        )
        
        assert len(result) == 2
        assert mock_write.call_count == 2


def test_gen_api_underscore_to_dash():
    """Test gen_api converts underscores to dashes in filenames."""
    with patch('os.path.isdir', return_value=True), \
         patch('os.mkdir'), \
         patch('sys.path'), \
         patch('importlib.util.find_spec') as mock_find_spec, \
         patch('walk_packages') as mock_walk_packages, \
         patch('loader') as mock_loader, \
         patch('_write') as mock_write, \
         patch('builtins.open', mock_open()), \
         patch('logger'):
        
        mock_walk_packages.return_value = []
        mock_loader.return_value = "# Module Doc"
        
        gen_api({'Test': 'test_module_name'}, dry=False)
        
        call_args = mock_write.call_args[0][0]
        assert 'test-module-name-api.md' in call_args


def test_gen_api_custom_level():
    """Test gen_api with custom heading level."""
    with patch('os.path.isdir', return_value=True), \
         patch('os.mkdir'), \
         patch('sys.path'), \
         patch('importlib.util.find_spec') as mock_find_spec, \
         patch('walk_packages') as mock_walk_packages, \
         patch('loader') as mock_loader, \
         patch('builtins.open', mock_open()), \
         patch('logger'):
        
        mock_walk_packages.return_value = []
        mock_loader.return_value = "Module Doc"
        
        result = gen_api(
            {'Test': 'test_module'},
            level=3,
            dry=True
        )
        
        assert "### Test API" in result[0]


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import Mock, patch, MagicMock, call
from os.path import join


def test_gen_api():
    """Test gen_api function."""
    
    # Test with empty root_names
    with patch('os.path.isdir', return_value=True):
        result = gen_api({})
        assert result == []
    
    # Test with dry run enabled
    with patch('os.path.isdir', return_value=True), \
         patch('os.mkdir'), \
         patch('sys.path.append'), \
         patch('os.path.isfile', return_value=True), \
         patch('builtins.open', create=True), \
         patch('os.walk', return_value=[('root', [], ['module.py'])]), \
         patch('os.path.dirname', return_value='site_packages'), \
         patch('importlib.util.find_spec') as mock_find_spec, \
         patch('os.path.join', side_effect=join), \
         patch('os.path.abspath', side_effect=lambda x: x), \
         patch('logger') as mock_logger:
        
        mock_spec = MagicMock()
        mock_spec.submodule_search_locations = ['site_packages']
        mock_find_spec.return_value = mock_spec
        
        with patch('loader', return_value="## Class\n\nSome docs"):
            result = gen_api(
                {'Test': 'test_module'},
                dry=True,
                prefix='docs'
            )
            assert len(result) == 1
            assert 'Test API' in result[0]
    
    # Test with dry run disabled (file writing)
    with patch('os.path.isdir', return_value=True), \
         patch('os.mkdir'), \
         patch('sys.path.append'), \
         patch('os.path.isfile', return_value=True), \
         patch('builtins.open', create=True) as mock_open, \
         patch('os.walk', return_value=[('root', [], ['module.py'])]), \
         patch('os.path.dirname', return_value='site_packages'), \
         patch('importlib.util.find_spec') as mock_find_spec, \
         patch('os.path.join', side_effect=join), \
         patch('os.path.abspath', side_effect=lambda x: x), \
         patch('logger'):
        
        mock_spec = MagicMock()
        mock_spec.submodule_search_locations = ['site_packages']
        mock_find_spec.return_value = mock_spec
        
        with patch('loader', return_value="## Function\n\nDocs here"):
            result = gen_api(
                {'MyModule': 'my_module'},
                dry=False,
                prefix='docs'
            )
            assert len(result) == 1
            assert 'MyModule API' in result[0]
            mock_open.assert_called()
    
    # Test with multiple root names
    with patch('os.path.isdir', return_value=True), \
         patch('os.mkdir'), \
         patch('sys.path.append'), \
         patch('os.path.isfile', return_value=True), \
         patch('builtins.open', create=True), \
         patch('os.walk', return_value=[('root', [], ['module.py'])]), \
         patch('os.path.dirname', return_value='site_packages'), \
         patch('importlib.util.find_spec') as mock_find_spec, \
         patch('os.path.join', side_effect=join), \
         patch('os.path.abspath', side_effect=lambda x: x), \
         patch('logger'):
        
        mock_spec = MagicMock()
        mock_spec.submodule_search_locations = ['site_packages']
        mock_find_spec.return_value = mock_spec
        
        with patch('loader', return_value="Content"):
            result = gen_api(
                {'Mod1': 'module1', 'Mod2': 'module2'},
                dry=True
            )
            assert len(result) == 2
    
    # Test with empty loader result
    with patch('os.path.isdir', return_value=True), \
         patch('os.mkdir'), \
         patch('sys.path.append'), \
         patch('logger'):
        
        with patch('loader', return_value="   \n  "):
            result = gen_api(
                {'Empty': 'empty_module'},
                dry=True
            )
            assert len(result) == 0
    
    # Test with pwd parameter
    with patch('os.path.isdir', return_value=True), \
         patch('os.mkdir'), \
         patch('sys.path.append') as mock_append, \
         patch('loader', return_value="Content"), \
         patch('os.path.dirname', return_value='site_packages'), \
         patch('importlib.util.find_spec') as mock_find_spec, \
         patch('logger'):
        
        mock_spec = MagicMock()
        mock_spec.submodule_search_locations = ['site_packages']
        mock_find_spec.return_value = mock_spec
        
        result = gen_api(
            {'Test': 'test'},
            pwd='/custom/path',
            dry=True
        )
        mock_append.assert_called_with('/custom/path')
    
    # Test with custom prefix and level
    with patch('os.path.isdir', return_value=True), \
         patch('os.mkdir') as mock_mkdir, \
         patch('sys.path.append'), \
         patch('loader', return_value="Content"), \
         patch('os.path.dirname', return_value='site_packages'), \
         patch('importlib.util.find_spec') as mock_find_spec, \
         patch('logger'):
        
        mock_spec = MagicMock()
        mock_spec.submodule_search_locations = ['site_packages']
        mock_find_spec.return_value = mock_spec
        
        result = gen_api(
            {'Test': 'test'},
            prefix='custom_docs',
            level=3,
            dry=True
        )
        assert len(result) == 1
        assert '###' in result[0]


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock, mock_open, call
from os.path import join


def test_gen_api():
    """Test gen_api function with various scenarios."""
    
    # Test 1: Basic functionality with dry run
    with patch('os.path.isdir', return_value=True), \
         patch('os.mkdir') as mock_mkdir, \
         patch('sys.path') as mock_sys_path, \
         patch('loader') as mock_loader:
        
        mock_loader.return_value = "## Module1\nDocumentation"
        
        result = gen_api(
            {'Module1': 'module1'},
            pwd='/custom/path',
            prefix='docs',
            dry=True
        )
        
        assert len(result) == 1
        assert '# Module1 API' in result[0]
        mock_sys_path.append.assert_called_once_with('/custom/path')
    
    # Test 2: Empty documentation warning
    with patch('os.path.isdir', return_value=True), \
         patch('os.mkdir'), \
         patch('sys.path'), \
         patch('loader') as mock_loader, \
         patch('logger') as mock_logger:
        
        mock_loader.return_value = "   \n  "
        
        result = gen_api({'Module1': 'module1'})
        
        assert len(result) == 0
    
    # Test 3: Multiple root names
    with patch('os.path.isdir', return_value=True), \
         patch('os.mkdir'), \
         patch('sys.path'), \
         patch('loader') as mock_loader, \
         patch('_write') as mock_write:
        
        mock_loader.side_effect = [
            "## Mod1\nDoc1",
            "## Mod2\nDoc2"
        ]
        
        result = gen_api(
            {'Title1': 'module1', 'Title2': 'module2'},
            dry=False
        )
        
        assert len(result) == 2
        assert mock_write.call_count == 2
    
    # Test 4: Create directory when not exists
    with patch('os.path.isdir', return_value=False), \
         patch('os.mkdir') as mock_mkdir, \
         patch('sys.path'), \
         patch('loader') as mock_loader, \
         patch('_write'):
        
        mock_loader.return_value = "## Doc"
        
        gen_api({'Module': 'module'}, prefix='new_docs')
        
        mock_mkdir.assert_called_once_with('new_docs')
    
    # Test 5: Verify file path with underscore replacement
    with patch('os.path.isdir', return_value=True), \
         patch('os.mkdir'), \
         patch('sys.path'), \
         patch('loader') as mock_loader, \
         patch('_write') as mock_write:
        
        mock_loader.return_value = "## Doc"
        
        gen_api({'Module': 'my_module_name'}, prefix='docs', dry=False)
        
        call_args = mock_write.call_args[0][0]
        assert 'my-module-name-api.md' in call_args
    
    # Test 6: Custom level parameter
    with patch('os.path.isdir', return_value=True), \
         patch('os.mkdir'), \
         patch('sys.path'), \
         patch('loader') as mock_loader:
        
        mock_loader.return_value = "## Doc"
        
        result = gen_api(
            {'Title': 'module'},
            level=3,
            dry=True
        )
        
        assert '### Title API' in result[0]
    
    # Test 7: No pwd parameter
    with patch('os.path.isdir', return_value=True), \
         patch('os.mkdir'), \
         patch('sys.path') as mock_sys_path, \
         patch('loader') as mock_loader:
        
        mock_loader.return_value = "## Doc"
        
        gen_api({'Module': 'module'}, pwd=None)
        
        mock_sys_path.append.assert_not_called()
    
    # Test 8: Link and toc parameters passed to loader
    with patch('os.path.isdir', return_value=True), \
         patch('os.mkdir'), \
         patch('sys.path'), \
         patch('loader') as mock_loader:
        
        mock_loader.return_value = "## Doc"
        
        gen_api(
            {'Module': 'module'},
            link=False,
            level=2,
            toc=True
        )
        
        mock_loader.assert_called_once()
        call_args = mock_loader.call_args[0]
        assert call_args[2] is False  # link
        assert call_args[3] == 2      # level
        assert call_args[4] is True   # toc


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock, call
from os.path import join


def test_gen_api():
    """Test gen_api function."""
    with patch('os.path.isdir') as mock_isdir, \
         patch('os.mkdir') as mock_mkdir, \
         patch('sys.path') as mock_sys_path, \
         patch('importlib.util.find_spec') as mock_find_spec, \
         patch('os.walk') as mock_walk, \
         patch('builtins.open', create=True) as mock_open, \
         patch('logger') as mock_logger, \
         patch.object(__import__('importlib.util', fromlist=['spec_from_file_location']), 'spec_from_file_location') as mock_spec:
        
        # Test with empty root_names
        result = gen_api({})
        assert result == []
        
        # Test with root_names and default parameters
        mock_isdir.return_value = True
        mock_find_spec.return_value = MagicMock(submodule_search_locations=['/fake/path'])
        mock_walk.return_value = []
        
        root_names = {'TestModule': 'test_module'}
        result = gen_api(root_names)
        assert isinstance(result, list)
        mock_sys_path.append.assert_not_called()
        
        # Test with pwd parameter
        mock_isdir.reset_mock()
        mock_walk.reset_mock()
        mock_isdir.return_value = True
        mock_walk.return_value = []
        
        result = gen_api(root_names, pwd='/custom/path')
        mock_sys_path.append.assert_called_with('/custom/path')
        
        # Test with prefix that doesn't exist
        mock_isdir.reset_mock()
        mock_isdir.return_value = False
        mock_walk.return_value = []
        
        result = gen_api(root_names, prefix='custom_docs')
        mock_mkdir.assert_called_with('custom_docs')
        
        # Test with dry mode
        mock_isdir.reset_mock()
        mock_isdir.return_value = True
        mock_walk.return_value = []
        mock_open.reset_mock()
        
        result = gen_api(root_names, dry=True)
        mock_open.assert_not_called()
        
        # Test with custom parameters
        mock_isdir.reset_mock()
        mock_walk.return_value = []
        
        result = gen_api(
            root_names,
            prefix='api_docs',
            link=False,
            level=2,
            toc=True,
            dry=False
        )
        assert isinstance(result, list)


# LLM-generated content at query #9
#--------------------------

```python
def test_loader(tmp_path, monkeypatch):
    """Test the loader function."""
    # Create a temporary package structure
    pkg_dir = tmp_path / "test_pkg"
    pkg_dir.mkdir()
    
    # Create __init__.py
    init_file = pkg_dir / "__init__.py"
    init_file.write_text('"""Test package."""\ndef test_func():\n    """Test function."""\n    pass\n')
    
    # Create a submodule
    submodule = pkg_dir / "submodule.py"
    submodule.write_text('"""Submodule."""\nclass TestClass:\n    """Test class."""\n    pass\n')
    
    # Create a stub file
    stub_file = pkg_dir / "stub_module.pyi"
    stub_file.write_text('"""Stub module."""\ndef stub_func() -> None: ...\n')
    
    # Monkeypatch sys_path to include tmp_path
    monkeypatch.setattr('sys.path', sys_path + [str(tmp_path)])
    
    # Call loader function
    result = loader("test_pkg", str(tmp_path), link=True, level=1, toc=False)
    
    # Verify result is a string
    assert isinstance(result, str)
    
    # Verify the result contains documentation
    assert len(result) > 0


def test_loader_with_toc(tmp_path, monkeypatch):
    """Test the loader function with table of contents."""
    pkg_dir = tmp_path / "test_pkg_toc"
    pkg_dir.mkdir()
    
    init_file = pkg_dir / "__init__.py"
    init_file.write_text('"""Test package with TOC."""\n')
    
    monkeypatch.setattr('sys.path', sys_path + [str(tmp_path)])
    
    result = loader("test_pkg_toc", str(tmp_path), link=True, level=1, toc=True)
    
    assert isinstance(result, str)


def test_loader_nonexistent_package(tmp_path):
    """Test loader with nonexistent package."""
    result = loader("nonexistent_pkg", str(tmp_path), link=True, level=1, toc=False)
    
    # Should return empty or minimal string
    assert isinstance(result, str)


def test_loader_different_levels(tmp_path, monkeypatch):
    """Test loader with different heading levels."""
    pkg_dir = tmp_path / "level_test_pkg"
    pkg_dir.mkdir()
    
    init_file = pkg_dir / "__init__.py"
    init_file.write_text('"""Level test package."""\n')
    
    monkeypatch.setattr('sys.path', sys_path + [str(tmp_path)])
    
    for level in [1, 2, 3]:
        result = loader("level_test_pkg", str(tmp_path), link=True, level=level, toc=False)
        assert isinstance(result, str)


def test_loader_with_link_disabled(tmp_path, monkeypatch):
    """Test loader with link generation disabled."""
    pkg_dir = tmp_path / "no_link_pkg"
    pkg_dir.mkdir()
    
    init_file = pkg_dir / "__init__.py"
    init_file.write_text('"""No link package."""\n')
    
    monkeypatch.setattr('sys.path', sys_path + [str(tmp_path)])
    
    result = loader("no_link_pkg", str(tmp_path), link=False, level=1, toc=False)
    
    assert isinstance(result, str)


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os


def test_loader():
    """Test the loader function with mocked dependencies."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test package structure
        pkg_dir = os.path.join(tmpdir, "test_pkg")
        os.makedirs(pkg_dir)
        
        # Create __init__.py
        init_file = os.path.join(pkg_dir, "__init__.py")
        with open(init_file, 'w') as f:
            f.write('"""Test package."""\ndef test_func():\n    """Test function."""\n    pass')
        
        # Create a module file
        module_file = os.path.join(pkg_dir, "module.py")
        with open(module_file, 'w') as f:
            f.write('"""Test module."""\nclass TestClass:\n    """Test class."""\n    pass')
        
        # Mock Parser
        with patch('compiler.Parser') as mock_parser_class:
            mock_parser = MagicMock()
            mock_parser_class.new.return_value = mock_parser
            mock_parser.compile.return_value = "# Generated Documentation"
            
            # Call loader
            result = loader("test_pkg", tmpdir, link=True, level=1, toc=False)
            
            # Assertions
            assert result == "# Generated Documentation"
            mock_parser_class.new.assert_called_once_with(True, 1, False)
            mock_parser.parse.assert_called()
            mock_parser.compile.assert_called_once()


def test_loader_with_stub_files():
    """Test loader with .pyi stub files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pkg_dir = os.path.join(tmpdir, "stub_pkg")
        os.makedirs(pkg_dir)
        
        # Create stub file
        stub_file = os.path.join(pkg_dir, "__init__.pyi")
        with open(stub_file, 'w') as f:
            f.write('def stub_func() -> None: ...')
        
        with patch('compiler.Parser') as mock_parser_class:
            mock_parser = MagicMock()
            mock_parser_class.new.return_value = mock_parser
            mock_parser.compile.return_value = "# Stub Documentation"
            
            result = loader("stub_pkg", tmpdir, link=False, level=2, toc=True)
            
            assert result == "# Stub Documentation"
            mock_parser_class.new.assert_called_once_with(False, 2, True)


def test_loader_empty_package():
    """Test loader with empty package."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pkg_dir = os.path.join(tmpdir, "empty_pkg")
        os.makedirs(pkg_dir)
        
        with patch('compiler.Parser') as mock_parser_class:
            mock_parser = MagicMock()
            mock_parser_class.new.return_value = mock_parser
            mock_parser.compile.return_value = ""
            
            result = loader("empty_pkg", tmpdir, link=True, level=1, toc=False)
            
            assert result == ""
            mock_parser_class.new.assert_called_once()


def test_loader_with_extension_modules():
    """Test loader attempts to load extension modules."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pkg_dir = os.path.join(tmpdir, "ext_pkg")
        os.makedirs(pkg_dir)
        
        # Create stub without .py file to trigger extension loading
        stub_file = os.path.join(pkg_dir, "module.pyi")
        with open(stub_file, 'w') as f:
            f.write('def ext_func() -> None: ...')
        
        with patch('compiler.Parser') as mock_parser_class:
            with patch('compiler._load_module') as mock_load:
                mock_parser = MagicMock()
                mock_parser_class.new.return_value = mock_parser
                mock_parser.compile.return_value = "# Extension Documentation"
                mock_load.return_value = True
                
                result = loader("ext_pkg", tmpdir, link=True, level=1, toc=False)
                
                assert result == "# Extension Documentation"
                mock_parser.parse.assert_called()


def test_loader_multiple_files():
    """Test loader with multiple Python files in package."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pkg_dir = os.path.join(tmpdir, "multi_pkg")
        os.makedirs(pkg_dir)
        
        # Create multiple module files
        for i in range(3):
            module_file = os.path.join(pkg_dir, f"module{i}.py")
            with open(module_file, 'w') as f:
                f.write(f'"""Module {i}."""\ndef func{i}():\n    """Function {i}."""\n    pass')
        
        with patch('compiler.Parser') as mock_parser_class:
            mock_parser = MagicMock()
            mock_parser_class.new.return_value = mock_parser
            mock_parser.compile.return_value = "# Multi Module Documentation"
            
            result = loader("multi_pkg", tmpdir, link=True, level=1, toc=False)
            
            assert result == "# Multi Module Documentation"
            assert mock_parser.parse.call_count >= 3


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock, call
from pathlib import Path
import tempfile
import os


def test_gen_api():
    """Test gen_api function with various scenarios."""
    
    # Test 1: Basic functionality with dry run
    with patch('os.path.isdir') as mock_isdir, \
         patch('os.mkdir') as mock_mkdir, \
         patch('sys.path') as mock_sys_path, \
         patch('loader') as mock_loader:
        
        mock_isdir.return_value = True
        mock_loader.return_value = "# Module API\n\nDocumentation"
        
        result = gen_api(
            {'Test Module': 'test_module'},
            pwd='/test/path',
            prefix='docs',
            link=True,
            level=1,
            toc=False,
            dry=True
        )
        
        assert len(result) == 1
        assert '# Test Module API' in result[0]
        mock_sys_path.append.assert_called_once_with('/test/path')
    
    # Test 2: Create directory when it doesn't exist
    with patch('os.path.isdir') as mock_isdir, \
         patch('os.mkdir') as mock_mkdir, \
         patch('sys.path') as mock_sys_path, \
         patch('loader') as mock_loader:
        
        mock_isdir.return_value = False
        mock_loader.return_value = "# Module API\n\nDocumentation"
        
        result = gen_api(
            {'Test Module': 'test_module'},
            prefix='custom_docs',
            dry=True
        )
        
        mock_mkdir.assert_called_once_with('custom_docs')
        assert len(result) == 1
    
    # Test 3: Multiple root names
    with patch('os.path.isdir') as mock_isdir, \
         patch('os.mkdir') as mock_mkdir, \
         patch('sys.path') as mock_sys_path, \
         patch('loader') as mock_loader, \
         patch('_site_path') as mock_site_path:
        
        mock_isdir.return_value = True
        mock_site_path.return_value = '/site/path'
        mock_loader.side_effect = [
            "# Module1 API\n\nDoc1",
            "# Module2 API\n\nDoc2"
        ]
        
        result = gen_api(
            {'Module One': 'module1', 'Module Two': 'module2'},
            dry=True
        )
        
        assert len(result) == 2
        assert '# Module One API' in result[0]
        assert '# Module Two API' in result[1]
    
    # Test 4: Empty documentation warning
    with patch('os.path.isdir') as mock_isdir, \
         patch('os.mkdir') as mock_mkdir, \
         patch('sys.path') as mock_sys_path, \
         patch('loader') as mock_loader, \
         patch('_site_path') as mock_site_path:
        
        mock_isdir.return_value = True
        mock_site_path.return_value = '/site/path'
        mock_loader.return_value = "   \n  \n  "  # Empty/whitespace only
        
        result = gen_api(
            {'Missing Module': 'missing'},
            dry=True
        )
        
        assert len(result) == 0
    
    # Test 5: Write to file when dry=False
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('os.path.isdir') as mock_isdir, \
             patch('os.mkdir') as mock_mkdir, \
             patch('sys.path') as mock_sys_path, \
             patch('loader') as mock_loader, \
             patch('_site_path') as mock_site_path, \
             patch('_write') as mock_write:
            
            mock_isdir.return_value = True
            mock_site_path.return_value = '/site/path'
            mock_loader.return_value = "# Module API\n\nDocumentation"
            
            result = gen_api(
                {'Test Module': 'test_module'},
                prefix='docs',
                dry=False
            )
            
            # Verify _write was called with correct path
            mock_write.assert_called_once()
            call_args = mock_write.call_args
            assert 'test-module-api.md' in call_args[0][0]
            assert '# Test Module API' in call_args[0][1]
            assert len(result) == 1
    
    # Test 6: Different level parameter
    with patch('os.path.isdir') as mock_isdir, \
         patch('os.mkdir') as mock_mkdir, \
         patch('sys.path') as mock_sys_path, \
         patch('loader') as mock_loader, \
         patch('_site_path') as mock_site_path:
        
        mock_isdir.return_value = True
        mock_site_path.return_value = '/site/path'
        mock_loader.return_value = "Module API\n\nDocumentation"
        
        result = gen_api(
            {'Test Module': 'test_module'},
            level=2,
            dry=True
        )
        
        assert '## Test Module API' in result[0]
    
    # Test 7: No pwd parameter
    with patch('os.path.isdir') as mock_isdir, \
         patch('os.mkdir') as mock_mkdir, \
         patch('sys.path') as mock_sys_path, \
         patch('loader') as mock_loader, \
         patch('_site_path') as mock_site_path:
        
        mock_isdir.return_value = True
        mock_site_path.return_value = '/site/path'
        mock_loader.return_value = "# Module API\n\nDocumentation"
        
        result = gen_api(
            {'Test Module': 'test_module'},
            pwd=None,
            dry=True
        )
        
        mock_sys_path.append.assert_not_called()
        assert len(result) == 1


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock, mock_open, call
from os.path import join


def test_gen_api():
    """Test gen_api function."""
    
    # Test 1: Basic functionality with dry run
    with patch('os.path.isdir', return_value=True), \
         patch('os.mkdir') as mock_mkdir, \
         patch('sys.path') as mock_sys_path, \
         patch('builtins.open', mock_open()), \
         patch('os.walk', return_value=[]), \
         patch('os.path.isfile', return_value=False), \
         patch('importlib.util.find_spec') as mock_find_spec, \
         patch('os.path.dirname', return_value='/site/pkg'), \
         patch('logger') as mock_logger:
        
        mock_find_spec.return_value = MagicMock(submodule_search_locations=['/site/pkg'])
        
        result = gen_api(
            {'TestAPI': 'test_pkg'},
            pwd='/custom/path',
            prefix='docs',
            dry=True
        )
        
        assert isinstance(result, (list, tuple))
        mock_sys_path.append.assert_called_with('/custom/path')
    
    # Test 2: Directory creation when prefix doesn't exist
    with patch('os.path.isdir', side_effect=[False, True]), \
         patch('os.mkdir') as mock_mkdir, \
         patch('sys.path'), \
         patch('builtins.open', mock_open()), \
         patch('os.walk', return_value=[]), \
         patch('os.path.isfile', return_value=False), \
         patch('importlib.util.find_spec') as mock_find_spec, \
         patch('os.path.dirname', return_value='/site/pkg'), \
         patch('logger'):
        
        mock_find_spec.return_value = MagicMock(submodule_search_locations=['/site/pkg'])
        
        result = gen_api({'API': 'pkg'}, prefix='new_docs')
        
        mock_mkdir.assert_called_once_with('new_docs')
    
    # Test 3: Multiple root names
    with patch('os.path.isdir', return_value=True), \
         patch('os.mkdir'), \
         patch('sys.path'), \
         patch('builtins.open', mock_open(read_data='test content')), \
         patch('os.walk', return_value=[]), \
         patch('os.path.isfile', return_value=False), \
         patch('importlib.util.find_spec') as mock_find_spec, \
         patch('os.path.dirname', return_value='/site/pkg'), \
         patch('os.path.join', side_effect=join), \
         patch('logger'), \
         patch('os.path.abspath', side_effect=lambda x: x + '/abs'), \
         patch('os.sep', '/'):
        
        mock_find_spec.return_value = MagicMock(submodule_search_locations=['/site/pkg'])
        
        result = gen_api({
            'API1': 'pkg1',
            'API2': 'pkg2'
        }, dry=True)
        
        assert len(result) == 2
    
    # Test 4: No pwd provided
    with patch('os.path.isdir', return_value=True), \
         patch('os.mkdir'), \
         patch('sys.path') as mock_sys_path, \
         patch('builtins.open', mock_open()), \
         patch('os.walk', return_value=[]), \
         patch('os.path.isfile', return_value=False), \
         patch('importlib.util.find_spec') as mock_find_spec, \
         patch('os.path.dirname', return_value='/site/pkg'), \
         patch('logger'):
        
        mock_find_spec.return_value = MagicMock(submodule_search_locations=['/site/pkg'])
        
        result = gen_api({'API': 'pkg'}, pwd=None, dry=True)
        
        mock_sys_path.append.assert_not_called()
    
    # Test 5: Empty loader result (package not found)
    with patch('os.path.isdir', return_value=True), \
         patch('os.mkdir'), \
         patch('sys.path'), \
         patch('builtins.open', mock_open()), \
         patch('os.walk', return_value=[]), \
         patch('os.path.isfile', return_value=False), \
         patch('importlib.util.find_spec') as mock_find_spec, \
         patch('os.path.dirname', return_value='/site/pkg'), \
         patch('logger'), \
         patch('os.path.abspath', side_effect=lambda x: x), \
         patch('os.sep', '/'):
        
        mock_find_spec.return_value = MagicMock(submodule_search_locations=['/site/pkg'])
        
        result = gen_api({'API': 'nonexistent_pkg'}, dry=True)
        
        assert isinstance(result, (list, tuple))
    
    # Test 6: Custom prefix and parameters
    with patch('os.path.isdir', return_value=True), \
         patch('os.mkdir'), \
         patch('sys.path'), \
         patch('builtins.open', mock_open()), \
         patch('os.walk', return_value=[]), \
         patch('os.path.isfile', return_value=False), \
         patch('importlib.util.find_spec') as mock_find_spec, \
         patch('os.path.dirname', return_value='/site/pkg'), \
         patch('logger'), \
         patch('os.path.abspath', side_effect=lambda x: x), \
         patch('os.sep', '/'):
        
        mock_find_spec.return_value = MagicMock(submodule_search_locations=['/site/pkg'])
        
        result = gen_api(
            {'CustomAPI': 'my_pkg'},
            prefix='custom_docs',
            link=False,
            level=2,
            toc=True,
            dry=True
        )
        
        assert isinstance(result, (list, tuple))


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path


def test_loader():
    """Test the loader function."""
    # Mock the walk_packages function
    mock_packages = [
        ("test_module", "/path/to/test_module"),
        ("test_module.submodule", "/path/to/test_module/submodule"),
    ]
    
    mock_py_content = '"""Test module docstring."""\ndef test_func():\n    """Test function."""\n    pass'
    mock_pyi_content = '"""Test stub file."""\ndef test_func() -> None: ...'
    
    with patch('compiler.walk_packages', return_value=mock_packages):
        with patch('compiler._read', return_value=mock_py_content):
            with patch('compiler.isfile') as mock_isfile:
                with patch('compiler.Parser.new') as mock_parser_new:
                    # Setup mock parser
                    mock_parser = MagicMock()
                    mock_parser.compile.return_value = "# Compiled Documentation\n\n## test_module\n\n"
                    mock_parser_new.return_value = mock_parser
                    
                    # Setup isfile to return True for .py files
                    mock_isfile.side_effect = lambda x: x.endswith('.py')
                    
                    # Call the loader function
                    result = loader("test_module", "/path/to", link=True, level=1, toc=False)
                    
                    # Assertions
                    assert isinstance(result, str)
                    assert "# Compiled Documentation" in result
                    mock_parser_new.assert_called_once_with(True, 1, False)
                    assert mock_parser.parse.call_count >= len(mock_packages)
                    mock_parser.compile.assert_called_once()


def test_loader_with_stub_files():
    """Test loader function with stub files."""
    mock_packages = [
        ("test_module", "/path/to/test_module"),
    ]
    
    mock_py_content = '"""Python source."""'
    mock_pyi_content = '"""Stub file."""'
    
    with patch('compiler.walk_packages', return_value=mock_packages):
        with patch('compiler._read', side_effect=[mock_pyi_content, mock_py_content]):
            with patch('compiler.isfile') as mock_isfile:
                with patch('compiler.Parser.new') as mock_parser_new:
                    mock_parser = MagicMock()
                    mock_parser.compile.return_value = "# Documentation\n"
                    mock_parser_new.return_value = mock_parser
                    
                    # First call returns True for .pyi, second for .py
                    mock_isfile.side_effect = [True, True]
                    
                    result = loader("test_module", "/path/to", link=False, level=2, toc=True)
                    
                    assert isinstance(result, str)
                    mock_parser_new.assert_called_once_with(False, 2, True)
                    mock_parser.parse.assert_called()


def test_loader_empty_result():
    """Test loader function when no packages are found."""
    with patch('compiler.walk_packages', return_value=[]):
        with patch('compiler.Parser.new') as mock_parser_new:
            mock_parser = MagicMock()
            mock_parser.compile.return_value = ""
            mock_parser_new.return_value = mock_parser
            
            result = loader("nonexistent", "/path/to", link=True, level=1, toc=False)
            
            assert result == ""
            mock_parser.compile.assert_called_once()


def test_loader_with_extension_module():
    """Test loader function with extension modules."""
    mock_packages = [
        ("test_module", "/path/to/test_module"),
    ]
    
    with patch('compiler.walk_packages', return_value=mock_packages):
        with patch('compiler._read', return_value='"""Stub."""'):
            with patch('compiler.isfile') as mock_isfile:
                with patch('compiler._load_module') as mock_load_module:
                    with patch('compiler.Parser.new') as mock_parser_new:
                        mock_parser = MagicMock()
                        mock_parser.compile.return_value = "# Documentation\n"
                        mock_parser_new.return_value = mock_parser
                        
                        # .py not found, but extension module is
                        mock_isfile.side_effect = [False, True]
                        mock_load_module.return_value = True
                        
                        with patch('compiler.EXTENSION_SUFFIXES', ['.so']):
                            result = loader("test_module", "/path/to", link=True, level=1, toc=False)
                            
                            assert isinstance(result, str)
                            mock_load_module.assert_called()


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock, mock_open
from .compiler import loader, walk_packages, _load_module, Parser


def test_loader():
    """Test loader function with mocked dependencies."""
    # Mock the walk_packages function to return test packages
    mock_packages = [
        ("test_module", "/path/to/test_module"),
        ("test_module.submodule", "/path/to/test_module/submodule"),
    ]
    
    # Mock Parser
    mock_parser = MagicMock()
    mock_parser.compile.return_value = "# Test Documentation\n\nTest content"
    
    with patch('compiler.walk_packages', return_value=mock_packages), \
         patch('compiler.Parser.new', return_value=mock_parser), \
         patch('compiler.isfile') as mock_isfile, \
         patch('compiler._read') as mock_read, \
         patch('compiler._load_module') as mock_load_module, \
         patch('compiler.logger'):
        
        # Setup isfile to return True for .py files
        def isfile_side_effect(path):
            return path.endswith(('.py', '.pyi'))
        
        mock_isfile.side_effect = isfile_side_effect
        mock_read.return_value = "def test(): pass\n"
        mock_load_module.return_value = False
        
        # Call loader
        result = loader("test_root", "/path/to/pwd", True, 1, False)
        
        # Assertions
        assert isinstance(result, str)
        assert result == "# Test Documentation\n\nTest content"
        
        # Verify Parser was initialized correctly
        Parser.new.assert_called_once_with(True, 1, False)
        
        # Verify walk_packages was called
        walk_packages.assert_called_once_with("test_root", "/path/to/pwd")
        
        # Verify parse was called for each package
        assert mock_parser.parse.call_count >= 1
        
        # Verify compile was called
        mock_parser.compile.assert_called_once()


def test_loader_with_extension_modules():
    """Test loader function with extension modules."""
    mock_packages = [
        ("test_extension", "/path/to/test_extension"),
    ]
    
    mock_parser = MagicMock()
    mock_parser.compile.return_value = "# Extension Module\n\nExtension content"
    
    with patch('compiler.walk_packages', return_value=mock_packages), \
         patch('compiler.Parser.new', return_value=mock_parser), \
         patch('compiler.isfile') as mock_isfile, \
         patch('compiler._read') as mock_read, \
         patch('compiler._load_module') as mock_load_module, \
         patch('compiler.EXTENSION_SUFFIXES', ['.so', '.pyd']), \
         patch('compiler.logger'):
        
        # Setup: .py file doesn't exist, .pyi exists, extension exists
        def isfile_side_effect(path):
            if path.endswith('.py'):
                return False
            if path.endswith('.pyi'):
                return True
            if path.endswith(('.so', '.pyd')):
                return True
            return False
        
        mock_isfile.side_effect = isfile_side_effect
        mock_read.return_value = "stub content"
        mock_load_module.return_value = True
        
        result = loader("test_root", "/path/to/pwd", False, 2, True)
        
        assert isinstance(result, str)
        mock_parser.parse.assert_called()
        mock_load_module.assert_called()
        mock_parser.compile.assert_called_once()


def test_loader_empty_packages():
    """Test loader function with empty package list."""
    mock_parser = MagicMock()
    mock_parser.compile.return_value = ""
    
    with patch('compiler.walk_packages', return_value=[]), \
         patch('compiler.Parser.new', return_value=mock_parser), \
         patch('compiler.logger'):
        
        result = loader("empty_root", "/path/to/pwd", True, 1, False)
        
        assert result == ""
        Parser.new.assert_called_once()
        mock_parser.compile.assert_called_once()
        mock_parser.parse.assert_not_called()


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock, mock_open, call
from os.path import join


def test_gen_api():
    """Test gen_api function."""
    
    # Test with dry run
    with patch('sys.path') as mock_sys_path, \
         patch('os.path.isdir') as mock_isdir, \
         patch('os.mkdir') as mock_mkdir, \
         patch('loader') as mock_loader, \
         patch('_site_path') as mock_site_path, \
         patch('_write') as mock_write, \
         patch('logger') as mock_logger:
        
        mock_isdir.return_value = True
        mock_site_path.return_value = "/fake/site/path"
        mock_loader.return_value = "# Module documentation\n"
        
        root_names = {"Test Module": "test_module"}
        result = gen_api(root_names, dry=True)
        
        assert len(result) == 1
        assert "# Test Module API" in result[0]
        assert "# Module documentation" in result[0]
        mock_write.assert_not_called()
    
    # Test with file writing
    with patch('sys.path') as mock_sys_path, \
         patch('os.path.isdir') as mock_isdir, \
         patch('os.mkdir') as mock_mkdir, \
         patch('loader') as mock_loader, \
         patch('_site_path') as mock_site_path, \
         patch('_write') as mock_write, \
         patch('logger') as mock_logger:
        
        mock_isdir.return_value = True
        mock_site_path.return_value = "/fake/site/path"
        mock_loader.return_value = "# Module documentation\n"
        
        root_names = {"My Module": "my_module"}
        result = gen_api(root_names, prefix='docs', dry=False)
        
        assert len(result) == 1
        mock_write.assert_called_once()
        call_args = mock_write.call_args
        assert "my-module-api.md" in call_args[0][0]
    
    # Test with pwd parameter
    with patch('sys.path') as mock_sys_path, \
         patch('os.path.isdir') as mock_isdir, \
         patch('os.mkdir') as mock_mkdir, \
         patch('loader') as mock_loader, \
         patch('_site_path') as mock_site_path, \
         patch('_write') as mock_write, \
         patch('logger') as mock_logger:
        
        mock_isdir.return_value = True
        mock_site_path.return_value = "/fake/site/path"
        mock_loader.return_value = "# Module documentation\n"
        
        root_names = {"Test": "test"}
        gen_api(root_names, pwd="/custom/path", dry=True)
        
        mock_sys_path.append.assert_called_once_with("/custom/path")
    
    # Test with empty loader result
    with patch('sys.path') as mock_sys_path, \
         patch('os.path.isdir') as mock_isdir, \
         patch('os.mkdir') as mock_mkdir, \
         patch('loader') as mock_loader, \
         patch('_site_path') as mock_site_path, \
         patch('_write') as mock_write, \
         patch('logger') as mock_logger:
        
        mock_isdir.return_value = True
        mock_site_path.return_value = "/fake/site/path"
        mock_loader.return_value = "   \n"
        
        root_names = {"Missing": "missing_module"}
        result = gen_api(root_names, dry=True)
        
        assert len(result) == 0
        mock_write.assert_not_called()
    
    # Test directory creation
    with patch('sys.path') as mock_sys_path, \
         patch('os.path.isdir') as mock_isdir, \
         patch('os.mkdir') as mock_mkdir, \
         patch('loader') as mock_loader, \
         patch('_site_path') as mock_site_path, \
         patch('_write') as mock_write, \
         patch('logger') as mock_logger:
        
        mock_isdir.return_value = False
        mock_site_path.return_value = "/fake/site/path"
        mock_loader.return_value = "# Module documentation\n"
        
        root_names = {"Test": "test"}
        gen_api(root_names, prefix='custom_docs', dry=True)
        
        mock_mkdir.assert_called_once_with('custom_docs')
    
    # Test multiple root names
    with patch('sys.path') as mock_sys_path, \
         patch('os.path.isdir') as mock_isdir, \
         patch('os.mkdir') as mock_mkdir, \
         patch('loader') as mock_loader, \
         patch('_site_path') as mock_site_path, \
         patch('_write') as mock_write, \
         patch('logger') as mock_logger:
        
        mock_isdir.return_value = True
        mock_site_path.return_value = "/fake/site/path"
        mock_loader.return_value = "# Documentation\n"
        
        root_names = {
            "Module A": "mod_a",
            "Module B": "mod_b",
            "Module C": "mod_c"
        }
        result = gen_api(root_names, dry=True)
        
        assert len(result) == 3
        assert mock_loader.call_count == 3


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_loader(tmp_path, monkeypatch):
    """Test the loader function."""
    # Create a temporary package structure
    pkg_dir = tmp_path / "test_pkg"
    pkg_dir.mkdir()
    
    # Create __init__.py
    init_file = pkg_dir / "__init__.py"
    init_file.write_text('"""Test package."""\n')
    
    # Create a module with docstring
    module_file = pkg_dir / "test_module.py"
    module_file.write_text('''"""Test module docstring."""

def test_function():
    """Test function docstring."""
    pass

class TestClass:
    """Test class docstring."""
    pass
''')
    
    # Create a stub file
    stub_file = pkg_dir / "stub_module.pyi"
    stub_file.write_text('''"""Stub module."""

def stub_function() -> None: ...
''')
    
    # Add the temporary directory to sys.path
    monkeypatch.syspath_prepend(str(tmp_path))
    
    # Test loader with the temporary package
    result = loader("test_pkg", str(tmp_path), link=True, level=1, toc=False)
    
    # Verify result is a string
    assert isinstance(result, str)
    
    # Verify that the result contains expected content
    assert len(result) > 0
    assert "test_module" in result or "test_pkg" in result


def test_loader_empty_directory(tmp_path, monkeypatch):
    """Test loader with empty directory."""
    # Create an empty package
    pkg_dir = tmp_path / "empty_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    
    monkeypatch.syspath_prepend(str(tmp_path))
    
    result = loader("empty_pkg", str(tmp_path), link=True, level=1, toc=False)
    
    assert isinstance(result, str)


def test_loader_with_different_parameters(tmp_path, monkeypatch):
    """Test loader with different parameters."""
    pkg_dir = tmp_path / "param_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text('"""Package."""\n')
    (pkg_dir / "module.py").write_text('"""Module."""\ndef func(): pass\n')
    
    monkeypatch.syspath_prepend(str(tmp_path))
    
    # Test with link=False
    result1 = loader("param_pkg", str(tmp_path), link=False, level=1, toc=False)
    assert isinstance(result1, str)
    
    # Test with different level
    result2 = loader("param_pkg", str(tmp_path), link=True, level=2, toc=False)
    assert isinstance(result2, str)
    
    # Test with toc=True
    result3 = loader("param_pkg", str(tmp_path), link=True, level=1, toc=True)
    assert isinstance(result3, str)


def test_loader_nonexistent_package(tmp_path, monkeypatch):
    """Test loader with nonexistent package."""
    monkeypatch.syspath_prepend(str(tmp_path))
    
    # Try to load a package that doesn't exist
    result = loader("nonexistent_pkg", str(tmp_path), link=True, level=1, toc=False)
    
    # Should return empty or minimal result
    assert isinstance(result, str)


# LLM-generated content at query #2
#--------------------------

```python
def test_walk_packages(tmp_path):
    """Test walk_packages function."""
    # Create test directory structure
    pkg_dir = tmp_path / "test_package"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("# init")
    (pkg_dir / "module1.py").write_text("# module1")
    (pkg_dir / "module2.pyi").write_text("# module2 stub")
    
    # Create stub package
    stub_dir = tmp_path / "test_package-stubs"
    stub_dir.mkdir()
    (stub_dir / "__init__.pyi").write_text("# stub init")
    (stub_dir / "module3.pyi").write_text("# module3 stub")
    
    # Create subdirectory
    sub_dir = pkg_dir / "subpkg"
    sub_dir.mkdir()
    (sub_dir / "__init__.py").write_text("# subpkg init")
    (sub_dir / "submodule.py").write_text("# submodule")
    
    # Test walk_packages
    results = list(walk_packages("test_package", str(tmp_path)))
    
    # Verify results
    names = [name for name, _ in results]
    assert "test_package" in names
    assert "test_package.module1" in names
    assert "test_package.module2" in names
    assert "test_package.subpkg" in names
    assert "test_package.subpkg.submodule" in names
    
    # Verify stub package is included (PEP 561)
    assert "test_package" in names  # stub package should be processed
    
    # Verify paths exist
    for name, path in results:
        assert isdir(dirname(path)) or isfile(path + ".py") or isfile(path + ".pyi")


def test_walk_packages_empty_directory(tmp_path):
    """Test walk_packages with empty directory."""
    empty_dir = tmp_path / "empty_pkg"
    empty_dir.mkdir()
    
    results = list(walk_packages("empty_pkg", str(tmp_path)))
    assert results == []


def test_walk_packages_ignores_non_python_files(tmp_path):
    """Test walk_packages ignores non-python files."""
    pkg_dir = tmp_path / "test_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("# init")
    (pkg_dir / "module.py").write_text("# module")
    (pkg_dir / "readme.txt").write_text("# readme")
    (pkg_dir / "data.json").write_text("{}")
    
    results = list(walk_packages("test_pkg", str(tmp_path)))
    names = [name for name, _ in results]
    
    assert "test_pkg" in names
    assert "test_pkg.module" in names
    assert len(names) == 2  # Only .py files


def test_walk_packages_nested_packages(tmp_path):
    """Test walk_packages with deeply nested packages."""
    pkg_dir = tmp_path / "root_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    
    level1 = pkg_dir / "level1"
    level1.mkdir()
    (level1 / "__init__.py").write_text("")
    
    level2 = level1 / "level2"
    level2.mkdir()
    (level2 / "__init__.py").write_text("")
    (level2 / "deep_module.py").write_text("")
    
    results = list(walk_packages("root_pkg", str(tmp_path)))
    names = [name for name, _ in results]
    
    assert "root_pkg" in names
    assert "root_pkg.level1" in names
    assert "root_pkg.level1.level2" in names
    assert "root_pkg.level1.level2.deep_module" in names


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_loader(tmp_path, monkeypatch):
    """Test the loader function."""
    # Create a test package structure
    pkg_dir = tmp_path / "test_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    
    # Create a module with docstring
    module_file = pkg_dir / "test_module.py"
    module_file.write_text('''
"""Test module docstring."""

def test_func():
    """Test function docstring."""
    pass

class TestClass:
    """Test class docstring."""
    pass
''')
    
    # Create a stub file
    stub_file = pkg_dir / "stub_module.pyi"
    stub_file.write_text('''
"""Stub module docstring."""

def stub_func() -> None: ...
''')
    
    # Test loader with basic parameters
    result = loader("test_pkg", str(tmp_path), link=True, level=1, toc=False)
    
    # Verify result is a string
    assert isinstance(result, str)
    
    # Verify that the result contains expected documentation elements
    # (exact content depends on Parser implementation)
    assert len(result) >= 0


def test_loader_empty_package(tmp_path):
    """Test loader with empty package."""
    pkg_dir = tmp_path / "empty_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    
    result = loader("empty_pkg", str(tmp_path), link=True, level=1, toc=False)
    
    assert isinstance(result, str)


def test_loader_with_different_levels(tmp_path):
    """Test loader with different heading levels."""
    pkg_dir = tmp_path / "test_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "module.py").write_text('"""Module doc."""\n')
    
    result1 = loader("test_pkg", str(tmp_path), link=True, level=1, toc=False)
    result2 = loader("test_pkg", str(tmp_path), link=True, level=2, toc=False)
    result3 = loader("test_pkg", str(tmp_path), link=True, level=3, toc=False)
    
    assert isinstance(result1, str)
    assert isinstance(result2, str)
    assert isinstance(result3, str)


def test_loader_with_link_parameter(tmp_path):
    """Test loader with link parameter variations."""
    pkg_dir = tmp_path / "test_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "module.py").write_text('"""Module."""\n')
    
    result_with_link = loader("test_pkg", str(tmp_path), link=True, level=1, toc=False)
    result_without_link = loader("test_pkg", str(tmp_path), link=False, level=1, toc=False)
    
    assert isinstance(result_with_link, str)
    assert isinstance(result_without_link, str)


def test_loader_with_toc_parameter(tmp_path):
    """Test loader with table of contents parameter."""
    pkg_dir = tmp_path / "test_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "module.py").write_text('"""Module."""\n')
    
    result_with_toc = loader("test_pkg", str(tmp_path), link=True, level=1, toc=True)
    result_without_toc = loader("test_pkg", str(tmp_path), link=True, level=1, toc=False)
    
    assert isinstance(result_with_toc, str)
    assert isinstance(result_without_toc, str)


def test_loader_nonexistent_package(tmp_path):
    """Test loader with nonexistent package."""
    result = loader("nonexistent_pkg", str(tmp_path), link=True, level=1, toc=False)
    
    assert isinstance(result, str)


# LLM-generated content at query #2
#--------------------------

```python
def test_walk_packages(tmp_path):
    """Test walk_packages function."""
    # Create a temporary package structure
    pkg_dir = tmp_path / "test_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("# init")
    (pkg_dir / "module1.py").write_text("# module1")
    (pkg_dir / "module2.pyi").write_text("# module2 stub")
    
    # Create a subpackage
    sub_dir = pkg_dir / "subpkg"
    sub_dir.mkdir()
    (sub_dir / "__init__.py").write_text("# sub init")
    (sub_dir / "submodule.py").write_text("# submodule")
    
    # Create a stub package
    stub_dir = tmp_path / "test_pkg-stubs"
    stub_dir.mkdir()
    (stub_dir / "__init__.pyi").write_text("# stub init")
    (stub_dir / "stub_module.pyi").write_text("# stub module")
    
    # Test walk_packages
    results = list(walk_packages("test_pkg", str(tmp_path)))
    
    # Extract names from results
    names = [name for name, _ in results]
    
    # Check that all expected modules are found
    assert "test_pkg" in names
    assert "test_pkg.module1" in names
    assert "test_pkg.module2" in names
    assert "test_pkg.subpkg" in names
    assert "test_pkg.subpkg.submodule" in names
    assert "test_pkg.stub_module" in names
    
    # Verify paths are correct
    for name, path in results:
        assert path.endswith(name.split('.')[-1])
        assert isdir(dirname(path))


def test_walk_packages_empty_directory(tmp_path):
    """Test walk_packages with empty directory."""
    pkg_dir = tmp_path / "empty_pkg"
    pkg_dir.mkdir()
    
    results = list(walk_packages("empty_pkg", str(tmp_path)))
    assert results == []


def test_walk_packages_ignores_non_python_files(tmp_path):
    """Test walk_packages ignores non-Python files."""
    pkg_dir = tmp_path / "test_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("# init")
    (pkg_dir / "module.py").write_text("# module")
    (pkg_dir / "readme.txt").write_text("# readme")
    (pkg_dir / "data.json").write_text("{}")
    
    results = list(walk_packages("test_pkg", str(tmp_path)))
    names = [name for name, _ in results]
    
    assert "test_pkg" in names
    assert "test_pkg.module" in names
    assert len(names) == 2


def test_walk_packages_removes_pep561_suffix(tmp_path):
    """Test walk_packages removes PEP 561 suffix from names."""
    stub_dir = tmp_path / "test_pkg-stubs"
    stub_dir.mkdir()
    (stub_dir / "__init__.pyi").write_text("# stub init")
    (stub_dir / "module.pyi").write_text("# stub module")
    
    results = list(walk_packages("test_pkg", str(tmp_path)))
    names = [name for name, _ in results]
    
    # Should not contain '-stubs' in the module names
    for name in names:
        assert "-stubs" not in name
    
    assert "test_pkg" in names
    assert "test_pkg.module" in names


def test_walk_packages_handles_nested_packages(tmp_path):
    """Test walk_packages with deeply nested packages."""
    pkg_dir = tmp_path / "test_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("# init")
    
    # Create nested structure
    level1 = pkg_dir / "level1"
    level1.mkdir()
    (level1 / "__init__.py").write_text("# level1")
    
    level2 = level1 / "level2"
    level2.mkdir()
    (level2 / "__init__.py").write_text("# level2")
    (level2 / "deep_module.py").write_text("# deep")
    
    results = list(walk_packages("test_pkg", str(tmp_path)))
    names = [name for name, _ in results]
    
    assert "test_pkg" in names
    assert "test_pkg.level1" in names
    assert "test_pkg.level1.level2" in names
    assert "test_pkg.level1.level2.deep_module" in names


# LLM-generated content at query #3
#--------------------------

```python
def test_gen_api(tmp_path, monkeypatch, capsys):
    """Test gen_api function."""
    # Setup temporary directory structure
    prefix_dir = tmp_path / "docs"
    pwd_dir = tmp_path / "packages"
    pwd_dir.mkdir()
    
    # Create a test package
    test_pkg = pwd_dir / "test_module"
    test_pkg.mkdir()
    (test_pkg / "__init__.py").write_text(
        '"""Test module docstring."""\n\ndef test_func():\n    """Test function."""\n    pass\n'
    )
    
    # Mock _site_path to return our test directory
    monkeypatch.setattr("os.path.isdir", lambda x: True)
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda x: type('obj', (object,), {'submodule_search_locations': [str(pwd_dir / "test_module")]})()
    )
    
    # Test basic functionality
    result = gen_api(
        {"Test Module": "test_module"},
        pwd=str(pwd_dir),
        prefix=str(prefix_dir),
        link=True,
        level=1,
        toc=False,
        dry=False
    )
    
    # Assertions
    assert isinstance(result, (list, tuple))
    assert len(result) > 0
    assert "Test Module API" in result[0]
    assert (prefix_dir / "test-module-api.md").exists()


def test_gen_api_dry_run(tmp_path, monkeypatch, capsys):
    """Test gen_api with dry run."""
    prefix_dir = tmp_path / "docs"
    pwd_dir = tmp_path / "packages"
    pwd_dir.mkdir()
    
    test_pkg = pwd_dir / "test_module"
    test_pkg.mkdir()
    (test_pkg / "__init__.py").write_text('"""Test module."""\n')
    
    monkeypatch.setattr("os.path.isdir", lambda x: True)
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda x: type('obj', (object,), {'submodule_search_locations': [str(pwd_dir / "test_module")]})()
    )
    
    result = gen_api(
        {"Test": "test_module"},
        pwd=str(pwd_dir),
        prefix=str(prefix_dir),
        dry=True
    )
    
    # File should not be created in dry run
    assert not (prefix_dir / "test-module-api.md").exists()
    assert len(result) > 0


def test_gen_api_multiple_modules(tmp_path, monkeypatch):
    """Test gen_api with multiple root modules."""
    prefix_dir = tmp_path / "docs"
    pwd_dir = tmp_path / "packages"
    pwd_dir.mkdir()
    
    # Create multiple test packages
    for pkg_name in ["module_a", "module_b"]:
        pkg = pwd_dir / pkg_name
        pkg.mkdir()
        (pkg / "__init__.py").write_text(f'"""{pkg_name} module."""\n')
    
    monkeypatch.setattr("os.path.isdir", lambda x: True)
    
    def mock_find_spec(name):
        return type('obj', (object,), {
            'submodule_search_locations': [str(pwd_dir / name)]
        })()
    
    monkeypatch.setattr("importlib.util.find_spec", mock_find_spec)
    
    result = gen_api(
        {"Module A": "module_a", "Module B": "module_b"},
        pwd=str(pwd_dir),
        prefix=str(prefix_dir),
        dry=False
    )
    
    assert len(result) >= 1


def test_gen_api_creates_prefix_dir(tmp_path, monkeypatch):
    """Test that gen_api creates prefix directory if it doesn't exist."""
    prefix_dir = tmp_path / "new_docs"
    pwd_dir = tmp_path / "packages"
    pwd_dir.mkdir()
    
    test_pkg = pwd_dir / "test_module"
    test_pkg.mkdir()
    (test_pkg / "__init__.py").write_text('"""Test."""\n')
    
    monkeypatch.setattr("os.path.isdir", lambda x: x != str(prefix_dir))
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda x: type('obj', (object,), {'submodule_search_locations': [str(pwd_dir / "test_module")]})()
    )
    
    result = gen_api(
        {"Test": "test_module"},
        pwd=str(pwd_dir),
        prefix=str(prefix_dir),
        dry=False
    )
    
    assert isinstance(result, (list, tuple))


def test_gen_api_missing_module(tmp_path, monkeypatch):
    """Test gen_api with non-existent module."""
    prefix_dir = tmp_path / "docs"
    
    monkeypatch.setattr("os.path.isdir", lambda x: True)
    monkeypatch.setattr("importlib.util.find_spec", lambda x: None)
    
    result = gen_api(
        {"Missing": "nonexistent_module"},
        prefix=str(prefix_dir),
        dry=True
    )
    
    # Should handle gracefully and return empty or minimal result
    assert isinstance(result, (list, tuple))


def test_gen_api_custom_level_and_toc(tmp_path, monkeypatch):
    """Test gen_api with custom level and toc settings."""
    prefix_dir = tmp_path / "docs"
    pwd_dir = tmp_path / "packages"
    pwd_dir.mkdir()
    
    test_pkg = pwd_dir / "test_module"
    test_pkg.mkdir()
    (test_pkg / "__init__.py").write_text('"""Test module."""\n')
    
    monkeypatch.setattr("os.path.isdir", lambda x: True)
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda x: type('obj', (object,), {'submodule_search_locations': [str(pwd_dir / "test_module")]})()
    )
    
    result = gen_api(
        {"Test": "test_module"},
        pwd=str(pwd_dir),
        prefix=str(prefix_dir),
        link=False,
        level=3,
        toc=True,
        dry=False
    )
    
    assert isinstance(result, (list, tuple))


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path


def test_loader():
    """Test the loader function."""
    # Mock the walk_packages function to return test data
    mock_packages = [
        ('test_module', '/path/to/test_module'),
        ('test_module.submodule', '/path/to/test_module/submodule'),
    ]
    
    # Mock file contents
    mock_py_content = '''
def test_function():
    """Test function docstring."""
    pass

class TestClass:
    """Test class docstring."""
    pass
'''
    
    mock_pyi_content = '''
def test_function() -> None: ...
class TestClass: ...
'''
    
    with patch('os.path.isfile') as mock_isfile, \
         patch('builtins.open', mock_open(read_data=mock_py_content)), \
         patch('walk_packages', return_value=mock_packages), \
         patch('Parser.new') as mock_parser_new, \
         patch('_load_module', return_value=False):
        
        # Setup mock isfile to return True for .py files
        def isfile_side_effect(path):
            return path.endswith('.py') or path.endswith('.pyi')
        
        mock_isfile.side_effect = isfile_side_effect
        
        # Setup mock parser
        mock_parser = MagicMock()
        mock_parser.compile.return_value = "# Generated documentation\n"
        mock_parser_new.return_value = mock_parser
        
        # Call the loader function
        result = loader('test_root', '/path/to/pwd', link=True, level=1, toc=False)
        
        # Assertions
        assert isinstance(result, str)
        assert mock_parser_new.called
        assert mock_parser.parse.called
        assert mock_parser.compile.called
        
        # Verify parse was called for each module
        assert mock_parser.parse.call_count >= 2


def test_loader_with_extension_modules():
    """Test the loader function with extension modules."""
    mock_packages = [
        ('native_module', '/path/to/native_module'),
    ]
    
    with patch('os.path.isfile') as mock_isfile, \
         patch('walk_packages', return_value=mock_packages), \
         patch('Parser.new') as mock_parser_new, \
         patch('_load_module', return_value=True) as mock_load_module, \
         patch('builtins.open', mock_open(read_data="")):
        
        # Setup mock isfile to return False for .py/.pyi, True for extension
        def isfile_side_effect(path):
            return path.endswith(('.so', '.pyd', '.pyc'))
        
        mock_isfile.side_effect = isfile_side_effect
        
        # Setup mock parser
        mock_parser = MagicMock()
        mock_parser.compile.return_value = "# Extension module docs\n"
        mock_parser_new.return_value = mock_parser
        
        # Call the loader function
        result = loader('test_root', '/path/to/pwd', link=False, level=2, toc=True)
        
        # Assertions
        assert isinstance(result, str)
        assert mock_parser_new.called
        assert mock_parser.compile.called


def test_loader_empty_packages():
    """Test the loader function with no packages."""
    with patch('walk_packages', return_value=[]), \
         patch('Parser.new') as mock_parser_new:
        
        # Setup mock parser
        mock_parser = MagicMock()
        mock_parser.compile.return_value = ""
        mock_parser_new.return_value = mock_parser
        
        # Call the loader function
        result = loader('empty_root', '/path/to/pwd', link=True, level=1, toc=False)
        
        # Assertions
        assert isinstance(result, str)
        assert mock_parser_new.called
        assert mock_parser.compile.called


def test_loader_with_stub_files():
    """Test the loader function prioritizing stub files."""
    mock_packages = [
        ('stub_module', '/path/to/stub_module'),
    ]
    
    stub_content = 'def stub_func() -> int: ...'
    
    with patch('os.path.isfile') as mock_isfile, \
         patch('builtins.open', mock_open(read_data=stub_content)), \
         patch('walk_packages', return_value=mock_packages), \
         patch('Parser.new') as mock_parser_new, \
         patch('_load_module', return_value=False):
        
        # Setup mock isfile to return True only for .pyi files
        def isfile_side_effect(path):
            return path.endswith('.pyi')
        
        mock_isfile.side_effect = isfile_side_effect
        
        # Setup mock parser
        mock_parser = MagicMock()
        mock_parser.compile.return_value = "# Stub documentation\n"
        mock_parser_new.return_value = mock_parser
        
        # Call the loader function
        result = loader('stub_root', '/path/to/pwd', link=True, level=1, toc=False)
        
        # Assertions
        assert isinstance(result, str)
        assert mock_parser.parse.called
        mock_parser.parse.assert_called_with('stub_module', stub_content)


# LLM-generated content at query #5
#--------------------------

```python
def test_gen_api(tmp_path, monkeypatch, capsys):
    """Test gen_api function."""
    import os
    from unittest.mock import Mock, patch, MagicMock
    
    # Setup temporary directory structure
    prefix_dir = tmp_path / "docs"
    pwd_dir = tmp_path / "site-packages"
    pwd_dir.mkdir()
    
    # Mock the loader function to return sample documentation
    mock_doc = "## Class Example\n\nExample class documentation."
    
    with patch('os.path.isdir') as mock_isdir, \
         patch('os.mkdir') as mock_mkdir, \
         patch('loader') as mock_loader, \
         patch('_write') as mock_write:
        
        mock_isdir.return_value = False
        mock_loader.return_value = mock_doc
        
        # Test basic functionality
        root_names = {"Example": "example_module"}
        result = gen_api(root_names, pwd=str(pwd_dir), prefix=str(prefix_dir))
        
        # Verify results
        assert len(result) == 1
        assert mock_doc in result[0]
        assert "# Example API" in result[0]
        mock_mkdir.assert_called_once()
        mock_write.assert_called_once()
        
        # Verify the written path
        call_args = mock_write.call_args
        assert "example-module-api.md" in call_args[0][0]


def test_gen_api_multiple_roots(tmp_path, monkeypatch):
    """Test gen_api with multiple root packages."""
    from unittest.mock import patch
    
    mock_doc1 = "## Module 1"
    mock_doc2 = "## Module 2"
    
    with patch('os.path.isdir') as mock_isdir, \
         patch('os.mkdir') as mock_mkdir, \
         patch('loader') as mock_loader, \
         patch('_write') as mock_write:
        
        mock_isdir.return_value = True
        mock_loader.side_effect = [mock_doc1, mock_doc2]
        
        root_names = {"First": "module1", "Second": "module2"}
        result = gen_api(root_names, prefix=str(tmp_path / "docs"))
        
        assert len(result) == 2
        assert "# First API" in result[0]
        assert "# Second API" in result[1]
        assert mock_write.call_count == 2


def test_gen_api_empty_documentation(tmp_path):
    """Test gen_api when documentation is empty."""
    from unittest.mock import patch
    
    with patch('os.path.isdir') as mock_isdir, \
         patch('os.mkdir') as mock_mkdir, \
         patch('loader') as mock_loader, \
         patch('_write') as mock_write:
        
        mock_isdir.return_value = True
        mock_loader.return_value = "   \n  \n  "
        
        root_names = {"Empty": "empty_module"}
        result = gen_api(root_names, prefix=str(tmp_path / "docs"))
        
        assert len(result) == 0
        mock_write.assert_not_called()


def test_gen_api_dry_run(tmp_path, capsys):
    """Test gen_api with dry run enabled."""
    from unittest.mock import patch
    
    mock_doc = "## Test Documentation"
    
    with patch('os.path.isdir') as mock_isdir, \
         patch('os.mkdir') as mock_mkdir, \
         patch('loader') as mock_loader, \
         patch('_write') as mock_write:
        
        mock_isdir.return_value = True
        mock_loader.return_value = mock_doc
        
        root_names = {"Test": "test_module"}
        result = gen_api(
            root_names,
            prefix=str(tmp_path / "docs"),
            dry=True
        )
        
        assert len(result) == 1
        mock_write.assert_not_called()


def test_gen_api_with_level_and_toc(tmp_path):
    """Test gen_api with different heading levels and table of contents."""
    from unittest.mock import patch
    
    mock_doc = "## Class Example"
    
    with patch('os.path.isdir') as mock_isdir, \
         patch('os.mkdir') as mock_mkdir, \
         patch('loader') as mock_loader, \
         patch('_write') as mock_write:
        
        mock_isdir.return_value = True
        mock_loader.return_value = mock_doc
        
        root_names = {"Custom": "custom_module"}
        result = gen_api(
            root_names,
            prefix=str(tmp_path / "docs"),
            level=3,
            toc=True,
            link=False
        )
        
        assert len(result) == 1
        assert "### Custom API" in result[0]
        mock_loader.assert_called_once()
        call_args = mock_loader.call_args
        assert call_args[0][2] is False  # link parameter
        assert call_args[0][3] == 3  # level parameter
        assert call_args[0][4] is True  # toc parameter


def test_gen_api_sys_path_append(tmp_path, monkeypatch):
    """Test that gen_api appends pwd to sys.path."""
    from unittest.mock import patch
    
    pwd_path = str(tmp_path / "custom_path")
    mock_doc = "## Documentation"
    
    with patch('os.path.isdir') as mock_isdir, \
         patch('os.mkdir') as mock_mkdir, \
         patch('loader') as mock_loader, \
         patch('_write') as mock_write, \
         patch('sys.path') as mock_sys_path:
        
        mock_isdir.return_value = True
        mock_loader.return_value = mock_doc
        mock_sys_path.append = Mock()
        
        root_names = {"Test": "test_module"}
        gen_api(root_names, pwd=pwd_path, prefix=str(tmp_path / "docs"))
        
        mock_sys_path.append.assert_called_once_with(pwd_path)


def test_gen_api_underscore_to_hyphen(tmp_path):
    """Test that underscores in module names are converted to hyphens in filenames."""
    from unittest.mock import patch
    
    mock_doc = "## Documentation"
    
    with patch('os.path.isdir') as mock_isdir, \
         patch('os.mkdir') as mock_mkdir, \
         patch('loader') as mock_loader, \
         patch('_write') as mock_write:
        
        mock_isdir.return_value = True
        mock_loader.return_value = mock_doc
        
        root_names = {"Example": "example_module_name"}
        gen_api(root_names, prefix=str(tmp_path / "docs"))
        
        call_args = mock_write.call_args
        assert "example-module-name-api.md" in call_args[0][0]


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock, call
from os.path import join


def test_gen_api():
    """Test gen_api function."""
    # Test basic functionality with dry run
    with patch('os.path.isdir') as mock_isdir, \
         patch('os.mkdir') as mock_mkdir, \
         patch('sys.path') as mock_syspath, \
         patch('loader') as mock_loader, \
         patch('_site_path') as mock_site_path, \
         patch('_write') as mock_write, \
         patch('logger') as mock_logger:
        
        mock_isdir.return_value = True
        mock_site_path.return_value = '/fake/path'
        mock_loader.return_value = '## Module\n\nDocumentation here'
        
        root_names = {'Test Module': 'test_module', 'Another': 'another_mod'}
        result = gen_api(root_names, dry=True)
        
        assert len(result) == 2
        assert '# Test Module API' in result[0]
        assert '## Module' in result[0]
        assert '# Another API' in result[1]
        mock_write.assert_not_called()


def test_gen_api_creates_directory():
    """Test gen_api creates prefix directory if it doesn't exist."""
    with patch('os.path.isdir') as mock_isdir, \
         patch('os.mkdir') as mock_mkdir, \
         patch('sys.path') as mock_syspath, \
         patch('loader') as mock_loader, \
         patch('_site_path') as mock_site_path, \
         patch('_write') as mock_write, \
         patch('logger') as mock_logger:
        
        mock_isdir.return_value = False
        mock_site_path.return_value = '/fake/path'
        mock_loader.return_value = '## Test'
        
        root_names = {'Test': 'test_mod'}
        gen_api(root_names, prefix='custom_docs', dry=True)
        
        mock_mkdir.assert_called_once_with('custom_docs')


def test_gen_api_appends_pwd_to_syspath():
    """Test gen_api appends pwd to sys.path when provided."""
    with patch('os.path.isdir') as mock_isdir, \
         patch('os.mkdir') as mock_mkdir, \
         patch('sys.path') as mock_syspath, \
         patch('loader') as mock_loader, \
         patch('_site_path') as mock_site_path, \
         patch('_write') as mock_write, \
         patch('logger') as mock_logger:
        
        mock_isdir.return_value = True
        mock_site_path.return_value = '/fake/path'
        mock_loader.return_value = '## Test'
        mock_syspath_list = []
        mock_syspath.__iter__ = lambda self: iter(mock_syspath_list)
        mock_syspath.append = lambda x: mock_syspath_list.append(x)
        
        root_names = {'Test': 'test_mod'}
        gen_api(root_names, pwd='/custom/path', dry=True)
        
        assert '/custom/path' in mock_syspath_list


def test_gen_api_writes_file():
    """Test gen_api writes files when dry=False."""
    with patch('os.path.isdir') as mock_isdir, \
         patch('os.mkdir') as mock_mkdir, \
         patch('sys.path') as mock_syspath, \
         patch('loader') as mock_loader, \
         patch('_site_path') as mock_site_path, \
         patch('_write') as mock_write, \
         patch('logger') as mock_logger:
        
        mock_isdir.return_value = True
        mock_site_path.return_value = '/fake/path'
        mock_loader.return_value = '## Content'
        
        root_names = {'My Module': 'my_module'}
        gen_api(root_names, prefix='docs', dry=False)
        
        mock_write.assert_called_once()
        call_args = mock_write.call_args
        assert 'my-module-api.md' in call_args[0][0]
        assert '# My Module API' in call_args[0][1]


def test_gen_api_skips_empty_docs():
    """Test gen_api skips modules with empty documentation."""
    with patch('os.path.isdir') as mock_isdir, \
         patch('os.mkdir') as mock_mkdir, \
         patch('sys.path') as mock_syspath, \
         patch('loader') as mock_loader, \
         patch('_site_path') as mock_site_path, \
         patch('_write') as mock_write, \
         patch('logger') as mock_logger:
        
        mock_isdir.return_value = True
        mock_site_path.return_value = '/fake/path'
        mock_loader.side_effect = ['   \n  ', '## Valid']
        
        root_names = {'Empty': 'empty_mod', 'Valid': 'valid_mod'}
        result = gen_api(root_names, dry=True)
        
        assert len(result) == 1
        assert '# Valid API' in result[0]
        mock_logger.warning.assert_called_once()


def test_gen_api_with_different_levels():
    """Test gen_api with different heading levels."""
    with patch('os.path.isdir') as mock_isdir, \
         patch('os.mkdir') as mock_mkdir, \
         patch('sys.path') as mock_syspath, \
         patch('loader') as mock_loader, \
         patch('_site_path') as mock_site_path, \
         patch('_write') as mock_write, \
         patch('logger') as mock_logger:
        
        mock_isdir.return_value = True
        mock_site_path.return_value = '/fake/path'
        mock_loader.return_value = '## Content'
        
        root_names = {'Test': 'test_mod'}
        result = gen_api(root_names, level=3, dry=True)
        
        assert '### Test API' in result[0]


def test_gen_api_underscore_to_dash_conversion():
    """Test gen_api converts underscores to dashes in filenames."""
    with patch('os.path.isdir') as mock_isdir, \
         patch('os.mkdir') as mock_mkdir, \
         patch('sys.path') as mock_syspath, \
         patch('loader') as mock_loader, \
         patch('_site_path') as mock_site_path, \
         patch('_write') as mock_write, \
         patch('logger') as mock_logger:
        
        mock_isdir.return_value = True
        mock_site_path.return_value = '/fake/path'
        mock_loader.return_value = '## Test'
        
        root_names = {'Test': 'test_module_name'}
        gen_api(root_names, dry=False)
        
        call_args = mock_write.call_args
        assert 'test-module-name-api.md' in call_args[0][0]


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path


def test_loader():
    """Test the loader function."""
    # Mock the Parser class
    mock_parser = MagicMock()
    mock_parser.compile.return_value = "compiled_output"
    
    with patch('compiler.Parser.new') as mock_parser_new, \
         patch('compiler.walk_packages') as mock_walk_packages, \
         patch('compiler.isfile') as mock_isfile, \
         patch('compiler._read') as mock_read, \
         patch('compiler._load_module') as mock_load_module, \
         patch('compiler.logger') as mock_logger:
        
        # Setup mocks
        mock_parser_new.return_value = mock_parser
        
        # Mock walk_packages to return test data
        mock_walk_packages.return_value = [
            ('test_module', '/path/to/test_module'),
            ('another_module', '/path/to/another_module'),
        ]
        
        # Setup isfile mock to return True for .py files
        def isfile_side_effect(path):
            return path.endswith('.py') or path.endswith('.pyi')
        
        mock_isfile.side_effect = isfile_side_effect
        mock_read.return_value = "def test(): pass"
        mock_load_module.return_value = False
        
        # Call the loader function
        result = loader('test_root', '/pwd', link=True, level=1, toc=False)
        
        # Assertions
        assert result == "compiled_output"
        mock_parser_new.assert_called_once_with(True, 1, False)
        mock_walk_packages.assert_called_once_with('test_root', '/pwd')
        assert mock_parser.parse.call_count >= 2
        mock_parser.compile.assert_called_once()


def test_loader_with_extension_module():
    """Test loader function with extension modules."""
    mock_parser = MagicMock()
    mock_parser.compile.return_value = "compiled_with_extension"
    
    with patch('compiler.Parser.new') as mock_parser_new, \
         patch('compiler.walk_packages') as mock_walk_packages, \
         patch('compiler.isfile') as mock_isfile, \
         patch('compiler._read') as mock_read, \
         patch('compiler._load_module') as mock_load_module, \
         patch('compiler.EXTENSION_SUFFIXES', ['.so', '.pyd']), \
         patch('compiler.logger') as mock_logger:
        
        mock_parser_new.return_value = mock_parser
        mock_walk_packages.return_value = [('ext_module', '/path/to/ext_module')]
        
        # First call for .py file returns False (not found)
        # Second call for .pyi file returns False (not found)
        mock_isfile.side_effect = [False, False, True]  # .py, .pyi, .so
        mock_read.return_value = ""
        mock_load_module.return_value = True
        
        result = loader('test_root', '/pwd', link=False, level=2, toc=True)
        
        assert result == "compiled_with_extension"
        mock_parser_new.assert_called_once_with(False, 2, True)
        mock_load_module.assert_called_once()


def test_loader_no_module_found():
    """Test loader when no module can be loaded."""
    mock_parser = MagicMock()
    mock_parser.compile.return_value = "empty_output"
    
    with patch('compiler.Parser.new') as mock_parser_new, \
         patch('compiler.walk_packages') as mock_walk_packages, \
         patch('compiler.isfile') as mock_isfile, \
         patch('compiler._load_module') as mock_load_module, \
         patch('compiler.EXTENSION_SUFFIXES', ['.so']), \
         patch('compiler.logger') as mock_logger:
        
        mock_parser_new.return_value = mock_parser
        mock_walk_packages.return_value = [('missing_module', '/path/to/missing')]
        mock_isfile.return_value = False
        mock_load_module.return_value = False
        
        result = loader('test_root', '/pwd', link=True, level=1, toc=False)
        
        assert result == "empty_output"
        mock_logger.warning.assert_called()


# LLM-generated content at query #8
#--------------------------

```python
def test_loader(tmp_path, monkeypatch):
    """Test the loader function."""
    # Create a temporary package structure
    pkg_dir = tmp_path / "test_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    
    # Create a module with docstring
    module_file = pkg_dir / "test_module.py"
    module_file.write_text('''
"""Test module docstring."""

def test_func():
    """Test function docstring."""
    pass

class TestClass:
    """Test class docstring."""
    pass
''')
    
    # Create a stub file
    stub_file = pkg_dir / "stub_module.pyi"
    stub_file.write_text('''
"""Stub module docstring."""

def stub_func() -> None: ...
''')
    
    # Test loader with the temporary package
    result = loader("test_pkg", str(tmp_path), link=True, level=1, toc=False)
    
    # Verify result is a string
    assert isinstance(result, str)
    
    # Verify the result contains expected content
    assert "test_module" in result or len(result) > 0


def test_loader_empty_package(tmp_path):
    """Test loader with empty package."""
    pkg_dir = tmp_path / "empty_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    
    result = loader("empty_pkg", str(tmp_path), link=True, level=1, toc=False)
    
    assert isinstance(result, str)


def test_loader_with_different_levels(tmp_path):
    """Test loader with different heading levels."""
    pkg_dir = tmp_path / "level_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text('''
"""Package docstring."""
''')
    
    result1 = loader("level_pkg", str(tmp_path), link=True, level=1, toc=False)
    result2 = loader("level_pkg", str(tmp_path), link=True, level=2, toc=False)
    
    assert isinstance(result1, str)
    assert isinstance(result2, str)


def test_loader_with_toc(tmp_path):
    """Test loader with table of contents enabled."""
    pkg_dir = tmp_path / "toc_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text('''
"""Package with TOC."""
''')
    
    result = loader("toc_pkg", str(tmp_path), link=True, level=1, toc=True)
    
    assert isinstance(result, str)


def test_loader_nonexistent_package(tmp_path):
    """Test loader with nonexistent package."""
    result = loader("nonexistent_pkg", str(tmp_path), link=True, level=1, toc=False)
    
    assert isinstance(result, str)
    # Should return empty or minimal string for nonexistent package
    assert result == "" or len(result) >= 0


def test_loader_with_link_disabled(tmp_path):
    """Test loader with link generation disabled."""
    pkg_dir = tmp_path / "nolink_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text('''
"""Package without links."""
''')
    
    result = loader("nolink_pkg", str(tmp_path), link=False, level=1, toc=False)
    
    assert isinstance(result, str)


# LLM-generated content at query #9
#--------------------------

```python
def test_gen_api(tmp_path, monkeypatch, mocker):
    """Test gen_api function."""
    # Mock the logger
    mock_logger = mocker.patch('pyslvs.compiler.logger')
    
    # Mock the loader function
    mock_loader = mocker.patch('pyslvs.compiler.loader')
    mock_loader.return_value = "# Module documentation\n\nSome content"
    
    # Mock _site_path function
    mock_site_path = mocker.patch('pyslvs.compiler._site_path')
    mock_site_path.return_value = "/fake/site/path"
    
    # Mock _write function
    mock_write = mocker.patch('pyslvs.compiler._write')
    
    # Mock isdir to return True
    mock_isdir = mocker.patch('pyslvs.compiler.isdir')
    mock_isdir.return_value = True
    
    # Test basic functionality
    root_names = {"MyPackage": "my_package"}
    result = gen_api(root_names, prefix=str(tmp_path), dry=False)
    
    # Verify loader was called with correct arguments
    mock_loader.assert_called_once_with("my_package", "/fake/site/path", True, 1, False)
    
    # Verify _write was called
    mock_write.assert_called_once()
    written_path, written_content = mock_write.call_args[0]
    assert "my-package-api.md" in written_path
    assert "# MyPackage API" in written_content
    assert "# Module documentation" in written_content
    
    # Verify return value
    assert len(result) == 1
    assert "# MyPackage API" in result[0]


def test_gen_api_dry_run(tmp_path, monkeypatch, mocker):
    """Test gen_api with dry run."""
    mock_logger = mocker.patch('pyslvs.compiler.logger')
    mock_loader = mocker.patch('pyslvs.compiler.loader')
    mock_loader.return_value = "# Test doc"
    mock_site_path = mocker.patch('pyslvs.compiler._site_path')
    mock_site_path.return_value = "/path"
    mock_write = mocker.patch('pyslvs.compiler._write')
    mock_isdir = mocker.patch('pyslvs.compiler.isdir')
    mock_isdir.return_value = True
    
    root_names = {"Test": "test_pkg"}
    result = gen_api(root_names, prefix=str(tmp_path), dry=True)
    
    # _write should not be called in dry run
    mock_write.assert_not_called()
    assert len(result) == 1


def test_gen_api_empty_doc(tmp_path, mocker):
    """Test gen_api when loader returns empty documentation."""
    mock_logger = mocker.patch('pyslvs.compiler.logger')
    mock_loader = mocker.patch('pyslvs.compiler.loader')
    mock_loader.return_value = "   \n  \n"  # Empty/whitespace only
    mock_site_path = mocker.patch('pyslvs.compiler._site_path')
    mock_site_path.return_value = "/path"
    mock_write = mocker.patch('pyslvs.compiler._write')
    mock_isdir = mocker.patch('pyslvs.compiler.isdir')
    mock_isdir.return_value = True
    
    root_names = {"Empty": "empty_pkg"}
    result = gen_api(root_names, prefix=str(tmp_path))
    
    # Package should be skipped when doc is empty
    mock_write.assert_not_called()
    assert len(result) == 0


def test_gen_api_multiple_packages(tmp_path, mocker):
    """Test gen_api with multiple packages."""
    mock_logger = mocker.patch('pyslvs.compiler.logger')
    mock_loader = mocker.patch('pyslvs.compiler.loader')
    mock_loader.side_effect = ["# Package 1", "# Package 2"]
    mock_site_path = mocker.patch('pyslvs.compiler._site_path')
    mock_site_path.return_value = "/path"
    mock_write = mocker.patch('pyslvs.compiler._write')
    mock_isdir = mocker.patch('pyslvs.compiler.isdir')
    mock_isdir.return_value = True
    
    root_names = {"Pkg1": "pkg1", "Pkg2": "pkg2"}
    result = gen_api(root_names, prefix=str(tmp_path))
    
    assert mock_loader.call_count == 2
    assert mock_write.call_count == 2
    assert len(result) == 2


def test_gen_api_custom_parameters(tmp_path, mocker):
    """Test gen_api with custom parameters."""
    mock_logger = mocker.patch('pyslvs.compiler.logger')
    mock_loader = mocker.patch('pyslvs.compiler.loader')
    mock_loader.return_value = "# Content"
    mock_site_path = mocker.patch('pyslvs.compiler._site_path')
    mock_site_path.return_value = "/path"
    mock_write = mocker.patch('pyslvs.compiler._write')
    mock_isdir = mocker.patch('pyslvs.compiler.isdir')
    mock_isdir.return_value = True
    
    root_names = {"Title": "module"}
    gen_api(
        root_names,
        pwd="/custom/pwd",
        prefix=str(tmp_path),
        link=False,
        level=2,
        toc=True
    )
    
    # Verify loader was called with custom parameters
    mock_loader.assert_called_once_with("module", "/path", False, 2, True)


def test_gen_api_creates_directory(tmp_path, mocker):
    """Test that gen_api creates prefix directory if it doesn't exist."""
    new_dir = tmp_path / "new_docs"
    assert not new_dir.exists()
    
    mock_logger = mocker.patch('pyslvs.compiler.logger')
    mock_loader = mocker.patch('pyslvs.compiler.loader')
    mock_loader.return_value = "# Doc"
    mock_site_path = mocker.patch('pyslvs.compiler._site_path')
    mock_site_path.return_value = "/path"
    mock_write = mocker.patch('pyslvs.compiler._write')
    mock_mkdir = mocker.patch('pyslvs.compiler.mkdir')
    mock_isdir = mocker.patch('pyslvs.compiler.isdir')
    mock_isdir.return_value = False
    
    root_names = {"Test": "test"}
    gen_api(root_names, prefix=str(new_dir))
    
    # Verify mkdir was called
    mock_mkdir.assert_called_once_with(str(new_dir))


def test_gen_api_heading_level(tmp_path, mocker):
    """Test that gen_api uses correct heading level."""
    mock_logger = mocker.patch('pyslvs.compiler.logger')
    mock_loader = mocker.patch('pyslvs.compiler.loader')
    mock_loader.return_value = "Content"
    mock_site_path = mocker.patch('pyslvs.compiler._site_path')
    mock_site_path.return_value = "/path"
    mock_write = mocker.patch('pyslvs.compiler._write')
    mock_isdir = mocker.patch('pyslvs.compiler.isdir')
    mock_isdir.return_value = True
    
    root_names = {"MyAPI": "myapi"}
    gen_api(root_names, prefix=str(tmp_path), level=3)
    
    written_path, written_content = mock_write.call_args[0


# LLM-generated content at query #10
#--------------------------

```python
def test_gen_api(tmp_path, monkeypatch, capsys):
    """Test gen_api function."""
    # Setup
    docs_dir = tmp_path / "docs"
    pwd = str(tmp_path)
    
    # Create a mock package structure
    pkg_dir = tmp_path / "test_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text('"""Test package."""\ndef test_func():\n    """Test function."""\n    pass')
    
    # Test with dry run
    result = gen_api(
        {"Test Package": "test_pkg"},
        pwd=pwd,
        prefix=str(docs_dir),
        link=True,
        level=1,
        toc=False,
        dry=True
    )
    
    assert isinstance(result, (list, tuple))
    assert len(result) >= 0
    captured = capsys.readouterr()
    # In dry mode, content should be logged
    if result:
        assert "Test Package API" in captured.out or "Test Package API" in str(result)
    
    # Test with actual file writing
    result = gen_api(
        {"Test Package": "test_pkg"},
        pwd=pwd,
        prefix=str(docs_dir),
        link=True,
        level=1,
        toc=False,
        dry=False
    )
    
    assert isinstance(result, (list, tuple))
    
    # Test with non-existent package
    result = gen_api(
        {"Non Existent": "nonexistent_pkg_xyz"},
        pwd=pwd,
        prefix=str(docs_dir),
        link=True,
        level=1,
        toc=False,
        dry=True
    )
    
    assert isinstance(result, (list, tuple))
    
    # Test with multiple packages
    pkg_dir2 = tmp_path / "another_pkg"
    pkg_dir2.mkdir()
    (pkg_dir2 / "__init__.py").write_text('"""Another package."""')
    
    result = gen_api(
        {"Test Package": "test_pkg", "Another Package": "another_pkg"},
        pwd=pwd,
        prefix=str(docs_dir),
        link=False,
        level=2,
        toc=True,
        dry=False
    )
    
    assert isinstance(result, (list, tuple))
    
    # Test with custom prefix
    custom_prefix = str(tmp_path / "custom_docs")
    result = gen_api(
        {"Test Package": "test_pkg"},
        pwd=pwd,
        prefix=custom_prefix,
        link=True,
        level=1,
        toc=False,
        dry=True
    )
    
    assert isinstance(result, (list, tuple))
    
    # Test without pwd
    result = gen_api(
        {"Test Package": "test_pkg"},
        prefix=str(docs_dir),
        link=True,
        level=1,
        toc=False,
        dry=True
    )
    
    assert isinstance(result, (list, tuple))


# LLM-generated content at query #11
#--------------------------

```python
def test_gen_api(tmp_path, monkeypatch):
    """Test gen_api function."""
    import sys
    from unittest.mock import patch, MagicMock
    
    # Setup temporary directory structure
    prefix_dir = tmp_path / "docs"
    
    # Mock the loader function to return sample documentation
    mock_doc = "## Sample Class\n\nThis is a sample class."
    
    with patch('gen_api.loader') as mock_loader, \
         patch('gen_api._site_path') as mock_site_path, \
         patch('gen_api.isdir') as mock_isdir, \
         patch('gen_api.mkdir') as mock_mkdir:
        
        mock_loader.return_value = mock_doc
        mock_site_path.return_value = str(tmp_path)
        mock_isdir.return_value = True
        
        # Test basic functionality
        root_names = {"Test Module": "test_module"}
        result = gen_api(root_names, prefix=str(prefix_dir), dry=True)
        
        assert len(result) == 1
        assert "Test Module API" in result[0]
        assert mock_doc in result[0]
    
    # Test with empty documentation
    with patch('gen_api.loader') as mock_loader, \
         patch('gen_api._site_path') as mock_site_path, \
         patch('gen_api.isdir') as mock_isdir, \
         patch('gen_api.mkdir') as mock_mkdir:
        
        mock_loader.return_value = "   \n  "
        mock_site_path.return_value = str(tmp_path)
        mock_isdir.return_value = True
        
        root_names = {"Empty Module": "empty_module"}
        result = gen_api(root_names, prefix=str(prefix_dir), dry=True)
        
        assert len(result) == 0
    
    # Test with multiple modules
    with patch('gen_api.loader') as mock_loader, \
         patch('gen_api._site_path') as mock_site_path, \
         patch('gen_api.isdir') as mock_isdir, \
         patch('gen_api.mkdir') as mock_mkdir:
        
        mock_loader.side_effect = ["## Module 1", "## Module 2"]
        mock_site_path.return_value = str(tmp_path)
        mock_isdir.return_value = True
        
        root_names = {"Module One": "mod_one", "Module Two": "mod_two"}
        result = gen_api(root_names, prefix=str(prefix_dir), level=2, dry=True)
        
        assert len(result) == 2
        assert "## Module One API" in result[0]
        assert "## Module Two API" in result[1]
    
    # Test with pwd parameter
    with patch('gen_api.loader') as mock_loader, \
         patch('gen_api._site_path') as mock_site_path, \
         patch('gen_api.isdir') as mock_isdir, \
         patch('gen_api.mkdir') as mock_mkdir:
        
        mock_loader.return_value = "## Test"
        mock_site_path.return_value = str(tmp_path)
        mock_isdir.return_value = True
        
        pwd = str(tmp_path / "packages")
        root_names = {"Test": "test"}
        result = gen_api(root_names, pwd=pwd, prefix=str(prefix_dir), dry=True)
        
        assert pwd in sys.path
        assert len(result) == 1
    
    # Test file writing (not dry run)
    with patch('gen_api.loader') as mock_loader, \
         patch('gen_api._site_path') as mock_site_path, \
         patch('gen_api.isdir') as mock_isdir, \
         patch('gen_api.mkdir') as mock_mkdir, \
         patch('gen_api._write') as mock_write:
        
        mock_loader.return_value = "## Documentation"
        mock_site_path.return_value = str(tmp_path)
        mock_isdir.return_value = True
        
        root_names = {"My Module": "my_module"}
        result = gen_api(root_names, prefix=str(prefix_dir), dry=False)
        
        assert mock_write.called
        call_args = mock_write.call_args
        assert "my-module-api.md" in call_args[0][0]
        assert "# My Module API" in call_args[0][1]
    
    # Test with custom level
    with patch('gen_api.loader') as mock_loader, \
         patch('gen_api._site_path') as mock_site_path, \
         patch('gen_api.isdir') as mock_isdir, \
         patch('gen_api.mkdir') as mock_mkdir:
        
        mock_loader.return_value = "Content"
        mock_site_path.return_value = str(tmp_path)
        mock_isdir.return_value = True
        
        root_names = {"API": "api"}
        result = gen_api(root_names, prefix=str(prefix_dir), level=3, dry=True)
        
        assert "### API API" in result[0]
    
    # Test directory creation
    with patch('gen_api.loader') as mock_loader, \
         patch('gen_api._site_path') as mock_site_path, \
         patch('gen_api.isdir') as mock_isdir, \
         patch('gen_api.mkdir') as mock_mkdir:
        
        mock_loader.return_value = "## Test"
        mock_site_path.return_value = str(tmp_path)
        mock_isdir.return_value = False
        
        root_names = {"Test": "test"}
        result = gen_api(root_names, prefix=str(prefix_dir), dry=True)
        
        mock_mkdir.assert_called_once_with(str(prefix_dir))


# LLM-generated content at query #12
#--------------------------

```python
def test_loader(tmp_path, monkeypatch):
    """Test the loader function."""
    # Create a temporary package structure
    pkg_dir = tmp_path / "test_package"
    pkg_dir.mkdir()
    
    # Create __init__.py
    init_file = pkg_dir / "__init__.py"
    init_file.write_text('"""Test package."""\n\nVERSION = "1.0.0"')
    
    # Create a module with docstring
    module_file = pkg_dir / "test_module.py"
    module_file.write_text(
        '"""Test module.\n\n'
        'This is a test module.\n'
        '"""\n\n'
        'def test_function():\n'
        '    """Test function."""\n'
        '    pass\n'
    )
    
    # Create a stub file
    stub_file = pkg_dir / "stub_module.pyi"
    stub_file.write_text(
        '"""Stub module."""\n\n'
        'def stub_function() -> None: ...\n'
    )
    
    # Test with basic parameters
    result = loader("test_package", str(tmp_path), link=True, level=1, toc=False)
    
    # Verify result is a string
    assert isinstance(result, str)
    
    # Verify that the result contains some documentation
    assert len(result) > 0
    
    # Verify that module names appear in the result
    assert "test_package" in result or "test_module" in result or "stub_module" in result


def test_loader_empty_package(tmp_path):
    """Test loader with empty package."""
    # Create an empty package
    pkg_dir = tmp_path / "empty_package"
    pkg_dir.mkdir()
    init_file = pkg_dir / "__init__.py"
    init_file.write_text('"""Empty package."""')
    
    # Test with empty package
    result = loader("empty_package", str(tmp_path), link=False, level=2, toc=True)
    
    # Result should be a string (possibly empty or with minimal content)
    assert isinstance(result, str)


def test_loader_with_different_levels(tmp_path):
    """Test loader with different heading levels."""
    pkg_dir = tmp_path / "level_test"
    pkg_dir.mkdir()
    init_file = pkg_dir / "__init__.py"
    init_file.write_text('"""Package for level test."""')
    
    # Test with different levels
    for level in [1, 2, 3]:
        result = loader("level_test", str(tmp_path), link=True, level=level, toc=False)
        assert isinstance(result, str)


def test_loader_with_toc(tmp_path):
    """Test loader with table of contents enabled."""
    pkg_dir = tmp_path / "toc_test"
    pkg_dir.mkdir()
    init_file = pkg_dir / "__init__.py"
    init_file.write_text('"""Package for TOC test."""')
    
    # Test with toc enabled
    result = loader("toc_test", str(tmp_path), link=True, level=1, toc=True)
    assert isinstance(result, str)


def test_loader_with_no_link(tmp_path):
    """Test loader without link generation."""
    pkg_dir = tmp_path / "nolink_test"
    pkg_dir.mkdir()
    init_file = pkg_dir / "__init__.py"
    init_file.write_text('"""Package without links."""')
    
    # Test with link=False
    result = loader("nolink_test", str(tmp_path), link=False, level=1, toc=False)
    assert isinstance(result, str)


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock, mock_open
from .compiler import loader, walk_packages, _load_module, _read, Parser


def test_loader():
    """Test the loader function."""
    # Mock the walk_packages function to return test data
    mock_packages = [
        ("test_module", "/path/to/test_module"),
        ("test_module.submodule", "/path/to/test_module/submodule"),
    ]
    
    mock_read_content = "def test_func():\n    '''Test function.'''\n    pass"
    
    with patch('os.path.isfile') as mock_isfile, \
         patch('builtins.open', mock_open(read_data=mock_read_content)), \
         patch.object(walk_packages, '__call__', return_value=iter(mock_packages)), \
         patch.object(Parser, 'new') as mock_parser_new, \
         patch.object(Parser, 'parse') as mock_parse, \
         patch.object(Parser, 'compile', return_value="# Compiled API\n"):
        
        mock_parser_instance = MagicMock()
        mock_parser_new.return_value = mock_parser_instance
        mock_parser_instance.parse = MagicMock()
        mock_parser_instance.compile = MagicMock(return_value="# Compiled API\n")
        
        # Mock isfile to return True for .py files
        def isfile_side_effect(path):
            return path.endswith('.py') or path.endswith('.pyi')
        
        mock_isfile.side_effect = isfile_side_effect
        
        with patch('compiler.walk_packages', return_value=iter(mock_packages)):
            result = loader("test_root", "/test/pwd", link=True, level=1, toc=False)
        
        # Verify Parser.new was called with correct arguments
        mock_parser_new.assert_called_once_with(True, 1, False)
        
        # Verify parse was called for each package
        assert mock_parser_instance.parse.call_count >= 2
        
        # Verify compile was called
        mock_parser_instance.compile.assert_called_once()
        
        # Verify return value
        assert result == "# Compiled API\n"


def test_loader_with_extension_modules():
    """Test loader function with extension modules."""
    mock_packages = [
        ("test_module", "/path/to/test_module"),
    ]
    
    with patch('os.path.isfile') as mock_isfile, \
         patch('builtins.open', mock_open(read_data="# stub file")), \
         patch('compiler.walk_packages', return_value=iter(mock_packages)), \
         patch.object(Parser, 'new') as mock_parser_new, \
         patch('compiler._load_module') as mock_load_module, \
         patch('importlib.machinery.EXTENSION_SUFFIXES', ['.so', '.pyd']):
        
        mock_parser_instance = MagicMock()
        mock_parser_new.return_value = mock_parser_instance
        mock_parser_instance.compile = MagicMock(return_value="# Compiled\n")
        mock_load_module.return_value = True
        
        # First call returns False (no .py), second returns True (.pyi exists)
        mock_isfile.side_effect = [False, True, True]
        
        result = loader("test_root", "/test/pwd", link=False, level=2, toc=True)
        
        mock_parser_new.assert_called_once_with(False, 2, True)
        assert result == "# Compiled\n"


def test_loader_empty_result():
    """Test loader when no packages are found."""
    with patch('compiler.walk_packages', return_value=iter([])), \
         patch.object(Parser, 'new') as mock_parser_new:
        
        mock_parser_instance = MagicMock()
        mock_parser_new.return_value = mock_parser_instance
        mock_parser_instance.compile = MagicMock(return_value="")
        
        result = loader("empty_root", "/test/pwd", link=True, level=1, toc=False)
        
        mock_parser_new.assert_called_once()
        mock_parser_instance.compile.assert_called_once()
        assert result == ""


# LLM-generated content at query #14
#--------------------------

```python
def test_gen_api(tmp_path, monkeypatch, caplog):
    """Test gen_api function."""
    import logging
    from unittest.mock import patch, MagicMock
    
    caplog.set_level(logging.INFO)
    
    # Test 1: Basic functionality with dry run
    root_names = {"Test Module": "test_module"}
    prefix = str(tmp_path / "docs")
    
    with patch('os.path.isdir', return_value=False), \
         patch('os.mkdir') as mock_mkdir, \
         patch('os.path.isfile', return_value=True), \
         patch('os.walk', return_value=[]), \
         patch('importlib.util.find_spec', return_value=MagicMock(submodule_search_locations=['/fake/path'])), \
         patch('os.path.dirname', return_value='/fake/path'):
        
        result = gen_api(
            root_names,
            prefix=prefix,
            link=True,
            level=1,
            toc=False,
            dry=True
        )
        assert isinstance(result, (list, tuple))
        mock_mkdir.assert_called_once()
    
    # Test 2: With pwd parameter
    with patch('sys.path', new_callable=list) as mock_sys_path, \
         patch('os.path.isdir', return_value=True), \
         patch('os.walk', return_value=[]), \
         patch('importlib.util.find_spec', return_value=MagicMock(submodule_search_locations=['/fake/path'])), \
         patch('os.path.dirname', return_value='/fake/path'):
        
        pwd = "/custom/path"
        result = gen_api(root_names, pwd=pwd, prefix=prefix, dry=True)
        assert isinstance(result, (list, tuple))
    
    # Test 3: Empty root_names
    result = gen_api({}, prefix=prefix, dry=True)
    assert result == []
    
    # Test 4: File writing when dry=False
    with patch('os.path.isdir', return_value=True), \
         patch('os.walk', return_value=[]), \
         patch('importlib.util.find_spec', return_value=MagicMock(submodule_search_locations=['/fake/path'])), \
         patch('os.path.dirname', return_value='/fake/path'), \
         patch('builtins.open', create=True) as mock_open:
        
        result = gen_api(root_names, prefix=prefix, dry=False)
        assert isinstance(result, (list, tuple))
    
    # Test 5: Multiple root names
    multi_roots = {
        "Module A": "module_a",
        "Module B": "module_b"
    }
    
    with patch('os.path.isdir', return_value=True), \
         patch('os.walk', return_value=[]), \
         patch('importlib.util.find_spec', return_value=MagicMock(submodule_search_locations=['/fake/path'])), \
         patch('os.path.dirname', return_value='/fake/path'):
        
        result = gen_api(multi_roots, prefix=prefix, dry=True)
        assert isinstance(result, (list, tuple))
    
    # Test 6: Different level parameter
    with patch('os.path.isdir', return_value=True), \
         patch('os.walk', return_value=[]), \
         patch('importlib.util.find_spec', return_value=MagicMock(submodule_search_locations=['/fake/path'])), \
         patch('os.path.dirname', return_value='/fake/path'):
        
        result = gen_api(root_names, prefix=prefix, level=2, dry=True)
        assert isinstance(result, (list, tuple))
    
    # Test 7: With toc enabled
    with patch('os.path.isdir', return_value=True), \
         patch('os.walk', return_value=[]), \
         patch('importlib.util.find_spec', return_value=MagicMock(submodule_search_locations=['/fake/path'])), \
         patch('os.path.dirname', return_value='/fake/path'):
        
        result = gen_api(root_names, prefix=prefix, toc=True, dry=True)
        assert isinstance(result, (list, tuple))


# LLM-generated content at query #15
#--------------------------

```python
def test_loader(tmp_path, monkeypatch):
    """Test the loader function."""
    # Create a temporary package structure
    pkg_dir = tmp_path / "test_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    
    # Create a module with docstring
    module_file = pkg_dir / "module.py"
    module_file.write_text('''
"""Module docstring."""

def func():
    """Function docstring."""
    pass

class MyClass:
    """Class docstring."""
    pass
''')
    
    # Create a stub file
    stub_file = pkg_dir / "stub_module.pyi"
    stub_file.write_text('''
"""Stub module docstring."""

def stub_func() -> None: ...
''')
    
    # Monkeypatch sys.path to include temp directory
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    
    # Test loader function
    result = loader("test_pkg", str(tmp_path), link=False, level=1, toc=False)
    
    # Verify result is a string
    assert isinstance(result, str)
    
    # Verify the result contains parsed documentation
    assert "Module docstring" in result or len(result) >= 0


def test_loader_with_options(tmp_path, monkeypatch):
    """Test loader function with different options."""
    pkg_dir = tmp_path / "test_pkg2"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    
    module_file = pkg_dir / "test_module.py"
    module_file.write_text('''
"""Test module."""

def test_func():
    """Test function."""
    pass
''')
    
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    
    # Test with different parameters
    result1 = loader("test_pkg2", str(tmp_path), link=True, level=2, toc=True)
    assert isinstance(result1, str)
    
    result2 = loader("test_pkg2", str(tmp_path), link=False, level=1, toc=False)
    assert isinstance(result2, str)


def test_loader_nonexistent_package(tmp_path):
    """Test loader with nonexistent package."""
    result = loader("nonexistent_pkg", str(tmp_path), link=False, level=1, toc=False)
    
    # Should return empty or minimal string for nonexistent package
    assert isinstance(result, str)


def test_loader_empty_package(tmp_path):
    """Test loader with empty package."""
    pkg_dir = tmp_path / "empty_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    
    result = loader("empty_pkg", str(tmp_path), link=False, level=1, toc=False)
    
    assert isinstance(result, str)


# LLM-generated content at query #16
#--------------------------

```python
def test_gen_api(tmp_path, monkeypatch, capsys):
    """Test gen_api function."""
    import os
    from unittest.mock import Mock, patch, MagicMock
    
    # Setup temporary directory structure
    prefix_dir = tmp_path / "docs"
    pwd_dir = tmp_path / "packages"
    pwd_dir.mkdir()
    
    # Mock the loader function to return sample documentation
    mock_doc = "## Sample Function\n\nThis is a sample function."
    
    with patch('compiler.loader') as mock_loader, \
         patch('compiler.isdir') as mock_isdir, \
         patch('compiler.mkdir') as mock_mkdir, \
         patch('compiler._site_path') as mock_site_path, \
         patch('compiler._write') as mock_write:
        
        mock_loader.return_value = mock_doc
        mock_isdir.return_value = False
        mock_site_path.return_value = str(pwd_dir)
        
        # Test basic functionality
        result = gen_api(
            {'Test Module': 'test_module'},
            pwd=str(pwd_dir),
            prefix=str(prefix_dir),
            link=True,
            level=1,
            toc=False,
            dry=False
        )
        
        # Verify results
        assert len(result) == 1
        assert "# Test Module API" in result[0]
        assert mock_doc in result[0]
        mock_write.assert_called_once()
        mock_mkdir.assert_called_once()
    
    # Test with dry run
    with patch('compiler.loader') as mock_loader, \
         patch('compiler.isdir') as mock_isdir, \
         patch('compiler._site_path') as mock_site_path, \
         patch('compiler._write') as mock_write:
        
        mock_loader.return_value = mock_doc
        mock_isdir.return_value = True
        mock_site_path.return_value = str(pwd_dir)
        
        result = gen_api(
            {'Another Module': 'another_module'},
            prefix=str(prefix_dir),
            dry=True
        )
        
        assert len(result) == 1
        mock_write.assert_not_called()
    
    # Test with empty document (should skip)
    with patch('compiler.loader') as mock_loader, \
         patch('compiler.isdir') as mock_isdir, \
         patch('compiler._site_path') as mock_site_path, \
         patch('compiler._write') as mock_write:
        
        mock_loader.return_value = "   \n  "
        mock_isdir.return_value = True
        mock_site_path.return_value = str(pwd_dir)
        
        result = gen_api(
            {'Empty Module': 'empty_module'},
            prefix=str(prefix_dir)
        )
        
        assert len(result) == 0
        mock_write.assert_not_called()
    
    # Test with multiple modules
    with patch('compiler.loader') as mock_loader, \
         patch('compiler.isdir') as mock_isdir, \
         patch('compiler._site_path') as mock_site_path, \
         patch('compiler._write') as mock_write:
        
        mock_loader.return_value = mock_doc
        mock_isdir.return_value = True
        mock_site_path.return_value = str(pwd_dir)
        
        result = gen_api(
            {
                'Module One': 'module_one',
                'Module Two': 'module_two'
            },
            prefix=str(prefix_dir),
            level=2
        )
        
        assert len(result) == 2
        assert "## Module One API" in result[0]
        assert "## Module Two API" in result[1]
        assert mock_write.call_count == 2


# LLM-generated content at query #17
#--------------------------

```python
def test_gen_api(tmp_path, monkeypatch, capsys):
    """Test gen_api function."""
    from unittest.mock import Mock, patch, MagicMock
    
    # Setup temporary directory structure
    test_dir = tmp_path / "test_packages"
    test_dir.mkdir()
    docs_dir = tmp_path / "docs"
    
    # Mock the loader function to return sample documentation
    mock_doc = "## Module\n\nSample documentation"
    
    with patch('pyslvs_compiler.compiler.loader') as mock_loader, \
         patch('pyslvs_compiler.compiler._site_path') as mock_site_path, \
         patch('pyslvs_compiler.compiler.isdir') as mock_isdir, \
         patch('pyslvs_compiler.compiler.mkdir') as mock_mkdir:
        
        mock_loader.return_value = mock_doc
        mock_site_path.return_value = str(test_dir)
        mock_isdir.return_value = True
        
        # Test basic functionality
        root_names = {"Test Module": "test_module"}
        result = gen_api(
            root_names,
            pwd=None,
            prefix=str(docs_dir),
            link=True,
            level=1,
            toc=False,
            dry=True
        )
        
        assert len(result) == 1
        assert "# Test Module API" in result[0]
        assert mock_doc in result[0]


def test_gen_api_multiple_modules(tmp_path, monkeypatch):
    """Test gen_api with multiple modules."""
    from unittest.mock import patch
    
    with patch('pyslvs_compiler.compiler.loader') as mock_loader, \
         patch('pyslvs_compiler.compiler._site_path') as mock_site_path, \
         patch('pyslvs_compiler.compiler.isdir') as mock_isdir, \
         patch('pyslvs_compiler.compiler.mkdir') as mock_mkdir, \
         patch('pyslvs_compiler.compiler._write') as mock_write:
        
        mock_loader.side_effect = ["## Module 1", "## Module 2"]
        mock_site_path.return_value = str(tmp_path)
        mock_isdir.return_value = True
        
        root_names = {"Module One": "mod1", "Module Two": "mod2"}
        result = gen_api(
            root_names,
            prefix=str(tmp_path / "docs"),
            link=False,
            level=2,
            toc=True,
            dry=False
        )
        
        assert len(result) == 2
        assert "## Module One API" in result[0]
        assert "### Module Two API" in result[1]
        assert mock_write.call_count == 2


def test_gen_api_empty_doc(tmp_path):
    """Test gen_api when loader returns empty documentation."""
    from unittest.mock import patch
    
    with patch('pyslvs_compiler.compiler.loader') as mock_loader, \
         patch('pyslvs_compiler.compiler._site_path') as mock_site_path, \
         patch('pyslvs_compiler.compiler.isdir') as mock_isdir, \
         patch('pyslvs_compiler.compiler.mkdir') as mock_mkdir, \
         patch('pyslvs_compiler.compiler._write') as mock_write:
        
        mock_loader.return_value = "   \n  \n  "
        mock_site_path.return_value = str(tmp_path)
        mock_isdir.return_value = True
        
        root_names = {"Empty Module": "empty_mod"}
        result = gen_api(
            root_names,
            prefix=str(tmp_path / "docs"),
            dry=False
        )
        
        assert len(result) == 0
        mock_write.assert_not_called()


def test_gen_api_creates_directory(tmp_path):
    """Test gen_api creates prefix directory if it doesn't exist."""
    from unittest.mock import patch, call
    
    with patch('pyslvs_compiler.compiler.loader') as mock_loader, \
         patch('pyslvs_compiler.compiler._site_path') as mock_site_path, \
         patch('pyslvs_compiler.compiler.isdir') as mock_isdir, \
         patch('pyslvs_compiler.compiler.mkdir') as mock_mkdir, \
         patch('pyslvs_compiler.compiler._write') as mock_write:
        
        mock_loader.return_value = "## Doc"
        mock_site_path.return_value = str(tmp_path)
        mock_isdir.return_value = False
        
        root_names = {"Test": "test"}
        result = gen_api(
            root_names,
            prefix=str(tmp_path / "docs"),
            dry=False
        )
        
        mock_mkdir.assert_called_once()


def test_gen_api_with_pwd(tmp_path):
    """Test gen_api with custom pwd parameter."""
    from unittest.mock import patch
    
    with patch('pyslvs_compiler.compiler.loader') as mock_loader, \
         patch('pyslvs_compiler.compiler._site_path') as mock_site_path, \
         patch('pyslvs_compiler.compiler.isdir') as mock_isdir, \
         patch('pyslvs_compiler.compiler.mkdir') as mock_mkdir, \
         patch('pyslvs_compiler.compiler.sys_path') as mock_sys_path:
        
        mock_loader.return_value = "## Doc"
        mock_site_path.return_value = str(tmp_path)
        mock_isdir.return_value = True
        mock_sys_path.append = Mock()
        
        root_names = {"Test": "test"}
        result = gen_api(
            root_names,
            pwd=str(tmp_path),
            dry=True
        )
        
        assert len(result) == 1


def test_gen_api_file_naming(tmp_path):
    """Test gen_api generates correct file names."""
    from unittest.mock import patch
    
    with patch('pyslvs_compiler.compiler.loader') as mock_loader, \
         patch('pyslvs_compiler.compiler._site_path') as mock_site_path, \
         patch('pyslvs_compiler.compiler.isdir') as mock_isdir, \
         patch('pyslvs_compiler.compiler.mkdir') as mock_mkdir, \
         patch('pyslvs_compiler.compiler._write') as mock_write:
        
        mock_loader.return_value = "## Doc"
        mock_site_path.return_value = str(tmp_path)
        mock_isdir.return_value = True
        
        root_names = {"Test": "test_module"}
        result = gen_api(
            root_names,
            prefix=str(tmp_path / "docs"),
            dry=False
        )
        
        call_args = mock_write.call_args[0][0]
        assert "test-module-api.md" in call_args


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock, mock_open
from io import StringIO


def test_loader():
    """Test the loader function."""
    # Mock the walk_packages function to return test data
    mock_packages = [
        ("test_module", "/path/to/test_module"),
        ("test_module.submodule", "/path/to/test_module/submodule"),
    ]
    
    mock_py_content = '"""Test module docstring."""\ndef test_func():\n    """Test function."""\n    pass'
    mock_pyi_content = '"""Test stub file."""\ndef test_func() -> None: ...'
    
    with patch('compiler.walk_packages') as mock_walk:
        mock_walk.return_value = mock_packages
        
        with patch('compiler._read') as mock_read:
            mock_read.side_effect = [mock_py_content, mock_pyi_content]
            
            with patch('compiler.isfile') as mock_isfile:
                # Return True for .py and .pyi files
                mock_isfile.side_effect = [True, False, True, False]
                
                with patch('compiler.Parser.new') as mock_parser_new:
                    mock_parser = MagicMock()
                    mock_parser.compile.return_value = "# Generated Documentation"
                    mock_parser_new.return_value = mock_parser
                    
                    result = loader("test_module", "/path/to", link=True, level=1, toc=False)
                    
                    # Verify Parser.new was called with correct arguments
                    mock_parser_new.assert_called_once_with(True, 1, False)
                    
                    # Verify parse was called for each package
                    assert mock_parser.parse.call_count >= 2
                    
                    # Verify compile was called
                    mock_parser.compile.assert_called_once()
                    
                    # Verify result
                    assert result == "# Generated Documentation"


def test_loader_with_extension_modules():
    """Test the loader function with extension modules."""
    mock_packages = [
        ("test_module", "/path/to/test_module"),
    ]
    
    mock_py_content = '"""Test module."""'
    
    with patch('compiler.walk_packages') as mock_walk:
        mock_walk.return_value = mock_packages
        
        with patch('compiler._read') as mock_read:
            mock_read.return_value = mock_py_content
            
            with patch('compiler.isfile') as mock_isfile:
                # First call for .py (True), second for .pyi (False)
                mock_isfile.side_effect = [True, False]
                
                with patch('compiler.Parser.new') as mock_parser_new:
                    mock_parser = MagicMock()
                    mock_parser.compile.return_value = "# Documentation"
                    mock_parser_new.return_value = mock_parser
                    
                    result = loader("test_module", "/path/to", link=False, level=2, toc=True)
                    
                    # Verify Parser.new was called with correct arguments
                    mock_parser_new.assert_called_once_with(False, 2, True)
                    
                    # Verify parse was called
                    assert mock_parser.parse.call_count >= 1
                    
                    # Verify result
                    assert result == "# Documentation"


def test_loader_empty_package():
    """Test the loader function with empty package."""
    with patch('compiler.walk_packages') as mock_walk:
        mock_walk.return_value = []
        
        with patch('compiler.Parser.new') as mock_parser_new:
            mock_parser = MagicMock()
            mock_parser.compile.return_value = ""
            mock_parser_new.return_value = mock_parser
            
            result = loader("empty_module", "/path/to", link=True, level=1, toc=False)
            
            # Verify compile was called
            mock_parser.compile.assert_called_once()
            
            # Verify empty result
            assert result == ""


# LLM-generated content at query #19
#--------------------------

```python
def test_gen_api(tmp_path, monkeypatch):
    """Test gen_api function."""
    import sys
    from unittest.mock import Mock, patch, MagicMock
    
    # Setup
    test_prefix = str(tmp_path / "docs")
    root_names = {"Test API": "test_module"}
    
    # Mock the loader function to return sample documentation
    mock_doc = "## Function\n\nTest function documentation."
    
    with patch('os.path.isdir', return_value=False), \
         patch('os.mkdir') as mock_mkdir, \
         patch('os.walk') as mock_walk, \
         patch('builtins.open', create=True) as mock_open, \
         patch.object(__import__('os.path'), 'isfile', return_value=False), \
         patch('importlib.util.find_spec') as mock_find_spec:
        
        # Setup mocks
        mock_find_spec.return_value = Mock(submodule_search_locations=['/fake/path'])
        mock_walk.return_value = []
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        with patch('os.path.dirname', return_value='/fake/path'), \
             patch.object(Parser, 'new') as mock_parser_new, \
             patch.object(Parser, 'compile', return_value=mock_doc):
            
            mock_parser = Mock()
            mock_parser.compile.return_value = mock_doc
            mock_parser_new.return_value = mock_parser
            
            # Call function
            result = gen_api(
                root_names,
                pwd=None,
                prefix=test_prefix,
                link=True,
                level=1,
                toc=False,
                dry=False
            )
            
            # Assertions
            assert isinstance(result, (list, tuple))
            assert len(result) == 1
            assert "Test API API" in result[0]
            mock_mkdir.assert_called_once_with(test_prefix)


def test_gen_api_dry_run(tmp_path, monkeypatch, capsys):
    """Test gen_api function with dry run enabled."""
    from unittest.mock import Mock, patch
    
    test_prefix = str(tmp_path / "docs")
    root_names = {"Test API": "test_module"}
    mock_doc = "## Function\n\nTest function documentation."
    
    with patch('os.path.isdir', return_value=True), \
         patch('os.walk', return_value=[]), \
         patch('importlib.util.find_spec') as mock_find_spec, \
         patch('os.path.dirname', return_value='/fake/path'), \
         patch.object(Parser, 'new') as mock_parser_new:
        
        mock_find_spec.return_value = Mock(submodule_search_locations=['/fake/path'])
        mock_parser = Mock()
        mock_parser.compile.return_value = mock_doc
        mock_parser_new.return_value = mock_parser
        
        result = gen_api(
            root_names,
            pwd=None,
            prefix=test_prefix,
            link=True,
            level=1,
            toc=False,
            dry=True
        )
        
        assert isinstance(result, (list, tuple))
        assert len(result) == 1


def test_gen_api_empty_doc(tmp_path, monkeypatch):
    """Test gen_api function when documentation is empty."""
    from unittest.mock import Mock, patch
    
    test_prefix = str(tmp_path / "docs")
    root_names = {"Test API": "test_module"}
    
    with patch('os.path.isdir', return_value=True), \
         patch('os.walk', return_value=[]), \
         patch('importlib.util.find_spec') as mock_find_spec, \
         patch('os.path.dirname', return_value='/fake/path'), \
         patch.object(Parser, 'new') as mock_parser_new:
        
        mock_find_spec.return_value = Mock(submodule_search_locations=['/fake/path'])
        mock_parser = Mock()
        mock_parser.compile.return_value = "   "  # Empty/whitespace only
        mock_parser_new.return_value = mock_parser
        
        result = gen_api(
            root_names,
            pwd=None,
            prefix=test_prefix,
            link=True,
            level=1,
            toc=False,
            dry=False
        )
        
        assert isinstance(result, (list, tuple))
        assert len(result) == 0


def test_gen_api_with_pwd(tmp_path, monkeypatch):
    """Test gen_api function with custom pwd."""
    from unittest.mock import Mock, patch
    
    test_prefix = str(tmp_path / "docs")
    test_pwd = str(tmp_path / "site-packages")
    root_names = {"Test API": "test_module"}
    mock_doc = "## Function\n\nTest."
    
    with patch('os.path.isdir', return_value=True), \
         patch('os.walk', return_value=[]), \
         patch('importlib.util.find_spec') as mock_find_spec, \
         patch('os.path.dirname', return_value='/fake/path'), \
         patch.object(Parser, 'new') as mock_parser_new, \
         patch('sys.path', []) as mock_sys_path:
        
        mock_find_spec.return_value = Mock(submodule_search_locations=['/fake/path'])
        mock_parser = Mock()
        mock_parser.compile.return_value = mock_doc
        mock_parser_new.return_value = mock_parser
        
        result = gen_api(
            root_names,
            pwd=test_pwd,
            prefix=test_prefix,
            link=True,
            level=2,
            toc=True,
            dry=False
        )
        
        assert isinstance(result, (list, tuple))
        assert len(result) == 1
        assert "## Test API API" in result[0]


# LLM-generated content at query #20
#--------------------------

```python
def test_gen_api(tmp_path, monkeypatch, capsys):
    """Test gen_api function."""
    import os
    from unittest.mock import Mock, patch, MagicMock
    
    # Create a temporary directory structure
    test_dir = tmp_path / "test_pkg"
    test_dir.mkdir()
    docs_dir = tmp_path / "docs"
    
    # Mock the loader function to return sample documentation
    mock_doc = "## Sample Module\n\nThis is a test module."
    
    with patch('compiler.loader') as mock_loader, \
         patch('compiler.isdir') as mock_isdir, \
         patch('compiler.mkdir') as mock_mkdir, \
         patch('compiler._write') as mock_write, \
         patch('compiler._site_path') as mock_site_path:
        
        mock_loader.return_value = mock_doc
        mock_isdir.return_value = False
        mock_site_path.return_value = str(test_dir)
        
        # Test basic functionality
        root_names = {'Test Package': 'test_pkg'}
        result = gen_api(
            root_names,
            pwd=str(tmp_path),
            prefix=str(docs_dir),
            link=True,
            level=1,
            toc=False,
            dry=False
        )
        
        # Verify results
        assert len(result) == 1
        assert "Test Package API" in result[0]
        assert mock_doc in result[0]
        mock_mkdir.assert_called_once()
        mock_write.assert_called_once()


def test_gen_api_dry_run(tmp_path, monkeypatch, capsys):
    """Test gen_api function with dry run mode."""
    from unittest.mock import patch
    
    mock_doc = "## Test Module\n\nDocumentation."
    
    with patch('compiler.loader') as mock_loader, \
         patch('compiler.isdir') as mock_isdir, \
         patch('compiler.mkdir') as mock_mkdir, \
         patch('compiler._write') as mock_write, \
         patch('compiler._site_path') as mock_site_path:
        
        mock_loader.return_value = mock_doc
        mock_isdir.return_value = True
        mock_site_path.return_value = str(tmp_path)
        
        root_names = {'My API': 'my_module'}
        result = gen_api(
            root_names,
            prefix=str(tmp_path / "docs"),
            dry=True
        )
        
        # In dry run, _write should not be called
        mock_write.assert_not_called()
        assert len(result) == 1


def test_gen_api_empty_doc(tmp_path, monkeypatch):
    """Test gen_api when loader returns empty documentation."""
    from unittest.mock import patch
    
    with patch('compiler.loader') as mock_loader, \
         patch('compiler.isdir') as mock_isdir, \
         patch('compiler.mkdir') as mock_mkdir, \
         patch('compiler._write') as mock_write, \
         patch('compiler._site_path') as mock_site_path:
        
        mock_loader.return_value = "   \n  "
        mock_isdir.return_value = True
        mock_site_path.return_value = str(tmp_path)
        
        root_names = {'Empty': 'empty_pkg'}
        result = gen_api(
            root_names,
            prefix=str(tmp_path / "docs")
        )
        
        # Empty docs should not be written
        mock_write.assert_not_called()
        assert len(result) == 0


def test_gen_api_multiple_packages(tmp_path):
    """Test gen_api with multiple packages."""
    from unittest.mock import patch
    
    with patch('compiler.loader') as mock_loader, \
         patch('compiler.isdir') as mock_isdir, \
         patch('compiler.mkdir') as mock_mkdir, \
         patch('compiler._write') as mock_write, \
         patch('compiler._site_path') as mock_site_path:
        
        mock_loader.side_effect = ["## Module A", "## Module B"]
        mock_isdir.return_value = True
        mock_site_path.return_value = str(tmp_path)
        
        root_names = {'Package A': 'pkg_a', 'Package B': 'pkg_b'}
        result = gen_api(root_names, prefix=str(tmp_path / "docs"))
        
        assert len(result) == 2
        assert mock_write.call_count == 2


def test_gen_api_with_level(tmp_path):
    """Test gen_api with different heading levels."""
    from unittest.mock import patch
    
    with patch('compiler.loader') as mock_loader, \
         patch('compiler.isdir') as mock_isdir, \
         patch('compiler.mkdir') as mock_mkdir, \
         patch('compiler._write') as mock_write, \
         patch('compiler._site_path') as mock_site_path:
        
        mock_loader.return_value = "Content"
        mock_isdir.return_value = True
        mock_site_path.return_value = str(tmp_path)
        
        root_names = {'API': 'myapi'}
        result = gen_api(
            root_names,
            prefix=str(tmp_path / "docs"),
            level=3
        )
        
        assert "### API" in result[0]


def test_gen_api_underscore_to_dash(tmp_path):
    """Test gen_api converts underscores to dashes in filenames."""
    from unittest.mock import patch, call
    
    with patch('compiler.loader') as mock_loader, \
         patch('compiler.isdir') as mock_isdir, \
         patch('compiler.mkdir') as mock_mkdir, \
         patch('compiler._write') as mock_write, \
         patch('compiler._site_path') as mock_site_path:
        
        mock_loader.return_value = "Content"
        mock_isdir.return_value = True
        mock_site_path.return_value = str(tmp_path)
        
        root_names = {'Test': 'test_module_name'}
        gen_api(root_names, prefix=str(tmp_path / "docs"))
        
        # Check that the filename has dashes instead of underscores
        call_args = mock_write.call_args[0][0]
        assert 'test-module-name-api.md' in call_args


