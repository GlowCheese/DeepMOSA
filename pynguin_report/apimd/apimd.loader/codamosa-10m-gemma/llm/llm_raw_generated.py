####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch
from os import mkdir, remove, rmdir
from os.path import isdir, join, exists

@pytest.fixture
def temp_package_structure(tmp_path):
    """Creates a temporary directory structure simulating a package."""
    pkg_dir = tmp_path / "test_pkg"
    pkg_dir.mkdir()
    
    # Create __init__.py
    init_file = pkg_dir / "__init__.py"
    init_file.write_text("# init doc", encoding="utf-8")
    
    # Create a module with a stub file
    sub_dir = pkg_dir / "submodule"
    sub_dir.mkdir()
    stub_file = sub_dir / "__init__.pyi"
    stub_file.write_text('"""Stub docstring"""\ndef func(): pass', encoding="utf-8")
    
    # Create an extension module (simulated via .so/.pyd suffix)
    # Note: EXTENSION_SUFFIXES varies by platform, but we will mock the check
    ext_file = sub_dir / "ext.so"
    ext_file.write_text("dummy content", encoding="utf-8")
    
    yield pkg_dir

@patch("compiler.Parser")
@patch("compiler.walk_packages")
@patch("compiler.isfile")
@patch("compiler._read")
@patch("compiler._load_module")
def test_loader(
    mock_load_module,
    mock_read,
    mock_isfile,
    mock_walk_packages,
    mock_parser_class,
    temp_package_structure
):
    # Setup Mock Parser
    mock_parser_instance = MagicMock()
    mock_parser_instance.compile.return_value = "Compiled Docstring"
    mock_parser_class.new.return_value = mock_parser_instance
    
    # Setup Mock walk_packages to return our temp package
    # Format: (name, path)
    mock_walk_packages.return_value = [
        ("test_pkg", str(temp_package_structure)),
        ("test_pkg.submodule", str(temp_package_structure / "submodule"))
    ]
    
    # Setup Mock isfile to return True for our files
    def side_effect_isfile(path):
        return True
    mock_isfile.side_effect = side_effect_isfile
    
    # Setup Mock _read
    mock_read.return_value = "module docstring"
    
    # Setup Mock _load_module
    mock_load_module.return_value = True
    
    # Execute loader
    result = loader(
        root="test_pkg",
        pwd=str(temp_package_structure),
        link=True,
        level=1,
        toc=True
    )
    
    # Assertions
    assert result == "Compiled Docstring"
    
    # Verify Parser was initialized correctly
    mock_parser_class.new.assert_called_once_with(True, 1, True)
    
    # Verify parser.parse was called for the files found
    # Since isfile always returned True, it should attempt to parse the names found in walk_packages
    assert mock_parser_instance.parse.called
    
    # Verify _load_module was called for the submodule (since it has a .pyi stub)
    # The logic in loader skips 'pure_py' (files that only have .py)
    # If it finds .pyi, it proceeds to check extension suffixes
    mock_load_module.assert_called()
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch
from os.path import join, sep

def test_loader():
    """
    Unit test for the loader function.
    Tests the orchestration of walking packages, parsing source/stubs,
    and loading extension modules.
    """
    # Mocking inputs
    root_path = "/tmp/root"
    pwd_path = "/tmp/pwd"
    link_val = True
    level_val = 1
    toc_val = False

    # Mocking the Parser
    mock_parser = MagicMock()
    mock_parser.compile.return_value = "Compiled Docstring"
    
    # Mocking the Parser.new factory method
    with patch('parser.Parser.new', return_value=mock_parser), \
         patch('compiler.walk_packages') as mock_walk, \
         patch('compiler.isfile') as mock_isfile, \
         patch('compiler.read') as mock_read, \
         patch('compiler._load_module') as mock_load_module, \
         patch('compiler.logger') as mock_logger:

        # Scenario 1: A pure Python package (only .py exists)
        # walk_packages yields: (name, path_base)
        mock_walk.return_value = [('my_pkg', join(root_path, 'my_pkg'))]
        
        # Setup file existence: .py exists, but .pyi and extensions do not
        def isfile_side_effect(path):
            return path.endswith('.py')
        mock_isfile.side_effect = isfile_side_effect
        
        # Setup reading content
        mock_read.return_value = "def hello(): pass"

        result = loader(root_path, pwd_path, link_val, level_val, toc_val)

        # Assertions for Scenario 1
        assert result == "Compiled Docstring"
        mock_parser.parse.assert_called_with('my_pkg', "def hello(): pass")
        # Should NOT attempt to load extension module because pure_py is True
        mock_load_module.assert_not_called()

        # Scenario 2: A package with stubs (.pyi) that requires extension loading
        mock_walk.return_value = [('ext_pkg', join(root_path, 'ext_pkg'))]
        
        # Setup: .pyi exists, but .py does not. Extension (.so/.pyd) exists.
        def isfile_side_effect_ext(path):
            # Return true for .pyi and a dummy extension suffix
            return path.endswith('.pyi') or path.endswith('.pyd')
        mock_isfile.side_effect = isfile_side_effect_ext
        
        mock_read.return_value = "# Stub content"
        mock_load_module.return_value = True

        result_ext = loader(root_path, pwd_path, link_val, level_val, toc_val)

        # Assertions for Scenario 2
        assert result_ext == "Compiled Docstring"
        # Verify parse was called for the stub
        mock_parser.parse.assert_any_call('ext_pkg', "# Stub content")
        # Verify _load_module was triggered because pure_py was False
        mock_load_module.assert_called()

        # Scenario 3: A package where no module is found (warning case)
        mock_walk.return_value = [('broken_pkg', join(root_path, 'broken_pkg'))]
        mock_isfile.side_effect = lambda p: p.endswith('.pyi') # only stub exists, no extension
        mock_load_module.return_value = False
        
        loader(root_path, pwd_path, link_val, level_val, toc_val)
        mock_logger.warning.assert_called_with("no module for broken_pkg in this platform")
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
import os
from pathlib import Path

def test_gen_api():
    """
    Tests the gen_api function by mocking filesystem operations, 
    module loading, and the parser/loader logic.
    """
    # Test data
    root_names = {"My Module": "my_module"}
    prefix = "test_docs_output"
    
    # Mocking the dependencies and filesystem
    # We mock:
    # 1. logger to prevent actual logging output in tests
    # 2. os.mkdir to prevent actual directory creation
    # 3. os.path.isdir to simulate directory existence
    # 4. _site_path to return a dummy path
    # 5. loader to return a dummy docstring
    # 6. _write to prevent actual file writing
    # 7. sys_path append/pop
    
    with patch('pyslvs.compiler.logger') as mock_logger, \
         patch('pyslvs.compiler.mkdir') as mock_mkdir, \
         patch('pyslvs.compiler.isdir') as mock_isdir, \
         patch('pyslvs.compiler._site_path') as mock_site_path, \
         patch('pyslvs.compiler.loader') as mock_loader, \
         patch('pyslvs.compiler._write') as mock_write, \
         patch('pyslvs.compiler.sys_path', ['/tmp']) as mock_sys_path:
        
        # Setup mock behaviors
        mock_isdir.return_value = False  # Simulate that prefix directory doesn't exist
        mock_site_path.return_value = "/fake/path"
        mock_loader.return_value = "This is the parsed docstring content."
        
        # --- Case 1: Normal execution (Writing files) ---
        results = gen_api(
            root_names=root_names,
            pwd="/fake/pwd",
            prefix=prefix,
            link=True,
            level=1,
            toc=True,
            dry=False
        )
        
        # Assertions for Case 1
        assert len(results) == 1
        assert "# My Module API\n\nThis is the parsed docstring content." in results[0]
        mock_mkdir.assert_called_once_with(prefix)
        mock_write.assert_called_once()
        # Verify file path construction: my_module -> my-module-api.md
        expected_path = os.path.join(prefix, "my-module-api.md")
        args, _ = mock_write.call_args
        assert args[0] == expected_path
        
        # --- Case 2: Dry run (No writing, just printing) ---
        mock_write.reset_mock()
        results_dry = gen_api(
            root_names=root_names,
            prefix=prefix,
            dry=True
        )
        
        # Assertions for Case 2
        assert len(results_dry) == 1
        mock_write.assert_not_called()
        # Verify that logger.info was called with the doc content in dry mode
        # The doc content is passed as a single string argument to the second logger.info call
        # (First is "Create directory...", second is the doc)
        found_doc_in_logs = any(
            "This is the parsed docstring content." in str(call) 
            for call in mock_logger.info.call_args_list
        )
        assert found_doc_in_logs is True

        # --- Case 3: Module not found (Empty docstring) ---
        mock_loader.return_value = "" # Simulate failed parsing
        results_empty = gen_api(root_names={"Empty": "empty_mod"})
        
        # Assertions for Case 3
        assert len(results_empty) == 0
        mock_logger.warning.assert_called_with("'empty_mod' can not be found")

        # --- Case 4: Verification of sys_path manipulation ---
        # Ensure pwd was added to sys_path
        assert "/fake/pwd" in mock_sys_path
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock, mock_open
from os.path import join, exists

@pytest.fixture
def mock_env(tmp_path):
    """Setup a temporary environment for testing."""
    prefix_dir = tmp_path / "docs_output"
    # Create a dummy structure
    return {
        "prefix": str(prefix_dir),
        "tmp_path": str(tmp_path)
    }

def test_gen_api(mock_exports, mock_env):
    """
    Test gen_api function by mocking the underlying heavy-lifting functions:
    loader, _site_path, _write, and os operations.
    """
    # Setup inputs
    root_names = {"my_package": "My Package"}
    prefix = mock_env["prefix"]
    
    # Mocking dependencies
    # We mock 'loader' to avoid actual file system walking and module loading
    # We mock '_site_path' to avoid searching sys.path
    # We mock '_write' to avoid actual disk writes
    # We mock 'mkdir' to avoid actual directory creation
    
    with patch('compiler.loader') as mock_loader, \
         patch('compiler._site_path') as mock_site_path, \
         patch('compiler._write') as mock_write, \
         patch('compiler.mkdir') as mock_mkdir, \
         patch('compiler.isdir', return_value=True), \
         patch('compiler.logger') as mock_logger:
        
        # Define what the loader returns (the compiled docstring)
        mock_loader.return_value = "Contents of the API docstring."
        mock_site_path.return_value = "/fake/path/to/package"
        
        # Execute the function
        result = gen_api(
            root_names=root_names,
            prefix=prefix,
            link=True,
            level=2,
            toc=True,
            dry=False
        )
        
        # Assertions
        
        # 1. Check if the returned content is correctly formatted
        expected_doc = "## My Package API\n\nContents of the API docstring."
        assert len(result) == 1
        assert result[0] == expected_doc
        
        # 2. Check if loader was called with correct arguments
        mock_loader.assert_called_once_with(
            "my_package", 
            "/fake/path/to/package", 
            True, 1, True
        )
        
        # 3. Check if the file writing was attempted at the correct path
        expected_file_path = join(prefix, "my-package-api.md")
        mock_write.assert_called_once_with(expected_file_path, expected_doc)

def test_gen_api_dry_run(mock_env):
    """Test gen_api with dry=True to ensure no files are written."""
    root_names = {"pkg": "Pkg"}
    
    with patch('compiler.loader', return_value="doc"), \
         patch('compiler._site_path', return_value="."), \
         patch('compiler._write') as mock_write, \
         patch('compiler.mkdir'), \
         patch('compiler.isdir', return_value=True), \
         patch('compiler.logger') as mock_logger:
        
        result = gen_api(root_names, prefix="docs", dry=True)
        
        # Verify no write occurred
        mock_write.assert_not_called()
        # Verify logger was used to print the doc instead
        # The doc contains the content returned by loader
        found_log = False
        for call in mock_logger.info.call_args_list:
            if "doc" in str(call) or "Pkg API" in str(call):
                found_log = True
        assert found_log

def test_gen_api_empty_loader(mock_env):
    """Test gen_api when the loader returns an empty string (package not found)."""
    root_names = {"missing_pkg": "Missing"}
    
    with patch('compiler.loader', return_value="   "), \
         patch('compiler._site_path', return_value="."), \
         patch('compiler._write') as mock_write, \
         patch('compiler.mkdir'), \
         patch('compiler.isdir', return_value=True), \
         patch('compiler.logger') as mock_logger:
        
        result = gen_api(root_names, prefix="docs")
        
        # Result should be empty because doc.strip() was empty
        assert len(result) == 0
        mock_write.assert_not_called()
        # Check if warning was logged
        mock_logger.warning.assert_called_with("'missing_pkg' can not be found")
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch
from os import mkdir, remove, rmdir
from os.path import isdir, join, exists

@pytest.fixture
def temp_package_structure(tmp_path):
    """Creates a temporary directory structure representing a package."""
    pkg_dir = tmp_path / "test_pkg"
    pkg_dir.mkdir()
    
    # Create __init__.py
    init_file = pkg_dir / "__init__.py"
    init_file.write_text("# init content", encoding="utf-8")
    
    # Create a stub file
    stub_file = pkg_dir / "module.pyi"
    stub_file.write_text("# stub content", encoding="utf-8")
    
    # Create a dummy extension file (simulating a compiled module)
    # Note: In a real test, we'd need a valid loader, 
    # but here we mock the existence of the file.
    ext_file = pkg_dir / "module.so" 
    ext_file.write_text("dummy binary", encoding="utf-8")

    yield pkg_dir

    # Cleanup is handled by tmp_path fixture automatically

@patch("compiler.Parser")
@patch("compiler.walk_packages")
@patch("compiler.isfile")
@patch("compiler.read")
@patch("compiler.logger")
def test_loader(mock_logger, mock_read, mock_isfile, mock_walk_packages, mock_parser_class, temp_package_structure):
    """
    Tests the loader function by mocking the file system, 
    the parser, and the package walking process.
    """
    # Setup Mock Parser
    mock_parser_instance = MagicMock()
    mock_parser_instance.compile.return_value = "Compiled Docstring"
    mock_parser_class.new.return_value = mock_parser_instance
    
    # Setup Mock walk_packages
    # Yielding (name, path)
    pkg_path = str(temp_package_module_path := temp_package_structure / "module")
    mock_walk_packages.return_value = [("test_pkg.module", pkg_path)]
    
    # Setup Mock isfile
    # We simulate that .py and .pyi exist for the walk, but we control the extension logic
    def side_effect_isfile(path):
        # Return True for the .py and .pyi files we created in the fixture
        return exists(path) or path.endswith(".py") or path.endswith(".pyi")
    mock_isfile.side_effect = side_effect_isfile

    # Setup Mock _read
    mock_read.return_value = "mock file content"

    # Setup Mock _load_module to return False to avoid complex import machinery in unit test
    with patch("compiler._load_module", return_value=False) as mock_load_mod:
        
        # Execute the function
        result = loader(
            root="test_pkg",
            pwd=str(temp_package_structure),
            link=True,
            level=1,
            toc=True
        )

        # Assertions
        assert result == "Compiled Docstring"
        
        # Verify Parser was initialized with correct params
        mock_parser_class.new.assert_called_once_with(True, 1, True)
        
        # Verify walk_packages was called with correct root and pwd
        # Note: walk_packages uses abspath internally
        import os
        expected_pwd = os.path.abspath(str(temp_package_structure))
        mock_walk_packages.assert_called_once_with("test_pkg", expected_pwd)

        # Verify parser.parse was called for the files found
        assert mock_parser_instance.parse.called
        
        # Verify the logic for pure_py (if .py exists, it should skip extension loading)
        # Since our fixture has an __init__.py, the walk finds it.
        # If we find a .py, pure_py becomes True, and it should 'continue' (skip extension loop)
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch
from os import mkdir, remove, rmdir
from os.path import isdir, join, exists

@pytest.fixture
def temp_package_structure(tmp_path):
    """Creates a temporary directory structure representing a package."""
    pkg_dir = tmp_path / "test_pkg"
    pkg_dir.mkdir()
    
    # Create __init__.py
    init_file = pkg_dir / "__init__.py"
    init_file.write_text("'''Root Docstring'''\npass")
    
    # Create a submodule with a .pyi stub
    sub_dir = pkg_dir / "submodule"
    sub_dir.mkdir()
    stub_file = sub_dir / "__init__.pyi"
    stub_file.write_text("'''Stub Docstring'''\ndef func(): pass")
    
    # Create an extension-like file (.pyd or .so simulation via .py)
    # Note: loader looks for EXTENSION_SUFFIXES. 
    # We will mock the extension check in the test.
    
    yield pkg_dir
    
    # Cleanup is handled by tmp_path fixture

@patch("compiler.Parser")
@patch("compiler.walk_packages")
@patch("compiler.isfile")
@patch("compiler.ext_suffix_exists", new_callable=pytest.importorskip("unittest.mock").MagicMock)
def test_loader(mock_ext_exists, mock_isfile, mock_walk, mock_parser_class, temp_package_structure):
    """
    Test the loader function logic:
    1. Iterates through packages.
    2. Parses .py and .pyi files.
    3. If it's an extension module (no .py), attempts to load via _load_module.
    4. Returns compiled docstring.
    """
    # Setup Mock Parser
    mock_parser = MagicMock()
    mock_parser_class.new.return_value = mock_parser
    mock_parser.compile.return_value = "Compiled Docstring"
    
    # Setup Mock walk_packages to return our temp package
    # format: (name, path)
    pkg_path = str(temp_package_structure)
    pkg_name = "test_pkg"
    mock_walk.return_value = [(pkg_name, pkg_path)]
    
    # Setup Mock isfile
    # We simulate that .py exists for the root, but for the submodule we simulate it's an extension
    def side_effect_isfile(path):
        # Simulate root __init__.py exists
        if "test_pkg/__init__.py" in path:
            return True
        # Simulate submodule .py does NOT exist, but we want to trigger extension loading
        if "submodule" in path and path.endswith(".py"):
            return False
        # Simulate .pyi exists
        if path.endswith(".pyi"):
            return True
        # Simulate extension exists
        if ".so" in path or ".pyd" in path:
            return True
        return False
    
    mock_isfile.side_effect = side_effect_isfile

    # Mock _read to return dummy content
    with patch("compiler._read", return_value="dummy content"):
        # Mock _load_module to return True
        with patch("compiler._load_module", return_value=True) as mock_load:
            
            result = loader(pkg_path, pkg_path, link=True, level=1, toc=True)
            
            # Assertions
            assert result == "Compiled Docstring"
            assert mock_parser.parse.called
            # Verify it tried to load the module because we simulated .py missing for submodule
            # (In our side_effect, we trigger the 'else' block for extension loading)
            mock_parser_class.new.assert_called_with(True, 1, True)

@patch("compiler.Parser")
@patch("compiler.walk_packages")
def test_loader_pure_python(mock_walk, mock_parser_class):
    """Test that loader skips extension loading if a .py file is found (pure python)."""
    mock_parser = MagicMock()
    mock_parser_class.new.return_value = mock_parser
    mock_parser.compile.return_value = "Pure Py Doc"
    
    mock_walk.return_value = [("pure_pkg", "/tmp/pure_pkg")]
    
    with patch("compiler.isfile", return_value=True), \
         patch("compiler._read", return_value="content"), \
         patch("compiler._load_module") as mock_load:
        
        result = loader("/tmp/pure_pkg", "/tmp/pure_pkg", True, 1, False)
        
        assert result == "Pure Py Doc"
        # _load_module should NOT be called because .py was found
        mock_load.assert_not_called()
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock, mock_open
from os import removedir, exists, rmdir

@pytest.fixture
def cleanup_docs():
    yield
    if exists("test_docs_dir"):
        rmdir("test_docs_dir")

def test_gen_api(tmp_path, cleanup_docs):
    """
    Test gen_api function by mocking internal dependencies to avoid 
    actual filesystem/import side effects.
    """
    # Setup mock data
    root_names = {"MyModule": "my_module"}
    prefix = "test_docs_dir"
    
    # Mocking the components of gen_api:
    # 1. loader: returns a dummy docstring
    # 2. _site_path: returns a dummy path
    # 3. _write: performs no real IO
    # 4. isdir: simulates directory check
    # 5. mkdir: simulates directory creation
    
    mock_doc_content = "Generated Docstring Content"
    
    with patch("compiler.loader") as mock_loader, \
         patch("compiler._site_path") as mock_site_path, \
         patch("compiler._write") as mock_write, \
         patch("compiler.isdir") as mock_isdir, \
         patch("compiler.mkdir") as mock_mkdir, \
         patch("compiler.logger") as mock_logger:
        
        # Configuration
        mock_site_path.return_value = "/fake/path/to/site-packages"
        mock_loader.return_value = mock_doc_content
        mock_isdir.return_value = False  # Force mkdir to be called
        
        # Execute
        result = gen_api(
            root_names=root_names,
            pwd="/fake/pwd",
            prefix=prefix,
            link=True,
            level=2,
            toc=True,
            dry=False
        )
        
        # Assertions
        
        # Check if the output doc matches the expected format (Level 2 header + content)
        expected_doc = "## MyModule API\n\n" + mock_doc_content
        assert len(result) == 1
        assert result[0] == expected_doc
        
        # Verify loader was called with correct arguments
        mock_loader.assert_called_once_with("my_module", "/fake/path/to/site-packages", True, 2, True)
        
        # Verify directory creation logic
        mock_mkdir.assert_called_once_with(prefix)
        
        # Verify file writing logic
        expected_file_path = f"{prefix}/my-module-api.md"
        mock_write.assert_called_once_with(expected_file_path, expected_doc)
        
        # Verify logger calls
        mock_logger.info.assert_any_call(f"Load root: my_module (MyModule)")
        mock_logger.info.assert_any_call(f"Write file: {expected_file_path}")

def test_gen_api_dry_run(tmp_path):
    """Test gen_api with dry=True to ensure no files are written and logs are printed."""
    root_names = {"MyModule": "my_module"}
    
    with patch("compiler.loader") as mock_loader, \
         patch("compiler._site_path") as mock_site_path, \
         patch("compiler._write") as mock_write, \
         patch("compiler.isdir") as mock_isdir, \
         patch("compiler.mkdir") as mock_mkdir, \
         patch("compiler.logger") as mock_logger:
        
        mock_loader.return_value = "Some content"
        mock_isdir.return_value = True
        
        result = gen_api(root_names, prefix="test_docs_dry", dry=True)
        
        # In dry mode, _write should NOT be called
        mock_write.assert_not_called()
        
        # Logger should print the doc content instead
        # The first log is 'Load root...', subsequent logs are the content in dry mode
        found_dry_log = False
        for call in mock_logger.info.call_args_list:
            if "Some content" in str(call):
                found_dry_log = True
                break
        assert found_dry_log
        assert len(result) == 1

def test_gen_api_empty_doc(tmp_path):
    """Test gen_api when loader returns an empty string (module not found)."""
    root_names = {"EmptyModule": "empty_mod"}
    
    with patch("compiler.loader") as mock_loader, \
         patch("compiler._site_path") as mock_site_path, \
         patch("compiler.isdir") as mock_isdir, \
         patch("compiler.logger") as mock_logger:
        
        mock_loader.return_value = "   "  # Empty/Whitespace doc
        mock_isdir.return_value = True
        
        result = gen_api(root_names, prefix="test_empty")
        
        # Result should be empty because the module was skipped
        assert len(result) == 0
        mock_logger.warning.assert_called_with("'empty_mod' can not be found")
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock, mock_open
from os import removedir, exists

def test_gen_api():
    # Test data
    root_names = {"MyModule": "my_module"}
    prefix = "test_docs_dir"
    pwd = "/fake/path"
    
    # Mocking dependencies
    # We need to mock:
    # 1. logger (to prevent actual logging)
    # 2. _site_path (to control the path returned)
    # 3. loader (to control the returned docstring)
    # 4. _write (to prevent actual file writing)
    # 5. isdir (to simulate directory existence)
    # 6. mkdir (to simulate directory creation)
    # 7. sys_path (to prevent polluting actual sys.path)
    
    with patch('compiler.logger') as mock_logger, \
         patch('compiler._site_path') as mock_site_path, \
         patch('compiler.loader') as mock_loader, \
         patch('compiler._write') as mock_write, \
         patch('compiler.isdir') as mock_isdir, \
         patch('compiler.mkdir') as mock_mkdir, \
         patch('compiler.sys_path', []):
        
        # Setup mocks
        mock_site_path.return_value = "/fake/site-packages"
        mock_loader.return_value = "Parsed Content"
        mock_isdir.return_value = False  # Simulate directory does not exist
        
        # Execute function
        result = gen_api(
            root_names=root_names,
            pwd=pwd,
            prefix=prefix,
            link=True,
            level=2,
            toc=True,
            dry=False
        )
        
        # Assertions
        # 1. Check if directory creation was attempted
        mock_mkdir.assert_called_once_with(prefix)
        
        # 2. Check if loader was called with correct arguments
        mock_loader.assert_called_once_with("my_module", "/fake/site-packages", True, 2, True)
        
        # 3. Check if the output docstring is formatted correctly
        # level 2 -> '## '
        expected_doc = "## MyModule API\n\nParsed Content"
        assert result[0] == expected_doc
        
        # 4. Check if _write was called with the correct path and content
        expected_file_path = f"{prefix}/my-module-api.md"
        mock_write.assert_called_once_with(expected_file_path, expected_doc)
        
        # 5. Check if result list contains the generated doc
        assert len(result) == 1
        assert result[0] == expected_doc

    # Test Dry Run mode
    with patch('compiler.logger') as mock_logger, \
         patch('compiler._site_path') as mock_site_path, \
         patch('compiler.loader') as mock_loader, \
         patch('compiler.isdir') as mock_isdir, \
         patch('compiler.sys_path', []):
        
        mock_site_path.return_value = "/fake/site-packages"
        mock_loader.return_value = "Dry Run Content"
        mock_isdir.return_value = True
        
        result_dry = gen_api(
            root_names={"Test": "test_pkg"},
            prefix=prefix,
            dry=True
        )
        
        # In dry mode, _write should NOT be called, but logger should log the doc
        # We verify by checking if the doc was passed to logger.info
        found_dry_log = False
        for call in mock_logger.info.call_args_list:
            if "Dry Run Content" in str(call):
                found_dry_log = True
        assert found_dry_log
        assert len(result_dry) == 1

    # Test Empty/Missing Module case
    with patch('compiler.logger') as mock_logger, \
         patch('compiler._site_path') as mock_site_path, \
         patch('compiler.loader') as mock_loader, \
         patch('compiler.isdir') as mock_isdir, \
         patch('compiler.sys_path', []):
        
        mock_site_path.return_value = "/fake/site-packages"
        mock_loader.return_value = ""  # Empty string simulates failed load
        mock_isdir.return_value = True
        
        result_empty = gen_api(root_names={"Empty": "empty_pkg"}, prefix=prefix)
        
        # Should skip adding to results and log a warning
        assert len(result_empty) == 0
        mock_logger.warning.assert_called_with("'empty_pkg' can not be found")
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock, mock_open
from os.path import isdir, join

@pytest.fixture
def mock_env(tmp_path):
    """Setup a temporary directory for testing file operations."""
    prefix_dir = tmp_path / "docs_out"
    return {
        "tmp_path": tmp_path,
        "prefix": str(prefix_dir),
        "root_names": {"test_pkg": "Test Package"}
    }

def test_gen_api(mock_env):
    """
    Test gen_api functionality including directory creation, 
    module loading, and file writing.
    """
    prefix = mock_env["prefix"]
    root_names = mock_env["root_names"]
    
    # Mocking dependencies to avoid actual filesystem/import side effects
    # 1. Mock _site_path to return a fake directory
    # 2. Mock loader to return a dummy docstring
    # 3. Mock isdir to simulate prefix directory doesn't exist initially
    # 4. Mock mkdir to prevent actual directory creation
    # 5. Mock _write to prevent actual file writing
    # 6. Mock logger to avoid polluting stdout
    
    with patch("compiler.gen_api._site_path") as mock_site_path, \
         patch("compiler.loader") as mock_loader, \
         patch("compiler.isdir") as mock_isdir, \
         patch("compiler.mkdir") as mock_mkdir, \
         patch("compiler._write") as mock_write, \
         patch("compiler.logger") as mock_logger:
        
        # Setup behaviors
        mock_site_path.return_value = "/fake/site-packages"
        mock_loader.return_value = "This is a docstring content."
        mock_isdir.return_value = False  # Simulate that 'docs' dir doesn't exist
        
        # Execute function
        results = gen_api(
            root_names=root_names,
            pwd="/fake/pwd",
            prefix=prefix,
            link=True,
            level=1,
            toc=True,
            dry=False
        )
        
        # Assertions
        # Verify directory creation was attempted
        mock_mkdir.assert_called_once_with(prefix)
        
        # Verify the content of the generated doc
        expected_doc = "# Test Package API\n\nThis is a docstring content."
        assert len(results) == 1
        assert results[0] == expected_doc
        
        # Verify file writing path
        expected_file_path = join(prefix, "test-pkg-api.md")
        mock_write.assert_called_once_with(expected_file_path, expected_doc)
        
        # Verify logger was used
        assert mock_logger.info.called

def test_gen_api_dry_run(mock_env):
    """Test gen_api with dry=True to ensure no files are written."""
    root_names = {"pkg": "Pkg"}
    
    with patch("compiler.gen_api._site_path") as mock_site_path, \
         patch("compiler.loader") as mock_loader, \
         patch("compiler.isdir") as mock_isdir, \
         patch("compiler.mkdir") as mock_mkdir, \
         patch("compiler._write") as mock_write, \
         patch("compiler.logger") as mock_logger:
        
        mock_site_path.return_value = "/fake/path"
        mock_loader.return_value = "Content"
        mock_isdir.return_value = True
        
        gen_api(root_names=root_names, dry=True, prefix="docs")
        
        # In dry mode, _write should NOT be called
        mock_write.assert_not_called()
        # Logger should show the content in dry mode
        mock_logger.info.assert_any_call('=' * 12)
        # Check if content was logged (the doc content)
        # Note: doc construction includes level (default 1) -> "# Pkg API\n\nContent"
        found_log = False
        for call in mock_logger.info.call_args_list:
            if "# Pkg API" in call[0][0]:
                found_log = True
        assert found_log

def test_gen_api_empty_doc(mock_env):
    """Test gen_api when loader returns empty string (package not found)."""
    root_names = {"missing_pkg": "Missing"}
    
    with patch("compiler.gen_api._site_path") as mock_site_path, \
         patch("compiler.loader") as mock_loader, \
         patch("compiler.isdir") as mock_isdir, \
         patch("compiler.logger") as mock_logger:
        
        mock_site_path.return_value = "/fake/path"
        mock_loader.return_value = "   "  # Empty/whitespace doc
        mock_isdir.return_value = True
        
        results = gen_api(root_names=root_names, prefix="docs")
        
        # Results should be empty because doc.strip() is false
        assert len(results) == 0
        mock_logger.warning.assert_called_with("'missing_pkg' can not be found")
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock, mock_open
from os import path

def test_gen_api():
    # Test data
    root_names = {"MyModule": "my_module"}
    prefix = "test_docs"
    
    # Mocking dependencies
    # 1. Mock sys_path.append to avoid side effects on real sys.path
    # 2. Mock isdir to simulate directory existence
    # 3. Mock mkdir to avoid real directory creation
    # 4. Mock _site_path to return a dummy path
    # 5. Mock loader to return a dummy docstring
    # 6. Mock _write to avoid real file writing
    # 7. Mock logger to avoid console spam
    
    with patch("sys.path", []), \
         patch("os.path.isdir", return_value=True), \
         patch("os.mkdir") as mock_mkdir, \
         patch("compiler._site_path", return_value="/fake/path"), \
         patch("compiler.loader", return_value="Generated Docstring Content"), \
         patch("compiler._write") as mock_write, \
         patch("compiler.logger") as mock_logger:
        
        # --- Test Case 1: Standard execution (Writing files) ---
        docs = gen_api(root_names, prefix=prefix, link=True, level=1, toc=True)
        
        # Assertions for Case 1
        assert len(docs) == 1
        assert "# MyModule API\n\nGenerated Docstring Content" in docs[0]
        mock_write.assert_called_once_with(
            path.join(prefix, "my-module-api.md"),
            "# MyModule API\n\nGenerated Docstring Content"
        )
        
        # --- Test Case 2: Dry run (No file writing, just logging) ---
        mock_write.reset_mock()
        docs_dry = gen_api(root_names, prefix=prefix, dry=True)
        
        assert len(docs_dry) == 1
        mock_write.assert_not_called()
        # Check if logger was used for printing doc in dry mode
        mock_logger.info.assert_any_call('=' * 12)
        
        # --- Test Case 3: Module not found (Empty docstring) ---
        with patch("compiler.loader", return_value=""):
            docs_empty = gen_api(root_names, prefix=prefix)
            assert len(docs_empty) == 0
            mock_logger.warning.assert_any_call("'my_module' can not be found")

        # --- Test Case 4: Directory creation ---
        with patch("os.path.isdir", return_value=False):
            gen_api({"Test": "test_pkg"}, prefix="new_dir")
            mock_mkdir.assert_called_with("new_dir")
            mock_logger.info.assert_any_call("Create directory: new_dir")

    # --- Test Case 5: PWD integration ---
    with patch("sys.path.append") as mock_append_path, \
         patch("os.path.isdir", return_value=True), \
         patch("compiler.loader", return_value="doc"), \
         patch("compiler._write"), \
         patch("compiler.logger"):
        
        gen_api({"A": "a"}, pwd="/fake/pwd")
        mock_append_path.assert_called_with("/fake/pwd")
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch
from os import mkdir, remove, rmdir
from os.path import isdir, join, exists

@pytest.fixture
def temp_package_structure(tmp_path):
    """Creates a dummy package structure for testing loader."""
    pkg_dir = tmp_path / "test_pkg"
    pkg_dir.mkdir()
    
    # Create __init__.py
    init_file = pkg_dir / "__init__.py"
    init_file.write_text("def hello():\n    '''Hello docstring'''\n    pass\n")
    
    # Create a .pyi stub file
    stub_file = pkg_dir / "module.pyi"
    stub_file.write_text("def func(): ...")
    
    # Create a .so/.pyd extension dummy (simulated by a file with extension)
    # Note: EXTENSION_SUFFIXES depends on platform, we'll mock the discovery
    ext_file = pkg_dir / "ext.pyd"
    ext_file.write_text("dummy content")

    yield tmp_path

@patch('compiler.Parser')
@patch('compiler.walk_packages')
@patch('compiler.isfile')
@patch('compiler.ext_suffixes', ['.pyd']) # Mocking suffix for predictability
def test_loader(mock_ext, mock_isfile, mock_walk, mock_parser_class, temp_package_structure):
    # Setup Mock Parser
    mock_parser = MagicMock()
    mock_parser.compile.return_value = "Compiled Docstring"
    mock_parser_class.new.return_value = mock_parser
    
    # Setup Mock walk_packages to return our dummy package
    pkg_path = str(temp_package_structure / "test_pkg")
    mock_walk.return_value = [("test_pkg.module", pkg_path + "/module")]
    
    # Mock isfile to return True for our specific files
    def side_effect_isfile(path):
        return True
    mock_isfile.side_effect = side_effect_isfile

    # Mock _read to return content
    with patch('compiler._read', return_value="dummy content"):
        # Mock _load_module to return True
        with patch('compiler._load_module', return_value=True) as mock_load:
            
            # Execute loader
            result = loader(
                root="test_pkg", 
                pwd=str(temp_package_structure), 
                link=True, 
                level=1, 
                toc=True
            )

            # Assertions
            assert result == "Compiled Docstring"
            mock_parser_class.new.assert_called_once_with(True, 1, True)
            
            # Check if parser.parse was called (for the .pyi file)
            assert mock_parser.parse.called
            
            # Check if extension loading was attempted
            assert mock_load.called

def test_loader_pure_python_skips_extension_loading(temp_package_structure):
    """Test that if a .py file exists, it doesn't attempt to load extensions."""
    pkg_dir = temp_package_structure / "pure_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("pass")
    
    with patch('compiler.walk_packages') as mock_walk:
        mock_walk.return_value = [("pure_pkg", str(pkg_dir))]
        with patch('compiler.Parser.new') as mock_parser_new:
            mock_parser = MagicMock()
            mock_parser_new.return_value = mock_parser
            
            with patch('compiler._load_module') as mock_load:
                loader("pure_pkg", str(temp_package_structure), True, 1, False)
                
                # _load_module should NOT be called because pure_py becomes True
                mock_load.assert_not_called()
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
import os
import shutil
import tempfile
from unittest.mock import patch

def test_walk_packages():
    # Create a temporary directory for testing
    tmp_dir = tempfile.mkdtemp()
    try:
        # Setup a dummy package structure
        # root/pkg/
        # root/pkg/__init__.py
        # root/pkg/module.py
        # root/pkg/subpkg/__init__.py
        # root/pkg/subpkg/submodule.pyi
        # root/pkg/other.txt (should be ignored)
        
        pkg_dir = os.path.join(tmp_dir, "pkg")
        subpkg_dir = os.path.join(pkg_dir, "subpkg")
        os.mkdir(pkg_dir)
        os.mkdir(subpkg_dir)
        
        init_py = os.path.join(pkg_dir, "__init__.py")
        mod_py = os.path.join(pkg_dir, "module.py")
        sub_init_py = os.path.join(subpkg_dir, "__init__.py")
        sub_mod_pyi = os.path.join(subpkg_dir, "submodule.pyi")
        ignored_file = os.path.join(pkg_dir, "other.txt")
        
        with open(init_py, 'w') as f: f.write("")
        with open(mod_py, 'w') as f: f.write("")
        with open(sub_init_py, 'w') as f: f.write("")
        with open(sub_mod_pyi, 'w') as f: f.write("")
        with open(ignored_file, 'w') as f: f.write("")

        # Test Case 1: Walking a specific package name
        # We expect to find:
        # 1. pkg (from __init__.py)
        # 2. pkg.module (from module.py)
        # 3. pkg.subpkg (from subpkg/__init__.py)
        # 4. pkg.subpkg.submodule (from submodule.pyi)
        
        results = list(walk_packages("pkg", tmp_dir))
        
        # Flatten names and paths for easier assertion
        found_names = [name for name, path in results]
        
        assert "pkg" in found_names
        assert "pkg.module" in found_names
        assert "pkg.subpkg" in found_names
        assert "pkg.subpkg.submodule" in found_names
        assert "pkg.other" not in found_names
        
        # Verify that the paths returned are absolute and correct
        for name, path in results:
            assert os.path.isabs(path)
            # Ensure the path actually exists
            assert os.path.exists(path)

        # Test Case 2: Walking a non-existent package
        results_empty = list(walk_packages("nonexistent", tmp_dir))
        assert len(results_empty) == 0

        # Test Case 3: Verify PEP561_SUFFIX handling
        # Create a stub directory: pkg.subpkg-stubs
        stub_dir = os.path.join(pkg_dir, "subpkg-stubs")
        os.mkdir(stub_dir)
        stub_file = os.path.join(stub_dir, "__init__.pyi")
        with open(stub_file, 'w') as f: f.write("")
        
        results_stubs = list(walk_packages("pkg", tmp_dir))
        found_names_stubs = [name for name, path in results_stubs]
        # The logic removes PEP561_SUFFIX, so 'subpkg-stubs' should appear as 'subpkg'
        assert "pkg.subpkg" in found_names_stubs

    finally:
        # Cleanup
        shutil.rmtree(tmp_dir)
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock, mock_open
from os import removedir, exists, rmdir

@pytest.fixture
def cleanup_docs():
    yield
    if exists("test_docs_dir"):
        rmdir("test_docs_dir")

def test_gen_api(tmp_path, cleanup_docs):
    """
    Test gen_api function by mocking the heavy lifting (loader, site_path, and file IO).
    """
    # Setup mock data
    root_names = {"my_package": "My Package"}
    prefix = "test_docs_dir"
    
    # Mock content to be returned by loader
    mock_doc_content = "## Module docstring\nclass MyClass: pass"
    
    # Mocking the internal dependencies
    # We mock:
    # 1. _site_path to return a dummy path
    # 2. loader to return our mock_doc_content
    # 3. _write to avoid actual disk writes of the content, but we allow it to 
    #    create the file structure so isfile/isdir checks pass.
    # 4. logger to prevent cluttering test output
    
    with patch("compiler._site_path") as mock_site_path, \
         patch("compiler.loader") as mock_loader, \
         patch("compiler._write") as mock_write, \
         patch("compiler.logger") as mock_logger, \
         patch("compiler.mkdir") as mock_mkdir, \
         patch("compiler.isdir", return_value=True):
        
        mock_site_path.return_value = "/fake/path"
        mock_loader.return_value = mock_doc_content
        
        # Execute gen_api
        # We use a real prefix name but mock the directory checks
        result = gen_api(
            root_names=root_names,
            pwd=None,
            prefix=prefix,
            link=True,
            level=1,
            toc=False,
            dry=False
        )
        
        # Assertions
        assert len(result) == 1
        # The function prepends '#' * level + title + API
        expected_doc = "# My Package API\n\n" + mock_doc_content
        assert result[0] == expected_doc
        
        # Verify loader was called with correct args
        mock_loader.assert_called_once_with("my_package", "/fake/path", True, 1, False)
        
        # Verify _write was called with the correct filename and content
        expected_filename = f"{prefix}/my-package-api.md"
        mock_write.assert_called_once_with(expected_filename, expected_doc)

def test_gen_api_dry_run(tmp_path):
    """Test gen_api with dry=True to ensure it logs instead of writing."""
    root_names = {"pkg": "Pkg"}
    
    with patch("compiler._site_path", return_value="/path"), \
         patch("compiler.loader", return_value="content"), \
         patch("compiler._write") as mock_write, \
         patch("compiler.logger") as mock_logger, \
         patch("compiler.isdir", return_value=True):
        
        result = gen_api(root_names, prefix="dry_test", dry=True)
        
        # Verify no write occurred
        mock_write.assert_not_called()
        # Verify logger was used to print the doc (as per implementation)
        # The implementation calls logger.info('=' * 12) and logger.info(doc)
        assert any("content" in str(call) for call in mock_logger.info.call_args_list)
        assert len(result) == 1

def test_gen_api_empty_doc(tmp_path):
    """Test gen_api when the loader returns an empty string."""
    root_names = {"empty_pkg": "Empty"}
    
    with patch("compiler._site_path", return_value="/path"), \
         patch("compiler.loader", return_value="   "), \
         patch("compiler.logger") as mock_logger, \
         patch("compiler.isdir", return_value=True):
        
        result = gen_api(root_names, prefix="empty_test")
        
        # Result should be empty because doc.strip() was empty
        assert len(result) == 0
        # Check if warning was logged
        mock_logger.warning.assert_called_with("'empty_pkg' can not be found")
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch
import os

def test_loader():
    """
    Test the loader function by mocking the Parser and the walk_packages iterator.
    """
    # 1. Setup Mock Parser
    mock_parser = MagicMock()
    mock_parser.compile.return_value = "Compiled Documentation"
    
    # 2. Define Mock Data for walk_packages
    # We simulate one package found in the directory
    mock_packages = [
        ("my_package", "/fake/path/my_package")
    ]
    
    # 3. Mocking external dependencies
    # We need to mock:
    # - walk_packages: to control which packages are 'discovered'
    # - isfile: to simulate presence of .py, .pyi, or extension files
    # - _read: to simulate reading file content
    # - _load_module: to simulate the loading of extension modules
    # - logger: to prevent actual logging output during test
    
    with patch('__main__.walk_packages', return_value=mock_packages), \
         patch('os.path.isfile') as mock_isfile, \
         patch('__main__._read') as mock_read, \
         patch('__main__._load_module') as mock_load_module, \
         patch('__main__.Parser.new', return_value=mock_parser), \
         patch('__main__.logger') as mock_logger:
        
        # Scenario A: Pure Python package (has .py, no extensions)
        # Setup: .py exists, .pyi does not, extension suffixes do not exist
        mock_isfile.side_effect = lambda p: p.endswith(".py")
        mock_read.return_value = "def hello(): pass"
        
        result_pure = loader("my_package", "/fake/path", True, 1, True)
        
        assert result_pure == "Compiled Documentation"
        mock_parser.parse.assert_called()
        # In pure py, _load_module should NOT be called
        mock_load_module.assert_not_called()

        # Scenario B: Extension module (has .pyi, needs loading)
        # Reset mocks
        mock_parser.reset_mock()
        mock_load_module.reset_mock()
        
        # Setup: .pyi exists, but .py does not. Extension .so/.pyd exists.
        # We simulate .pyi exists and the extension file exists.
        def isfile_side_effect(p):
            # Return True for the .pyi stub and the extension suffix
            return p.endswith(".pyi") or p.endswith(".pyd") or p.endswith(".so")
        
        mock_isfile.side_effect = isfile_side_effect
        mock_load_module.return_value = True
        
        result_ext = loader("my_package", "/fake/path", True, 1, True)
        
        assert result_ext == "Compiled Documentation"
        # Verify _load_module was attempted because pure_py was False
        assert mock_load_module.called
        
        # Scenario C: Package found but no files exist
        mock_parser.reset_mock()
        mock_isfile.return_value = False
        
        result_empty = loader("my_package", "/fake/path", True, 1, True)
        
        # If no files are found, parser.parse is never called
        mock_parser.parse.assert_not_called()
        assert result_empty == "Compiled Documentation"
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock, mock_open
import os

def test_gen_api():
    # Setup input data
    root_names = {"MyModule": "my_module", "OtherPkg": "other_pkg"}
    prefix = "test_docs"
    
    # Mocking dependencies
    # 1. Mock logger to prevent actual logging output during tests
    # 2. Mock isdir to simulate directory existence
    # 3. Mock mkdir to prevent actual directory creation
    # 4. Mock _site_path to return a controlled path
    # 5. Mock loader to return controlled documentation string
    # 6. Mock _write to prevent actual file writing
    # 7. Mock sys_path.append/pop to prevent polluting environment
    
    with patch("compiler.logger") as mock_logger, \
         patch("compiler.isdir", return_value=True), \
         patch("compiler.mkdir") as mock_mkdir, \
         patch("compiler._site_path", return_value="/fake/path"), \
         patch("compiler.loader") as mock_loader, \
         patch("compiler._write") as mock_write, \
         patch("compiler.sys_path", ["/original/path"]) as mock_sys_path:
        
        # Define what the loader returns for each package
        def loader_side_effect(name, path, link, level, toc):
            if name == "my_module":
                return "Content for my_module"
            if name == "other_pkg":
                return "Content for other_pkg"
            return ""

        mock_loader.side_effect = loader_side_effect

        # --- Test Case 1: Standard execution (Writing files) ---
        results = gen_api(
            root_names=root_names,
            pwd="/fake/pwd",
            prefix=prefix,
            link=True,
            level=2,
            toc=True,
            dry=False
        )

        # Assertions for Case 1
        assert len(results) == 2
        assert "#" * 2 + " MyModule API\n\nContent for my_module" in results
        assert "#" * 2 + " OtherPkg API\n\nContent for other_pkg" in results
        
        # Check if _write was called with correct paths
        expected_path_1 = os.path.join(prefix, "my-module-api.md")
        expected_path_2 = os.path.join(prefix, "other-pkg-api.md")
        mock_write.assert_any_call(expected_path_1, pytest.any)
        mock_write.assert_any_call(expected_path_2, pytest.any)
        
        # Check if pwd was appended to sys_path
        assert "/fake/pwd" in mock_sys_path

        # --- Test Case 2: Dry run (No files written, logs output) ---
        mock_write.reset_mock()
        results_dry = gen_api(
            root_names={"my_module": "my_module"},
            prefix=prefix,
            dry=True
        )
        
        assert len(results_dry) == 1
        mock_write.assert_not_called()
        # Check if logger was used to print the doc in dry mode
        mock_logger.info.assert_any_call('=' * 12)

        # --- Test Case 3: Directory creation ---
        with patch("compiler.isdir", return_value=False):
            gen_api({"a": "a"}, prefix="new_dir")
            mock_mkdir.assert_called_with("new_dir")

        # --- Test Case 4: Handling missing packages (Empty docstring) ---
        with patch("compiler.loader", return_value="   "): # Empty/whitespace doc
            results_empty = gen_api({"missing": "missing"}, prefix=prefix)
            assert len(results_empty) == 0
            mock_logger.warning.assert_any_call("'missing' can not be found")

```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch
from os import mkdir, remove, rmdir
from os.path import isdir, join, exists

@pytest.fixture
def setup_test_env(tmp_path):
    """Sets up a temporary directory structure for testing loader."""
    pkg_dir = tmp_path / "test_pkg"
    pkg_dir.mkdir()
    
    # Create __init__.py
    init_file = pkg_dir / "__init__.py"
    init_file.write_text("# Init docstring", encoding="utf-8")
    
    # Create a sub-module with a stub file
    sub_dir = pkg_dir / "submodule"
    sub_dir.mkdir()
    stub_file = sub_dir / "__init__.pyi"
    stub_file.write_text("'''Stub docstring'''", encoding="utf-8")
    
    # Create a module with an extension (simulated by .py)
    ext_dir = pkg_dir / "ext_mod"
    ext_dir.mkdir()
    ext_py = ext_dir / "__init__.py"
    ext_py.write_text("'''Extension docstring'''", encoding="utf-8")

    yield {
        "root": str(pkg_dir),
        "pkg_name": "test_pkg",
        "sub_name": "test_pkg.submodule",
        "ext_name": "test_pkg.ext_mod"
    }

    # Cleanup is handled by tmp_path fixture

def test_loader(setup_test_env):
    """Test the loader function with mocked Parser and walk_packages."""
    env = setup_test_env
    
    # Mock Parser
    mock_parser = MagicMock()
    mock_parser.compile.return_value = "Compiled Documentation"
    mock_parser.load_docstring = MagicMock()
    
    # Mock Parser.new to return our mock_parser
    with patch('pyslvs.parser.Parser.new', return_value=mock_parser), \
         patch('pyslvs.compiler._read', return_value="'''Mock Content'''"), \
         patch('pyslvs.compiler.walk_packages') as mock_walk, \
         patch('pyslvs.compiler.isfile', return_value=True), \
         patch('pyslvs.compiler._load_module', return_value=True):
        
        # Define what walk_packages yields
        # Format: (name, path)
        mock_walk.return_value = [
            (env["pkg_name"], join(env["root"], "__init__.py")),
            (env["sub_name"], join(env["root"], "submodule", "__init__.pyi")),
            (env["ext_name"], join(env["root"], "ext_mod", "__init__.py"))
        ]

        # Execute loader
        result = loader(
            root=env["root"],
            pwd=env["root"],
            link=True,
            level=1,
            toc=True
        )

        # Assertions
        assert result == "Compiled Documentation"
        assert mock_parser.parse.called
        # Check if it attempted to parse the modules found in walk_packages
        assert mock_parser.parse.call_count >= 1
        
        # Verify that the parser.compile() was called to return the final result
        mock_parser.compile.assert_called_once()

def test_loader_pure_python_skip(setup_test_env):
    """Test that loader skips extension loading if a pure .py file is found."""
    env = setup_test_env
    
    mock_parser = MagicMock()
    mock_parser.compile.return_value = "Pure Python Result"
    
    with patch('pyslvs.parser.Parser.new', return_value=mock_parser), \
         patch('pyslvs.compiler.walk_packages') as mock_walk, \
         patch('pyslvs.compiler._read', return_value="content"), \
         patch('pyslvs.compiler.isfile', return_value=True), \
         patch('pyslvs.compiler._load_module') as mock_load:
        
        # Case: Module has both .py and .pyi, but .py exists (pure_py = True)
        mock_walk.return_value = [
            (env["pkg_name"], join(env["root"], "__init__.py"))
        ]
        
        # We need to simulate that .py exists so pure_py becomes True
        # and the loop for EXTENSION_SUFFIXES is skipped.
        # isfile is patched to return True for the .py file.
        
        loader(env["root"], env["root"], True, 1, False)
        
        # _load_module should NOT be called because pure_py = True
        mock_load.assert_not_called()
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch, mock_open
from os import path as os_path

def test_loader():
    """
    Unit test for the loader function.
    Tests the orchestration of walking packages, parsing files, 
    and loading extension modules.
    """
    # Setup mock data
    root_dir = "/fake/root"
    pwd_dir = "/fake/pwd"
    link_param = True
    level_param = 1
    toc_param = True
    
    # Mocking Parser.new to return a mock Parser object
    mock_parser = MagicMock()
    mock_parser.compile.return_value = "Compiled Docstring"
    
    # Mocking walk_packages to return a specific package and path
    # (name, path)
    package_name = "my_package"
    package_path = "/fake/pwd/my_package"
    mock_packages = [(package_name, package_path)]

    # Mocking file existence and reading
    # We will simulate:
    # 1. my_package.pyi exists (stub)
    # 2. my_package.pyd (extension) exists
    # 3. my_package.py does NOT exist (so pure_py remains False)
    
    # We need to mock isfile, _read, _load_module, and walk_packages
    with patch('__main__.Parser.new', return_value=mock_parser), \
         patch('__main__.walk_packages', return_value=mock_packages), \
         patch('__main__.isfile') as mock_isfile, \
         patch('__main__._read') as mock_read, \
         patch('__main__._load_module') as mock_load_module, \
         patch('__main__.logger') as mock_logger:

        # Define behavior for isfile
        # .pyi exists, .py does not, .pyd (extension) exists
        def isfile_side_effect(p):
            if p.endswith('.pyi'):
                return True
            if p.endswith('.py'):
                return False
            if p.endswith('.pyd') or p.endswith('.so'):
                return True
            return False
        
        mock_isfile.side_effect = isfile_side_effect
        mock_read.return_value = "stub content"
        mock_load_module.return_value = True

        # Execute the function
        result = loader(root_dir, pwd_dir, link_param, level_param, toc_param)

        # Assertions
        # 1. Check if Parser was initialized correctly
        from __main__ import Parser
        Parser.new.assert_called_once_with(link_param, level_param, toc_param)

        # 2. Check if parser.parse was called with the content of the .pyi file
        mock_parser.parse.assert_called_with(package_name, "stub content")

        # 3. Check if _load_module was called for the extension
        # Since .py was False, it should attempt to load an extension
        assert mock_load_module.called
        
        # 4. Check if the final result is the compiled docstring
        assert result == "Compiled Docstring"

        # 5. Verify that it didn't skip the loading logic (pure_py check)
        # If pure_py was True, it would 'continue'. 
        # Since we mocked .py to be False, it proceeds to extension loading.
        assert mock_logger.debug.called

def test_loader_pure_python_skip():
    """
    Tests that if a .py file is found, the loader skips extension loading.
    """
    mock_parser = MagicMock()
    mock_parser.compile.return_value = "Pure Python Doc"
    
    package_name = "pure_pkg"
    package_path = "/fake/pwd/pure_pkg"
    mock_packages = [(package_name, package_path)]

    with patch('__main__.Parser.new', return_value=mock_parser), \
         patch('__main__.walk_packages', return_value=mock_packages), \
         patch('__main__.isfile') as mock_isfile, \
         patch('__main__._read') as mock_read, \
         patch('__main__._load_module') as mock_load_module, \
         patch('__main__.logger') as mock_logger:

        # Simulate .py exists
        mock_isfile.side_effect = lambda p: p.endswith('.py')
        mock_read.return_value = "python code"

        loader("root", "pwd", True, 1, False)

        # Assert parse was called
        mock_parser.parse.assert_called()
        # Assert _load_module was NEVER called because pure_py was True
        mock_load_module.assert_not_called()
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
import os
import shutil
from unittest.mock import patch, MagicMock

def test_gen_api(tmp_path):
    """
    Tests the gen_api function by mocking the heavy lifting (loader and site_path)
    to verify the orchestration logic: directory creation, file writing, and return values.
    """
    # Setup temporary directory for documentation output
    prefix_dir = tmp_path / "docs_output"
    root_names = {"my_module": "My Module Title", "other_pkg": "Other Package"}
    
    # Mocking the dependencies
    # 1. _site_path: return a dummy path
    # 2. loader: return a dummy docstring content
    # 3. _write: we let it run to check if files are actually created
    # 4. isdir: to control the directory creation logic
    
    mock_doc_content = "Generated Docstring Content"
    
    with patch('os.path.isdir', return_value=True), \
         patch('os.mkdir') as mock_mkdir, \
         patch('compiler. _site_path', return_value=str(tmp_path)), \
         patch('compiler.loader', return_value=mock_doc_content), \
         patch('compiler._write', wraps=lambda p, d: (open(p, 'w', encoding='utf-8').write(d))) as mock_write, \
         patch('compiler.logger') as mock_logger:
        
        # Test Case 1: Standard execution (Write to disk)
        # We use the actual prefix_dir path for the test
        prefix_str = str(prefix_dir)
        
        # Execute
        results = gen_api(
            root_names=root_names,
            pwd=str(tmp_path),
            prefix=prefix_str,
            link=True,
            level=2,
            toc=True,
            dry=False
        )
        
        # Assertions for Case 1
        assert len(results) == 2
        assert '#' * 2 + " My Module Title API\n\n" + mock_doc_content in results[0]
        
        # Verify files were "written" (since we used wraps)
        expected_file_1 = os.path.join(prefix_str, "my-module-api.md")
        expected_file_2 = os.path.join(prefix_str, "other-pkg-api.md")
        
        assert os.path.exists(expected_file_1)
        assert os.path.exists(expected_file_2)
        
        with open(expected_file_1, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "My Module Title API" in content

        # Test Case 2: Dry run (No files written, prints to logger)
        results_dry = gen_api(
            root_names=root_names,
            pwd=str(tmp_path),
            prefix=prefix_str,
            dry=True
        )
        
        assert len(results_dry) == 2
        # In dry run, the content should still be returned in the list
        assert mock_doc_content in results_dry[0]
        # Verify logger was called for dry run output
        mock_logger.info.assert_any_call('=' * 12)

        # Test Case 3: Module not found (Loader returns empty string)
        with patch('compiler.loader', return_value=""), \
             patch('compiler.logger') as mock_log_empty:
            
            results_empty = gen_api(
                root_names={"missing_pkg": "Missing"},
                prefix=prefix_str,
                dry=True
            )
            assert len(results_empty) == 0
            mock_log_empty.warning.assert_called_with("'missing_pkg' can not be found")

    # Cleanup
    if prefix_dir.exists():
        shutil.rmtree(prefix_dir)
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch
from os import mkdir, remove, rmdir
from os.path import isdir, join, exists

@pytest.fixture
def temp_package_structure(tmp_path):
    """Creates a dummy package structure for testing loader."""
    pkg_dir = tmp_path / "test_pkg"
    pkg_dir.mkdir()
    
    # Create __init__.py
    init_file = pkg_dir / "__init__.py"
    init_file.write_text("__version__ = '1.0.0'", encoding='utf-8')
    
    # Create a submodule with a stub
    sub_dir = pkg_dir / "submodule"
    sub_dir.mkdir()
    stub_file = sub_dir / "__init__.pyi"
    stub_file.write_text('"""Stub docstring"""\ndef func(): pass', encoding='utf-8')
    
    # Create a real submodule (to test extension loading/pure py logic)
    real_sub_dir = pkg_dir / "real_sub"
    real_sub_dir.mkdir()
    real_file = real_sub_dir / "__init__.py"
    real_file.write_text('"""Real docstring"""\ndef func(): pass', encoding='utf-8')

    yield tmp_path

def test_loader(temp_package_structure):
    """Test the loader function with mocked Parser and filesystem."""
    root = str(temp_package_structure)
    
    # Mock Parser to track calls
    mock_parser = MagicMock()
    mock_parser.compile.return_value = "Compiled Documentation"
    
    # Mock Parser.new to return our mock_parser
    with patch('compiler.Parser.new', return_value=mock_parser), \
         patch('compiler.walk_packages') as mock_walk, \
         patch('compiler.isfile') as mock_isfile, \
         patch('compiler._read') as mock_read, \
         patch('compiler._load_module') as mock_load_mod:
        
        # Setup walk_packages to return our dummy package
        # Format: (name, path)
        mock_walk.return_value = [
            ('test_pkg.submodule', join(root, 'test_pkg/submodule')),
            ('test_pkg.real_sub', join(root, 'test_pkg/real_sub'))
        ]
        
        # Setup isfile: 
        # 1. For submodule: .pyi exists, .py does not.
        # 2. For real_sub: .py exists.
        def isfile_side_effect(path):
            if 'submodule' in path and path.endswith('.pyi'):
                return True
            if 'real_sub' in path and path.endswith('.py'):
                return True
            return False
        mock_isfile.side_effect = isfile_side_effect
        
        # Setup _read to return content
        mock_read.return_value = "dummy content"
        
        # Setup _load_module behavior
        # For submodule (which has .pyi but no .py), it should try to load extension
        mock_load_mod.return_value = True

        # Execute loader
        result = loader('test_pkg', root, link=True, level=1, toc=True)

        # Assertions
        assert result == "Compiled Documentation"
        assert mock_parser.parse.called
        # Verify that it attempted to parse the stub
        mock_parser.parse.assert_any_call('test_pkg.submodule', "dummy content")
        # Verify that it attempted to load the extension for the submodule
        mock_load_mod.assert_called()

def test_loader_pure_python_skips_extension_loading(temp_package_structure):
    """Test that if a module is pure .py, it skips extension loading."""
    root = str(temp_package_structure)
    
    mock_parser = MagicMock()
    mock_parser.compile.return_value = "Pure Py Result"
    
    with patch('compiler.Parser.new', return_value=mock_parser), \
         patch('compiler.walk_packages') as mock_walk, \
         patch('compiler.isfile') as mock_isfile, \
         patch('compiler._read') as mock_read, \
         patch('compiler._load_module') as mock_load_mod:
        
        mock_walk.return_value = [('test_pkg.real_sub', join(root, 'test_pkg/real_sub'))]
        
        # .py exists, but .pyi does not
        def isfile_side_effect(path):
            return path.endswith('.py')
        mock_isfile.side_effect = isfile_side_effect
        mock_read.return_value = "content"

        result = loader('test_pkg', root, link=True, level=1, toc=True)

        assert result == "Pure Py Result"
        # _load_module should NOT be called for pure python modules
        mock_load_mod.assert_not_called()
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch, mock_open

def test_loader():
    # Setup Mock Parser
    mock_parser = MagicMock()
    mock_parser.compile.return_value = "compiled_doc_string"
    
    # Setup Mock Data for walk_packages
    # We will mock walk_packages to return a controlled list of (name, path)
    # Tuple: (package_name, directory_path)
    mock_packages = [
        ("my_package", "/tmp/test_pkg"),
        ("my_package.submodule", "/tmp/test_pkg/sub")
    ]
    
    # We need to simulate files existing on disk for the logic:
    # 1. .py file exists (pure_py = True)
    # 2. .pyi file exists (extension module logic)
    # 3. .so/.pyd (extension suffix) exists
    
    with patch('compiler.walk_packages', return_value=mock_packages), \
         patch('compiler.Parser.new', return_value=mock_parser), \
         patch('compiler.isfile') as mock_isfile, \
         patch('compiler.is_file', side_effect=mock_isfile), \
         patch('compiler.logger') as mock_logger, \
         patch('compiler._read') as mock_read, \
         patch('compiler._load_module') as mock_load_module:
        
        # Scenario 1: A pure Python package (has .py, no extension needed)
        # For 'my_package', let's say .py exists
        def isfile_side_effect(path):
            if path == "/tmp/test_pkg/init.py" or path == "/tmp/test_pkg/sub/init.py":
                return True
            return False
        
        mock_isfile.side_effect = isfile_side_effect
        mock_read.return_value = "content"
        
        # Execute loader
        result = loader("root", "/tmp/test_pkg", True, 1, False)
        
        # Assertions for Scenario 1
        assert result == "compiled_doc_string"
        # Verify parser.parse was called for the .py files
        assert mock_parser.parse.called
        # Verify it didn't try to load extensions because pure_py was True
        assert not mock_load_module.called

        # Scenario 2: A Stub/Extension package (has .pyi, but NO .py)
        # Reset mocks
        mock_parser.reset_mock()
        mock_load_module.reset_mock()
        
        # For 'my_package.submodule', let's say only .pyi exists
        def isfile_side_effect_ext(path):
            # Only allow .pyi for the submodule
            if path == "/tmp/test_pkg/sub/init.pyi":
                return True
            # Allow the extension suffix (e.g., .so) to exist for the submodule
            if path.endswith(".so") or path.endswith(".pyd"):
                return True
            return False
            
        mock_isfile.side_effect = isfile_side_effect_ext
        mock_load_module.return_value = True
        
        # Execute loader
        result = loader("root", "/tmp/test_pkg", True, 1, False)
        
        # Assertions for Scenario 2
        assert result == "compiled_doc_string"
        # Verify _load_module was called because pure_py was False
        assert mock_load_module.called
        # Verify logger was notified about extension loading
        mock_logger.debug.assert_any_call("loading extension module for fully documented:")

    # Scenario 3: Package found but no extension module found on platform
    with patch('compiler.walk_packages', return_value=[("ext_pkg", "/tmp/ext")]), \
         patch('compiler.Parser.new', return_value=mock_parser), \
         patch('compiler.isfile', return_value=True), \
         patch('compiler._read', return_value=""), \
         patch('compiler._load_module', return_value=False), \
         patch('compiler.logger') as mock_logger:
        
        # Force pure_py to be False by only having .pyi exist
        # (In our mock, isfile returns True for everything, so we must ensure .py is not in the logic)
        # We'll use a custom side effect to ensure .py is False but .pyi is True
        def isfile_no_py(path):
            return path.endswith(".pyi") or path.endswith(".so")
            
        with patch('compiler.isfile', side_effect=isfile_no_py):
            loader("root", "/tmp/ext", True, 1, False)
            mock_logger.warning.assert_any_call("no module for ext_pkg in this platform")
```


