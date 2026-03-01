####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_check_file():
    import tempfile
    from pathlib import Path
    from io import StringIO
    from isort import Config

    # Test 1: Check file with correctly sorted imports
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("import os\nimport sys\n")
        tmp_path = tmp.name
    
    try:
        result = check_file(tmp_path, show_diff=False)
        assert result is True
    finally:
        Path(tmp_path).unlink()

    # Test 2: Check file with incorrectly sorted imports
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("import sys\nimport os\n")
        tmp_path = tmp.name
    
    try:
        result = check_file(tmp_path, show_diff=False)
        assert result is False
    finally:
        Path(tmp_path).unlink()

    # Test 3: Check file with show_diff=True
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("import sys\nimport os\n")
        tmp_path = tmp.name
    
    try:
        output = StringIO()
        result = check_file(tmp_path, show_diff=output)
        assert result is False
        output.seek(0)
        diff_output = output.read()
        assert len(diff_output) > 0
    finally:
        Path(tmp_path).unlink()

    # Test 4: Check file with custom config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("import sys\nimport os\n")
        tmp_path = tmp.name
    
    try:
        config = Config(force_sort_within_sections=True)
        result = check_file(tmp_path, config=config, show_diff=False)
        assert result is False
    finally:
        Path(tmp_path).unlink()

    # Test 5: Check file with skip comment
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("# isort: skip_file\nimport sys\nimport os\n")
        tmp_path = tmp.name
    
    try:
        result = check_file(tmp_path, disregard_skip=False, show_diff=False)
        assert result is True
    finally:
        Path(tmp_path).unlink()

    # Test 6: Check file with extension parameter
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("import sys\nimport os\n")
        tmp_path = tmp.name
    
    try:
        result = check_file(tmp_path, extension='py', show_diff=False)
        assert result is False
    finally:
        Path(tmp_path).unlink()

    # Test 7: Check file with file_path parameter
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("import sys\nimport os\n")
        tmp_path = tmp.name
    
    try:
        result = check_file(tmp_path, file_path=Path(tmp_path), show_diff=False)
        assert result is False
    finally:
        Path(tmp_path).unlink()

    # Test 8: Check non-existent file (should raise FileNotFoundError)
    try:
        check_file("non_existent_file.py", show_diff=False)
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass

    # Test 9: Check file with config_trie parameter
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("import sys\nimport os\n")
        tmp_path = tmp.name
    
    try:
        config_trie = None
        result = check_file(tmp_path, config_trie=config_trie, show_diff=False)
        assert result is False
    finally:
        Path(tmp_path).unlink()

    # Test 10: Check empty file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("")
        tmp_path = tmp.name
    
    try:
        result = check_file(tmp_path, show_diff=False)
        assert result is True
    finally:
        Path(tmp_path).unlink()


# LLM-generated content at query #2
#--------------------------

```python
def test_find_imports_in_paths():
    import tempfile
    import os
    from pathlib import Path
    from isort import Config
    from isort.api import find_imports_in_paths
    
    # Create temporary directory structure
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create test Python files with imports
        file1 = tmpdir_path / "module1.py"
        file1.write_text("import os\nimport sys\nfrom collections import defaultdict\n")
        
        file2 = tmpdir_path / "module2.py"
        file2.write_text("from typing import List, Dict\nimport json\n")
        
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        file3 = subdir / "module3.py"
        file3.write_text("import pytest\nfrom unittest.mock import Mock\n")
        
        # Test 1: Find all imports in paths
        paths = [tmpdir_path]
        imports = list(find_imports_in_paths(paths, config=Config()))
        
        assert len(imports) == 7
        import_modules = {imp.module for imp in imports}
        expected_modules = {"os", "sys", "collections", "typing", "json", "pytest", "unittest.mock"}
        assert import_modules == expected_modules
        
        # Test 2: Find unique imports only
        file4 = tmpdir_path / "module4.py"
        file4.write_text("import os\nimport sys\n")  # Duplicate imports
        
        imports_unique = list(find_imports_in_paths([tmpdir_path], config=Config(), unique=True))
        
        # Should still have 7 unique imports (os and sys appear multiple times but only counted once)
        assert len(imports_unique) == 7
        
        # Test 3: Test with empty paths
        empty_imports = list(find_imports_in_paths([], config=Config()))
        assert len(empty_imports) == 0
        
        # Test 4: Test with non-existent path (should handle gracefully)
        non_existent = list(find_imports_in_paths(["/non/existent/path"], config=Config()))
        assert len(non_existent) == 0
        
        # Test 5: Test with config modifications
        config = Config(skip=["pytest"])
        imports_with_skip = list(find_imports_in_paths([tmpdir_path], config=config))
        
        import_modules_skipped = {imp.module for imp in imports_with_skip}
        assert "pytest" not in import_modules_skipped
        assert len(imports_with_skip) == 6
        
        # Test 6: Test top_only parameter
        file5 = tmpdir_path / "module5.py"
        file5.write_text("import top_level\n\ndef function():\n    import nested\n")
        
        imports_top_only = list(find_imports_in_paths([tmpdir_path], config=Config(), top_only=True))
        
        # Should find top_level but not nested
        top_modules = {imp.module for imp in imports_top_only if imp.module in {"top_level", "nested"}}
        assert "top_level" in top_modules
        assert "nested" not in top_modules
        
        # Test 7: Test with ImportKey uniqueness
        from isort.identify import ImportKey
        
        # Create file with same module imported multiple ways
        file6 = tmpdir_path / "module6.py"
        file6.write_text("import os\nimport os as operating_system\nfrom os import path\n")
        
        # Test unique by module only
        imports_module_unique = list(find_imports_in_paths(
            [tmpdir_path], 
            config=Config(), 
            unique=ImportKey.MODULE
        ))
        
        # Should count os only once despite multiple import statements
        os_imports = [imp for imp in imports_module_unique if imp.module == "os"]
        assert len(os_imports) == 1
        
        # Test 8: Test with multiple path inputs
        paths_list = [tmpdir_path, subdir]
        imports_multi_path = list(find_imports_in_paths(paths_list, config=Config()))
        
        # Should find imports from all specified paths
        assert len(imports_multi_path) >= 7  # At least the original 7 plus new ones
        
        # Test 9: Test with file path instead of directory
        file_imports = list(find_imports_in_paths([file1], config=Config()))
        assert len(file_imports) == 3
        file_modules = {imp.module for imp in file_imports}
        assert file_modules == {"os", "sys", "collections"}


# LLM-generated content at query #3
#--------------------------

```python
def test_find_imports_in_file():
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, mock_open
    import isort.io
    from isort import Config
    from isort.identify import Import

    # Test 1: Normal imports in file
    test_code = """import os
import sys
from collections import defaultdict
import numpy as np
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_code)
        temp_path = Path(f.name)
    
    try:
        imports = list(find_imports_in_file(temp_path))
        assert len(imports) == 4
        assert all(isinstance(imp, Import) for imp in imports)
        assert imports[0].module == 'os'
        assert imports[1].module == 'sys'
        assert imports[2].module == 'collections'
        assert imports[3].module == 'numpy'
    finally:
        temp_path.unlink()

    # Test 2: File with unique=True
    test_code = """import os
import sys
import os  # Duplicate
from os.path import join
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_code)
        temp_path = Path(f.name)
    
    try:
        imports = list(find_imports_in_file(temp_path, unique=True))
        assert len(imports) == 3  # os, sys, os.path
        modules = [imp.module for imp in imports]
        assert 'os' in modules
        assert 'sys' in modules
        assert 'os.path' in modules
    finally:
        temp_path.unlink()

    # Test 3: File with top_only=True
    test_code = """import os

def my_func():
    import sys
    return sys.version

class MyClass:
    from datetime import datetime
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_code)
        temp_path = Path(f.name)
    
    try:
        imports = list(find_imports_in_file(temp_path, top_only=True))
        assert len(imports) == 1
        assert imports[0].module == 'os'
    finally:
        temp_path.unlink()

    # Test 4: File with custom config
    test_code = """import os
import sys
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_code)
        temp_path = Path(f.name)
    
    try:
        custom_config = Config(known_third_party=['os'])
        imports = list(find_imports_in_file(temp_path, config=custom_config))
        assert len(imports) == 2
    finally:
        temp_path.unlink()

    # Test 5: Non-existent file (should warn but not crash)
    with patch('isort.api.warn') as mock_warn:
        imports = list(find_imports_in_file('/non/existent/file.py'))
        assert len(imports) == 0
        assert mock_warn.called

    # Test 6: File with OSError during reading
    mock_file = mock_open()
    mock_file.return_value.read.side_effect = OSError("Permission denied")
    
    with patch('isort.api.io.File.read') as mock_file_read:
        mock_file_read.return_value.__enter__.return_value.stream = mock_file.return_value
        with patch('isort.api.warn') as mock_warn:
            imports = list(find_imports_in_file('/some/file.py'))
            assert len(imports) == 0
            assert mock_warn.called

    # Test 7: File with mixed imports and code
    test_code = '''"""Module docstring."""
import os
print("Hello")
from sys import version
import math
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_code)
        temp_path = Path(f.name)
    
    try:
        imports = list(find_imports_in_file(temp_path))
        assert len(imports) == 3
        assert imports[0].module == 'os'
        assert imports[1].module == 'sys'
        assert imports[2].module == 'math'
    finally:
        temp_path.unlink()

    # Test 8: Empty file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("")
        temp_path = Path(f.name)
    
    try:
        imports = list(find_imports_in_file(temp_path))
        assert len(imports) == 0
    finally:
        temp_path.unlink()


# LLM-generated content at query #4
#--------------------------

```python
def test_find_imports_in_file():
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    
    # Test 1: Basic imports from a file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("import os\nimport sys\nfrom collections import defaultdict\n")
        tmp_path = tmp.name
    
    try:
        imports = list(find_imports_in_file(tmp_path))
        assert len(imports) == 3
        modules = {imp.module for imp in imports}
        assert "os" in modules
        assert "sys" in modules
        assert "collections" in modules
    finally:
        Path(tmp_path).unlink()
    
    # Test 2: File with unique=True
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("import os\nimport os\nimport sys\nimport sys\n")
        tmp_path = tmp.name
    
    try:
        imports = list(find_imports_in_file(tmp_path, unique=True))
        assert len(imports) == 2
        modules = {imp.module for imp in imports}
        assert "os" in modules
        assert "sys" in modules
    finally:
        Path(tmp_path).unlink()
    
    # Test 3: File with top_only=True
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("import os\n\ndef foo():\n    import sys\n")
        tmp_path = tmp.name
    
    try:
        imports = list(find_imports_in_file(tmp_path, top_only=True))
        assert len(imports) == 1
        assert imports[0].module == "os"
    finally:
        Path(tmp_path).unlink()
    
    # Test 4: File with config modifications
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("import os\nimport sys\n")
        tmp_path = tmp.name
    
    try:
        imports = list(find_imports_in_file(tmp_path, config_kwargs={"verbose": True}))
        assert len(imports) == 2
    finally:
        Path(tmp_path).unlink()
    
    # Test 5: File path provided separately
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("import os\n")
        tmp_path = tmp.name
    
    try:
        file_path = Path(tmp_path)
        imports = list(find_imports_in_file(tmp_path, file_path=file_path))
        assert len(imports) == 1
        assert imports[0].module == "os"
    finally:
        Path(tmp_path).unlink()
    
    # Test 6: Non-existent file (should warn but not crash)
    with patch('builtins.warn') as mock_warn:
        imports = list(find_imports_in_file("/non/existent/file.py"))
        assert len(imports) == 0
        assert mock_warn.called
    
    # Test 7: File with different import types
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("import os\nfrom sys import argv\nimport numpy as np\n")
        tmp_path = tmp.name
    
    try:
        imports = list(find_imports_in_file(tmp_path))
        assert len(imports) == 3
        import_types = {type(imp).__name__ for imp in imports}
        assert "Import" in import_types
    finally:
        Path(tmp_path).unlink()
    
    # Test 8: Empty file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("")
        tmp_path = tmp.name
    
    try:
        imports = list(find_imports_in_file(tmp_path))
        assert len(imports) == 0
    finally:
        Path(tmp_path).unlink()


# LLM-generated content at query #5
#--------------------------

```python
def test_find_imports_in_file():
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    
    # Test 1: Basic imports from a file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("import os\nimport sys\nfrom collections import defaultdict\n")
        tmp_path = Path(tmp.name)
    
    try:
        imports = list(find_imports_in_file(tmp_path))
        assert len(imports) == 3
        module_names = {imp.module for imp in imports}
        assert "os" in module_names
        assert "sys" in module_names
        assert "collections" in module_names
    finally:
        tmp_path.unlink()
    
    # Test 2: With unique=True
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("import os\nimport sys\nimport os\nfrom sys import path\n")
        tmp_path = Path(tmp.name)
    
    try:
        imports = list(find_imports_in_file(tmp_path, unique=True))
        assert len(imports) == 2  # Only unique modules
        module_names = {imp.module for imp in imports}
        assert "os" in module_names
        assert "sys" in module_names
    finally:
        tmp_path.unlink()
    
    # Test 3: With top_only=True
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("import os\n\ndef func():\n    import sys\n")
        tmp_path = Path(tmp.name)
    
    try:
        imports = list(find_imports_in_file(tmp_path, top_only=True))
        assert len(imports) == 1
        assert imports[0].module == "os"
    finally:
        tmp_path.unlink()
    
    # Test 4: File not found/OSError
    non_existent = Path("/non/existent/file.py")
    imports = list(find_imports_in_file(non_existent))
    assert len(imports) == 0
    
    # Test 5: With custom config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("import os\nimport sys\n")
        tmp_path = Path(tmp.name)
    
    try:
        custom_config = Config(known_third_party=["os"])
        imports = list(find_imports_in_file(tmp_path, config=custom_config))
        assert len(imports) == 2
    finally:
        tmp_path.unlink()
    
    # Test 6: With file_path parameter
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("import os\n")
        tmp_path = Path(tmp.name)
    
    try:
        imports = list(find_imports_in_file(tmp_path, file_path=tmp_path))
        assert len(imports) == 1
        assert imports[0].module == "os"
    finally:
        tmp_path.unlink()
    
    # Test 7: With config_kwargs
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("import os\n")
        tmp_path = Path(tmp.name)
    
    try:
        imports = list(find_imports_in_file(tmp_path, profile="black"))
        assert len(imports) == 1
    finally:
        tmp_path.unlink()


# LLM-generated content at query #6
#--------------------------

```python
def test_check_file():
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from io import StringIO

    # Test 1: File with correctly sorted imports should return True
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("import os\nimport sys\n")
        tmp_path = tmp.name
    
    try:
        result = check_file(tmp_path, show_diff=False)
        assert result is True
    finally:
        Path(tmp_path).unlink()

    # Test 2: File with incorrectly sorted imports should return False
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("import sys\nimport os\n")
        tmp_path = tmp.name
    
    try:
        result = check_file(tmp_path, show_diff=False)
        assert result is False
    finally:
        Path(tmp_path).unlink()

    # Test 3: Show diff with stdout
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("import sys\nimport os\n")
        tmp_path = tmp.name
    
    try:
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = check_file(tmp_path, show_diff=True)
            assert result is False
            # Should have printed diff output
            assert mock_stdout.getvalue() != ""
    finally:
        Path(tmp_path).unlink()

    # Test 4: Show diff with custom TextIO stream
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("import sys\nimport os\n")
        tmp_path = tmp.name
    
    try:
        diff_output = StringIO()
        result = check_file(tmp_path, show_diff=diff_output)
        assert result is False
        diff_output.seek(0)
        assert diff_output.read() != ""
    finally:
        Path(tmp_path).unlink()

    # Test 5: With custom config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("import sys\nimport os\n")
        tmp_path = tmp.name
    
    try:
        custom_config = Config(profile="black")
        result = check_file(tmp_path, config=custom_config, show_diff=False)
        # Just verify it runs without error
        assert isinstance(result, bool)
    finally:
        Path(tmp_path).unlink()

    # Test 6: File with skip comment should be skipped
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("# isort: skip_file\nimport sys\nimport os\n")
        tmp_path = tmp.name
    
    try:
        result = check_file(tmp_path, disregard_skip=False, show_diff=False)
        # Should return True when skipped
        assert result is True
    finally:
        Path(tmp_path).unlink()

    # Test 7: File with skip comment but disregard_skip=True
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("# isort: skip_file\nimport sys\nimport os\n")
        tmp_path = tmp.name
    
    try:
        result = check_file(tmp_path, disregard_skip=True, show_diff=False)
        # Should check the file and return False (unsorted)
        assert result is False
    finally:
        Path(tmp_path).unlink()

    # Test 8: Non-existent file should raise FileNotFoundError
    try:
        check_file("/non/existent/file.py", show_diff=False)
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass

    # Test 9: Test with config_trie parameter
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("import os\nimport sys\n")
        tmp_path = tmp.name
    
    try:
        mock_trie = MagicMock()
        mock_trie.search.return_value = ("some/path", {})
        
        with patch('sys.stdout', new_callable=StringIO):
            result = check_file(
                tmp_path, 
                show_diff=False,
                config_trie=mock_trie
            )
            assert result is True
            mock_trie.search.assert_called_once_with(tmp_path)
    finally:
        Path(tmp_path).unlink()

    # Test 10: Empty file should return True
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("")
        tmp_path = tmp.name
    
    try:
        result = check_file(tmp_path, show_diff=False)
        assert result is True
    finally:
        Path(tmp_path).unlink()

    # Test 11: File with only comments should return True
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("# This is a comment\n# Another comment\n")
        tmp_path = tmp.name
    
    try:
        result = check_file(tmp_path, show_diff=False)
        assert result is True
    finally:
        Path(tmp_path).unlink()


# LLM-generated content at query #7
#--------------------------

```python
def test_find_imports_in_stream():
    from io import StringIO
    from isort import Config
    from isort.api import find_imports_in_stream
    from isort import identify

    # Test basic import detection
    code = "import os\nimport sys\nfrom collections import defaultdict"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream))
    assert len(imports) == 3
    assert imports[0].module == "os"
    assert imports[1].module == "sys"
    assert imports[2].module == "collections"
    assert imports[2].attribute == "defaultdict"

    # Test with unique=True
    code = "import os\nimport sys\nimport os\nimport sys"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream, unique=True))
    assert len(imports) == 2
    assert {imp.module for imp in imports} == {"os", "sys"}

    # Test with unique=ImportKey.MODULE
    code = "import os.path\nimport os\nfrom os import path"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream, unique=identify.ImportKey.MODULE))
    assert len(imports) == 1
    assert imports[0].module == "os"

    # Test with top_only=True
    code = "import os\ndef func():\n    import sys"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream, top_only=True))
    assert len(imports) == 1
    assert imports[0].module == "os"

    # Test with config
    config = Config(force_sort_within_sections=True)
    code = "import sys\nimport os"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream, config=config))
    assert len(imports) == 2

    # Test with file_path
    from pathlib import Path
    code = "import os"
    stream = StringIO(code)
    file_path = Path("test.py")
    imports = list(find_imports_in_stream(stream, file_path=file_path))
    assert len(imports) == 1
    assert imports[0].module == "os"

    # Test with _seen parameter
    code = "import os\nimport sys"
    stream = StringIO(code)
    seen = {"os"}
    imports = list(find_imports_in_stream(stream, unique=True, _seen=seen))
    assert len(imports) == 1
    assert imports[0].module == "sys"

    # Test with config_kwargs
    code = "import os"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream, force_sort_within_sections=True))
    assert len(imports) == 1

    # Test empty stream
    stream = StringIO("")
    imports = list(find_imports_in_stream(stream))
    assert len(imports) == 0

    # Test with complex imports
    code = "from module.submodule import Class1, function2 as fn"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream))
    assert len(imports) == 1
    assert imports[0].module == "module.submodule"
    assert imports[0].attribute == "Class1"

    # Test with multiple from imports
    code = "from a import b\nfrom c import d, e"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream))
    assert len(imports) == 3
    assert imports[0].module == "a"
    assert imports[0].attribute == "b"
    assert imports[1].module == "c"
    assert imports[1].attribute == "d"
    assert imports[2].module == "c"
    assert imports[2].attribute == "e"


# LLM-generated content at query #8
#--------------------------

```python
def test_find_imports_in_stream():
    from io import StringIO
    from pathlib import Path
    from isort import Config
    from isort.api import find_imports_in_stream
    from isort import identify

    # Test basic import detection
    code = "import os\nimport sys\nfrom collections import defaultdict"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream))
    assert len(imports) == 3
    assert imports[0].module == "os"
    assert imports[1].module == "sys"
    assert imports[2].module == "collections"
    assert imports[2].attribute == "defaultdict"

    # Test with unique=True
    code = "import os\nimport sys\nimport os"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream, unique=True))
    assert len(imports) == 2
    assert imports[0].module == "os"
    assert imports[1].module == "sys"

    # Test with top_only=True
    code = "import os\ndef func():\n    import sys"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream, top_only=True))
    assert len(imports) == 1
    assert imports[0].module == "os"

    # Test with file_path parameter
    code = "import os"
    stream = StringIO(code)
    file_path = Path("test.py")
    imports = list(find_imports_in_stream(stream, file_path=file_path))
    assert len(imports) == 1
    assert imports[0].module == "os"

    # Test with custom config
    code = "import os\nimport sys"
    stream = StringIO(code)
    config = Config(force_sort_within_sections=True)
    imports = list(find_imports_in_stream(stream, config=config))
    assert len(imports) == 2

    # Test with _seen parameter
    code = "import os\nimport sys"
    stream = StringIO(code)
    seen = {"os"}
    imports = list(find_imports_in_stream(stream, unique=True, _seen=seen))
    assert len(imports) == 1
    assert imports[0].module == "sys"

    # Test empty stream
    stream = StringIO("")
    imports = list(find_imports_in_stream(stream))
    assert len(imports) == 0

    # Test with from imports
    code = "from datetime import datetime, timedelta"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream))
    assert len(imports) == 1
    assert imports[0].module == "datetime"
    assert imports[0].attribute == "datetime"
    assert imports[0].names == ["datetime", "timedelta"]

    # Test with config_kwargs
    code = "import os"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream, profile="black"))
    assert len(imports) == 1


# LLM-generated content at query #9
#--------------------------

```python
def test_find_imports_in_file():
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    
    # Test 1: Basic imports found in file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("import os\nimport sys\nfrom collections import defaultdict\n")
        temp_path = Path(f.name)
    
    try:
        imports = list(find_imports_in_file(temp_path))
        assert len(imports) == 3
        modules = {imp.module for imp in imports}
        assert "os" in modules
        assert "sys" in modules
        assert "collections" in modules
    finally:
        temp_path.unlink()
    
    # Test 2: Unique imports only
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("import os\nimport os\nimport sys\nimport sys\n")
        temp_path = Path(f.name)
    
    try:
        imports = list(find_imports_in_file(temp_path, unique=True))
        assert len(imports) == 2
        modules = {imp.module for imp in imports}
        assert "os" in modules
        assert "sys" in modules
    finally:
        temp_path.unlink()
    
    # Test 3: Top only imports
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("import os\n\ndef func():\n    import sys\n")
        temp_path = Path(f.name)
    
    try:
        imports = list(find_imports_in_file(temp_path, top_only=True))
        assert len(imports) == 1
        assert imports[0].module == "os"
    finally:
        temp_path.unlink()
    
    # Test 4: File read error - should warn but not crash
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("import os\n")
        temp_path = Path(f.name)
    
    temp_path.unlink()  # Delete to cause OSError
    
    with patch('warnings.warn') as mock_warn:
        imports = list(find_imports_in_file(temp_path))
        assert len(imports) == 0
        assert mock_warn.called
    
    # Test 5: With custom config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("import os\nimport sys\n")
        temp_path = Path(f.name)
    
    try:
        config = Config(known_third_party=['os'])
        imports = list(find_imports_in_file(temp_path, config=config))
        assert len(imports) == 2
    finally:
        temp_path.unlink()
    
    # Test 6: File path parameter
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("import os\n")
        temp_path = Path(f.name)
    
    try:
        imports = list(find_imports_in_file(temp_path, file_path=temp_path))
        assert len(imports) == 1
        assert imports[0].module == "os"
    finally:
        temp_path.unlink()
    
    # Test 7: ImportKey uniqueness
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("import os as os1\nimport os as os2\nfrom os import path\n")
        temp_path = Path(f.name)
    
    try:
        # Test with ImportKey.MODULE
        imports = list(find_imports_in_file(temp_path, unique=ImportKey.MODULE))
        assert len(imports) == 2  # Only 2 unique modules: os and os.path
        
        # Test with ImportKey.ALIAS (default when unique=True)
        imports = list(find_imports_in_file(temp_path, unique=True))
        assert len(imports) == 3  # All 3 are unique statements
        
        # Test with ImportKey.PACKAGE
        imports = list(find_imports_in_file(temp_path, unique=ImportKey.PACKAGE))
        assert len(imports) == 1  # Only 'os' package
    finally:
        temp_path.unlink()


# LLM-generated content at query #10
#--------------------------

```python
def test_find_imports_in_stream():
    from io import StringIO
    from isort import Config
    from isort.identify import Import
    from isort.api import find_imports_in_stream

    # Test basic import detection
    code = "import os\nimport sys\n"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream))
    assert len(imports) == 2
    assert isinstance(imports[0], Import)
    assert imports[0].module == "os"
    assert imports[1].module == "sys"

    # Test with unique=True
    code = "import os\nimport sys\nimport os\n"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream, unique=True))
    assert len(imports) == 2  # Duplicate 'os' should be filtered
    modules = {imp.module for imp in imports}
    assert modules == {"os", "sys"}

    # Test with ImportKey.MODULE
    code = "import os\nimport os.path\n"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream, unique="module"))
    assert len(imports) == 1  # Both are module 'os'
    assert imports[0].module == "os"

    # Test with ImportKey.PACKAGE
    code = "import os.path\nimport os\n"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream, unique="package"))
    assert len(imports) == 1  # Both are package 'os'
    assert imports[0].module == "os.path"

    # Test with top_only=True
    code = "import os\ndef foo():\n    import sys\n"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream, top_only=True))
    assert len(imports) == 1
    assert imports[0].module == "os"

    # Test with file_path parameter
    stream = StringIO("import os\n")
    imports = list(find_imports_in_stream(stream, file_path="test.py"))
    assert len(imports) == 1

    # Test with config parameter
    config = Config()
    stream = StringIO("import os\n")
    imports = list(find_imports_in_stream(stream, config=config))
    assert len(imports) == 1

    # Test with from imports
    code = "from collections import defaultdict\n"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream))
    assert len(imports) == 1
    assert imports[0].module == "collections"
    assert imports[0].attribute == "defaultdict"

    # Test with _seen parameter
    stream = StringIO("import os\n")
    seen = {"os"}
    imports = list(find_imports_in_stream(stream, _seen=seen))
    assert len(imports) == 0  # Already in seen set

    # Test empty stream
    stream = StringIO("")
    imports = list(find_imports_in_stream(stream))
    assert len(imports) == 0

    # Test mixed imports
    code = "import os\nfrom sys import path\nimport numpy as np\n"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream))
    assert len(imports) == 3
    assert imports[0].module == "os"
    assert imports[1].module == "sys"
    assert imports[2].module == "numpy"
    assert imports[2].alias == "np"


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_sort_stream():
    from io import StringIO
    from pathlib import Path
    from unittest.mock import patch, mock_open
    import pytest
    
    from isort.api import sort_stream, Config, DEFAULT_CONFIG
    from isort.exceptions import FileSkipSetting, FileSkipComment, ExistingSyntaxErrors, IntroducedSyntaxErrors
    
    # Test 1: Basic sorting functionality
    input_stream = StringIO("import b\nimport a\n")
    output_stream = StringIO()
    result = sort_stream(input_stream, output_stream)
    output_stream.seek(0)
    assert output_stream.read() == "import a\nimport b\n"
    assert result is True
    
    # Test 2: No changes needed
    input_stream = StringIO("import a\nimport b\n")
    output_stream = StringIO()
    result = sort_stream(input_stream, output_stream)
    output_stream.seek(0)
    assert output_stream.read() == "import a\nimport b\n"
    assert result is False
    
    # Test 3: With file_path and extension
    input_stream = StringIO("import b\nimport a\n")
    output_stream = StringIO()
    file_path = Path("test.py")
    result = sort_stream(input_stream, output_stream, file_path=file_path)
    output_stream.seek(0)
    assert output_stream.read() == "import a\nimport b\n"
    assert result is True
    
    # Test 4: With custom config
    input_stream = StringIO("import b\nimport a\n")
    output_stream = StringIO()
    config = Config(force_sort_within_sections=True)
    result = sort_stream(input_stream, output_stream, config=config)
    output_stream.seek(0)
    assert output_stream.read() == "import a\nimport b\n"
    assert result is True
    
    # Test 5: Show diff to stdout
    input_stream = StringIO("import b\nimport a\n")
    output_stream = StringIO()
    with patch('isort.api.show_unified_diff') as mock_show_diff:
        result = sort_stream(input_stream, output_stream, show_diff=True)
        assert mock_show_diff.called
    
    # Test 6: Show diff to TextIO stream
    input_stream = StringIO("import b\nimport a\n")
    output_stream = StringIO()
    diff_stream = StringIO()
    with patch('isort.api.show_unified_diff') as mock_show_diff:
        result = sort_stream(input_stream, output_stream, show_diff=diff_stream)
        assert mock_show_diff.called
    
    # Test 7: File skip setting
    input_stream = StringIO("import b\nimport a\n")
    output_stream = StringIO()
    file_path = Path("test.py")
    config = Config(skip=["test.py"])
    
    with pytest.raises(FileSkipSetting):
        sort_stream(input_stream, output_stream, config=config, file_path=file_path)
    
    # Test 8: Disregard skip
    input_stream = StringIO("import b\nimport a\n")
    output_stream = StringIO()
    file_path = Path("test.py")
    config = Config(skip=["test.py"])
    
    result = sort_stream(input_stream, output_stream, config=config, file_path=file_path, disregard_skip=True)
    output_stream.seek(0)
    assert output_stream.read() == "import a\nimport b\n"
    assert result is True
    
    # Test 9: File skip comment
    input_stream = StringIO("# isort: skip_file\nimport b\nimport a\n")
    output_stream = StringIO()
    
    with pytest.raises(FileSkipComment):
        sort_stream(input_stream, output_stream)
    
    # Test 10: Atomic mode with valid syntax
    input_stream = StringIO("import b\nimport a\n")
    output_stream = StringIO()
    config = Config(atomic=True)
    
    result = sort_stream(input_stream, output_stream, config=config)
    output_stream.seek(0)
    assert output_stream.read() == "import a\nimport b\n"
    assert result is True
    
    # Test 11: Atomic mode with existing syntax errors
    input_stream = StringIO("import b\nimport a\ninvalid syntax here")
    output_stream = StringIO()
    config = Config(atomic=True)
    
    with pytest.raises(ExistingSyntaxErrors):
        sort_stream(input_stream, output_stream, config=config)
    
    # Test 12: Atomic mode with introduced syntax errors
    # Mock core.process to return invalid code
    input_stream = StringIO("import b\nimport a\n")
    output_stream = StringIO()
    config = Config(atomic=True)
    
    with patch('isort.api.core.process') as mock_process:
        mock_process.return_value = True
        mock_process.side_effect = lambda *args, **kwargs: args[1].write("invalid python code")
        
        with pytest.raises(IntroducedSyntaxErrors):
            sort_stream(input_stream, output_stream, config=config)
    
    # Test 13: Cython extension with syntax errors
    input_stream = StringIO("import b\nimport a\ninvalid syntax")
    output_stream = StringIO()
    config = Config(atomic=True, verbose=True)
    
    with patch('warnings.warn') as mock_warn:
        result = sort_stream(input_stream, output_stream, extension="pyx", config=config)
        assert mock_warn.called
    
    # Test 14: Raise on skip disabled
    input_stream = StringIO("# isort: skip_file\nimport b\nimport a\n")
    output_stream = StringIO()
    
    result = sort_stream(input_stream, output_stream, raise_on_skip=False)
    assert result is False
    
    # Test 15: Complex import sorting
    input_stream = StringIO("from z import b, a\nimport y\nimport x\n")
    output_stream = StringIO()
    
    result = sort_stream(input_stream, output_stream)
    output_stream.seek(0)
    output = output_stream.read()
    assert "import x" in output
    assert "import y" in output
    assert "from z import a, b" in output
    assert result is True
    
    # Test 16: Empty input
    input_stream = StringIO("")
    output_stream = StringIO()
    
    result = sort_stream(input_stream, output_stream)
    output_stream.seek(0)
    assert output_stream.read() == ""
    assert result is False
    
    # Test 17: Only whitespace
    input_stream = StringIO("\n\n  \n\t\n")
    output_stream = StringIO()
    
    result = sort_stream(input_stream, output_stream)
    output_stream.seek(0)
    assert output_stream.read() == "\n\n  \n\t\n"
    assert result is False


# LLM-generated content at query #2
#--------------------------

```python
def test_find_imports_in_paths():
    import tempfile
    from pathlib import Path
    from isort import Config
    from isort.api import find_imports_in_paths
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        file1 = tmpdir_path / "module1.py"
        file1.write_text("import os\nimport sys\nfrom collections import defaultdict")
        
        file2 = tmpdir_path / "module2.py"
        file2.write_text("import json\nimport os\nfrom typing import List, Dict")
        
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        file3 = subdir / "module3.py"
        file3.write_text("import math\nimport random")
        
        paths = [tmpdir_path]
        
        imports = list(find_imports_in_paths(paths, config=Config()))
        assert len(imports) == 7
        
        module_names = {imp.module for imp in imports}
        expected_modules = {"os", "sys", "collections", "json", "typing", "math", "random"}
        assert module_names == expected_modules
        
        imports_unique = list(find_imports_in_paths(paths, config=Config(), unique=True))
        assert len(imports_unique) == 7
        
        imports_unique_module = list(find_imports_in_paths(paths, config=Config(), unique="module"))
        assert len(imports_unique_module) == 7
        
        single_path = [file1]
        imports_single = list(find_imports_in_paths(single_path, config=Config()))
        assert len(imports_single) == 3
        
        empty_paths = []
        imports_empty = list(find_imports_in_paths(empty_paths, config=Config()))
        assert len(imports_empty) == 0
        
        non_existent_paths = [tmpdir_path / "nonexistent"]
        imports_nonexistent = list(find_imports_in_paths(non_existent_paths, config=Config()))
        assert len(imports_nonexistent) == 0


# LLM-generated content at query #3
#--------------------------

```python
def test_find_imports_in_stream():
    from io import StringIO
    from isort import Config
    from isort.identify import Import

    # Test basic import detection
    code = "import os\nimport sys\nfrom collections import defaultdict"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream))
    assert len(imports) == 3
    assert all(isinstance(imp, Import) for imp in imports)
    assert imports[0].module == "os"
    assert imports[1].module == "sys"
    assert imports[2].module == "collections"
    assert imports[2].attribute == "defaultdict"

    # Test with unique=True
    code = "import os\nimport sys\nimport os\nfrom sys import path"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream, unique=True))
    assert len(imports) == 3  # Only unique imports: os, sys, sys.path
    modules = [imp.module for imp in imports]
    assert "os" in modules
    assert "sys" in modules

    # Test with unique=ImportKey.MODULE
    code = "import os.path\nimport os\nfrom os import path"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream, unique=ImportKey.MODULE))
    assert len(imports) == 1  # Only unique module: os
    assert imports[0].module == "os"

    # Test with top_only=True
    code = "import os\ndef func():\n    import sys"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream, top_only=True))
    assert len(imports) == 1
    assert imports[0].module == "os"

    # Test with file_path parameter
    stream = StringIO("import os")
    imports = list(find_imports_in_stream(stream, file_path="test.py"))
    assert len(imports) == 1

    # Test with config parameter
    config = Config()
    stream = StringIO("import os")
    imports = list(find_imports_in_stream(stream, config=config))
    assert len(imports) == 1

    # Test with config_kwargs
    stream = StringIO("import os")
    imports = list(find_imports_in_stream(stream, profile="black"))
    assert len(imports) == 1

    # Test with _seen parameter (internal use)
    stream = StringIO("import os\nimport sys")
    seen = {"os"}
    imports = list(find_imports_in_stream(stream, _seen=seen))
    assert len(imports) == 2  # Both imports should still be returned
    assert "os" in seen
    assert "sys" in seen

    # Test empty stream
    stream = StringIO("")
    imports = list(find_imports_in_stream(stream))
    assert len(imports) == 0

    # Test with complex imports
    code = "from module.submodule import Class1, Class2\nimport pandas as pd"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream))
    assert len(imports) == 3  # Two imports from from statement + one import
    assert any(imp.module == "module.submodule" and imp.attribute == "Class1" for imp in imports)
    assert any(imp.module == "module.submodule" and imp.attribute == "Class2" for imp in imports)
    assert any(imp.module == "pandas" and imp.alias == "pd" for imp in imports)


# LLM-generated content at query #4
#--------------------------

```python
def test_find_imports_in_file():
    import tempfile
    from pathlib import Path
    from unittest.mock import Mock, patch
    from isort import Config
    from isort.api import find_imports_in_file
    from isort.identify import Import

    # Test 1: Basic import detection
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("import os\nimport sys\nfrom collections import defaultdict\n")
        tmp_path = Path(tmp.name)

    try:
        imports = list(find_imports_in_file(tmp_path))
        assert len(imports) == 3
        assert all(isinstance(imp, Import) for imp in imports)
        module_names = [imp.module for imp in imports]
        assert "os" in module_names
        assert "sys" in module_names
        assert "collections" in module_names
    finally:
        tmp_path.unlink()

    # Test 2: With unique=True
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("import os\nimport os\nimport sys\n")
        tmp_path = Path(tmp.name)

    try:
        imports = list(find_imports_in_file(tmp_path, unique=True))
        assert len(imports) == 2  # Only unique imports
        module_names = [imp.module for imp in imports]
        assert "os" in module_names
        assert "sys" in module_names
    finally:
        tmp_path.unlink()

    # Test 3: With top_only=True
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("import os\ndef func():\n    import sys\n")
        tmp_path = Path(tmp.name)

    try:
        imports = list(find_imports_in_file(tmp_path, top_only=True))
        assert len(imports) == 1
        assert imports[0].module == "os"
    finally:
        tmp_path.unlink()

    # Test 4: With custom config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("import os\nimport sys\n")
        tmp_path = Path(tmp.name)

    try:
        custom_config = Config(profile="black")
        imports = list(find_imports_in_file(tmp_path, config=custom_config))
        assert len(imports) == 2
    finally:
        tmp_path.unlink()

    # Test 5: File not found/OSError
    non_existent = Path("/non/existent/file.py")
    imports = list(find_imports_in_file(non_existent))
    assert len(imports) == 0  # Should handle OSError gracefully

    # Test 6: With file_path parameter
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("import os\n")
        tmp_path = Path(tmp.name)

    try:
        custom_file_path = Path("/custom/path.py")
        imports = list(find_imports_in_file(tmp_path, file_path=custom_file_path))
        assert len(imports) == 1
    finally:
        tmp_path.unlink()

    # Test 7: Empty file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("")
        tmp_path = Path(tmp.name)

    try:
        imports = list(find_imports_in_file(tmp_path))
        assert len(imports) == 0
    finally:
        tmp_path.unlink()

    # Test 8: Complex import patterns
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("""
import os.path as osp
from collections import defaultdict, OrderedDict
import sys
""")
        tmp_path = Path(tmp.name)

    try:
        imports = list(find_imports_in_file(tmp_path))
        assert len(imports) >= 3
    finally:
        tmp_path.unlink()


# LLM-generated content at query #5
#--------------------------

```python
def test_find_imports_in_paths():
    import tempfile
    from pathlib import Path
    from isort import Config
    from isort.api import find_imports_in_paths
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        file1 = tmpdir_path / "module1.py"
        file1.write_text("import os\nimport sys\nfrom collections import defaultdict")
        
        file2 = tmpdir_path / "module2.py"
        file2.write_text("import json\nimport os\nfrom typing import List, Dict")
        
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        file3 = subdir / "module3.py"
        file3.write_text("import math\nimport random")
        
        paths = [tmpdir_path]
        imports = list(find_imports_in_paths(paths, config=Config()))
        
        assert len(imports) == 7
        
        module_names = {imp.module for imp in imports}
        expected_modules = {"os", "sys", "collections", "json", "typing", "math", "random"}
        assert module_names == expected_modules
        
        imports_unique = list(find_imports_in_paths(paths, config=Config(), unique=True))
        assert len(imports_unique) == 7
        
        os_imports = [imp for imp in imports if imp.module == "os"]
        assert len(os_imports) == 2
        
        imports_unique_module = list(find_imports_in_paths(paths, config=Config(), unique="module"))
        assert len(imports_unique_module) == 7
        
        single_file_paths = [file1]
        single_file_imports = list(find_imports_in_paths(single_file_paths, config=Config()))
        assert len(single_file_imports) == 3
        
        empty_paths = []
        empty_imports = list(find_imports_in_paths(empty_paths, config=Config()))
        assert len(empty_imports) == 0
        
        non_existent_paths = [tmpdir_path / "nonexistent"]
        non_existent_imports = list(find_imports_in_paths(non_existent_paths, config=Config()))
        assert len(non_existent_imports) == 0


# LLM-generated content at query #6
#--------------------------

```python
def test_sort_stream():
    import io
    from pathlib import Path
    from unittest.mock import patch, mock_open, MagicMock
    
    from isort.api import sort_stream, Config, DEFAULT_CONFIG
    from isort.exceptions import FileSkipSetting, FileSkipComment
    
    # Test 1: Basic sorting functionality
    input_code = "import b\nimport a\n"
    expected_output = "import a\nimport b\n"
    
    input_stream = io.StringIO(input_code)
    output_stream = io.StringIO()
    
    result = sort_stream(input_stream, output_stream)
    output_stream.seek(0)
    assert output_stream.read() == expected_output
    assert result is True
    
    # Test 2: No changes needed
    input_code = "import a\nimport b\n"
    
    input_stream = io.StringIO(input_code)
    output_stream = io.StringIO()
    
    result = sort_stream(input_stream, output_stream)
    output_stream.seek(0)
    assert output_stream.read() == input_code
    assert result is False
    
    # Test 3: With file_path and extension
    input_code = "import b\nimport a\n"
    expected_output = "import a\nimport b\n"
    
    input_stream = io.StringIO(input_code)
    output_stream = io.StringIO()
    file_path = Path("test.py")
    
    result = sort_stream(input_stream, output_stream, file_path=file_path)
    output_stream.seek(0)
    assert output_stream.read() == expected_output
    
    # Test 4: With custom config
    input_code = "import b\nimport a\n"
    
    input_stream = io.StringIO(input_code)
    output_stream = io.StringIO()
    config = Config(force_sort_within_sections=True)
    
    result = sort_stream(input_stream, output_stream, config=config)
    output_stream.seek(0)
    assert "import a" in output_stream.read()
    
    # Test 5: Show diff to stdout
    input_code = "import b\nimport a\n"
    
    input_stream = io.StringIO(input_code)
    output_stream = io.StringIO()
    
    with patch("isort.api.show_unified_diff") as mock_diff:
        result = sort_stream(input_stream, output_stream, show_diff=True)
        assert mock_diff.called
    
    # Test 6: Show diff to TextIO stream
    input_code = "import b\nimport a\n"
    
    input_stream = io.StringIO(input_code)
    output_stream = io.StringIO()
    diff_stream = io.StringIO()
    
    with patch("isort.api.show_unified_diff") as mock_diff:
        result = sort_stream(input_stream, output_stream, show_diff=diff_stream)
        assert mock_diff.called
    
    # Test 7: File skipped by setting
    input_code = "import b\nimport a\n"
    
    input_stream = io.StringIO(input_code)
    output_stream = io.StringIO()
    file_path = Path("skipped.py")
    
    config = Config(skip=["skipped.py"])
    
    try:
        sort_stream(input_stream, output_stream, config=config, file_path=file_path)
        assert False, "Should have raised FileSkipSetting"
    except FileSkipSetting:
        pass
    
    # Test 8: File skipped by setting with disregard_skip
    input_code = "import b\nimport a\n"
    expected_output = "import a\nimport b\n"
    
    input_stream = io.StringIO(input_code)
    output_stream = io.StringIO()
    file_path = Path("skipped.py")
    
    config = Config(skip=["skipped.py"])
    
    result = sort_stream(
        input_stream, output_stream, config=config, 
        file_path=file_path, disregard_skip=True
    )
    output_stream.seek(0)
    assert output_stream.read() == expected_output
    
    # Test 9: File skipped by comment
    input_code = "# isort: skip_file\nimport b\nimport a\n"
    
    input_stream = io.StringIO(input_code)
    output_stream = io.StringIO()
    
    try:
        sort_stream(input_stream, output_stream)
        assert False, "Should have raised FileSkipComment"
    except FileSkipComment:
        pass
    
    # Test 10: Atomic mode with valid code
    input_code = "import b\nimport a\n"
    expected_output = "import a\nimport b\n"
    
    input_stream = io.StringIO(input_code)
    output_stream = io.StringIO()
    
    config = Config(atomic=True)
    result = sort_stream(input_stream, output_stream, config=config)
    output_stream.seek(0)
    assert output_stream.read() == expected_output
    
    # Test 11: Atomic mode with syntax errors (non-Cython)
    input_code = "import b\nimport a\nx = \n"  # Syntax error
    
    input_stream = io.StringIO(input_code)
    output_stream = io.StringIO()
    
    config = Config(atomic=True)
    
    try:
        sort_stream(input_stream, output_stream, config=config)
        assert False, "Should have raised ExistingSyntaxErrors"
    except Exception as e:
        assert "ExistingSyntaxErrors" in str(type(e).__name__)
    
    # Test 12: Cython extension with syntax errors
    input_code = "import b\nimport a\nx = \n"  # Syntax error
    
    input_stream = io.StringIO(input_code)
    output_stream = io.StringIO()
    
    config = Config(atomic=True)
    
    with patch("isort.api.warn") as mock_warn:
        result = sort_stream(
            input_stream, output_stream, config=config, extension="pyx"
        )
        assert mock_warn.called
    
    # Test 13: Raise on skip disabled
    input_code = "# isort: skip_file\nimport b\nimport a\n"
    
    input_stream = io.StringIO(input_code)
    output_stream = io.StringIO()
    
    result = sort_stream(input_stream, output_stream, raise_on_skip=False)
    assert result is False
    
    # Test 14: Config kwargs
    input_code = "import b\nimport a\n"
    expected_output = "import a\nimport b\n"
    
    input_stream = io.StringIO(input_code)
    output_stream = io.StringIO()
    
    result = sort_stream(input_stream, output_stream, force_sort_within_sections=True)
    output_stream.seek(0)
    assert output_stream.read() == expected_output
    
    # Test 15: Complex import structure
    input_code = """from z import b, a
import y
import x
from a import c, b
"""
    expected_sorted = """import x
import y
from a import b, c
from z import a, b
"""
    
    input_stream = io.StringIO(input_code)
    output_stream = io.StringIO()
    
    result = sort_stream(input_stream, output_stream)
    output_stream.seek(0)
    output = output_stream.read()
    assert "import x" in output
    assert "import y" in output
    assert output.count("import") == 4


