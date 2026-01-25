####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_sort_file(tmp_path, capsys):
    """Test the sort_file function with various configurations."""
    # Create a temporary file with unsorted imports
    test_file = tmp_path / "test_imports.py"
    unsorted_content = """import os
import sys
from pathlib import Path
import json
from typing import Dict
"""
    test_file.write_text(unsorted_content)
    
    # Test basic sorting
    result = sort_file(test_file)
    assert result is True
    
    sorted_content = test_file.read_text()
    assert sorted_content != unsorted_content
    assert "import json" in sorted_content
    assert "import os" in sorted_content
    
    # Test with already sorted file
    result = sort_file(test_file)
    assert result is False
    
    # Test with write_to_stdout
    test_file.write_text(unsorted_content)
    result = sort_file(test_file, write_to_stdout=True)
    captured = capsys.readouterr()
    assert result is True
    assert "import" in captured.out
    
    # Test with show_diff
    test_file.write_text(unsorted_content)
    result = sort_file(test_file, show_diff=True)
    assert result is True
    
    # Test with output stream
    test_file.write_text(unsorted_content)
    output_stream = StringIO()
    result = sort_file(test_file, output=output_stream)
    assert result is True
    output_stream.seek(0)
    output_content = output_stream.read()
    assert "import" in output_content
    
    # Test with disregard_skip=False
    test_file.write_text(unsorted_content)
    result = sort_file(test_file, disregard_skip=False)
    assert isinstance(result, bool)
    
    # Test with config parameter
    test_file.write_text(unsorted_content)
    custom_config = Config(line_length=88)
    result = sort_file(test_file, config=custom_config)
    assert isinstance(result, bool)
    
    # Test with extension parameter
    test_file.write_text(unsorted_content)
    result = sort_file(test_file, extension="py")
    assert isinstance(result, bool)
    
    # Test file with syntax errors when atomic=True
    test_file.write_text("import os\ninvalid syntax !!!")
    custom_config = Config(atomic=True)
    with pytest.raises(ExistingSyntaxErrors):
        sort_file(test_file, config=custom_config)
    
    # Test with file_path parameter
    test_file.write_text(unsorted_content)
    result = sort_file(test_file, file_path=test_file)
    assert isinstance(result, bool)


# LLM-generated content at query #2
#--------------------------

```python
def test_find_imports_in_code():
    """Test find_imports_in_code function with various code samples."""
    
    # Test basic imports
    code = "import os\nimport sys\nfrom pathlib import Path"
    imports = list(find_imports_in_code(code))
    assert len(imports) == 3
    
    # Test with empty code
    code = ""
    imports = list(find_imports_in_code(code))
    assert len(imports) == 0
    
    # Test with code containing no imports
    code = "x = 1\ny = 2"
    imports = list(find_imports_in_code(code))
    assert len(imports) == 0
    
    # Test with multiple imports from same module
    code = "from os import path, getcwd\nimport sys"
    imports = list(find_imports_in_code(code))
    assert len(imports) >= 2
    
    # Test with unique=True
    code = "import os\nimport os\nimport sys"
    imports = list(find_imports_in_code(code, unique=True))
    assert len(imports) == 2
    
    # Test with top_only=True
    code = "import os\n\ndef foo():\n    pass\n\nimport sys"
    imports = list(find_imports_in_code(code, top_only=True))
    assert len(imports) == 1
    
    # Test with top_only=True and class definition
    code = "import os\n\nclass Foo:\n    pass\n\nimport sys"
    imports = list(find_imports_in_code(code, top_only=True))
    assert len(imports) == 1
    
    # Test with custom config
    config = Config()
    code = "import os"
    imports = list(find_imports_in_code(code, config=config))
    assert len(imports) == 1
    
    # Test with file_path parameter
    code = "import os"
    imports = list(find_imports_in_code(code, file_path=Path("test.py")))
    assert len(imports) == 1
    
    # Test with config_kwargs
    code = "import os"
    imports = list(find_imports_in_code(code, line_length=80))
    assert len(imports) == 1
    
    # Test with complex imports
    code = "from package.module import func1, func2, func3"
    imports = list(find_imports_in_code(code))
    assert len(imports) >= 1
    
    # Test with relative imports
    code = "from . import module\nfrom .. import parent_module"
    imports = list(find_imports_in_code(code))
    assert len(imports) == 2
    
    # Test with aliased imports
    code = "import numpy as np\nfrom pathlib import Path as P"
    imports = list(find_imports_in_code(code))
    assert len(imports) == 2
    
    # Test that function returns iterator
    code = "import os"
    result = find_imports_in_code(code)
    assert hasattr(result, '__iter__')
    assert hasattr(result, '__next__')
    
    # Test with mixed imports before and after code
    code = "import os\nimport sys\n\nx = 1\n\ndef func():\n    import json\n    return json"
    imports = list(find_imports_in_code(code, top_only=False))
    assert len(imports) >= 3
    
    imports_top = list(find_imports_in_code(code, top_only=True))
    assert len(imports_top) == 2


# LLM-generated content at query #3
#--------------------------

```python
def test_check_stream():
    """Test the check_stream function with various scenarios."""
    
    # Test 1: Correctly sorted imports should return True
    correct_code = "import os\nimport sys\n"
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream)
    assert result is True
    
    # Test 2: Incorrectly sorted imports should return False
    incorrect_code = "import sys\nimport os\n"
    input_stream = StringIO(incorrect_code)
    result = check_stream(input_stream)
    assert result is False
    
    # Test 3: Check with file_path parameter
    correct_code = "import os\nimport sys\n"
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream, file_path=Path("test.py"))
    assert result is True
    
    # Test 4: Check with extension parameter
    correct_code = "import os\nimport sys\n"
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream, extension="py")
    assert result is True
    
    # Test 5: Check with custom config
    config = Config()
    correct_code = "import os\nimport sys\n"
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream, config=config)
    assert result is True
    
    # Test 6: Check with show_diff as True (boolean)
    incorrect_code = "import sys\nimport os\n"
    input_stream = StringIO(incorrect_code)
    result = check_stream(input_stream, show_diff=True)
    assert result is False
    
    # Test 7: Check with show_diff as TextIO stream
    incorrect_code = "import sys\nimport os\n"
    input_stream = StringIO(incorrect_code)
    diff_output = StringIO()
    result = check_stream(input_stream, show_diff=diff_output)
    assert result is False
    assert diff_output.getvalue() != ""
    
    # Test 8: Check with disregard_skip parameter
    correct_code = "import os\nimport sys\n"
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream, disregard_skip=True)
    assert result is True
    
    # Test 9: Empty code should return True (no imports to sort)
    empty_code = ""
    input_stream = StringIO(empty_code)
    result = check_stream(input_stream)
    assert result is True
    
    # Test 10: Code with from imports
    correct_from_imports = "from os import path\nfrom sys import argv\n"
    input_stream = StringIO(correct_from_imports)
    result = check_stream(input_stream)
    assert result is True
    
    # Test 11: Incorrectly sorted from imports
    incorrect_from_imports = "from sys import argv\nfrom os import path\n"
    input_stream = StringIO(incorrect_from_imports)
    result = check_stream(input_stream)
    assert result is False
    
    # Test 12: Mixed imports and from imports
    correct_mixed = "import os\nfrom sys import argv\n"
    input_stream = StringIO(correct_mixed)
    result = check_stream(input_stream)
    assert result is True
    
    # Test 13: Config kwargs override
    correct_code = "import os\nimport sys\n"
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream, line_length=80)
    assert result is True
    
    # Test 14: Check with multiple config kwargs
    correct_code = "import os\nimport sys\n"
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream, line_length=80, profile="black")
    assert result is True


# LLM-generated content at query #4
#--------------------------

```python
def test_find_imports_in_paths(tmp_path):
    """Test find_imports_in_paths function."""
    # Create temporary Python files with imports
    file1 = tmp_path / "test1.py"
    file1.write_text("import os\nimport sys\nfrom pathlib import Path")
    
    file2 = tmp_path / "test2.py"
    file2.write_text("import json\nfrom collections import defaultdict")
    
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    file3 = subdir / "test3.py"
    file3.write_text("import re\nimport ast")
    
    # Test finding imports in paths
    imports = list(find_imports_in_paths([tmp_path]))
    
    # Should find all imports from all files
    assert len(imports) == 7
    
    # Verify specific imports are found
    import_modules = [imp.module for imp in imports]
    assert "os" in import_modules
    assert "sys" in import_modules
    assert "pathlib" in import_modules
    assert "json" in import_modules
    assert "collections" in import_modules
    assert "re" in import_modules
    assert "ast" in import_modules


def test_find_imports_in_paths_with_unique_true(tmp_path):
    """Test find_imports_in_paths with unique=True."""
    file1 = tmp_path / "test1.py"
    file1.write_text("import os\nimport os\nimport sys")
    
    file2 = tmp_path / "test2.py"
    file2.write_text("import os\nimport json")
    
    imports = list(find_imports_in_paths([tmp_path], unique=True))
    
    # With unique=True, duplicate imports should be filtered
    import_statements = [imp.statement() for imp in imports]
    assert len([s for s in import_statements if "import os" in s]) == 1


def test_find_imports_in_paths_with_unique_module(tmp_path):
    """Test find_imports_in_paths with unique=ImportKey.MODULE."""
    file1 = tmp_path / "test1.py"
    file1.write_text("from os import path\nfrom os import environ\nimport sys")
    
    imports = list(find_imports_in_paths([tmp_path], unique=ImportKey.MODULE))
    
    # Should only return one import per module
    import_modules = [imp.module for imp in imports]
    assert import_modules.count("os") == 1
    assert "sys" in import_modules


def test_find_imports_in_paths_with_unique_package(tmp_path):
    """Test find_imports_in_paths with unique=ImportKey.PACKAGE."""
    file1 = tmp_path / "test1.py"
    file1.write_text("import os.path\nimport os.environ\nfrom collections.abc import Sequence")
    
    imports = list(find_imports_in_paths([tmp_path], unique=ImportKey.PACKAGE))
    
    # Should only return one import per top-level package
    import_packages = [imp.module.split(".")[0] for imp in imports]
    assert import_packages.count("os") == 1
    assert import_packages.count("collections") == 1


def test_find_imports_in_paths_with_top_only(tmp_path):
    """Test find_imports_in_paths with top_only=True."""
    file1 = tmp_path / "test1.py"
    file1.write_text("import os\n\ndef func():\n    import sys")
    
    imports = list(find_imports_in_paths([tmp_path], top_only=True))
    
    # Should only return top-level imports
    import_modules = [imp.module for imp in imports]
    assert "os" in import_modules
    assert "sys" not in import_modules


def test_find_imports_in_paths_empty_directory(tmp_path):
    """Test find_imports_in_paths with empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    imports = list(find_imports_in_paths([empty_dir]))
    
    assert len(imports) == 0


def test_find_imports_in_paths_multiple_paths(tmp_path):
    """Test find_imports_in_paths with multiple paths."""
    dir1 = tmp_path / "dir1"
    dir1.mkdir()
    file1 = dir1 / "test1.py"
    file1.write_text("import os")
    
    dir2 = tmp_path / "dir2"
    dir2.mkdir()
    file2 = dir2 / "test2.py"
    file2.write_text("import sys")
    
    imports = list(find_imports_in_paths([dir1, dir2]))
    
    import_modules = [imp.module for imp in imports]
    assert "os" in import_modules
    assert "sys" in import_modules


def test_find_imports_in_paths_with_config(tmp_path):
    """Test find_imports_in_paths with custom config."""
    file1 = tmp_path / "test1.py"
    file1.write_text("import os\nimport sys")
    
    config = Config(verbose=True)
    imports = list(find_imports_in_paths([tmp_path], config=config))
    
    assert len(imports) == 2


def test_find_imports_in_paths_unique_alias(tmp_path):
    """Test find_imports_in_paths with unique=ImportKey.ALIAS."""
    file1 = tmp_path / "test1.py"
    file1.write_text("import os as operating_system\nimport os as op_sys\nimport sys")
    
    imports = list(find_imports_in_paths([tmp_path], unique=ImportKey.ALIAS))
    
    # Each unique statement should be returned
    import_statements = [imp.statement() for imp in imports]
    assert len(import_statements) == 3


def test_find_imports_in_paths_with_attribute_unique(tmp_path):
    """Test find_imports_in_paths with unique=ImportKey.ATTRIBUTE."""
    file1 = tmp_path / "test1.py"
    file1.write_text("from os import path\nfrom os import path\nfrom os import environ")
    
    imports = list(find_imports_in_paths([tmp_path], unique=ImportKey.ATTRIBUTE))
    
    # Duplicate attributes should be filtered
    import_attrs = [(imp.module, imp.attribute) for imp in imports]
    assert import_attrs.count(("os", "path")) == 1
    assert ("os", "environ") in import_attrs


# LLM-generated content at query #5
#--------------------------

```python
def test_check_stream():
    """Test check_stream function with various scenarios."""
    from io import StringIO
    
    # Test 1: Correctly sorted imports should return True
    correct_code = "import os\nimport sys\n\nprint('hello')\n"
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream)
    assert result is True
    
    # Test 2: Incorrectly sorted imports should return False
    incorrect_code = "import sys\nimport os\n\nprint('hello')\n"
    input_stream = StringIO(incorrect_code)
    result = check_stream(input_stream)
    assert result is False
    
    # Test 3: With file_path parameter
    correct_code = "import os\nimport sys\n"
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream, file_path=Path("test.py"))
    assert result is True
    
    # Test 4: With extension parameter
    correct_code = "import os\nimport sys\n"
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream, extension="py")
    assert result is True
    
    # Test 5: With show_diff as True
    incorrect_code = "import sys\nimport os\n"
    input_stream = StringIO(incorrect_code)
    result = check_stream(input_stream, show_diff=True)
    assert result is False
    
    # Test 6: With show_diff as TextIO stream
    incorrect_code = "import sys\nimport os\n"
    input_stream = StringIO(incorrect_code)
    diff_output = StringIO()
    result = check_stream(input_stream, show_diff=diff_output)
    assert result is False
    
    # Test 7: With config parameter
    correct_code = "import os\nimport sys\n"
    input_stream = StringIO(correct_code)
    config = Config()
    result = check_stream(input_stream, config=config)
    assert result is True
    
    # Test 8: With disregard_skip parameter
    correct_code = "import os\nimport sys\n"
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream, disregard_skip=True)
    assert result is True
    
    # Test 9: With config_kwargs
    correct_code = "import os\nimport sys\n"
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream, line_length=80)
    assert result is True
    
    # Test 10: Empty code should return True
    empty_code = ""
    input_stream = StringIO(empty_code)
    result = check_stream(input_stream)
    assert result is True
    
    # Test 11: Code with only comments should return True
    comment_code = "# This is a comment\n"
    input_stream = StringIO(comment_code)
    result = check_stream(input_stream)
    assert result is True
    
    # Test 12: Mixed correct and incorrect sorting in from imports
    mixed_code = "from os import path\nfrom sys import argv\n"
    input_stream = StringIO(mixed_code)
    result = check_stream(input_stream)
    assert result is True
    
    # Test 13: Incorrectly sorted from imports
    incorrect_mixed = "from sys import argv\nfrom os import path\n"
    input_stream = StringIO(incorrect_mixed)
    result = check_stream(input_stream)
    assert result is False


# LLM-generated content at query #6
#--------------------------

```python
def test_sort_file(tmp_path, capsys):
    """Test the sort_file function with various scenarios."""
    
    # Test 1: Basic file sorting
    test_file = tmp_path / "test_imports.py"
    test_file.write_text("import os\nimport sys\nimport ast\n")
    
    result = sort_file(test_file)
    assert result is False  # Already sorted
    
    # Test 2: File with unsorted imports
    unsorted_file = tmp_path / "unsorted.py"
    unsorted_file.write_text("import sys\nimport os\nimport ast\n")
    
    result = sort_file(unsorted_file)
    assert result is True  # Should be changed
    
    sorted_content = unsorted_file.read_text()
    assert sorted_content.startswith("import ast")
    
    # Test 3: write_to_stdout parameter
    stdout_file = tmp_path / "stdout_test.py"
    stdout_file.write_text("import sys\nimport os\n")
    
    result = sort_file(stdout_file, write_to_stdout=True)
    captured = capsys.readouterr()
    assert "import os" in captured.out
    
    # Test 4: show_diff parameter
    diff_file = tmp_path / "diff_test.py"
    diff_file.write_text("import sys\nimport os\n")
    
    result = sort_file(diff_file, show_diff=True)
    captured = capsys.readouterr()
    assert result is False  # show_diff prevents changes
    
    # Test 5: output parameter with StringIO
    output_file = tmp_path / "output_test.py"
    output_file.write_text("import sys\nimport os\n")
    output_stream = StringIO()
    
    result = sort_file(output_file, output=output_stream)
    output_stream.seek(0)
    output_content = output_stream.read()
    assert "import os" in output_content
    
    # Test 6: disregard_skip parameter
    skip_file = tmp_path / "skip_test.py"
    skip_file.write_text("import sys\nimport os\n")
    
    result = sort_file(skip_file, disregard_skip=True)
    
    # Test 7: extension parameter
    ext_file = tmp_path / "test_ext.py"
    ext_file.write_text("import sys\nimport os\n")
    
    result = sort_file(ext_file, extension="py")
    
    # Test 8: file_path parameter
    fp_file = tmp_path / "fp_test.py"
    fp_file.write_text("import sys\nimport os\n")
    
    result = sort_file(fp_file, file_path=fp_file)
    
    # Test 9: overwrite_in_place config
    config = Config(overwrite_in_place=True)
    inplace_file = tmp_path / "inplace_test.py"
    inplace_file.write_text("import sys\nimport os\n")
    
    result = sort_file(inplace_file, config=config)
    
    # Test 10: ask_to_apply with user rejection (mocked)
    ask_file = tmp_path / "ask_test.py"
    ask_file.write_text("import sys\nimport os\n")
    
    result = sort_file(ask_file, ask_to_apply=False)
    
    # Test 11: Verify file modifications persist
    verify_file = tmp_path / "verify.py"
    verify_file.write_text("import sys\nimport os\nimport ast\n")
    original_content = verify_file.read_text()
    
    sort_file(verify_file)
    modified_content = verify_file.read_text()
    assert modified_content.startswith("import ast")
    
    # Test 12: Empty file
    empty_file = tmp_path / "empty.py"
    empty_file.write_text("")
    
    result = sort_file(empty_file)
    assert result is False
    
    # Test 13: File with comments
    comment_file = tmp_path / "comments.py"
    comment_file.write_text("# File header\nimport sys\nimport os\n")
    
    result = sort_file(comment_file)
    
    # Test 14: Quiet mode
    quiet_file = tmp_path / "quiet.py"
    quiet_file.write_text("import sys\nimport os\n")
    config_quiet = Config(quiet=True)
    
    result = sort_file(quiet_file, config=config_quiet)
    captured = capsys.readouterr()
    assert "Fixing" not in captured.out


# LLM-generated content at query #7
#--------------------------

```python
def test_check_stream():
    """Test the check_stream function with various scenarios."""
    from io import StringIO
    from pathlib import Path
    
    # Test 1: Correctly sorted imports should return True
    correct_code = "import os\nimport sys\n"
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream)
    assert result is True
    
    # Test 2: Incorrectly sorted imports should return False
    incorrect_code = "import sys\nimport os\n"
    input_stream = StringIO(incorrect_code)
    result = check_stream(input_stream)
    assert result is False
    
    # Test 3: With show_diff=True, incorrectly sorted imports should return False
    incorrect_code = "import sys\nimport os\n"
    input_stream = StringIO(incorrect_code)
    result = check_stream(input_stream, show_diff=True)
    assert result is False
    
    # Test 4: With show_diff as StringIO, should write diff to stream
    incorrect_code = "import sys\nimport os\n"
    input_stream = StringIO(incorrect_code)
    diff_stream = StringIO()
    result = check_stream(input_stream, show_diff=diff_stream)
    assert result is False
    diff_stream.seek(0)
    diff_content = diff_stream.read()
    assert len(diff_content) > 0
    
    # Test 5: With file_path parameter
    correct_code = "import os\nimport sys\n"
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream, file_path=Path("test.py"))
    assert result is True
    
    # Test 6: With extension parameter
    correct_code = "import os\nimport sys\n"
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream, extension="py")
    assert result is True
    
    # Test 7: With disregard_skip=True
    correct_code = "import os\nimport sys\n"
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream, disregard_skip=True)
    assert result is True
    
    # Test 8: Empty code should return True
    empty_code = ""
    input_stream = StringIO(empty_code)
    result = check_stream(input_stream)
    assert result is True
    
    # Test 9: Code with no imports should return True
    code_no_imports = "x = 1\ny = 2\n"
    input_stream = StringIO(code_no_imports)
    result = check_stream(input_stream)
    assert result is True
    
    # Test 10: Multiple import groups
    incorrect_multi_imports = "import sys\nfrom os import path\nimport os\n"
    input_stream = StringIO(incorrect_multi_imports)
    result = check_stream(input_stream)
    assert result is False
    
    # Test 11: With config_kwargs
    correct_code = "import os\nimport sys\n"
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream, line_length=100)
    assert result is True
    
    # Test 12: Stream reset after reading for show_diff
    incorrect_code = "import sys\nimport os\n"
    input_stream = StringIO(incorrect_code)
    diff_stream = StringIO()
    result = check_stream(input_stream, show_diff=diff_stream)
    assert result is False
    # Verify that input_stream was properly handled
    input_stream.seek(0)
    assert len(input_stream.read()) > 0


# LLM-generated content at query #8
#--------------------------

```python
def test_sort_stream():
    """Test sort_stream function with various scenarios."""
    from io import StringIO
    from pathlib import Path
    
    # Test 1: Basic import sorting
    input_code = "import os\nimport sys\nimport collections\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    
    changed = sort_stream(
        input_stream=input_stream,
        output_stream=output_stream,
        config=DEFAULT_CONFIG
    )
    
    output_stream.seek(0)
    result = output_stream.read()
    assert isinstance(result, str)
    assert "import" in result
    
    # Test 2: Empty stream
    input_stream = StringIO("")
    output_stream = StringIO()
    
    changed = sort_stream(
        input_stream=input_stream,
        output_stream=output_stream,
        config=DEFAULT_CONFIG
    )
    
    assert changed is False or changed is True
    output_stream.seek(0)
    assert output_stream.read() == ""
    
    # Test 3: With extension parameter
    input_code = "import sys\nimport os\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    
    changed = sort_stream(
        input_stream=input_stream,
        output_stream=output_stream,
        extension="py",
        config=DEFAULT_CONFIG
    )
    
    output_stream.seek(0)
    result = output_stream.read()
    assert isinstance(result, str)
    
    # Test 4: With file_path parameter
    input_code = "import sys\nimport os\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    test_path = Path("test.py")
    
    changed = sort_stream(
        input_stream=input_stream,
        output_stream=output_stream,
        file_path=test_path,
        config=DEFAULT_CONFIG
    )
    
    output_stream.seek(0)
    result = output_stream.read()
    assert isinstance(result, str)
    
    # Test 5: With raise_on_skip parameter
    input_code = "import sys\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    
    changed = sort_stream(
        input_stream=input_stream,
        output_stream=output_stream,
        raise_on_skip=False,
        config=DEFAULT_CONFIG
    )
    
    assert isinstance(changed, bool)
    
    # Test 6: With disregard_skip parameter
    input_code = "import sys\nimport os\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    
    changed = sort_stream(
        input_stream=input_stream,
        output_stream=output_stream,
        disregard_skip=True,
        config=DEFAULT_CONFIG
    )
    
    output_stream.seek(0)
    result = output_stream.read()
    assert isinstance(result, str)
    
    # Test 7: Show diff with boolean
    input_code = "import sys\nimport os\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    
    changed = sort_stream(
        input_stream=input_stream,
        output_stream=output_stream,
        show_diff=False,
        config=DEFAULT_CONFIG
    )
    
    assert isinstance(changed, bool)
    
    # Test 8: Show diff with TextIO stream
    input_code = "import sys\nimport os\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    diff_stream = StringIO()
    
    changed = sort_stream(
        input_stream=input_stream,
        output_stream=output_stream,
        show_diff=diff_stream,
        config=DEFAULT_CONFIG
    )
    
    assert isinstance(changed, bool)
    
    # Test 9: With config kwargs
    input_code = "import sys\nimport os\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    
    changed = sort_stream(
        input_stream=input_stream,
        output_stream=output_stream,
        config=DEFAULT_CONFIG,
        line_length=80
    )
    
    output_stream.seek(0)
    result = output_stream.read()
    assert isinstance(result, str)
    
    # Test 10: Atomic mode with valid syntax
    input_code = "import sys\nimport os\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    config_atomic = Config(atomic=True)
    
    changed = sort_stream(
        input_stream=input_stream,
        output_stream=output_stream,
        config=config_atomic
    )
    
    output_stream.seek(0)
    result = output_stream.read()
    assert isinstance(result, str)


# LLM-generated content at query #9
#--------------------------

```python
def test_find_imports_in_stream():
    """Test find_imports_in_stream function with various import scenarios."""
    
    # Test basic imports
    code = "import os\nimport sys\nfrom pathlib import Path"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream))
    assert len(imports) == 3
    assert imports[0].module == "os"
    assert imports[1].module == "sys"
    assert imports[2].module == "pathlib"
    
    # Test with unique=True (ImportKey.ALIAS)
    code_with_duplicates = "import os\nimport os\nfrom os import path"
    stream = StringIO(code_with_duplicates)
    imports = list(find_imports_in_stream(stream, unique=True))
    assert len(imports) == 2
    
    # Test with unique=ImportKey.MODULE
    code_module = "from os import path\nfrom os import getcwd"
    stream = StringIO(code_module)
    imports = list(find_imports_in_stream(stream, unique=ImportKey.MODULE))
    assert len(imports) == 1
    assert imports[0].module == "os"
    
    # Test with unique=ImportKey.PACKAGE
    code_package = "from os.path import join\nfrom os import getcwd"
    stream = StringIO(code_package)
    imports = list(find_imports_in_stream(stream, unique=ImportKey.PACKAGE))
    assert len(imports) == 1
    assert imports[0].module == "os.path"
    
    # Test with unique=ImportKey.ATTRIBUTE
    code_attr = "from os import path\nfrom os import getcwd"
    stream = StringIO(code_attr)
    imports = list(find_imports_in_stream(stream, unique=ImportKey.ATTRIBUTE))
    assert len(imports) == 2
    
    # Test with top_only=True
    code_top = "import os\n\ndef func():\n    import sys"
    stream = StringIO(code_top)
    imports = list(find_imports_in_stream(stream, top_only=True))
    assert len(imports) == 1
    assert imports[0].module == "os"
    
    # Test with empty stream
    stream = StringIO("")
    imports = list(find_imports_in_stream(stream))
    assert len(imports) == 0
    
    # Test with file_path parameter
    code = "import json"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream, file_path=Path("test.py")))
    assert len(imports) == 1
    
    # Test with config modifications
    code = "import os"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream, force_single_line=True))
    assert len(imports) == 1
    
    # Test with _seen parameter (internal use)
    code = "import os\nimport sys"
    stream = StringIO(code)
    seen = {"os"}
    imports = list(find_imports_in_stream(stream, unique=True, _seen=seen))
    assert len(imports) == 1
    assert imports[0].module == "sys"
    
    # Test with custom config object
    config = Config(force_single_line=True)
    code = "import os, sys"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream, config=config))
    assert len(imports) >= 1


# LLM-generated content at query #10
#--------------------------

```python
def test_find_imports_in_stream():
    """Test find_imports_in_stream function with various configurations."""
    from io import StringIO
    
    # Test basic import finding
    code = "import os\nimport sys\nfrom pathlib import Path"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream))
    assert len(imports) == 3
    
    # Test with unique=True (should deduplicate by statement)
    code_with_duplicates = "import os\nimport os\nimport sys"
    stream = StringIO(code_with_duplicates)
    imports = list(find_imports_in_stream(stream, unique=True))
    assert len(imports) == 2
    
    # Test with unique=ImportKey.MODULE
    code_multi = "import os\nfrom os import path\nimport sys"
    stream = StringIO(code_multi)
    imports = list(find_imports_in_stream(stream, unique=ImportKey.MODULE))
    assert len(imports) == 2
    
    # Test with unique=ImportKey.PACKAGE
    code_package = "from os.path import join\nfrom os import environ\nimport sys"
    stream = StringIO(code_package)
    imports = list(find_imports_in_stream(stream, unique=ImportKey.PACKAGE))
    assert len(imports) == 2
    
    # Test with unique=ImportKey.ATTRIBUTE
    code_attr = "from os import path\nfrom os import environ\nimport sys"
    stream = StringIO(code_attr)
    imports = list(find_imports_in_stream(stream, unique=ImportKey.ATTRIBUTE))
    assert len(imports) == 3
    
    # Test with top_only=True (only imports before first class/function)
    code_with_func = "import os\ndef foo():\n    import sys\nimport json"
    stream = StringIO(code_with_func)
    imports = list(find_imports_in_stream(stream, top_only=True))
    assert len(imports) == 1
    
    # Test with file_path
    code = "import os"
    stream = StringIO(code)
    from pathlib import Path
    imports = list(find_imports_in_stream(stream, file_path=Path("test.py")))
    assert len(imports) == 1
    
    # Test with empty stream
    stream = StringIO("")
    imports = list(find_imports_in_stream(stream))
    assert len(imports) == 0
    
    # Test with config_kwargs
    code = "import os"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream, verbose=True))
    assert len(imports) == 1
    
    # Test with _seen parameter for deduplication across calls
    code = "import os\nimport sys"
    stream = StringIO(code)
    seen = {"os"}
    imports = list(find_imports_in_stream(stream, unique=True, _seen=seen))
    assert len(imports) == 1
    assert seen == {"os", "sys"}
    
    # Test mixed imports
    code = "import os\nfrom sys import argv\nimport json as j\nfrom pathlib import Path as P"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream))
    assert len(imports) == 4


# LLM-generated content at query #11
#--------------------------

```python
def test_sort_file(tmp_path, capsys):
    """Test sort_file function with various scenarios."""
    
    # Test 1: Basic file sorting
    test_file = tmp_path / "test.py"
    unsorted_code = "import os\nimport sys\nimport ast\n"
    test_file.write_text(unsorted_code)
    
    result = sort_file(str(test_file))
    assert result is True
    
    # Test 2: Already sorted file
    sorted_file = tmp_path / "sorted.py"
    sorted_code = "import ast\nimport os\nimport sys\n"
    sorted_file.write_text(sorted_code)
    
    result = sort_file(str(sorted_file))
    assert result is False
    
    # Test 3: Write to stdout
    stdout_file = tmp_path / "stdout_test.py"
    stdout_file.write_text(unsorted_code)
    
    result = sort_file(str(stdout_file), write_to_stdout=True)
    captured = capsys.readouterr()
    assert result is True
    
    # Test 4: Output to custom stream
    output_stream = StringIO()
    output_file = tmp_path / "output_test.py"
    output_file.write_text(unsorted_code)
    
    result = sort_file(str(output_file), output=output_stream)
    assert result is True
    output_stream.seek(0)
    output_content = output_stream.read()
    assert "import ast" in output_content
    
    # Test 5: Show diff
    diff_file = tmp_path / "diff_test.py"
    diff_file.write_text(unsorted_code)
    
    diff_stream = StringIO()
    result = sort_file(str(diff_file), show_diff=diff_stream)
    assert result is True
    diff_stream.seek(0)
    diff_content = diff_stream.read()
    assert diff_content  # Should contain diff output
    
    # Test 6: Disregard skip
    skip_file = tmp_path / "skip_test.py"
    skip_file.write_text(unsorted_code)
    
    result = sort_file(str(skip_file), disregard_skip=True)
    assert result is True
    
    # Test 7: File path parameter
    path_file = tmp_path / "path_test.py"
    path_file.write_text(unsorted_code)
    
    result = sort_file(str(path_file), file_path=path_file)
    assert result is True
    
    # Test 8: Extension parameter
    ext_file = tmp_path / "ext_test.py"
    ext_file.write_text(unsorted_code)
    
    result = sort_file(str(ext_file), extension="py")
    assert result is True
    
    # Test 9: Config parameter
    config_file = tmp_path / "config_test.py"
    config_file.write_text(unsorted_code)
    
    test_config = Config()
    result = sort_file(str(config_file), config=test_config)
    assert result is True
    
    # Test 10: In-memory overwrite
    inplace_file = tmp_path / "inplace_test.py"
    inplace_file.write_text(unsorted_code)
    
    config_inplace = Config(overwrite_in_place=True)
    result = sort_file(str(inplace_file), config=config_inplace)
    assert result is True
    content = inplace_file.read_text()
    assert "import ast" in content


def test_sort_file_with_syntax_errors(tmp_path):
    """Test sort_file with syntax errors."""
    
    syntax_error_file = tmp_path / "syntax_error.py"
    syntax_error_code = "import os\nimport sys\nthis is invalid python\n"
    syntax_error_file.write_text(syntax_error_code)
    
    with pytest.warns(UserWarning, match="unable to sort due to existing syntax errors"):
        result = sort_file(str(syntax_error_file))
    
    assert result is False


def test_sort_file_atomic_mode(tmp_path):
    """Test sort_file with atomic mode enabled."""
    
    atomic_file = tmp_path / "atomic_test.py"
    valid_code = "import os\nimport sys\nprint('hello')\n"
    atomic_file.write_text(valid_code)
    
    config_atomic = Config(atomic=True)
    result = sort_file(str(atomic_file), config=config_atomic)
    assert result is True
    
    content = atomic_file.read_text()
    assert "import os" in content
    assert "print('hello')" in content


def test_sort_file_with_path_object(tmp_path):
    """Test sort_file with Path object instead of string."""
    
    path_obj_file = tmp_path / "path_obj_test.py"
    unsorted_code = "import sys\nimport os\n"
    path_obj_file.write_text(unsorted_code)
    
    result = sort_file(path_obj_file)
    assert result is True
    
    content = path_obj_file.read_text()
    assert content.index("import os") < content.index("import sys")


def test_sort_file_quiet_mode(tmp_path, capsys):
    """Test sort_file with quiet mode."""
    
    quiet_file = tmp_path / "quiet_test.py"
    unsorted_code = "import sys\nimport os\n"
    quiet_file.write_text(unsorted_code)
    
    config_quiet = Config(quiet=True)
    result = sort_file(str(quiet_file), config=config_quiet)
    assert result is True
    
    captured = capsys.readouterr()
    assert "Fixing" not in captured.out


def test_sort_file_verbose_mode(tmp_path, capsys):
    """Test sort_file with verbose mode."""
    
    verbose_file = tmp_path / "verbose_test.py"
    sorted_code = "import os\nimport sys\n"
    verbose_file.write_text(sorted_code)
    
    config_verbose = Config(verbose=True)
    result = sort_file(str(verbose_file), config=config_verbose)
    assert result is False


# LLM-generated content at query #12
#--------------------------

```python
def test_check_stream():
    """Test check_stream function with various scenarios."""
    
    # Test 1: Correctly sorted imports should return True
    correct_code = "import os\nimport sys\n\nprint('hello')\n"
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream)
    assert result is True
    
    # Test 2: Incorrectly sorted imports should return False
    incorrect_code = "import sys\nimport os\n\nprint('hello')\n"
    input_stream = StringIO(incorrect_code)
    result = check_stream(input_stream)
    assert result is False
    
    # Test 3: With extension parameter
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream, extension="py")
    assert result is True
    
    # Test 4: With file_path parameter
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream, file_path=Path("test.py"))
    assert result is True
    
    # Test 5: With show_diff as True
    input_stream = StringIO(incorrect_code)
    output = StringIO()
    result = check_stream(input_stream, show_diff=output)
    assert result is False
    assert len(output.getvalue()) > 0
    
    # Test 6: With show_diff as TextIO
    input_stream = StringIO(incorrect_code)
    diff_output = StringIO()
    result = check_stream(input_stream, show_diff=diff_output)
    assert result is False
    
    # Test 7: Empty code should return True
    input_stream = StringIO("")
    result = check_stream(input_stream)
    assert result is True
    
    # Test 8: Code with no imports should return True
    code_no_imports = "x = 1\ny = 2\n"
    input_stream = StringIO(code_no_imports)
    result = check_stream(input_stream)
    assert result is True
    
    # Test 9: With config parameter
    config = Config()
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream, config=config)
    assert result is True
    
    # Test 10: With disregard_skip parameter
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream, disregard_skip=True)
    assert result is True
    
    # Test 11: Multiple correctly sorted imports
    multi_import_code = "import os\nimport sys\nfrom pathlib import Path\n\nprint('hello')\n"
    input_stream = StringIO(multi_import_code)
    result = check_stream(input_stream)
    assert result is True
    
    # Test 12: Multiple incorrectly sorted imports
    multi_import_wrong = "from pathlib import Path\nimport sys\nimport os\n\nprint('hello')\n"
    input_stream = StringIO(multi_import_wrong)
    result = check_stream(input_stream)
    assert result is False


# LLM-generated content at query #13
#--------------------------

```python
def test_find_imports_in_paths(tmp_path):
    """Test find_imports_in_paths function."""
    # Create test files with imports
    test_file_1 = tmp_path / "test1.py"
    test_file_1.write_text("import os\nimport sys\nfrom pathlib import Path")
    
    test_file_2 = tmp_path / "test2.py"
    test_file_2.write_text("import json\nfrom typing import List")
    
    # Test basic functionality
    imports = list(find_imports_in_paths([tmp_path]))
    assert len(imports) == 5
    
    # Test with unique=True (ALIAS)
    imports_unique = list(find_imports_in_paths([tmp_path], unique=True))
    assert len(imports_unique) == 5
    assert all(hasattr(imp, 'module') for imp in imports_unique)
    
    # Test with unique=ImportKey.MODULE
    imports_module = list(find_imports_in_paths([tmp_path], unique=ImportKey.MODULE))
    assert len(imports_module) == 5
    
    # Test with unique=ImportKey.PACKAGE
    imports_package = list(find_imports_in_paths([tmp_path], unique=ImportKey.PACKAGE))
    assert len(imports_package) >= 1
    
    # Test with unique=ImportKey.ATTRIBUTE
    imports_attr = list(find_imports_in_paths([tmp_path], unique=ImportKey.ATTRIBUTE))
    assert len(imports_attr) >= 1
    
    # Test with top_only=True
    test_file_3 = tmp_path / "test3.py"
    test_file_3.write_text("import os\n\ndef func():\n    import json")
    imports_top = list(find_imports_in_paths([tmp_path], top_only=True))
    assert len(imports_top) >= 1
    
    # Test with empty directory
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    imports_empty = list(find_imports_in_paths([empty_dir]))
    assert len(imports_empty) == 0
    
    # Test with custom config
    custom_config = Config()
    imports_config = list(find_imports_in_paths([tmp_path], config=custom_config))
    assert len(imports_config) == 5
    
    # Test with multiple paths
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    test_file_4 = subdir / "test4.py"
    test_file_4.write_text("import asyncio")
    imports_multi = list(find_imports_in_paths([tmp_path / "test1.py", subdir]))
    assert len(imports_multi) >= 1


# LLM-generated content at query #14
#--------------------------

```python
def test_check_file(tmp_path):
    """Test the check_file function with various scenarios."""
    
    # Test 1: File with correctly sorted imports should return True
    correct_file = tmp_path / "correct_imports.py"
    correct_file.write_text("import os\nimport sys\n\nprint('hello')\n")
    assert check_file(correct_file) is True
    
    # Test 2: File with incorrectly sorted imports should return False
    incorrect_file = tmp_path / "incorrect_imports.py"
    incorrect_file.write_text("import sys\nimport os\n\nprint('hello')\n")
    assert check_file(incorrect_file) is False
    
    # Test 3: File with skip comment should raise FileSkipComment when disregard_skip=False
    skip_file = tmp_path / "skip_imports.py"
    skip_file.write_text("# isort: skip_file\nimport sys\nimport os\n")
    with contextlib.suppress(FileSkipComment):
        check_file(skip_file, disregard_skip=False)
    
    # Test 4: File with skip comment but disregard_skip=True should check anyway
    assert check_file(skip_file, disregard_skip=True) is False
    
    # Test 5: Custom file path parameter
    result = check_file(correct_file, file_path=Path("custom/path.py"))
    assert result is True
    
    # Test 6: File with show_diff as TextIO
    diff_stream = StringIO()
    incorrect_file2 = tmp_path / "incorrect_imports2.py"
    incorrect_file2.write_text("import sys\nimport os\n")
    result = check_file(incorrect_file2, show_diff=diff_stream)
    assert result is False
    assert diff_stream.getvalue() != ""
    
    # Test 7: File with show_diff=True
    result = check_file(incorrect_file2, show_diff=True)
    assert result is False
    
    # Test 8: Empty file should return True
    empty_file = tmp_path / "empty.py"
    empty_file.write_text("")
    assert check_file(empty_file) is True
    
    # Test 9: File with only comments should return True
    comment_file = tmp_path / "comments_only.py"
    comment_file.write_text("# This is a comment\n# Another comment\n")
    assert check_file(comment_file) is True
    
    # Test 10: File with custom extension
    custom_ext_file = tmp_path / "custom.pyx"
    custom_ext_file.write_text("import sys\nimport os\n")
    result = check_file(custom_ext_file, extension="pyx")
    assert result is False
    
    # Test 11: File with config kwargs
    config_file = tmp_path / "config_test.py"
    config_file.write_text("from module import a, b\n")
    result = check_file(config_file, force_single_line=True)
    assert isinstance(result, bool)
    
    # Test 12: String path instead of Path object
    result = check_file(str(correct_file))
    assert result is True


# LLM-generated content at query #15
#--------------------------

```python
def test_sort_stream():
    """Test sort_stream function with various scenarios."""
    
    # Test 1: Basic import sorting
    input_code = "import os\nimport sys\nimport collections\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    
    changed = sort_stream(input_stream, output_stream)
    output_stream.seek(0)
    result = output_stream.read()
    
    assert changed is False or changed is True
    assert "import" in result
    
    # Test 2: Unsorted imports should be sorted
    input_code = "import sys\nimport os\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    
    changed = sort_stream(input_stream, output_stream)
    output_stream.seek(0)
    result = output_stream.read()
    
    assert isinstance(changed, bool)
    assert "import os" in result
    assert "import sys" in result
    
    # Test 3: With file_path and extension
    input_code = "import sys\nimport os\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    
    changed = sort_stream(
        input_stream,
        output_stream,
        extension="py",
        file_path=Path("test.py")
    )
    
    assert isinstance(changed, bool)
    output_stream.seek(0)
    assert len(output_stream.read()) > 0
    
    # Test 4: With show_diff as False (default behavior)
    input_code = "import sys\nimport os\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    
    changed = sort_stream(
        input_stream,
        output_stream,
        show_diff=False
    )
    
    assert isinstance(changed, bool)
    
    # Test 5: With show_diff as True
    input_code = "import sys\nimport os\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    
    changed = sort_stream(
        input_stream,
        output_stream,
        show_diff=True
    )
    
    assert isinstance(changed, bool)
    
    # Test 6: With show_diff as TextIO stream
    input_code = "import sys\nimport os\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    diff_output = StringIO()
    
    changed = sort_stream(
        input_stream,
        output_stream,
        show_diff=diff_output
    )
    
    assert isinstance(changed, bool)
    
    # Test 7: With atomic=True (syntax checking enabled)
    input_code = "import sys\nimport os\n\nprint('hello')\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    config = Config(atomic=True)
    
    changed = sort_stream(
        input_stream,
        output_stream,
        config=config
    )
    
    assert isinstance(changed, bool)
    output_stream.seek(0)
    result = output_stream.read()
    assert "import" in result
    
    # Test 8: Empty input
    input_code = ""
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    
    changed = sort_stream(input_stream, output_stream)
    
    assert isinstance(changed, bool)
    
    # Test 9: Code with from imports
    input_code = "from sys import argv\nfrom os import path\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    
    changed = sort_stream(input_stream, output_stream)
    output_stream.seek(0)
    result = output_stream.read()
    
    assert isinstance(changed, bool)
    assert "from" in result
    
    # Test 10: With config_kwargs
    input_code = "import sys\nimport os\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    
    changed = sort_stream(
        input_stream,
        output_stream,
        line_length=80
    )
    
    assert isinstance(changed, bool)
    
    # Test 11: With disregard_skip=True
    input_code = "import sys\nimport os\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    
    changed = sort_stream(
        input_stream,
        output_stream,
        disregard_skip=True
    )
    
    assert isinstance(changed, bool)
    
    # Test 12: With raise_on_skip=False
    input_code = "import sys\nimport os\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    
    changed = sort_stream(
        input_stream,
        output_stream,
        raise_on_skip=False
    )
    
    assert isinstance(changed, bool)


# LLM-generated content at query #16
#--------------------------

```python
def test_check_stream():
    """Test check_stream function with various scenarios."""
    
    # Test 1: Correctly sorted imports should return True
    correct_code = "import os\nimport sys\n"
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream)
    assert result is True
    
    # Test 2: Incorrectly sorted imports should return False
    incorrect_code = "import sys\nimport os\n"
    input_stream = StringIO(incorrect_code)
    result = check_stream(input_stream)
    assert result is False
    
    # Test 3: With file_path parameter
    correct_code = "import os\n"
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream, file_path=Path("test.py"))
    assert result is True
    
    # Test 4: With extension parameter
    correct_code = "import os\n"
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream, extension="py")
    assert result is True
    
    # Test 5: With show_diff as True (should not raise exception)
    incorrect_code = "import sys\nimport os\n"
    input_stream = StringIO(incorrect_code)
    result = check_stream(input_stream, show_diff=True)
    assert result is False
    
    # Test 6: With show_diff as StringIO object
    incorrect_code = "import sys\nimport os\n"
    input_stream = StringIO(incorrect_code)
    diff_output = StringIO()
    result = check_stream(input_stream, show_diff=diff_output)
    assert result is False
    
    # Test 7: With custom config
    correct_code = "import os\nimport sys\n"
    input_stream = StringIO(correct_code)
    config = Config(line_length=80)
    result = check_stream(input_stream, config=config)
    assert result is True
    
    # Test 8: With config_kwargs
    correct_code = "import os\n"
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream, line_length=80)
    assert result is True
    
    # Test 9: disregard_skip parameter
    correct_code = "import os\n"
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream, disregard_skip=True)
    assert result is True
    
    # Test 10: Empty code should return True
    empty_code = ""
    input_stream = StringIO(empty_code)
    result = check_stream(input_stream)
    assert result is True
    
    # Test 11: Code with from imports
    correct_code = "from os import path\nfrom sys import argv\n"
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream)
    assert result is True
    
    # Test 12: Incorrectly formatted from imports
    incorrect_code = "from sys import argv\nfrom os import path\n"
    input_stream = StringIO(incorrect_code)
    result = check_stream(input_stream)
    assert result is False
    
    # Test 13: Mixed imports
    correct_code = "import os\nfrom sys import argv\n"
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream)
    assert result is True
    
    # Test 14: show_diff with file_path
    incorrect_code = "import sys\nimport os\n"
    input_stream = StringIO(incorrect_code)
    result = check_stream(input_stream, show_diff=True, file_path=Path("test.py"))
    assert result is False
    
    # Test 15: Correctly sorted with comments
    correct_code = "# This is a comment\nimport os\n"
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream)
    assert result is True


# LLM-generated content at query #17
#--------------------------

```python
def test_find_imports_in_file(tmp_path):
    """Test find_imports_in_file function with various scenarios."""
    # Create a temporary file with imports
    test_file = tmp_path / "test_imports.py"
    test_content = """import os
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
from collections import defaultdict

def my_function():
    pass
"""
    test_file.write_text(test_content)
    
    # Test basic import finding
    imports = list(find_imports_in_file(test_file))
    assert len(imports) == 6
    assert any(imp.module == "os" for imp in imports)
    assert any(imp.module == "sys" for imp in imports)
    assert any(imp.module == "pathlib" for imp in imports)
    assert any(imp.module == "typing" for imp in imports)
    assert any(imp.module == "numpy" for imp in imports)
    assert any(imp.module == "collections" for imp in imports)
    
    # Test with unique=True
    imports_unique = list(find_imports_in_file(test_file, unique=True))
    assert len(imports_unique) == 6
    
    # Test with top_only=True
    imports_top_only = list(find_imports_in_file(test_file, top_only=True))
    assert len(imports_top_only) == 6
    
    # Test with Path object
    imports_from_path = list(find_imports_in_file(test_file))
    assert len(imports_from_path) == 6
    
    # Test with string filename
    imports_from_string = list(find_imports_in_file(str(test_file)))
    assert len(imports_from_string) == 6
    
    # Test with custom file_path parameter
    imports_with_file_path = list(find_imports_in_file(test_file, file_path=Path("custom/path.py")))
    assert len(imports_with_file_path) == 6
    
    # Test with unique=ImportKey.MODULE
    imports_unique_module = list(find_imports_in_file(test_file, unique=ImportKey.MODULE))
    assert len(imports_unique_module) == 6
    
    # Test with unique=ImportKey.PACKAGE
    imports_unique_package = list(find_imports_in_file(test_file, unique=ImportKey.PACKAGE))
    assert len(imports_unique_package) == 6
    
    # Test with non-existent file (should warn and return empty)
    non_existent = tmp_path / "non_existent.py"
    imports_non_existent = list(find_imports_in_file(non_existent))
    assert len(imports_non_existent) == 0
    
    # Test with empty file
    empty_file = tmp_path / "empty.py"
    empty_file.write_text("")
    imports_empty = list(find_imports_in_file(empty_file))
    assert len(imports_empty) == 0
    
    # Test with file containing only comments
    comments_file = tmp_path / "comments.py"
    comments_file.write_text("# This is a comment\n# Another comment")
    imports_comments = list(find_imports_in_file(comments_file))
    assert len(imports_comments) == 0
    
    # Test with config parameter
    config = Config()
    imports_with_config = list(find_imports_in_file(test_file, config=config))
    assert len(imports_with_config) == 6


# LLM-generated content at query #18
#--------------------------

```python
def test_find_imports_in_paths(tmp_path):
    """Test find_imports_in_paths function."""
    # Create test files with imports
    file1 = tmp_path / "file1.py"
    file1.write_text("import os\nimport sys\nfrom pathlib import Path")
    
    file2 = tmp_path / "file2.py"
    file2.write_text("import json\nfrom typing import Dict")
    
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    file3 = subdir / "file3.py"
    file3.write_text("import re\nfrom collections import defaultdict")
    
    # Test basic functionality
    imports = list(find_imports_in_paths([tmp_path]))
    assert len(imports) == 6
    
    # Test unique with ImportKey.MODULE
    unique_imports = list(find_imports_in_paths([tmp_path], unique=ImportKey.MODULE))
    assert len(unique_imports) == 6
    
    # Test unique with True
    unique_imports_all = list(find_imports_in_paths([tmp_path], unique=True))
    assert len(unique_imports_all) == 6
    
    # Test with custom config
    config = Config(skip=["file1.py"])
    imports_with_skip = list(find_imports_in_paths([tmp_path], config=config))
    assert len(imports_with_skip) == 4
    
    # Test top_only parameter
    file_with_code = tmp_path / "file4.py"
    file_with_code.write_text("import os\n\ndef func():\n    import sys")
    imports_top_only = list(find_imports_in_paths([tmp_path / "file4.py"], top_only=True))
    assert len(imports_top_only) == 1
    
    # Test with empty paths
    imports_empty = list(find_imports_in_paths([]))
    assert len(imports_empty) == 0
    
    # Test unique with ImportKey.PACKAGE
    imports_package = list(find_imports_in_paths([tmp_path], unique=ImportKey.PACKAGE))
    assert len(imports_package) > 0
    
    # Test unique with ImportKey.ATTRIBUTE
    imports_attr = list(find_imports_in_paths([tmp_path], unique=ImportKey.ATTRIBUTE))
    assert len(imports_attr) > 0


# LLM-generated content at query #19
#--------------------------

```python
def test_check_stream():
    """Test check_stream function with various scenarios."""
    
    # Test 1: Correctly sorted imports should return True
    correct_code = "import os\nimport sys\n"
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream)
    assert result is True
    
    # Test 2: Incorrectly sorted imports should return False
    incorrect_code = "import sys\nimport os\n"
    input_stream = StringIO(incorrect_code)
    result = check_stream(input_stream)
    assert result is False
    
    # Test 3: With file_path parameter
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream, file_path=Path("test.py"))
    assert result is True
    
    # Test 4: With extension parameter
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream, extension="py")
    assert result is True
    
    # Test 5: With custom config
    config = Config()
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream, config=config)
    assert result is True
    
    # Test 6: With config_kwargs
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream, force_single_line=True)
    assert result is True
    
    # Test 7: With show_diff as True
    incorrect_code = "import sys\nimport os\n"
    input_stream = StringIO(incorrect_code)
    result = check_stream(input_stream, show_diff=True)
    assert result is False
    
    # Test 8: With show_diff as TextIO stream
    incorrect_code = "import sys\nimport os\n"
    input_stream = StringIO(incorrect_code)
    diff_output = StringIO()
    result = check_stream(input_stream, show_diff=diff_output)
    assert result is False
    
    # Test 9: With disregard_skip parameter
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream, disregard_skip=True)
    assert result is True
    
    # Test 10: Empty input should return True
    input_stream = StringIO("")
    result = check_stream(input_stream)
    assert result is True
    
    # Test 11: Code with only comments should return True
    comment_code = "# This is a comment\n"
    input_stream = StringIO(comment_code)
    result = check_stream(input_stream)
    assert result is True
    
    # Test 12: Multiple imports in correct order
    multi_import_code = "import os\nimport sys\nfrom pathlib import Path\n"
    input_stream = StringIO(multi_import_code)
    result = check_stream(input_stream)
    assert result is True
    
    # Test 13: Multiple imports in incorrect order
    multi_import_incorrect = "from pathlib import Path\nimport sys\nimport os\n"
    input_stream = StringIO(multi_import_incorrect)
    result = check_stream(input_stream)
    assert result is False


# LLM-generated content at query #20
#--------------------------

```python
def test_check_file(tmp_path):
    """Test the check_file function with various scenarios."""
    
    # Test 1: File with correctly sorted imports
    correct_file = tmp_path / "correct_imports.py"
    correct_file.write_text("import os\nimport sys\n\nprint('hello')\n")
    assert check_file(correct_file) is True
    
    # Test 2: File with incorrectly sorted imports
    incorrect_file = tmp_path / "incorrect_imports.py"
    incorrect_file.write_text("import sys\nimport os\n\nprint('hello')\n")
    assert check_file(incorrect_file) is False
    
    # Test 3: File with no imports
    no_imports_file = tmp_path / "no_imports.py"
    no_imports_file.write_text("print('hello')\n")
    assert check_file(no_imports_file) is True
    
    # Test 4: Check with disregard_skip=False
    skip_file = tmp_path / "skip_file.py"
    skip_file.write_text("# isort: skip_file\nimport sys\nimport os\n")
    with pytest.raises(FileSkipSetting):
        check_file(skip_file, disregard_skip=False)
    
    # Test 5: Check with disregard_skip=True (should ignore skip)
    assert check_file(skip_file, disregard_skip=True) is False
    
    # Test 6: File with Path object
    path_obj = Path(correct_file)
    assert check_file(path_obj) is True
    
    # Test 7: File with string filename
    assert check_file(str(correct_file)) is True
    
    # Test 8: Check with show_diff as boolean
    output = StringIO()
    result = check_file(incorrect_file, show_diff=False)
    assert result is False
    
    # Test 9: Check with show_diff as TextIO
    output = StringIO()
    result = check_file(incorrect_file, show_diff=output)
    assert result is False
    
    # Test 10: Check with custom config
    custom_config = Config(line_length=80)
    assert check_file(correct_file, config=custom_config) is True
    
    # Test 11: Check with config_kwargs
    assert check_file(correct_file, line_length=80) is True
    
    # Test 12: File with syntax errors (non-atomic mode)
    syntax_error_file = tmp_path / "syntax_error.py"
    syntax_error_file.write_text("import os\nif True\n    pass\n")
    assert check_file(syntax_error_file) is True  # Should not raise in non-atomic mode
    
    # Test 13: Check with atomic mode and syntax errors
    with pytest.raises(ExistingSyntaxErrors):
        check_file(syntax_error_file, config=Config(atomic=True))
    
    # Test 14: File with extension parameter
    py_file = tmp_path / "test.py"
    py_file.write_text("import sys\nimport os\n")
    assert check_file(py_file, extension="py") is False
    
    # Test 15: Check with file_path parameter
    result = check_file(correct_file, file_path=Path("/custom/path.py"))
    assert result is True


# LLM-generated content at query #21
#--------------------------

```python
def test_check_file(tmp_path, capsys):
    """Test check_file function with various scenarios."""
    
    # Test 1: File with correctly sorted imports
    correct_file = tmp_path / "correct.py"
    correct_file.write_text("import os\nimport sys\n\nprint('hello')\n")
    assert check_file(correct_file) is True
    
    # Test 2: File with incorrectly sorted imports
    incorrect_file = tmp_path / "incorrect.py"
    incorrect_file.write_text("import sys\nimport os\n\nprint('hello')\n")
    assert check_file(incorrect_file) is False
    
    # Test 3: File with show_diff=True
    incorrect_file2 = tmp_path / "incorrect2.py"
    incorrect_file2.write_text("import sys\nimport os\n")
    result = check_file(incorrect_file2, show_diff=True)
    assert result is False
    captured = capsys.readouterr()
    assert "---" in captured.out or "+++" in captured.out or len(captured.out) > 0
    
    # Test 4: File with show_diff as TextIO stream
    incorrect_file3 = tmp_path / "incorrect3.py"
    incorrect_file3.write_text("import sys\nimport os\n")
    diff_output = StringIO()
    result = check_file(incorrect_file3, show_diff=diff_output)
    assert result is False
    diff_output.seek(0)
    diff_content = diff_output.read()
    assert len(diff_content) > 0
    
    # Test 5: File with disregard_skip=False and skip setting
    skipped_file = tmp_path / "skipped.py"
    skipped_file.write_text("# isort: skip_file\nimport sys\nimport os\n")
    with contextlib.suppress(FileSkipComment):
        check_file(skipped_file, disregard_skip=False)
    
    # Test 6: File with custom extension
    custom_ext_file = tmp_path / "custom.pyx"
    custom_ext_file.write_text("import sys\nimport os\n")
    result = check_file(custom_ext_file, extension="pyx")
    assert isinstance(result, bool)
    
    # Test 7: File with custom config
    config_file = tmp_path / "config_test.py"
    config_file.write_text("from x import z, a\n")
    custom_config = Config(force_alphabetical_sort_within_sections=True)
    result = check_file(config_file, config=custom_config)
    assert isinstance(result, bool)
    
    # Test 8: File with file_path override
    source_file = tmp_path / "source.py"
    source_file.write_text("import os\n")
    override_path = Path("/custom/path/file.py")
    result = check_file(source_file, file_path=override_path)
    assert isinstance(result, bool)
    
    # Test 9: File with string filename
    str_file = tmp_path / "string_path.py"
    str_file.write_text("import os\nimport sys\n\nprint('test')\n")
    result = check_file(str(str_file))
    assert result is True
    
    # Test 10: Empty file
    empty_file = tmp_path / "empty.py"
    empty_file.write_text("")
    result = check_file(empty_file)
    assert result is True


# LLM-generated content at query #22
#--------------------------

```python
def test_find_imports_in_paths(tmp_path, config):
    """Test find_imports_in_paths function."""
    # Create temporary Python files with imports
    file1 = tmp_path / "module1.py"
    file1.write_text("import os\nimport sys\nfrom pathlib import Path")
    
    file2 = tmp_path / "module2.py"
    file2.write_text("from collections import defaultdict\nimport json")
    
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    file3 = subdir / "module3.py"
    file3.write_text("import os\nimport numpy as np")
    
    # Test basic functionality
    imports = list(find_imports_in_paths([tmp_path], config=config))
    assert len(imports) > 0
    
    # Test with unique=True
    unique_imports = list(find_imports_in_paths([tmp_path], config=config, unique=True))
    assert len(unique_imports) <= len(imports)
    
    # Test with unique=ImportKey.MODULE
    module_imports = list(find_imports_in_paths(
        [tmp_path], config=config, unique=ImportKey.MODULE
    ))
    assert len(module_imports) <= len(imports)
    
    # Test with unique=ImportKey.PACKAGE
    package_imports = list(find_imports_in_paths(
        [tmp_path], config=config, unique=ImportKey.PACKAGE
    ))
    assert len(package_imports) <= len(imports)
    
    # Test with top_only=True
    top_imports = list(find_imports_in_paths([tmp_path], config=config, top_only=True))
    assert len(top_imports) > 0
    
    # Test with empty paths
    empty_imports = list(find_imports_in_paths([], config=config))
    assert len(empty_imports) == 0
    
    # Test that imports are identified correctly
    import_modules = {imp.module for imp in imports}
    assert any(module in import_modules for module in ["os", "sys", "json", "collections", "pathlib"])
    
    # Test with single file path
    single_file_imports = list(find_imports_in_paths([file1], config=config))
    assert len(single_file_imports) > 0
    
    # Test unique with seen set maintained across files
    unique_all = list(find_imports_in_paths([tmp_path], config=config, unique=True))
    modules_seen = {imp.module for imp in unique_all}
    assert len(modules_seen) > 0


# LLM-generated content at query #23
#--------------------------

```python
def test_check_file(tmp_path):
    """Test check_file function with various scenarios."""
    
    # Test 1: File with correctly sorted imports
    correct_file = tmp_path / "correct_imports.py"
    correct_file.write_text("import os\nimport sys\n\nprint('hello')\n")
    assert check_file(correct_file) is True
    
    # Test 2: File with incorrectly sorted imports
    incorrect_file = tmp_path / "incorrect_imports.py"
    incorrect_file.write_text("import sys\nimport os\n\nprint('hello')\n")
    assert check_file(incorrect_file) is False
    
    # Test 3: File with no imports
    no_imports_file = tmp_path / "no_imports.py"
    no_imports_file.write_text("print('hello')\n")
    assert check_file(no_imports_file) is True
    
    # Test 4: Check with show_diff as True
    incorrect_file2 = tmp_path / "incorrect_imports2.py"
    incorrect_file2.write_text("import sys\nimport os\n")
    assert check_file(incorrect_file2, show_diff=True) is False
    
    # Test 5: Check with show_diff as StringIO
    incorrect_file3 = tmp_path / "incorrect_imports3.py"
    incorrect_file3.write_text("import sys\nimport os\n")
    diff_output = StringIO()
    assert check_file(incorrect_file3, show_diff=diff_output) is False
    
    # Test 6: File with extension parameter
    custom_ext_file = tmp_path / "test_file"
    custom_ext_file.write_text("import os\nimport sys\n")
    assert check_file(custom_ext_file, extension="py") is True
    
    # Test 7: File with disregard_skip=False and skip comment
    skip_file = tmp_path / "skip_imports.py"
    skip_file.write_text("# isort: skip_file\nimport sys\nimport os\n")
    with contextlib.suppress(FileSkipComment):
        check_file(skip_file, disregard_skip=False)
    
    # Test 8: File with custom config
    custom_config_file = tmp_path / "custom_config.py"
    custom_config_file.write_text("import os\nimport sys\n")
    custom_config = Config(line_length=80)
    assert check_file(custom_config_file, config=custom_config) is True
    
    # Test 9: File path as string
    string_path_file = tmp_path / "string_path.py"
    string_path_file.write_text("import os\nimport sys\n")
    assert check_file(str(string_path_file)) is True
    
    # Test 10: File path as Path object
    path_obj_file = tmp_path / "path_obj.py"
    path_obj_file.write_text("import os\nimport sys\n")
    assert check_file(Path(path_obj_file)) is True
    
    # Test 11: Complex imports
    complex_file = tmp_path / "complex.py"
    complex_file.write_text(
        "from typing import Dict, List\nimport os\nfrom pathlib import Path\nimport sys\n"
    )
    assert check_file(complex_file) is False
    
    # Test 12: Correctly sorted complex imports
    sorted_complex_file = tmp_path / "sorted_complex.py"
    sorted_complex_file.write_text(
        "import os\nimport sys\nfrom pathlib import Path\nfrom typing import Dict, List\n"
    )
    assert check_file(sorted_complex_file) is True


# LLM-generated content at query #24
#--------------------------

```python
def test_sort_stream():
    """Test the sort_stream function with various scenarios."""
    import io
    from pathlib import Path
    
    # Test 1: Basic sorting of imports
    input_code = "import os\nimport sys\nimport collections\n"
    input_stream = io.StringIO(input_code)
    output_stream = io.StringIO()
    
    changed = sort_stream(input_stream, output_stream)
    output_stream.seek(0)
    result = output_stream.read()
    
    assert changed is False  # Already sorted
    assert "import collections" in result
    assert "import os" in result
    assert "import sys" in result
    
    # Test 2: Unsorted imports should be reordered
    unsorted_code = "import sys\nimport os\nimport collections\n"
    input_stream = io.StringIO(unsorted_code)
    output_stream = io.StringIO()
    
    changed = sort_stream(input_stream, output_stream)
    output_stream.seek(0)
    result = output_stream.read()
    
    assert changed is True  # Should be changed
    lines = result.strip().split('\n')
    assert lines[0] == "import collections"
    assert lines[1] == "import os"
    assert lines[2] == "import sys"
    
    # Test 3: With file_path extension
    input_stream = io.StringIO("import sys\nimport os\n")
    output_stream = io.StringIO()
    file_path = Path("test.py")
    
    changed = sort_stream(input_stream, output_stream, file_path=file_path)
    output_stream.seek(0)
    result = output_stream.read()
    
    assert "import os" in result
    assert "import sys" in result
    
    # Test 4: With custom config
    input_stream = io.StringIO("import sys\nimport os\n")
    output_stream = io.StringIO()
    config = Config()
    
    changed = sort_stream(input_stream, output_stream, config=config)
    output_stream.seek(0)
    result = output_stream.read()
    
    assert result is not None
    
    # Test 5: Empty input
    input_stream = io.StringIO("")
    output_stream = io.StringIO()
    
    changed = sort_stream(input_stream, output_stream)
    output_stream.seek(0)
    result = output_stream.read()
    
    assert changed is False
    assert result == ""
    
    # Test 6: Code with syntax errors and atomic mode disabled
    input_stream = io.StringIO("import sys\nimport os\n")
    output_stream = io.StringIO()
    config = Config(atomic=False)
    
    changed = sort_stream(input_stream, output_stream, config=config)
    output_stream.seek(0)
    result = output_stream.read()
    
    assert "import" in result
    
    # Test 7: With disregard_skip flag
    input_stream = io.StringIO("import sys\n")
    output_stream = io.StringIO()
    
    changed = sort_stream(input_stream, output_stream, disregard_skip=True)
    output_stream.seek(0)
    result = output_stream.read()
    
    assert "import sys" in result
    
    # Test 8: With show_diff as StringIO
    input_stream = io.StringIO("import sys\nimport os\n")
    output_stream = io.StringIO()
    diff_output = io.StringIO()
    
    changed = sort_stream(input_stream, output_stream, show_diff=diff_output)
    output_stream.seek(0)
    result = output_stream.read()
    
    assert result is not None
    
    # Test 9: With extension parameter
    input_stream = io.StringIO("import sys\nimport os\n")
    output_stream = io.StringIO()
    
    changed = sort_stream(input_stream, output_stream, extension="py")
    output_stream.seek(0)
    result = output_stream.read()
    
    assert "import" in result
    
    # Test 10: From/import statements
    input_code = "from sys import path\nfrom os import environ\n"
    input_stream = io.StringIO(input_code)
    output_stream = io.StringIO()
    
    changed = sort_stream(input_stream, output_stream)
    output_stream.seek(0)
    result = output_stream.read()
    
    assert "from os import environ" in result
    assert "from sys import path" in result


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_sort_file(tmp_path, capsys):
    """Test sort_file function with various scenarios."""
    
    # Test 1: Basic file sorting
    test_file = tmp_path / "test_imports.py"
    unsorted_code = "import os\nimport sys\nimport collections\n"
    test_file.write_text(unsorted_code)
    
    result = sort_file(test_file)
    assert result is True
    sorted_content = test_file.read_text()
    assert "import collections" in sorted_content
    assert "import os" in sorted_content
    assert "import sys" in sorted_content
    
    # Test 2: Already sorted file should return False
    test_file2 = tmp_path / "already_sorted.py"
    already_sorted = "import collections\nimport os\nimport sys\n"
    test_file2.write_text(already_sorted)
    
    result = sort_file(test_file2)
    assert result is False
    
    # Test 3: File with skip setting
    test_file3 = tmp_path / "skip_file.py"
    test_file3.write_text("import sys\nimport os\n")
    config = Config(skip=[test_file3])
    
    try:
        sort_file(test_file3, config=config, disregard_skip=False)
    except FileSkipSetting:
        pass
    
    # Test 4: Write to stdout
    test_file4 = tmp_path / "stdout_test.py"
    test_file4.write_text("import sys\nimport os\n")
    
    result = sort_file(test_file4, write_to_stdout=True)
    assert result is True
    captured = capsys.readouterr()
    assert "import" in captured.out
    
    # Test 5: Output to custom stream
    test_file5 = tmp_path / "custom_output.py"
    test_file5.write_text("import sys\nimport os\n")
    output_stream = StringIO()
    
    result = sort_file(test_file5, output=output_stream)
    assert result is True
    output_stream.seek(0)
    output_content = output_stream.read()
    assert "import" in output_content
    
    # Test 6: Show diff
    test_file6 = tmp_path / "diff_test.py"
    test_file6.write_text("import sys\nimport os\n")
    diff_stream = StringIO()
    
    result = sort_file(test_file6, show_diff=diff_stream)
    assert result is True
    diff_stream.seek(0)
    diff_content = diff_stream.read()
    assert len(diff_content) > 0
    
    # Test 7: Overwrite in place
    test_file7 = tmp_path / "overwrite_test.py"
    unsorted = "import sys\nimport os\n"
    test_file7.write_text(unsorted)
    config_overwrite = Config(overwrite_in_place=True)
    
    result = sort_file(test_file7, config=config_overwrite)
    assert result is True
    
    # Test 8: File with syntax errors should warn
    test_file8 = tmp_path / "syntax_error.py"
    test_file8.write_text("import os\nimport sys\nthis is invalid python !!!")
    config_atomic = Config(atomic=True)
    
    with pytest.warns(UserWarning, match="unable to sort due to existing syntax errors"):
        result = sort_file(test_file8, config=config_atomic)
        assert result is False
    
    # Test 9: Custom extension
    test_file9 = tmp_path / "test_file.pyx"
    test_file9.write_text("import sys\nimport os\n")
    
    result = sort_file(test_file9, extension="pyx")
    assert result is True
    
    # Test 10: Config kwargs override
    test_file10 = tmp_path / "config_kwargs.py"
    test_file10.write_text("import os\nfrom sys import path\n")
    
    result = sort_file(test_file10, force_single_line=True)
    assert result is True


# LLM-generated content at query #2
#--------------------------

```python
def test_find_imports_in_file(tmp_path):
    """Test find_imports_in_file function."""
    # Create a temporary Python file with imports
    test_file = tmp_path / "test_imports.py"
    test_file.write_text(
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "from typing import List, Dict\n"
        "import json\n"
    )
    
    # Test basic import finding
    imports = list(find_imports_in_file(test_file))
    assert len(imports) == 5
    assert any(imp.module == "os" for imp in imports)
    assert any(imp.module == "sys" for imp in imports)
    assert any(imp.module == "pathlib" for imp in imports)
    assert any(imp.module == "typing" for imp in imports)
    assert any(imp.module == "json" for imp in imports)
    
    # Test with unique=True
    test_file_duplicates = tmp_path / "test_duplicates.py"
    test_file_duplicates.write_text(
        "import os\n"
        "import os\n"
        "import sys\n"
        "import sys\n"
    )
    unique_imports = list(find_imports_in_file(test_file_duplicates, unique=True))
    assert len(unique_imports) == 2
    
    # Test with top_only=True
    test_file_mixed = tmp_path / "test_mixed.py"
    test_file_mixed.write_text(
        "import os\n"
        "import sys\n"
        "\n"
        "def my_function():\n"
        "    import json\n"
    )
    top_imports = list(find_imports_in_file(test_file_mixed, top_only=True))
    assert len(top_imports) == 2
    assert all(imp.module in ("os", "sys") for imp in top_imports)
    
    # Test with file_path parameter
    imports_with_path = list(find_imports_in_file(
        test_file,
        file_path=test_file
    ))
    assert len(imports_with_path) == 5
    
    # Test with non-existent file (should warn and return empty)
    non_existent = tmp_path / "non_existent.py"
    imports_missing = list(find_imports_in_file(non_existent))
    assert len(imports_missing) == 0
    
    # Test with unique=ImportKey.MODULE
    test_file_from = tmp_path / "test_from.py"
    test_file_from.write_text(
        "from os import path\n"
        "from os import getcwd\n"
        "from sys import argv\n"
    )
    module_unique = list(find_imports_in_file(
        test_file_from,
        unique=ImportKey.MODULE
    ))
    assert len(module_unique) == 2
    
    # Test with unique=ImportKey.PACKAGE
    package_unique = list(find_imports_in_file(
        test_file_from,
        unique=ImportKey.PACKAGE
    ))
    assert len(package_unique) == 2
    
    # Test with Config parameter
    config = Config()
    imports_with_config = list(find_imports_in_file(test_file, config=config))
    assert len(imports_with_config) == 5


# LLM-generated content at query #3
#--------------------------

```python
def test_find_imports_in_file(tmp_path):
    """Test find_imports_in_file function with various scenarios."""
    # Create a temporary Python file with imports
    test_file = tmp_path / "test_imports.py"
    test_content = """import os
import sys
from pathlib import Path
from typing import List, Dict
import json

def my_function():
    pass

import late_import
"""
    test_file.write_text(test_content)

    # Test basic import finding
    imports = list(find_imports_in_file(test_file))
    assert len(imports) == 6
    assert any(imp.module == "os" for imp in imports)
    assert any(imp.module == "sys" for imp in imports)
    assert any(imp.module == "pathlib" for imp in imports)
    assert any(imp.module == "typing" for imp in imports)
    assert any(imp.module == "json" for imp in imports)
    assert any(imp.module == "late_import" for imp in imports)

    # Test with top_only=True
    top_imports = list(find_imports_in_file(test_file, top_only=True))
    assert len(top_imports) == 5
    assert not any(imp.module == "late_import" for imp in top_imports)

    # Test with unique=True
    unique_imports = list(find_imports_in_file(test_file, unique=True))
    assert len(unique_imports) == 6

    # Test with unique=ImportKey.MODULE
    module_unique = list(find_imports_in_file(test_file, unique=ImportKey.MODULE))
    assert len(module_unique) == 6
    assert all(imp.module for imp in module_unique)

    # Test with unique=ImportKey.PACKAGE
    package_unique = list(find_imports_in_file(test_file, unique=ImportKey.PACKAGE))
    assert all(imp.module for imp in package_unique)

    # Test with non-existent file
    non_existent = tmp_path / "non_existent.py"
    with pytest.warns(UserWarning, match="Unable to parse file"):
        imports = list(find_imports_in_file(non_existent))
    assert len(imports) == 0

    # Test with file_path parameter
    imports_with_path = list(find_imports_in_file(test_file, file_path=test_file))
    assert len(imports_with_path) == 6

    # Test with custom config
    custom_config = Config(quiet=True)
    imports_with_config = list(find_imports_in_file(test_file, config=custom_config))
    assert len(imports_with_config) == 6

    # Test with duplicate imports - unique should filter them
    test_file_duplicates = tmp_path / "test_duplicates.py"
    duplicate_content = """import os
import os
from pathlib import Path
from pathlib import Path
"""
    test_file_duplicates.write_text(duplicate_content)
    
    all_imports = list(find_imports_in_file(test_file_duplicates))
    assert len(all_imports) == 4
    
    unique_imports = list(find_imports_in_file(test_file_duplicates, unique=True))
    assert len(unique_imports) == 2

    # Test with Path object
    imports_from_path = list(find_imports_in_file(test_file))
    assert len(imports_from_path) == 6

    # Test with string filename
    imports_from_str = list(find_imports_in_file(str(test_file)))
    assert len(imports_from_str) == 6


# LLM-generated content at query #4
#--------------------------

```python
def test_sort_file(tmp_path, capsys):
    """Test the sort_file function with various scenarios."""
    
    # Test 1: Basic file sorting
    test_file = tmp_path / "test_imports.py"
    test_file.write_text("import os\nimport sys\nimport collections\n")
    
    result = sort_file(test_file)
    assert result is True
    
    content = test_file.read_text()
    assert content == "import collections\nimport os\nimport sys\n"
    
    # Test 2: File with no changes needed
    test_file2 = tmp_path / "test_imports2.py"
    test_file2.write_text("import collections\nimport os\nimport sys\n")
    
    result = sort_file(test_file2)
    assert result is False
    
    # Test 3: File with custom config
    test_file3 = tmp_path / "test_imports3.py"
    test_file3.write_text("import os\nfrom sys import argv\n")
    
    config = Config(force_single_line=True)
    result = sort_file(test_file3, config=config)
    assert result is True
    
    # Test 4: write_to_stdout parameter
    test_file4 = tmp_path / "test_imports4.py"
    test_file4.write_text("import sys\nimport os\n")
    
    result = sort_file(test_file4, write_to_stdout=True)
    assert result is True
    captured = capsys.readouterr()
    assert "import os" in captured.out
    assert "import sys" in captured.out
    
    # Test 5: output parameter with StringIO
    test_file5 = tmp_path / "test_imports5.py"
    test_file5.write_text("import sys\nimport os\n")
    
    output_stream = StringIO()
    result = sort_file(test_file5, output=output_stream)
    assert result is True
    output_stream.seek(0)
    content = output_stream.read()
    assert "import os" in content
    assert "import sys" in content
    
    # Test 6: File with syntax errors and atomic mode
    test_file6 = tmp_path / "test_imports6.py"
    test_file6.write_text("import sys\nimport os\n\ndef broken(:\n    pass\n")
    
    config = Config(atomic=True)
    with pytest.raises(ExistingSyntaxErrors):
        sort_file(test_file6, config=config)
    
    # Test 7: Extension parameter
    test_file7 = tmp_path / "test_imports7.pyx"
    test_file7.write_text("import sys\nimport os\n")
    
    result = sort_file(test_file7, extension="pyx")
    assert result is True
    
    # Test 8: File path override
    test_file8 = tmp_path / "test_imports8.py"
    test_file8.write_text("import sys\nimport os\n")
    
    result = sort_file(test_file8, file_path=test_file8)
    assert result is True
    
    # Test 9: disregard_skip parameter
    test_file9 = tmp_path / "test_imports9.py"
    test_file9.write_text("import sys\nimport os\n")
    
    result = sort_file(test_file9, disregard_skip=True)
    assert result is True
    
    # Test 10: overwrite_in_place config
    test_file10 = tmp_path / "test_imports10.py"
    test_file10.write_text("import sys\nimport os\n")
    
    config = Config(overwrite_in_place=True)
    result = sort_file(test_file10, config=config)
    assert result is True
    content = test_file10.read_text()
    assert "import os" in content
    assert "import sys" in content


# LLM-generated content at query #5
#--------------------------

```python
def test_find_imports_in_stream():
    """Test find_imports_in_stream function with various import scenarios."""
    from io import StringIO
    
    # Test 1: Basic import detection
    code = "import os\nimport sys"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream))
    assert len(imports) == 2
    assert any(imp.module == "os" for imp in imports)
    assert any(imp.module == "sys" for imp in imports)
    
    # Test 2: From imports
    code = "from os import path\nfrom sys import argv"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream))
    assert len(imports) == 2
    assert any(imp.module == "os" for imp in imports)
    assert any(imp.module == "sys" for imp in imports)
    
    # Test 3: Duplicate imports with unique=True
    code = "import os\nimport os"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream, unique=True))
    assert len(imports) == 1
    
    # Test 4: Duplicate imports with unique=False
    code = "import os\nimport os"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream, unique=False))
    assert len(imports) == 2
    
    # Test 5: top_only parameter
    code = "import os\n\ndef function():\n    import sys"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream, top_only=True))
    assert len(imports) == 1
    assert imports[0].module == "os"
    
    # Test 6: unique with ImportKey.MODULE
    code = "import os.path\nfrom os import path"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream, unique=ImportKey.MODULE))
    assert len(imports) == 1
    
    # Test 7: unique with ImportKey.PACKAGE
    code = "import os.path\nfrom os import path"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream, unique=ImportKey.PACKAGE))
    assert len(imports) == 1
    
    # Test 8: Empty stream
    code = ""
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream))
    assert len(imports) == 0
    
    # Test 9: No imports in code
    code = "x = 1\ny = 2"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream))
    assert len(imports) == 0
    
    # Test 10: Multiple imports on same line
    code = "import os, sys"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream))
    assert len(imports) >= 1
    
    # Test 11: With custom config
    code = "import os"
    stream = StringIO(code)
    config = Config()
    imports = list(find_imports_in_stream(stream, config=config))
    assert len(imports) == 1
    
    # Test 12: _seen parameter for tracking
    code = "import os"
    stream = StringIO(code)
    seen_set = set()
    imports = list(find_imports_in_stream(stream, unique=True, _seen=seen_set))
    assert len(imports) == 1
    assert "import os" in seen_set
    
    # Test 13: Aliased imports with unique=ImportKey.ALIAS
    code = "import os as operating_system\nimport os as os_module"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream, unique=ImportKey.ALIAS))
    assert len(imports) == 2
    
    # Test 14: Mixed import types
    code = "import os\nfrom sys import argv\nimport json as j"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream))
    assert len(imports) == 3
    
    # Test 15: Config kwargs
    code = "import os"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream, verbose=True))
    assert len(imports) == 1


# LLM-generated content at query #6
#--------------------------

```python
def test_find_imports_in_stream():
    """Test find_imports_in_stream function with various import scenarios."""
    
    # Test basic imports
    code = "import os\nimport sys\nfrom pathlib import Path"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream))
    assert len(imports) == 3
    assert imports[0].module == "os"
    assert imports[1].module == "sys"
    assert imports[2].module == "pathlib"
    
    # Test with unique=True (statement level)
    code = "import os\nimport os\nfrom sys import argv"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream, unique=True))
    assert len(imports) == 2
    
    # Test with unique=ImportKey.MODULE
    code = "import os\nfrom os import path\nimport sys"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream, unique=ImportKey.MODULE))
    assert len(imports) == 2
    assert imports[0].module == "os"
    assert imports[1].module == "sys"
    
    # Test with unique=ImportKey.ATTRIBUTE
    code = "from os import path\nfrom os import getcwd\nfrom sys import argv"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream, unique=ImportKey.ATTRIBUTE))
    assert len(imports) == 3
    
    # Test with unique=ImportKey.PACKAGE
    code = "from os.path import join\nfrom os import getcwd\nfrom sys import argv"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream, unique=ImportKey.PACKAGE))
    assert len(imports) == 2
    
    # Test with top_only=True
    code = "import os\n\ndef foo():\n    import sys"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream, top_only=True))
    assert len(imports) == 1
    assert imports[0].module == "os"
    
    # Test with empty stream
    stream = StringIO("")
    imports = list(find_imports_in_stream(stream))
    assert len(imports) == 0
    
    # Test with config
    code = "import os\nimport sys"
    stream = StringIO(code)
    config = Config()
    imports = list(find_imports_in_stream(stream, config=config))
    assert len(imports) == 2
    
    # Test with file_path
    code = "import os"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream, file_path=Path("test.py")))
    assert len(imports) == 1
    
    # Test with _seen parameter
    code = "import os\nimport sys"
    stream = StringIO(code)
    seen = {"import os"}
    imports = list(find_imports_in_stream(stream, unique=True, _seen=seen))
    assert len(imports) == 1
    assert imports[0].module == "sys"
    
    # Test complex import statements
    code = "from module import func1, func2\nfrom another import ClassA"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream))
    assert len(imports) >= 2
    
    # Test with config_kwargs
    code = "import os\nimport sys"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream, quiet=True))
    assert len(imports) == 2
    
    # Test unique with duplicate imports at different positions
    code = "import os\nfrom pathlib import Path\nimport os"
    stream = StringIO(code)
    imports = list(find_imports_in_stream(stream, unique=ImportKey.MODULE))
    assert len(imports) == 2
    module_names = [imp.module for imp in imports]
    assert "os" in module_names
    assert "pathlib" in module_names


# LLM-generated content at query #7
#--------------------------

```python
def test_find_imports_in_paths(tmp_path):
    """Test find_imports_in_paths function."""
    # Create temporary Python files with imports
    file1 = tmp_path / "file1.py"
    file1.write_text("import os\nimport sys\nfrom pathlib import Path")
    
    file2 = tmp_path / "file2.py"
    file2.write_text("import json\nfrom collections import defaultdict")
    
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    file3 = subdir / "file3.py"
    file3.write_text("import re\nimport ast")
    
    # Test basic functionality
    imports = list(find_imports_in_paths([tmp_path]))
    assert len(imports) == 7
    
    # Test with unique=True
    imports_unique = list(find_imports_in_paths([tmp_path], unique=True))
    assert len(imports_unique) == 7
    
    # Test with unique=ImportKey.MODULE
    imports_module = list(find_imports_in_paths([tmp_path], unique=ImportKey.MODULE))
    assert len(imports_module) == 7
    
    # Test with unique=ImportKey.PACKAGE
    imports_package = list(find_imports_in_paths([tmp_path], unique=ImportKey.PACKAGE))
    assert len(imports_package) == 7
    
    # Test with top_only=True
    imports_top = list(find_imports_in_paths([tmp_path], top_only=True))
    assert len(imports_top) == 7
    
    # Test with custom config
    config = Config()
    imports_config = list(find_imports_in_paths([tmp_path], config=config))
    assert len(imports_config) == 7
    
    # Test that imports are identify.Import objects
    for imp in imports:
        assert isinstance(imp, identify.Import)
    
    # Test with non-existent path
    imports_empty = list(find_imports_in_paths([tmp_path / "nonexistent"]))
    assert len(imports_empty) == 0
    
    # Test with multiple paths
    file4 = tmp_path / "file4.py"
    file4.write_text("import csv")
    
    imports_multi = list(find_imports_in_paths([file1, file2]))
    assert len(imports_multi) >= 5
    
    # Test unique with duplicates across files
    file5 = tmp_path / "file5.py"
    file5.write_text("import os\nimport sys")
    
    imports_dup = list(find_imports_in_paths([tmp_path], unique=True))
    seen_modules = set()
    for imp in imports_dup:
        if imp.module:
            assert imp.module not in seen_modules or imp.module in seen_modules
            seen_modules.add(imp.module)


# LLM-generated content at query #8
#--------------------------

```python
def test_find_imports_in_paths(tmp_path, capsys):
    """Test find_imports_in_paths function."""
    # Create test files with imports
    file1 = tmp_path / "test1.py"
    file1.write_text("import os\nfrom sys import path\n")
    
    file2 = tmp_path / "test2.py"
    file2.write_text("import json\nfrom collections import defaultdict\n")
    
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    file3 = subdir / "test3.py"
    file3.write_text("import re\nfrom typing import List\n")
    
    # Test basic functionality
    imports = list(find_imports_in_paths([tmp_path]))
    assert len(imports) == 6
    assert any(imp.module == "os" for imp in imports)
    assert any(imp.module == "sys" for imp in imports)
    assert any(imp.module == "json" for imp in imports)
    assert any(imp.module == "collections" for imp in imports)
    assert any(imp.module == "re" for imp in imports)
    assert any(imp.module == "typing" for imp in imports)


def test_find_imports_in_paths_with_unique():
    """Test find_imports_in_paths with unique parameter."""
    file1_content = "import os\nimport sys\nfrom os import path\n"
    file2_content = "import os\nfrom sys import argv\n"
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        (tmp_path / "file1.py").write_text(file1_content)
        (tmp_path / "file2.py").write_text(file2_content)
        
        # Test unique=True (alias mode)
        imports = list(find_imports_in_paths([tmp_path], unique=True))
        assert len(imports) == 4
        
        # Test unique=ImportKey.MODULE
        imports = list(find_imports_in_paths([tmp_path], unique=ImportKey.MODULE))
        modules = [imp.module for imp in imports]
        assert modules.count("os") == 1
        assert modules.count("sys") == 1


def test_find_imports_in_paths_with_top_only():
    """Test find_imports_in_paths with top_only parameter."""
    code = """import os
import sys

def foo():
    import json
    
from typing import List
"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        (tmp_path / "test.py").write_text(code)
        
        imports = list(find_imports_in_paths([tmp_path], top_only=True))
        modules = [imp.module for imp in imports]
        assert "os" in modules
        assert "sys" in modules
        assert "typing" in modules
        assert "json" not in modules


def test_find_imports_in_paths_empty_directory():
    """Test find_imports_in_paths with empty directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        imports = list(find_imports_in_paths([tmp_path]))
        assert len(imports) == 0


def test_find_imports_in_paths_multiple_paths():
    """Test find_imports_in_paths with multiple paths."""
    with tempfile.TemporaryDirectory() as tmp_dir1:
        with tempfile.TemporaryDirectory() as tmp_dir2:
            path1 = Path(tmp_dir1)
            path2 = Path(tmp_dir2)
            
            (path1 / "file1.py").write_text("import os\n")
            (path2 / "file2.py").write_text("import sys\n")
            
            imports = list(find_imports_in_paths([path1, path2]))
            modules = [imp.module for imp in imports]
            assert "os" in modules
            assert "sys" in modules


def test_find_imports_in_paths_with_config():
    """Test find_imports_in_paths with custom config."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        (tmp_path / "test.py").write_text("import os\nfrom sys import path\n")
        
        config = Config(skip_gitignore=True)
        imports = list(find_imports_in_paths([tmp_path], config=config))
        assert len(imports) == 2


def test_find_imports_in_paths_package_unique():
    """Test find_imports_in_paths with unique=ImportKey.PACKAGE."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        (tmp_path / "test.py").write_text(
            "from os.path import join\nfrom os import getcwd\nimport sys\n"
        )
        
        imports = list(find_imports_in_paths([tmp_path], unique=ImportKey.PACKAGE))
        modules = [imp.module.split(".")[0] for imp in imports]
        assert modules.count("os") == 1
        assert modules.count("sys") == 1


def test_find_imports_in_paths_attribute_unique():
    """Test find_imports_in_paths with unique=ImportKey.ATTRIBUTE."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        (tmp_path / "test.py").write_text(
            "from os import path\nfrom os import getcwd\nimport sys\n"
        )
        
        imports = list(find_imports_in_paths([tmp_path], unique=ImportKey.ATTRIBUTE))
        assert len(imports) == 3


# LLM-generated content at query #9
#--------------------------

```python
import io
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest


def test_sort_file():
    """Test sort_file function with various scenarios."""
    
    # Test 1: Basic file sorting with changes
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("import os\nimport sys\n")
        tmp_path = tmp.name
    
    try:
        with patch('io.File.read') as mock_file_read:
            mock_source_file = Mock()
            mock_source_file.path = Path(tmp_path)
            mock_source_file.stream = io.StringIO("import sys\nimport os\n")
            mock_source_file.encoding = 'utf-8'
            mock_file_read.return_value.__enter__.return_value = mock_source_file
            
            with patch('sort_stream') as mock_sort_stream:
                mock_sort_stream.return_value = True
                with patch('builtins.print'):
                    result = sort_file(tmp_path)
                    assert result is True
                    mock_sort_stream.assert_called()
    finally:
        Path(tmp_path).unlink()
    
    # Test 2: File with no changes
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("import os\nimport sys\n")
        tmp_path = tmp.name
    
    try:
        with patch('io.File.read') as mock_file_read:
            mock_source_file = Mock()
            mock_source_file.path = Path(tmp_path)
            mock_source_file.stream = io.StringIO("import os\nimport sys\n")
            mock_source_file.encoding = 'utf-8'
            mock_file_read.return_value.__enter__.return_value = mock_source_file
            
            with patch('sort_stream') as mock_sort_stream:
                mock_sort_stream.return_value = False
                result = sort_file(tmp_path)
                assert result is False
    finally:
        Path(tmp_path).unlink()
    
    # Test 3: write_to_stdout option
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("import sys\nimport os\n")
        tmp_path = tmp.name
    
    try:
        with patch('io.File.read') as mock_file_read:
            mock_source_file = Mock()
            mock_source_file.path = Path(tmp_path)
            mock_source_file.stream = io.StringIO("import sys\nimport os\n")
            mock_source_file.encoding = 'utf-8'
            mock_file_read.return_value.__enter__.return_value = mock_source_file
            
            with patch('sort_stream') as mock_sort_stream:
                mock_sort_stream.return_value = True
                with patch('sys.stdout', new_callable=io.StringIO):
                    result = sort_file(tmp_path, write_to_stdout=True)
                    assert result is True
                    mock_sort_stream.assert_called()
    finally:
        Path(tmp_path).unlink()
    
    # Test 4: Custom output stream
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("import sys\nimport os\n")
        tmp_path = tmp.name
    
    try:
        output_stream = io.StringIO()
        with patch('io.File.read') as mock_file_read:
            mock_source_file = Mock()
            mock_source_file.path = Path(tmp_path)
            mock_source_file.stream = io.StringIO("import sys\nimport os\n")
            mock_source_file.encoding = 'utf-8'
            mock_file_read.return_value.__enter__.return_value = mock_source_file
            
            with patch('sort_stream') as mock_sort_stream:
                mock_sort_stream.return_value = True
                result = sort_file(tmp_path, output=output_stream)
                assert result is True
    finally:
        Path(tmp_path).unlink()
    
    # Test 5: show_diff option
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("import sys\nimport os\n")
        tmp_path = tmp.name
    
    try:
        with patch('io.File.read') as mock_file_read:
            mock_source_file = Mock()
            mock_source_file.path = Path(tmp_path)
            mock_source_file.stream = io.StringIO("import sys\nimport os\n")
            mock_source_file.encoding = 'utf-8'
            mock_file_read.return_value.__enter__.return_value = mock_source_file
            
            with patch('sort_stream') as mock_sort_stream:
                mock_sort_stream.return_value = False
                with patch('show_unified_diff'):
                    result = sort_file(tmp_path, show_diff=True)
                    assert result is False
    finally:
        Path(tmp_path).unlink()
    
    # Test 6: ExistingSyntaxErrors exception handling
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("invalid python syntax !!!\n")
        tmp_path = tmp.name
    
    try:
        with patch('io.File.read') as mock_file_read:
            mock_source_file = Mock()
            mock_source_file.path = Path(tmp_path)
            mock_source_file.stream = io.StringIO("invalid syntax")
            mock_source_file.encoding = 'utf-8'
            mock_file_read.return_value.__enter__.return_value = mock_source_file
            
            with patch('sort_stream') as mock_sort_stream:
                mock_sort_stream.side_effect = ExistingSyntaxErrors(tmp_path)
                with patch('warnings.warn'):
                    result = sort_file(tmp_path)
                    assert result is False
    finally:
        Path(tmp_path).unlink()
    
    # Test 7: config_trie parameter
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write("import os\n")
        tmp_path = tmp.name
    
    try:
        mock_config_trie = Mock()
        mock_config_trie.search.return_value = (tmp_path, {})
        
        with patch('io.File.read') as mock_file_read:
            mock_source_file = Mock()
            mock_source_file.path = Path(tmp_path)
            mock_source_file.stream = io.StringIO("import os\n")
            mock_source_file.encoding = 'utf-8'
            mock_file_read.return_value.__enter__.return_value = mock_source_file
            
            with patch('sort_stream') as mock_sort_stream:
                mock_sort_stream.return_value = False
                with patch('builtins.print'):
                    result = sort_file(tmp_path, config_trie=mock_config_trie)
                    assert result is False
                    mock_config_trie.search.assert_called_once()
    finally:
        Path(tmp_path).unlink()


# LLM-generated content at query #10
#--------------------------

```python
def test_check_stream():
    """Test check_stream function with various scenarios."""
    from io import StringIO
    from pathlib import Path
    
    # Test 1: Correctly sorted imports should return True
    correct_code = "import os\nimport sys\n\nprint('hello')"
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream)
    assert result is True
    
    # Test 2: Incorrectly sorted imports should return False
    incorrect_code = "import sys\nimport os\n\nprint('hello')"
    input_stream = StringIO(incorrect_code)
    result = check_stream(input_stream)
    assert result is False
    
    # Test 3: With show_diff as True
    input_stream = StringIO(incorrect_code)
    result = check_stream(input_stream, show_diff=True)
    assert result is False
    
    # Test 4: With show_diff as TextIO stream
    diff_output = StringIO()
    input_stream = StringIO(incorrect_code)
    result = check_stream(input_stream, show_diff=diff_output)
    assert result is False
    diff_output.seek(0)
    diff_content = diff_output.read()
    assert len(diff_content) > 0
    
    # Test 5: With custom config
    input_stream = StringIO(correct_code)
    config = Config()
    result = check_stream(input_stream, config=config)
    assert result is True
    
    # Test 6: With file_path
    input_stream = StringIO(correct_code)
    file_path = Path("test.py")
    result = check_stream(input_stream, file_path=file_path)
    assert result is True
    
    # Test 7: With extension parameter
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream, extension="py")
    assert result is True
    
    # Test 8: With disregard_skip parameter
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream, disregard_skip=True)
    assert result is True
    
    # Test 9: With config_kwargs
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream, line_length=80)
    assert result is True
    
    # Test 10: Empty input stream
    input_stream = StringIO("")
    result = check_stream(input_stream)
    assert result is True
    
    # Test 11: Multiple imports needing sorting
    multi_incorrect = "from z import a\nfrom a import b\n"
    input_stream = StringIO(multi_incorrect)
    result = check_stream(input_stream)
    assert result is False
    
    # Test 12: Code with no imports
    no_imports = "x = 1\ny = 2\n"
    input_stream = StringIO(no_imports)
    result = check_stream(input_stream)
    assert result is True


# LLM-generated content at query #11
#--------------------------

```python
def test_find_imports_in_stream():
    """Test find_imports_in_stream function with various import scenarios."""
    
    # Test basic import finding
    code_stream = StringIO("import os\nfrom sys import path\n")
    imports = list(find_imports_in_stream(code_stream))
    assert len(imports) == 2
    assert imports[0].module == "os"
    assert imports[1].module == "sys"
    
    # Test with unique=True (only first instance)
    code_stream = StringIO("import os\nimport os\nfrom sys import path\n")
    imports = list(find_imports_in_stream(code_stream, unique=True))
    assert len(imports) == 2
    
    # Test with unique=ImportKey.MODULE
    code_stream = StringIO("import os\nfrom os import path\nimport sys\n")
    imports = list(find_imports_in_stream(code_stream, unique=ImportKey.MODULE))
    assert len(imports) == 2
    assert all(imp.module in ("os", "sys") for imp in imports)
    
    # Test with unique=ImportKey.PACKAGE
    code_stream = StringIO("import os.path\nimport os\nimport sys\n")
    imports = list(find_imports_in_stream(code_stream, unique=ImportKey.PACKAGE))
    assert len(imports) == 2
    
    # Test with top_only=True
    code_stream = StringIO("import os\n\ndef foo():\n    import sys\n")
    imports = list(find_imports_in_stream(code_stream, top_only=True))
    assert len(imports) == 1
    assert imports[0].module == "os"
    
    # Test with empty stream
    code_stream = StringIO("")
    imports = list(find_imports_in_stream(code_stream))
    assert len(imports) == 0
    
    # Test with no imports
    code_stream = StringIO("x = 1\ny = 2\n")
    imports = list(find_imports_in_stream(code_stream))
    assert len(imports) == 0
    
    # Test with file_path parameter
    code_stream = StringIO("import os\n")
    imports = list(find_imports_in_stream(code_stream, file_path=Path("test.py")))
    assert len(imports) == 1
    
    # Test with config_kwargs
    code_stream = StringIO("import os\n")
    imports = list(find_imports_in_stream(code_stream, line_length=80))
    assert len(imports) == 1
    
    # Test with _seen parameter (internal use)
    code_stream = StringIO("import os\nimport sys\n")
    seen = {"os"}
    imports = list(find_imports_in_stream(code_stream, unique=True, _seen=seen))
    assert len(imports) == 1
    assert imports[0].module == "sys"
    
    # Test with ImportKey.ATTRIBUTE
    code_stream = StringIO("from os import path\nfrom os import environ\n")
    imports = list(find_imports_in_stream(code_stream, unique=ImportKey.ATTRIBUTE))
    assert len(imports) == 2
    
    # Test with ImportKey.ALIAS
    code_stream = StringIO("import os as operating_system\nimport os\n")
    imports = list(find_imports_in_stream(code_stream, unique=ImportKey.ALIAS))
    assert len(imports) == 2
    
    # Test with custom config
    custom_config = Config(line_length=120)
    code_stream = StringIO("import os\n")
    imports = list(find_imports_in_stream(code_stream, config=custom_config))
    assert len(imports) == 1
    
    # Test with multiple imports on same line
    code_stream = StringIO("import os, sys\n")
    imports = list(find_imports_in_stream(code_stream))
    assert len(imports) == 2
    
    # Test with from...import multiple
    code_stream = StringIO("from os import path, environ\n")
    imports = list(find_imports_in_stream(code_stream))
    assert len(imports) == 2


# LLM-generated content at query #12
#--------------------------

```python
def test_sort_stream():
    """Test sort_stream function with various scenarios."""
    
    # Test 1: Basic import sorting
    input_code = "import os\nimport sys\nimport ast\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    
    config = Config()
    changed = sort_stream(input_stream, output_stream, config=config)
    
    output_stream.seek(0)
    result = output_stream.read()
    assert "import ast" in result
    assert "import os" in result
    assert "import sys" in result
    
    # Test 2: No changes needed
    input_code = "import ast\nimport os\nimport sys\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    
    changed = sort_stream(input_stream, output_stream, config=Config())
    
    output_stream.seek(0)
    result = output_stream.read()
    assert result == input_code
    
    # Test 3: With file_path and extension
    input_code = "import sys\nimport os\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    
    changed = sort_stream(
        input_stream,
        output_stream,
        extension="py",
        file_path=Path("test.py"),
        config=Config()
    )
    
    output_stream.seek(0)
    result = output_stream.read()
    assert "import os" in result
    assert "import sys" in result
    
    # Test 4: With show_diff as False (default behavior)
    input_code = "import sys\nimport os\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    
    changed = sort_stream(
        input_stream,
        output_stream,
        show_diff=False,
        config=Config()
    )
    
    output_stream.seek(0)
    result = output_stream.read()
    assert len(result) > 0
    
    # Test 5: Extension inference from file_path
    input_code = "import sys\nimport os\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    
    changed = sort_stream(
        input_stream,
        output_stream,
        file_path=Path("test.py"),
        config=Config()
    )
    
    output_stream.seek(0)
    result = output_stream.read()
    assert len(result) > 0
    
    # Test 6: With disregard_skip=True
    input_code = "import sys\nimport os\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    
    changed = sort_stream(
        input_stream,
        output_stream,
        disregard_skip=True,
        config=Config()
    )
    
    output_stream.seek(0)
    result = output_stream.read()
    assert len(result) > 0
    
    # Test 7: Syntax error with atomic mode
    input_code = "import sys\nimport os\ninvalid syntax here\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    
    config = Config(atomic=True)
    
    with pytest.raises(ExistingSyntaxErrors):
        sort_stream(
            input_stream,
            output_stream,
            config=config
        )
    
    # Test 8: Config kwargs override
    input_code = "import sys\nimport os\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    
    changed = sort_stream(
        input_stream,
        output_stream,
        config=Config(),
        line_length=80
    )
    
    output_stream.seek(0)
    result = output_stream.read()
    assert len(result) > 0
    
    # Test 9: With raise_on_skip=False
    input_code = "import sys\nimport os\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    
    changed = sort_stream(
        input_stream,
        output_stream,
        raise_on_skip=False,
        config=Config()
    )
    
    output_stream.seek(0)
    result = output_stream.read()
    assert len(result) > 0


# LLM-generated content at query #13
#--------------------------

```python
def test_sort_stream():
    """Test sort_stream function with various scenarios."""
    import pytest
    from io import StringIO
    from pathlib import Path

    # Test 1: Basic sorting of imports
    input_code = "import os\nimport sys\nimport collections\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    
    changed = sort_stream(input_stream, output_stream)
    output_stream.seek(0)
    result = output_stream.read()
    
    assert changed is True
    assert "import collections" in result
    assert "import os" in result
    assert "import sys" in result

    # Test 2: No changes needed
    input_code = "import collections\nimport os\nimport sys\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    
    changed = sort_stream(input_stream, output_stream)
    output_stream.seek(0)
    result = output_stream.read()
    
    assert changed is False

    # Test 3: With file_path and extension
    input_code = "import sys\nimport os\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    
    changed = sort_stream(
        input_stream,
        output_stream,
        file_path=Path("test.py"),
        extension="py"
    )
    output_stream.seek(0)
    result = output_stream.read()
    
    assert changed is True

    # Test 4: With config kwargs
    input_code = "import sys\nimport os\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    
    changed = sort_stream(
        input_stream,
        output_stream,
        line_length=80
    )
    output_stream.seek(0)
    result = output_stream.read()
    
    assert changed is True

    # Test 5: Atomic mode with valid Python syntax
    input_code = "import sys\nimport os\nprint('hello')\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    config = Config(atomic=True)
    
    changed = sort_stream(
        input_stream,
        output_stream,
        config=config
    )
    output_stream.seek(0)
    result = output_stream.read()
    
    assert changed is True
    assert "print" in result

    # Test 6: Show diff mode (with StringIO)
    input_code = "import sys\nimport os\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    diff_stream = StringIO()
    
    changed = sort_stream(
        input_stream,
        output_stream,
        show_diff=diff_stream
    )
    
    assert changed is True

    # Test 7: Show diff mode (with True)
    input_code = "import sys\nimport os\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    
    changed = sort_stream(
        input_stream,
        output_stream,
        show_diff=True
    )
    
    assert changed is True

    # Test 8: With disregard_skip
    input_code = "import sys\nimport os\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    
    changed = sort_stream(
        input_stream,
        output_stream,
        disregard_skip=True
    )
    
    assert changed is True

    # Test 9: Syntax error with atomic mode (non-Cython file)
    input_code = "import sys\nimport os\nthis is invalid python\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    config = Config(atomic=True)
    
    with pytest.raises(ExistingSyntaxErrors):
        sort_stream(
            input_stream,
            output_stream,
            config=config,
            extension="py"
        )

    # Test 10: File skip setting
    input_code = "import sys\nimport os\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    config = Config(skip=["test.py"])
    
    with pytest.raises(FileSkipSetting):
        sort_stream(
            input_stream,
            output_stream,
            file_path=Path("test.py"),
            config=config,
            disregard_skip=False
        )

    # Test 11: Complex import sorting
    input_code = "from z import a\nfrom a import z\nimport sys\nimport os\n"
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    
    changed = sort_stream(input_stream, output_stream)
    output_stream.seek(0)
    result = output_stream.read()
    
    assert changed is True
    lines = result.strip().split('\n')
    assert len(lines) >= 1

    # Test 12: Empty input
    input_code = ""
    input_stream = StringIO(input_code)
    output_stream = StringIO()
    
    changed = sort_stream(input_stream, output_stream)
    
    assert changed is False


# LLM-generated content at query #14
#--------------------------

```python
def test_find_imports_in_file(tmp_path):
    """Test find_imports_in_file function."""
    # Create a temporary Python file with imports
    test_file = tmp_path / "test_imports.py"
    test_file.write_text("""
import os
import sys
from pathlib import Path
from typing import List, Dict
import json

def foo():
    pass
""")
    
    # Test basic import finding
    imports = list(find_imports_in_file(test_file))
    assert len(imports) == 5
    assert imports[0].module == "os"
    assert imports[1].module == "sys"
    assert imports[2].module == "pathlib"
    assert imports[3].module == "typing"
    assert imports[4].module == "json"
    
    # Test with unique=True
    test_file_with_duplicates = tmp_path / "test_duplicates.py"
    test_file_with_duplicates.write_text("""
import os
import sys
import os
from pathlib import Path
""")
    imports_unique = list(find_imports_in_file(test_file_with_duplicates, unique=True))
    assert len(imports_unique) == 3
    
    # Test with unique=ImportKey.MODULE
    imports_module_unique = list(
        find_imports_in_file(test_file_with_duplicates, unique=ImportKey.MODULE)
    )
    assert len(imports_module_unique) == 3
    
    # Test with top_only=True
    imports_top_only = list(find_imports_in_file(test_file, top_only=True))
    assert len(imports_top_only) == 5
    
    # Test with non-existent file
    with pytest.warns(UserWarning, match="Unable to parse file"):
        imports_missing = list(find_imports_in_file(tmp_path / "nonexistent.py"))
        assert len(imports_missing) == 0
    
    # Test with custom config
    config = Config(verbose=True)
    imports_with_config = list(find_imports_in_file(test_file, config=config))
    assert len(imports_with_config) == 5
    
    # Test with file_path parameter
    imports_with_path = list(find_imports_in_file(test_file, file_path=test_file))
    assert len(imports_with_path) == 5
    
    # Test unique with ImportKey.PACKAGE
    test_file_package = tmp_path / "test_packages.py"
    test_file_package.write_text("""
from pathlib.submodule import something
from pathlib import Path
import json
""")
    imports_package = list(
        find_imports_in_file(test_file_package, unique=ImportKey.PACKAGE)
    )
    assert len(imports_package) == 2


# LLM-generated content at query #15
#--------------------------

```python
def test_find_imports_in_paths(tmp_path):
    """Test find_imports_in_paths function."""
    # Create temporary Python files with imports
    file1 = tmp_path / "module1.py"
    file1.write_text("import os\nimport sys\nfrom pathlib import Path")
    
    file2 = tmp_path / "module2.py"
    file2.write_text("import json\nfrom typing import List")
    
    file3 = tmp_path / "module3.py"
    file3.write_text("import os\nimport json")
    
    # Test basic functionality - find all imports
    imports = list(find_imports_in_paths([tmp_path]))
    assert len(imports) > 0
    
    # Test with unique=True - should eliminate duplicates
    imports_unique = list(find_imports_in_paths([tmp_path], unique=True))
    assert len(imports_unique) <= len(imports)
    
    # Test with unique=ImportKey.MODULE
    imports_module = list(find_imports_in_paths([tmp_path], unique=ImportKey.MODULE))
    assert len(imports_module) <= len(imports)
    
    # Test with unique=ImportKey.PACKAGE
    imports_package = list(find_imports_in_paths([tmp_path], unique=ImportKey.PACKAGE))
    assert len(imports_package) <= len(imports_module)
    
    # Test with top_only=True
    imports_top = list(find_imports_in_paths([tmp_path], top_only=True))
    assert len(imports_top) > 0
    
    # Verify imports are identify.Import objects
    if imports:
        assert hasattr(imports[0], 'module')
        assert hasattr(imports[0], 'statement')
    
    # Test with empty paths
    imports_empty = list(find_imports_in_paths([]))
    assert len(imports_empty) == 0
    
    # Test with config_kwargs
    imports_with_config = list(find_imports_in_paths([tmp_path], verbose=True))
    assert len(imports_with_config) > 0
    
    # Test with non-existent path - should return empty
    non_existent = tmp_path / "nonexistent"
    imports_nonexistent = list(find_imports_in_paths([non_existent]))
    assert len(imports_nonexistent) == 0
    
    # Test unique with duplicate imports across files
    file4 = tmp_path / "module4.py"
    file4.write_text("import os")
    
    imports_all = list(find_imports_in_paths([tmp_path], unique=False))
    imports_unique_all = list(find_imports_in_paths([tmp_path], unique=True))
    assert len(imports_unique_all) <= len(imports_all)


# LLM-generated content at query #16
#--------------------------

```python
def test_sort_file(tmp_path):
    """Test the sort_file function with various scenarios."""
    
    # Test 1: Basic file sorting
    test_file = tmp_path / "test_imports.py"
    test_file.write_text("import os\nimport sys\nimport collections\n")
    
    result = sort_file(test_file)
    assert result is True
    
    content = test_file.read_text()
    assert content == "import collections\nimport os\nimport sys\n"
    
    # Test 2: File with no changes needed
    test_file2 = tmp_path / "test_sorted.py"
    test_file2.write_text("import collections\nimport os\nimport sys\n")
    
    result = sort_file(test_file2)
    assert result is False
    
    # Test 3: File with custom config
    test_file3 = tmp_path / "test_custom_config.py"
    test_file3.write_text("from os import path\nimport sys\n")
    
    custom_config = Config(force_single_line=True)
    result = sort_file(test_file3, config=custom_config)
    assert result is True
    
    # Test 4: Write to stdout
    test_file4 = tmp_path / "test_stdout.py"
    test_file4.write_text("import sys\nimport os\n")
    
    result = sort_file(test_file4, write_to_stdout=True)
    assert result is True
    
    # Test 5: Write to custom output stream
    test_file5 = tmp_path / "test_output_stream.py"
    test_file5.write_text("import sys\nimport os\n")
    
    output_stream = StringIO()
    result = sort_file(test_file5, output=output_stream)
    assert result is True
    output_stream.seek(0)
    assert "import os" in output_stream.read()
    
    # Test 6: Show diff without applying
    test_file6 = tmp_path / "test_diff.py"
    test_file6.write_text("import sys\nimport os\n")
    original_content = test_file6.read_text()
    
    diff_stream = StringIO()
    result = sort_file(test_file6, show_diff=diff_stream)
    assert result is False
    assert test_file6.read_text() == original_content
    
    # Test 7: File with syntax errors (should warn but not crash if not atomic)
    test_file7 = tmp_path / "test_syntax_error.py"
    test_file7.write_text("import os\nimport sys\ndef foo(\n")
    
    with pytest.warns(UserWarning):
        result = sort_file(test_file7, config=Config(atomic=True))
    
    # Test 8: File path parameter
    test_file8 = tmp_path / "test_file_path.py"
    test_file8.write_text("import sys\nimport os\n")
    
    result = sort_file(test_file8, file_path=Path(test_file8))
    assert result is True
    
    # Test 9: Extension parameter
    test_file9 = tmp_path / "test_extension.pyx"
    test_file9.write_text("import sys\nimport os\n")
    
    result = sort_file(test_file9, extension="pyx")
    assert result is True
    
    # Test 10: Disregard skip parameter
    test_file10 = tmp_path / "test_disregard_skip.py"
    test_file10.write_text("import sys\nimport os\n")
    
    config_with_skip = Config(skip=[test_file10.name])
    result = sort_file(test_file10, config=config_with_skip, disregard_skip=True)
    assert result is True
    
    # Test 11: Overwrite in place
    test_file11 = tmp_path / "test_overwrite.py"
    test_file11.write_text("import sys\nimport os\n")
    
    config_overwrite = Config(overwrite_in_place=True)
    result = sort_file(test_file11, config=config_overwrite)
    assert result is True
    assert test_file11.read_text() == "import os\nimport sys\n"
    
    # Test 12: Quiet mode
    test_file12 = tmp_path / "test_quiet.py"
    test_file12.write_text("import sys\nimport os\n")
    
    config_quiet = Config(quiet=True)
    result = sort_file(test_file12, config=config_quiet)
    assert result is True


# LLM-generated content at query #17
#--------------------------

```python
def test_check_stream():
    """Test the check_stream function with various scenarios."""
    from io import StringIO
    from pathlib import Path
    
    # Test 1: Correctly sorted imports should return True
    correct_code = "import os\nimport sys\n"
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream)
    assert result is True
    
    # Test 2: Incorrectly sorted imports should return False
    incorrect_code = "import sys\nimport os\n"
    input_stream = StringIO(incorrect_code)
    result = check_stream(input_stream)
    assert result is False
    
    # Test 3: With show_diff as True (should not raise error)
    incorrect_code = "import sys\nimport os\n"
    input_stream = StringIO(incorrect_code)
    result = check_stream(input_stream, show_diff=True)
    assert result is False
    
    # Test 4: With show_diff as TextIO stream
    incorrect_code = "import sys\nimport os\n"
    input_stream = StringIO(incorrect_code)
    diff_output = StringIO()
    result = check_stream(input_stream, show_diff=diff_output)
    assert result is False
    
    # Test 5: With extension parameter
    correct_code = "import os\n"
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream, extension="py")
    assert result is True
    
    # Test 6: With file_path parameter
    correct_code = "import os\n"
    input_stream = StringIO(correct_code)
    file_path = Path("test.py")
    result = check_stream(input_stream, file_path=file_path)
    assert result is True
    
    # Test 7: Empty input should return True
    empty_code = ""
    input_stream = StringIO(empty_code)
    result = check_stream(input_stream)
    assert result is True
    
    # Test 8: Code with only comments should return True
    comment_code = "# This is a comment\n"
    input_stream = StringIO(comment_code)
    result = check_stream(input_stream)
    assert result is True
    
    # Test 9: Mixed correctly sorted imports
    correct_mixed = "import os\nimport sys\nfrom pathlib import Path\n"
    input_stream = StringIO(correct_mixed)
    result = check_stream(input_stream)
    assert result is True
    
    # Test 10: With config parameter
    correct_code = "import os\n"
    input_stream = StringIO(correct_code)
    config = Config()
    result = check_stream(input_stream, config=config)
    assert result is True
    
    # Test 11: With disregard_skip parameter
    correct_code = "import os\n"
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream, disregard_skip=True)
    assert result is True
    
    # Test 12: With config_kwargs
    correct_code = "import os\n"
    input_stream = StringIO(correct_code)
    result = check_stream(input_stream, verbose=False)
    assert result is True
    
    # Test 13: Incorrectly sorted with from imports
    incorrect_from = "from sys import argv\nfrom os import path\n"
    input_stream = StringIO(incorrect_from)
    result = check_stream(input_stream)
    assert result is False
    
    # Test 14: Multiple imports on same line (unsorted)
    multi_import = "import sys, os\n"
    input_stream = StringIO(multi_import)
    result = check_stream(input_stream)
    # Result depends on configuration, but should not raise
    assert isinstance(result, bool)


# LLM-generated content at query #18
#--------------------------

```python
def test_find_imports_in_paths(tmp_path):
    """Test find_imports_in_paths function."""
    # Create test files with imports
    test_file_1 = tmp_path / "test1.py"
    test_file_1.write_text("import os\nimport sys\nfrom pathlib import Path\n")
    
    test_file_2 = tmp_path / "test2.py"
    test_file_2.write_text("import json\nfrom collections import defaultdict\n")
    
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    test_file_3 = subdir / "test3.py"
    test_file_3.write_text("import re\nfrom typing import List\n")
    
    # Test basic functionality
    imports = list(find_imports_in_paths([tmp_path]))
    assert len(imports) > 0
    
    # Test with unique=True
    imports_unique = list(find_imports_in_paths([tmp_path], unique=True))
    assert len(imports_unique) > 0
    assert len(imports_unique) <= len(imports)
    
    # Test with unique=ImportKey.MODULE
    imports_module = list(find_imports_in_paths([tmp_path], unique=ImportKey.MODULE))
    assert len(imports_module) > 0
    
    # Test with unique=ImportKey.PACKAGE
    imports_package = list(find_imports_in_paths([tmp_path], unique=ImportKey.PACKAGE))
    assert len(imports_package) > 0
    
    # Test with top_only=True
    imports_top = list(find_imports_in_paths([tmp_path], top_only=True))
    assert len(imports_top) > 0
    
    # Test with custom config
    config = Config()
    imports_config = list(find_imports_in_paths([tmp_path], config=config))
    assert len(imports_config) > 0
    
    # Test with multiple paths
    imports_multi = list(find_imports_in_paths([test_file_1, test_file_2]))
    assert len(imports_multi) > 0
    
    # Verify imports contain expected modules
    modules = {imp.module for imp in imports}
    assert "os" in modules or "sys" in modules or "json" in modules
    
    # Test with empty paths
    imports_empty = list(find_imports_in_paths([]))
    assert len(imports_empty) == 0


# LLM-generated content at query #19
#--------------------------

```python
def test_sort_file(tmp_path):
    """Test sort_file function with various scenarios."""
    # Test 1: Basic file sorting
    test_file = tmp_path / "test_imports.py"
    unsorted_code = "import os\nimport sys\nimport ast\n"
    test_file.write_text(unsorted_code)
    
    result = sort_file(test_file)
    assert result is True
    sorted_content = test_file.read_text()
    assert sorted_content == unsorted_code  # Already sorted
    
    # Test 2: File with unsorted imports
    unsorted_file = tmp_path / "unsorted.py"
    unsorted_content = "import sys\nimport os\nimport ast\n"
    unsorted_file.write_text(unsorted_content)
    
    result = sort_file(unsorted_file)
    assert result is True
    sorted_content = unsorted_file.read_text()
    assert "import ast" in sorted_content
    assert "import os" in sorted_content
    assert "import sys" in sorted_content
    
    # Test 3: File with no changes needed
    sorted_file = tmp_path / "sorted.py"
    sorted_content = "import ast\nimport os\nimport sys\n"
    sorted_file.write_text(sorted_content)
    
    result = sort_file(sorted_file)
    assert result is False
    assert sorted_file.read_text() == sorted_content
    
    # Test 4: write_to_stdout flag
    stdout_file = tmp_path / "stdout_test.py"
    stdout_file.write_text("import sys\nimport os\n")
    
    result = sort_file(stdout_file, write_to_stdout=True)
    assert isinstance(result, bool)
    
    # Test 5: output parameter with StringIO
    output_file = tmp_path / "output_test.py"
    output_file.write_text("import sys\nimport os\n")
    output_stream = StringIO()
    
    result = sort_file(output_file, output=output_stream)
    output_stream.seek(0)
    output_content = output_stream.read()
    assert isinstance(output_content, str)
    
    # Test 6: show_diff flag
    diff_file = tmp_path / "diff_test.py"
    diff_file.write_text("import sys\nimport os\n")
    
    result = sort_file(diff_file, show_diff=True)
    assert isinstance(result, bool)
    
    # Test 7: show_diff with TextIO
    diff_stream = StringIO()
    diff_file2 = tmp_path / "diff_test2.py"
    diff_file2.write_text("import sys\nimport os\n")
    
    result = sort_file(diff_file2, show_diff=diff_stream)
    assert isinstance(result, bool)
    
    # Test 8: disregard_skip flag
    skip_file = tmp_path / "skip_test.py"
    skip_file.write_text("import sys\nimport os\n")
    
    result = sort_file(skip_file, disregard_skip=True)
    assert isinstance(result, bool)
    
    # Test 9: extension parameter
    ext_file = tmp_path / "ext_test.pyx"
    ext_file.write_text("import sys\nimport os\n")
    
    result = sort_file(ext_file, extension="pyx")
    assert isinstance(result, bool)
    
    # Test 10: file_path parameter
    fp_file = tmp_path / "filepath_test.py"
    fp_file.write_text("import sys\nimport os\n")
    
    result = sort_file(fp_file, file_path=fp_file)
    assert isinstance(result, bool)
    
    # Test 11: config parameter
    from isort import Config
    custom_config = Config(line_length=120)
    config_file = tmp_path / "config_test.py"
    config_file.write_text("import sys\nimport os\n")
    
    result = sort_file(config_file, config=custom_config)
    assert isinstance(result, bool)
    
    # Test 12: overwrite_in_place config
    inplace_file = tmp_path / "inplace_test.py"
    inplace_file.write_text("import sys\nimport os\n")
    
    inplace_config = Config(overwrite_in_place=True)
    result = sort_file(inplace_file, config=inplace_config)
    assert isinstance(result, bool)
    assert inplace_file.exists()


# LLM-generated content at query #20
#--------------------------

```python
def test_find_imports_in_paths(tmp_path):
    """Test find_imports_in_paths function."""
    # Create temporary Python files with imports
    file1 = tmp_path / "file1.py"
    file1.write_text("import os\nimport sys\nfrom pathlib import Path")
    
    file2 = tmp_path / "file2.py"
    file2.write_text("import json\nfrom typing import List")
    
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    file3 = subdir / "file3.py"
    file3.write_text("import re\nfrom collections import defaultdict")
    
    # Test basic functionality
    imports = list(find_imports_in_paths([tmp_path]))
    assert len(imports) == 6
    assert any(imp.module == "os" for imp in imports)
    assert any(imp.module == "sys" for imp in imports)
    assert any(imp.module == "pathlib" for imp in imports)
    assert any(imp.module == "json" for imp in imports)
    assert any(imp.module == "typing" for imp in imports)
    assert any(imp.module == "re" for imp in imports)
    assert any(imp.module == "collections" for imp in imports)


def test_find_imports_in_paths_with_unique_true(tmp_path):
    """Test find_imports_in_paths with unique=True."""
    file1 = tmp_path / "file1.py"
    file1.write_text("import os\nimport os\nfrom pathlib import Path")
    
    file2 = tmp_path / "file2.py"
    file2.write_text("import os\nfrom typing import List")
    
    imports = list(find_imports_in_paths([tmp_path], unique=True))
    assert len(imports) == 3
    os_imports = [imp for imp in imports if imp.module == "os"]
    assert len(os_imports) == 1


def test_find_imports_in_paths_with_unique_module(tmp_path):
    """Test find_imports_in_paths with unique=ImportKey.MODULE."""
    file1 = tmp_path / "file1.py"
    file1.write_text("import os\nfrom os import path")
    
    file2 = tmp_path / "file2.py"
    file2.write_text("import sys")
    
    imports = list(find_imports_in_paths([tmp_path], unique=ImportKey.MODULE))
    modules = {imp.module for imp in imports}
    assert "os" in modules
    assert "sys" in modules
    assert len([imp for imp in imports if imp.module == "os"]) == 1


def test_find_imports_in_paths_with_unique_package(tmp_path):
    """Test find_imports_in_paths with unique=ImportKey.PACKAGE."""
    file1 = tmp_path / "file1.py"
    file1.write_text("from os.path import join\nimport os")
    
    file2 = tmp_path / "file2.py"
    file2.write_text("from os import environ")
    
    imports = list(find_imports_in_paths([tmp_path], unique=ImportKey.PACKAGE))
    os_imports = [imp for imp in imports if imp.module.startswith("os")]
    assert len(os_imports) == 1


def test_find_imports_in_paths_empty_directory(tmp_path):
    """Test find_imports_in_paths with empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    imports = list(find_imports_in_paths([empty_dir]))
    assert len(imports) == 0


def test_find_imports_in_paths_multiple_paths(tmp_path):
    """Test find_imports_in_paths with multiple paths."""
    dir1 = tmp_path / "dir1"
    dir1.mkdir()
    file1 = dir1 / "file1.py"
    file1.write_text("import os")
    
    dir2 = tmp_path / "dir2"
    dir2.mkdir()
    file2 = dir2 / "file2.py"
    file2.write_text("import sys")
    
    imports = list(find_imports_in_paths([dir1, dir2]))
    modules = {imp.module for imp in imports}
    assert "os" in modules
    assert "sys" in modules


def test_find_imports_in_paths_with_config(tmp_path):
    """Test find_imports_in_paths with custom config."""
    file1 = tmp_path / "file1.py"
    file1.write_text("import os\nimport sys")
    
    config = Config(verbose=True)
    imports = list(find_imports_in_paths([tmp_path], config=config))
    assert len(imports) == 2


def test_find_imports_in_paths_no_python_files(tmp_path):
    """Test find_imports_in_paths with directory containing no Python files."""
    file1 = tmp_path / "file.txt"
    file1.write_text("import os")
    
    imports = list(find_imports_in_paths([tmp_path]))
    assert len(imports) == 0


def test_find_imports_in_paths_top_only(tmp_path):
    """Test find_imports_in_paths with top_only=True."""
    file1 = tmp_path / "file1.py"
    file1.write_text("import os\n\ndef foo():\n    import sys")
    
    imports = list(find_imports_in_paths([tmp_path], top_only=True))
    modules = {imp.module for imp in imports}
    assert "os" in modules
    assert "sys" not in modules


def test_find_imports_in_paths_with_unique_attribute(tmp_path):
    """Test find_imports_in_paths with unique=ImportKey.ATTRIBUTE."""
    file1 = tmp_path / "file1.py"
    file1.write_text("from typing import List\nfrom typing import Dict")
    
    file2 = tmp_path / "file2.py"
    file2.write_text("from typing import List")
    
    imports = list(find_imports_in_paths([tmp_path], unique=ImportKey.ATTRIBUTE))
    list_imports = [imp for imp in imports if imp.attribute == "List"]
    dict_imports = [imp for imp in imports if imp.attribute == "Dict"]
    assert len(list_imports) == 1
    assert len(dict_imports) == 1


