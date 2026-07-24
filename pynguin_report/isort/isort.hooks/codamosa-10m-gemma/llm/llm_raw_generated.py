####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_get_lines():
    # Test Case 1: Basic functionality with multiple lines and whitespace
    mock_output = "line1\n  line2  \nline3\t"
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = mock_output.encode()
        result = get_lines(["dummy", "command"])
        assert result == ["line1", "line2", "line3"]

    # Test Case 2: Single line output
    mock_output = "single_line"
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = mock_output.encode()
        result = get_lines(["dummy", "command"])
        assert result == ["single_line"]

    # Test Case 3: Empty output
    mock_output = ""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = mock_output.encode()
        result = get_lines(["dummy", "command"])
        assert result == []

    # Test Case 4: Verification that subprocess.run is called with correct arguments
    command = ["git", "diff-index", "--cached"]
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = b"file1.py\nfile2.py"
        get_lines(command)
        mock_run.assert_called_once_with(command, stdout=subprocess.PIPE, check=True)
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock

@pytest.mark.parametrize("strict,modify,lazy,directories,files,is_sorted,expected_exit_code", [
    # Case 1: No files modified -> return 0
    (True, False, False, None, [], True, 0),
    
    # Case 2: Files modified, sorted, strict=True -> return 0
    (True, False, False, None, ["file1.py"], True, 0),
    
    # Case 3: Files modified, NOT sorted, strict=False -> return 0 (warning mode)
    (False, False, False, None, ["file1.py"], False, 0),
    
    # Case 4: Files modified, NOT sorted, strict=True -> return error count
    (True, False, False, None, ["file1.py", "file2.py"], False, 2),
    
    # Case 5: Files modified, NOT sorted, strict=True, modify=True -> return error count (with sort call)
    (True, True, False, None, ["file1.py"], False, 1),
    
    # Case 6: Lazy mode (check unstaged) -> check if git command changes
    (True, False, True, None, ["file1.py"], False, 1),
    
    # Case 7: Filter by directories -> check if git command extends
    (True, False, False, ["src/"], ["src/file1.py"], False, 1),
    
    # Case 8: Non-python files -> should be ignored
    (True, False, False, None, ["README.md", "script.sh"], False, 0),
])
def test_git_hook(strict, modify, lazy, directories, files, is_sorted, expected_exit_code):
    # Mocking subprocess/get_lines to return our controlled file list
    # We mock get_lines to return our 'files' list
    # We mock get_output to return dummy content for git show
    # We mock api.check_code_string to return our 'is_sorted' boolean
    
    with patch("isort.api.check_code_string") as mock_check, \
         patch("isort.api.sort_file") as mock_sort, \
         patch("isort.Config") as mock_config, \
         patch("isort.exceptions.FileSkipped", exception=Exception), \
         patch("your_module_name.get_lines") as mock_get_lines, \
         patch("your_module_name.get_output") as mock_get_output:
        
        # Setup mocks
        mock_get_lines.return_value = files
        mock_get_output.return_value = "import os\nimport sys" # dummy content
        mock_check.return_value = is_sorted
        
        # Execute
        result = git_hook(
            strict=strict,
            modify=modify,
            lazy=lazy,
            settings_file="config.ini",
            directories=directories
        )
        
        # Assertions
        assert result == expected_exit_code
        
        # Verify Git Command Construction
        if lazy:
            # Check if --cached was removed from the call
            args, _ = mock_get_lines.call_args
            assert "--cached" not in args[0]
        else:
            args, _ = mock_get_lines.call_args
            assert "--cached" in args[0]
            
        if directories:
            args, _ = mock_get_lines.call_args
            for d in directories:
                assert d in args[0]

        # Verify modification logic
        if modify and not is_sorted and any(f.endswith(".py") for f in files):
            assert mock_sort.called
        elif not modify and not is_sorted and any(f.endswith(".py") for f in files):
            assert not mock_sort.called

        # Verify Config instantiation
        if files:
            mock_config.assert_called()
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock

@pytest.mark.parametrize(
    "strict, modify, lazy, directories, files_output, is_sorted, expected_exit_code",
    [
        # Case 1: No files modified -> exit 0
        (True, False, False, None, [], True, 0),
        
        # Case 2: Files modified, all sorted, strict mode -> exit 0
        (True, False, False, None, ["file1.py", "file2.py"], True, 0),
        
        # Case 3: Files modified, one unsorted, strict mode -> exit 1
        (True, False, False, None, ["file1.py", "file2.py"], False, 1),
        
        # Case 4: Files modified, one unsorted, non-strict mode -> exit 0
        (False, False, False, None, ["file1.py"], False, 0),
        
        # Case 5: Files modified, one unsorted, strict mode, modify=True -> exit 1 (but calls sort)
        (True, True, False, None, ["file1.py"], False, 1),
        
        # Case 6: Lazy mode (checks unstaged files) -> check command construction
        (True, False, True, None, ["file1.py"], True, 0),
        
        # Case 7: Specific directories provided -> check command construction
        (True, False, False, ["src/"], ["src/file1.py"], True, 0),
        
        # Case 8: Non-python files should be ignored
        (True, False, False, None, ["README.md", "script.sh"], False, 0),
    ],
)
def test_git_hook(
    strict,
    modify,
    lazy,
    directories,
    files_output,
    is_sorted,
    expected_exit_code,
):
    # Mock subprocess.run to simulate git commands
    with patch("subprocess.run") as mock_run:
        # Mock get_lines output (git diff-index)
        # We need to handle multiple calls: 1 for diff, then 1 for git show per py file
        mock_diff_result = MagicMock()
        mock_diff_result.stdout = "\n".join(files_output).encode()
        
        # Mock git show output (file contents)
        mock_show_result = MagicMock()
        mock_show_result.stdout = b"import os\nimport sys"
        
        # Side effect logic for subprocess.run
        def side_effect(command, **kwargs):
            if "diff-index" in command:
                return mock_diff_result
            if "show" in command:
                return mock_show_result
            return MagicMock(stdout=b"")

        mock_run.side_effect = side_effect

        # Mock isort API
        with patch("isort.api.check_code_string") as mock_check, \
             patch("isort.api.sort_file") as mock_sort, \
             patch("isort.Config") as mock_config:
            
            mock_check.return_value = is_sorted
            
            # Execute the hook
            result = git_hook(
                strict=strict,
                modify=modify,
                lazy=lazy,
                settings_file="pyproject.toml",
                directories=directories,
            )

            # Assertions
            assert result == expected_exit_code
            
            # Verify command construction for diff
            diff_call_args = [arg for arg in mock_run.call_args_list if "diff-index" in mock_run.call_args_list[0][0]]
            if not files_output:
                # If no files, git_hook returns 0 immediately
                pass
            else:
                # Check if lazy removed --cached
                if lazy:
                    assert "--cached" not in mock_run.call_args_list[0][0][0]
                else:
                    assert "--cached" in mock_run.call_args_list[0][0][0]

            # Verify modification call
            if modify and not is_sorted and any(f.endswith(".py") for f in files_output):
                assert mock_sort.called
            elif not modify and not is_sorted and any(f.endswith(".py") for f in files_output):
                assert not mock_sort.called
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock

@pytest.mark.parametrize("strict, modify, lazy, directories, staged_files, is_sorted, expected_exit_code", [
    # Case 1: No files modified -> exit 0
    (False, False, False, None, [], True, 0),
    (True, False, False, None, [], True, 0),
    
    # Case 2: Python files modified, all sorted -> exit 0
    (False, False, False, None, ["file1.py", "file2.py"], True, 0),
    (True, False, False, None, ["file1.py", "file2.py"], True, 0),
    
    # Case 3: Python files modified, some unsorted, strict=False -> exit 0 (warning mode)
    (False, False, False, None, ["file1.py"], False, 0),
    (False, True, False, None, ["file1.py"], False, 0),
    
    # Case 4: Python files modified, some unsorted, strict=True -> exit > 0 (error mode)
    (True, False, False, None, ["file1.py"], False, 1),
    (True, False, False, None, ["file1.py", "file2.py"], False, 2),
    
    # Case 5: Non-python files modified -> ignored -> exit 0
    (True, False, False, None, ["README.md", "script.sh"], True, 0),
    
    # Case 6: Lazy mode (check unstaged) -> diff_cmd should not have --cached
    (True, False, True, None, ["file1.py"], False, 1),
])
def test_git_hook(strict, modify, lazy, directories, staged_files, is_sorted, expected_exit_code):
    with patch("subprocess.run") as mock_run, \
         patch("isort.api.check_code_string") as mock_check, \
         patch("isort.api.sort_file") as mock_sort, \
         patch("os.path.abspath") as mock_abspath, \
         patch("os.path.dirname") as mock_dirname:
        
        # Mock subprocess.run for git diff-index
        # We need to return the list of files in stdout
        diff_output = "\n".join(staged_files) + "\n"
        mock_run.side_effect = [
            MagicMock(stdout=diff_output.encode()), # git diff-index
            *[MagicMock(stdout=b"content") for _ in staged_files if _ in staged_files and _ .endswith(".py")] # git show
        ]
        
        # Mock isort behavior
        mock_check.return_value = is_sorted
        mock_abspath.return_value = "/path/to/file.py"
        mock_dirname.return_value = "/path/to"
        
        # Setup the call to git_hook
        result = git_hook(
            strict=strict,
            modify=modify,
            lazy=lazy,
            settings_file="config.ini",
            directories=directories
        )
        
        assert result == expected_exit_code
        
        # Verify git command construction for lazy mode
        if lazy:
            # Check if --cached was removed from the first call's args
            first_call_args = mock_run.call_args_list[0][0][0]
            assert "--cached" not in first_call_args
        else:
            first_call_args = mock_run.call_args_list[0][0][0]
            assert "--cached" in first_call_args

        # Verify sort_file was called if modify=True and error found
        if modify and not is_sorted and any(f.endswith(".py") for f in staged_files):
            assert mock_sort.called
        elif modify and is_sorted:
            assert not mock_sort.called

def test_git_hook_file_skipped_exception():
    with patch("subprocess.run") as mock_run, \
         patch("isort.api.check_code_string") as mock_check, \
         patch("isort.exceptions.FileSkipped", Exception):
        
        mock_run.side_effect = [
            MagicMock(stdout=b"file1.py\n"), # git diff-index
            MagicMock(stdout=b"content")     # git show
        ]
        mock_check.side_effect = exceptions.FileSkipped
        
        # Should not crash and should return 0 because error was skipped
        result = git_hook(strict=True)
        assert result == 0
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock

@pytest.mark.parametrize("strict,modify,lazy,directories,files_output,is_sorted,expected_exit_code,expected_calls", [
    # Case 1: No files modified
    (True, False, False, None, [], 0, []),
    
    # Case 2: Python files are sorted, strict mode
    (True, False, False, None, ["file1.py", "file2.py"], True, 0, [
        ["git", "diff-index", "--cached", "--name-only", "--diff-filter=ACMRTUXB", "HEAD"],
        ["git", "show", ":file1.py"],
        ["git", "show", ":file2.py"],
    ]),
    
    # Case 3: Python files are NOT sorted, strict mode, modify True
    (True, True, False, None, ["file1.py"], False, 1, [
        ["git", "diff-index", "--cached", "--name-only", "--diff-filter=ACMRTUXB", "HEAD"],
        ["git", "show", ":file1.py"],
    ]),
    
    # Case 4: Python files are NOT sorted, strict mode, modify False
    (True, False, False, None, ["file1.py"], False, 1, [
        ["git", "diff-index", "--cached", "--name-only", "--diff-filter=ACMRTUXB", "HEAD"],
        ["git", "show", ":file1.py"],
    ]),
    
    # Case 5: Non-python files should be ignored
    (True, False, False, None, ["README.md", "script.sh"], False, 0, [
        ["git", "diff-index", "--cached", "--name-only", "--diff-filter=ACMRTUXB", "HEAD"],
    ]),
    
    # Case 6: Lazy mode (removes --cached) and directories provided
    (True, False, True, ["src/"], ["src/file1.py"], True, 0, [
        ["git", "diff-index", "--name-only", "--diff-filter=ACMRTUXB", "HEAD", "src/"],
        ["git", "show", ":src/file1.py"],
    ]),
    
    # Case 7: Non-strict mode (always returns 0 even if errors found)
    (False, False, False, None, ["file1.py"], False, 0, [
        ["git", "diff-index", "--cached", "--name-only", "--diff-filter=ACMRTUXB", "HEAD"],
        ["git", "show", ":file1.py"],
    ]),
])
def test_git_hook(strict, modify, lazy, directories, files_output, is_sorted, expected_exit_code, expected_calls):
    with patch("subprocess.run") as mock_run, \
         patch("isort.api.check_code_string") as mock_check, \
         patch("isort.api.sort_file") as mock_sort, \
         patch("isort.Config") as mock_config:
        
        # Setup subprocess mock for git diff and git show
        mock_stdout_diff = "\n".join(files_output).encode()
        mock_stdout_show = b"import os\nimport sys"
        
        # Side effect to handle sequential calls to subprocess.run
        # First call is git diff, subsequent calls are git show
        run_side_effects = []
        # Diff call
        diff_res = MagicMock()
        diff_res.stdout = mock_stdout_diff
        run_side_effects.append(diff_res)
        # Show calls
        for _ in range(len([f for f in files_output if f.endswith(".py")])):
            show_res = MagicMock()
            show_res.stdout = mock_stdout_show
            run_side_effects.append(show_res)
        
        mock_run.side_effect = run_side_effects
        mock_check.return_value = is_sorted
        
        # Execute
        result = git_hook(
            strict=strict,
            modify=modify,
            lazy=lazy,
            settings_file="config.py",
            directories=directories
        )
        
        # Assertions
        assert result == expected_exit_code
        
        # Verify git commands
        actual_run_commands = [call.args[0] for call in mock_run.call_args_list]
        for expected_cmd in expected_calls:
            assert expected_cmd in actual_run_commands
            
        # Verify isort modification
        if modify and not is_sorted and files_output:
            assert mock_sort.called
        else:
            assert not mock_sort.called
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock

@pytest.mark.parametrize(
    "strict, modify, lazy, directories, staged_files, isort_check_result, expected_exit_code",
    [
        # Case 1: No files modified -> return 0
        (True, False, False, None, [], True, 0),
        
        # Case 2: Files modified, but no .py files -> return 0
        (True, False, False, None, ["README.md", "script.sh"], True, 0),
        
        # Case 3: .py files modified, all correct, strict mode -> return 0
        (True, False, False, None, ["app/main.py"], True, 0),
        
        # Case 4: .py files modified, error found, strict mode -> return 1
        (True, False, False, None, ["app/main.py"], False, 1),
        
        # Case 5: .py files modified, error found, non-strict mode -> return 0
        (False, False, False, None, ["app/main.py"], False, 0),
        
        # Case 6: .py files modified, error found, strict mode, modify=True -> return 1 (but calls sort_file)
        (True, True, False, None, ["app/main.py"], False, 1),
        
        # Case 7: Lazy mode (checks unstaged) -> diff_cmd should not have --cached
        (True, False, True, None, ["app/main.py"], True, 0),
        
        # Case 8: Directories restriction -> diff_cmd should include directories
        (True, False, False, ["src/"], ["src/lib.py"], True, 0),
    ],
)
def test_git_hook(
    strict,
    modify,
    lazy,
    directories,
    staged_files,
    isort_check_result,
    expected_exit_code,
):
    # Mocking subprocess.run to control git output
    # We mock get_lines (which calls get_output) to return our staged_files
    # We mock get_output to return dummy content for git show
    # We mock isort api.check_code_string and api.sort_file
    
    with patch("subprocess.run") as mock_run, \
         patch("isort.api.check_code_string") as mock_check, \
         patch("isort.api.sort_file") as mock_sort, \
         patch("os.path.abspath", return_value="/fake/path/file.py"), \
         patch("os.path.dirname", return_value="/fake/path"):

        # Setup subprocess mock for git diff-index
        mock_diff_result = MagicMock()
        mock_diff_result.stdout = b"\n".join(
            [f.encode() for f in staged_files]
        ) + b"\n"
        
        # Setup subprocess mock for git show
        mock_show_result = MagicMock()
        mock_show_result.stdout = b"import os\nimport sys"
        
        # Side effect for subprocess.run
        def run_side_effect(command, **kwargs):
            if "diff-index" in command:
                return mock_diff_result
            if "show" in command:
                return mock_show_result
            return MagicMock(stdout=b"")

        mock_run.side_effect = run_side_effect
        
        # Setup isort mock
        mock_check.return_value = isort_check_result

        # Execute
        result = git_hook(
            strict=strict,
            modify=modify,
            lazy=lazy,
            settings_file="",
            directories=directories,
        )

        # Assertions
        assert result == expected_exit_code
        
        if len(staged_files) > 0 and ".py" in staged_files[0] and not isort_check_result and modify:
            assert mock_sort.called
        
        if lazy and staged_files:
            # Check if --cached was removed from the command
            # The first call to subprocess.run is the diff command
            diff_cmd_called = mock_run.call_args_list[0][0][0]
            if lazy:
                assert "--cached" not in diff_cmd_called
            else:
                assert "--cached" in diff_cmd_called
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock

@pytest.mark.parametrize("strict,modify,lazy,directories,files_to_return,is_sorted,expected_exit_code", [
    # Case 1: No files modified
    (True, False, False, None, [], False, 0),
    
    # Case 2: Python files are sorted, no errors
    (True, False, False, None, ["test.py", "app.py"], True, 0),
    
    # Case 3: Python files are NOT sorted, strict mode (returns error count)
    (True, False, False, None, ["test.py"], False, 1),
    
    # Case 4: Python files are NOT sorted, non-strict mode (returns 0)
    (False, False, False, None, ["test.py"], False, 0),
    
    # Case 5: Python files NOT sorted, modify=True (calls sort_file)
    (True, True, False, None, ["test.py"], False, 1),
    
    # Case 6: Non-python files (should be ignored)
    (True, False, False, None, ["README.md", "script.sh"], False, 0),
    
    # Case 7: Lazy mode (removes --cached from git command)
    (True, False, True, None, ["test.py"], False, 1),
    
    # Case 8: Directories argument added to git command
    (True, False, False, ["src/"], ["src/test.py"], False, 1),
])
def test_git_hook(strict, modify, lazy, directories, files_to_return, is_sorted, expected_exit_code):
    # Mock subprocess.run to simulate git commands
    # We need to mock get_output and get_lines indirectly via subprocess.run
    
    # Setup the mock for subprocess.run
    # 1. The diff command (get_lines)
    # 2. The git show command (get_output)
    
    mock_diff_output = "\n".join(files_to_return) + "\n"
    mock_staged_content = "import os\nimport sys"
    
    with patch("subprocess.run") as mock_run, \
         patch("isort.api.check_code_string") as mock_check, \
         patch("isort.api.sort_file") as mock_sort, \
         patch("os.path.abspath", return_value="/root/" + (files_to_return[0] if files_to_return else "dummy.py")), \
         patch("os.path.dirname", return_value="/root"):
        
        # Define side effects for subprocess.run
        # First call: git diff-index
        # Second call: git show
        mock_diff_result = MagicMock()
        mock_diff_result.stdout = mock_diff_output.encode()
        
        mock_show_result = MagicMock()
        mock_show_result.stdout = mock_staged_content.encode()
        
        mock_run.side_effect = [mock_diff_result, mock_show_result]
        
        # Define side effect for isort check
        mock_check.return_value = is_sorted
        
        # Execute the hook
        result = git_hook(
            strict=strict,
            modify=modify,
            lazy=lazy,
            settings_file="",
            directories=directories
        )
        
        # Assertions
        assert result == expected_exit_code
        
        # Verify git diff command construction
        diff_cmd_called = mock_run.call_args_list[0][0][0]
        if lazy:
            assert "--cached" not in diff_cmd_called
        if directories:
            for d in directories:
                assert d in diff_cmd_called
                
        # Verify sort_file was called if modify is True and errors exist
        if modify and files_to_return and files_to_return[0].endswith(".py") and not is_sorted:
            assert mock_sort.called
        elif modify:
            assert not mock_sort.called

def test_git_hook_file_skipped_exception():
    with patch("subprocess.run") as mock_run, \
         patch("isort.api.check_code_string") as mock_check, \
         patch("isort.exceptions.FileSkipped", new=Exception), \
         patch("isort.api.FileSkipped", new=Exception):
        
        # Mock git diff to return one file
        mock_diff_result = MagicMock()
        mock_diff_result.stdout = b"test.py\n"
        
        # Mock git show
        mock_show_result = MagicMock()
        mock_show_result.stdout = b"import os"
        
        mock_run.side_effect = [mock_diff_result, mock_show_result]
        
        # Simulate the exception
        from isort.exceptions import FileSkipped
        mock_check.side_effect = FileSkipped
        
        # Should not crash and return 0 errors for skipped files
        result = git_hook(strict=True)
        assert result == 0
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock

@pytest.mark.parametrize("strict, modify, lazy, directories, staged_files, is_sorted, expected_exit_code", [
    # No files modified -> return 0
    (True, False, False, None, [], True, 0),
    
    # Files modified, all sorted -> return 0
    (True, False, False, None, ["file1.py", "file2.py"], True, 0),
    
    # Files modified, not sorted, strict=False -> return 0 (warning mode)
    (False, False, False, None, ["file1.py"], False, 0),
    
    # Files modified, not sorted, strict=True -> return error count
    (True, False, False, None, ["file1.py", "file2.py"], False, 2),
    
    # Files modified, not sorted, strict=True, modify=True -> return error count, calls sort_file
    (True, True, False, None, ["file1.py"], False, 1),
    
    # Lazy mode: removes --cached from command
    (True, False, True, None, ["file1.py"], True, 0),
    
    # Directories: adds directories to command
    (True, False, False, ["src/"], ["src/file1.py"], True, 0),
    
    # Non-python files should be ignored
    (True, False, False, None, ["README.md", "script.sh"], False, 0),
])
def test_git_hook(
    strict, 
    modify, 
    lazy, 
    directories, 
    staged_files, 
    is_sorted, 
    expected_exit_code
):
    # Mock subprocess.run to simulate git commands
    with patch("subprocess.run") as mock_run, \
         patch("isort.api.check_code_string") as mock_check, \
         patch("isort.api.sort_file") as mock_sort, \
         patch("os.path.abspath", return_value="/tmp/file1.py"), \
         patch("os.path.dirname", return_value="/tmp"):

        # Setup mock for get_lines (git diff-index)
        # We simulate the output of the diff command as the list of files
        mock_diff_output = "\n".join(staged_files) + "\n"
        
        # Setup mock for get_output (git show)
        # We simulate the content of the staged file
        mock_staged_content = "import os\nimport sys"
        
        # Configure the side effect for subprocess.run
        # 1st call: git diff-index
        # 2nd call: git show (for each .py file)
        mock_process_diff = MagicMock()
        mock_process_diff.stdout = mock_diff_output.encode()
        
        mock_process_show = MagicMock()
        mock_process_show.stdout = mock_staged_content.encode()
        
        mock_run.side_effect = [mock_process_diff, mock_process_show] * len(staged_files)

        # Setup isort behavior
        # If is_sorted is True, check_code_string returns True. If False, returns False.
        mock_check.return_value = is_sorted

        # Execute the hook
        result = git_hook(
            strict=strict,
            modify=modify,
            lazy=lazy,
            directories=directories
        )

        # Assertions
        assert result == expected_exit_code
        
        if staged_files and ".py" in staged_files[0]:
            assert mock_check.called
            if modify and not is_sorted:
                assert mock_sort.called
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_get_lines():
    # Test case 1: Successful execution with multiple lines
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = b"  line1  \nline2\n  line3  \n"
        command = ["git", "status", "--short"]
        result = get_lines(command)
        
        assert result == ["line1", "line2", "line3"]
        mock_run.assert_called_once_with(command, stdout=subprocess.PIPE, check=True)

    # Test case 2: Empty output
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = b""
        command = ["git", "ls-files"]
        result = get_lines(command)
        
        assert result == []

    # Test case 3: Single line with whitespace
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = b"  single_line  \n"
        command = ["echo", "test"]
        result = get_lines(command)
        
        assert result == ["single_lag"] # Note: input is stripped
        assert result == ["single_line"]

    # Test case 4: Command raises CalledProcessError
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(1, ["git", "error"])
        command = ["git", "error"]
        
        with pytest.raises(subprocess.CalledProcessError):
            get_lines(command)
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock

def test_git_hook():
    # Test Case 1: No files modified
    with patch("git_hook.get_lines", return_value=[]):
        assert git_hook() == 0

    # Test Case 2: Files modified, strict=False (Warning mode)
    with patch("git_hook.get_lines", return_value=["file1.py", "file2.txt"]), \
         patch("git_hook.get_output", return_value="import os\n"), \
         patch("isort.api.check_code_string", return_value=False):
        # Should return 0 because strict=False
        assert git_hook(strict=False) == 0

    # Test Case 3: Files modified, strict=True (Error mode)
    with patch("git_hook.get_lines", return_value=["file1.py"]), \
         patch("git_hook.get_output", return_value="import os\n"), \
         patch("isort.api.check_code_string", return_value=False):
        # Should return 1 error
        assert git_hook(strict=True) == 1

    # Test Case 4: Files modified, strict=True, modify=True (Auto-fix mode)
    with patch("git_hook.get_lines", return_value=["file1.py"]), \
         patch("git_hook.get_output", return_value="import os\n"), \
         patch("isort.api.check_code_string", return_value=False), \
         patch("isort.api.sort_file") as mock_sort:
        assert git_hook(strict=True, modify=True) == 1
        mock_sort.assert_called_once()

    # Test Case 5: Lazy mode (checking unstaged files)
    with patch("git_hook.get_lines") as mock_get_lines, \
         patch("git_hook.get_output", return_value="import os\n"), \
         patch("isort.api.check_code_string", return_value=True):
        mock_get_lines.return_value = ["file1.py"]
        git_hook(lazy=True)
        # Verify --cached was removed from the command
        args, _ = mock_get_lines.call_args
        assert "--cached" not in args[0]

    # Test Case 6: Directory restriction
    with patch("git_hook.get_lines") as mock_get_lines, \
         patch("git_hook.get_output", return_value=""), \
         patch("isort.api.check_code_string", return_value=True):
        mock_get_lines.return_value = []
        git_hook(directories=["src/"])
        args, _ = mock_get_lines.call_args
        assert "src/" in args[0]

    # Test Case 7: FileSkipped exception handling
    with patch("git_hook.get_lines", return_value=["file1.py"]), \
         patch("git_hook.get_output", return_value="import os\n"), \
         patch("isort.api.check_code_string", side_effect=exceptions.FileSkipped):
        # Should not crash and return 0 errors
        assert git_hook(strict=True) == 0

    # Test Case 8: Non-python files are ignored
    with patch("git_hook.get_lines", return_value=["README.md"]), \
         patch("git_hook.get_output", return_value=""):
        # Should return 0 because it only iterates over .py files
        assert git_hook(strict=True) == 0
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_get_lines():
    # Test case 1: Standard output with multiple lines
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = b"line1\n  line2  \nline3\n"
        result = get_lines(["dummy", "command"])
        assert result == ["line1", "line2", "line3"]

    # Test case 2: Empty output
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = b""
        result = get_lines(["dummy", "command"])
        assert result == []

    # Test case 3: Output with only whitespace
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = b"  \n\t\n  line  \n"
        result = get_lines(["dummy", "command"])
        assert result == ["", "", "line"]

    # Test case 4: Verify subprocess.run is called with correct arguments
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = b"test"
        command = ["ls", "-l"]
        get_lines(command)
        mock_run.assert_called_once_with(
            command, stdout=subprocess.PIPE, check=True
        )
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_get_lines():
    # Test case 1: Successful command output with multiple lines
    mock_output = "line1\n  line2  \nline3\n"
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = mock_output.encode()
        result = get_lines(["dummy_cmd"])
        assert result == ["line1", "line2", "line3"]

    # Test case 2: Empty output
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = b""
        result = get_lines(["dummy_cmd"])
        assert result == []

    # Test case 3: Command raises CalledProcessError
    with patch("subprocess.run") as mock_run:
        from subprocess import CalledProcessError
        mock_run.side_effect = CalledProcessError(1, ["dummy_cmd"])
        with pytest.raises(CalledProcessError):
            get_lines(["dummy_cmd"])

    # Test case 4: Single line output
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = b"single_line"
        result = get_lines(["dummy_cmd"])
        assert result == ["single_line"]
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_get_lines():
    # Test Case 1: Standard output with multiple lines
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = b"line1\n  line2  \nline3\n"
        result = get_lines(["some", "command"])
        assert result == ["line1", "line2", "line3"]
        mock_run.assert_called_once_with(["some", "command"], stdout=subprocess.PIPE, check=True)

    # Test Case 2: Empty output
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = b""
        result = get_lines(["some", "command"])
        assert result == []

    # Test Case 3: Command fails (subprocess.CalledProcessError)
    with patch("subprocess.run") as mock_run:
        import subprocess
        mock_run.side_effect = subprocess.CalledProcessError(1, ["cmd"])
        with pytest.raises(subprocess.CalledProcessError):
            get_lines(["cmd"])

    # Test Case 4: Output with only whitespace/newlines
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = b"\n\n  \n"
        result = get_lines(["some", "command"])
        # splitlines() on "\n\n  \n" results in empty strings after stripping
        # Depending on splitlines behavior, we expect list of stripped empty strings
        assert all(len(line) == 0 for line in result)
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_get_lines():
    # Test case 1: Basic functionality with multiple lines and whitespace
    mock_output = "line1\n  line2  \nline3\t"
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = mock_output.encode()
        result = get_lines(["dummy", "command"])
        assert result == ["line1", "line2", "line3"]

    # Test case 2: Empty output
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = b""
        result = get_lines(["dummy", "command"])
        assert result == []

    # Test case 3: Single line
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = b"only_one_line"
        result = get_lines(["dummy", "command"])
        assert result == ["only_one_line"]

    # Test case 4: Error propagation
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(1, ["dummy"])
        with pytest.raises(subprocess.CalledProcessError):
            get_lines(["dummy"])
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock

@pytest.mark.parametrize(
    "strict, modify, lazy, directories, files, staged_contents, check_result, expected_errors, expected_call_count",
    [
        # 1. No files modified: should return 0 immediately
        (True, False, False, None, [], "", False, 0, 0),
        
        # 2. Files modified, but none are .py: should return 0
        (True, False, False, None, ["test.txt", "README.md"], "", False, 0, 0),
        
        # 3. Python file, check passes: should return 0
        (True, False, False, None, ["app.py"], "import os", True, 0, 1),
        
        # 4. Python file, check fails, strict=True: should return error count
        (True, False, False, None, ["app.py"], "import os\nimport sys", False, 1, 1),
        
        # 5. Python file, check fails, strict=False: should return 0 (warning mode)
        (False, False, False, None, ["app.py"], "import os\nimport sys", False, 0, 1),
        
        # 6. Python file, check fails, modify=True: should call sort_file
        (True, True, False, None, ["app.py"], "import os\nimport sys", False, 1, 1),
        
        # 7. Lazy mode: should change git command (remove --cached)
        (True, False, True, None, ["app.py"], "import os", True, 0, 1),
        
        # 8. Directories filter: should add directories to git command
        (True, False, False, ["src/"], ["src/app.py"], "import os", True, 0, 1),
        
        # 9. Multiple files: count errors correctly
        (True, False, False, None, ["a.py", "b.py"], "import os", False, 2, 2),
    ],
)
def test_git_hook(
    strict,
    modify,
    lazy,
    directories,
    files,
    staged_contents,
    check_result,
    expected_errors,
    expected_call_count,
):
    # Mocking subprocess.run for get_output and get_lines
    # We need to handle different calls: 
    # 1. git diff-index (to get file list)
    # 2. git show (to get file content)
    
    with patch("subprocess.run") as mock_run, \
         patch("isort.api.check_code_string") as mock_check, \
         patch("isort.api.sort_file") as mock_sort, \
         patch("isort.Config") as mock_config:
        
        # Setup mock for git diff-index
        diff_output = "\n".join(files) + "\n"
        # Setup mock for git show (return staged content for each file)
        show_output = staged_contents.encode()
        
        # Configure subprocess.run side effects
        # First call is get_lines(diff_cmd), subsequent calls are get_output(staged_cmd)
        mock_run.side_effect = [
            MagicMock(stdout=diff_output.encode()),
            MagicMock(stdout=show_output),
        ] * (len(files) if files else 1)

        # Setup isort mocks
        mock_check.return_value = check_result
        
        # Execute
        result = git_hook(
            strict=strict,
            modify=modify,
            lazy=lazy,
            settings_file="config.py",
            directories=directories,
        )

        # Assertions
        assert result == expected_errors
        assert mock_check.call_count == expected_call_count
        
        if modify and len(files) > 0 and not check_result and files[0].endswith(".py"):
            assert mock_sort.called
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock

@pytest.mark.parametrize("strict,modify,lazy,directories,files,staged_contents,check_result,expect_errors,expect_exit_code", [
    # Case 1: No files modified
    (True, False, False, None, [], "", True, 0),
    
    # Case 2: Python files, no errors, strict mode
    (True, False, False, None, ["file1.py", "file2.py"], "import os\nimport sys", True, 0, 0),
    
    # Case 3: Python files, errors found, strict mode (return error count)
    (True, False, False, None, ["file1.py"], "import sys\nimport os", False, True, 1),
    
    # Case 4: Python files, errors found, non-strict mode (return 0)
    (False, False, False, None, ["file1.py"], "import sys\nimport os", False, True, 0),
    
    # Case 5: Python files, errors found, modify mode (calls sort_file)
    (True, True, False, None, ["file1.py"], "import sys\nimport os", False, True, 1),
    
    # Case 6: Lazy mode (removes --cached)
    (True, False, True, None, ["file1.py"], "import os", True, 0, 0),
    
    # Case 7: Non-python files (should be ignored)
    (True, False, False, None, ["README.md", "script.sh"], "", True, 0, 0),
    
    # Case 8: Directories provided (extends git command)
    (True, False, False, ["src/"], ["src/file1.py"], "import os", True, 0, 0),
])
def test_git_hook(strict, modify, lazy, directories, files, staged_contents, check_result, expect_errors, expect_exit_code):
    diff_cmd_expected = ["git", "diff-index", "--cached", "--name-only", "--diff-filter=ACMRTUXB", "HEAD"]
    if lazy:
        diff_cmd_expected.remove("--cached")
    if directories:
        diff_cmd_expected.extend(directories)

    # Mocking subprocess.run for get_lines and get_output
    # First call for get_lines (diff_cmd)
    # Subsequent calls for get_output (git show)
    mock_diff_output = "\n".join(files) + "\n"
    mock_show_output = staged_contents

    with patch("subprocess.run") as mock_run, \
         patch("isort.api.check_code_string") as mock_check, \
         patch("isort.api.sort_file") as mock_sort, \
         patch("isort.Config") as mock_config_class:
        
        # Setup subprocess mock
        # First call is get_lines (diff_cmd)
        # Second call is get_output (git show :filename)
        mock_run.side_effect = [
            MagicMock(stdout=mock_diff_output.encode()),
            MagicMock(stdout=mock_show_output.encode())
        ]
        
        mock_check.return_value = check_result
        
        # Execute
        exit_code = git_hook(
            strict=strict,
            modify=modify,
            lazy=lazy,
            directories=directories
        )
        
        # Assertions
        assert exit_code == expect_exit_code
        
        # Verify git command construction
        actual_diff_cmd = mock_run.call_args_list[0][0][0]
        assert actual_diff_cmd == diff_cmd_expected
        
        # Verify isort logic
        if files and files[0].endswith(".py"):
            assert mock_check.called
            if modify and not check_result:
                assert mock_sort.called
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock

@pytest.mark.parametrize(
    "strict, modify, lazy, directories, staged_files, isort_check_pass, expected_exit_code",
    [
        # Case 1: No files modified
        (True, False, False, None, [], True, 0),
        
        # Case 2: Files modified, all pass isort (strict mode)
        (True, False, False, None, ["file1.py", "file2.py"], True, 0),
        
        # Case 3: Files modified, one fails isort (strict mode)
        (True, False, False, None, ["file1.py", "file2.py"], False, 1),
        
        # Case 4: Files modified, one fails isort (non-strict mode)
        (False, False, False, None, ["file1.py"], False, 0),
        
        # Case 5: Files modified, one fails isort, modify=True (checks if sort_file called)
        (True, True, False, None, ["file1.py"], False, 1),
        
        # Case 6: Non-python files should be ignored
        (True, False, False, None, ["script.sh", "data.txt"], False, 0),
        
        # Case 7: Lazy mode (removes --cached from command)
        (True, False, True, None, ["file1.py"], True, 0),
        
        # Case 8: Specific directories provided
        (True, False, False, ["src/"], ["src/file1.py"], True, 0),
    ],
)
def test_git_hook(
    strict,
    modify,
    lazy,
    directories,
    staged_files,
    isort_check_pass,
    expected_exit_code,
):
    # Mock subprocess.run to simulate git commands
    with patch("subprocess.run") as mock_run, \
         patch("isort.api.check_code_string") as mock_check, \
         patch("isort.api.sort_file") as mock_sort, \
         patch("isort.Config") as mock_config_class:
        
        # Setup mock for get_lines (git diff-index)
        # We return the staged_files list as the output of the git diff command
        mock_diff_output = "\n".join(staged_files) + "\n"
        
        # Setup mock for get_output (git show)
        mock_git_show_output = b"import os\nimport sys\n"
        
        # Configure the side effects for subprocess.run
        # First call: git diff-index
        # Second call: git show for each python file
        run_side_effects = []
        
        # Create a mock object for the first call (diff-index)
        diff_result = MagicMock()
        diff_result.stdout = mock_diff_output.encode()
        run_side_effects.append(diff_result)
        
        # Create mock objects for subsequent git show calls
        for _ in staged_files:
            show_result = MagicMock()
            show_result.stdout = mock_git_show_output
            run_side_effects.append(show_result)
            
        mock_run.side_effect = run_side_effects
        
        # Mock isort behavior
        mock_check.return_value = isort_check_pass
        
        # Execute the function
        result = git_hook(
            strict=strict,
            modify=modify,
            lazy=lazy,
            settings_file="config.py",
            directories=directories,
        )
        
        # Assertions
        assert result == expected_exit_code
        
        if modify and not isort_check_pass and any(f.endswith(".py") for f in staged_files):
            assert mock_sort.called
        
        if lazy:
            # Verify that --cached was removed from the first call's arguments
            first_cmd = mock_run.call_args_list[0][0][0]
            assert "--cached" not in first_cmd
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock

@pytest.mark.parametrize("strict,modify,lazy,directories,files,is_sorted,expected_exit_code", [
    # Case 1: No files modified
    (True, False, False, None, [], True, 0),
    
    # Case 2: Python files are sorted, no errors
    (True, False, False, None, ["test.py", "app/main.py"], True, 0),
    
    # Case 3: Python files are NOT sorted, strict mode (returns error count)
    (True, False, False, None, ["test.py"], False, 1),
    
    # Case 4: Python files are NOT sorted, non-strict mode (returns 0)
    (False, False, False, None, ["test.py"], False, 0),
    
    # Case 5: Python files are NOT sorted, modify is True (calls sort_file)
    (True, True, False, None, ["test.py"], False, 1),
    
    # Case 6: Non-python files are ignored (no errors)
    (True, False, False, None, ["README.md", "script.sh"], False, 0),
    
    # Case 7: Lazy mode (diff command changes)
    (True, False, True, None, ["test.py"], False, 1),
    
    # Case 8: Directories filter (diff command changes)
    (True, False, False, ["src/"], ["src/test.py"], False, 1),
])
def test_git_hook(strict, modify, lazy, directories, files, is_sorted, expected_exit_code):
    with patch("subprocess.run") as mock_run, \
         patch("isort.api.check_code_string") as mock_check, \
         patch("isort.api.sort_file") as mock_sort, \
         patch("os.path.abspath", return_value="/repo/test.py"), \
         patch("os.path.dirname", return_value="/repo"):
        
        # Setup Mock for git diff-index
        # We simulate the output of 'git diff-index' as the list of files
        mock_diff_output = "\n".join(files) + "\n"
        
        # Setup Mock for git show (contents of the file)
        mock_git_show_output = "import os\nimport sys"
        
        def side_effect_run(command, **kwargs):
            mock_result = MagicMock()
            if "--cached" in command:
                mock_result.stdout = mock_diff_output.encode()
            elif "show" in command:
                mock_result.stdout = mock_git_show_output.encode()
            else:
                mock_result.stdout = b""
            return mock_result

        mock_run.side_effect = side_effect_run
        
        # Setup Mock for isort check
        mock_check.return_value = is_sorted

        # Execute
        result = git_hook(
            strict=strict,
            modify=modify,
            lazy=lazy,
            settings_file="",
            directories=directories
        )

        # Assertions
        assert result == expected_exit_code
        
        # Verify sort_file was called if modify is True and error exists
        if modify and not is_sorted and files:
            assert mock_sort.called
        elif not modify and not is_sorted and files:
            assert not mock_sort.called

        # Verify git commands called
        if files:
            # Check if diff command was called with correct flags
            diff_call_args = [arg for arg in mock_run.call_args_list if "--diff-filter" in arg[0][0]]
            assert len(diff_call_args) > 0
            
            if lazy:
                assert "--cached" not in diff_call_args[0][0]
            else:
                assert "--cached" in diff_call_args[0][0]
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock

@pytest.mark.parametrize(
    "strict, modify, lazy, directories, staged_files, isort_check_pass, expected_exit_code",
    [
        # Case 1: No files modified -> exit 0
        (True, False, False, None, [], True, 0),
        # Case 2: Files modified, isort passes -> exit 0
        (True, False, False, None, ["file1.py", "file2.py"], True, 0),
        # Case 3: Files modified, isort fails, strict=True -> exit count of errors
        (True, False, False, None, ["file1.py", "file2.py"], False, 2),
        # Case 4: Files modified, isort fails, strict=False -> exit 0 (warning mode)
        (False, False, False, None, ["file1.py"], False, 0),
        # Case 5: Files modified, isort fails, modify=True -> calls sort_file
        (True, True, False, None, ["file1.py"], False, 1),
        # Case 6: Lazy mode (checks unstaged files) -> diff command changes
        (True, False, True, None, ["file1.py"], True, 0),
        # Case 7: Non-python files should be ignored
        (True, False, False, None, ["script.sh", "README.md"], False, 0),
        # Case 8: Specific directories provided
        (True, False, False, ["src/"], ["src/module.py"], False, 1),
    ],
)
def test_git_hook(
    strict,
    modify,
    lazy,
    directories,
    staged_files,
    isort_check_pass,
    expected_exit_code,
):
    # Mock subprocess.run for get_output/get_lines
    # We need to handle both get_lines (diff_cmd) and get_output (git show)
    with patch("subprocess.run") as mock_run, \
         patch("isort.api.check_code_string") as mock_check, \
         patch("isort.api.sort_file") as mock_sort, \
         patch("isort.Config") as mock_config:

        # Setup mock for get_lines (git diff-index)
        # Return the staged_files as lines of stdout
        diff_output = "\n".join(staged_files).encode("utf-8")
        
        # Setup mock for get_output (git show)
        staged_content = b"import os\nimport sys"
        
        # Define side effects for subprocess.run
        # First call is git diff, subsequent calls are git show
        mock_run.side_effect = [
            MagicMock(stdout=diff_output),  # git diff-index
            MagicMock(stdout=staged_content), # git show file1.py
            MagicMock(stdout=staged_content), # git show file2.py (if exists)
        ]

        # Setup isort mocks
        mock_check.return_value = isort_check_pass
        
        # Execute the hook
        result = git_hook(
            strict=strict,
            modify=modify,
            lazy=lazy,
            settings_file="config.ini",
            directories=directories,
        )

        # Assertions
        assert result == expected_exit_code
        
        # Verify git command construction for diff
        first_call_args = mock_run.call_args_list[0][0][0]
        if lazy:
            assert "--cached" not in first_call_args
        else:
            assert "--cached" in first_call_args
            
        if directories:
            for d in directories:
                assert d in first_call_args

        # Verify isort logic
        if modify and not isort_check_pass and staged_files:
            # Check if sort_file was called for the failing python files
            assert mock_sort.called

        # Verify config pathing (based on the first file in staged_files)
        if staged_files and staged_files[0].endswith(".py"):
            assert mock_config.called
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock

@pytest.mark.parametrize("strict, modify, lazy, directories, files_output, isort_check_result, expected_errors", [
    # Case 1: No files modified, should return 0
    (True, False, False, None, [], False, 0),
    
    # Case 2: Files modified, no errors, strict mode
    (True, False, False, None, ["file1.py", "file2.py"], True, 0),
    
    # Case 3: Files modified, error found, strict mode, no modify
    (True, False, False, None, ["file1.py"], False, 1),
    
    # Case 4: Files modified, error found, strict mode, with modify
    (True, True, False, None, ["file1.py"], False, 1),
    
    # Case 5: Files modified, error found, non-strict mode (should return 0)
    (False, False, False, None, ["file1.py"], False, 0),
    
    # Case 6: Lazy mode (removes --cached from command)
    (True, False, True, None, ["file1.py"], True, 0),
    
    # Case 7: Non-python files should be ignored
    (True, False, False, None, ["file1.txt", "file2.py"], True, 0),
    
    # Case 8: Directories passed to git command
    (True, False, False, ["src/"], ["src/file1.py"], True, 0),
])
def test_git_hook(strict, modify, lazy, directories, files_output, isort_check_result, expected_errors):
    with patch("subprocess.run") as mock_run, \
         patch("isort.api.check_code_string") as mock_check, \
         patch("isort.api.sort_file") as mock_sort, \
         patch("isort.Config") as mock_config:
        
        # Setup mock for get_lines (git diff command)
        # We mock subprocess.run to return the list of files as stdout
        mock_stdout_diff = "\n".join(files_output) + "\n"
        mock_run.side_effect = [
            MagicMock(stdout=mock_stdout_diff.encode()), # git diff command
            MagicMock(stdout=b"content"),                # git show command for first file
        ] * len(files_output)
        
        # Setup mock for isort check
        mock_check.return_value = isort_check_result
        
        # Execute
        result = git_hook(
            strict=strict,
            modify=modify,
            lazy=lazy,
            settings_file="test_config.ini",
            directories=directories
        )
        
        # Assertions
        assert result == expected_errors
        
        if files_output:
            # Verify git diff command construction
            diff_call_args = mock_run.call_args_list[0][0][0]
            if lazy:
                assert "--cached" not in diff_call_args
            if directories:
                for d in directories:
                    assert d in diff_call_args

            # Verify isort interaction
            if isort_check_result is False:
                if modify:
                    assert mock_sort.called
                else:
                    assert not mock_sort.called
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_get_lines():
    # Test case 1: Basic functionality with multiple lines
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = b"line1\n  line2  \nline3\n"
        result = get_lines(["some", "command"])
        assert result == ["line1", "line2", "line3"]
        mock_run.assert_called_once_with(["some", "command"], stdout=subprocess.PIPE, check=True)

    # Test case 2: Empty output
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = b""
        result = get_lines(["some", "command"])
        assert result == []

    # Test case 3: Single line with whitespace
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = b"   single_line   \n"
        result = get_lines(["some", "command"])
        assert result == ["single_line"]

    # Test case 4: Command failure (subprocess.CalledProcessError)
    with patch("subprocess.run") as mock_run:
        import subprocess
        mock_run.side_effect = subprocess.CalledProcessError(1, ["cmd"])
        with pytest.raises(subprocess.CalledProcessError):
            get_lines(["cmd"])
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock

@pytest.mark.parametrize("strict,modify,lazy,directories,files,staged_contents,is_sorted,expected_errors", [
    # 1. No files modified -> return 0
    (True, False, False, None, [], "", True, 0),
    
    # 2. Files modified, but not .py files -> return 0
    (True, False, False, None, ["README.md", "script.sh"], "", True, 0),
    
    # 3. Python file modified, is sorted, strict mode -> return 0
    (True, False, False, None, ["app.py"], "import os\nimport sys", True, 0),
    
    # 4. Python file modified, NOT sorted, strict mode -> return 1
    (True, False, False, None, ["app.py"], "import sys\nimport os", False, 1),
    
    # 5. Python file modified, NOT sorted, strict mode, modify=True -> return 1 (but calls sort_file)
    (True, True, False, None, ["app.py"], "import sys\nimport os", False, 1),
    
    # 6. Python file modified, NOT sorted, NOT strict mode -> return 0
    (False, False, False, None, ["app.py"], "import sys\nimport os", False, 0),
    
    # 7. Lazy mode (no --cached) -> check logic for command construction
    (True, False, True, None, ["app.py"], "import os", True, 0),
])
def test_git_hook(strict, modify, lazy, directories, files, staged_contents, is_sorted, expected_errors):
    # Mocking subprocess.run for get_output and get_lines
    # We need to simulate 'git diff-index' returning our 'files' list
    # and 'git show' returning our 'staged_contents'
    
    with patch("subprocess.run") as mock_run, \
         patch("isort.api.check_code_string") as mock_check, \
         patch("isort.api.sort_file") as mock_sort, \
         patch("os.path.abspath", return_value="/root/app.py"), \
         patch("os.path.dirname", return_value="/root"):

        # Setup mock for git diff-index
        diff_output = "\n".join(files).encode() + b"\n"
        # Setup mock for git show
        show_output = staged_contents.encode()
        
        # Define side effect for subprocess.run
        def side_effect(command, stdout, check, **kwargs):
            mock_result = MagicMock()
            if "--cached" in command:
                mock_result.stdout = diff_output
            elif "show" in command:
                mock_result.stdout = show_output
            else:
                mock_result.stdout = b""
            return mock_result

        mock_run.side_effect = side_effect
        mock_check.return_value = is_sorted

        # Execute the hook
        result = git_hook(
            strict=strict,
            modify=modify,
            lazy=lazy,
            settings_file="",
            directories=directories
        )

        # Assertions
        assert result == expected_errors
        
        if files and not is_sorted and modify:
            mock_sort.assert_called()
        
        if not files:
            assert mock_check.call_count == 0

def test_git_hook_directories_filter():
    """Test that directories argument is appended to git command."""
    with patch("subprocess.run") as mock_run, \
         patch("isort.api.check_code_string", return_value=True):
        
        mock_result = MagicMock()
        mock_result.stdout = b"src/main.py\n"
        mock_run.return_value = mock_result
        
        git_hook(directories=["src", "tests"])
        
        # Check if directories were added to the command
        called_command = mock_run.call_args_list[0][0][0]
        assert "src" in called_command
        assert "tests" in called_command

def test_git_hook_file_skipped_exception():
    """Test that isort.exceptions.FileSkipped is handled gracefully."""
    from isort.exceptions import FileSkipped
    
    with patch("subprocess.run") as mock_run, \
         patch("isort.api.check_code_string", side_effect=FileSkipped), \
         patch("os.path.abspath", return_value="/root/app.py"), \
         patch("os.path.dirname", return_value="/root"):
        
        mock_result = MagicMock()
        mock_result.stdout = b"app.py\n"
        mock_run.return_value = mock_result
        
        # Should not raise exception and return 0 errors
        result = git_hook(strict=True)
        assert result == 0
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock

@pytest.mark.parametrize(
    "strict, modify, lazy, directories, files_modified, staged_contents, check_result, expected_return",
    [
        # Case 1: No files modified
        (True, False, False, None, [], "", True, 0),
        
        # Case 2: Files modified, all pass isort
        (True, False, False, None, ["file1.py", "file2.py"], "import os\nimport sys", True, 0),
        
        # Case 3: Files modified, one fails isort, strict=True (returns error count)
        (True, False, False, None, ["file1.py"], "import sys\nimport os", False, 1),
        
        # Case 4: Files modified, one fails isort, strict=False (returns 0)
        (False, False, False, None, ["file1.py"], "import sys\nimport os", False, 0),
        
        # Case 5: Files modified, one fails isort, modify=True (should call sort_file)
        (True, True, False, None, ["file1.py"], "import sys\nimport os", False, 1),
        
        # Case 6: Lazy mode (removes --cached from command)
        (True, False, True, None, ["file1.py"], "import os", True, 0),
        
        # Case 7: Non-python files should be ignored
        (True, False, False, None, ["README.md", "script.sh"], "", True, 0),
        
        # Case 8: Directories filter applied to git command
        (True, False, False, ["src/"], ["src/file1.py"], "import os", True, 0),
    ],
)
def test_git_hook(
    strict,
    modify,
    lazy,
    directories,
    files_modified,
    staged_contents,
    check_result,
    expected_return,
):
    with patch("subprocess.run") as mock_run, \
         patch("isort.api.check_code_string") as mock_check, \
         patch("isort.api.sort_file") as mock_sort, \
         patch("isort.Config") as mock_config_class:
        
        # Setup mock for get_lines (git diff-index)
        # We need to return the files_modified list
        mock_diff_result = MagicMock()
        mock_diff_result.stdout = "\n".join(files_modified).encode()
        
        # Setup mock for get_output (git show)
        mock_show_result = MagicMock()
        mock_show_result.stdout = staged_contents.encode()
        
        # Side effect for subprocess.run
        # First call is git diff, subsequent calls are git show
        mock_run.side_effect = [mock_diff_result, mock_show_result]
        
        # Setup isort mocks
        mock_check.return_value = check_result
        
        # Execute
        result = git_hook(
            strict=strict,
            modify=modify,
            lazy=lazy,
            settings_file="config.ini",
            directories=directories,
        )
        
        # Assertions
        assert result == expected_return
        
        # Verify git diff command construction
        diff_call_args = mock_run.call_args_list[0][0][0]
        if lazy:
            assert "--cached" not in diff_call_args
        if directories:
            for d in directories:
                assert d in diff_call_args
        
        # Verify sort_file was called if modify is True and error occurred
        if modify and check_result is False and files_modified and files_modified[0].endswith(".py"):
            assert mock_sort.called
        elif modify and check_result is True:
            assert not mock_sort.called

def test_git_hook_file_skipped_exception():
    with patch("subprocess.run") as mock_run, \
         patch("isort.api.check_code_string") as mock_check, \
         patch("isort.exceptions.FileSkipped", new=Exception), \
         patch("isort.Config"):
        
        mock_diff_result = MagicMock()
        mock_diff_result.stdout = b"file1.py"
        
        mock_show_result = MagicMock()
        mock_show_result.stdout = b"import os"
        
        mock_run.side_effect = [mock_diff_result, mock_show_result]
        mock_check.side_effect = Exception("FileSkipped") # Simulating the error
        
        # Should not crash and should return 0 errors if we treat it as skipped
        # Note: In the original code, FileSkipped is caught and passed.
        # Since we can't easily redefine the exception class without side effects, 
        # we just ensure the logic handles the try/except block.
        
        # We manually trigger the logic for the test
        try:
            result = git_hook(strict=True)
            assert result == 0
        except Exception:
            pytest.fail("git_hook raised exception unexpectedly")
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_get_lines():
    # Test case 1: Standard output with multiple lines and whitespace
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = b"line1\n  line2  \nline3\n"
        result = get_lines(["dummy", "command"])
        assert result == ["line1", "line2", "line3"]

    # Test case 2: Empty output
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = b""
        result = get_lines(["dummy", "command"])
        assert result == []

    # Test case 3: Output with only newlines
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = b"\n\n"
        result = get_lines(["dummy", "command"])
        # splitlines() on "\n\n" results in ['', ''] or similar depending on trailing
        # but strip() on empty strings remains empty strings
        # Testing the behavior of the specific implementation:
        # stdout.splitlines() for "\n\n" is ['', '']
        assert result == ["", ""]

    # Test case 4: Command failure (subprocess.run with check=True)
    import subprocess
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(1, ["dummy"])
        with pytest.raises(subprocess.CalledProcessError):
            get_lines(["dummy", "command"])
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock

@pytest.mark.parametrize(
    "strict, modify, lazy, directories, staged_files, isort_check_result, expected_exit_code",
    [
        # Case 1: No files modified -> return 0
        (True, False, False, None, [], True, 0),
        
        # Case 2: Python files staged, all sorted -> return 0 (strict or not)
        (True, False, False, None, ["file1.py", "file2.py"], True, 0),
        (False, False, False, None, ["file1.py"], True, 0),
        
        # Case 3: Python files staged, one error, strict=True -> return 1
        (True, False, False, None, ["file1.py", "file2.py"], False, 1),
        
        # Case 4: Python files staged, one error, strict=False -> return 0 (warning mode)
        (False, False, False, None, ["file1.py"], False, 0),
        
        # Case 5: Non-python files staged -> should be ignored
        (True, False, False, None, ["script.sh", "README.md"], False, 0),
        
        # Case 6: Modify=True, error found -> calls sort_file
        (True, True, False, None, ["file1.py"], False, 1),
        
        # Case 7: Lazy=True -> uses different git diff command (no --cached)
        (True, False, True, None, ["file1.py"], False, 1),
        
        # Case 8: Directories provided -> extends git diff command
        (True, False, False, ["src/"], ["src/file1.py"], False, 1),
    ],
)
def test_git_hook(
    strict,
    modify,
    lazy,
    directories,
    staged_files,
    isort_check_result,
    expected_exit_code,
):
    # Mocking subprocess.run to control git command outputs
    with patch("subprocess.run") as mock_run, \
         patch("isort.api.check_code_string") as mock_check, \
         patch("isort.api.sort_file") as mock_sort, \
         patch("isort.Config") as mock_config:
        
        # Setup Mock for git diff-index (get_lines)
        # We need to return the list of files based on the test case
        diff_output = "\n".join(staged_files) + "\n"
        
        # Setup Mock for git show (get_output)
        # Return dummy content for each file
        staged_content = "import os\nimport sys"
        
        # Configure the mock_run behavior
        # First call: git diff-index
        # Subsequent calls: git show
        mock_run.side_effect = [
            MagicMock(stdout=diff_output.encode()),  # git diff-index
            MagicMock(stdout=staged_content.encode()), # git show file1
            MagicMock(stdout=staged_content.encode()), # git show file2 (if exists)
        ]
        
        # Configure isort behavior
        mock_check.return_value = isort_check_result
        
        # Execute the hook
        result = git_hook(
            strict=strict,
            modify=modify,
            lazy=lazy,
            settings_file="config.ini",
            directories=directories,
        )
        
        # Assertions
        assert result == expected_exit_code
        
        # Verify git command construction for lazy mode
        if lazy:
            # Check if '--cached' was removed from the first call args
            first_call_args = mock_run.call_args_list[0][0][0]
            assert "--cached" not in first_call_args
        
        # Verify if sort_file was called when modify=True and error exists
        if modify and not isort_check_result and any(f.endswith(".py") for f in staged_files):
            assert mock_sort.called
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock

@pytest.mark.parametrize("strict,modify,lazy,directories,files_to_return,is_sorted,expected_exit_code", [
    # Case 1: No files modified -> return 0
    (False, False, False, None, [], False, 0),
    (True, False, False, None, [], False, 0),
    
    # Case 2: Python files modified, all sorted -> return 0
    (False, False, False, None, ["test.py", "app.py"], True, 0),
    (True, False, False, None, ["test.py", "app.py"], True, 0),
    
    # Case 3: Python files modified, one unsorted, strict=False -> return 0 (warning mode)
    (False, False, False, None, ["test.py", "app.py"], False, 0),
    
    # Case 4: Python files modified, one unsorted, strict=True -> return 1 (error mode)
    (True, False, False, None, ["test.py", "app.py"], False, 1),
    
    # Case 5: Python files modified, one unsorted, strict=True, modify=True -> return 1, trigger sort
    (True, True, False, None, ["test.py"], False, 1),
    
    # Case 6: Non-python files modified -> return 0
    (True, False, False, None, ["README.md", "script.sh"], False, 0),
    
    # Case 7: Lazy mode (no --cached)
    (True, False, True, None, ["test.py"], False, 1),
])
def test_git_hook(strict, modify, lazy, directories, files_to_return, is_sorted, expected_exit_code):
    # Mock subprocess.run for get_lines (git diff-index)
    # Mock subprocess.run for get_output (git show)
    # Mock isort api.check_code_string
    # Mock isort api.sort_file
    
    with patch("subprocess.run") as mock_run, \
         patch("isort.api.check_code_string") as mock_check, \
         patch("isort.api.sort_file") as mock_sort, \
         patch("os.path.abspath") as mock_abs, \
         patch("os.path.dirname") as mock_dir:
        
        # Setup mock for git diff-index
        diff_output = "\n".join(files_to_return).encode()
        if not files_to_return:
            diff_output = b""
            
        # Setup mock for git show (content of files)
        staged_content = b"import os\nimport sys"
        
        # Configure the side effects for subprocess.run
        # First call is git diff-index, subsequent calls are git show
        mock_run.side_effect = [
            MagicMock(stdout=diff_output.decode().encode()), # git diff-index
            MagicMock(stdout=staged_content)                 # git show for first file
        ] * len(files_to_return)
        
        # Ensure we handle the case where no files are returned
        if not files_to_return:
            mock_run.side_effect = [MagicMock(stdout=b"")]
        else:
            # If multiple files, we need more mock returns for git show
            # We'll provide enough to cover all files in files_to_return
            mock_run.side_effect = [
                MagicMock(stdout=diff_output.decode().encode()), # diff-index
            ] + [MagicMock(stdout=staged_content) for _ in files_to_return]

        # Mock isort behavior
        mock_check.return_value = is_sorted
        mock_abs.return_value = "/tmp/test_dir"
        mock_dir.return_value = "/tmp"

        # Execute the hook
        result = git_hook(
            strict=strict,
            modify=modify,
            lazy=lazy,
            settings_file="",
            directories=directories
        )

        # Assertions
        assert result == expected_exit_code
        
        if is_sorted is False and modify is True and files_to_return and files_to_return[0].endswith(".py"):
            assert mock_sort.called
        
        if not files_to_return:
            assert mock_check.call_count == 0
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_get_lines():
    # Test case 1: Standard output with multiple lines and whitespace
    mock_output = "line1\n  line2  \nline3\n"
    expected_output = ["line1", "line_2", "line3"] # Note: strip() removes leading/trailing whitespace
    # Correcting expected based on strip() behavior:
    expected_output = ["line1", "line2", "line3"]

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = mock_output.encode()
        
        result = get_lines(["git", "status"])
        
        assert result == ["line1", "line2", "line3"]
        mock_run.assert_called_once_with(["git", "status"], stdout=subprocess.PIPE, check=True)

    # Test case 2: Empty output
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = b""
        
        result = get_lines(["git", "diff"])
        
        assert result == []

    # Test case 3: Output with only whitespace/newlines
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = b"\n \n\t\n"
        
        result = get_lines(["git", "ls-files"])
        
        # splitlines() on "\n \n\t\n" results in [' ', ' ', '\t'] or similar depending on split logic
        # but strip() on those strings results in empty strings
        # Filter out empty strings if that's the logic, but get_lines doesn't filter, it only strips.
        # Let's verify behavior: [line.strip() for line in "\n \n\t\n".splitlines()]
        # "\n \n\t\n".splitlines() -> ['', ' ', '\t']
        # result -> ['', '', '']
        assert all(len(line) == 0 for line in result)

    # Test case 4: subprocess error (check=True)
    with patch("subprocess.run") as mock_run:
        import subprocess
        mock_run.side_effect = subprocess.CalledProcessError(1, "cmd")
        
        with pytest.raises(subprocess.CalledProcessError):
            get_lines(["false"])
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock

@pytest.mark.parametrize(
    "strict, modify, lazy, directories, files_modified, staged_contents, is_sorted, expected_exit_code, expected_api_calls",
    [
        # Case 1: No files modified
        (True, False, False, None, [], "", True, 0, 0),
        
        # Case 2: Files modified, all sorted, strict mode (return 0)
        (True, False, False, None, ["file1.py"], "import os", True, 0, 1),
        
        # Case 3: Files modified, one error, strict mode (return 1)
        (True, False, False, None, ["file1.py"], "import sys\nimport os", False, 1, 1),
        
        # Case 4: Files modified, one error, strict mode, modify=True (return 1, calls sort_file)
        (True, True, False, None, ["file1.py"], "import sys\nimport os", False, 1, 2),
        
        # Case 5: Files modified, one error, non-strict mode (return 0)
        (False, False, False, None, ["file1.py"], "import sys\nimport os", False, 0, 1),
        
        # Case 6: Lazy mode (removes --cached from git command)
        (True, False, True, None, ["file1.py"], "import os", True, 0, 1),
        
        # Case 7: Non-python files (should be ignored)
        (True, False, False, None, ["README.md"], "", True, 0, 0),
        
        # Case 8: Directories filtering
        (True, False, False, ["src/"], ["src/file1.py"], "import os", True, 0, 1),
    ],
)
def test_git_hook(
    strict,
    modify,
    lazy,
    directories,
    files_modified,
    staged_contents,
    is_sorted,
    expected_exit_code,
    expected_api_calls,
):
    with patch("subprocess.run") as mock_run, \
         patch("isort.api.check_code_string") as mock_check, \
         patch("isort.api.sort_file") as mock_sort, \
         patch("isort.Config") as mock_config:
        
        # Setup subprocess mock for git diff-index and git show
        mock_diff_output = "\n".join(files_modified) + "\n"
        mock_show_output = staged_contents.encode()
        
        def side_effect_run(command, **kwargs):
            mock_result = MagicMock()
            if "diff-index" in command:
                mock_result.stdout = mock_diff_output.encode()
            elif "show" in command:
                mock_result.stdout = mock_show_output
            return mock_result

        mock_run.side_effect = side_effect_run
        mock_check.return_value = is_sorted

        # Execute
        result = git_hook(
            strict=strict,
            modify=modify,
            lazy=lazy,
            settings_file="config.ini",
            directories=directories,
        )

        # Assertions
        assert result == expected_exit_code
        
        # Verify API calls
        # Check if check_code_string was called the expected number of times
        # (only for .py files in files_modified)
        py_files_count = len([f for f in files_modified if f.endswith(".py")])
        assert mock_check.call_count == py_files_count
        
        if modify and py_files_count > 0 and not is_sorted:
            assert mock_sort.called
        else:
            assert not mock_sort.called

        # Verify git command construction for lazy mode
        if lazy:
            # Check that --cached was removed from the diff command
            diff_call = [arg for arg, kwargs in mock_run.call_args_list if "--cached" not in arg[0] and "diff-index" in arg[0]]
            if diff_call:
                assert "--cached" not in diff_call[0]
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock

@pytest.mark.parametrize("strict,modify,lazy,directories,staged_files,isort_check_pass,expected_exit_code", [
    # Case 1: No files modified -> returns 0
    (True, False, False, None, [], True, 0),
    
    # Case 2: Files modified, but all pass isort -> returns 0
    (True, False, False, None, ["file1.py", "file2.py"], True, 0),
    
    # Case 3: Strict mode, one file fails -> returns 1
    (True, False, False, None, ["file1.py"], False, 1),
    
    # Case 4: Non-strict mode, one file fails -> returns 0 (warning only)
    (False, False, False, None, ["file1.py"], False, 0),
    
    # Case 5: Modify mode, one file fails -> calls sort_file
    (True, True, False, None, ["file1.py"], False, 1),
    
    # Case 6: Lazy mode -> uses different git command (removes --cached)
    (True, False, True, None, ["file1.py"], True, 0),
    
    # Case 7: Non-python files are ignored
    (True, False, False, None, ["README.md", "script.sh"], True, 0),
])
def test_git_hook(strict, modify, lazy, directories, staged_files, isort_check_pass, expected_exit_code):
    # Mock subprocess.run to simulate git commands
    # We need to handle:
    # 1. git diff-index (get_lines)
    # 2. git show (get_output)
    
    with patch("subprocess.run") as mock_run, \
         patch("isort.api.check_code_string") as mock_check, \
         patch("isort.api.sort_file") as mock_sort, \
         patch("os.path.abspath") as mock_abs:
        
        # Setup mock for git diff-index
        # First call is the diff command
        diff_output = "\n".join(staged_files) + "\n"
        
        # Setup mock for git show
        # Subsequent calls are for each file in staged_files
        show_output = "import os\nimport sys\n"
        
        # Configure mock_run side effects
        # 1. The diff command
        # 2. One 'git show' call for every file in staged_files
        mock_run_results = []
        
        # Result for git diff-index
        diff_res = MagicMock()
        diff_res.stdout = diff_output.encode()
        mock_run_results.append(diff_res)
        
        # Results for git show
        for _ in staged_files:
            show_res = MagicMock()
            show_res.stdout = show_output.encode()
            mock_run_results.append(show_res)
            
        mock_run.side_effect = mock_run_results
        
        # Setup isort mocks
        mock_check.return_value = isort_check_pass
        mock_abs.return_value = "/repo/path/file1.py"
        
        # Execute
        result = git_hook(
            strict=strict,
            modify=modify,
            lazy=lazy,
            directories=directories
        )
        
        # Assertions
        assert result == expected_exit_code
        
        # Verify git command construction for lazy mode
        if lazy:
            # Check if '--cached' was removed from the first call
            first_cmd = mock_run.call_args_list[0][0][0]
            assert "--cached" not in first_cmd
        
        # Verify sort_file was called if modify=True and error occurred
        if modify and not isort_check_pass and staged_files:
            assert mock_sort.called
        elif not modify and not isort_check_pass and staged_files:
            assert not mock_sort.called
```


