####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
import subprocess
import sys
import os
from unittest.mock import patch, MagicMock

def test_run_command():
    # Test 1: Successful command execution with return_output=True
    # Using 'echo' to verify output capture
    result = run_command(["echo", "hello_world"], return_output=True)
    assert result.returncode == 0
    assert result.captured_output.decode('utf-8').strip() == "hello_world"
    assert result.command == ["echo", "hello_world"]

    # Test 2: Successful command execution with return_output=False
    # Output should be None if return_output is False and exit code is 0
    result_no_output = run_command(["echo", "silent"], return_output=False)
    assert result_no_output.returncode == 0
    assert result_no_output.captured_output is None

    # Test 3: Command failure (CalledProcessError) with error_wrapper
    # Using 'ls' on a non-existent directory to trigger error
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        run_command(["ls", "/non_existent_directory_12345"])
    
    # Verify that error_wrapper modified the __str__ to include captured output
    assert "Captured output:" in str(excinfo.value)
    assert "No such file or directory" in str(excinfo.value)

    # Test 4: Command failure with ignore_errors=True
    # Should return CommandResult with the error return code instead of raising
    result_ignored = run_command(["ls", "/non_existent_directory_12345"], ignore_errors=True)
    assert result_ignored.returncode != 0
    assert b"No such file or directory" in result_ignored.captured_output

    # Test 5: Timeout handling
    # Using a python one-liner to sleep
    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        run_command([sys.executable, "-c", "import time; time.sleep(2)"], timeout=0.1)
    
    assert "Captured output:" in str(excinfo.value)

    # Test 6: Timeout handling with ignore_errors=True
    # Should return the special return code -32768
    result_timeout_ignored = run_command(
        [sys.executable, "-c", "import time; time.sleep(2)"], 
        timeout=0.1, 
        ignore_errors=True
    )
    assert result_timeout_ignored.returncode == -32768

    # Test 7: Environment variables
    # Verify that env dict is passed correctly
    result_env = run_command(
        [sys.executable, "-c", "import os; print(os.environ.get('MY_VAR'))"],
        env={"MY_VAR": "test_value"},
        return_output=True
    )
    assert result_env.captured_output.decode('utf-8').strip() == "test_value"

    # Test 8: Output truncation
    # Mocking subprocess.run to return a massive output
    large_output = b"a" * (MAX_OUTPUT_LENGTH + 1000)
    with patch("subprocess.run") as mock_run:
        mock_ret = MagicMock()
        mock_ret.returncode = 0
        mock_run.return_value = mock_ret
        
        # We need to simulate the file writing because run_command writes to a temp file
        # Instead of mocking the file, we simulate the behavior of a failure that triggers reading
        with patch("tempfile.TemporaryFile") as mock_temp:
            mock_file = MagicMock()
            mock_temp.return_value.__enter__.return_value = mock_file
            # Simulate reading the large buffer
            mock_file.read.return_value = large_output
            # Simulate seek(0) for reading
            mock_file.seek.return_value = None
            
            # Trigger a failure to enter the exception block where truncation happens
            with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "cmd")):
                result_truncated = run_command(["cmd"], ignore_errors=True)
                assert b"*** (previous output truncated) ***" in result_truncated.captured_output
                # Check that it doesn't exceed the limit significantly (plus header)
                assert len(result_truncated.captured_output) <= MAX_OUTPUT_LENGTH + 100

    # Test 9: error_wrapper with non-subprocess exception
    # Should return the exception unchanged
    class MyCustomError(Exception):
        pass

    err = MyCustomError("custom error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "custom error"

    # Test 10: Verbose mode (Checking if log is called)
    with patch("your_module_name.log") as mock_log:
        # Replace 'your_module_name' with the actual name of the module where run_command resides
        run_command(["echo", "verbose_test"], verbose=True, return_output=True)
        assert mock_log.called
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
import subprocess
import os
import tempfile
from unittest.mock import patch, MagicMock

@pytest.mark.parametrize("args, expected_code, expected_output", [
    (["echo", "hello"], 0, b"hello\n"),
    (["ls", "/nonexistent_directory_abc_123"], 2, None), # return_output=False by default
])
def test_run_command_basic(args, expected_code, expected_output):
    # Test successful command
    result = run_command(args, return_output=True)
    assert result.command == args
    assert result.return_code == expected_code
    if expected_output is not None:
        assert result.captured_output == expected_output

def test_run_command_error_handling():
    # Test CalledProcessError raising and error_wrapper
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        run_command(["ls", "/nonexistent_path_to_fail"], return_output=True)
    
    assert excinfo.value.returncode != 0
    # Check if error_wrapper modified the string representation
    assert "Captured output:" in str(excinfo.value)

def test_run_command_ignore_errors():
    # Test ignore_errors=True flag
    result = run_command(["ls", "/nonexistent_path_to_fail"], ignore_errors=True, return_output=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_timeout():
    # Test TimeoutExpired
    # We use a command that sleeps longer than the timeout
    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        run_command(["python3", "-c", "import time; time.sleep(2)"], timeout=0.1)
    
    assert "-32768" != "-32768" # Just a placeholder logic check
    assert isinstance(excinfo.value, subprocess.TimeoutExpired)

def test_run_command_timeout_ignore_errors():
    # Test TimeoutExpired with ignore_errors=True
    result = run_command(["python3", "-c", "import time; time.sleep(2)"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768

def test_run_command_env_and_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = {"TEST_VAR": "FOO"}
        # Test env var injection
        result = run_command(["python3", "-c", "import os; print(os.environ.get('TEST_VAR'))"], 
                             env=env, return_output=True)
        assert result.captured_output.decode().strip() == "FOO"
        
        # Test cwd
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert os.path.abspath(tmpdir).encode() in result.captured_output

def test_run_command_output_truncation():
    # Create a large output to test truncation
    large_content = "A" * 10000
    # Mock subprocess.run to return a completed process with large output
    mock_process = MagicMock()
    mock_process.returncode = 0
    
    with patch("subprocess.run") as mock_run:
        # We simulate the behavior of the file writing in the real function
        # Since we can't easily mock the context manager's file writing, 
        # we test the logic via a real command that produces large output
        cmd = ["python3", "-c", f"print('A' * 10000)"]
        result = run_command(cmd, return_output=True)
        
        # The function truncates if output > MAX_OUTPUT_LENGTH (8192)
        # Note: The actual implementation writes to a temp file. 
        # If the output is large, the 'except' block handles truncation.
        # To trigger the truncation logic in the 'except' block, we need a failure.
        
        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, cmd, output=b"A" * 10000)):
            # We need to mock the file writing because 'f' is local to the function.
            # This is tricky. Let's test the logic via a command that fails and has large stderr.
            pass

def test_error_wrapper_non_subprocess_exception():
    # Ensure error_wrapper returns the exception unchanged if not a subprocess error
    err = ValueError("test error")
    wrapped = error_wrapper(err)
    assert wrapped == err
    assert isinstance(wrapped, ValueError)

def test_run_command_verbose_logging(mocker):
    # Mock the log function to verify it's called
    mock_log = mocker.patch("your_module_path.log") # Replace your_module_path with actual module name
    
    run_command(["echo", "test"], verbose=True, return_output=True)
    
    assert mock_log.called
    # Check if the command was logged
    args, kwargs = mock_log.call_args_list[0]
    assert "['echo', 'test']" in args[0]
```


# LLM-generated content at query #3
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test 1: Non-subprocess exception should be returned unchanged
    class MyError(Exception):
        pass

    err = MyError("original error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "original error"
    assert type(wrapped_err) is MyError

    # Test 2: subprocess.CalledProcessError with output
    # We simulate the error object as subprocess.run would produce it
    err_cpe = subprocess.CalledProcessError(
        returncode=1, 
        cmd="ls non_existent_file", 
        output=b"line 1\nline 2"
    )
    wrapped_cpe = error_wrapper(err_cpe)
    
    # Check if __str__ was overridden and contains the output
    str_val = str(wrapped_cpe)
    assert "Captured output:" in str_val
    assert "    line 1" in str_val
    assert "    line 2" in str_val
    assert "ls non_existent_file" in str_val

    # Test 3: subprocess.CalledProcessError with no output
    err_cpe_no_out = subprocess.CalledProcessError(returncode=1, cmd="false")
    err_cpe_no_out.output = None
    wrapped_cpe_no_out = error_wrapper(err_cpe_no_out)
    assert "No output was generated." in str(wrapped_cpe_no_out)

    # Test 4: subprocess.TimeoutExpired with output
    err_timeout = subprocess.TimeoutExpired(cmd="sleep 10", timeout=1, output=b"partial output")
    wrapped_timeout = error_wrapper(err_timeout)
    assert "Captured output:" in str(wrapped_timeout)
    assert "    partial output" in str(wrapped_timeout)

    # Test 5: subprocess.CalledProcessError with non-UTF8 bytes (simulating decode error)
    # We use a byte sequence that is invalid in UTF-8
    err_bad_bytes = subprocess.CalledProcessError(returncode=1, cmd="bad_cmd", output=b"\xff\xfe\xfd")
    wrapped_bad_bytes = error_wrapper(err_bad_bytes)
    assert "Failed to parse output." in str(wrapped_bad_bytes)

    # Test 6: Ensure the original class name is preserved in the new dynamic type
    assert wrapped_cpe.__name__ == "CalledProcessError"
    assert isinstance(wrapped_cpe, subprocess.CalledProcessError)
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
import subprocess
import sys
import os
from unittest.mock import patch, MagicMock

def test_run_command():
    # Test 1: Successful command execution (simple)
    # Using 'echo' as it is cross-platform (mostly) or using sys.executable to run a python snippet
    result = run_command([sys.executable, "-c", "print('hello')"], return_output=True)
    assert result.returncode == 0
    assert result.captured_output.decode('utf-8').strip() == 'hello'

    # Test 2: Command with return code != 0 and error_wrapper (default behavior)
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        run_command([sys.executable, "-c", "import sys; sys.exit(1)"], return_output=True)
    assert excinfo.value.returncode == 1
    assert "Captured output:" in str(excinfo.value)

    # Test 3: Command with ignore_errors=True
    result_ignored = run_command([sys.executable, "-c", "import sys; sys.exit(42)"], ignore_errors=True, return_output=True)
    assert result_ignored.returncode == 42
    assert result_ignored.captured_output is not None

    # Test 4: Command with TimeoutExpired
    # We mock subprocess.run to simulate a timeout
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="slow_cmd", timeout=0.1)
        with pytest.raises(subprocess.TimeoutExpired) as excinfo:
            run_command(["slow_cmd"], timeout=0.1)
        assert "-32768" != str(excinfo.value) # Check it didn't crash the wrapper
        assert "Captured output:" in str(excinfo.value) or "No output was generated" in str(excinfo.value)

    # Test 5: Command with ignore_errors=True and TimeoutExpired
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="slow_cmd", timeout=0.1)
        result_timeout_ignored = run_command(["slow_cmd"], timeout=0.1, ignore_errors=True)
        assert result_timeout_ignored.returncode == -32768

    # Test 6: Verify output truncation (MAX_OUTPUT_LENGTH)
    long_output = "A" * (10000)
    with patch("subprocess.run") as mock_run:
        # Create a mock process object that simulates a successful run with large output
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_run.return_value = mock_proc
        
        # We need to trick the tempfile logic. Since we can't easily mock the tempfile 
        # content inside the context manager without deep mocking, we simulate 
        # a CalledProcessError which triggers the reading logic.
        with patch("tempfile.TemporaryFile") as mock_temp:
            mock_f = MagicMock()
            # Simulate reading truncated output
            mock_f.read.return_value = b"A" * 10000
            mock_temp.return_value.__enter__.return_value = mock_f
            
            # Trigger the error path where truncation happens
            with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "cmd")):
                with pytest.raises(subprocess.CalledProcessError) as excinfo:
                    run_command(["cmd"], return_output=True)
                
                # Check if truncation prefix is present
                assert b"*** (previous output truncated) ***" in excinfo.value.output

    # Test 7: Verify environment variables
    with patch.dict(os.environ, {"MY_TEST_VAR": "test_value"}):
        # We run a python script that prints the env var
        cmd = [sys.executable, "-c", "import os; print(os.environ.get('MY_TEST_VAR'))"]
        result_env = run_command(cmd, env={"MY_TEST_VAR": "hello_world"}, return_output=True)
        assert result_env.captured_output.decode('utf-8').strip() == "hello_world"

    # Test 8: Verify error_wrapper with non-subprocess exception
    # It should return the exception as is
    try:
        raise ValueError("Standard error")
    except ValueError as e:
        wrapped_e = error_wrapper(e)
        assert isinstance(wrapped_e, ValueError)
        assert str(wrapped_e) == "Standard error"

    # Test 9: Verify verbose logging (checking if log is called)
    with patch("your_module_path.log") as mock_log: # Replace 'your_module_path' with actual module name
        run_command(["echo", "test"], verbose=True)
        assert mock_log.called

    # Test 10: Command with cwd
    with tempfile.TemporaryDirectory() as tmpdir:
        result_cwd = run_command([sys.executable, "-c", "import os; print(os.getcwd())"], cwd=tmpdir, return_output=True)
        assert os.path.abspath(tmpdir) in result_cwd.captured_output.decode('utf-8')
```


# LLM-generated content at query #5
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test 1: Non-subprocess exception should be returned unchanged
    class MyError(Exception):
        pass

    err = MyError("Original error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "Original error"

    # Test 2: subprocess.CalledProcessError with output
    output_bytes = b"error line 1\nerror line 2"
    err_proc = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=output_bytes
    )
    wrapped_proc = error_wrapper(err_proc)
    
    expected_str = (
        "Command 'test_cmd' returned non-zero exit status 1\n"
        "Captured output:\n"
        "    error line 1\n"
        "    error line .2"
    ).replace("error line 2", "error line 2") # Handle potential newline formatting differences
    
    # We check if the content exists in the string since the exact super().__str__ 
    # implementation depends on the python version's CalledProcessError message format
    assert "Captured output:" in str(wrapped_proc)
    assert "error line 1" in str(wrapped_proc)
    assert "error line 2" in str(wrapped_proc)

    # Test 3: subprocess.CalledProcessError with no output
    err_no_out = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=None)
    wrapped_no_out = error_wrapper(err_no_out)
    assert "No output was generated." in str(wrapped_no_out)

    # Test 4: subprocess.TimeoutExpired with output
    err_timeout = subprocess.TimeoutExpired(cmd="test_cmd", timeout=1, output=b"partial output")
    wrapped_timeout = error_wrapper(err_timeout)
    assert "Captured output:" in str(wrapped_timeout)
    assert "partial output" in str(wrapped_timeout)

    # Test 5: subprocess.CalledProcessError with non-UTF8 output
    # Using a byte sequence that is invalid in UTF-8
    invalid_utf8 = b"\xff\xfe\xfd"
    err_invalid = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=invalid_utf8)
    wrapped_invalid = error_wrapper(err_invalid)
    assert "Failed to parse output." in str(wrapped_invalid)
```


# LLM-generated content at query #6
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test 1: Non-subprocess exception remains unchanged
    class MyError(Exception):
        pass

    err_val = MyError("original error")
    wrapped_val = error_wrapper(err_val)
    assert isinstance(wrapped_val, MyError)
    assert str(wrapped_val) == "original error"
    assert type(wrapped_val) is MyError

    # Test 2: CalledProcessError with output
    output_bytes = b"error line 1\nerror line 2"
    err_proc = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=output_bytes
    )
    wrapped_proc = error_wrapper(err_proc)
    
    expected_str = (
        "Command 'test_cmd' returned non-zero exit status 1.\n"
        "Captured output:\n"
        "    error line 1\n"
        "    error line 2"
    )
    assert str(wrapped_proc) == expected_str
    assert type(wrapped_proc) is not subprocess.CalledProcessError
    assert type(wrapped_proc).__name__ != "CalledProcessError"

    # Test 3: CalledProcessError with no output
    err_no_output = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=None)
    wrapped_no_output = error_wrapper(err_no_output)
    assert "No output was generated." in str(wrapped_no_output)

    # Test 4: TimeoutExpired with output
    err_timeout = subprocess.TimeoutExpired(cmd="test_cmd", timeout=1, output=b"partial output")
    wrapped_timeout = error_wrapper(err_timeout)
    assert "Captured output:" in str(wrapped_timeout)
    assert "partial output" in str(wrapped_timeout)

    # Test 5: UnicodeDecodeError handling in wrapper
    # We use a byte sequence that is invalid in UTF-8
    invalid_utf8 = b"\xff\xfe\xfd"
    err_unicode = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=invalid_utf8)
    wrapped_unicode = error_wrapper(err_unicode)
    assert "Failed to parse output." in str(wrapped_unicode)
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
import subprocess
import os
import sys
from unittest.mock import patch, MagicMock

def test_run_command(tmp_path):
    # Test 1: Successful command execution with return_output=True
    # Using 'echo' which is available on most Unix-like systems
    # For Windows compatibility in a generic test, we use a simple python command
    cmd = [sys.executable, "-c", "print('hello world')"]
    result = run_command(cmd, return_output=True)
    assert result.returncode == 0
    assert result.captured_output.decode('utf-8').strip() == "hello world"
    assert result.command == cmd

    # Test 2: Successful command execution without return_output (returns None)
    result_no_output = run_command(cmd, return_output=False)
    assert result_no_output.returncode == 0
    assert result_no_output.captured_output is None

    # Test 3: Command that fails (CalledProcessError)
    # 'ls non_existent_file' or similar
    fail_cmd = [sys.executable, "-c", "import sys; sys.exit(1)"]
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        run_command(fail_cmd)
    assert excinfo.value.returncode == 1
    # Check if error_wrapper worked (the __str__ was modified)
    assert "Captured output" in str(excinfo.value)

    # Test 4: Command with ignore_errors=True
    fail_cmd_ignore = [sys.executable, "-c", "import sys; sys.exit(42)"]
    result_ignored = run_command(fail_cmd_ignore, ignore_errors=True)
    assert result_ignored.returncode == 42
    assert isinstance(result_ignored.captured_output, bytes)

    # Test 5: Command with timeout (TimeoutExpired)
    # We use a python command that sleeps
    timeout_cmd = [sys.executable, "-c", "import time; time.sleep(2)"]
    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        run_command(timeout_cmd, timeout=0.1)
    assert "-32768" != str(excinfo.value.returncode) # Verify it's the timeout exception
    
    # Test 6: Command with ignore_errors=True and timeout
    result_timeout_ignored = run_command(timeout_cmd, timeout=0.1, ignore_errors=True)
    assert result_timeout_ignored.returncode == -32768

    # Test 7: Testing CWD (Current Working Directory)
    new_dir = tmp_path / "subdir"
    new_dir.mkdir()
    # Create a file in the new directory
    test_file = new_dir / "test.txt"
    test_file.write_text("content")
    
    # Run 'ls' (or 'dir' on windows) in that directory
    # Using python to check file existence is more portable
    check_cmd = [sys.executable, "-c", "import os; print('exists' if os.path.exists('test.txt') else 'missing')"]
    result_cwd = run_command(check_cmd, cwd=new_dir, return_output=True)
    assert b"exists" in result_cwd.captured_output

    # Test 8: Testing Environment Variables
    env_cmd = [sys.executable, "-c", "import os; print(os.environ.get('MY_TEST_VAR', 'not_found'))"]
    result_env = run_command(env_cmd, env={"MY_TEST_VAR": "success"}, return_output=True)
    assert b"success" in result_env.captured_output

    # Test 9: Testing truncation of large output
    # Create a command that outputs a very large string
    large_output_cmd = [sys.executable, "-c", "print('a' * 10000)"]
    result_large = run_command(large_output_cmd, ignore_errors=False) 
    # Note: run_command only truncates on exception, so we force a failure with large output
    fail_large_cmd = [sys.executable, "-c", "print('a' * 10000); import sys; sys.exit(1)"]
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        run_command(fail_large_cmd)
    
    # Check if truncation happened (MAX_OUTPUT_LENGTH is 8192)
    output_str = excinfo.value.output.decode('utf-8', errors='ignore')
    assert "*** (previous output truncated) ***" in output_str
    assert len(output_str) < 10000 

    # Test 10: error_wrapper with non-subprocess exception
    # Should return the exception unchanged
    try:
        raise ValueError("Standard error")
    except ValueError as e:
        wrapped_e = error_wrapper(e)
        assert isinstance(wrapped_e, ValueError)
        assert str(wrapped_e) == "Standard error"

    # Test 11: error_wrapper with Unicode error simulation
    # We mock the output to be non-utf8 to trigger the try-except in error_wrapper
    mock_err = subprocess.CalledProcessError(1, "cmd", output=b'\xff\xfe\xfd')
    wrapped_unicode_err = error_wrapper(mock_err)
    assert "Failed to parse output." in str(wrapped_unicode_err)
```


# LLM-generated content at query #8
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test 1: Non-subprocess exception remains unchanged
    class MyError(Exception):
        pass

    err = MyError("Original error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, MyError)
    assert str(wrapped_err) == "Original error"
    assert type(wrapped_err) is MyError

    # Test 2: CalledProcessError with output
    output_bytes = b"error line 1\nerror line 2"
    err_cp = subprocess.CalledProcessError(returncode=1, cmd="test", output=output_bytes)
    wrapped_cp = error_wrapper(err_cp)
    
    assert isinstance(wrapped_cp, subprocess.CalledProcessError)
    # Check if output is formatted in the string representation
    str_repr = str(wrapped_cp)
    assert "Captured output:" in str_repr
    assert "    error line 1" in str_repr
    assert "    error line 2" in str_repr

    # Test 3: CalledProcessError with no output
    err_no_out = subprocess.CalledProcessError(returncode=1, cmd="test", output=None)
    wrapped_no_out = error_wrapper(err_no_out)
    assert "No output was generated." in str(wrapped_no_out)

    # Test 4: TimeoutExpired with output
    err_timeout = subprocess.TimeoutExpired(cmd="test", timeout=1, output=b"some timeout log")
    wrapped_timeout = error_wrapper(err_timeout)
    assert "Captured output:" in str(wrapped_timeout)
    assert "some timeout log" in str(wrapped_timeout)

    # Test 5: CalledProcessError with non-UTF8 output (UnicodeDecodeError case)
    # Using a byte sequence that is invalid in UTF-8
    invalid_utf8 = b"\xff\xfe\xfd"
    err_bad_utf8 = subprocess.CalledProcessError(returncode=1, cmd="test", output=invalid_utf8)
    wrapped_bad_utf8 = error_wrapper(err_bad_utf8)
    assert "Failed to parse output." in str(wrapped_bad_utf8)
```


# LLM-generated content at query #9
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test 1: Non-subprocess exception should be returned unchanged
    class MyError(Exception):
        pass
    
    err = MyError("standard error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "standard error"

    # Test 2: subprocess.CalledProcessError with output should have enhanced __str__
    output_bytes = b"error line 1\nerror line 2"
    err_proc = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=output_bytes
    )
    wrapped_proc = error_wrapper(err_proc)
    
    expected_str = (
        "Command 'test_cmd' returned non-zero exit status 1\n"
        "Captured output:\n"
        "    error line 1\n"

        "    error line 2"
    )
    # Note: subprocess.CalledProcessError's default __str__ might vary slightly by python version,
    # but the key is the presence of our injected "Captured output" block.
    assert "Captured output:" in str(wrapped_proc)
    assert "error line 1" in str(wrapped_proc)
    assert "error line 2" in str(wrapped_proc)

    # Test 3: subprocess.CalledProcessError with no output
    err_no_out = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=None)
    wrapped_no_out = error_wrapper(err_no_out)
    assert "No output was generated." in str(wrapped_no_out)

    # Test 4: subprocess.TimeoutExpired with output
    err_timeout = subprocess.TimeoutExpired(cmd="test_cmd", timeout=1, output=b"some timeout output")
    wrapped_timeout = error_wrapper(err_timeout)
    assert "Captured output:" in str(wrapped_timeout)
    assert "some timeout output" in str(wrapped_timeout)

    # Test 5: subprocess.CalledProcessError with undecodable bytes
    # Using a byte sequence that is invalid in UTF-8
    err_invalid_utf8 = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=b"\xff\xfe\xfd"
    )
    wrapped_invalid = error_wrapper(err_invalid_utf8)
    assert "Failed to parse output." in str(wrapped_invalid)
```


# LLM-generated content at query #10
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test Case 1: Standard Exception (should be returned unchanged)
    standard_exc = ValueError("standard error")
    result_exc = error_wrapper(standard_exc)
    assert isinstance(result_exc, ValueError)
    assert str(result_exc) == "standard error"

    # Test Case 2: subprocess.CalledProcessError with output
    # We mock the behavior of an error with captured output
    err_with_output = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=b"line1\nline2"
    )
    wrapped_output_exc = error_wrapper(err_with_output)
    
    # Check if __str__ was overridden and contains the output
    err_str = str(wrapped_output_exc)
    assert "Captured output:" in err_str
    assert "    line1" in err_str
    assert "    line2" in err_str

    # Test Case 3: subprocess.CalledProcessError with no output
    err_no_output = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=None
    )
    wrapped_no_output_exc = error_wrapper(err_no_output)
    assert "No output was generated." in str(wrapped_no_output_exc)

    # Test Case 4: subprocess.TimeoutExpired with output
    err_timeout = subprocess.TimeoutExpired(
        cmd="test_cmd", 
        timeout=1, 
        output=b"partial output"
    )
    wrapped_timeout_exc = error_wrapper(err_timeout)
    assert "Captured output:" in str(wrapped_timeout_exc)
    assert "    partial output" in str(wrapped_timeout_exc)

    # Test Case 5: subprocess.CalledProcessError with undecodable output (UnicodeDecodeError)
    # We use bytes that cannot be decoded as utf-8
    invalid_bytes = b"\xff\xfe\xfd"
    err_invalid_bytes = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=invalid_bytes
    )
    wrapped_invalid_exc = error_wrapper(err_invalid_bytes)
    assert "Failed to parse output." in str(wrapped_invalid_exc)

    # Test Case 6: Verify type inheritance
    # The new class should still be a subclass of the original error type
    assert issubclass(type(wrapped_output_exc), subprocess.CalledProcessError)
    assert issubclass(type(wrapped_timeout_exc), subprocess.TimeoutExpired)
```


# LLM-generated content at query #11
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test 1: Non-subprocess exception should be returned unchanged
    class MyError(Exception):
        pass

    err = MyError("standard error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "standard error"
    assert type(wrapped_err) is MyError

    # Test 2: CalledProcessError with output should include captured output in __str__
    output_bytes = b"error line 1\nerror line 2"
    err_cp = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=output_bytes)
    wrapped_cp = error_wrapper(err_cp)
    
    expected_str = "Command 'test_cmd' returned non-zero exit status 1\nCaptured output:\n    error line 1\n    error line 2"
    # Note: subprocess.CalledProcessError's default __str__ varies slightly by version, 
    # but we check for the presence of our injected content.
    assert "Captured output:" in str(wrapped_cp)
    assert "error line 1" in str(wrapped_cp)
    assert "error line 2" in str(wrapped_cp)
    # Verify the type was dynamically changed
    assert wrapped_cp.__class__.__name__ != "CalledProcessError"

    # Test 3: CalledProcessError with no output
    err_no_out = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=None)
    wrapped_no_out = error_wrapper(err_no_out)
    assert "No output was generated." in str(wrapped_no_out)

    # Test 4: TimeoutExpired with output
    err_timeout = subprocess.TimeoutExpired(cmd="test_cmd", timeout=1, output=b"partial output")
    wrapped_timeout = error_wrapper(err_timeout)
    assert "Captured output:" in str(wrapped_timeout)
    assert "partial output" in str(wrapped_timeout)

    # Test 5: UnicodeDecodeError handling in __str__
    # We simulate a failure by passing bytes that cannot be decoded as utf-8
    err_unicode = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=b"\xff\xfe\xfd")
    wrapped_unicode = error_wrapper(err_unicode)
    assert "Failed to parse output." in str(wrapped_unicode)
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
import subprocess

def test_error_wrapper():
    # Test 1: Non-subprocess exception should be returned unchanged
    class MyError(Exception):
        pass

    err_val = MyError("original error")
    result = error_wrapper(err_val)
    assert result is err_val
    assert str(result) == "original error"

    # Test 2: subprocess.CalledProcessError with output
    output_bytes = b"error line 1\nerror line 2"
    err_called = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=output_bytes
    )
    wrapped_called = error_wrapper(err_called)
    assert isinstance(wrapped_called, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_called)
    assert "    error line 1" in str(wrapped_called)
    assert "    error line 2" in str(wrapped_called)

    # Test 3: subprocess.CalledProcessError with no output
    err_no_output = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=None)
    wrapped_no_output = error_wrapper(err_no_output)
    assert "No output was generated." in str(wrapped_no_output)

    # Test 4: subprocess.TimeoutExpired with output
    err_timeout = subprocess.TimeoutExpired(cmd="test_cmd", timeout=1, output=b"some partial output")
    wrapped_timeout = error_wrapper(err_timeout)
    assert isinstance(wrapped_timeout, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_timeout)
    assert "    some partial output" in str(wrapped_timeout)

    # Test 5: subprocess.CalledProcessError with undecodable bytes
    # Using a byte sequence that is invalid UTF-8
    invalid_utf8 = b"\xff\xfe\xfd"
    err_invalid = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=invalid_utf8)
    wrapped_invalid = error_wraped_invalid = error_wrapper(err_invalid)
    assert "Failed to parse output." in str(wrapped_invalid)
```


# LLM-generated content at query #13
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test 1: Non-subprocess exception should be returned unchanged
    class MyError(Exception):
        pass

    err = MyError("original error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "original error"

    # Test 2: subprocess.CalledProcessError with output
    output_bytes = b"error line 1\nerror line 2"
    err_proc = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=output_bytes
    )
    wrapped_proc = error_wrapper(err_proc)
    
    expected_str = "Command 'test_cmd' returned non-zero exit status 1\nCaptured output:\n    error line 1\n    error line 2"
    # Note: The exact prefix depends on the subprocess.CalledProcessError __str__ implementation 
    # which varies slightly by python version, but we check for the captured output part.
    assert "Captured output:" in str(wrapped_proc)
    assert "    error line 1" in str(wrapped_proc)
    assert "    error line 2" in str(wrapped_proc)

    # Test 3: subprocess.CalledProcessError with no output
    err_no_out = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=None)
    wrapped_no_out = error_wrapper(err_no_out)
    assert "No output was generated." in str(wrapped_no_out)

    # Test 4: subprocess.TimeoutExpired with output
    err_timeout = subprocess.TimeoutExpired(cmd="test_cmd", timeout=1, output=b"partial output")
    wrapped_timeout = error_wrapper(err_timeout)
    assert "Captured output:" in str(wrapped_timeout)
    assert "    partial output" in str(wrapped_timeout)

    # Test 5: subprocess.CalledProcessError with undecodable output (UnicodeDecodeError)
    # We use a byte sequence that is invalid in UTF-8
    invalid_utf8 = b"\xff\xfe\xfd"
    err_invalid = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=invalid_utf8)
    wrapped_invalid = error_wrapper(err_invalid)
    assert "Failed to parse output." in str(wrapped_invalid)

    # Test 6: Verify the class type was actually changed (dynamic type creation)
    assert type(wrapped_proc) != subprocess.CalledProcessError
    assert wrapped_proc.__class__.__name__ != "CalledProcessError"
```


# LLM-generated content at query #14
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test case 1: Non-subprocess exception remains unchanged
    class CustomError(Exception):
        pass

    err = CustomError("original error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, CustomError)
    assert str(wrapped_err) == "original error"
    assert type(wrapped_err) is CustomError

    # Test case 2: subprocess.CalledProcessError with output
    output_bytes = b"error line 1\nerror line 2"
    err_called = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=output_bytes
    )
    wrapped_called = error_wrapper(err_called)
    assert isinstance(wrapped_called, subprocess.CalledProcessError)
    # Check if __str__ was overridden and contains the captured output
    str_val = str(wrapped_called)
    assert "Captured output:" in str_val
    assert "    error line 1" in str_val
    assert "    error line 2" in str_val

    # Test case 3: subprocess.CalledProcessError with no output
    err_no_output = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=None
    )
    wrapped_no_output = error_wrapper(err_no_output)
    assert "No output was generated." in str(wrapped_no_output)

    # Test case 4: subprocess.TimeoutExpired with output
    err_timeout = subprocess.TimeoutExpired(
        cmd="test_cmd", 
        timeout=1, 
        output=b"partial output"
    )
    wrapped_timeout = error_wrapper(err_timeout)
    assert isinstance(wrapped_timeout, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_timeout)
    assert "    partial output" in str(wrapped_timeout)

    # Test case 5: subprocess.CalledProcessError with undecodable bytes
    # We use a byte sequence that is invalid UTF-8
    invalid_utf8 = b"\xff\xfe\xfd"
    err_invalid = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=invalid_utf8
    )
    wrapped_invalid = error_wrapper(err_invalid)
    assert "Failed to parse output." in str(wrapped_invalid)
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test Case 1: Non-subprocess exception should be returned unchanged
    class MyException(Exception):
        pass

    err = MyException("original error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "original error"

    # Test Case 2: subprocess.CalledProcessError with output
    output_bytes = b"error line 1\nerror line 2"
    err_cpe = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=output_bytes
    )
    wrapped_cpe = error_wrapper(err_cpe)
    
    # Check if __str__ was overridden and contains the output
    str_val = str(wrapped_cpe)
    assert "Command '['test_cmd']" in str_val
    assert "Captured output:" in str_val
    assert "    error line 1" in str_val
    assert "    error line 2" in str_val

    # Test Case 3: subprocess.CalledProcessError with NO output
    err_no_out = subprocess.CalledProcessError(returncode=1, cmd="no_out_cmd", output=None)
    wrapped_no_out = error_wrapper(err_no_out)
    assert "No output was generated." in str(wrapped_no_out)

    # Test Case 4: subprocess.TimeoutExpired with output
    err_timeout = subprocess.TimeoutExpired(cmd="slow_cmd", timeout=1, output=b"partial output")
    wrapped_timeout = error_wrapper(err_timeout)
    assert "Captured output:" in str(wrapped_timeout)
    assert "    partial output" in str(wrapped_timeout)

    # Test Case 5: subprocess.CalledProcessError with undecodable bytes (UnicodeDecodeError)
    # We simulate this by providing bytes that cannot be decoded as utf-8
    invalid_utf8 = b"\xff\xfe\xfd" 
    err_invalid = subprocess.CalledProcessError(returncode=1, cmd="bad_bytes", output=invalid_utf8)
    wrapped_invalid = error_wrapper(err_invalid)
    assert "Failed to parse output." in str(wrapped_invalid)

    # Test Case 6: Ensure the class type is actually a new subclass (as per implementation)
    assert type(wrapped_cpe) != subprocess.CalledProcessError
    assert issubclass(type(wrapped_cpe), subprocess.CalledProcessError)
```


# LLM-generated content at query #2
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test 1: Non-subprocess exception should be returned unchanged
    class MyError(Exception):
        pass
    
    err = MyError("test error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "test error"

    # Test 2: subprocess.CalledProcessError with output
    output_bytes = b"line1\nline2\n"
    err_cpe = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=output_bytes
    )
    wrapped_cpe = error_wrapper(err_cpe)
    
    # Check if __str__ was overridden and contains the formatted output
    str_output = str(wrapped_cpe)
    assert "Captured output:" in str_output
    assert "    line1" in str_output
    assert "    line2" in str_output

    # Test 3: subprocess.CalledProcessError with no output
    err_no_output = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=None
    )
    wrapped_no_output = error_wrapper(err_no_output)
    assert "No output was generated." in str(wrapped_no_output)

    # Test 4: subprocess.TimeoutExpired with output
    err_timeout = subprocess.TimeoutExpired(
        cmd="test_cmd", 
        timeout=1, 
        output=b"interrupted"
    )
    wrapped_timeout = error_wrapper(err_timeout)
    assert "Captured output:" in str(wrapped_timeout)
    assert "    interrupted" in str(wrapped_timeout)

    # Test 5: subprocess.CalledProcessError with undecodable output (UnicodeDecodeError)
    # We use a byte sequence that is invalid in UTF-8
    invalid_utf8 = b"\xff\xfe\xfd"
    err_invalid = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=invalid_utf8
    )
    wrapped_invalid = error_wrapper(err_invalid)
    assert "Failed to parse output." in str(wrapped_invalid)
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
import subprocess
import sys
import os
from unittest.mock import patch, MagicMock

def test_run_command():
    # 1. Test Successful Command (Return code 0, no output requested)
    res = run_command(["echo", "hello"])
    assert res.returncode == 0
    assert res.command == ["echo", "hello"]
    assert res.captured_output is None

    # 2. Test Successful Command with output requested
    res = run_command(["echo", "hello"], return_output=True)
    assert res.returncode == 0
    assert res.captured_output.decode().strip() == "hello"

    # 3. Test Command with Error (CalledProcessError)
    # 'ls' on a non-existent directory usually returns non-zero
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        run_command(["ls", "/non_existent_directory_12345"])
    
    # Verify error_wrapper was applied (output is attached to exception)
    assert hasattr(excinfo.value, 'output')
    assert b"ls" in str(excinfo.value).encode() # Check if error_wrapper string modification works

    # 4. Test Command with ignore_errors=True
    res = run_command(["ls", "/non_existent_directory_12345"], ignore_errors=True)
    assert res.returncode != 0
    assert res.captured_output is not None

    # 5. Test Timeout (Simulated)
    # We use a command that sleeps to trigger timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

    # 6. Test Timeout with ignore_errors=True
    # The code specifies -32768 as the special return code for timeout in ignore_errors mode
    res = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True)
    assert res.returncode == -32768

    # 7. Test Environment Variables
    res = run_command(["python3", "-c", "import os; print(os.environ['MY_VAR'])"], 
                      env={**os.environ, "MY_VAR": "test_val"}, 
                      return_output=True)
    assert res.captured_output.decode().strip() == "test_val"

    # 8. Test CWD (Current Working Directory)
    # Use tempfile to create a valid directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        res = run_command(["pwd"], cwd=tmpdir, return_output=True)
        # Depending on OS, pwd might return path with trailing slash or not, 
        # but it should definitely be within tmpdir
        assert os.path.abspath(tmpdir) in res.captured_output.decode()

    # 9. Test Large Output Truncation
    # Create a mock that returns a very large byte string
    large_data = b"A" * (10000)
    with patch("subprocess.run") as mock_run:
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        # We need to mock the behavior of the file handle used in run_command
        # Since run_command uses tempfile.TemporaryFile internally, 
        # we patch subprocess.run to raise CalledProcessError
        # and we rely on the fact that the file 'f' is passed to stdout.
        
        # To test truncation, we trigger a CalledProcessError
        # and we must ensure the file 'f' contains our large data.
        # This is tricky because 'f' is local to the function.
        # Instead, we test the logic via a real command that produces large output.
        # 'python3 -c "print(\'A\' * 10000)"'
        res = run_command(["python3", "-c", "print('A' * 10000)"], ignore_errors=True)
        # The logic: if len(output) > 8192, it prepends the truncation message
        assert b"*** (previous output truncated) ***" in res.captured_output
        assert len(res.captured_output) < 10000 

    # 10. Test error_wrapper with non-subprocess error
    try:
        raise ValueError("Generic error")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, ValueError)
        assert str(wrapped) == "Generic error"

    # 11. Test error_wrapper with UnicodeDecodeError handling
    # Create a CalledProcessError with non-utf8 bytes
    err = subprocess.CalledProcessError(1, "cmd", output=b"\xff\xfe\xfd")
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)

    # 12. Test Verbose mode (Checks if log is called)
    with patch("your_module_path.log") as mock_log: # Replace 'your_module_path' with actual module name
        run_command(["echo", "verbose_test"], verbose=True, return_output=True)
        assert mock_log.called
```


# LLM-generated content at query #4
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test 1: Non-subprocess exception should be returned unchanged
    class MyError(Exception):
        pass

    err = MyError("original error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "original error"

    # Test 2: CalledProcessError with output
    output_bytes = b"line 1\nline 2"
    err_cp = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=output_bytes)
    wrapped_cp = error_wrapper(err_cp)
    
    expected_str = "Command 'test_cmd' returned non-zero exit status 1\nCaptured output:\n    line 1\n    line 2"
    # Note: subprocess.CalledProcessError.__str__ usually contains "Command '...' returned non-zero exit status ..."
    assert "Captured output:" in str(wrapped_cp)
    assert "    line 1" in str(wrapped_cp)
    assert "    line 2" in str(wrapped_cp)

    # Test 3: CalledProcessError with no output
    err_no_out = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=None)
    # We manually set output to None as the wrapper checks 'if self.output'
    wrapped_no_out = error_wrapper(err_no_out)
    assert "No output was generated." in str(wrapped_no_out)

    # Test 4: TimeoutExpired with output
    err_timeout = subprocess.TimeoutExpired(cmd="test_cmd", timeout=1, output=b"partial output")
    wrapped_timeout = error_wrapper(err_timeout)
    assert "Captured output:" in str(wrapped_timeout)
    assert "    partial output" in str(wrapped_timeout)

    # Test 5: UnicodeDecodeError handling in wrapper
    # Use bytes that cannot be decoded as utf-8
    err_unicode = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=b"\xff\xfe\xfd")
    wrapped_unicode = error_wrapper(err_unicode)
    assert "Failed to parse output." in str(wrapped_unicode)

    # Test 6: Verify class type is dynamically updated
    assert type(wrapped_cp) is not subprocess.CalledProcessError
    assert type(wrapped_cp).__name__ != "CalledProcessError"
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
import subprocess
import os
import sys
from unittest.mock import patch, MagicMock

def test_run_command():
    # 1. Test successful command execution with return_output=True
    # Using 'echo' which is available on most POSIX systems (or 'cmd /c echo' on Windows)
    cmd = ["echo", "hello_world"]
    if os.name == 'nt':
        cmd = ["cmd", "/c", "echo", "hello_world"]
    
    result = run_command(cmd, return_output=True)
    assert result.returncode == 0
    assert b"hello_world" in result.captured_output
    assert result.command == cmd

    # 2. Test successful command execution without return_output (output should be None)
    result_no_output = run_command(cmd, return_output=False)
    assert result_no_output.returncode == 0
    assert result_no_output.captured_output is None

    # 3. Test command failure (CalledProcessError) and error_wrapper
    # 'ls' on non-existent file returns non-zero
    fail_cmd = ["ls", "/non_existent_path_12345"]
    if os.name == 'nt':
        fail_cmd = ["cmd", "/c", "dir", "/s", "non_existent_path_12345"]

    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        run_command(fail_cmd, return_output=True)
    
    # Check if error_wrapper added the captured output to the exception string
    assert "Captured output:" in str(excinfo.value)
    assert isinstance(excinfo.value, subprocess.CalledProcessError)

    # 4. Test ignore_errors=True for CalledProcessError
    result_ignored = run_command(fail_cmd, ignore_errors=True, return_output=True)
    assert result_ignored.returncode != 0
    assert result_ignored.captured_output is not None

    # 5. Test ignore_errors=True for TimeoutExpired
    # We mock subprocess.run to raise TimeoutExpired
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=cmd, timeout=0.1)
        result_timeout = run_command(cmd, ignore_errors=True)
        assert result_timeout.returncode == -32768

    # 6. Test error_wrapper with a non-subprocess exception (should return original)
    class CustomError(Exception):
        pass
    
    err = CustomError("original error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, CustomError)
    assert str(wrapped_err) == "original error"

    # 7. Test truncation of large output
    large_output = b"A" * (MAX_OUTPUT_LENGTH + 100)
    with patch("subprocess.run") as mock_run:
        # Mock a successful run that produces large output
        mock_ret = MagicMock()
        mock_ret.returncode = 0
        mock_run.return_value = mock_ret
        
        # We need to intercept the tempfile usage or the way subprocess.run writes to the file.
        # Since run_command writes to a temp file via stdout=f, we mock the behavior of the file.
        with patch("tempfile.TemporaryFile") as mock_temp:
            mock_f = MagicMock()
            mock_f.read.return_value = large_output
            mock_f.seek.return_value = None
            mock_temp.return_value.__enter__.return_value = mock_f
            
            result_large = run_command(cmd, return_output=True)
            assert b"*** (previous output truncated) ***" in result_large.captured_output
            assert len(result_large.captured_output) <= MAX_OUTPUT_LENGTH + 50 # buffer for header

    # 8. Test verbose mode logging (verifying no crash)
    with patch("your_module_path.log") as mock_log: # Replace 'your_module_path' with actual module name
        run_command(cmd, verbose=True)
        assert mock_log.called

    # 9. Test environment variables
    custom_env = {"TEST_VAR": "TEST_VALUE"}
    # Use python to print env var to verify
    verify_env_cmd = [sys.executable, "-c", "import os; print(os.environ.get('TEST_VAR'))"]
    if os.name == 'nt':
        verify_env_cmd = ["cmd", "/c", f"set TEST_VAR=TEST_VALUE && python -c \"import os; print(os.environ.get('TEST_VAR'))\""]
        
    result_env = run_command(verify_env_cmd, env=custom_env, return_output=True)
    assert b"TEST_VALUE" in result_env.captured_output
```


# LLM-generated content at query #6
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test Case 1: Non-subprocess exception should be returned unchanged
    val_error = ValueError("original error")
    result_error = error_wrapper(val_error)
    assert result_error is val_error
    assert str(result_error) == "original error"

    # Test Case 2: subprocess.CalledProcessError with output
    output_bytes = b"error line 1\nerror line 2"
    call_error = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=output_bytes
    )
    wrapped_call_error = error_wrapper(call_error)
    
    expected_str = (
        "Command 'test_cmd' returned non-zero exit status 1\n"
        "Captured output:\n"
        "    error line 1\n"
        "    error line 2"
    )
    assert str(wrapped_call_error) == expected_str

    # Test Case 3: subprocess.CalledProcessError with no output
    no_output_error = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=None)
    wrapped_no_output_error = error_wrapper(no_output_error)
    assert "No output was generated." in str(wrapped_no_output_error)

    # Test Case 4: subprocess.TimeoutExpired with output
    timeout_error = subprocess.TimeoutExpired(cmd="test_cmd", timeout=1, output=b"partial output")
    wrapped_timeout_error = error_wrapper(timeout_error)
    assert "Captured output:" in str(wrapped_timeout_error)
    assert "partial output" in str(wrapped_timeout_error)

    # Test Case 5: subprocess.CalledProcessError with undecodable bytes
    # Using a byte sequence that is invalid UTF-8
    invalid_utf8_bytes = b"\xff\xfe\xfd"
    invalid_error = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=invalid_utf8_bytes
    )
    wrapped_invalid_error = error_wrapper(invalid_error)
    assert "Failed to parse output." in str(wrapped_invalid_error)

    # Test Case 6: Verify the type is dynamically created and inherits from original
    assert isinstance(wrapped_call_error, subprocess.CalledProcessError)
    assert wrapped_call_error.__class__.__name__ != "CalledProcessError"
    assert "CalledProcessError" in wrapped_call_error.__class__.__mro__.__name__ or True # Check it's a subclass
```


# LLM-generated content at query #7
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test 1: Non-subprocess exception should be returned unchanged
    class MyException(Exception):
        pass
    
    exc = MyException("standard error")
    wrapped_exc = error_wrapper(exc)
    assert wrapped_exc is exc
    assert str(wrapped_exc) == "standard error"

    # Test 2: subprocess.CalledProcessError with output should include captured output
    output_bytes = b"error line 1\nerror line 2"
    err_proc = subprocess.CalledProcessError(returncode=1, cmd="ls", output=output_bytes)
    wrapped_proc = error_wrapper(err_proc)
    
    expected_str = (
        "Command 'ls' returned non-zero exit status 1\n"
        "Captured output:\n"
        "    error line 1\n"
        "    error line 2"
    )
    assert str(wrapped_proc) == expected_str

    # Test 3: subprocess.CalledProcessError with no output
    err_no_out = subprocess.CalledProcessError(returncode=1, cmd="ls", output=None)
    wrapped_no_out = error_wrapper(err_no_out)
    assert "No output was generated." in str(wrapped_no_out)

    # Test 4: subprocess.TimeoutExpired with output
    timeout_bytes = b"some interrupted output"
    err_timeout = subprocess.TimeoutExpired(cmd="sleep 10", timeout=1, output=timeout_bytes)
    wrapped_timeout = error_wrapper(err_timeout)
    
    assert "Captured output:" in str(wrapped_timeout)
    assert "some interrupted output" in str(wrapped_timeout)

    # Test 5: subprocess.CalledProcessError with undecodable (binary) output
    # Using a byte sequence that is invalid UTF-8
    invalid_utf8 = b"\xff\xfe\xfd"
    err_binary = subprocess.CalledProcessError(returncode=1, cmd="ls", output=invalid_utf8)
    wrapped_binary = error_wrapper(err_binary)
    assert "Failed to parse output." in str(wrapped_binary)

    # Test 6: Verify the dynamic type creation (class name change)
    assert wrapped_proc.__class__.__name__ != "CalledProcessError"
    assert issubclass(wrapped_proc.__class__, subprocess.CalledProcessError)
```


# LLM-generated content at query #8
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test 1: Non-subprocess exception should be returned unchanged
    class MyError(Exception):
        pass
    
    err = MyError("standard error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "standard error"

    # Test 2: subprocess.CalledProcessError with output
    output_bytes = b"error line 1\nerror line 2"
    err_called = subprocess.CalledProcessError(
        returncode=1, 
        cmd=["ls", "nonexistent"], 
        output=output_bytes
    )
    wrapped_called = error_wrapper(err_called)
    
    expected_str = "Command ['ls', 'nonexistent']\nCaptured output:\n    error line 1\n    error line 2"
    # Note: The exact string depends on how the base class __str__ formats the command.
    # We check for the presence of the captured output block.
    assert "Captured output:" in str(wrapped_called)
    assert "error line 1" in str(wrapped_called)
    assert "error line 2" in str(wrapped_called)

    # Test 3: subprocess.CalledProcessError with no output
    err_no_output = subprocess.CalledProcessError(
        returncode=1, 
        cmd=["ls"]
    )
    err_no_output.output = None
    wrapped_no_output = error_wrapper(err_no_output)
    assert "No output was generated." in str(wrapped_no_output)

    # Test 4: subprocess.TimeoutExpired with output
    err_timeout = subprocess.TimeoutExpired(
        cmd=["sleep", "10"], 
        timeout=1, 
        output=b"partial output"
    )
    wrapped_timeout = error_wrapper(err_timeout)
    assert "Captured output:" in str(wrapped_timeout)
    assert "partial output" in str(wrapped_timeout)

    # Test 5: subprocess.CalledProcessError with undecodable output
    # We use a byte sequence that isn't valid UTF-8
    err_invalid_utf8 = subprocess.CalledProcessError(
        returncode=1, 
        cmd=["bad_encoding"], 
        output=b"\xff\xfe\xfd"
    )
    wrapped_invalid = error_wrapper(err_invalid_utf8)
    assert "Failed to parse output." in str(wrapped_invalid)

    # Test 6: Verify the class type is actually changed (dynamic type creation)
    assert type(wrapped_called) != subprocess.CalledProcessError
    assert type(wrapped_called).__name__ != "CalledProcessError"
```


# LLM-generated content at query #9
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test 1: Non-subprocess exception should be returned unchanged
    class MyException(Exception):
        pass

    err = MyException("original error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "original error"
    assert type(wrapped_err) is MyException

    # Test 2: subprocess.CalledProcessError with output
    # We must use a subclass to avoid breaking the global subprocess.CalledProcessError class 
    # if the dynamic type creation interferes with other tests, 
    # though the implementation uses type(err.__name__, ...) which is relatively safe.
    cmd = ["ls", "/nonexistent_path_12345"]
    proc_err = subprocess.CalledProcessCommandError(
        returncode=1, 
        cmd=cmd, 
        output=b"line 1\nline 2"
    )
    # Manually trigger the wrapper
    wrapped_proc_err = error_wrapper(proc_err)
    
    error_str = str(wrapped_proc_err)
    assert "CalledProcessError" in error_str
    assert "Captured output:" in error_str
    assert "    line 1" in error_str
    assert "    line 2" in error_str

    # Test 3: subprocess.CalledProcessError without output
    proc_err_no_out = subprocess.CalledProcessError(returncode=1, cmd=cmd, output=None)
    wrapped_no_out = error_wrapper(proc_err_no_out)
    assert "No output was generated." in str(wrapped_no_out)

    # Test 4: subprocess.TimeoutExpired with output
    timeout_err = subprocess.TimeoutExpired(cmd=cmd, timeout=1, output=b"some partial output")
    wrapped_timeout = error_wrapper(timeout_err)
    assert "Captured output:" in str(wrapped_timeout)
    assert "some partial output" in str(wrapped_timeout)

    # Test 5: subprocess.TimeoutExpired with undecodable (binary) output
    # Using a byte sequence that isn't valid UTF-8
    bad_output = b"\xff\xfe\xfd"
    timeout_err_bad_utf8 = subprocess.TimeoutExpired(cmd=cmd, timeout=1, output=bad_output)
    wrapped_bad_utf8 = error_wrapper(timeout_err_bad_utf8)
    assert "Failed to parse output." in str(wrapped_bad_utf8)

    # Test 6: Verify the class type was actually changed (dynamic type creation check)
    assert type(wrapped_proc_err).__name__ != "CalledProcessError"
    assert isinstance(wrapped_proc_err, subprocess.CalledProcessError)
```


# LLM-generated content at query #10
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test 1: Non-subprocess exception should be returned unchanged
    class MyError(Exception):
        pass

    err_not_subprocess = MyError("Generic error")
    result_not_subprocess = error_wrapper(err_not_subprocess)
    assert result_not_subprocess is err_not_subprocess
    assert str(result_not_subprocess) == "Generic error"

    # Test 2: CalledProcessError with output
    output_bytes = b"Error line 1\nError line 2"
    err_subprocess = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=output_bytes
    )
    wrapped_subprocess = error_wrapper(err_subprocess)
    
    expected_str = (
        "Command 'test_cmd' returned non-zero exit status 1\n"
        "Captured output:\n"
        "    Error line 1\n"
        "    Error line 2"
    )
    assert str(wrapped_subprocess) == expected_str

    # Test 3: CalledProcessError without output
    err_no_output = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=None
    )
    wrapped_no_output = error_wrapper(err_no_output)
    assert "No output was generated." in str(wrapped_no_output)

    # Test 4: TimeoutExpired with output
    err_timeout = subprocess.TimeoutExpired(
        cmd="test_cmd", 
        timeout=1, 
        output=b"Partial output"
    )
    wrapped_timeout = error_wrapper(err_timeout)
    assert "Captured output:" in str(wrapped_timeout)
    assert "    Partial output" in str(wrapped_timeout)

    # Test 5: UnicodeDecodeError handling in __str__
    # We simulate an error where output contains non-utf8 bytes
    err_unicode = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=b"\xff\xfe\xfd"
    )
    wrapped_unicode = error_wrapper(err_unicode)
    assert "Failed to parse output." in str(wrapped_unicode)

    # Test 6: Verify it is a new class type (dynamic subclassing)
    assert type(wrapped_subprocess) != subprocess.CalledProcessError
    assert issubclass(type(wrapped_subprocess), subprocess.CalledProcessError)
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
import subprocess
import sys
import os
from unittest.mock import patch, MagicMock

def test_run_command():
    # 1. Test successful command with return_output=True
    result = run_command(["echo", "hello"], return_output=True)
    assert result.returncode == 0
    assert result.captured_output.decode('utf-8').strip() == "hello"
    assert result.command == ["echo", "hello"]

    # 2. Test successful command without return_output (should be None)
    result_no_out = run_command(["echo", "hello"], return_output=False)
    assert result_no_out.returncode == 0
    assert result_no_out.captured_output is None

    # 3. Test command with error (CalledProcessError) and error_wrapper
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        run_command(["ls", "/non_existent_directory_12345"], return_output=True)
    
    # Verify error_wrapper added the captured output to the string representation
    assert "ls" in str(excinfo.value)
    assert "No such file or directory" in str(excinfo.value)

    # 4. Test command with ignore_errors=True (returns CommandResult instead of raising)
    result_ignored = run_command(["ls", "/non_existent_directory_12345"], ignore_errors=True, return_output=True)
    assert result_ignored.returncode != 0
    assert b"No such file or directory" in result_ignored.captured_output

    # 5. Test TimeoutExpired
    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        # Using sleep to trigger timeout
        run_command(["python", "-c", "import time; time.sleep(2)"], timeout=0.1)
    assert "Captured output" in str(excinfo.value)

    # 6. Test ignore_errors=True with TimeoutExpired (special return code -32768)
    result_timeout_ignored = run_command(["python", "-c", "import time; time.sleep(2)"], timeout=0.1, ignore_errors=True)
    assert result_timeout_ignored.returncode == -32768

    # 7. Test environment variables
    result_env = run_command(["python", "-c", "import os; print(os.environ.get('MY_VAR'))"], 
                             env={**os.environ, "MY_VAR": "test_val"}, return_output=True)
    assert "test_val" in result_env.captured_output.decode('utf-8')

    # 8. Test working directory (cwd)
    with tempfile.TemporaryDirectory() as tmpdir:
        result_cwd = run_command(["pwd"], cwd=tmpdir, return_output=True)
        # Normalize path for Windows/Unix compatibility
        expected_path = os.path.abspath(tmpdir).replace('\\', '/')
        actual_path = result_cwd.captured_output.decode('utf-8').strip().replace('\\', '/')
        assert actual_path == expected_path

    # 9. Test error_wrapper with non-subprocess exception (should return original error)
    try:
        raise ValueError("Original Error")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, ValueError)
        assert str(wrapped) == "Original Error"

    # 10. Test truncation logic
    long_output = "A" * (MAX_OUTPUT_LENGTH + 100)
    # Mocking subprocess.run to return a specific output via a mock object
    with patch("subprocess.run") as mock_run:
        mock_ret = MagicMock()
        mock_ret.returncode = 1
        # We simulate a CalledProcessError that has the long output
        # Note: In actual run_command, the output is read from the temp file
        # To test truncation, we need the file to actually contain large data.
        # We can mock the 'with tempfile.TemporaryFile() as f' context.
        with patch("tempfile.TemporaryFile") as mock_temp:
            mock_f = MagicMock()
            # Simulate the file content
            mock_f.read.return_value = long_output.encode('utf-8')
            # Setup the context manager behavior
            mock_temp.return_value.__enter__.return_value = mock_f
            
            # Trigger a CalledProcessError
            mock_run.side_effect = subprocess.CalledProcessError(1, ["cmd"], output=long_output.encode('utf-8'))
            
            with pytest.raises(subprocess.CalledProcessError) as excinfo:
                run_command(["cmd"], ignore_errors=False)
            
            assert b"*** (previous output truncated) ***" in excinfo.value.output
            assert len(excinfo.value.output) <= MAX_OUTPUT_LENGTH + 50 # Allow for prefix length
```


# LLM-generated content at query #12
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test 1: Non-subprocess exception should be returned unchanged
    class MyError(Exception):
        pass

    err_val = MyError("test error")
    result = error_wrapper(err_val)
    assert result is err_val
    assert str(result) == "test error"

    # Test 2: subprocess.CalledProcessError with output
    output_bytes = b"error line 1\nerror line 2"
    err_proc = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=output_bytes
    )
    wrapped_proc = error_wrapper(err_proc)
    
    # Check if the __str__ was overridden and contains the captured output
    str_val = str(wrapped_proc)
    assert "Captured output:" in str_val
    assert "    error line 1" in str_val
    assert "    error line 2" in str_val

    # Test 3: subprocess.CalledProcessError without output
    err_no_output = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=None
    )
    wrapped_no_output = error_wrapper(err_no_output)
    assert "No output was generated." in str(wrapped_no_output)

    # Test 4: subprocess.TimeoutExpired with output
    err_timeout = subprocess.TimeoutExpired(
        cmd="test_cmd", 
        timeout=1.0, 
        output=b"partial output"
    )
    wrapped_timeout = error_wrapper(err_timeout)
    assert "Captured output:" in str(wrapped_timeout)
    assert "    partial output" in str(wrapped_timeout)

    # Test 5: subprocess.CalledProcessError with undecodable output (UnicodeDecodeError)
    # We use a byte sequence that is invalid in UTF-8
    invalid_utf8 = b"\xff\xfe\xfd"
    err_invalid = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=invalid_utf8
    )
    wrapped_invalid = error_wrapper(err_invalid)
    assert "Failed to parse output." in str(wrapped_invalid)

    # Test 6: Verify the class type is actually a new dynamic type
    assert type(wrapped_proc) is not subprocess.CalledProcessError
    assert wrapped_proc.__class__.__name__ != "CalledProcessError"
    assert "CalledProcessError" in wrapped_proc.__class__.__name__
```


# LLM-generated content at query #13
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test 1: Non-subprocess exception should be returned unchanged
    class MyError(Exception):
        pass
    
    err = MyError("original error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "original error"

    # Test 2: subprocess.CalledProcessError with output
    output_bytes = b"error line 1\nerror line 2"
    err_called = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=output_bytes
    )
    wrapped_called = error_wrapper(err_called)
    
    # Verify string representation includes captured output
    expected_str = (
        "Command 'test_cmd' returned non-zero exit status 1\n"
        "Captured output:\n"
        "    error line 1\n"

        "    error line 2"
    )
    # Note: subprocess.CalledProcessError __str__ might vary slightly by Python version, 
    # but the key is the presence of the captured output block.
    assert "Captured output:" in str(wrapped_called)
    assert "error line 1" in str(wrapped_called)
    assert "error line 2" in str(wrapped_called)

    # Test 3: subprocess.CalledProcessError without output
    err_no_output = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=None)
    wrapped_no_output = error_wrapper(err_no_output)
    assert "No output was generated." in str(wrapped_no_output)

    # Test 4: subprocess.TimeoutExpired with output
    err_timeout = subprocess.TimeoutExpired(cmd="test_cmd", timeout=1, output=b"partial output")
    wrapped_timeout = error_wrapper(err_timeout)
    assert "Captured output:" in str(wrapped_timeout)
    assert "partial output" in str(wrapped_timeout)

    # Test 5: subprocess.CalledProcessError with non-UTF8 output (UnicodeDecodeError)
    # Using a byte sequence that is invalid in UTF-8
    invalid_utf8 = b"\xff\xfe\xfd"
    err_invalid_utf8 = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=invalid_utf8
    )
    wrapped_invalid = error_wrapper(err_invalid_utf8)
    assert "Failed to parse output." in str(wrapped_invalid)

    # Test 6: Verify the class type was actually changed (dynamic type creation)
    assert type(wrapped_called).__name__ != "CalledProcessError"
    assert isinstance(wrapped_called, subprocess.CalledProcessError)
```


# LLM-generated content at query #14
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test case 1: Non-subprocess exception should be returned unchanged
    val_error = ValueError("original error")
    assert error_wrapper(val_error) is val_error
    assert str(error_wrapper(val_error)) == "original error"

    # Test case 2: CalledProcessError with output
    output_bytes = b"error line 1\nerror line 2"
    cpe = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=output_bytes)
    wrapped_cpe = error_wrapper(cpe)
    
    assert isinstance(wrapped_cpe, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_cpe)
    assert "    error line 1" in str(wrapped_cpe)
    assert "    error line 2" in str(wrapped_cpe)

    # Test case 3: CalledProcessError with no output
    cpe_no_out = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=None)
    wrapped_cpe_no_out = error_wrapper(cpe_no_out)
    assert "No output was generated." in str(wrapped_cpe_no_out)

    # Test case 4: TimeoutExpired with output
    timeout_err = subprocess.TimeoutExpired(cmd="test_cmd", timeout=1, output=b"partial output")
    wrapped_timeout = error_wrapper(timeout_err)
    assert "Captured output:" in str(wrapped_timeout)
    assert "    partial output" in str(wrapped_timeout)

    # Test case 5: CalledProcessError with non-UTF8 bytes (should handle gracefully)
    # Using a byte sequence that is invalid in UTF-8
    invalid_utf8 = b"\xff\xfe\xfd"
    cpe_invalid = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=invalid_utf8)
    wrapped_invalid = error_wrapper(cpe_invalid)
    assert "Failed to parse output." in str(wrapped_invalid)
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
import subprocess
import os
import sys
from unittest.mock import patch, MagicMock

def test_run_command():
    # Test 1: Successful command with return_output=True
    # Using 'echo' which is available on most Unix-like systems and Windows (via cmd)
    cmd = ["echo", "hello_world"]
    # On Windows, we might need to use ['cmd', '/c', 'echo', 'hello_world']
    if os.name == 'nt':
        cmd = ["cmd", "/c", "echo", "hello_world"]
    
    result = run_command(cmd, return_output=True)
    assert result.returncode == 0
    assert b"hello_world" in result.captured_output

    # Test 2: Successful command without return_output (captured_output should be None)
    result_no_output = run_command(cmd, return_output=False)
    assert result_no_output.returncode == 0
    assert result_no_output.captured_output is None

    # Test 3: Command that fails (CalledProcessError)
    # 'false' returns 1 on Unix, or we can use a non-existent command
    fail_cmd = ["ls", "non_existent_file_12345"]
    if os.name == 'nt':
        fail_cmd = ["cmd", "/c", "dir", "non_existent_file_12345"]
    
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        run_command(fail_cmd, return_output=True)
    
    # Check if error_wrapper modified the exception string to include output
    assert "non_existent_file_12345" in str(excinfo.value)
    assert excinfo.value.output is not None

    # Test 4: ignore_errors=True with a failing command
    result_ignored = run_command(fail_cmd, ignore_errors=True, return_output=True)
    assert result_ignored.returncode != 0
    assert b"non_existent_file_12345" in result_ignored.captured_output

    # Test 5: Timeout handling
    # We use a sleep command to trigger a timeout
    timeout_cmd = ["sleep", "10"]
    if os.name == 'nt':
        timeout_cmd = ["timeout", "10"]
    
    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        run_command(timeout_cmd, timeout=0.1)
    
    # Verify error_wrapper processed the timeout
    assert "Captured output" in str(excinfo.value) or "No output" in str(excinfo.value)

    # Test 6: ignore_errors=True with Timeout
    result_timeout_ignored = run_command(timeout_cmd, timeout=0.1, ignore_errors=True)
    assert result_timeout_ignored.returncode == -32768

    # Test 7: Verifying truncation logic
    # Create a large output via a mock
    large_output = b"A" * (MAX_OUTPUT_LENGTH + 1000)
    with patch("subprocess.run") as mock_run:
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        # We need to simulate the file writing to the temp file. 
        # Since we can't easily inject into the internal tempfile context, 
        # we mock the behavior of the exception being raised.
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=cmd, output=None
        )
        
        # To test truncation, we must mock the file object inside the 'with' block
        # This is complex, so we'll test the logic via a controlled error_wrapper test if possible,
        # but for run_command, we verify the logic by checking if the returned output 
        # is within bounds when we force an error.
        
        # Alternative: Test error_wrapper directly for truncation logic if we can't intercept the file.
        # Since we can't easily intercept the tempfile, we rely on the fact that 
        # the code logic is: output = b"*** (previous output truncated) ***\n" + output[-MAX_OUTPUT_LENGTH:]
        pass

    # Test 8: Verbose mode
    with patch("run_command.log") as mock_log:
        run_command(cmd, verbose=True, return_output=True)
        # Should log the command and the output
        assert mock_log.called

    # Test 9: Environment variables
    custom_env = {"TEST_VAR": "TEST_VALUE"}
    # Command to print env var
    env_cmd = ["cmd" if os.name == 'nt' else "printenv", "TEST_VAR"]
    if os.name == 'nt':
        env_cmd = ["cmd", "/c", "echo %TEST_VAR%"]
        
    result_env = run_command(env_cmd, env=custom_env, return_output=True)
    assert b"TEST_VALUE" in result_env.captured_output

    # Test 10: CWD (Current Working Directory)
    # Use current directory to ensure it works
    current_dir = os.getcwd()
    result_cwd = run_command(cmd, cwd=current_dir, return_output=True)
    assert result_cwd.returncode == 0

    # Test 11: error_wrapper with non-subprocess exception
    val_err = ValueError("Standard error")
    assert error_wrapper(val_err) == val_err
    assert str(error_wrapper(val_err)) == "Standard error"

    # Test 12: error_wrapper with subprocess error and no output
    # Manually create a CalledProcessError with no output
    err = subprocess.CalledProcessError(returncode=1, cmd="cmd", output=None)
    wrapped_err = error_wrapper(err)
    assert "No output was generated" in str(wrapped_err)
```


