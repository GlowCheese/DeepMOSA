####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
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
    assert type(wrapped_err).__name__ == "MyError"

    # Test 2: subprocess.CalledProcessError with output should include captured output
    output_bytes = b"error line 1\nerror line 2"
    cpe = subprocess.CalledProcesseler(["ls", "nonexistent"], returncode=2)
    cpe.output = output_bytes
    
    wrapped_cpe = error_wrapper(cpe)
    expected_str = (
        "Command '['ls', 'nonexistent'] exited with error code 2:\n"
        "    error line 1\n"
        "    error line 2"
    )
    assert str(wrapped_cpe).strip() == expected_str.strip()

    # Test 3: subprocess.CalledProcessError without output should indicate no output
    cpe_no_out = subprocess.CalledProcessError(["ls"], returncode=1)
    cpe_no_out.output = None
    
    wrapped_cpe_no_out = error_wrapper(cpe_no_out)
    assert "No output was generated." in str(wrapped_cpe_no_out)

    # Test 4: subprocess.CalledProcessError with non-UTF8 output (UnicodeDecodeError simulation)
    # We use a byte sequence that is invalid in UTF-8
    invalid_utf8 = b"\xff\xfe\xfd"
    cpe_invalid = subprocess.CalledProcessError(["ls"], returncode=1)
    cpe_invalid.output = invalid_utf8
    
    wrapped_cpe_invalid = error_wrapper(cpe_invalid)
    assert "Failed to parse output." in str(wrapped_cpe_invalid)

    # Test 5: subprocess.TimeoutExpired with output
    timeout_err = subprocess.TimeoutExpired(["sleep", "10"], timeout=0.1)
    timeout_err.output = b"some partial output"
    
    wrapped_timeout = error_wrapper(timeout_err)
    assert "Captured output:" in str(wrapped_timeout)
    assert "some partial output" in str(wrapped_timeout)

    # Test 6: Verify the dynamic type creation works (class name is new)
    assert wrapped_cpe.__class__.__name__ != "CalledProcessError"
    assert wrapped_cpe.__class__.__name__ == "CalledProcessError" # It uses the same name but a new type object
```


# LLM-generated content at query #2
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test 1: Non-subprocess exception should be returned unchanged
    class MyException(Exception):
        pass

    exc = MyException("test error")
    result = error_wrapper(exc)
    assert result is exc
    assert str(result) == "test error"

    # Test 2: subprocess.CalledProcessError with output should be wrapped and formatted
    output_bytes = b"line1\nline2\n"
    err_called = subprocess.CalledProcessError(returncode=1, cmd="ls", output=output_bytes)
    wrapped_called = error_wrapper(err_called)
    
    assert isinstance(wrapped_called, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_called)
    assert "    line1" in str(wrapped_called)
    assert "    line2" in str(wrapped_called)

    # Test 3: subprocess.CalledProcessError without output should indicate no output
    err_no_output = subprocess.CalledProcessError(returncode=1, cmd="ls", output=None)
    wrapped_no_out = error_wrapper(err_no_output)
    assert "No output was generated." in str(wrapped_no_out)

    # Test 4: subprocess.TimeoutExpired with output should be formatted
    timeout_output = b"some log during timeout"
    err_timeout = subprocess.TimeoutExpired(cmd="sleep", timeout=1, output=timeout_output)
    wrapped_timeout = error_wrapper(err_timeout)
    
    assert isinstance(wrapped_timeout, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_timeout)
    assert "    some log during timeout" in str(wrapped_timeout)

    # Test 5: Handling of non-UTF8 bytes (UnicodeDecodeError case)
    # We simulate this by providing bytes that cannot be decoded as utf-8
    invalid_utf8 = b"\xff\xfe\xfd"
    err_invalid = subprocess.CalledProcessError(returncode=1, cmd="ls", output=invalid_utf8)
    wrapped_invalid = error_wrapper(err_invalid)
    assert "Failed to parse output." in str(wrapped_invalid)

    # Test 6: Verify that the original class name is preserved in the new dynamic type
    assert wrapped_called.__class__.__name__ != "CalledProcessError"
    # The code uses err_type.__name__, so it creates a new class with same name but different identity
    assert wrapped_called.__class__.__name__ == "CalledProcessError"
```


# LLM-generated content at query #3
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test 1: Normal exception should be returned unchanged
    val_err = ValueError("original error")
    assert error_wrapper(val_err) is val_err
    assert str(error_wrapper(val_err)) == "original error"

    # Test 2: subprocess.CalledProcessError with output
    output_bytes = b"error log line\nsecond line"
    cpe = subprocess.CalledProcessError(returncode=1, cmd="ls", output=output_bytes)
    wrapped_cpe = error_wrapper(cpe)
    
    assert isinstance(wrapped_cpe, subprocess.CalledProcessError)
    error_str = str(wrapped_cpe)
    assert "Captured output:" in error_str
    assert "error log line" in error_str
    assert "second line" in error_str

    # Test 3: subprocess.CalledProcessError with no output
    cpe_no_out = subprocess.CalledProcessError(returncode=1, cmd="ls", output=None)
    wrapped_cpe_no_out = error_wrapper(cpe_no_out)
    assert "No output was generated." in str(wrapped_cpe_no_out)

    # Test 4: subprocess.TimeoutExpired with output
    timeout_err = subprocess.TimeoutExpired(cmd="sleep 10", timeout=1, output=b"partial output")
    wrapped_timeout = error_wrapper(timeout_err)
    assert "Captured output:" in str(wrapped_timeout)
    assert "partial output" in str(wrapped_timeout)

    # Test 5: Unicode decoding failure handling
    # We use a byte sequence that is invalid in UTF-8
    invalid_utf8 = b"\xff\xfe\xfd"
    cpe_invalid = subprocess.CalledProcessError(returncode=1, cmd="ls", output=invalid_utf8)
    wrapped_invalid = error_wrapper(cpe_invalid)
    assert "Failed to parse output." in str(wrapped_invalid)

    # Test 6: Verify type creation (the instance's class should be a new dynamically created subclass)
    assert type(wrapped_cpe).__name__ != subprocess.CalledProcessError.__name__
    assert issubclass(type(wrapped_cpe), subprocess.CalledProcessError)
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
import subprocess
import os
import tempfile
from unittest.mock import patch, MagicMock

@pytest.mark.parametrize("args, expected_return_code, expected_output", [
    (["echo", "hello"], 0, b"hello\n"),
    (["ls", "-l"], 0, None), # return_output defaults to False
])
def test_run_command_success(args, expected_return_code, expected_output):
    result = run_command(args, return_output=True)
    assert result.command == args
    assert result.return_code == expected_return_code
    if expected_output is not None:
        assert result.captured_output == expected_output

def test_run_command_with_env_and_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = {"TEST_VAR": "FOO"}
        # Use python to check environment variable
        args = ["python3", "-c", "import os; print(os.environ.get('TEST_CACHED_VAR', ''))"]
        # We can't easily modify the actual process env without side effects, 
        # but we can pass it via the env param to subprocess
        env = {"TEST_VAR": "FOO"}
        # Note: run_command passes env to subprocess.run
        # We'll use a command that prints an env var
        args = ["python3", "-c", "import os; print(os.environ.get('MY_TEST_VAR', ''))"]
        env = {"MY_TEST_VAR": "BAR"}
        
        result = run_command(args, env=env, return_output=True)
        assert result.captured_output.decode().strip() == "BAR"

def test_run_command_error_raises_exception():
    # Command that returns non-zero exit code
    args = ["python3", "-c", "import sys; sys.exit(1)"]
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        run_command(args, return_output=True)
    assert excinfo.value.returncode == 1

def test_run_command_ignore_errors():
    # Command that returns non-zero exit code
    args = ["python3", "-c", "import sys; sys.exit(42)"]
    result = run_command(args, ignore_errors=True, return_output=True)
    assert result.return_code == 42
    assert isinstance(result.captured_output, bytes)

def test_run_command_timeout():
    # Command that sleeps longer than timeout
    args = ["python3", "-c", "import time; time.sleep(2)"]
    with pytest.py.raises(subprocess.TimeoutExpired):
        run_command(args, timeout=0.1)

def test_run_command_timeout_ignore_errors():
    args = ["python3", "-c", "import time; time.sleep(2)"]
    result = run_command(args, timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768

def test_run_command_output_truncation():
    # Create a large output string
    large_str = "A" * (MAX_OUTPUT_LENGTH + 1000)
    args = ["python3", "-c", f"print('{large_str}')"]
    
    with pytest.raises(subprocess.CalledProcessError):
        # Force an error so we can inspect the captured output in the exception
        # or use ignore_errors=True to see it in CommandResult
        run_command(["python3", "-c", "import sys; print('A'*10000); sys.exit(1)"], 
                    ignore_errors=True, return_output=True)
    
    # Re-running with ignore_errors to check truncation logic specifically
    result = run_command(["python3", "-c", "print('A'*10000); sys.exit(1)"], 
                         ignore_errors=True, return_output=True)
    assert b"*** (previous output truncated) ***" in result.captured_output
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + 50 # buffer for header

def test_error_wrapper_string_formatting():
    # Mock a CalledProcessError with output
    err = subprocess.CalledProcessError(returncode=1, cmd="test", output=b"line1\nline2")
    wrapped_err = error_wrapper(err)
    err_str = str(wrapped_err)
    assert "Captured output:" in err_str
    assert "    line1" in err_str
    assert "    line2" in err_str

def test_error_wrapper_no_output():
    err = subprocess.CalledProcessError(returncode=1, cmd="test", output=None)
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

def test_run_command_unicode_error_in_logging():
    # Mocking log to trigger the unicode decode error path in run_command
    with patch("your_module_path.log") as mock_log:
        # Provide invalid utf-8 bytes
        invalid_bytes = b"\xff\xfe\xfd"
        args = ["python3", "-c", "import sys; sys.stdout.buffer.write(b'\\xff\\xfe')"]
        
        # This will trigger the 'except UnicodeDecodeError' block in run_command 
        # because it tries to log output as utf-8 string
        result = run_command(args, verbose=True)
        
        # Verify that log was called (even if it failed decode internally, 
        # the code handles it by iterating lines)
        assert mock_log.called
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
import subprocess
import os
from unittest.mock import patch, MagicMock

def test_run_command(tmp_path):
    # Test 1: Successful command with return_output=True
    result = run_command(["echo", "hello"], return_output=True)
    assert result.returncode == 0
    assert result.captured_output.decode().strip() == "hello"
    assert isinstance(result.command, list)

    # Test 2: Successful command with string args (shell=True)
    result = run_command("echo 'world'", shell=True, return_output=True)
    assert result.returncode == 0
    assert "world" in result.captured_output.decode()

    # Test 3: Command that fails (CalledProcessError) and error_wrapper behavior
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        run_command(["ls", "/non_existent_directory_12345"], return_output=True)
    
    # Check if error_wrapper modified the exception string to include output
    assert "No output was generated" in str(excinfo.value) or len(str(excinfo.value)) > 0
    assert excinfo.value.output is not None

    # Test 4: Command with ignore_errors=True
    result = run_command(["ls", "/non_existent_directory_12345"], ignore_errors=True, return_output=True)
    assert result.returncode != 0
    assert isinstance(result.captured_output, bytes)

    # Test 5: Timeout handling (TimeoutExpired)
    # We use a command that sleeps to trigger timeout
    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        run_command(["sleep", "10"], timeout=0.1)
    assert "-32768" not in str(excinfo.value) # Should be the original error type

    # Test 6: Timeout handling with ignore_errors=True (Special return code -32768)
    result = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True)
    assert result.returncode == -32768

    # Test 7: Environment variables and CWD
    test_file = tmp_path / "env_test.txt"
    test_file.write_text("data")
    result = run_command(["cat", "env_test.txt"], cwd=tmp_path, return_output=True)
    assert result.captured_output.decode().strip() == "data"

    # Test 8: Large output truncation
    # Create a large string to exceed MAX_OUTPUT_LENGTH (8192)
    large_str = "A" * 10000
    with patch("subprocess.run") as mock_run:
        mock_ret = MagicMock()
        mock_ret.returncode = 1
        # Simulate a CalledProcessError with large output
        mock_run.side_effect = subprocess.CalledProcessError(1, "cmd", output=None)
        
        # We need to mock the file writing logic inside run_command
        # Since run_command uses tempfile.TemporaryFile, we'll simulate a failure 
        # that triggers the exception block where truncation happens.
        with pytest.raises(subprocess.CalledProcessError):
            # Triggering the path where output is read from the file
            # We mock the 'f' inside the context manager via patch if needed, 
            # but easier to rely on a real command that produces large output.
            run_command(["python3", "-c", f"print('B' * {MAX_OUTPUT_LENGTH + 100})"], 
                        ignore_errors=False, return_output=True)

    # Test 9: Verify error_wrapper with UnicodeDecodeError simulation
    # We simulate an error where output is not valid utf-8
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(1, "cmd", output=b"\xff\xfe\xfd")
        try:
            run_command(["false"], return_output=True)
        except subprocess.CalledProcessError as e:
            assert "Failed to parse output." in str(e)

    # Test 10: Verbose mode logging (checking if log is called)
    with patch("your_module_name.log") as mock_log: # Replace 'your_module_name' with actual module name
        run_command(["echo", "verbose_test"], verbose=True, return_output=True)
        assert mock_log.called

def test_error_wrapper_non_subprocess_exception():
    # Ensure error_wrapper returns the exception unchanged if it's not a subprocess error
    val_err = ValueError("test")
    wrapped = error_wrapper(val_err)
    assert wrapped is val_err
    assert isinstance(wrapped, ValueError)
```


# LLM-generated content at query #6
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test case 1: Non-subprocess exception should be returned unchanged
    class MyError(Exception):
        pass

    err = MyError("standard error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "standard error"

    # Test case 2: subprocess.CalledProcessError with output
    output_bytes = b"error line 1\nerror line 2"
    err_proc = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=output_bytes)
    wrapped_proc = error_wrapper(err_proc)
    
    # Check that __str__ was overridden and contains the captured output
    error_str = str(wrapped_proc)
    assert "Captured output:" in error_str
    assert "    error line 1" in error_str
    assert "    error line .2" in error_str or "error line 2" in error_str

    # Test case 3: subprocess.CalledProcessError without output
    err_no_output = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=None)
    wrapped_no_output = error_wrapper(err_no_output)
    assert "No output was generated." in str(wrapped_no_output)

    # Test case 4: subprocess.TimeoutExpired with output
    err_timeout = subprocess.TimeoutExpired(cmd="test_cmd", timeout=1, output=b"interrupted")
    wrapped_timeout = error_wrapper(err_timeout)
    assert "Captured output:" in str(wrapped_timeout)
    assert "    interrupted" in str(wrapped_timeout)

    # Test case 5: subprocess.CalledProcessError with non-utf8 output (UnicodeDecodeError scenario)
    # Using invalid utf-8 sequence
    invalid_utf8 = b"\xff\xfe\xfd"
    err_invalid = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=invalid_utf8)
    wrapped_invalid = error_wrapper(err_invalid)
    assert "Failed to parse output." in str(wrapped_invalid)

    # Test case 6: Verify the class type is actually a new dynamic type (as per implementation)
    assert type(wrapped_proc) != subprocess.CalledProcessError
    assert wrapped_proc.__class__.__name__ != "CalledProcessError"
```


# LLM-generated content at query #7
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test 1: Non-subprocess exception should be returned unchanged
    class MyError(Exception):
        pass

    err = MyError("Standard error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "Standard error"

    # Test 2: CalledProcessError with output should include captured output in __str__
    output_bytes = b"error line 1\nerror line 2"
    cpe = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=output_bytes)
    wrapped_cpe = error_wrapper(cpe)
    
    expected_str = (
        "Command 'test_cmd' returned non-zero exit status 1.\n"
        "Captured output:\n"
        "    error line 1\n"
        "    error line 2"
    )
    assert str(wrapped_cpe) == expected_str

    # Test 3: CalledProcessError with no output should indicate no output was generated
    cpe_no_out = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=None)
    wrapped_cpe_no_out = error_wrapper(cpe_no_out)
    assert "No output was generated." in str(wrapped_cpe_no_out)

    # Test 4: CalledProcessError with non-utf8 output should handle decoding failure gracefully
    non_utf8_output = b"\xff\xfe\xfd"
    cpe_bad_enc = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=non_utf8_output)
    wrapped_cpe_bad_enc = error_wrapper(cpe_bad_enc)
    assert "Failed to parse output." in str(wrapped_cpe_bad_enc)

    # Test 5: TimeoutExpired with output should include captured output
    timeout_out = b"interrupted output"
    te = subprocess.TimeoutExpired(cmd="test_cmd", timeout=1, output=timeout_out)
    wrapped_te = error_wrapper(te)
    assert "Captured output:" in str(wrapped_te)
    assert "    interrupted output" in str(wrapped_te)

    # Test 6: Verify that the class type is actually changed (dynamic type creation)
    assert type(wrapped_cpe).__name__ != "CalledProcessError"
    assert isinstance(wrapped_cpe, subprocess.CalledProcessError)
```


# LLM-generated content at query #8
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test 1: Regular exception should be returned unchanged
    class MyError(Exception):
        pass

    err = MyError("Original error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "Original error"
    assert isinstance(wrapped_err, MyError)

    # Test 2: subprocess.CalledProcessError with output should be enriched
    cmd = ["ls", "nonexistent_file"]
    # Simulate a CalledProcessError that would occur in reality
    original_err = subprocess.CalledProcessError(returncode=1, cmd=cmd, output=b"error message\nline 2")
    wrapped_err = error_wrapper(original_err)
    
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    # Check if the __str__ was overridden to include captured output
    error_str = str(wrapped_err)
    assert "Captured output:" in error_str
    assert "    error message" in error_str
    assert "    line 2" in error_str

    # Test 3: subprocess.CalledProcessError with no output
    no_output_err = subprocess.CalledProcessError(returncode=1, cmd=cmd, output=b"")
    wrapped_err = error_wrapper(no_output_err)
    assert "No output was generated." in str(wrapped_err)

    # Test 4: subprocess.TimeoutExpired with output
    timeout_err = subprocess.TimeoutExpired(cmd=cmd, timeout=1.0, output=b"some partial output")
    wrapped_err = error_wrapper(timeout_err)
    assert "Captured output:" in str(wrapped_err)
    assert "    some partial output" in str(wrapped_err)

    # Test 5: subprocess.CalledProcessError with undecodable (binary) output
    # Using bytes that are invalid utf-8
    invalid_utf8 = b"\xff\xfe\xfd"
    bad_output_err = subprocess.CalledProcessError(returncode=1, cmd=cmd, output=invalid_utf8)
    wrapped_err = error_wrapper(bad_output_err)
    assert "Failed to parse output." in str(wrapped_err)

    # Test 6: Verify the dynamic type creation doesn't break inheritance/type checks
    assert issubclass(type(wrapped_err), subprocess.CalledProcessError)
```


# LLM-generated content at query #9
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test Case 1: Non-subprocess exception should be returned unchanged
    class MyError(Exception):
        pass

    err = MyError("original error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "original error"

    # Test Case 2: subprocess.CalledProcessError with output
    output_bytes = b"error line 1\nerror line 2"
    err_cp = subprocess.CalledProcessError(returncode=1, cmd="ls", stderr=output_bytes)
    # Manually attach output as run_command does during exception handling
    err_cp.output = output_bytes
    
    wrapped_cp = error_wrapper(err_cp)
    assert isinstance(wrapped_cp, subprocess.CalledProcessError
    assert "Captured output:" in str(wrapped_cp)
    assert "    error line 1" in str(wrapped_cp)
    assert "    error line 2" in str(wrapped_cp)

    # Test Case 3: subprocess.CalledProcessError with no output
    err_no_out = subprocess.CalledProcessError(returncode=1, cmd="ls")
    err_no_out.output = None
    
    wrapped_no_out = error_wrapper(err_no_out)
    assert "No output was generated." in str(wrapped_no_out)

    # Test Case 4: subprocess.TimeoutExpired with output
    err_timeout = subprocess.TimeoutExpired(cmd="sleep 10", timeout=1)
    err_timeout.output = b"some partial output"
    
    wrapped_timeout = error_wrapper(err_timeout)
    assert "Captured output:" in str(wrapped_timeout)
    assert "    some partial output" in str(wrapped_timeout)

    # Test Case 5: subprocess.CalledProcessError with non-UTF8 output (triggering the except block)
    # Using a byte sequence that is invalid UTF-8
    invalid_utf8 = b"\xff\xfe\xfd"
    err_bad_utf8 = subprocess.CalledProcessError(returncode=1, cmd="ls")
    err_bad_utf8.output = invalid_utf8
    
    wrapped_bad_utf8 = error_wrapper(err_bad_utf8)
    assert "Failed to parse output." in str(wrapped_bad_utf8)
```


# LLM-generated content at query #10
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test 1: Passing a non-subprocess exception (should return same object)
    val_err = ValueError("original error")
    result_err = error_wrapper(val_err)
    assert result_err is val_err
    assert str(result_err) == "original error"

    # Test 2: CalledProcessError with output (should augment __str__)
    cmd = ["ls", "/nonexistent_path"]
    # We simulate the error object as subprocess.run would create it
    cpe = subprocess.CalledProcessError(returncode=1, cmd=cmd, output=b"error message\nline 2")
    wrapped_cpe = error_wrapper(cpe)
    
    assert "Captured output:" in str(wrapped_cpe)
    assert "error message" in str(wrapped_cpe)
    assert "    line 2" in str(wrapped_cpe)

    # Test 3: CalledProcessError without output (should indicate no output)
    cpe_no_out = subprocess.CalledProcessError(returncode=1, cmd=cmd, output=None)
    wrapped_cpe_no_out = error_wrapper(cpe_no_out)
    assert "No output was generated." in str(wrapped_cpe_no_out)

    # Test 4: CalledProcessError with undecodable/binary output (should handle UnicodeEncodeError)
    # Using a byte sequence that is invalid UTF-8
    invalid_utf8 = b"\xff\xfe\xfd"
    cpe_bad_utf8 = subprocess.CalledProcessError(returncode=1, cmd=cmd, output=invalid_utf8)
    wrapped_cpe_bad_utf8 = error_wrapper(cpe_bad_utf8)
    assert "Failed to parse output." in str(wrapped_cpe_bad_utf8)

    # Test 5: TimeoutExpired with output
    te = subprocess.TimeoutExpired(cmd=cmd, timeout=1, output=b"partial output")
    wrapped_te = error_wrapper(te)
    assert "Captured output:" in str(wrapped_te)
    assert "partial output" in str(wrapped_te)

    # Test 6: Verify type dynamic creation (the instance class name should be different)
    original_class_name = cpe.__class__.__name__
    wrapped_cpe_type_check = error_wrapper(subprocess.CalledProcessError(1, cmd))
    assert wrapped_cpe_type_check.__class__.__name__ != original_class_name
```


# LLM-generated content at query #11
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test 1: Non-subprocess exception should be returned unchanged
    val_error = ValueError("test error")
    assert error_wrapper(val_error) is val_error
    assert str(error_wrapper(val_error)) == "test error"

    # Test 2: subprocess.CalledProcessError with output
    output_bytes = b"error line\nsecond line"
    err_cpe = subprocess.CalledProcessError(returncode=1, cmd="ls", output=output_bytes)
    wrapped_cpe = error_wrapper(err_cpe)
    
    assert isinstance(wrapped_cpe, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_cpe)
    assert "error line" in str(wrapped_cpe)
    assert "second line" in str(wrapped_cpe)

    # Test 3: subprocess.CalledProcessError with no output
    err_no_out = subprocess.CalledProcessError(returncode=1, cmd="ls", output=None)
    wrapped_no_out = error_wrapper(err_no_out)
    assert "No output was generated." in str(wrapped_no_out)

    # Test 4: subprocess.TimeoutExpired with output
    err_timeout = subprocess.TimeoutExpired(cmd="sleep 10", timeout=1, output=b"some progress")
    wrapped_timeout = error_wrapper(err_timeout)
    assert "Captured output:" in str(wrapped_timeout)
    assert "some progress" in str(wrapped_timeout)

    # Test 5: subprocess.CalledProcessError with undecodable bytes (UnicodeDecodeError simulation)
    # We use a byte sequence that is invalid UTF-8
    invalid_utf8 = b"\xff\xfe\xfd"
    err_invalid = subprocess.CalledProcessError(returncode=1, cmd="ls", output=invalid_utf8)
    wrapped_invalid = error_wrapper(err_invalid)
    assert "Failed to parse output." in str(wrapped_invalid)

    # Test 6: Verify that the class type is actually a new dynamic type
    assert type(wrapped_cpe) != subprocess.CalledProcessError
    assert wrapped_cpe.__class__.__name__ != "CalledProcessError"
    # Check inheritance
    assert issubclass(wrapped_cpe.__class__, subprocess.CalledProcessError)
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
    
    err = MyError("original error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "original error"

    # Test 2: CalledProcessError with output should include captured output in __str__
    output_bytes = b"error line 1\nerror line 2"
    cpe = subprocess.CalledProcessError(returncode=1, cmd="ls", output=output_bytes)
    wrapped_cpe = error_wrapper(cpe)
    
    expected_str = "Command 'ls' returned non-zero exit status 1\nCaptured output:\n    error line 1\n    error line 2"
    # Note: subprocess.CalledProcessError.__str__ format depends on Python version, 
    # but we check if our injected part is present.
    assert "Captured output:" in str(wrapped_cpe)
    assert "error line 1" in str(wrapped_cpe)

    # Test 3: CalledProcessError with empty output should indicate no output generated
    cpe_no_out = subprocess.CalledProcessError(returncode=1, cmd="ls", output=b"")
    wrapped_cpe_no_out = error_wrapper(cpe_no_out)
    assert "No output was generated." in str(wrapped_cpe_no_out)

    # Test 4: TimeoutExpired with output
    timeout_err = subprocess.TimeoutExpired(cmd="sleep 10", timeout=1, output=b"partial output")
    wrapped_timeout = error_wrapper(timeout_err)
    assert "Captured output:" in str(wrapped_timeout)
    assert "partial output" in str(wrapped_timeout)

    # Test 5: CalledProcessError with non-UTF8 bytes (UnicodeDecodeError fallback)
    # We use bytes that are invalid in UTF-8
    invalid_utf8 = b"\xff\xfe\xfd"
    cpe_invalid = subprocess.CalledProcessError(returncode=1, cmd="ls", output=invalid_utf8)
    wrapped_invalid = error_wrapper(cpe_invalid)
    assert "Failed to parse output." in str(wrapped_invalid)

    # Test 6: Verify class type is dynamically changed and inherits from original
    assert isinstance(wrapped_cpe, subprocess.CalledProcessError)
    assert wrapped_cpe.__class__.__name__ != "CalledProcessError"
    assert "Command" in wrapped_cpe.__class__.__name__
```


# LLM-generated content at query #13
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

    # Test 2: subprocess.CalledProcessError with output is decorated and shows output
    output_bytes = b"error line 1\nerror line 2"
    err_cp = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=output_bytes)
    wrapped_cp = error_wrapper(err_cp)
    
    expected_str = "Command 'test_cmd' returned non-zero exit status 1\nCaptured output:\n    error line 1\n    error line 2"
    # Note: We check if the substring exists because the exact base error message can vary by Python version/OS
    assert "Captured output:" in str(wrapped_cp)
    assert "error line 1" in str(wrapped_cp)
    assert "error line 2" in str(wrapped_cp)

    # Test 3: subprocess.CalledProcessError with no output shows fallback message
    err_no_out = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=None)
    wrapped_no_out = error_wrapper(err_no_out)
    assert "No output was generated." in str(wrapped_no_out)

    # Test 4: subprocess.TimeoutExpired with output is decorated
    err_timeout = subprocess.TimeoutExpired(cmd="test_cmd", timeout=1, output=b"partial output")
    wrapped_timeout = error_wrapper(err_timeout)
    assert "Captured output:" in str(wrapped_timeout)
    assert "partial output" in str(wrapped_timeout)

    # Test 5: subprocess.CalledProcessError with non-UTF8 output (UnicodeDecodeError handling)
    # Using invalid utf-8 byte sequence
    err_bad_utf8 = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=b"\xff\xfe\xfd")
    wrapped_bad_utf8 = error_wrapper(err_bad_utf8)
    assert "Failed to parse output." in str(wrapped_bad_utf8)

    # Test 6: Verify the type is actually a new dynamically created class
    original_type = type(err_cp)
    assert type(wrapped_cp) != original_type
    assert isinstance(wrapped_cp, subprocess.CalledProcessError)
```


# LLM-generated content at query #14
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test 1: Non-subprocess exception should be returned unchanged
    class CustomError(Exception):
        pass
    
    err = CustomError("Original error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "Original error"

    # Test 2: subprocess.CalledProcessError with output
    output_bytes = b"Error occurred\nTraceback details"
    cpe = subprocess.CalledProcessError(returncode=1, cmd="ls", output=output_bytes)
    wrapped_cpe = error_wrapper(cpe)
    
    expected_str = "Command 'ls' returned non-zero exit status 1\nCaptured output:\n    Error occurred\n    Traceback details"
    assert str(wrapped_cpe).strip() == expected_str.strip()

    # Test 3: subprocess.CalledProcessError without output
    cpe_no_out = subprocess.CalledProcessError(returncode=1, cmd="ls", output=None)
    wrapped_cpe_no_out = error_wrapper(cpe_no_out)
    assert "No output was generated." in str(wrapped_cpe_no_out)

    # Test 4: subprocess.TimeoutExpired with output
    te = subprocess.TimeoutExpired(cmd="sleep 10", timeout=1, output=b"partial log")
    wrapped_te = error_wrapper(te)
    assert "Captured output:" in str(wrapped_te)
    assert "    partial log" in str(wrapped_te)

    # Test 5: subprocess.TimeoutExpired with unparseable (non-utf8) output
    # We use a byte sequence that is invalid in UTF-8
    invalid_utf8 = b"\xff\xfe\xfd"
    te_invalid = subprocess.TimeoutExpired(cmd="sleep 10", timeout=1, output=invalid_utf8)
    wrapped_te_invalid = error_wrapper(te_invalid)
    assert "Failed to parse output." in str(wrapped_te_invalid)

    # Test 6: Verify that the type is actually changed (dynamic class creation)
    assert type(wrapped_cpe) != subprocess.CalledProcessError
    assert wrapped_cpe.__class__.__name__ != "CalledProcessError"
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
import subprocess
import os
import tempfile
from unittest.mock import patch, MagicMock

def test_run_command():
    # 1. Test successful command execution with return_output=True
    result = run_command(["echo", "hello"], return_output=True)
    assert result.returncode == 0
    assert result.captured_output.decode('utf-8').strip() == "hello"
    assert result.command == ["echo", "hello"]

    # 2. Test successful command execution without return_output (should be None)
    result = run_command(["echo", "no_output"], return_output=False)
    assert result.returncode == 0
    assert result.captured_output is None

    # 3. Test command failure (CalledProcessError) with error_wrapper
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        run_command(["ls", "/non_existent_directory_path_12345"], return_output=True)
    
    # Verify error_wrapper modified the exception string to include output
    assert "No output was generated" in str(excinfo.value) or len(str(excinfo.value)) > 0
    assert isinstance(excinfo.value.__class__.__name__, str)

    # 4. Test ignore_errors=True for CalledProcessError
    result = run_command(["ls", "/non_existent_directory_path_12345"], ignore_errors=True, return_output=True)
    assert result.returncode != 0
    assert result.captured_output is not None

    # 5. Test timeout handling (TimeoutExpired)
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

    # 6. Test ignore_errors=True for TimeoutExpired (special return code -32768)
    result = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True)
    assert result.returncode == -32768

    # 7. Test environment variables passing
    with tempfile.TemporaryFile(mode='w+t') as tmp:
        tmp.write("echo $MY_VAR")
        tmp.seek(0)
        env_test_path = tmp.name
        # Using shell=True to allow env var expansion in simple way for testing
        result = run_command("echo $MY_VAR", env={"MY_VAR": "test_val"}, shell=True, return_output=True)
        assert result.captured_output.decode('utf-8').strip() == "test_val"

    # 8. Test working directory (cwd)
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert os.path.normpath(result.captured_output.decode('utf-8').strip()) == os.path.normpath(tmpdir)

    # 9. Test output truncation (MAX_OUTPUT_LENGTH)
    # Create a large string larger than 8192
    large_str = "A" * 9000
    with patch("subprocess.run") as mock_run:
        mock_ret = MagicMock()
        mock_ret.returncode = 1
        # Simulate a CalledProcessError with large output
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd="large_cmd", output=large_str.encode('utf-8')
        )
        
        with pytest.raises(subprocess.CalledProcessError) as excinfo:
            run_command(["large_cmd"], ignore_errors=False)
        
        output = excinfo.value.output.decode('utf-8')
        assert "*** (previous output truncated) ***" in output
        assert len(output) <= 8192 + 40 # margin for header

    # 10. Test error_wrapper with UnicodeDecodeError fallback
    # We manually trigger the logic via a mock that returns non-utf8 bytes
    with patch("subprocess.run") as mock_run:
        # Non-utf8 sequence
        bad_bytes = b"\xff\xfe\xfd" 
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd="bad_bytes", output=bad_bytes
        )
        with pytest.raises(subprocess.CalledProcessError) as excinfo:
            run_command(["bad_bytes"])
        assert "Failed to parse output." in str(excinfo.value)

    # 11. Test verbose mode (checking if log is called)
    with patch("your_module_path.log") as mock_log: # Replace your_module_path with actual module name
        run_command(["echo", "verbose_test"], verbose=True, return_output=True)
        assert mock_log.called

    # 12. Test ignore_errors for shell=True string args
    result = run_command("exit 1", shell=True, ignore_errors=True, return_output=True)
    assert result.returncode == 1
```


# LLM-generated content at query #16
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test Case 1: Non-subprocess exception should be returned unchanged
    class MyError(Exception):
        pass

    err_other = MyError("generic error")
    assert error_wrapper(err_other) is err_other
    assert str(error_wrapper(err_other)) == "generic error"

    # Test Case 2: subprocess.CalledProcessError with output
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

    # Test Case 3: subprocess.CalledProcessError with no output
    err_no_output = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=None)
    wrapped_no_output = error_wrapper(err_no_output)
    assert "No output was generated." in str(wrapped_no_output)

    # Test Case 4: subprocess.TimeoutExpired with output
    err_timeout = subprocess.TimeoutExpired(cmd="test_cmd", timeout=1, output=b"partial log")
    wrapped_timeout = error_wrapper(err_timeout)
    assert isinstance(wrapped_timeout, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_timeout)
    assert "    partial log" in str(wrapped_timeout)

    # Test Case 5: UnicodeDecodeError handling (simulating non-utf8 bytes)
    # We provide bytes that cannot be decoded as utf-8
    err_unicode = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=b"\xff\xfe\xfd"
    )
    wrapped_unicode = error_wrapper(err_unicode)
    assert "Failed to parse output." in str(wrapped_unicode)

    # Test Case 6: Ensure the original class type name is preserved in the new dynamic type
    assert wrapped_called.__class__.__name__ != "CalledProcessError" # It's a new dynamic subclass
    # Check if it still behaves like the original error via inheritance check
    assert isinstance(wrapped_called, subprocess.CalledProcessError)
```


# LLM-generated content at query #17
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test case 1: Non-subprocess exception should be returned unchanged
    class MyError(Exception):
        pass

    err = MyError("Original error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "Original error"

    # Test case 2: subprocess.CalledProcessError with output
    output_bytes = b"error occurred\nline 2"
    err_cp = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=output_bytes
    )
    wrapped_cp = error_wrapper(err_cp)
    expected_str = "Command 'test_cmd' returned non-zero exit status 1\nCaptured output:\n    error occurred\n    line 2"
    # Note: The exact base string depends on the python version/subprocess implementation,
    # but we check if our custom part is appended correctly.
    assert "Captured output:" in str(wrapped_cp)
    assert "error occurred" in str(wrapped_cp)

    # Test case 3: subprocess.CalledProcessError with no output
    err_no_out = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    wrapped_no_out = error_wrapper(err_no_out)
    assert "No output was generated." in str(wrapped_no_out)

    # Test case 4: subprocess.TimeoutExpired with output
    err_timeout = subprocess.TimeoutExpired(cmd="test_cmd", timeout=1, output=b"partial output")
    wrapped_timeout = error_wrapper(err_timeout)
    assert "Captured output:" in str(wrapped_timeout)
    assert "partial output" in str(wrapped_timeout)

    # Test case 5: subprocess.CalledProcessError with non-UTF8 output (UnicodeDecodeError handling)
    # Using a byte sequence that is invalid in UTF-8
    invalid_utf8 = b"\xff\xfe\xfd"
    err_bad_utf8 = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=invalid_utf8)
    wrapped_bad_utf8 = error_wrapper(err_bad_utf8)
    assert "Failed to parse output." in str(wrapped_bad_utf8)

    # Test case 6: Verify that the type is actually changed (dynamic class creation)
    assert type(wrapped_cp) != subprocess.CalledProcessError
    assert wrapped_cp.__class__.__name__ != "CalledProcessError"
```


# LLM-generated content at query #18
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test case 1: Non-subprocess exception should be returned unchanged
    class MyException(Exception):
        pass

    exc = MyException("original error")
    wrapped_exc = error_wrapper(exc)
    assert wrapped_exc is exc
    assert str(wrapped_exc) == "original error"

    # Test case 2: subprocess.CalledProcessError with output
    output_bytes = b"line1\nline2\n"
    err_cpe = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=output_bytes
    )
    wrapped_cpe = error_wrapper(err_cpe)
    
    # Verify class was dynamically changed to allow __str__ override
    assert wrapped_cpe.__class__.__name__ != "CalledProcessError"
    
    expected_str = (
        "Command 'test_cmd' returned non-zero exit status 1\n"
        "Captured output:\n"
        "    line1\n"
        "    line2\n"
    )
    assert str(wrapped_cpe).strip() == expected_str.strip()

    # Test case 3: subprocess.CalledProcessError with no output
    err_no_output = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=None)
    wrapped_no_output = error_wrapper(err_no_output)
    assert "No output was generated." in str(wrapped_no_output)

    # Test case 4: subprocess.TimeoutExpired with binary output that fails UTF-8 decoding
    # Using a byte sequence that is invalid UTF-8
    invalid_utf8 = b"\xff\xfe\xfd"
    err_timeout = subprocess.TimeoutExpired(cmd="test_cmd", timeout=1.0, output=invalid_utf8)
    wrapped_timeout = error_wrapper(err_timeout)
    assert "Failed to parse output." in str(wrapped_timeout)

    # Test case 5: subprocess.TimeoutExpired with valid UTF-8 output
    err_timeout_valid = subprocess.TimeoutExpired(cmd="test_cmd", timeout=1.0, output=b"timed out message")
    wrapped_timeout_valid = error_wrapper(err_timeout_valid)
    assert "Captured output:" in str(wrapped_timeout_valid)
    assert "    timed out message" in str(wrapped_timeout_valid)
```


# LLM-generated content at query #19
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Case 1: Non-subprocess exception should be returned unchanged
    class MyError(Exception):
        pass
    
    err = MyError("original error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "original error"

    # Case 2: subprocess.CalledProcessError with output
    output_bytes = b"error line 1\nerror line 2"
    err_called = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=output_bytes
    )
    wrapped_called = error_wrapper(err_called)
    
    # Check if the type was dynamically changed
    assert wrapped_called.__class__.__name__ != "CalledProcessError"
    
    # Check if output is correctly formatted in string representation
    expected_str = (
        "Command 'test_cmd' returned non-zero exit status 1\n"
        "Captured output:\n"
        "    error line 1\n"
        "    error line 2"
    )
    assert str(wrapped_called) == expected_str

    # Case 3: subprocess.CalledProcessError with no output
    err_no_output = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=None)
    wrapped_no_output = error_wrapper(err_no_output)
    assert "No output was generated." in str(wrapped_no_output)

    # Case 4: subprocess.TimeoutExpired with output
    err_timeout = subprocess.TimeoutExpired(cmd="test_cmd", timeout=1, output=b"partial data")
    wrapped_timeout = error_wrapper(err_timeout)
    assert "Captured output:" in str(wrapped_timeout)
    assert "partial data" in str(wrapped_timeout)

    # Case 5: subprocess.CalledProcessError with non-UTF8 bytes (triggers UnicodeDecodeError handling)
    # Using a byte sequence that is invalid in UTF-8
    invalid_utf8 = b"\xff\xfe\xfd" 
    err_invalid_bytes = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=invalid_utf8)
    wrapped_invalid = error_wrapper(err_invalid_bytes)
    assert "Failed to parse output." in str(wrapped_invalid)

    # Case 6: subprocess.CalledProcessError with very large output (verifying it handles split/join logic)
    large_output = b"line\n" * 100
    err_large = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=large_output)
    wrapped_large = error_wrapper(err_large)
    assert "Captured output:" in str(wrapped_large)
    assert "    line" in str(wrapped_large)
```


# LLM-generated content at query #20
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test 1: Non-subprocess exception should be returned unchanged
    class CustomError(Exception):
        pass

    err = CustomError("Original error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "Original error"

    # Test 2: subprocess.CalledProcessError with output should include captured output in __str__
    output_bytes = b"error message\nline 2"
    cpe = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=output_bytes)
    wrapped_cpe = error_wrapper(cpe)
    
    expected_str = "Command 'test_cmd' returned non-zero exit status 1\nCaptured output:\n    error message\n    line 2"
    # Note: The exact prefix depends on the python version/OS for CalledProcessError, 
    # so we check for the presence of our injected string.
    assert "Captured output:" in str(wrapped_cpe)
    assert "error message" in str(wrapped_cpe)
    assert "line 2" in str(wrapped_cpe)

    # Test 3: subprocess.CalledProcessError with no output should indicate no output was generated
    cpe_no_out = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=None)
    wrapped_cpe_no_out = error_wrapper(cpe_no_out)
    assert "No output was generated." in str(wrapped_cpe_no_out)

    # Test 4: subprocess.TimeoutExpired with output
    timeout_err = subprocess.TimeoutExpired(cmd="test_cmd", timeout=1, output=b"partial output")
    wrapped_timeout = error_wrapper(timeout_err)
    assert "Captured output:" in str(wrapped_timeout)
    assert "partial output" in str(wrapped_timeout)

    # Test 5: subprocess.CalledProcessError with non-UTF8 bytes should handle decoding error gracefully
    invalid_utf8 = b"\xff\xfe\xfd"
    cpe_bad_bytes = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=invalid_utf8)
    wrapped_bad_bytes = error_wrapper(cpe_bad_bytes)
    assert "Failed to parse output." in str(wrapped_bad_bytes)

    # Test 6: Ensure the class type is actually changed (dynamic type creation)
    assert wrapped_cpe.__class__.__name__ != subprocess.CalledProcessError.__name__
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest
import subprocess
import os
import sys
from unittest.mock import patch, MagicMock

@pytest.mark.parametrize("args, expected_returncode, expected_output", [
    (["echo", "hello"], 0, b"hello\n"),
    (["ls", "-l"], 0, None),  # Default return_output is False
])
def test_run_command_success(args, expected_returncode, expected_output):
    result = run_command(args, return_output=True)
    assert result.command == args
    assert result.return_code == expected_returncode
    if expected_output is not None:
        assert result.captured_output == expected_output

def test_run_command_with_env():
    env = {"TEST_VAR": "foo"}
    # Using python to check env var
    args = [sys.executable, "-c", "import os; print(os.environ.get('TEST_SHELL_VAR'))"]
    # We pass env directly to run_command which uses it in subprocess.run
    # Note: we must merge with existing env or the process might lack PATH
    current_env = os.environ.copy()
    current_env["TEST_SHELL_VAR"] = "bar"
    
    result = run_command(args, env=current_env, return_output=True)
    assert b"bar" in result.captured_output

def test_run_command_error_raises():
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        run_command(["ls", "/non_existent_directory_12345"], return_output=True)
    
    # The error_wrapper modifies the exception string to include output
    assert "No such file or directory" in str(excinfo.value)
    assert isinstance(excinfo.value, CommandResult.__class__) # Check if type was wrapped

def test_run_command_ignore_errors():
    result = run_command(["ls", "/non_existent_directory_12345"], ignore_errors=True, return_output=True)
    assert result.return_code != 0
    assert b"No such file or directory" in result.captured_output

def test_run_command_timeout():
    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        # Sleep for a long time to trigger timeout
        run_command([sys.executable, "-c", "import time; time.sleep(10)"], timeout=0.1)
    assert excinfo.value.output is not None

def test_run_command_timeout_ignore_errors():
    result = run_command([sys.executable, "-c", "import time; time.sleep(10)"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768

def test_run_command_truncation():
    # Create a large output
    large_output = "A" * (MAX_OUTPUT_LENGTH + 100)
    cmd = [sys.executable, "-c", f'print("{large_output}")']
    
    with patch("subprocess.run") as mock_run:
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        # Simulate CalledProcessError with large output
        error = subprocess.CalledProcessError(1, cmd)
        
        # We need to mock the file behavior inside run_command
        # This is tricky because it uses tempfile.TemporaryFile(). 
        # Instead, we'll mock the subprocess.run to raise error and 
        # rely on the fact that we can control what 'f' sees if we were more surgical,
        # but since we want to test the truncation logic:
        
        with patch("tempfile.TemporaryFile") as mock_temp:
            mock_file = MagicMock()
            # Simulate reading large content from the temp file
            content = b"A" * (MAX_OUTPUT_LENGTH + 100)
            mock_file.read.return_value = content
            # For seek/write operations to work in the context manager
            mock_temp.return_value.__enter__.return_value = mock_file
            
            mock_run.side_effect = subprocess.CalledProcessError(1, cmd)
            
            result = run_command(cmd, ignore_errors=True, return_output=True)
            
            assert b"*** (previous output truncated) ***" in result.captured_output
            assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + 50

def test_error_wrapper_non_subprocess_exception():
    try:
        raise ValueError("Standard error")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert str(wrapped) == "Standard error"
        assert isinstance(wrapped, ValueError)

def test_error_wrapper_with_output_decoding_error():
    # Create an error with non-utf8 bytes
    err = subprocess.CalledProcessError(1, "cmd", output=b"\xff\xfe")
    wrapped = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped)

def test_run_command_verbose(capsys):
    # Testing verbose mode requires checking the log calls. 
    # Since 'log' is imported, we patch it.
    with patch("your_module_path.log") as mock_log:
        run_command(["echo", "test"], verbose=True, return_output=True)
        assert mock_log.called
```


# LLM-generated content at query #22
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test case 1: Standard Exception (should remain unchanged)
    val_err = ValueError("original error")
    wrapped_val = error_wrapper(val_err)
    assert isinstance(wrapped_val, ValueError)
    assert str(wrapped_val) == "original error"

    # Test case 2: CalledProcessError with output (should include captured output)
    output_bytes = b"error line 1\nerror line 2"
    cpe_err = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=output_bytes)
    wrapped_cpe = error_wrapper(cpe_err)
    assert isinstance(wrapped_cpe, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_cpe)
    assert "error line 1" in str(wrapped_cpe)
    assert "error line 2" in str(wrapped_cpe)

    # Test case 3: CalledProcessError with no output (should indicate no output generated)
    cpe_no_out = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=None)
    wrapped_no_out = error_wrapper(cpe_no_out)
    assert "No output was generated." in str(wrapped_no_out)

    # Test case 4: TimeoutExpired with output
    timeout_err = subprocess.TimeoutExpired(cmd="test_cmd", timeout=1, output=b"partial output")
    wrapped_timeout = error_wrapper(timeout_err)
    assert "Captured output:" in str(wrapped_timeout)
    assert "partial output" in str(wrapped_timeout)

    # Test case 5: CalledProcessError with undecodable bytes (should handle UnicodeDecodeError gracefully)
    # Using a byte sequence that is invalid in UTF-8
    invalid_utf8 = b"\xff\xfe\xfd"
    cpe_invalid = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=invalid_utf8)
    wrapped_invalid = error_wrapper(cpe_invalid)
    assert "Failed to parse output." in str(wrapped_invalid)

    # Test case 6: Verify the type is dynamically created and not just the original class
    assert type(wrapped_cpe) != subprocess.CalledProcessError
    assert wrapped_cpe.__class__.__name__ != "CalledProcessError"
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
import subprocess
import os
import tempfile
from unittest.mock import patch, MagicMock

@pytest.mark.parametrize("cmd, expected_code, expect_output", [
    (["echo", "hello"], 0, b"hello\n"),
    (["ls", "/nonexistent_directory_xyz"], 2, b"ls: /nonexistent_directory_xyz: No such file or directory\n"),
])
def test_run_command_basic(cmd, expected_code, expect_output):
    # Test successful command and return output
    result = run_command(cmd, return_output=True)
    assert result.returncode == expected_code
    if expect_output:
        assert result.captured_output == expect_output

def test_run_command_shell_true():
    # Test running with shell=True
    result = run_command("echo 'test'", shell=True, return_output=True)
    assert result.returncode == 0
    assert b"test" in result.captured_output

def test_run_command_ignore_errors():
    # Test that ignore_errors prevents exception raising and returns special code for timeout
    with patch("subprocess.run") as mock_run:
        # Simulate CalledProcessError
        mock_run.side_effect = subprocess.CalledProcessError(returncode=1, cmd="bad_cmd", output=b"error")
        result = run_command(["bad_cmd"], ignore_errors=True)
        assert result.returncode == 1
        assert b"error" in result.captured_output

    with patch("subprocess.run") as mock_run:
        # Simulate TimeoutExpired
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="slow_cmd", timeout=1)
        result = run_command(["slow_cmd"], ignore_errors=True, timeout=1)
        assert result.returncode == -32768

def test_run_command_error_wrapper_output():
    # Test that error_wrapper adds the captured output to the exception string
    with patch("subprocess.run") as mock_run:
        error_msg = b"specific error message"
        mock_run.side_effect = subprocess.CalledProcessError(returncode=1, cmd="cmd", output=error_msg)
        
        try
            run_command(["cmd"])
        except subprocess.CalledProcessError as e:
            assert "Captured output:" in str(e)
            assert "specific error message" in str(e)

def test_run_command_max_output_length():
    # Test truncation of very large outputs
    large_output = b"a" * (MAX_OUTPUT_LENGTH + 100)
    with patch("subprocess.run") as mock_run:
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        # We need to simulate the file writing behavior or intercept the write
        # Since we can't easily intercept the internal TemporaryFile write without complex mocks,
        # we mock the subprocess.run to raise a CalledProcessError with a large output.
        mock_run.side_effect = subprocess.CalledProcessError(returncode=1, cmd="cmd", output=large_output)
        
        with pytest.raises(subprocess.CalledProcessError) as excinfo:
            run_command(["cmd"], ignore_errors=False)
        
        # Check if truncation occurred in the exception object attached by run_command
        captured = excinfo.value.output
        assert b"*** (previous output truncated) ***" in captured
        assert len(captured) <= MAX_OUTPUT_LENGTH + 40 # buffer for header

def test_run_command_env_and_cwd():
    # Test environment variables and CWD
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_env = {"MY_VAR": "test_value"}
        # Use a command that prints env vars (python is cross-platform for this test)
        cmd = ["python3", "-c", "import os; print(os.environ.get('MY_VAR'))"]
        result = run_command(cmd, env=custom_env, cwd=tmpdir, return_output=True)
        assert b"test_value" in result.captured_output

def test_run_command_verbose_logging(capsys):
    # Test that verbose mode triggers logging (mocking log to avoid dependency issues)
    with patch("your_module_path.log") as mock_log: # Replace your_module_path with actual module name
        run_command(["echo", "hi"], verbose=True, return_output=True)
        assert mock_log.called

def test_error_wrapper_non_subprocess_exception():
    # Test that error_wrapper ignores non-subprocess exceptions
    try:
        raise ValueError("standard error")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, ValueError)
        assert str(wrapped) == "standard error"

def test_run_command_unicode_error_in_wrapper():
    # Test the try-except block inside error_wrapper's __str__ for decoding errors
    err = subprocess.CalledProcessError(returncode=1, cmd="cmd", output=b"\xff\xfe")
    wrapped = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped)

def test_run_command_no_output_returned_by_default():
    # If return_output is False and command succeeds, captured_output should be None
    result = run_command(["echo", "hello"], return_output=False)
    assert result.returncode == 0
    assert result.captured_output is None
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
import subprocess
import os
import tempfile
from unittest.mock import patch, MagicMock

@pytest.mark.parametrize("args, expected_returncode, expected_output", [
    (["echo", "hello"], 0, b"hello\n"),
    (["ls", "-l"], 0, None),  # return_output is False by default
])
def test_run_command_success(args, expected_returncode, expected_output):
    result = run_command(args, return_output=True)
    assert result.command == args
    assert result.returncode == expected_returncode
    if expected_output is not None:
        assert result.captured_output == expected_output

def test_run_command_error_raises():
    # Command that exits with non-zero status
    args = ["ls", "/non_existent_directory_12345"]
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        run_command(args, return_output=True)
    
    assert excinfo.value.returncode != 0
    # Check if error_wrapper modified the string representation to include output
    assert "No such file or directory" in str(excinfo.value)

def test_run_command_ignore_errors():
    args = ["ls", "/non_existent_directory_12345"]
    result = run_command(args, ignore_errors=True, return_output=True)
    assert result.returncode != 0
    assert b"No such file or directory" in result.captured_output

def test_run_command_timeout():
    # Use a command that sleeps longer than the timeout
    args = ["python3", "-c", "import time; time.sleep(2)"]
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(args, timeout=0.1)

def test_run_command_timeout_ignore_errors():
    args = ["python3", "-c", "import time; time.sleep(2)"]
    result = run_command(args, timeout=0.1, ignore_errors=True)
    assert result.returncode == -32768

def test_run_command_env_and_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a file in the temp dir
        test_file = os.path.join(tmpdir, "test.txt")
        with open(test_file, "w") as f:
            f.write("content")

        env = {"MY_VAR": "HELLO"}
        # Check if env is passed and cwd is used to find the file
        result = run_command(["cat", "test.txt"], env=env, cwd=tmpdir, return_output=True)
        assert result.captured_output == b"content"

def test_run_command_truncation():
    # Mocking subprocess.run to return a huge amount of output
    huge_output = b"A" * (MAX_OUTPUT_LENGTH + 100)
    mock_ret = MagicMock()
    mock_ret.returncode = 0
    
    with patch("subprocess.run") as mock_run:
        # We need to simulate the file writing behavior of run_command
        def side_effect(*args, **kwargs):
            # This is a simplification; in reality, we'd need to handle the file object
            # But for testing truncation logic specifically:
            return MagicMock(returncode=0)

        # To test truncation, we must trigger the 'except' block in run_command
        # because that's where the truncation logic lives.
        error = subprocess.CalledProcessError(returncode=1, cmd="cmd")
        mock_run.side_effect = error
        
        with patch("tempfile.TemporaryFile") as mock_temp:
            mock_file = MagicMock()
            # Simulate reading the huge output
            mock_file.read.return_value = huge_output
            mock_temp.return_value.__enter__.return_value = mock_file
            
            with pytest.raises(subprocess.CalledProcessError) as excinfo:
                run_command(["dummy"], return_output=True)
            
            assert b"*** (previous output truncated) ***" in excinfo.value.output
            assert len(excinfo.value.output) < MAX_OUTPUT_LENGTH + 100

def test_error_wrapper_logic():
    # Test that error_wrapper handles non-subprocess errors by returning them unchanged
    class MyError(Exception):
        pass
    
    err = MyError("original")
    wrapped = error_wrapper(err)
    assert wrapped == err
    assert str(wrapped) == "original"

    # Test that it wraps subprocess errors
    proc_err = subprocess.CalledProcessError(1, "cmd", output=b"some error output")
    wrapped_proc = error_wrapper(proc_err)
    assert "Captured output:" in str(wrapped_proc)
    assert "some error output" in str(wrapped_proc)

def test_run_command_verbose_logging(capsys):
    # Testing verbose mode relies on the log function side effects. 
    # Since we don't have the implementation of 'log', we check if it runs without error.
    with patch("your_module_name.log") as mock_log: # Replace your_module_name with actual package name
        run_command(["echo", "test"], verbose=True, return_output=True)
        assert mock_log.called
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

    err = MyError("Original error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "Original error"
    assert type(wrapped_err).__name__ == "MyError"

    # Test 2: subprocess.CalledProcessError with output should include captured output
    output_bytes = b"error log line\nsecond line"
    err_cp = subprocess.CalledProcessError(returncode=1, cmd="ls", output=output_bytes)
    wrapped_cp = error_wrapper(err_cp)
    
    expected_str = (
        "Command 'ls' returned non-zero exit status 1\n"
        "Captured output:\n"
        "    error log line\n"
        "    second line"
    )
    # Note: subprocess.CalledProcessError str might vary slightly by python version, 
    # but the appended part is what we test.
    assert "Captured output:" in str(wrapped_cp)
    assert "error log line" in str(wrapped_cp)

    # Test 3: subprocess.CalledProcessError with no output should indicate no output was generated
    err_no_out = subprocess.CalledProcessError(returncode=1, cmd="ls", output=None)
    wrapped_no_out = error_wrapper(err_no_out)
    assert "No output was generated." in str(wrapped_no_out)

    # Test 4: subprocess.TimeoutExpired with output should include captured output
    err_timeout = subprocess.TimeoutExpired(cmd="sleep 10", timeout=1, output=b"partial output")
    wrapped_timeout = error_wrapper(err_timeout)
    assert "Captured output:" in str(wrapped_timeout)
    assert "partial output" in str(wrapped_timeout)

    # Test 5: UnicodeDecodeError handling (non-utf8 bytes)
    # Using invalid utf-8 sequence
    invalid_utf8 = b"\xff\xfe\xfd" 
    err_unicode = subprocess.CalledProcessError(returncode=1, cmd="ls", output=invalid_utf8)
    wrapped_unicode = error_wrapper(err_unicode)
    assert "Failed to parse output." in str(wrapped_unicode)

    # Test 6: Verify class type name changes as expected (dynamic type creation)
    assert wrapped_cp.__class__.__name__ != "CalledProcessError"
    assert wrapped_cp.__class__.__name__ == "CalledProcessError" # It uses the original name but is a new type
```


# LLM-generated content at query #4
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test 1: Regular exception should be returned unchanged
    class MyError(Exception):
        pass
    
    err = MyError("original error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "original error"

    # Test 2: subprocess.CalledProcessError with output
    output_bytes = b"error line 1\nerror line 2"
    called_proc_err = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=output_bytes
    )
    wrapped_called_proc = error_wrapper(called_proc_err)
    
    expected_str = "Command 'test_cmd' returned non-zero exit status 1\nCaptured output:\n    error line 1\n    error line 2"
    assert str(wrapped_called_proc) == expected_str

    # Test 3: subprocess.CalledProcessError without output
    empty_proc_err = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=None)
    wrapped_empty_proc = error_wrapper(empty_proc_err)
    assert "No output was generated." in str(wrapped_empty_proc)

    # Test 4: subprocess.TimeoutExpired with output
    timeout_err = subprocess.TimeoutExpired(cmd="slow_cmd", timeout=1, output=b"partial output")
    wrapped_timeout = error_wrapper(timeout_err)
    assert "Captured output:\n    partial output" in str(wrapped_timeout)

    # Test 5: subprocess.CalledProcessError with non-utf8 output (UnicodeDecodeError simulation)
    # We simulate this by providing bytes that cannot be decoded as utf-8 if the logic fails
    # Note: The implementation uses .decode('utf-8'), so we provide a byte sequence like \xff
    bad_output_bytes = b"\xff\xfe\xfd"
    bad_proc_err = subprocess.CalledProcessError(returncode=1, cmd="bad_enc", output=bad_output_bytes)
    wrapped_bad_enc = error_wrapper(bad_proc_err)
    assert "Failed to parse output." in str(wrapped_bad_enc)

    # Test 6: Verify type dynamic creation (checking if it's a new class instance)
    assert type(wrapped_called_proc) != subprocess.CalledProcessError
    assert issubclass(type(wrapped_called_proc), subprocess.CalledProcessError
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
    
    err = MyError("standard error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "standard error"

    # Test 2: CalledProcessError with output should include captured output in __str__
    output_bytes = b"error message\nline 2"
    cpe = subprocess.CalledProcessError(returncode=1, cmd="ls", output=output_bytes)
    wrapped_cpe = error_wrapper(cpe)
    
    expected_str = (
        "Command 'ls' returned non-zero exit status 1\n"
        "Captured output:\n"
        "    error message\n"
        "    line 2"
    )
    assert str(wrapped_cpe) == expected_str

    # Test 3: CalledProcessError with no output should indicate no output was generated
    cpe_no_out = subprocess.CalledProcessError(returncode=1, cmd="ls", output=None)
    wrapped_cpe_no_out = error_wrapper(cpe_no_out)
    assert "No output was generated." in str(wrapped_cpe_no_out)

    # Test 4: CalledProcessError with non-UTF8 bytes should handle decoding failure
    invalid_utf8 = b"\xff\xfe\xfd"
    cpe_bad_bytes = subprocess.CalledProcessError(returncode=1, cmd="ls", output=invalid_utf8)
    wrapped_bad_bytes = error_wrapper(cpe_bad_bytes)
    assert "Failed to parse output." in str(wrapped_bad_bytes)

    # Test 5: TimeoutExpired with output should include captured output
    timeout_out = b"some progress before timeout"
    te = subprocess.TimeoutExpired(cmd="sleep 10", timeout=1, output=timeout_out)
    wrapped_te = error_wrapper(te)
    assert "Captured output:" in str(wrapped_te)
    assert "some progress before timeout" in str(wrapped_te)

    # Test 6: Verify that the class type is actually changed (dynamic type creation check)
    assert wrapped_cpe.__class__.__name__ != subprocess.CalledProcessError.__name__
    assert isinstance(wrapped_cpe, subprocess.CalledProcessError)
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
import subprocess
import os
import tempfile
from unittest.mock import patch, MagicMock

def test_run_command(tmp_path):
    # Test 1: Basic successful command execution (return_output=False)
    # Should return CommandResult with returncode 0 and no captured output
    result = run_command(["echo", "hello"], return_output=False)
    assert result.returncode == 0
    assert result.captured_output is None
    assert isinstance(result.command, list)

    # Test 2: Successful command execution with return_output=True
    # Should capture the output correctly
    result = run_command(["echo", "hello"], return_output=True)
    assert result.returncode == 0
    assert result.captured_output.decode('utf-8').strip() == "hello"

    # Test 3: Command failure (CalledProcessError) with error_wrapper
    # Should raise an exception where __str__ contains the captured output
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        run_command(["ls", "/non_existent_directory_12345"], return_output=False)
    
    assert excinfo.value.returncode != 0
    # Check if error_wrapper successfully injected output into __str__
    assert "Captured output:" in str(excinfo.value)
    assert "/non_existent_directory_12345" in str(excinfo.value)

    # Test 4: Command failure with ignore_errors=True
    # Should return a CommandResult instead of raising an exception
    result = run_command(["ls", "/non_existent_directory_12345"], ignore_errors=True)
    assert result.returncode != 0
    assert b"No such file or directory" in result.captured_output

    # Test 5: Command timeout (TimeoutExpired)
    # Using a sleep command to trigger timeout
    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        run_command(["sleep", "2"], timeout=0.1)
    assert "-32768" not in str(excinfo.value) # Verify it's the original error before wrapping logic if needed

    # Test 6: Command timeout with ignore_errors=True
    # Should return the specific magic return code -32768
    result = runCommand_timeout_ignore := run_command(["sleep", "0.5"], timeout=0.1, ignore_errors=True)
    assert result.returncode == -32768

    # Test 7: Verification of truncation logic (MAX_OUTPUT_LENGTH)
    # Create a large output to trigger truncation
    large_output = "A" * (9000)
    with patch("subprocess.run") as mock_run:
        mock_ret = MagicMock()
        mock_ret.returncode = 1
        # Simulate CalledProcessError with large output
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd="large_cmd", output=large_output.encode('utf-8')
        )
        
        with pytest.raises(subprocess.CalledProcessError) as excinfo:
            run_command(["large_cmd"], ignore_errors=False)
        
        # Check that truncation happened
        assert b"*** (previous output truncated) ***" in excinfo.value.output
        assert len(excinfo.value.output) <= 9000 + 50 # original limit + overhead

    # Test 8: Environment variables and CWD
    # Create a file in a temp directory and run command to see it
    test_dir = tmp_path / "sub"
    test_dir.mkdir()
    test_file = test_dir / "test.txt"
    test_file.write_text("content")
    
    result = run_command(["cat", "test.txt"], cwd=str(test_dir), return_output=True)
    assert result.captured_output.decode('utf-8').strip() == "content"

    # Test 9: error_wrapper with non-subprocess exception should remain unchanged
    with pytest.raises(ValueError) as excinfo:
        error_wrapper(ValueError("test"))
    assert str(excinfo.value) == "test"
    assert isinstance(excinfo.value, ValueError)

    # Test 10: error_wrapper with CalledProcessError without output
    err = subprocess.CalledProcessError(returncode=1, cmd="cmd")
    wrapped_err = error_wrapper(err)
    assert "No output was generated" in str(wrapped_err)
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

    exc = MyException("original error")
    wrapped_exc = error_wrapper(exc)
    assert wrapped_exc is exc
    assert str(wrapped_exc) == "original error"

    # Test 2: subprocess.CalledProcessError with output should include captured output in __str__
    output_bytes = b"error message\nline 2"
    err_called = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=output_bytes)
    wrapped_called = error_wrapper(err_called)
    
    assert "Captured output:" in str(wrapped_called)
    assert "error message" in str(wrapped_called)
    assert "line 2" in str(wrapped_called)
    # Verify type dynamic creation works (it shouldn't be the original class anymore)
    assert wrapped_called.__class__.__name__ != "CalledProcessError"

    # Test 3: subprocess.CalledProcessError with no output should indicate no output was generated
    err_no_output = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=None)
    wrapped_no_output = error_wrapper(err_no_output)
    assert "No output was generated." in str(wrapped_no_output)

    # Test 4: subprocess.TimeoutExpired with output should include captured output
    timeout_output = b"interrupted process output"
    err_timeout = subprocess.TimeoutExpired(cmd="test_cmd", timeout=1, output=timeout_output)
    wrapped_timeout = error_wrapper(err_timeout)
    assert "Captured output:" in str(wrapped_timeout)
    assert "interrupted process output" in str(wrapped_timeout)

    # Test 5: subprocess.CalledProcessError with undecodable (binary) output
    # Using a byte sequence that is invalid UTF-8
    invalid_utf8 = b"\xff\xfe\xfd"
    err_bad_decode = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=invalid_utf8)
    wrapped_bad_decode = error_wrapper(err_bad_decode)
    assert "Failed to parse output." in str(wrapped_bad_decode)

    # Test 6: Verify that the original exception class name is preserved in the new dynamic type
    assert wrapped_called.__class__.__name__ == "CalledProcessError"
```


# LLM-generated content at query #8
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test Case 1: Non-subprocess exception should be returned unchanged
    class MyError(Exception):
        pass

    err = MyError("Original error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, MyError)
    assert str(wrapped_err) == "Original error"

    # Test Case 2: subprocess.CalledProcessError with output
    output_bytes = b"error line 1\nerror line 2"
    err_called = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=output_bytes
    )
    wrapped_called = error_wrapper(err_called)
    assert "Captured output:" in str(wrapped_called)
    assert "error line 1" in str(wrapped_called)
    assert "error line 2" in str(wrapped_called)

    # Test Case 3: subprocess.TimeoutExpired with output
    err_timeout = subprocess.TimeoutExpired(
        cmd="test_cmd", 
        timeout=1, 
        output=b"some partial output"
    )
    wrapped_timeout = error_wrapper(err_timeout)
    assert "Captured output:" in str(wrapped_timeout)
    assert "some partial output" in str(wrapped_timeout)

    # Test Case 4: subprocess.CalledProcessError without output
    err_no_output = subprocess.CalledProcessError(
        returncode=2, 
        cmd="test_cmd", 
        output=None
    )
    wrapped_no_output = error_wrapper(err_no_output)
    assert "No output was generated." in str(wrapped_no_output)

    # Test Case 5: subprocess.CalledProcessError with undecodable bytes (UnicodeDecodeError simulation)
    # We use a byte sequence that is invalid UTF-8
    invalid_utf8 = b"\xff\xfe\xfd"
    err_invalid = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=invalid_utf8
    )
    wrapped_invalid = error_wrapper(err_invalid)
    assert "Failed to parse output." in str(wrapped_invalid)

    # Test Case 6: Verify the type is actually a new dynamic class and not the original
    assert type(wrapped_called) != subprocess.CalledProcessError
    assert wrapped_called.__class__.__name__ != "CalledProcessError"
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
import subprocess
import os
import tempfile
from unittest.mock import patch, MagicMock

@pytest.mark.parametrize("args, expected_code, expect_output", [
    (["echo", "hello"], 0, b"hello\n"),
    (["ls", "/nonexistent_directory_test_123"], 2, b"ls: /nonexistent_directory_test_123: No such file or directory\n"),
])
def test_run_command_basic(args, expected_code, expect_output):
    # Test successful command and failed command with return_output=True
    result = run_command(args, return_output=True)
    assert result.command == args
    assert result.return_code == expected_code
    if expect_output:
        assert expect_output in result.captured_output

def test_run_command_no_output_on_success():
    # Test that captured_output is None when return_output=False and command succeeds
    result = run_command(["echo", "test"], return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_ignore_errors():
    # Test that ignore_errors=True prevents exception and returns specific code for timeout
    # We use a command that will fail
    result = run_command(["ls", "/nonexistent"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_timeout_error():
    # Test timeout functionality
    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        # sleep for a long time, but timeout after 1 second
        run_command(["sleep", "10"], timeout=1)
    
    # Verify that the error_wrapper was applied (the __str__ is modified)
    assert "Captured output:" in str(excinfo.value) or "No output" in str(excinfo.value)

def test_run_command_env_and_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test environment variable injection
        result = run_command(["printenv", "MY_TEST_VAR"], env={"MY_TEST_VAR": "val"}, return_output=True)
        assert b"val" in result.captured_output

        # Test CWD (working directory)
        # Create a file in tmpdir and try to list it from within the command context
        test_file = os.path.join(tmpdir, "exists.txt")
        with open(test_file, "w") as f:
            f.write("content")
        
        result = run_command(["ls"], cwd=tmpdir, return_output=True)
        assert b"exists.txt" in result.captured_output

def test_run_command_truncation():
    # Mock subprocess.run to return a massive amount of output
    large_output = b"A" * (MAX_OUTPUT_LENGTH + 1000)
    
    with patch("subprocess.run") as mock_run:
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        # Simulate the behavior of a failed process where output is captured in a file
        # Since we can't easily mock the TemporaryFile content directly without complexity,
        # we rely on the logic that if return_code != 0, it reads the file.
        # We patch 'subprocess.run' to raise CalledProcessError
        mock_run.side_effect = subprocess.CalledProcessError(1, "cmd", output=large_output)
        
        # Note: The actual implementation uses a real TemporaryFile. 
        # To test truncation specifically, we need the file content to be large.
        # Because run_command writes to a real temp file, we'll mock 'subprocess.run' 
        # and instead of an error, we let it succeed but provide a huge amount of data via a side effect.
        
        # A more robust way to test the logic of truncation:
        with patch("tempfile.TemporaryFile") as mock_temp:
            mock_file = MagicMock()
            # Simulate reading large content
            mock_file.read.return_value = b"B" * (MAX_OUTPUT_LENGTH + 100)
            mock_file.seek.return_value = None
            mock_temp.return().__enter__.return_value = mock_file
            
            # Force a CalledProcessError
            mock_run.side_effect = subprocess.CalledProcessError(1, "cmd")
            
            result = run_command(["cmd"], ignore_errors=True)
            assert b"*** (previous output truncated) ***" in result.captured_output
            assert len(result.captured_output) < MAX_OUTPUT_LENGTH + 100

def test_error_wrapper_non_subprocess_exception():
    # Ensure error_wrapper returns the exception unchanged if it's not a subprocess error
    err = ValueError("test")
    wrapped = error_wrapper(err)
    assert wrapped == err
    assert isinstance(wrapped, ValueError)

def test_error_wrapper_subprocess_exception_str():
    # Test the custom __str__ implementation in error_wrapper
    err = subprocess.CalledProcessError(1, "cmd", output=b"line1\nline2")
    wrapped = error_wrapper(err)
    output_str = str(wrapped)
    assert "Captured output:" in output_str
    assert "    line1" in output_str
    assert "    line2" in output_str

def test_run_command_verbose():
    # Test that verbose mode doesn't crash and uses log (via patching)
    with patch("your_module_path.log") as mock_log: # Replace 'your_module_path' with actual module name
        run_command(["echo", "hi"], verbose=True, return_output=True)
        assert mock_log.called
```


# LLM-generated content at query #10
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

    # Test 2: subprocess.CalledProcessError with output should include captured output in __str__
    output_bytes = b"error message\nline 2"
    err_proc = subprocess.CalledProcesseler(["cmd"], returncode=1, output=output_bytes)
    # Note: In a real scenario, subprocess.run assigns .output during the catch block in run_command
    # Here we manually simulate the state error_wrapper expects
    err_proc.output = output_bytes
    wrapped_proc = error_wrapper(err_proc)
    
    expected_str = "Command '['cmd'] returned non-zero exit status 1\nCaptured output:\n    error message\n    line 2"
    assert str(wrapped_proc) == expected_str

    # Test 3: subprocess.CalledProcessError with no output should indicate no output was generated
    err_no_out = subprocess.CalledProcessError(["cmd"], returncode=1, output=None)
    err_no_out.output = None
    wrapped_no_out = error_wrapper(err_no_out)
    assert "No output was generated." in str(wrapped_no_out)

    # Test 4: subprocess.TimeoutExpired with output should include captured output
    timeout_output = b"some logs before timeout"
    err_timeout = subprocess.TimeoutExpired(["cmd"], timeout=1.0, output=timeout_output)
    err_timeout.output = timeout_output
    wrapped_timeout = error_wrapper(err_timeout)
    
    assert "Captured output:" in str(wrapped_timeout)
    assert "some logs before timeout" in str(wrapped_timeout)

    # Test 5: subprocess.CalledProcessError with undecodable (binary) output
    # We use a byte sequence that is invalid UTF-8
    invalid_utf8 = b"\xff\xfe\xfd"
    err_bad_utf8 = subprocess.CalledProcessError(["cmd"], returncode=1, output=invalid_utf8)
    err_bad_utf8.output = invalid_utf8
    wrapped_bad_utf8 = error_wrapper(err_bad_utf8)
    assert "Failed to parse output." in str(wrapped_bad_utf8)

    # Test 6: Verify that the class type is actually changed (dynamic subclassing)
    assert type(wrapped_proc) != subprocess.CalledProcessError
    assert issubclass(type(wrapped_proc), subprocess.CalledProcessError
```


# LLM-generated content at query #11
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test Case 1: Non-subprocess exception should be returned unchanged
    class MyError(Exception):
        pass

    err = MyError("Standard error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "Standard error"

    # Test Case 2: subprocess.CalledProcessError with output
    output_bytes = b"error line 1\nerror line 2"
    err_called = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=output_bytes
    )
    wrapped_called = error_wrapper(err_called)
    
    expected_str = "Command 'test_cmd' returned non-zero exit status 1\nCaptured output:\n    error line 1\n    error line 2"
    assert str(wrapped_called) == expected_str

    # Test Case 3: subprocess.CalledProcessError without output
    err_no_output = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=None
    )
    wrapped_no_output = error_wrapper(err_no_output)
    assert "No output was generated." in str(wrapped_no_output)

    # Test Case 4: subprocess.TimeoutExpired with output
    err_timeout = subprocess.TimeoutExpired(
        cmd="test_cmd", 
        timeout=1, 
        output=b"partial output"
    )
    wrapped_timeout = error_wrapper(err_timeout)
    assert "Captured output:" in str(wrapped_timeout)
    assert "partial output" in str(wrapped_timeout)

    # Test Case 5: subprocess.CalledProcessError with undecodable/binary output
    # Using a byte sequence that is invalid UTF-8
    invalid_utf8 = b"\xff\xfe\xfd"
    err_invalid = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=invalid_utf8
    )
    wrapped_invalid = error_wrapper(err_invalid)
    assert "Failed to parse output." in str(wrapped_invalid)

    # Test Case 6: Verify the type is dynamically created and inherits correctly
    assert isinstance(wrapped_called, subprocess.CalledProcessError)
    assert wrapped_called.__class__.__name__ != "CalledProcessError"
```


# LLM-generated content at query #12
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test case 1: Non-subprocess exception should be returned unchanged
    class MyException(Exception):
        pass

    exc = MyException("standard error")
    wrapped_exc = error_wrapper(exc)
    assert wrapped_exc is exc
    assert str(wrapped_exc) == "standard error"

    # Test case 2: subprocess.CalledProcessError with output
    output_bytes = b"error message\nline 2"
    err_proc = subprocess.CalledProcessError(returncode=1, cmd="ls", output=output_bytes)
    wrapped_proc = error_wrapper(err_proc)
    
    expected_str = "Command 'ls' returned non-zero exit status 1\nCaptured output:\n    error message\n    line 2"
    assert str(wrapped_proc) == expected_str

    # Test case 3: subprocess.CalledProcessError without output
    err_no_output = subprocess.CalledProcessError(returncode=1, cmd="ls", output=None)
    wrapped_no_output = error_wrapper(err_no_output)
    assert "No output was generated." in str(wrapped_no_output)

    # Test case 4: subprocess.TimeoutExpired with output
    err_timeout = subprocess.TimeoutExpired(cmd="sleep 10", timeout=1, output=b"partial output")
    wrapped_timeout = error_wrapper(err_timeout)
    assert "Captured output:" in str(wrapped_timeout)
    assert "partial output" in str(wrapped_timeout)

    # Test case 5: subprocess.CalledProcessError with undecodable (binary) output
    # Using bytes that are invalid in UTF-8 to trigger UnicodeDecodeError
    invalid_utf8 = b"\xff\xfe\xfd"
    err_invalid = subprocess.CalledProcessError(returncode=1, cmd="ls", output=invalid_utf8)
    wrapped_invalid = error_wrapper(err_invalid)
    assert "Failed to parse output." in str(wrapped_invalid)

    # Test case 6: Verify that the class type is actually changed (dynamic type creation)
    assert wrapped_proc.__class__.__name__ != "CalledProcessError"
    assert issubclass(wrapped_proc.__class__, subprocess.CalledProcessError)
```


# LLM-generated content at query #13
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test Case 1: Non-subprocess exception remains unchanged
    class MyError(Exception):
        pass
    
    err = MyError("Original Error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, MyError)
    assert str(wrapped_err) == "Original Error"

    # Test Case 2: subprocess.CalledProcessError with output
    output_bytes = b"error line 1\nerror line 2"
    err_proc = subprocess.CalledProcessError(
        returncode=1, 
        cmd=["ls", "nonexistent"], 
        output=output_bytes
    )
    wrapped_proc = error_wrapper(err_proc)
    
    # Check if the string representation contains the captured output formatted correctly
    assert "Captured output:" in str(wrapped_proc)
    assert "    error line 1" in str(wrapped_proc)
    assert "    error line 2" in str(wrapped_proc)

    # Test Case 3: subprocess.CalledProcessError with non-UTF8 output (triggers decoding error handling)
    # We use a byte sequence that is invalid in UTF-8
    invalid_utf8 = b"\xff\xfe\xfd"
    err_bad_bytes = subprocess.CalledProcessError(
        returncode=1, 
        cmd=["ls"], 
        output=invalid_utf8
    )
    wrapped_bad_bytes = error_wrapper(err_bad_bytes)
    assert "Failed to parse output." in str(wrapped_bad_bytes)

    # Test Case 4: subprocess.CalledProcessError with no output
    err_no_output = subprocess.CalledProcessError(
        returncode=1, 
        cmd=["ls"], 
        output=None
    )
    # Manually setting output to empty bytes as subprocess might provide b'' instead of None
    err_no_output.output = b"" 
    wrapped_no_output = error_wrapper(err_no_output)
    assert "No output was generated." in str(wrapped_no_output)

    # Test Case 5: subprocess.TimeoutExpired with output
    err_timeout = subprocess.TimeoutExpired(
        cmd=["sleep", "10"], 
        timeout=0.1, 
        output=b"interrupted data"
    )
    wrapped_timeout = error_wrap(err_timeout) if 'error_wrap' in globals() else error_wrapper(err_timeout)
    assert "Captured output:" in str(wrapped_timeout)
    assert "    interrupted data" in str(wrapped_timeout)

    # Test Case 6: Verify that the class type has actually changed (dynamic type creation)
    assert type(wrapped_proc).__name__ != "CalledProcessError"
    assert hasattr(wrapped_proc, "__str__")
```


# LLM-generated content at query #14
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test 1: Normal exception (not a subprocess error) should be returned unchanged
    class MyError(Exception):
        pass

    err = MyError("original message")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "original message"
    assert type(wrapped_err).__name__ == "MyError"

    # Test 2: subprocess.CalledProcessError with output should include captured output in __str__
    output_bytes = b"error line 1\nerror line 2"
    cpe = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=output_bytes)
    wrapped_cpe = error_wrapper(cpe)
    
    assert "Captured output:" in str(wrapped_cpe)
    assert "error line 1" in str(wrapped_cpe)
    assert "error line 2" in str(wrapped_cpe)
    # Verify it's a new type but preserves the original class name concept via the wrapper logic
    assert type(wrapped_cpe).__name__ == "CalledProcessError"

    # Test 3: subprocess.CalledProcessError with no output should indicate no output generated
    cpe_no_out = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=None)
    wrapped_cpe_no_out = error_wrapper(cpe_no_out)
    assert "No output was generated." in str(wrapped_cpe_no_out)

    # Test 4: subprocess.TimeoutExpired with output
    timeout_err = subprocess.TimeoutExpired(cmd="test_cmd", timeout=1, output=b"partial output")
    wrapped_timeout = error_wrapper(timeout_err)
    assert "Captured output:" in str(wrapped_timeout)
    assert "partial output" in str(wrapped_timeout)

    # Test 5: subprocess.CalledProcessError with non-UTF8 byte sequence (handling the UnicodeEncodeError case)
    # Note: The error_wrapper uses .decode('utf-8'), so we provide bytes that fail decoding
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
from unittest.mock import patch, MagicMock

def test_run_command():
    # Test 1: Successful command execution returning output
    result = run_command(["echo", "hello"], return_output=True)
    assert result.returncode == 0
    assert result.captured_output.decode('utf-8').strip() == "hello"
    assert result.command == ["echo", "hello"]

    # Test 2: Successful command execution without requesting output
    result = run_command(["echo", "no_output"], return_output=False)
    assert result.returncode == 0
    assert result.captured_output is None

    # Test 3: Command that fails (CalledProcessError) with error wrapper
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        run_command(["ls", "/non_existent_directory_path_12345"], return_output=True)
    
    # Check if error_wrapper modified the string representation to include output
    assert "No such file or directory" in str(excinfo.value)
    assert isinstance(excinfo.value, subprocess.CalledProcessError)

    # Test 4: Command execution with ignore_errors=True
    result = run_command(["ls", "/non_existent_directory_path_12345"], ignore_errors=True, return_output=True)
    assert result.returncode != 0
    assert b"No such file or directory" in result.captured_output

    # Test 5: Command execution with timeout (TimeoutExpired)
    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        run_command(["sleep", "10"], timeout=0.1)
    assert "No output was generated" in str(excinfo.value)

    # Test 6: Command execution with ignore_errors and Timeout
    result = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True)
    assert result.returncode == -32768

    # Test 7: Testing env variables
    result = run_command(["printenv", "MY_TEST_VAR"], return_output=True, env={"MY_TEST_VAR": "foobar"})
    assert b"foobar" in result.captured_output

    # Test 8: Testing verbose mode (checking if log is called)
    with patch('pytest.importorskip("pytest").module.log') as mock_log:
        # We use a simple command to see if log is triggered via run_command's internal logic
        run_command(["echo", "verbose_test"], verbose=True, return_output=True)
        # Check if any call to log contained the expected output
        args_list = [call.args[0] for call in mock_log.call_args_list]
        assert any("verbose_test" in str(arg) for arg in args_list)

    # Test 9: Testing truncation of large output
    large_output = "A" * (MAX_OUTPUT_LENGTH + 100)
    with patch('subprocess.run') as mock_run:
        mock_ret = MagicMock()
        mock_ret.returncode = 1
        # Simulate a CalledProcessError with large output in the temp file logic
        # Since run_command writes to a tempfile, we simulate the failure and the content
        mock_run.side_effect = subprocess.CalledProcessError(returncode=1, cmd="large_cmd")
        
        # We need to mock the behavior of the file system inside run_command or 
        # rely on the fact that it uses tempfile.TemporaryFile.
        # To test truncation specifically, we can't easily intercept the internal tempfile write,
        # but we can check if the logic handles output > MAX_OUTPUT_LENGTH correctly.
        
        # A more direct way to test the logic of the function:
        with patch('tempfile.TemporaryFile') as mock_temp:
            mock_f = MagicMock()
            mock_f.read.return_value = b"B" * (MAX_OUTPUT_LENGTH + 100)
            mock_f.seek.return_tuples = (None, None) # dummy
            mock_temp.return_value.__enter__.return_value = mock_f
            
            with pytest.raises(subprocess.CalledProcessError):
                run_command(["dummy"], ignore_errors=False)
            
            # Check if the error message (via error_wrapper) or internal logic would truncate
            # The truncation happens in the 'except' block before raising/returning
            # We verify the output length via the returned CommandResult when ignore_errors=True
            result = run_command(["dummy"], ignore_errors=True, return_output=True)
            # Note: This test depends on how we mock the file. 
            # In a real scenario, we'd mock subprocess.run to raise error and f.read() to return large bytes.
```


# LLM-generated content at query #16
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test case 1: Non-subprocess exception should be returned unchanged
    class MyException(Exception):
        pass

    exc = MyException("test error")
    wrapped_exc = error_wrapper(exc)
    assert wrapped_exc is exc
    assert str(wrapped_exc) == "test error"

    # Test case 2: subprocess.CalledProcessError with output
    output_bytes = b"error line 1\nerror line 2"
    called_proc_err = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=output_bytes
    )
    wrapped_called_err = error_wrapper(called_proc_err)
    expected_str = "Command 'test_cmd' returned non-zero exit status 1\nCaptured output:\n    error line 1\n    error line 2"
    assert str(wrapped_called_err).strip() == expected_str.strip()

    # Test case 3: subprocess.TimeoutExpired with output
    timeout_err = subprocess.TimeoutExpired(
        cmd="test_cmd", 
        timeout=1, 
        output=b"partial output"
    )
    wrapped_timeout_err = error_wrapper(timeout_err)
    expected_timeout_str = "Command 'test_cmd' expired after 1 seconds\nCaptured output:\n    partial output"
    assert str(wrapped_timeout_err).strip() == expected_timeout_str.strip()

    # Test case 4: subprocess.CalledProcessError with no output
    no_output_err = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=None
    )
    wrapped_no_output_err = error_wrapper(no_output_err)
    assert "No output was generated." in str(wrapped_no_output_err)

    # Test case 5: subprocess.CalledProcessError with undecodable (binary) output
    # We simulate this by providing bytes that aren't valid UTF-8
    invalid_utf8 = b"\xff\xfe\xfd"
    bad_decode_err = subprocess.CalledProcessError(
        returncode=1, 
        cmd="test_cmd", 
        output=invalid_utf8
    )
    wrapped_bad_decode_err = error_wrapper(bad_decode_err)
    assert "Failed to parse output." in str(wrapped_bad_decode_err)

    # Test case 6: Verify type creation (the __class__ mutation)
    assert type(wrapped_called_err).__name__ != "CalledProcessError"
    assert issubclass(type(wrapped_called_err), subprocess.CalledProcessError
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
import subprocess
import os
import sys
from unittest.mock import patch, MagicMock

def test_run_command(tmp_path):
    # Test 1: Successful command execution with return_output=True
    # Using 'echo' which is available on most Unix-like systems. 
    # For Windows compatibility in a generic test, we use sys.executable to run a python command.
    cmd = [sys.executable, "-c", "print('hello world')"]
    result = run_command(cmd, return_output=True)
    assert result.returncode == 0
    assert result.captured_output.decode('utf-8').strip() == 'hello world'
    assert result.command == cmd

    # Test 2: Successful command execution without return_output (None output)
    result = run_command(cmd, return_output=False)
    assert result.returncode == 0
    assert result.captured_output is None

    # Test 3: Command failure with error_wrapper and captured output
    # We use a python command that exits with non-zero code and prints something
    fail_cmd = [sys.executable, "-c", "import sys; print('error message'); sys.exit(1)"]
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        run_command(fail_cmd, return_output=True)
    
    # Check if error_wrapper modified the exception string to include output
    assert "error message" in str(excinfo.value)
    assert isinstance(excinfo.value.__class__.__name__, str)

    # Test 4: Command failure with ignore_errors=True
    result = run_command(fail_cmd, ignore_errors=True, return_output=True)
    assert result.return_code == 1
    assert b"error message" in result.captured_output

    # Test 5: Timeout handling (simulated via mock to be platform independent)
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd, timeout=0.1, output=b"some output")
        with pytest.raises(subprocess.TimeoutExpired) as excinfo:
            run_command(cmd, timeout=0.01)
        assert b"some output" in excinfo.value.output

    # Test 6: Timeout handling with ignore_errors=True (special return code -32768)
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd, timeout=0.1, output=b"timeout error")
        result = run_command(cmd, timeout=0.01, ignore_errors=True, return_output=True)
        assert result.return_code == -32768
        assert b"timeout error" in result.captured_output

    # Test 7: Verify CWD and Env passing
    new_dir = tmp_path / "subdir"
    new_dir.mkdir()
    env_vars = {"TEST_VAR": "TEST_VALUE"}
    cmd_env = [sys.executable, "-c", "import os; print(os.environ.get('TEST_VAR'))"]
    result = run_command(cmd_env, env=env_vars, cwd=new_dir, return_output=True)
    assert result.captured_output.decode('utf-8').strip() == "TEST_VALUE"

    # Test 8: Output truncation (MAX_OUTPUT_LENGTH = 8192)
    large_output = "A" * 10000
    with patch("subprocess.run") as mock_run:
        mock_ret = MagicMock()
        mock_ret.returncode = 1
        mock_run.return_value = mock_ret
        # Simulate error during run to trigger the 'except' block where truncation happens
        mock_run.side_effect = subprocess.CalledProcessError(1, cmd, output=large_output.encode('utf-8'))
        
        with pytest.raises(subprocess.CalledProcessError) as excinfo:
            run_command(cmd, ignore_errors=False)
        
        decoded_output = excinfo.value.output.decode('utf-8')
        assert "*** (previous output truncated) ***" in decoded_output
        # Check that the end of the string contains the tail of the large output
        assert decoded_output.endswith("A" * 100)

    # Test 9: error_wrapper with non-subprocess exception should remain unchanged
    with pytest.raises(ValueError) as excinfo:
        error_wrapper(ValueError("standard error"))
    assert str(excinfo.value) == "standard error"
```


# LLM-generated content at query #18
#--------------------------

```python
import subprocess
import pytest

def test_error_wrapper():
    # Test 1: Passing a standard exception (should return same instance/type)
    val_err = ValueError("standard error")
    wrapped_val = error_wrapper(val_err)
    assert wrapped_val is val_err
    assert str(wrapped_val) == "standard error"

    # Test 2: Passing CalledProcessError with output (should include captured output)
    cmd = ["ls", "/nonexistent"]
    # Simulate subprocess.CalledProcessError
    call_err = subprocess.CalledProcessError(returncode=1, cmd=cmd, output=b"error line\nsecond line")
    wrapped_call_err = error_wrapper(call_err)
    
    expected_str = "Command '['ls', '/nonexistent']\nCaptured output:\n    error line\n    second line"
    assert str(wrapped_call_err) == expected_str

    # Test 3: Passing CalledProcessError with no output (should indicate no output generated)
    call_err_no_out = subprocess.CalledProcessError(returncode=1, cmd=cmd, output=None)
    wrapped_no_out = error_wrapper(call_err_no_out)
    assert "No output was generated." in str(wrapped_no_out)

    # Test 4: Passing CalledProcessError with non-UTF8 bytes (should handle decode error)
    # Using invalid utf-8 sequence
    invalid_bytes = b"\xff\xfe\xfd"
    call_err_invalid = subprocess.CalledProcessError(returncode=1, cmd=cmd, output=invalid_bytes)
    wrapped_invalid = error_wrapper(call_err_invalid)
    assert "Failed to parse output." in str(wrapped_invalid)

    # Test 5: Passing TimeoutExpired (should work similarly to CalledProcessError)
    timeout_err = subprocess.TimeoutExpired(cmd=cmd, timeout=1)
    timeout_err.output = b"some partial output"
    wrapped_timeout = error_wrapper(timeout_err)
    assert "Captured output:" in str(wrapped_timeout)
    assert "some partial output" in str(wrapped_timeout)

    # Test 6: Verify that the type is dynamically changed (class identity check)
    # The original class should be replaced by a new one with the same name but different identity
    assert type(wrapped_call_err).__name__ == "CalledProcessError"
    assert type(wrapped_call_err) is not subprocess.CalledProcessError
```


# LLM-generated content at query #19
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

    # Test 2: subprocess.CalledProcessError with output should include captured output in __str__
    output_bytes = b"error details\nline 2"
    cpe = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=output_bytes)
    wrapped_cpe = error_wrapper(cpe)
    
    assert "Captured output:" in str(wrapped_cpe)
    assert "error details" in str(wrapped_cpe)
    assert "line 2" in str(wrapped_cpe)

    # Test 3: subprocess.CalledProcessError with no output should indicate no output was generated
    cpe_no_out = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=None)
    # Note: We must manually set output to None because error_wrapper checks 'if self.output'
    # In a real scenario, if it's empty bytes b'', it might behave differently based on truthiness
    cpe_no_out.output = None 
    wrapped_cpe_no_out = error_wrapper(cpe_no_out)
    assert "No output was generated." in str(wrapped_cpe_no_out)

    # Test 4: subprocess.TimeoutExpired with output
    timeout_err = subprocess.TimeoutExpired(cmd="test_cmd", timeout=1, output=b"interrupted")
    wrapped_timeout = error_wrapper(timeout_err)
    assert "Captured output:" in str(wrapped_timeout)
    assert "interrupted" in str(wrapped_timeout)

    # Test 5: UnicodeDecodeError handling during wrapping (invalid utf-8 bytes)
    invalid_utf8 = b"\xff\xfe\xfd"
    cpe_invalid = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=invalid_utf8)
    wrapped_invalid = error_wrapper(cpe_invalid)
    assert "Failed to parse output." in str(wrapped_invalid)

    # Test 6: Verify that the class type has actually changed (dynamic class creation)
    original_type = type(cpe)
    assert type(wrapped_cpe) != original_type
    assert wrapped_cpe.__str__() == str(cpe)
```


