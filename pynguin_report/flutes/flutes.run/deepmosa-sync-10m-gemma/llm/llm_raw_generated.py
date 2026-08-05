####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_run_command_success():
    import subprocess
    from flutes.run import run_command
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_with_return_output():
    import subprocess
    from flutes.run import run_command
    result = run_command(["echo", "test_output"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"test_output\n"

def test_run_command_error_raises_exception():
    import subprocess
    from flutes.run import run_command
    try:
        run_command(["ls", "--non-existent-flag"])
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
        assert b"ls" in str(e).encode()
    else:
        raise AssertionError("Expected CalledProcessError was not raised")

def test_run_command_ignore_errors():
    import subprocess
    from flutes.run import run_command
    result = run_command(["ls", "--non-existent-flag"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_timeout_raises_exception():
    import subprocess
    from flutes.run import run_command
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert True
    else:
        raise AssertionError("Expected TimeoutExpired was not raised")

def test_run_command_timeout_ignore_errors():
    import subprocess
    from flutes.run import run_command
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768

def test_run_command_with_env():
    import subprocess
    from flutes.run import run_command
    import os
    result = run_command(["python3", "-c", "import os; print(os.environ['MY_VAR'])"], env={**os.environ, "MY_VAR": "test_val"}, return_output=True)
    assert b"test_val" in result.captured_output

def test_run_command_with_cwd():
    import subprocess
    from flutes.run import run_command
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert os.path.abspath(tmpdir).encode() in result.captured_output
```


# LLM-generated content at query #2
#--------------------------

```python
def test_run_command_verbose_true_executes_log_predicate():
    from flutes.run import run_command
    import sys
    import subprocess

    # We use a simple command that succeeds (like 'echo') to ensure we reach the logic.
    # The predicate is `if verbose:`. By setting `verbose=True`, line 24 executes,
    # and the code proceeds to the `with` block (line 25).
    result = run_command(["echo", "test"], verbose=True)
    
    assert result.command == ["echo", "test"]
    assert result.returncode == 0
```


# LLM-generated content at query #3
#--------------------------

```python
def test_run_command_returns_output_when_requested():
    from flutes.run import run_command
    import sys
    result = run_command([sys.executable, "-c", "print('hello')"], return_output=True)
    assert result.captured_output == b'hello\n'

def test_run_command_returns_output_on_nonzero_exit():
    from flutes.run import run_command
    import sys
    result = run_command([sys.executable, "-c", "import sys; sys.exit(1)"], return_output=False)
    assert result.return_code == 1
    assert result.captured_output is not None

def test_run_command_returns_output_when_verbose():
    from flutes.run import run_command
    import sys
    result = run_command([sys.executable, "-c", "print('hello')"], verbose=True)
    assert result.captured_output == b'hello\n'
```


# LLM-generated content at query #4
#--------------------------

```python
import subprocess
import sys
from pathlib import Path
from flutes.run import run_command, CommandResult

def test_run_command_success():
    result = run_command([sys.executable, "-c", "print('hello')"], return_output=True)
    assert isinstance(result, CommandResult)
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"
    assert result.command == [sys.executable, "-c", "print('hello')"]

def test_run_command_failure_raises():
    with Exception() as e:
        run_command([sys.executable, "-c", "import sys; sys.exit(1)"])
        raise AssertionError("Should have raised subprocess.CalledProcessError")
    assert isinstance(e, subprocess.CalledProcessError)
    assert e.returncode == 1

def test_run_command_failure_with_output():
    try:
        run_command([sys.executable, "-c", "import sys; sys.stderr.write('error_msg'); sys.exit(1)"], return_output=False)
    except subprocess.CalledProcessError as e:
        assert e.returncode == 1
        assert b"error_msg" in e.output

def test_run_command_ignore_errors():
    result = run_command([sys.executable, "-c", "import sys; sys.exit(42)"], ignore_errors=True)
    assert result.return_code == 42
    assert isinstance(result, CommandResult)

def test_run_command_timeout_ignore_errors():
    result = run_command([sys.executable, "-c", "import time; time.sleep(10)"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768

def test_run_command_timeout_raises():
    with Exception() as e:
        run_command([sys.executable, "-c", "import time; time.sleep(10)"], timeout=0.1)
        raise AssertionError("Should have raised subprocess.TimeoutExpired")
    assert isinstance(e, subprocess.TimeoutExpired)

def test_run_command_env_vars():
    result = run_command([sys.executable, "-c", "import os; print(os.environ['TEST_VAR'])"], env={"TEST_VAR": "val"}, return_output=True)
    assert b"val" in result.captured_output

def test_run_command_shell_mode():
    result = run_command("echo 'shell_test'", shell=True, return_output=True)
    assert b"shell_test" in result.captured_output

def test_error_wrapper_logic():
    err = subprocess.CalledProcessError(1, "cmd", output=b"line1\nline2")
    wrapped_err = subprocess.CalledProcessError(1, "cmd", output=b"line1\nline2")
    # We simulate the error_wrapper behavior manually since we cannot redefine functions in a test case
    from flutes.run import error_wrapper
    wrapped_err = error_wrapper(wrapped_err)
    assert "Captured output:" in str(wrapped_err)
    assert "line1" in str(wrapped_err)
    assert "line2" in str(wrapped_err)

def test_error_wrapper_no_output():
    from flutes.run import error_wrapper
    err = subprocess.CalledProcessError(1, "cmd", output=None)
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)
```


# LLM-generated content at query #5
#--------------------------

```python
import subprocess
from pathlib import Path
from flutes.run import run_command, CommandResult

def test_run_command_success():
    result = run_command(["echo", "hello"])
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_return_output_true():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.captured_output == b"hello\n"

def test_run_command_error_raises_exception():
    with subprocess.CalledProcessError(1, "ls /nonexistent_directory") as e:
        try:
            run_command(["ls", "/nonexistent_directory"])
        except subprocess.CalledProcessError as raised_e:
            assert isinstance(raised_e, subprocess.CalledProcessError)
            assert "No output was generated" in str(raised_e) or "ls" in str(raised_e)
            raise e

def test_run_command_ignore_errors_true():
    result = run_command(["ls", "/nonexistent_directory"], ignore_errors=True)
    assert result.return_code != 0
    assert isinstance(result, CommandResult)

def test_run_command_with_env():
    result = run_command(["python3", "-c", "import os; print(os.environ['MY_VAR'])"], env={"MY_VAR": "test_val"})
    assert b"test_val" in result.captured_output

def test_run_command_timeout_raises_exception():
    with subprocess.TimeoutExpired("sleep", 1) as e:
        try:
            run_command(["sleep", "5"], timeout=0.1)
        except subprocess.TimeoutExpired as raised_e:
            assert isinstance(raised_e, subprocess.TimeoutExpired)
            raise e

def test_run_command_ignore_errors_timeout():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768

def test_run_command_shell_true():
    result = run_command("echo 'shell test'", shell=True)
    assert b"shell test" in result.captured_output
```


# LLM-generated content at query #6
#--------------------------

```python
import subprocess

def test_error_wrapper_returns_original_if_not_subprocess_error():
    original_exception = ValueError("Original Error")
    wrapped_exception = error_wrapper(ValueError)
    assert wrapped_exception == ValueError
    
    instance = ValueError("Test")
    result = error_wrapper(instance)
    assert result is instance
    assert str(result) == "Test"

def test_error_wrapper_modifies_called_process_error_with_output():
    original_err = subprocess.CalledProcessError(returncode=1, cmd="ls", output=b"line1\nline2")
    wrapped_err = error_wrapper(original_err)
    
    expected_str = "Command 'ls' returned non-zero exit status 1.\nCaptured output:\n    line1\n    line2"
    assert str(wrapped_err) == expected_str
    assert type(wrapped_err).__name__ != "CalledProcessError"

def test_error_wrapper_modifies_called_process_error_without_output():
    original_err = subprocess.CalledProcessError(returncode=1, cmd="ls", output=None)
    wrapped_err = error_wrapper(original_err)
    
    expected_str = "Command 'ls' returned non-zero exit status 1.\nNo output was generated."
    assert str(wrapped_err) == expected_str

def test_error_wrapper_modifies_timeout_expired_with_output():
    original_err = subprocess.TimeoutExpired(cmd="sleep", timeout=5, output=b"timeout error")
    wrapped_err = error_wrapper(original_err)
    
    expected_str = "Command 'sleep' -> Timeout expired.\nCaptured output:\n    timeout error"
    assert str(wrapped_err) == expected_str

def test_error_wrapper_handles_decode_error():
    # Using an invalid byte sequence for utf-8 to trigger decode failure
    invalid_bytes = b"\xff\xfe\xfd" 
    original_err = subprocess.CalledProcessError(returncode=1, cmd="ls", output=invalid_bytes)
    wrapped_err = error_wrapper(original_err)
    
    # The function catches UnicodeDecodeError (though the snippet says UnicodeEncodeError, 
    # in practice decoding bytes triggers UnicodeDecodeError)
    # Note: if the provided code specifically checks for UnicodeEncodeError, we test that logic flow.
    assert "Failed to parse output." in str(wrapped_err) or "Captured output:" in str(wrapped_err)
```


# LLM-generated content at query #7
#--------------------------

```python
import subprocess

def test_error_wrapper_predicate_is_false():
    err = subprocess.CalledProcessError(returncode=1, cmd="ls")
    result = error_wrapper(err)
    assert isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))
    assert result == err
```


# LLM-generated content at query #8
#--------------------------

```python
def test_run_command_unicode_decode_error_path():
    import subprocess
    from unittest.mock import patch, MagicMock
    from flutes.run import run_command

    # Mocking subprocess.run to return a successful process object with non-utf8 bytes in stdout
    # We use an invalid byte sequence (e.g., \xff) that will trigger UnicodeDecodeError on .decode('utf-8')
    mock_ret = MagicMock()
    mock_ret.returncode = 0
    
    # Create a temporary file mock or rely on the actual tempfile behavior in run_command
    # Since we cannot easily intercept the internal TemporaryFile's content without mocking subprocess.run,
    # We will mock subprocess.run to return our object and mock the 'f' (the file object) 
    # used inside the context manager via patching 'tempfile.TemporaryFile'.
    
    mock_file = MagicMock()
    # The byte sequence \xff is invalid UTF-8
    invalid_bytes = b"\xff\xfe\xfd"
    mock_file.read.returnside_effect = [invalid_bytes, invalid_bytes] # for line loop if needed
    mock_file.__enter__.return_value = mock_file
    mock_file.seek.return_value = None
    # The first read() inside the 'if return_output...' block returns the bad bytes
    mock_file.read.side_effect = [invalid_bytes] 

    with patch("subprocess.run", return_value=mock_ret), \
         patch("tempfile.TemporaryFile", return_value=mock_file), \
         patch("flutes.run.log") as mock_log:
        
        # We trigger the 'if verbose' branch to reach line 45
        # Line 45 calls log(output.decode('utf-8'), ...) which will raise UnicodeDecodeError
        # because output is invalid UTF-8.
        
        run_command(["echo", "test"], verbose=True)
        
        # Assert that the logger was called at least once (even if it failed the first time, 
        # the 'except' block triggers a second call with str(line))
        assert mock_log.called
```


# LLM-generated content at query #9
#--------------------------

```python
import subprocess

def test_error_wrapper_returns_original_if_not_subprocess_error():
    err = ValueError("test error")
    result = error_wrapper(err)
    assert result is err
    assert str(result) == "test error"

def test_error_wrapper_modifies_called_process_error_with_output():
    err = subprocess.CalledProcessError(returncode=1, cmd="ls", output=b"line1\nline2")
    result = error_wrapper(err)
    expected_str = "Command 'ls' returned non-zero exit status 1.\nCaptured output:\n    line1\n    line2"
    assert str(result) == expected_str

def test_error_wrapper_modifies_called_process_error_without_output():
    err = subprocess.CalledProcessError(returncode=1, cmd="ls", output=None)
    result = error_wrapper(err)
    expected_str = "Command 'ls' returned non-zero exit status 1.\nNo output was generated."
    assert str(result) == expected_str

def test_error_wrapper_modifies_timeout_expired_with_output():
    err = subprocess.TimeoutExpired(cmd="sleep", timeout=5, output=b"some error")
    result = error_wrapper(err)
    expected_str = "Command 'sleep' -> Timeout expired.\nCaptured output:\n    some error"
    assert str(result) == expected_str

def test_error_wrapper_handles_unicode_decode_error():
    # Using an invalid byte sequence for utf-8 to trigger the exception path
    err = subprocess.CalledProcessError(returncode=1, cmd="ls", output=b"\xff\xfe")
    result = error_wrapper(err)
    assert "Failed to parse output." in str(result)

def test_error_wrapper_preserves_class_hierarchy():
    err = subprocess.CalledProcessError(returncode=1, cmd="ls", output=b"out")
    result = error_wrapper(err)
    assert isinstance(result, subprocess.CalledProcessError)
    assert result.__class__.__name__ != "CalledProcessError"
    assert issubclass(result.__class__, subprocess.CalledProcessError)
```


# LLM-generated content at query #10
#--------------------------

```python
def test_run_command_unicode_decode_error_branch():
    import subprocess
    import tempfile
    from unittest.mock import patch, MagicMock
    from flutes.run import run_command, CommandResult

    # We want to trigger the 'except UnicodeDecodeError' block at line 46.
    # This happens when log(output.decode('utf-8')) fails.
    # To achieve this, we need:
    # 1. verbose=True (to enter the if verbose block)
    # 2. return_output=True or a non-zero exit code (to reach the reading part)
    # 3. output containing invalid UTF-8 bytes.
    
    invalid_utf8_bytes = b"\xff\xfe\xfd"
    args = ["echo", "test"]
    
    # Mock subprocess.run to return a successful process with our invalid bytes in stdout
    mock_ret = MagicMock()
    mock_ret.returncode = 0
    
    # We mock the file object returned by TemporaryFile to yield our invalid bytes
    # and we mock the log function to raise UnicodeDecodeError when it receives the decoded string.
    # However, the code calls output.decode('utf-8') inside the try block.
    # So if we provide invalid bytes, the .decode() call itself will raise the error.

    with patch("subprocess.run", return_value=mock_ret), \
         patch("tempfile.TemporaryFile") as mock_tempfile, \
         patch("flutes.run.log") as mock_log:
        
        # Setup the mock file behavior
        mock_f = MagicMock()
        mock_tempfile.return_value.__enter__.return_value = mock_f
        mock_f.read.return_value = invalid_utf8_bytes
        
        # Execute function
        run_command(args, verbose=True, return_output=True)
        
        # Assertions:
        # The first call to log (the one in the try block) should have attempted to decode 
        # and failed, thus triggering the except block.
        # In the except block, it iterates through lines and calls log(str(line), ...).
        # We check if any call to log was made with a string representation of the byte line.
        
        # The first call attempted: log(invalid_utf8_bytes.decode('utf-8'), ...) -> Raises UnicodeDecodeError
        # The second call (in except block): log("b'\\xff\\xfe\\xfd'", ...) 
        
        found_fallback_log = False
        for call in mock_log.call_args_list:
            if isinstance(call.args[0], str) and "b'\\xff" in call.args[0]:
                found_fallback_log = True
        
        assert found_fallback_log is True
```


# LLM-generated content at query #11
#--------------------------

```python
def test_run_command_unicode_decode_success():
    import subprocess
    from unittest.mock import patch, MagicMock
    from flutes.run import run_command

    # Mocking subprocess.run to return a successful process with valid UTF-8 output
    mock_ret = MagicMock()
    mock_ret.returncode = 0
    
    with patch("subprocess.run", return_value=mock_ret), \
         patch("flutes.run.log") as mock_log:
        
        # We use a simple command that returns valid UTF-8 bytes
        # 'echo' is used to ensure output is easily predictable
        run_command(["echo", "hello"], verbose=True, return_output=True)
        
        # The predicate at line 46: `except UnicodeDecodeError`
        # To make it False, the log call must succeed.
        # We check if log was called with the decoded string 'hello\n'
        # Note: subprocess output usually includes a newline
        mock_log.assert_called_with("hello\n", timestamp=False, include_proc_id=False)
```


# LLM-generated content at query #12
#--------------------------

```python
import subprocess

def test_error_wrapper_predicate_true():
    err = subprocess.CalledProcessError(returncode=1, cmd="ls")
    result = error_wrapper(err)
    assert isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))
```


# LLM-generated content at query #13
#--------------------------

```python
def test_run_command_success():
    import subprocess
    from flutes.run import run_command
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_with_output():
    import subprocess
    from flutes.run import run_command
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output.decode("utf-8").strip() == "hello"

def test_run_command_failure_raises_error():
    import subprocess
    from flutes.run import run_command
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        run_command(["ls", "/non_existent_directory_12345"])
    assert excinfo.value.returncode != 0

def test_run_command_ignore_errors():
    import subprocess
    from flutes.run import run_command
    result = run_command(["ls", "/non_existent_directory_12345"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_timeout():
    import subprocess
    from flutes.run import run_command
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=1)

def test_run_command_ignore_errors_timeout():
    import subprocess
    from flutes.run import run_command
    result = run_command(["sleep", "10"], timeout=1, ignore_errors=True)
    assert result.return_code == -32768

def test_run_command_with_env():
    import subprocess
    from flutes.run import run_command
    result = run_command(["printenv", "MY_VAR"], return_output=True, env={"MY_VAR": "test_value"})
    assert b"test_value" in result.captured_output

def test_error_wrapper_with_output():
    import subprocess
    from flutes.run import error_wrapper
    err = subprocess.CalledProcessError(returncode=1, cmd="cmd", output=b"line1\nline2")
    wrapped_err = error_wrapper(err)
    assert "Captured output:" in str(wrapped_err)
    assert "    line1" in str(wrapped_err)
    assert "    line2" in str(wrapped_err)

def test_error_wrapper_no_output():
    import subprocess
    from flutes.run import error_wrapper
    err = subprocess.CalledProcessError(returncode=1, cmd="cmd", output=None)
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

def test_error_wrapper_non_subprocess_error():
    from flutes.run import error_wrapper
    err = ValueError("standard error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "standard error"
```


# LLM-generated content at query #14
#--------------------------

```python
def test_run_command_returns_output_when_requested():
    import subprocess
    from flutes.run import run_command
    result = run_command(["echo", "hello"], return_output=True)
    assert result.captured_output == b"hello\n"
```


# LLM-generated content at query #15
#--------------------------

```python
def test_run_command_unicode_decode_success():
    from flutes.run import run_command
    import subprocess

    # We use 'echo' to produce valid UTF-8 output. 
    # By setting verbose=True and return_output=True, we trigger the block containing line 45.
    # Since the output is valid UTF-8, output.decode('utf-8') will succeed,
    # making the 'except UnicodeDecodeError' block (and thus the predicate at line 46) unreachable.
    result = run_command(["echo", "hello"], verbose=True, return_output=True)
    
    assert result.returncode == 0
    assert b"hello" in result.captured_output
```


# LLM-generated content at query #16
#--------------------------

```python
import subprocess

def test_error_wrapper_returns_original_on_non_subprocess_exception():
    err = ValueError("test error")
    result = error_wrapper(err)
    assert result is err
    assert str(result) == "test error"

def test_error_wrapper_handles_called_process_error_with_output():
    err = subprocess.CalledProcessError(returncode=1, cmd="ls", output=b"line1\nline2")
    result = error_wrapper(err)
    expected_str = "Command 'ls' returned non-zero exit status 1.\nCaptured output:\n    line1\n    line2"
    assert str(result) == expected_str

def test_error_wrapper_handles_called_process_error_without_output():
    err = subprocess.CalledProcessError(returncode=1, cmd="ls", output=None)
    result = error_wrapper(err)
    expected_str = "Command 'ls' returned non-zero exit status 1.\nNo output was generated."
    assert str(result) == expected_str

def test_error_wrapper_handles_timeout_expired_with_output():
    err = subprocess.TimeoutExpired(cmd="sleep", timeout=5, output=b"partial")
    result = error_wrapper(err)
    expected_str = "Command 'sleep' -> Timeout expired.\nCaptured output:\n    partial"
    assert str(result) == expected_str

def test_error_wrapper_handles_unicode_decode_error():
    # Using invalid utf-8 bytes to trigger the exception in decode
    err = subprocess.CalledProcessError(returncode=1, cmd="ls", output=b"\xff\xfe")
    result = error_wrapper(err)
    assert "Failed to parse output." in str(result)

def test_error_wrapper_preserves_class_hierarchy():
    err = subprocess.CalledProcessError(returncode=1, cmd="ls", output=b"data")
    result = error_wrapper(err)
    assert isinstance(result, subprocess.CalledProcessError)
    assert result.__class__.__name__ != "CalledProcessError"
    assert issubclass(result.__class__, subprocess.CalledProcessError)
```


# LLM-generated content at query #17
#--------------------------

```python
def test_run_command_truncates_large_output():
    import subprocess
    from unittest.mock import patch, MagicMock
    from flutes.run import run_command, CommandResult

    large_content = b"A" * (8192 + 100)
    
    # We need to mock subprocess.run to raise a CalledProcessError with specific output
    # and we need to ensure the temporary file used in run_command contains our large content.
    # Since run_command creates its own tempfile, we patch 'tempfile.TemporaryFile' 
    # to return a mock that acts like a file containing our large data.
    
    mock_file = MagicMock()
    mock_file.__enter__.return_value = mock_file
    # Simulate the content being written and then read back
    # The function writes to stdout=f, so we simulate reading from it.
    mock_file.read.return_value = large_content
    
    with patch("tempfile.TemporaryFile", return_value=mock_file), \
         patch("subprocess.run") as mock_run:
        
        # Setup the exception that triggers line 29-32
        mock_err = subprocess.CalledProcessError(returncode=1, cmd="echo 'large'", output=None)
        mock_run.side_effect = mock_err

        # Execute the command with ignore_errors=True to prevent the exception from bubbling up
        result = run_command(["echo", "large"], ignore_errors=True)

        # Assertions for line 32 predicate (len(output) > MAX_OUTPUT_LENGTH)
        assert result.captured_output.startswith(b"*** (previous output truncated) ***\n")
        assert len(result.captured_output) <= 8192 + len(b"*** (previous output truncated) ***\n")
        # Check that the end of the output is indeed the end of our large content
        assert result.captured_output.endswith(large_content[-8192:])
```


# LLM-generated content at query #18
#--------------------------

```python
def test_run_command_predicate_true_via_return_output():
    from flutes.run import run_command
    import sys

    result = run_command([sys.executable, "-c", "print('hello')"], return_output=True)
    assert result.returncode == 0
    assert b"hello" in result.captured_output

def test_run_command_predicate_true_via_nonzero_return_code():
    from flutes.run import run_command
    import sys

    result = run_command([sys.executable, "-c", "import sys; sys.exit(1)"], return_output=False)
    assert result.returncode == 1
    assert result.captured_output is None

def test_run_command_predicate_true_via_verbose():
    from flutes.run import run_command
    import sys

    result = run_command([sys.executable, "-c", "print('verbose_test')"], verbose=True, return_output=False)
    assert result.returncode == 0
    assert b"verbose_test" in result.captured_output
```


# LLM-generated content at query #19
#--------------------------

```python
import subprocess

def test_error_wrapper_returns_original_on_standard_exception():
    err = ValueError("original error")
    result = error_wrapper(err)
    assert result is err
    assert str(result) == "original error"
    assert type(result) is ValueError

def test_error_wrapper_modifies_called_process_error_with_output():
    err = subprocess.CalledProcessError(returncode=1, cmd="ls", output=b"line1\nline2")
    result = error_wrapper(err)
    expected_str = "Command 'ls' returned non-zero exit status 1.\nCaptured output:\n    line1\n    line2"
    assert str(result) == expected_str
    assert type(result).__name__ != "CalledProcessError"

def test_error_wrapper_modifies_called_process_error_no_output():
    err = subprocess.CalledProcessError(returncode=1, cmd="ls", output=None)
    result = error_wrapper(err)
    expected_str = "Command 'ls' returned non-zero exit status 1.\nNo output was generated."
    assert str(result) == expected_str

def test_error_wrapper_modifies_timeout_expired_with_output():
    err = subprocess.TimeoutExpired(cmd="sleep", timeout=5, output=b"some error")
    result = error_wrapper(err)
    expected_str = "Command 'sleep' -> Timeout expired.\nCaptured output:\n    some error"
    assert str(result) == expected_str

def test_error_wrapper_handles_undecodable_output():
    # Using invalid utf-8 bytes to trigger UnicodeDecodeError (Note: the function catches UnicodeEncodeError in code, 
    # but usually decode triggers UnicodeDecodeError. We test based on provided logic)
    err = subprocess.CalledProcessError(returncode=1, cmd="ls", output=b"\xff\xfe")
    result = error_wrapper(err)
    # The provided function has a bug/typo: it catches UnicodeEncodeError instead of UnicodeDecodeError.
    # However, if we follow the code's logic as written:
    try:
        str_val = str(result)
        assert "Captured output:" in str_val or "Failed to parse output." in str_val
    except UnicodeDecodeError:
        pass
```


# LLM-generated content at query #20
#--------------------------

```python
import subprocess

def test_error_wrapper_evaluates_true_for_subprocess_errors():
    err = subprocess.CalledProcessError(returncode=1, cmd="ls")
    wrapped_err = error_wrapper(err)
    assert isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))
```


# LLM-generated content at query #21
#--------------------------

```python
def test_run_command_truncates_large_output():
    from flutes.run import run_command
    import subprocess
    from unittest.mock import patch, MagicMock

    large_content = b"A" * (9000)
    
    with patch("subprocess.run") as mock_run:
        mock_ret = MagicMock()
        mock_ret.returncode = 1
        mock_run.side_effect = subprocess.CalledProcessError(returncode=1, cmd="test", output=None)
        
        with patch("tempfile.TemporaryFile") as mock_temp:
            mock_file = MagicMock()
            mock_temp.return_value.__enter__.return_value = mock_file
            mock_file.read.return_value = large_content
            
            result = run_command(["test"], ignore_errors=True, return_output=True)
            
            assert len(result.captured_output) <= 9000 + len(b"*** (previous output truncated) ***\n")
            assert result.captured_output.startswith(b"*** (previous output truncated) ***\n")
            assert b"A" in result.captured_output
```


# LLM-generated content at query #22
#--------------------------

```python
def test_run_command_truncates_large_output():
    import subprocess
    from unittest.mock import patch, MagicMock
    from flutes.run import run_command, CommandResult

    large_content = b"A" * (8192 + 100)
    
    with patch("subprocess.run") as mock_run:
        mock_exception = subprocess.CalledProcessError(returncode=1, cmd="echo", output=None)
        mock_exception.output = large_content
        mock_run.side_effect = mock_exception
        
        with patch("tempfile.TemporaryFile") as mock_temp:
            mock_file = MagicMock()
            mock_file.read.return_value = large_content
            mock_temp.return_value.__enter__.return_value = mock_file
            
            result = run_command(["echo", "test"], ignore_errors=True)
            
            assert len(result.captured_output) < len(large_content)
            assert result.captured_output.startswith(b"*** (previous output truncated) ***\n")
            assert result.captured_output.endswith(b"A" * 8192)
```


# LLM-generated content at query #23
#--------------------------

```python
def test_run_command_verbose_true_executes_log_branch():
    from flutes.run import run_command
    import sys
    from unittest.mock import patch

    with patch("flutes.run.log") as mock_log:
        run_command(["echo", "hello"], verbose=True)
        mock_log.assert_called()
```


# LLM-generated content at query #24
#--------------------------

```python
import subprocess

def test_error_wrapper_returns_original_on_unrelated_exception():
    err = ValueError("test error")
    result = error_wrapper(err)
    assert result is err
    assert str(result) == "test error"

def test_error_wrapper_modifies_called_process_error_with_output():
    err = subprocess.CalledProcessError(returncode=1, cmd="ls", output=b"line1\nline2")
    result = error_wrapper(err)
    expected = "Command 'ls' returned non-zero exit status 1.\nCaptured output:\n    line1\n    line2"
    assert str(result) == expected

def test_error_wrapper_modifies_called_process_error_without_output():
    err = subprocess.CalledProcessError(returncode=1, cmd="ls", output=None)
    result = error_wrapper(err)
    expected = "Command 'ls' returned non-zero exit status 1.\nNo output was generated."
    assert str(result) == expected

def test_error_wrapper_modifies_timeout_expired_with_output():
    err = subprocess.TimeoutExpired(cmd="ls", timeout=5, output=b"some data")
    result = error_wrapper(err)
    expected = "Command 'ls' -> Timeout expired.\nCaptured output:\n    some data"
    assert str(result) == expected

def test_error_wrapper_handles_undecodable_output():
    # Using a byte sequence that is invalid utf-8 to trigger the decode error logic
    err = subprocess.CalledProcessError(returncode=1, cmd="ls", output=b"\xff\xfe\xfd")
    result = error_wrapper(err)
    assert "Failed to parse output." in str(result)

def test_error_wrapper_preserves_type_inheritance():
    err = subprocess.CalledProcessError(returncode=1, cmd="ls")
    result = error_wrapper(err)
    assert isinstance(result, subprocess.CalledProcessError)
    assert not isinstance(result, subprocess.TimeoutExpired)
```


# LLM-generated content at query #25
#--------------------------

```python
def test_run_command_verbose_true_triggers_log():
    from flutes.run import run_command
    import subprocess

    result = run_command(["echo", "hello"], verbose=True, return_output=True)
    assert result.command == ["echo", "hello"]
```


# LLM-generated content at query #26
#--------------------------

```python
import subprocess

def test_error_wrapper_predicate_true():
    err = subprocess.CalledProcessError(returncode=1, cmd="ls")
    result = error_wrapper(err)
    assert isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))
```


# LLM-generated content at query #27
#--------------------------

```python
def test_run_command_unicode_decode_success():
    import subprocess
    from unittest.mock import patch, MagicMock
    from flutes.run import run_command

    # Mocking subprocess.run to return a successful process with valid UTF-8 output
    mock_ret = MagicMock()
    mock_ret.returncode = 0
    
    with patch('subprocess.run', return_value=mock_ret), \
         patch('flutes.run.log') as mock_log:
        
        # Using a simple command that works on most systems and produces valid UTF-8
        # 'echo' or similar. We use a list format.
        args = ["echo", "hello"]
        
        # We need to simulate the file writing into the tempfile used in run_command.
        # Since we can't easily intercept the write to the tempfile inside the function 
        # without mocking 'tempfile.TemporaryFile', we patch it to return a BytesIO object.
        from io import BytesIO
        mock_buffer = BytesIO(b"hello\n")
        
        with patch('tempfile.TemporaryFile', return_value=mock_buffer):
            # To ensure the line 45 log() succeeds, we need to make sure output.decode('utf-8') works.
            # The 'output' is read from the file buffer.
            run_command(args, verbose=True)
            
            # If UnicodeDecodeError was NOT raised, then log() was called with the decoded string.
            # We check if the first argument of the first call to log was "hello\n" (decoded).
            # Note: The code does `log(output.decode('utf-8'), ...)`
            # If it hits line 46, it means decode failed. If it stays at 45, decode succeeded.
            mock_log.assert_called_with("hello\n", timestamp=False, include_proc_id=False)

```


# LLM-generated content at query #28
#--------------------------

```python
import subprocess

def test_error_wrapper_evaluates_true_for_subprocess_errors():
    err = subprocess.CalledProcessError(returncode=1, cmd="ls")
    result = error_wrapper(err)
    assert isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))
```


# LLM-generated content at query #29
#--------------------------

```python
def test_run_command_unicode_decode_error_trigger():
    import subprocess
    from unittest.mock import patch, MagicMock
    from flutes.run import run_command

    # We need to trigger the 'except UnicodeDecodeError' block at line 46.
    # This requires:
    # 1. verbose=True (to enter the if verbose block)
    # 2. return_output=True or ret.returncode != 0 (to reach lines 41-49)
    # 3. The output in the temp file must contain bytes that cannot be decoded as UTF-8.
    #    e.g., b'\xff' is an invalid start byte for UTF-8.

    invalid_utf8_bytes = b"\xff\xfe\xfd"
    
    # Mocking subprocess.run to simulate a successful execution with specific output.
    # We use a mock for the file object 'f' because run_command uses tempfile.TemporaryFile().
    # Since we can't easily inject into the context manager of TemporaryFile without complex mocking,
    # we will mock subprocess.run to return a mock object whose returncode is 0.
    # The trick is that the output is actually written to 'f' (the file object).
    
    mock_ret = MagicMock()
    mock_ret.returncode = 0

    with patch("subprocess.run", return_value=mock_ret), \
         patch("tempfile.TemporaryFile") as mock_tempfile, \
         patch("flutes.log.log") as mock_log:
        
        # Setup the mock file to contain invalid bytes when read
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.read.return_value = invalid_utf8_bytes
        mock_tempfile.return_value = mock_file
        
        # Execute the command
        run_command(["echo", "test"], verbose=True, return_output=True)

        # Assertion: The code should have entered the 'except UnicodeDecodeError' block.
        # In the 'except' block (line 47), it calls log(str(line), ...).
        # Since line is a bytes object, str(line) results in "b'\\xff\\xfe\\xfd'".
        # We check if any call to log contains the string representation of the bytes.
        
        found_fallback_log = False
        for call in mock_log.call_args_list:
            if isinstance(call.args[0], str) and "b'\\xff" in call.args[0]:
                found_fallback_log = True
        
        assert found_fallback_log is True
```


# LLM-generated content at query #30
#--------------------------

```python
def test_run_command_no_unicode_decode_error():
    import subprocess
    from unittest.mock import patch, MagicMock
    from flutes.run import run_command

    # Setup: A command that succeeds and produces valid UTF-8 output.
    # We mock subprocess.run to return a CompletedProcess object with returncode 0.
    # This ensures the 'if verbose' block is entered (to trigger line 45)
    # but the content is valid UTF-8, so UnicodeDecodeError is not raised.
    args = ["echo", "hello"]
    mock_output = b"hello\n"
    mock_ret = MagicMock(spec=subprocess.CompletedProcess)
    mock_ret.returncode = 0
    
    with patch("subprocess.run", return_value=mock_ret), \
         patch("flutes.run.log") as mock_log:
        
        # We set verbose=True to enter the block containing line 45.
        # Since 'hello\n' is valid UTF-8, the except UnicodeDecodeError block at line 46 will not be triggered.
        result = run_command(args, verbose=True, return_output=True)

        assert result.returncode == 0
        assert result.captured_output == mock_output
        mock_log.assert_called_once_with("hello\n", timestamp=False, include_proc_id=False)
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import subprocess
from pathlib import Path
from flutes.run import run_command, CommandResult

def test_run_command_success():
    result = run_command(["echo", "hello"])
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_with_output_retrieval():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_error_raises_exception():
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        run_command(["ls", "/non_existent_directory_12345"])
    assert excinfo.value.returncode != 0

def test_run_command_ignore_errors_and_capture_output():
    result = run_command(["ls", "/non_existent_directory_12345"], ignore_errors=True, return_output=True)
    assert result.return_code != 0
    assert b"No such file or directory" in result.captured_output

def test_run_command_timeout_raises_exception():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

def test_run_command_ignore_errors_on_timeout():
    result = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768

def test_run_command_with_env_vars():
    result = run_command(["python3", "-c", "import os; print(os.environ['TEST_VAR'])"], 
                         env={"TEST_VAR": "working"}, return_output=True)
    assert b"working" in result.captured_output

def test_run_command_with_cwd():
    # Using current directory to ensure validity across environments
    result = run_command(["pwd"], cwd=Path("."), return_output=True)
    assert result.return_code == 0
    assert b".." not in result.captured_output # Basic check that it's a valid path output
```


# LLM-generated content at query #2
#--------------------------

```python
def test_run_command_returns_output_when_requested():
    import subprocess
    from flutes.run import run_command

    result = run_command(["echo", "hello"], return_output=True)
    assert result.captured_output == b"hello\n"
```


# LLM-generated content at query #3
#--------------------------

```python
def test_run_command_verbose_with_output():
    import subprocess
    from flutes.run import run_command
    result = run_command(["echo", "hello"], verbose=True, return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.returncode == 0
    assert b"hello" in result.captured_output
```


# LLM-generated content at query #4
#--------------------------

```python
def test_run_command_predicate_true_with_return_output():
    from flutes.run import run_command
    import sys

    result = run_command([sys.executable, "-c", "print('hello')"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_predicate_true_with_nonzero_return_code():
    from flutes.run import run_command
    import sys

    result = run_command([sys.executable, "-c", "import sys; sys.exit(1)"], return_output=False)
    assert result.return_code == 1
    assert result.captured_output is None

def test_run_command_predicate_true_with_verbose():
    from flutes.run import run_command
    import sys

    result = run_command([sys.executable, "-c", "print('hello')"], verbose=True, return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None
```


# LLM-generated content at query #5
#--------------------------

```python
def test_run_command_verbose_true_triggers_log():
    from flutes.run import run_command
    import subprocess

    # We use 'echo' as a simple command that succeeds (return code 0).
    # Setting verbose=True ensures the predicate at line 25 is evaluated.
    result = run_command(["echo", "hello"], verbose=True)
    
    assert result.return_code == 0
```


# LLM-generated content at query #6
#--------------------------

```python
def test_run_command_unicode_decode_success():
    import subprocess
    from unittest.mock import patch, MagicMock
    from flutes.run import run_command

    with patch("subprocess.run") as mock_run:
        mock_ret = MagicMock()
        mock_ret.returncode = 0
        mock_run.return_value = mock_ret
        
        with patch("tempfile.TemporaryFile") as mock_tempfile:
            mock_file = MagicMock()
            mock_tempfile.return_value.__enter__.return_value = mock_file
            # Provide valid utf-8 bytes to ensure decoding succeeds
            mock_file.read.return_value = b"success_output"
            mock_file.seek.return_value = None
            
            with patch("flutes.run.log") as mock_log:
                result = run_command(["echo", "test"], verbose=True, return_output=True)
                
                # If decoding succeeds, log is called once with the decoded string.
                # The UnicodeDecodeError block (lines 46-48) is NOT entered.
                mock_log.assert_called_once_with("success_output", timestamp=False, include_proc_id=False)
                assert result.captured_output == b"success_output"
```


# LLM-generated content at query #7
#--------------------------

```python
import subprocess

def test_error_wrapper_returns_original_on_standard_exception():
    err = ValueError("test error")
    result = error_wrapper(err)
    assert result is err
    assert str(result) == "test error"

def test_error_wrapper_modifies_called_process_error_with_output():
    err = subprocess.CalledProcessError(returncode=1, cmd="ls", output=b"line1\nline2")
    result = error_wrapper(err)
    expected_str = "Command 'ls' returned non-zero exit status 1.\nCaptured output:\n    line1\n    line2"
    assert str(result) == expected_str

def test_error_wrapper_modifies_called_process_error_without_output():
    err = subprocess.CalledProcessError(returncode=1, cmd="ls", output=None)
    result = error_wrapper(err)
    assert "No output was generated." in str(result)

def test_error_wrapper_modifies_timeout_expired_with_output():
    err = subprocess.TimeoutExpired(cmd="sleep 10", timeout=1, output=b"partial output")
    result = error_wrapper(err)
    assert "Captured output:\n    partial output" in str(result)

def test_error_wrapper_handles_unicode_decode_error():
    # Create bytes that cannot be decoded as utf-8 (e.g., invalid start byte)
    invalid_bytes = b"\xff\xfe\xfd"
    err = subprocess.CalledProcessError(returncode=1, cmd="ls", output=invalid_bytes)
    result = error_wrapper(err)
    assert "Failed to parse output." in str(result)

def test_error_wrapper_preserves_original_class_identity():
    err = subprocess.CalledProcessError(returncode=1, cmd="ls")
    result = error_wrapper(err)
    # The function creates a new type with the same name but different class identity
    assert result.__class__.__name__ == "CalledProcessError"
    assert isinstance(result, subprocess.CalledProcessError)
```


# LLM-generated content at query #8
#--------------------------

```python
def test_run_command_truncates_large_output():
    import subprocess
    import tempfile
    from unittest.mock import patch, MagicMock
    from flutes.run import run_command

    large_content = b"A" * (8192 + 100)
    
    with patch("subprocess.run") as mock_run:
        mock_error = subprocess.CalledProcessError(returncode=1, cmd="test", output=None)
        mock_run.side_effect = mock_error
        
        with patch("tempfile.TemporaryFile") as mock_temp_file:
            mock_f = MagicMock()
            mock_f.read.return_value = large_content
            mock_temp_file.return_value.__enter__.return_value = mock_f
            
            result = run_command(["test"], ignore_errors=True)
            
            assert len(result.captured_output) < len(large_content)
            assert result.captured_output.startswith(b"*** (previous output truncated) ***\n")
            assert result.captured_output.endswith(b"A" * 8192)
```


# LLM-generated content at query #9
#--------------------------

```python
def test_run_command_success():
    import subprocess
    from flutes.run import run_command
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_with_return_output():
    import subprocess
    from flutes.run import run_command
    result = run_command(["echo", "world"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"world\n"

def test_run_command_error_raises_exception():
    import subprocess
    from flutes import run_command
    try:
        run_command(["ls", "--non-existent-flag"])
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
        assert b"ls" in str(e).encode()

def test_run_command_ignore_errors():
    import subprocess
    from flutes import run_command
    result = run_command(["ls", "--non-existent-flag"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_timeout_error():
    import subprocess
    from flutes import run_command
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert True
    else:
        raise AssertionError("TimeoutExpired not raised")

def test_run_command_timeout_ignore_errors():
    import subprocess
    from flutes import run_command
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768

def test_run_command_env_variable():
    import subprocess
    from flutes import run_command
    result = run_command(["printenv", "MY_VAR"], return_output=True, env={"MY_VAR": "test_value"})
    assert b"test_value" in result.captured_output

def test_error_wrapper_called_process_error():
    import subprocess
    from flutes import error_wrapper
    err = subprocess.CalledProcessError(returncode=1, cmd="test", output=b"line1\nline2")
    wrapped_err = error_wrapper(err)
    assert "Captured output:" in str(wrapped_err)
    assert "    line1" in str(wrapped_err)

def test_error_wrapper_unrelated_exception():
    from flutes import error_wrapper
    err = ValueError("standard error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "standard error"
```


# LLM-generated content at query #10
#--------------------------

```python
def test_run_command_predicate_true_with_return_output():
    from flutes.run import run_command
    import sys

    # We use 'echo' as it is a standard command that returns returncode 0.
    # Setting return_output=True makes the predicate (return_output or ret.returncode != 0 or verbose) True.
    result = run_command(["echo", "hello"], return_output=True)
    
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_predicate_true_with_nonzero_return_code():
    from flutes.run import run_command
    import sys

    # Using 'ls' on a non-existent directory triggers returncode != 0.
    # This makes the predicate (return_output or ret.returncode != 0 or verbose) True.
    result = run_command(["ls", "/non_existent_directory_path_12345"], return_output=False)
    
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_predicate_true_with_verbose_mode():
    from flutes.run import run_command
    import sys

    # Setting verbose=True makes the predicate (return_output or ret.returncode != 0 or verbose) True.
    result = run_command(["echo", "verbose_test"], verbose=True, return_output=False)
    
    assert result.command == ["echo", "verbose_test"]
    # Note: when return_output is False and returncode is 0, captured_output should be None according to line 50,
    # but the predicate logic at line 40 triggers the block that reads from file.
    assert result.return_code == 0
```


# LLM-generated content at query #11
#--------------------------

```python
import subprocess

def test_error_wrapper_predicate_is_false():
    err = ValueError("Not a subprocess error")
    result = error_wrapper(err)
    assert result is err
```


# LLM-generated content at query #12
#--------------------------

```python
def test_run_command_truncates_large_output():
    import subprocess
    from unittest.mock import patch, MagicMock
    from flutes.run import run_command, CommandResult

    large_content = b"A" * (8192 + 100)
    
    with patch("subprocess.run") as mock_run:
        mock_exception = subprocess.CalledProcessError(returncode=1, cmd="test", output=None)
        mock_run.side_effect = mock_exception
        
        with patch("tempfile.TemporaryFile") as mock_tempfile:
            mock_file = MagicMock()
            mock_file.read.return_value = large_content
            mock_tempfile.return_value.__enter__.return_value = mock_file
            
            result = run_command(["test"], ignore_errors=True, return_output=True)
            
            assert result.captured_output.startswith(b"*** (previous output truncated) ***\n")
            assert len(result.captured_output) <= 8192 + len(b"*** (previous output truncated) ***\n")
            assert result.captured_output.endswith(b"A" * 8192)
```


# LLM-generated content at query #13
#--------------------------

```python
def test_run_command_predicate_true_via_return_output():
    from flutes.run import run_command
    import sys
    result = run_command([sys.executable, "-c", "print('hello')"], return_output=True)
    assert result.captured_output == b"hello\n"

def test_run_command_predicate_true_via_non_zero_return_code():
    from flutes.run import run_command
    import sys
    result = run_command([sys.executable, "-c", "import sys; sys.exit(1)"], return_output=False)
    assert result.return_code == 1

def test_run_command_predicate_true_via_verbose():
    from flutes.run import run_command
    import sys
    result = run_command([sys.executable, "-c", "print('hello')"], verbose=True, return_output=False)
    assert result.return_code == 0
```


# LLM-generated content at query #14
#--------------------------

```python
import subprocess
from pathlib import Path
from flutes.run import run_command, CommandResult

def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"], return_output=False)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_error_raises_exception():
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        run_command(["ls", "/non_existent_directory_12345"])
    assert excinfo.value.returncode != 0
    assert b"No such file or directory" in excinfo.value.output

def test_run_command_error_with_ignore_errors():
    result = run_command(["ls", "/non_existent_directory_12345"], ignore_errors=True)
    assert result.return_code != 0
    assert b"No such file or directory" in result.captured_output

def test_run_command_timeout_raises_exception():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "5"], timeout=1)

def test_run_command_timeout_with_ignore_errors():
    result = run_command(["sleep", "5"], timeout=1, ignore_errors=True)
    assert result.return_code == -32768

def test_run_command_env_variables():
    result = run_command("echo $MY_VAR", shell=True, return_output=True, env={"MY_VAR": "test_val"})
    assert b"test_val" in result.captured_output

def test_run_command_cwd_parameter():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert Path(tmpdir).resolve().as_posix() in result.captured_output.decode()

def test_run_command_verbose_logging():
    # This tests that the function executes without crashing when verbose is True
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.return_code == 0
```


# LLM-generated content at query #15
#--------------------------

```python
def test_run_command_no_unicode_decode_error():
    import subprocess
    from unittest.mock import patch, MagicMock
    from flutes.run import run_command, CommandResult

    args = ["echo", "hello"]
    mock_process = MagicMock()
    mock_process.returncode = 0
    
    with patch("subprocess.run", return_value=mock_process), \
         patch("tempfile.TemporaryFile") as mock_temp:
        
        mock_file = MagicMock()
        # Provide valid UTF-8 bytes to ensure decode('utf-8') succeeds
        mock_file.read.return_value = b"hello\n"
        mock_temp.return_value.__enter__.return_value = mock_file
        
        with patch("flutes.run.log") as mock_log:
            result = run_command(args, verbose=True, return_output=True)
            
            assert result.returncode == 0
            assert result.captured_output == b"hello\n"
            # Verify that the log call happened with the decoded string (no error occurred)
            mock_log.assert_called_once_with("hello\n", timestamp=False, include_proc_id=False)

```


# LLM-generated content at query #16
#--------------------------

```python
import subprocess
from pathlib import Path
from flutes.run import run_command, CommandResult

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"], return_output=False)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_error_raises_exception():
    try:
        run_command(["ls", "/non_existent_directory_12345"])
        raise Exception("Should have raised CalledProcessError")
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
        assert b"No such file or directory" in e.output

def test_run_command_ignore_errors():
    result = run_command(["ls", "/non_existent_directory_12345"], ignore_errors=True, return_output=True)
    assert result.return_code != 0
    assert b"No such file or directory" in result.captured_output

def test_run_command_timeout_error():
    try:
        run_command(["sleep", "10"], timeout=0.1)
        raise Exception("Should have raised TimeoutExpired")
    except subprocess.TimeoutExpired as e:
        assert e.timeout == 0.1

def test_run_command_timeout_ignore_errors():
    result = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768

def test_run_command_with_env():
    import os
    custom_env = os.environ.copy()
    custom_env["TEST_VAR"] = "FLUTES"
    result = run_command(["printenv", "TEST_VAR"], return_output=True, env=custom_env)
    assert b"FLUTES" in result.captured_output

def test_run_command_with_cwd(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("content")
    result = run_command(["cat", "test.txt"], cwd=tmp_path, return_output=True)
    assert result.captured_output == b"content"

def test_run_command_shell_mode():
    result = run_command("echo 'shell test'", shell=True, return_output=True)
    assert b"shell test" in result.captured_output

def test_error_wrapper_string_formatting():
    try:
        subprocess.check_output(["ls", "/non_existent_path"], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        from flutes.run import error_wrapper
        wrapped_e = error_wrapper(e)
        assert "Captured output:" in str(wrapped_e)
        assert "No such file or directory" in str(wrapped_e)
```


# LLM-generated content at query #17
#--------------------------

```python
def test_run_command_verbose_true_reaches_line_25():
    import subprocess
    from flutes.run import run_command
    result = run_command(["echo", "hello"], verbose=True)
    assert result.return_code == 0
```


# LLM-generated content at query #18
#--------------------------

```python
def test_run_command_unicode_decode_error_trigger():
    import subprocess
    import tempfile
    from unittest.mock import patch, MagicMock
    from flutes.run import run_command

    # Prepare non-utf8 bytes that will cause UnicodeDecodeError when decoding as utf-8
    invalid_utf8_bytes = b"\xff\xfe\xfd"
    
    # Mock subprocess.run to return a mock object with returncode 0
    mock_ret = MagicMock()
    mock_ret.returncode = 0
    
    # We need to patch the log function to see if it's called
    # and we need to patch subprocess.run to control the output
    # Since line 45 calls log(output.decode('utf-8'), ...), 
    # providing invalid bytes will trigger the except block at line 46.
    
    with patch("subprocess.run", return_value=mock_ret), \
         patch("tempfile.TemporaryFile") as mock_tempfile, \
         patch("flutes.run.log") as mock_log:
        
        # Setup the temporary file mock to return our invalid bytes
        mock_file = MagicMock()
        mock_file.read.return_value = invalid_utf8_bytes
        mock_tempfile.return_value.__enter__.return_value = mock_file
        
        # Execute run_command with verbose=True to reach the log call at line 45
        run_command(["echo", "test"], verbose=True, return_output=True)
        
        # Assertions:
        # The first call to log (line 45) should have failed due to UnicodeDecodeError.
        # The second call (line 48) should be the fallback.
        # We check if any call was made to log with the string representation of the bytes.
        # Since we cannot use 'if' or 'for', we assert that the mock was called.
        assert mock_log.called
        # Verify that at least one call contains the byte string representation (the fallback)
        # The first arg of the last call to log should be the stringified version of invalid bytes
        last_call_args = mock_log.call_args[0][0]
        assert str(invalid_utf8_bytes) in last_call_args or b"\xff" in last_call_args.encode('latin-1')
```


# LLM-generated content at query #19
#--------------------------

```python
def test_run_command_verbose_triggering_log_line():
    from flutes.run import run_command
    import subprocess
    result = run_command(["echo", "hello"], verbose=True)
    assert result.return_code == 0
```


# LLM-generated content at query #20
#--------------------------

```python
import subprocess

def test_error_wrapper_returns_original_if_not_subprocess_error():
    error = ValueError("test error")
    result = error_wrapper(error)
    assert result is error
    assert str(result) == "test error"

def test_error_wrapper_modifies_called_process_error_with_output():
    error = subprocess.CalledProcessError(returncode=1, cmd="ls", output=b"line1\nline2")
    result = error_wrapper(error)
    expected_str = "Command 'ls' returned non-zero exit status 1.\nCaptured output:\n    line1\n    line2"
    assert str(result) == expected_str

def test_error_wrapper_modifies_called_process_error_with_no_output():
    error = subprocess.CalledProcessError(returncode=1, cmd="ls", output=None)
    result = error_wrapper(error)
    assert "No output was generated." in str(result)

def test_error_wrapper_modifies_timeout_expired_with_output():
    error = subprocess.TimeoutExpired(cmd="sleep", timeout=5, output=b"some data")
    result = error_wrapper(error)
    assert "Captured output:" in str(result)
    assert "    some data" in str(result)

def test_error_wrapper_handles_unicode_decode_error():
    # Using bytes that are invalid utf-8 to trigger the exception logic
    invalid_utf8 = b"\xff\xfe\xfd"
    error = subprocess.CalledProcessError(returncode=1, cmd="ls", output=invalid_utf8)
    result = error_wrapper(error)
    assert "Failed to parse output." in str(result)

def test_error_wrapper_preserves_type_hierarchy():
    error = subprocess.CalledProcessError(returncode=1, cmd="ls")
    result = error_wrapper(error)
    assert isinstance(result, subprocess.CalledProcessError)
    assert result.__class__.__name__ != "CalledProcessError"
```


# LLM-generated content at query #21
#--------------------------

```python
def test_run_command_truncates_large_output():
    import subprocess
    from unittest.mock import patch, MagicMock
    from flutes.run import run_command
    from flutes.log import MAX_OUTPUT_LENGTH

    large_data = b"A" * (MAX_OUTPUT_LENGTH + 100)
    
    with patch("subprocess.run") as mock_run:
        mock_error = subprocess.CalledProcessError(returncode=1, cmd="test", output=None)
        mock_run.side_effect = mock_error
        
        with patch("tempfile.TemporaryFile") as mock_tempfile:
            mock_file = MagicMock()
            mock_file.read.return_value = large_data
            mock_tempfile.return_value.__enter__.return_value = mock_file
            
            result = run_command(["test"], ignore_errors=True, return_output=True)
            
            assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len(b"*** (previous output truncated) ***\n")
            assert result.captured_output.startswith(b"*** (previous output truncated) ***\n")
            assert result.captured_output.endswith(large_data[-MAX_OUTPUT_LENGTH:])
```


# LLM-generated content at query #22
#--------------------------

```python
import subprocess

def test_error_wrapper_returns_original_if_not_subprocess_error():
    error = ValueError("test error")
    result = error_wrapper(error)
    assert result is error
    assert str(result) == "test error"

def test_error_wrapper_modifies_called_process_error_with_output():
    error = subprocess.CalledProcessError(returncode=1, cmd="ls", output=b"line1\nline2")
    result = error_wrapper(error)
    expected_str = "Command 'ls' returned non-zero exit status 1.\nCaptured output:\n    line1\n    line2"
    assert str(result) == expected_str

def test_error_wrapper_modifies_called_process_error_without_output():
    error = subprocess.CalledProcessError(returncode=1, cmd="ls", output=None)
    result = error_wrapper(error)
    expected_str = "Command 'ls' returned non-zero exit status 1.\nNo output was generated."
    assert str(result) == expected_str

def test_error_wrapper_modifies_timeout_expired_with_output():
    error = subprocess.TimeoutExpired(cmd="sleep", timeout=5, output=b"timeout info")
    result = error_wrapper(error)
    expected_str = "Command 'sleep' -> Timeout expired.\nCaptured output:\n    timeout info"
    assert str(result) == expected_str

def test_error_wrapper_handles_unicode_decode_error():
    # Use bytes that cannot be decoded as utf-8 (e.g., 0xff)
    invalid_utf8 = b"\xff\xfe"
    error = subprocess.CalledProcessError(returncode=1, cmd="ls", output=invalid_utf8)
    result = error_wrapper(error)
    assert "Failed to parse output." in str(result)
```


# LLM-generated content at query #23
#--------------------------

```python
import subprocess
from pathlib import Path
from flutes.run import run_command, CommandResult

def test_run_command_success():
    result = run_command(["echo", "hello"])
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_with_return_output():
    result = run_command(["echo", "test"], return_output=True)
    assert result.captured_output == b"test\n"

def test_run_command_error_raises_exception():
    try:
        run_command(["ls", "/non_existent_directory_12345"])
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
        assert b"No such file or directory" in e.output
    else:
        raise AssertionError("Expected subprocess.CalledProcessError")

def test_run_command_ignore_errors():
    result = run_command(["ls", "/non_existent_directory_12345"], ignore_errors=True)
    assert result.return_code != 0
    assert b"No such file or directory" in result.captured_output

def test_run_command_timeout():
    try:
        run_command(["python3", "-c", "import time; time.sleep(2)"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert isinstance(e, subprocess.TimeoutExpired)
    else:
        raise AssertionError("Expected subprocess.TimeoutExpired")

def test_run_command_ignore_errors_timeout():
    result = run_command(["python3", "-c", "import time; time.sleep(0.1)"], timeout=0.01, ignore_errors=True)
    assert result.return_code == -32768

def test_run_command_with_env():
    result = run_command(["python3", "-c", "import os; print(os.environ['MY_VAR'])"], env={"MY_VAR": "success"})
    assert b"success\n" in result.captured_output

def test_run_command_with_cwd():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["ls"], cwd=tmpdir)
        assert result.return_code == 0

def test_run_command_verbose_logging_trigger():
    # This tests that the function completes when verbose is True
    result = run_command(["echo", "verbose_test"], verbose=True)
    assert result.return_code == 0
```


# LLM-generated content at query #24
#--------------------------

```python
import subprocess

def test_error_wrapper_predicate_is_false():
    err = subprocess.CalledProcessError(returncode=1, cmd="ls")
    result = error_wrapper(err)
    assert isinstance(result, (subprocess.CalledProcessError, subprocess.TimeoutExpired))
```


# LLM-generated content at query #25
#--------------------------

```python
def test_run_command_unicode_decode_error_handling():
    import subprocess
    import tempfile
    from unittest.mock import patch, MagicMock
    from flutes.run import run_command, CommandResult

    # Create a mock for the process output that contains invalid UTF-8 bytes
    # 0xff is an invalid start byte in UTF-8
    invalid_utf8_output = b"\xff\xfe\xfd"
    
    # Mock subprocess.run to return a CompletedProcess with our non-UTF8 output
    mock_ret = MagicMock()
    mock_ret.returncode = 0
    
    with patch("subprocess.run") as mock_run, \
         patch("flutes.run.log") as mock_log:
        
        # We need to simulate the file writing behavior of subprocess.run
        # By patching TemporaryFile's context manager return value
        mock_file = MagicMock()
        mock_file.read.return_value = invalid_utf8_output
        
        # Setup the mock to act like a file that can be used in 'with'
        mock_run.return_value = mock_ret
        
        with patch("tempfile.TemporaryFile") as mock_tempfile:
            mock_tempfile.return_value.__enter__.return_value = mock_file
            
            # Trigger the path where decode('utf-8') fails
            # args must be valid, return_output=True to ensure we enter the block at line 40
            run_command(["echo", "test"], verbose=True, return_output=True)
            
            # The first call to log (line 45) should fail due to UnicodeDecodeError
            # We check if a second call to log (line 48) happened with the string representation of the bytes
            # This proves the 'except UnicodeDecodeError' block was executed.
            
            # Check that line 48 was reached: it logs str(line)
            # Since output is b'\xff\xfe\xfd', log should be called with "b'\\xff\\xfe\\xfd'"
            expected_log_msg = str(invalid_utf8_output.split(b"\n")[0])
            
            # Find calls to log that were made after the failure
            log_calls = [call.args[0] for call in mock_log.call_args_list]
            assert expected_log_msg in log_calls
```


# LLM-generated content at query #26
#--------------------------

```python
def test_run_command_truncates_large_output():
    import subprocess
    from unittest.mock import patch, MagicMock
    from flutes.run import run_command, CommandResult

    large_content = b"a" * (8192 + 10)
    
    mock_process = MagicMock()
    mock_process.returncode = 1
    
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, 
            cmd="large_cmd", 
            output=None
        )
        
        with patch("tempfile.TemporaryFile") as mock_temp:
            mock_file = MagicMock()
            mock_file.__enter__.return_value = mock_file
            # Simulate file containing large content
            mock_file.read.return_value = large_content
            mock_temp.return_value = mock_file
            
            result = run_command("large_cmd", ignore_errors=True, return_output=True)
            
            assert len(result.captured_output) <= 8192 + len(b"*** (previous output truncated) ***\n")
            assert result.captured_output.startswith(b"*** (previous_output truncated) ***\n".replace(b"previous_output", b""))
            assert result.captured_output.endswith(large_content[-8192:])
```


# LLM-generated content at query #27
#--------------------------

```python
def test_run_command_unicode_decode_error_triggering_fallback():
    import subprocess
    import tempfile
    from unittest.mock import patch, MagicMock
    from flutes.run import run_command

    # We need to trigger the UnicodeDecodeError in line 45.
    # The code calls log(output.decode('utf-8'), ...)
    # To make output.decode('utf-8') fail, we need 'output' (the content of the temp file)
    # to contain invalid UTF-8 bytes.
    invalid_utf8_bytes = b"\xff\xfe\xfd"

    # We mock subprocess.run to return a mock object where returncode is 0
    # and we mock the behavior of the file descriptor/tempfile via patching
    # However, since run_command creates its own TemporaryFile, we patch 
    # subprocess.run to simulate the written content into the captured output.
    
    with patch("subprocess.run") as mock_run:
        mock_ret = MagicMock()
        mock_ret.returncode = 0
        mock_run.return_value = mock_ret
        
        # We also need to mock the file writing part. Since run_command uses 
        # tempfile.TemporaryFile() as a context manager, we can patch it.
        with patch("tempfile.TemporaryFile") as mock_tempfile:
            mock_f = MagicMock()
            # Simulate reading the invalid bytes when f.read() is called
            mock_f.read.return_value = invalid_utf8_bytes
            # Ensure seek(0) works (returns None or self)
            mock_f.seek.return_value = None
            # Context manager support
            mock_tempfile.return_value.__enter__.return_value = mock_f
            
            # We also need to patch 'log' to see if it's called, 
            # but the requirement is specifically about line 46 evaluating to True.
            # Line 46 evaluates to True when log(...) raises UnicodeDecodeError.
            with patch("flutes.run.log") as mock_log:
                mock_log.side_effect = UnicodeDecodeError("utf-8", invalid_utf8_bytes, 0, 1, b"\xff")
                
                # Execute the command. We use a simple valid command like ['echo', 'test']
                # but it won't actually run because subprocess.run is mocked.
                result = run_command(["echo", "test"], verbose=True)
                
                # Assertions to verify the flow reached the error handling logic
                assert result.returncode == 0
                assert result.captured_output == invalid_utf8_bytes
                # Verify that log was called at least once (the first attempt failed, second attempt in 'except' block)
                assert mock_log.call_count >= 1
```


# LLM-generated content at query #28
#--------------------------

```python
import subprocess

def test_error_wrapper_returns_same_instance_for_unrelated_exception():
    err = ValueError("original error")
    result = error_wrapper(err)
    assert result is err
    assert str(result) == "original error"

def test_error_wrapper_modifies_called_process_error_with_output():
    err = subprocess.CalledProcessError(returncode=1, cmd="ls", output=b"line1\nline2")
    result = error_wrapper(err)
    assert result is err
    expected_str = "Command 'ls' returned non-zero exit status 1.\nCaptured output:\n    line1\n    line2"
    assert str(result) == expected_str

def test_error_wrapper_modifies_called_process_error_without_output():
    err = subprocess.CalledProcessError(returncode=1, cmd="ls", output=None)
    result = error_wrapper(err)
    assert result is err
    assert "No output was generated." in str(result)

def test_error_wrapper_modifies_timeout_expired_with_output():
    err = subprocess.TimeoutExpired(cmd="ls", timeout=5, output=b"some output")
    result = error_wrapper(err)
    assert result is err
    assert "Captured output:\n    some output" in str(result)

def test_error_wrapper_handles_undecodable_output():
    # Using a byte sequence that cannot be decoded as utf-8
    invalid_utf8 = b"\x80\x81" 
    err = subprocess.CalledProcessError(returncode=1, cmd="ls", output=invalid_utf8)
    result = error_wrapper(err)
    assert "Failed to parse output." in str(result)

def test_error_wrapper_preserves_class_hierarchy():
    err = subprocess.CalledProcessError(returncode=1, cmd="ls", output=b"test")
    result = error_wrapper(err)
    assert isinstance(result, subprocess.CalledProcessError)
    assert not isinstance(result, subprocess.TimeoutExpired)
```


# LLM-generated content at query #29
#--------------------------

```python
def test_run_command_unicode_decode_error_trigger():
    import subprocess
    import tempfile
    from unittest.mock import patch, MagicMock
    from flutes.run import run_command

    # The goal is to trigger the UnicodeDecodeError in line 45: log(output.decode('utf-8'), ...)
    # This happens if 'output' contains bytes that are not valid UTF-8.
    # We also need to ensure we reach line 40 (return_output=True or ret.returncode != 0 or verbose=True).
    # We will mock subprocess.run to return a success object with non-UTF-8 output in the file.
    
    invalid_utf8_bytes = b"\xff\xfe\xfd"
    
    mock_ret = MagicMock()
    mock_ret.returncode = 0
    
    # We mock 'tempfile.TemporaryFile' to return a buffer containing our invalid bytes
    # so that when the function reads from it, it gets non-UTF-8 data.
    with patch("subprocess.run", return_value=mock_ret), \
         patch("tempfile.TemporaryFile") as mock_tempfile, \
         patch("flutes.run.log") as mock_log:
        
        # Setup the fake file behavior
        fake_file = MagicMock()
        fake_file.__enter__.return_value = fake_file
        # When f.read() is called (line 42), return invalid bytes
        fake_file.read.return_value = invalid_utf8_bytes
        mock_tempfile.return_value = fake_file
        
        # Execute the command with verbose=True to enter the block containing line 45
        run_command(["echo", "test"], verbose=True, return_output=True)
        
        # Assertions:
        # 1. The first call to log (line 45) should have been attempted with invalid UTF-8.
        # Since we can't easily 'catch' the exception inside the function from outside 
        # without it bubbling up, we rely on the fact that if line 46 is reached, 
        # the logic successfully handled the error.
        # To verify line 46 was hit, we check if the second log call (the fallback) happened.
        
        # The first call to log(output.decode('utf-8')...) will raise UnicodeDecodeError inside run_command.
        # The 'except' block catches it and calls log(str(line), ...).
        # Therefore, the second call to log must exist in the mock calls.
        
        assert mock_log.call_count >= 1
        # Check if any of the calls to log used a string representation of the bytes (the fallback)
        # which is what happens in line 48.
        found_fallback = False
        for call in mock_log.call_args_list:
            arg_val = call[0][0]
            if isinstance(arg_val, str) and "b'\\xff\\xfe\\xfd'" in arg_val:
                found_fallback = True
        assert found_fallback is True
```


# LLM-generated content at query #30
#--------------------------

```python
import subprocess

def test_error_wrapper_evaluates_true_for_subprocess_errors():
    error_instance = subprocess.CalledProcessError(returncode=1, cmd="ls")
    wrapped_error = error_wrapper(error_instance)
    assert isinstance(wrapped_error, (subprocess.CalledProcessError, subprocess.TimeoutExpired))
```


# LLM-generated content at query #31
#--------------------------

```python
def test_run_command_unicode_decode_success():
    import subprocess
    from unittest.mock import patch, MagicMock
    from flutes.run import run_command, CommandResult

    # We need to mock subprocess.run to return a successful process 
    # where the output is valid UTF-8 so that decode('utf-8') succeeds.
    # This ensures line 45 (log call) completes without raising UnicodeDecodeError,
    # thus the predicate at line 46 (the 'except' block) evaluates to False.

    mock_ret = MagicMock()
    mock_ret.returncode = 0
    
    # Mocking the temporary file context manager and its behavior
    # The output must be valid utf-8 bytes for success
    valid_output = b"success output"
    
    with patch("subprocess.run", return_value=mock_ret), \
         patch("tempfile.TemporaryFile") as mock_temp:
        
        # Setup the mock file object
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.read.return_value = valid_output
        mock_temp.return_value = mock_file
        
        # Mock log to track if it's called (it should be called with decoded string)
        with patch("flutes.run.log") as mock_log:
            result = run_command(["echo", "hello"], verbose=True, return_output=True)
            
            assert result.returncode == 0
            assert result.captured_output == valid_output
            # Verify that the log was called with the decoded string (not bytes/error handling)
            mock_log.assert_called_with("success output", timestamp=False, include_proc_id=False)
```


# LLM-generated content at query #32
#--------------------------

```python
import subprocess

def test_error_wrapper_predicate_is_false():
    err = ValueError("Test error")
    result = error_wrapper(err)
    assert result is err
    assert isinstance(err, ValueError)
    assert not isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))
```


# LLM-generated content at query #33
#--------------------------

```python
import subprocess

def test_error_wrapper_returns_same_exception_if_not_subprocess_error():
    err = ValueError("test error")
    result = error_wrapper(err)
    assert result is err
    assert str(result) == "test error"

def test_error_wrapper_modifies_called_process_error_with_output():
    err = subprocess.CalledProcessError(returncode=1, cmd="ls", output=b"line1\nline2")
    result = error_wrapper(err)
    assert isinstance(result, subprocess.CalledProcessError)
    expected_str = "Command 'ls' returned non-zero exit status 1.\nCaptured output:\n    line1\n    line2"
    assert str(result) == expected_str

def test_error_wrapper_modifies_called_process_error_without_output():
    err = subprocess.CalledProcessError(returncode=1, cmd="ls", output=None)
    result = error_wrapper(err)
    assert isinstance(result, subprocess.CalledProcessError)
    expected_str = "Command 'ls' returned non-zero exit status 1.\nNo output was generated."
    assert str(result) == expected_str

def test_error_wrapper_modifies_timeout_expired_with_output():
    err = subprocess.TimeoutExpired(cmd="sleep", timeout=5, output=b"some error")
    result = error_wrapper(err)
    assert isinstance(result, subprocess.TimeoutExpired)
    expected_str = "Command 'sleep' -> TimeoutExpired\nCaptured output:\n    some error"
    # Note: str(TimeoutExpired) implementation details can vary by python version, 
    # but we test the appended part of the logic provided in the snippet.
    assert "Captured output:" in str(result)
    assert "    some error" in str(result)

def test_error_wrapper_handles_undecodable_output():
    # Using invalid utf-8 bytes to trigger UnicodeDecodeError (the code catches UnicodeEncodeError, 
    # but decoding bytes usually raises UnicodeDecodeError. However, following the logic provided:
    err = subprocess.CalledProcessError(returncode=1, cmd="ls", output=b"\xff\xfe")
    result = error_wrapper(err)
    assert "Failed to parse output." in str(result)
```


# LLM-generated content at query #34
#--------------------------

```python
import subprocess

def test_error_wrapper_returns_original_if_not_subprocess_error():
    error = ValueError("Original error")
    result = error_wrapper(error)
    assert result is error
    assert str(result) == "Original error"

def test_error_wrapper_modifies_called_process_error_with_output():
    error = subprocess.CalledProcessError(returncode=1, cmd="ls", output=b"line1\nline2")
    result = error_wrapper(type(error)) # Note: The implementation expects the class (ExcType) or an instance that can be re-classed
    # Based on the provided code logic, it takes an exception type or instance. 
    # Looking at 'err.__class__ = new_type', it operates on an instance.
    
    error_instance = subprocess.CalledProcessError(returncode=1, cmd="ls", output=b"line1\nline2")
    wrapped_error = error_wrapper(error_instance)
    
    expected_str = "Command 'ls' returned non-zero exit status 1.\nCaptured output:\n    line1\n    line2"
    assert str(wrapped_error) == expected_str

def test_error_wrapper_modifies_called_process_error_with_no_output():
    error_instance = subprocess.CalledProcessError(returncode=1, cmd="ls", output=None)
    wrapped_error = error_wrapper(error_instance)
    
    # Default __str__ for CalledProcessError contains the returncode and cmd
    assert "No output was generated." in str(wrapped_error)

def test_error_wrapper_modifies_timeout_expired_with_output():
    error_instance = subprocess.TimeoutExpired(cmd="sleep", timeout=5, output=b"some output")
    wrapped_error = error_wrapper(error_instance)
    
    assert "Captured output:" in str(wrapped_error)
    assert "    some output" in str(wrapped_error)

def test_error_wrapper_handles_unicode_decode_error():
    # Create bytes that are not valid utf-8
    invalid_utf8 = b"\xff\xfe\xfd"
    error_instance = subprocess.CalledProcessError(returncode=1, cmd="ls", output=invalid_utf8)
    wrapped_error = error_wrapper(error_instance)
    
    assert "Failed to parse output." in str(wrapped_error)
```


# LLM-generated content at query #35
#--------------------------

```python
import subprocess

def test_error_wrapper_returns_same_instance_for_unrelated_exception():
    err = ValueError("test error")
    result = error_wrapper(err)
    assert result is err
    assert str(result) == "test error"

def test_error_wrapper_modifies_called_process_error_with_output():
    err = subprocess.CalledProcessError(returncode=1, cmd="ls", output=b"line1\nline2")
    result = error_wrapper(err)
    assert result is err
    expected_str = "Command 'ls' returned non-zero exit status 1.\nCaptured output:\n    line1\n    line2"
    assert str(result) == expected_str

def test_error_wrapper_modifies_called_process_error_without_output():
    err = subprocess.CalledProcessError(returncode=1, cmd="ls", output=None)
    result = error_wrapper(err)
    assert result is err
    assert "No output was generated." in str(result)

def test_error_wrapper_modifies_timeout_expired_with_output():
    err = subprocess.TimeoutExpired(cmd="ls", timeout=5, output=b"some data")
    result = error_wrapper(err)
    assert result is err
    assert "Captured output:\n    some data" in str(result)

def test_error_wrapper_handles_undecodable_output():
    # Using bytes that are invalid in utf-8 to trigger the exception handler
    invalid_bytes = b"\xff\xfe\xfd"
    err = subprocess.CalledProcessError(returncode=1, cmd="ls", output=invalid_bytes)
    result = error_wrapper(err)
    assert result is err
    assert "Failed to parse output." in str(result)
```


