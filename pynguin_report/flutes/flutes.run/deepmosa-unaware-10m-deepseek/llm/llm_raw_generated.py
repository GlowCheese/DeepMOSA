####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"test output")
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped)
    assert "test output" in str(wrapped)
    
    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_cmd", 10, output=b"timeout output")
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped)
    assert "timeout output" in str(wrapped)
    
    # Test with output that cannot be decoded
    err = subprocess.CalledProcessError(1, "test_cmd", output=b'\xff\xfe\x00\x00')
    wrapped = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped)
    
    # Test with no output
    err = subprocess.CalledProcessError(1, "test_cmd", output=None)
    wrapped = error_wrapper(err)
    assert "No output was generated." in str(wrapped)
    
    # Test with empty output
    err = subprocess.CalledProcessError(1, "test_cmd", output=b'')
    wrapped = error_wrapper(err)
    assert "No output was generated." in str(wrapped)
    
    # Test with other exception types (should return unchanged)
    original_exc = ValueError("test error")
    result = error_wrapper(original_exc)
    assert result is original_exc
    assert str(result) == "test error"
    
    # Test with multi-line output formatting
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"line1\nline2\nline3")
    wrapped = error_wrapper(err)
    str_repr = str(wrapped)
    assert "    line1" in str_repr
    assert "    line2" in str_repr
    assert "    line3" in str_repr


# LLM-generated content at query #2
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    error.output = b"test output\nline 2"
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert wrapped.returncode == 1
    assert "Captured output:" in str(wrapped)
    assert "test output" in str(wrapped)
    assert "line 2" in str(wrapped)
    
    # Test with CalledProcessError without output
    error2 = subprocess.CalledProcessError(returncode=2, cmd="test_cmd2")
    error2.output = None
    wrapped2 = error_wrapper(error2)
    assert "No output was generated." in str(wrapped2)
    
    # Test with TimeoutExpired
    error3 = subprocess.TimeoutExpired(cmd="test_cmd3", timeout=5)
    error3.output = b"timeout output"
    wrapped3 = error_wrapper(error3)
    assert isinstance(wrapped3, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped3)
    assert "timeout output" in str(wrapped3)
    
    # Test with non-subprocess exception (should return unchanged)
    regular_error = ValueError("test error")
    wrapped4 = error_wrapper(regular_error)
    assert wrapped4 is regular_error
    assert str(wrapped4) == "test error"
    
    # Test with Unicode decode error in output
    error5 = subprocess.CalledProcessError(returncode=3, cmd="test_cmd5")
    error5.output = b'\xff\xfe'  # Invalid UTF-8
    wrapped5 = error_wrapper(error5)
    assert "Failed to parse output." in str(wrapped5)


# LLM-generated content at query #3
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError with output
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"error output")
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped)
    assert "error output" in str(wrapped)
    
    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_cmd", output=None)
    wrapped = error_wrapper(err)
    assert "No output was generated." in str(wrapped)
    
    # Test with TimeoutExpired with output
    err = subprocess.TimeoutExpired("test_cmd", 10, output=b"timeout output")
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped)
    assert "timeout output" in str(wrapped)
    
    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_cmd", 10, output=None)
    wrapped = error_wrapper(err)
    assert "No output was generated." in str(wrapped)
    
    # Test with other exception types (should return unchanged)
    original_err = ValueError("test error")
    wrapped = error_wrapper(original_err)
    assert wrapped is original_err
    assert str(wrapped) == "test error"
    
    # Test Unicode decode error handling
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"\xff\xfe")
    wrapped = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped)
    
    # Test output formatting with multiple lines
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"line1\nline2\nline3")
    wrapped = error_wrapper(err)
    assert "    line1" in str(wrapped)
    assert "    line2" in str(wrapped)
    assert "    line3" in str(wrapped)


# LLM-generated content at query #4
#--------------------------

```python
def test_run_command():
    # Test successful command execution without output
    result = run_command(["echo", "test"], return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None
    assert result.command == ["echo", "test"]

    # Test successful command execution with output
    result = run_command(["echo", "test"], return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output
    assert result.command == ["echo", "test"]

    # Test command with non-zero exit code
    result = run_command(["false"], return_output=True, ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test command with environment variables
    env = {"TEST_VAR": "test_value"}
    result = run_command(["env"], env=env, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=test_value" in result.captured_output

    # Test command with working directory
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output

    # Test command timeout
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

    # Test verbose mode
    result = run_command(["echo", "verbose_test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"verbose_test" in result.captured_output

    # Test shell command
    result = run_command("echo shell_test", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"shell_test" in result.captured_output

    # Test command with large output truncation
    long_output = "A" * 10000
    result = run_command(["echo", long_output], return_output=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output

    # Test error propagation without ignore_errors
    import subprocess
    try:
        run_command(["false"])
        assert False, "Should have raised CalledProcessError"
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
        assert hasattr(e, 'output')

    # Test timeout error propagation without ignore_errors
    try:
        run_command(["sleep", "2"], timeout=0.1)
        assert False, "Should have raised TimeoutExpired"
    except subprocess.TimeoutExpired as e:
        assert hasattr(e, 'output')

    # Test command with stderr redirection
    result = run_command(["ls", "/nonexistent"], return_output=True, ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test command with binary output
    result = run_command(["printf", "\\x00\\x01\\x02"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"\x00\x01\x02"


# LLM-generated content at query #5
#--------------------------

```python
def test_run_command():
    # Test successful command execution without output
    result = run_command(["echo", "test"], return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None
    
    # Test successful command execution with return_output=True
    result = run_command(["echo", "test"], return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output
    
    # Test command with non-zero exit code
    result = run_command(["false"], return_output=True, ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None
    
    # Test command with timeout
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None
    
    # Test verbose mode
    result = run_command(["echo", "verbose_test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"verbose_test" in result.captured_output
    
    # Test with environment variables
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=test_value" in result.captured_output
    
    # Test with working directory
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output
    
    # Test shell command
    result = run_command("echo shell_test", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"shell_test" in result.captured_output
    
    # Test error propagation without ignore_errors
    import subprocess
    try:
        run_command(["false"])
        assert False, "Should have raised CalledProcessError"
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
    
    # Test output truncation for long output
    long_output = "x" * 10000
    result = run_command(["echo", long_output], return_output=True, ignore_errors=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output
    
    # Test command as list vs string
    result_list = run_command(["echo", "test"], return_output=True)
    result_str = run_command("echo test", shell=True, return_output=True)
    assert result_list.return_code == 0
    assert result_str.return_code == 0


# LLM-generated content at query #6
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    err.output = b"test output\nline 2"
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    str_repr = str(wrapped)
    assert "Captured output:" in str_repr
    assert "    test output" in str_repr
    assert "    line 2" in str_repr

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired(cmd="test_cmd", timeout=5)
    err.output = b"timeout output"
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    str_repr = str(wrapped)
    assert "Captured output:" in str_repr
    assert "    timeout output" in str_repr

    # Test with no output
    err = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    err.output = None
    wrapped = error_wrapper(err)
    str_repr = str(wrapped)
    assert "No output was generated." in str_repr

    # Test with empty output
    err = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    err.output = b""
    wrapped = error_wrapper(err)
    str_repr = str(wrapped)
    assert "No output was generated." in str_repr

    # Test with non-subprocess exception (should return unchanged)
    original_err = ValueError("test error")
    wrapped = error_wrapper(original_err)
    assert wrapped is original_err
    assert str(wrapped) == "test error"

    # Test with Unicode decode error in output
    err = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    err.output = b"\xff\xfe"  # Invalid UTF-8
    wrapped = error_wrapper(err)
    str_repr = str(wrapped)
    assert "Failed to parse output." in str_repr

    # Test that original error attributes are preserved
    err = subprocess.CalledProcessError(returncode=42, cmd="special_cmd")
    err.output = b"output"
    wrapped = error_wrapper(err)
    assert wrapped.returncode == 42
    assert wrapped.cmd == "special_cmd"
    assert wrapped.output == b"output"


# LLM-generated content at query #7
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError with output
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped)
    assert "test output" in str(wrapped)
    
    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command", output=None)
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped)
    
    # Test with TimeoutExpired with output
    err = subprocess.TimeoutExpired("test_command", 10, output=b"timeout output")
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped)
    assert "timeout output" in str(wrapped)
    
    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", 10, output=None)
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped)
    
    # Test with other exception types (should return unchanged)
    original_exc = ValueError("test error")
    wrapped = error_wrapper(original_exc)
    assert wrapped is original_exc
    assert str(wrapped) == "test error"
    
    # Test with Unicode decode error in output
    err = subprocess.CalledProcessError(1, "test_command", output=b'\xff\xfe')
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped)


# LLM-generated content at query #8
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    error.output = b"test output"
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped)
    assert "test output" in str(wrapped)
    
    # Test with TimeoutExpired
    error = subprocess.TimeoutExpExpired(timeout=5, cmd="test_cmd")
    error.output = b"timeout output"
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.TimeoutExpExpired)
    assert "Captured output:" in str(wrapped)
    assert "timeout output" in str(wrapped)
    
    # Test with no output
    error = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    error.output = None
    wrapped = error_wrapper(error)
    assert "No output was generated." in str(wrapped)
    
    # Test with empty output
    error = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    error.output = b""
    wrapped = error_wrapper(error)
    assert "No output was generated." in str(wrapped)
    
    # Test with non-subprocess exception (should return unchanged)
    original_error = ValueError("test error")
    wrapped = error_wrapper(original_error)
    assert wrapped is original_error
    assert str(wrapped) == "test error"
    
    # Test with unicode output
    error = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    error.output = "test unicode output".encode('utf-8')
    wrapped = error_wrapper(error)
    assert "test unicode output" in str(wrapped)
    
    # Test that __str__ method is properly added
    error = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    error.output = b"line1\nline2\nline3"
    wrapped = error_wrapper(error)
    str_repr = str(wrapped)
    assert "    line1" in str_repr
    assert "    line2" in str_repr
    assert "    line3" in str_repr


# LLM-generated content at query #9
#--------------------------

```python
def test_run_command():
    # Test successful command execution without output
    result = run_command(["echo", "test"], return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None
    assert result.command == ["echo", "test"]
    
    # Test successful command execution with return_output=True
    result = run_command(["echo", "test"], return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output
    
    # Test command with non-zero exit code
    result = run_command(["false"], return_output=True, ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None
    
    # Test command with timeout
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None
    
    # Test verbose mode
    result = run_command(["echo", "verbose_test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"verbose_test" in result.captured_output
    
    # Test with custom environment
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=test_value" in result.captured_output
    
    # Test with custom working directory
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output
    
    # Test shell command with string argument
    result = run_command("echo shell_test", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"shell_test" in result.captured_output
    
    # Test error propagation without ignore_errors
    import subprocess
    try:
        run_command(["false"])
        assert False, "Should have raised CalledProcessError"
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
    
    # Test timeout propagation without ignore_errors
    try:
        run_command(["sleep", "2"], timeout=0.1)
        assert False, "Should have raised TimeoutExpired"
    except subprocess.TimeoutExpired:
        pass
    
    # Test output truncation for long output
    long_output = "x" * 10000
    result = run_command(["echo", long_output], return_output=True, ignore_errors=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output
    assert len(result.captured_output) <= 8192 + len("*** (previous output truncated) ***\n")
    
    # Test with stderr redirection
    result = run_command(["ls", "/nonexistent"], return_output=True, ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None


# LLM-generated content at query #10
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError with output
    err = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    err.output = b"Error output line 1\nError output line 2"
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert wrapped.returncode == 1
    assert wrapped.cmd == "test_cmd"
    assert "Captured output:" in str(wrapped)
    assert "    Error output line 1" in str(wrapped)
    assert "    Error output line 2" in str(wrapped)

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(returncode=2, cmd="test_cmd2")
    err.output = None
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped)

    # Test with CalledProcessError with empty output
    err = subprocess.CalledProcessError(returncode=3, cmd="test_cmd3")
    err.output = b""
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped)

    # Test with TimeoutExpired with output
    err = subprocess.TimeoutExpired(cmd="test_cmd", timeout=5)
    err.output = b"Timeout output line 1\nTimeout output line 2"
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert wrapped.timeout == 5
    assert "Captured output:" in str(wrapped)
    assert "    Timeout output line 1" in str(wrapped)
    assert "    Timeout output line 2" in str(wrapped)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired(cmd="test_cmd2", timeout=10)
    err.output = None
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped)

    # Test with non-subprocess exception (should return unchanged)
    original_err = ValueError("Test error")
    wrapped = error_wrapper(original_err)
    assert wrapped is original_err
    assert str(wrapped) == "Test error"

    # Test with Unicode decode error in output
    err = subprocess.CalledProcessError(returncode=4, cmd="test_cmd4")
    err.output = b"\xff\xfe"  # Invalid UTF-8
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped)

    # Test that original exception attributes are preserved
    err = subprocess.CalledProcessError(returncode=5, cmd="test_cmd5")
    err.output = b"Some output"
    err.some_custom_attr = "custom_value"
    wrapped = error_wrapper(err)
    assert hasattr(wrapped, 'some_custom_attr')
    assert wrapped.some_custom_attr == "custom_value"


# LLM-generated content at query #11
#--------------------------

```python
def test_run_command():
    # Test successful command execution without output
    result = run_command(["echo", "test"], return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None
    assert result.command == ["echo", "test"]
    
    # Test successful command execution with output
    result = run_command(["echo", "test"], return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output
    assert result.command == ["echo", "test"]
    
    # Test command with non-zero return code
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None
    
    # Test command with environment variables
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=test_value" in result.captured_output
    
    # Test command with working directory
    import os
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.return_code == 0
    assert b"/tmp" in result.captured_output
    
    # Test command timeout
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None
    
    # Test verbose mode
    result = run_command(["echo", "verbose_test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"verbose_test" in result.captured_output
    
    # Test shell command
    result = run_command("echo shell_test", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"shell_test" in result.captured_output
    
    # Test command with large output truncation
    result = run_command(["python3", "-c", "print('x'*10000)"], return_output=True)
    assert result.return_code == 0
    assert len(result.captured_output) <= 8192 + len("*** (previous output truncated) ***\n")
    
    # Test error propagation without ignore_errors
    import subprocess
    try:
        run_command(["false"])
        assert False, "Should have raised CalledProcessError"
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
        assert e.output is not None
    
    # Test timeout propagation without ignore_errors
    try:
        run_command(["sleep", "2"], timeout=0.1)
        assert False, "Should have raised TimeoutExpired"
    except subprocess.TimeoutExpired as e:
        assert e.output is not None
    
    # Test command with stderr redirection
    result = run_command(["python3", "-c", "import sys; sys.stderr.write('error output')"], 
                         return_output=True)
    assert result.return_code == 0
    assert b"error output" in result.captured_output
    
    # Test command with both stdout and stderr
    result = run_command(["python3", "-c", "import sys; sys.stdout.write('stdout'); sys.stderr.write('stderr')"], 
                         return_output=True)
    assert result.return_code == 0
    assert b"stdout" in result.captured_output
    assert b"stderr" in result.captured_output


# LLM-generated content at query #12
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError with output
    err = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    err.output = b"error line 1\nerror line 2"
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped)
    assert "error line 1" in str(wrapped)
    assert "error line 2" in str(wrapped)
    
    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    err.output = None
    wrapped = error_wrapper(err)
    assert "No output was generated." in str(wrapped)
    
    # Test with CalledProcessError with empty output
    err = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    err.output = b""
    wrapped = error_wrapper(err)
    assert "No output was generated." in str(wrapped)
    
    # Test with TimeoutExpired with output
    err = subprocess.TimeoutExpired(cmd="test_cmd", timeout=10)
    err.output = b"timeout output"
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped)
    assert "timeout output" in str(wrapped)
    
    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired(cmd="test_cmd", timeout=10)
    err.output = None
    wrapped = error_wrapper(err)
    assert "No output was generated." in str(wrapped)
    
    # Test with other exception types (should return unchanged)
    original_exc = ValueError("test error")
    wrapped = error_wrapper(original_exc)
    assert wrapped is original_exc
    assert str(wrapped) == "test error"
    
    # Test with Unicode decode error in output
    err = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    err.output = b"\xff\xfe"  # Invalid UTF-8
    wrapped = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped)
    
    # Test that __str__ method preserves original error message
    err = subprocess.CalledProcessError(returncode=2, cmd="ls -la")
    err.output = b"file not found"
    wrapped = error_wrapper(err)
    str_repr = str(wrapped)
    assert "Command 'ls -la' returned non-zero exit status 2." in str_repr
    assert "Captured output:" in str_repr
    assert "file not found" in str_repr


# LLM-generated content at query #13
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError with output
    err = subprocess.CalledProcessError(1, ["ls", "-la"], output=b"file1.txt\nfile2.txt")
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped)
    assert "file1.txt" in str(wrapped)
    assert "file2.txt" in str(wrapped)
    
    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(2, ["rm", "nonexistent"], output=None)
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped)
    
    # Test with TimeoutExpired with output
    err = subprocess.TimeoutExpired(["sleep", "10"], 5, output=b"processing...")
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped)
    assert "processing..." in str(wrapped)
    
    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired(["sleep", "10"], 5, output=None)
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped)
    
    # Test with other exception types (should return unchanged)
    original_exc = ValueError("test error")
    wrapped = error_wrapper(original_exc)
    assert wrapped is original_exc
    assert str(wrapped) == "test error"
    
    # Test with Unicode decode error in output
    err = subprocess.CalledProcessError(1, ["cmd"], output=b'\xff\xfe')
    wrapped = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped)
    
    # Test that __str__ method is properly added
    err = subprocess.CalledProcessError(1, ["test"], output=b"output")
    wrapped = error_wrapper(err)
    str_repr = str(wrapped)
    assert "return code 1" in str_repr
    assert "Captured output:" in str_repr
    assert "output" in str_repr


# LLM-generated content at query #14
#--------------------------

```python
def test_run_command():
    # Test successful command execution without output
    result = run_command(["echo", "test"], return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None
    
    # Test successful command execution with output
    result = run_command(["echo", "test"], return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output
    
    # Test command with non-zero return code
    result = run_command(["false"], return_output=True, ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None
    
    # Test command with timeout
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None
    
    # Test verbose mode
    result = run_command(["echo", "verbose_test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"verbose_test" in result.captured_output
    
    # Test with environment variables
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=test_value" in result.captured_output
    
    # Test with working directory
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output
    
    # Test shell command
    result = run_command("echo shell_test", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"shell_test" in result.captured_output
    
    # Test error propagation without ignore_errors
    import subprocess
    try:
        run_command(["false"])
        assert False, "Should have raised CalledProcessError"
    except subprocess.CalledProcessError:
        pass
    
    # Test output truncation for long output
    long_output = "x" * 10000
    result = run_command(["echo", long_output], return_output=True, ignore_errors=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output
    
    # Test command with list arguments
    result = run_command(["ls", "-la"], return_output=True, ignore_errors=True)
    assert isinstance(result.command, list)
    
    # Test command with string argument
    result = run_command("ls -la", shell=True, return_output=True, ignore_errors=True)
    assert isinstance(result.command, str)


# LLM-generated content at query #15
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped)
    assert "test output" in str(wrapped)
    
    # Test with CalledProcessError and no output
    error = subprocess.CalledProcessError(1, "test_command", output=None)
    wrapped = error_wrapper(error)
    assert "No output was generated." in str(wrapped)
    
    # Test with CalledProcessError and empty output
    error = subprocess.CalledProcessError(1, "test_command", output=b"")
    wrapped = error_wrapper(error)
    assert "No output was generated." in str(wrapped)
    
    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired("test_command", 10, output=b"timeout output")
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped)
    assert "timeout output" in str(wrapped)
    
    # Test with TimeoutExpired and no output
    error = subprocess.TimeoutExpired("test_command", 10, output=None)
    wrapped = error_wrapper(error)
    assert "No output was generated." in str(wrapped)
    
    # Test with other exception types (should return unchanged)
    original_error = ValueError("test error")
    wrapped = error_wrapper(original_error)
    assert wrapped is original_error
    assert str(wrapped) == "test error"
    
    # Test with CalledProcessError and Unicode decode error simulation
    error = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped = error_wrapper(error)
    assert "Failed to parse output." in str(wrapped)
    
    # Test that output is properly formatted with indentation
    error = subprocess.CalledProcessError(1, "test_command", output=b"line1\nline2\nline3")
    wrapped = error_wrapper(error)
    assert "    line1" in str(wrapped)
    assert "    line2" in str(wrapped)
    assert "    line3" in str(wrapped)


# LLM-generated content at query #16
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError with output
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped)
    assert "test output" in str(wrapped)
    
    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command", output=None)
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped)
    
    # Test with TimeoutExpired with output
    err = subprocess.TimeoutExpired("test_command", 10, output=b"timeout output")
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped)
    assert "timeout output" in str(wrapped)
    
    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", 10, output=None)
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped)
    
    # Test with other exception types (should return unchanged)
    original_err = ValueError("test error")
    wrapped = error_wrapper(original_err)
    assert wrapped is original_err
    assert str(wrapped) == "test error"
    
    # Test that output decoding error is handled gracefully
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")  # Invalid UTF-8
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped)
    
    # Test that __str__ method is properly added to exception class
    err = subprocess.CalledProcessError(1, "test_command", output=b"line1\nline2")
    wrapped = error_wrapper(err)
    str_repr = str(wrapped)
    assert "    line1" in str_repr
    assert "    line2" in str_repr


# LLM-generated content at query #17
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert wrapped.returncode == 1
    assert wrapped.output == b"test output"
    assert "Captured output:" in str(wrapped)
    assert "    test output" in str(wrapped)
    
    # Test with CalledProcessError and no output
    error = subprocess.CalledProcessError(1, "test_command", output=None)
    wrapped = error_wrapper(error)
    assert "No output was generated." in str(wrapped)
    
    # Test with CalledProcessError and empty output
    error = subprocess.CalledProcessError(1, "test_command", output=b"")
    wrapped = error_wrapper(error)
    assert "No output was generated." in str(wrapped)
    
    # Test with CalledProcessError and multi-line output
    error = subprocess.CalledProcessError(1, "test_command", output=b"line1\nline2\nline3")
    wrapped = error_wrapper(error)
    assert "    line1" in str(wrapped)
    assert "    line2" in str(wrapped)
    assert "    line3" in str(wrapped)
    
    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired("test_command", 10, output=b"timeout output")
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert wrapped.timeout == 10
    assert wrapped.output == b"timeout output"
    assert "Captured output:" in str(wrapped)
    
    # Test with TimeoutExpired and no output
    error = subprocess.TimeoutExpired("test_command", 10, output=None)
    wrapped = error_wrapper(error)
    assert "No output was generated." in str(wrapped)
    
    # Test with non-subprocess exception (should return unchanged)
    original_error = ValueError("test error")
    wrapped = error_wrapper(original_error)
    assert wrapped is original_error
    assert str(wrapped) == "test error"
    
    # Test with Unicode output
    error = subprocess.CalledProcessError(1, "test_command", output="test output".encode('utf-8'))
    wrapped = error_wrapper(error)
    assert "test output" in str(wrapped)
    
    # Test that __str__ method is properly added to exception class
    error = subprocess.CalledProcessError(1, "test_command", output=b"test")
    wrapped = error_wrapper(error)
    assert hasattr(wrapped, '__str__')
    assert callable(wrapped.__str__)


# LLM-generated content at query #18
#--------------------------

```python
def test_run_command():
    # Test successful command without output
    result = run_command(["echo", "test"], return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None
    assert result.command == ["echo", "test"]

    # Test successful command with return_output=True
    result = run_command(["echo", "test"], return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output
    assert result.command == ["echo", "test"]

    # Test command with non-zero exit code
    result = run_command(["false"], return_output=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test command with shell=True
    result = run_command("echo shell_test", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"shell_test" in result.captured_output

    # Test timeout handling
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True, return_output=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

    # Test timeout without ignore_errors raises exception
    import pytest
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "2"], timeout=0.1)

    # Test CalledProcessError without ignore_errors
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"])

    # Test CalledProcessError with ignore_errors
    result = run_command(["false"], ignore_errors=True, return_output=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test verbose mode
    result = run_command(["echo", "verbose_test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"verbose_test" in result.captured_output

    # Test with cwd parameter
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output

    # Test with env parameter
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=test_value" in result.captured_output

    # Test output truncation for long output
    long_output = "x" * 10000
    result = run_command(["echo", long_output], return_output=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output

    # Test error_wrapper integration
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert "Captured output:" in str(e) or "No output was generated." in str(e)

    # Test successful command with verbose=False and return_output=False
    result = run_command(["true"])
    assert result.return_code == 0
    assert result.captured_output is None


# LLM-generated content at query #19
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError with output
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"test output\nline2")
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped)
    assert "test output" in str(wrapped)
    assert "line2" in str(wrapped)
    
    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_cmd", output=None)
    wrapped = error_wrapper(err)
    assert "No output was generated." in str(wrapped)
    
    # Test with CalledProcessError with empty output
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"")
    wrapped = error_wrapper(err)
    assert "No output was generated." in str(wrapped)
    
    # Test with TimeoutExpired with output
    err = subprocess.TimeoutExpired("test_cmd", 10, output=b"timeout output")
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped)
    assert "timeout output" in str(wrapped)
    
    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_cmd", 10, output=None)
    wrapped = error_wrapper(err)
    assert "No output was generated." in str(wrapped)
    
    # Test with other exception types (should return unchanged)
    original_exc = ValueError("test error")
    wrapped = error_wrapper(original_exc)
    assert wrapped is original_exc
    assert str(wrapped) == "test error"
    
    # Test with Unicode decode error in output
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"\xff\xfe")
    wrapped = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped)
    
    # Verify the wrapped exception preserves original attributes
    err = subprocess.CalledProcessError(2, ["cmd", "arg"], output=b"output")
    wrapped = error_wrapper(err)
    assert wrapped.returncode == 2
    assert wrapped.cmd == ["cmd", "arg"]
    assert wrapped.output == b"output"


# LLM-generated content at query #20
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError with output
    err = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    err.output = b"error output line 1\nerror output line 2"
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped)
    assert "    error output line 1" in str(wrapped)
    assert "    error output line 2" in str(wrapped)
    
    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    err.output = None
    wrapped = error_wrapper(err)
    assert "No output was generated." in str(wrapped)
    
    # Test with CalledProcessError with empty output
    err = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    err.output = b""
    wrapped = error_wrapper(err)
    assert "No output was generated." in str(wrapped)
    
    # Test with CalledProcessError with unicode output
    err = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    err.output = "test output with unicode: é".encode('utf-8')
    wrapped = error_wrapper(err)
    assert "test output with unicode: é" in str(wrapped)
    
    # Test with TimeoutExpired with output
    err = subprocess.TimeoutExpired(cmd="test_cmd", timeout=10)
    err.output = b"timeout output"
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped)
    
    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired(cmd="test_cmd", timeout=10)
    err.output = None
    wrapped = error_wrapper(err)
    assert "No output was generated." in str(wrapped)
    
    # Test with other exception types (should return unchanged)
    original_exc = ValueError("test error")
    result = error_wrapper(original_exc)
    assert result is original_exc
    assert str(result) == "test error"
    
    # Test with CalledProcessError with output that fails to decode
    err = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    # Create invalid utf-8 bytes
    err.output = b'\xff\xfe\x00'
    wrapped = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped)


# LLM-generated content at query #21
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    error.output = b"test output"
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped)
    assert "test output" in str(wrapped)
    
    # Test with CalledProcessError but no output
    error = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    error.output = None
    wrapped = error_wrapper(error)
    assert "No output was generated." in str(wrapped)
    
    # Test with CalledProcessError and empty output
    error = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    error.output = b""
    wrapped = error_wrapper(error)
    assert "No output was generated." in str(wrapped)
    
    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired(cmd="test_cmd", timeout=10)
    error.output = b"timeout output"
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped)
    assert "timeout output" in str(wrapped)
    
    # Test with other exception types (should return unchanged)
    original_exception = ValueError("test")
    wrapped = error_wrapper(original_exception)
    assert wrapped is original_exception
    assert str(wrapped) == "test"
    
    # Test with Unicode decode error in output
    error = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    error.output = b"\xff\xfe"  # Invalid UTF-8
    wrapped = error_wrapper(error)
    assert "Failed to parse output." in str(wrapped)
    
    # Test that output is properly formatted with indentation
    error = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    error.output = b"line1\nline2\nline3"
    wrapped = error_wrapper(error)
    assert "    line1" in str(wrapped)
    assert "    line2" in str(wrapped)
    assert "    line3" in str(wrapped)


# LLM-generated content at query #22
#--------------------------

```python
def test_run_command():
    # Test successful command execution without output
    result = run_command(["echo", "test"], return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None
    
    # Test successful command execution with output
    result = run_command(["echo", "test"], return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output
    
    # Test command with non-zero exit code
    result = run_command(["false"], return_output=True, ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None
    
    # Test command with timeout
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None
    
    # Test verbose mode
    result = run_command(["echo", "verbose_test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"verbose_test" in result.captured_output
    
    # Test with environment variables
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=test_value" in result.captured_output
    
    # Test with working directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output
    
    # Test shell command
    result = run_command("echo shell_test", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"shell_test" in result.captured_output
    
    # Test error propagation without ignore_errors
    import subprocess
    try:
        run_command(["false"])
        assert False, "Should have raised CalledProcessError"
    except subprocess.CalledProcessError:
        pass
    
    # Test output truncation for long output
    long_output = "x" * 10000
    result = run_command(["echo", long_output], return_output=True, ignore_errors=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output
    
    # Test command with list arguments
    result = run_command(["echo", "arg1", "arg2"], return_output=True)
    assert result.return_code == 0
    assert b"arg1 arg2" in result.captured_output


# LLM-generated content at query #23
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError with output
    err = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    err.output = b"Error output line 1\nError output line 2"
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    str_repr = str(wrapped_err)
    assert "Command 'test_cmd' returned non-zero exit status 1." in str_repr
    assert "Captured output:" in str_repr
    assert "    Error output line 1" in str_repr
    assert "    Error output line 2" in str_repr

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(returncode=2, cmd="test_cmd2")
    err.output = None
    wrapped_err = error_wrapper(err)
    str_repr = str(wrapped_err)
    assert "No output was generated." in str_repr

    # Test with CalledProcessError with empty output
    err = subprocess.CalledProcessError(returncode=3, cmd="test_cmd3")
    err.output = b""
    wrapped_err = error_wrapper(err)
    str_repr = str(wrapped_err)
    assert "No output was generated." in str_repr

    # Test with TimeoutExpired with output
    err = subprocess.TimeoutExpired(cmd="test_cmd", timeout=5)
    err.output = b"Timeout output line 1\nTimeout output line 2"
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    str_repr = str(wrapped_err)
    assert "Command 'test_cmd' timed out after 5 seconds" in str_repr
    assert "Captured output:" in str_repr
    assert "    Timeout output line 1" in str_repr
    assert "    Timeout output line 2" in str_repr

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired(cmd="test_cmd2", timeout=10)
    err.output = None
    wrapped_err = error_wrapper(err)
    str_repr = str(wrapped_err)
    assert "No output was generated." in str_repr

    # Test with non-subprocess exception (should return unchanged)
    original_err = ValueError("Test error")
    wrapped_err = error_wrapper(original_err)
    assert wrapped_err is original_err
    assert str(wrapped_err) == "Test error"

    # Test with Unicode decode error in output
    err = subprocess.CalledProcessError(returncode=4, cmd="test_cmd4")
    err.output = b"\xff\xfe\x00\x00"  # Invalid UTF-8 bytes
    wrapped_err = error_wrapper(err)
    str_repr = str(wrapped_err)
    assert "Failed to parse output." in str_repr

    # Test that original exception attributes are preserved
    err = subprocess.CalledProcessError(returncode=5, cmd="test_cmd5")
    err.output = b"Some output"
    wrapped_err = error_wrapper(err)
    assert wrapped_err.returncode == 5
    assert wrapped_err.cmd == "test_cmd5"
    assert wrapped_err.output == b"Some output"

    # Test with TimeoutExpired attributes preserved
    err = subprocess.TimeoutExpired(cmd="test_cmd6", timeout=15)
    err.output = b"Timeout output"
    wrapped_err = error_wrapper(err)
    assert wrapped_err.cmd == "test_cmd6"
    assert wrapped_err.timeout == 15
    assert wrapped_err.output == b"Timeout output"


# LLM-generated content at query #24
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    error.output = b"test output"
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped)
    assert "test output" in str(wrapped)
    
    # Test with CalledProcessError but no output
    error = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    error.output = None
    wrapped = error_wrapper(error)
    assert "No output was generated." in str(wrapped)
    
    # Test with CalledProcessError and empty output
    error = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    error.output = b""
    wrapped = error_wrapper(error)
    assert "No output was generated." in str(wrapped)
    
    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired(cmd="test_cmd", timeout=10)
    error.output = b"timeout output"
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped)
    assert "timeout output" in str(wrapped)
    
    # Test with other exception types (should return unchanged)
    original_exception = ValueError("test")
    wrapped = error_wrapper(original_exception)
    assert wrapped is original_exception
    assert str(wrapped) == "test"
    
    # Test with Unicode decode error in output
    error = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    error.output = b"\xff\xfe"  # Invalid UTF-8
    wrapped = error_wrapper(error)
    assert "Failed to parse output." in str(wrapped)
    
    # Test that output is properly indented
    error = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    error.output = b"line1\nline2\nline3"
    wrapped = error_wrapper(error)
    assert "    line1" in str(wrapped)
    assert "    line2" in str(wrapped)
    assert "    line3" in str(wrapped)


# LLM-generated content at query #25
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError with output
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"error output")
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped)
    assert "error output" in str(wrapped)
    
    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_cmd", output=None)
    wrapped = error_wrapper(err)
    assert "No output was generated." in str(wrapped)
    
    # Test with CalledProcessError with empty output
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"")
    wrapped = error_wrapper(err)
    assert "No output was generated." in str(wrapped)
    
    # Test with TimeoutExpired with output
    err = subprocess.TimeoutExpired("test_cmd", 10, output=b"timeout output")
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped)
    assert "timeout output" in str(wrapped)
    
    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_cmd", 10, output=None)
    wrapped = error_wrapper(err)
    assert "No output was generated." in str(wrapped)
    
    # Test with other exception types (should return unchanged)
    original_exc = ValueError("test error")
    wrapped = error_wrapper(original_exc)
    assert wrapped is original_exc
    assert str(wrapped) == "test error"
    
    # Test with CalledProcessError with unicode decode error
    err = subprocess.CalledProcessError(1, "test_cmd", output=b'\xff\xfe')
    wrapped = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped)
    
    # Test that output lines are indented
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"line1\nline2\nline3")
    wrapped = error_wrapper(err)
    assert "    line1" in str(wrapped)
    assert "    line2" in str(wrapped)
    assert "    line3" in str(wrapped)


# LLM-generated content at query #26
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(returncode=1, cmd="test_command")
    error.output = b"Error output line 1\nError output line 2"
    wrapped_error = error_wrapper(error)
    error_str = str(wrapped_error)
    assert "Command 'test_command' returned non-zero exit status 1." in error_str
    assert "Captured output:" in error_str
    assert "    Error output line 1" in error_str
    assert "    Error output line 2" in error_str
    
    # Test with TimeoutExpired
    timeout_error = subprocess.TimeoutExpired(cmd="test_command", timeout=5)
    timeout_error.output = b"Timeout output"
    wrapped_timeout = error_wrapper(timeout_error)
    timeout_str = str(wrapped_timeout)
    assert "Command 'test_command' timed out after 5 seconds" in timeout_str
    assert "Captured output:" in timeout_str
    assert "    Timeout output" in timeout_str
    
    # Test with no output
    error_no_output = subprocess.CalledProcessError(returncode=2, cmd="test_command")
    error_no_output.output = None
    wrapped_no_output = error_wrapper(error_no_output)
    no_output_str = str(wrapped_no_output)
    assert "No output was generated." in no_output_str
    
    # Test with empty output
    error_empty = subprocess.CalledProcessError(returncode=3, cmd="test_command")
    error_empty.output = b""
    wrapped_empty = error_wrapper(error_empty)
    empty_str = str(wrapped_empty)
    assert "No output was generated." in empty_str
    
    # Test with non-subprocess exception (should return unchanged)
    regular_error = ValueError("Regular error")
    wrapped_regular = error_wrapper(regular_error)
    assert wrapped_regular is regular_error
    assert str(wrapped_regular) == "Regular error"
    
    # Test with Unicode decode error in output
    error_binary = subprocess.CalledProcessError(returncode=4, cmd="test_command")
    # Create bytes that can't be decoded as UTF-8
    error_binary.output = b'\xff\xfe\x00\x00'
    wrapped_binary = error_wrapper(error_binary)
    binary_str = str(wrapped_binary)
    assert "Failed to parse output." in binary_str
    
    # Test that output is truncated properly in __str__ method
    error_long = subprocess.CalledProcessError(returncode=5, cmd="test_command")
    long_output = b"Line 1\n" * 100
    error_long.output = long_output
    wrapped_long = error_wrapper(error_long)
    long_str = str(wrapped_long)
    assert "Captured output:" in long_str
    # Check that all lines are properly indented
    lines = long_str.split("\n")
    output_lines = [line for line in lines if line.strip() and "Captured output:" not in line 
                   and "Command '" not in line and "returned non-zero" not in line]
    for line in output_lines:
        if line:  # Skip empty lines
            assert line.startswith("    ")


# LLM-generated content at query #27
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(returncode=1, cmd="test_command")
    error.output = b"test output line 1\ntest output line 2"
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped)
    assert "    test output line 1" in str(wrapped)
    assert "    test output line 2" in str(wrapped)
    
    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired(cmd="test_command", timeout=5)
    error.output = b"timeout output"
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped)
    assert "    timeout output" in str(wrapped)
    
    # Test with no output
    error = subprocess.CalledProcessError(returncode=1, cmd="test_command")
    error.output = None
    wrapped = error_wrapper(error)
    assert "No output was generated." in str(wrapped)
    
    # Test with empty output
    error = subprocess.CalledProcessError(returncode=1, cmd="test_command")
    error.output = b""
    wrapped = error_wrapper(error)
    assert "No output was generated." in str(wrapped)
    
    # Test with non-subprocess exception (should return unchanged)
    original_error = ValueError("test error")
    wrapped = error_wrapper(original_error)
    assert wrapped is original_error
    assert str(wrapped) == "test error"
    
    # Test with Unicode decode error in output
    error = subprocess.CalledProcessError(returncode=1, cmd="test_command")
    error.output = b"\xff\xfe"  # Invalid UTF-8
    wrapped = error_wrapper(error)
    assert "Failed to parse output." in str(wrapped)
    
    # Test that original attributes are preserved
    error = subprocess.CalledProcessError(returncode=42, cmd="test_cmd")
    error.output = b"some output"
    wrapped = error_wrapper(error)
    assert wrapped.returncode == 42
    assert wrapped.cmd == "test_cmd"
    assert wrapped.output == b"some output"


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_run_command():
    # Test 1: Successful command without return_output
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None
    assert result.command == ["echo", "hello"]

    # Test 2: Successful command with return_output=True
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output
    assert result.command == ["echo", "hello"]

    # Test 3: Failed command (non-zero return code)
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None
    assert result.command == ["false"]

    # Test 4: Failed command with exception
    import subprocess
    try:
        run_command(["false"])
        assert False, "Should have raised CalledProcessError"
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
        assert hasattr(e, 'output')

    # Test 5: Command with timeout
    try:
        run_command(["sleep", "2"], timeout=0.1)
        assert False, "Should have raised TimeoutExpired"
    except subprocess.TimeoutExpired as e:
        assert hasattr(e, 'output')

    # Test 6: Command with timeout and ignore_errors
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

    # Test 7: Command with cwd parameter
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert tmpdir.encode() in result.captured_output

    # Test 8: Command with env parameter
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"TEST_VAR=test_value" in result.captured_output

    # Test 9: Command with shell=True
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

    # Test 10: Verbose mode (no exception, just check it runs)
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output

    # Test 11: Long output truncation
    long_output = "x" * 10000
    result = run_command(["echo", long_output], return_output=True, ignore_errors=True)
    assert result.captured_output is not None
    assert b"*** (previous output truncated) ***" in result.captured_output

    # Test 12: Command with string args
    result = run_command("echo test", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output

    # Test 13: Check error_wrapper integration
    try:
        run_command(["false"])
        assert False, "Should have raised CalledProcessError"
    except subprocess.CalledProcessError as e:
        str_repr = str(e)
        assert "Captured output:" in str_repr or "No output was generated." in str_repr

    # Test 14: Successful command with non-zero return code but no exception
    result = run_command(["false"], ignore_errors=True, return_output=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test 15: Command with kwargs passed to subprocess.run
    result = run_command(["echo", "hello"], return_output=True, stderr=subprocess.STDOUT)
    assert result.return_code == 0
    assert b"hello" in result.captured_output


# LLM-generated content at query #2
#--------------------------

```python
def test_run_command():
    # Test successful command execution without output
    result = run_command(["echo", "test"], return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None
    
    # Test successful command execution with output
    result = run_command(["echo", "test"], return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output
    
    # Test command with non-zero exit code
    result = run_command(["false"], return_output=True)
    assert result.return_code != 0
    assert result.captured_output is not None
    
    # Test command with timeout
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None
    
    # Test verbose mode
    result = run_command(["echo", "verbose_test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"verbose_test" in result.captured_output
    
    # Test with environment variables
    result = run_command(["printenv", "TEST_VAR"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"test_value" in result.captured_output
    
    # Test with working directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output
    
    # Test shell command
    result = run_command("echo shell_test", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"shell_test" in result.captured_output
    
    # Test error propagation without ignore_errors
    import subprocess
    try:
        run_command(["false"])
        assert False, "Should have raised CalledProcessError"
    except subprocess.CalledProcessError:
        pass
    
    # Test error wrapping with output
    try:
        run_command(["ls", "/nonexistent"])
        assert False, "Should have raised CalledProcessError"
    except subprocess.CalledProcessError as e:
        assert "Captured output:" in str(e)
    
    # Test long output truncation
    long_output = "x" * 10000
    result = run_command(["echo", long_output], return_output=True, ignore_errors=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output
    
    # Test command as list vs string
    result_list = run_command(["echo", "test"], return_output=True)
    result_str = run_command("echo test", shell=True, return_output=True)
    assert result_list.return_code == 0
    assert result_str.return_code == 0
    assert b"test" in result_list.captured_output
    assert b"test" in result_str.captured_output


# LLM-generated content at query #3
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(returncode=1, cmd="test_command")
    error.output = b"Error output line 1\nError output line 2"
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_error)
    assert "Error output line 1" in str(wrapped_error)
    assert "Error output line 2" in str(wrapped_error)
    
    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired(cmd="test_command", timeout=5)
    error.output = b"Timeout output"
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_error)
    assert "Timeout output" in str(wrapped_error)
    
    # Test with no output
    error = subprocess.CalledProcessError(returncode=1, cmd="test_command")
    error.output = None
    wrapped_error = error_wrapper(error)
    assert "No output was generated." in str(wrapped_error)
    
    # Test with empty output
    error = subprocess.CalledProcessError(returncode=1, cmd="test_command")
    error.output = b""
    wrapped_error = error_wrapper(error)
    assert "No output was generated." in str(wrapped_error)
    
    # Test with non-subprocess exception (should return unchanged)
    original_error = ValueError("Test error")
    wrapped_error = error_wrapper(original_error)
    assert wrapped_error is original_error
    assert str(wrapped_error) == "Test error"
    
    # Test with Unicode decode error in output
    error = subprocess.CalledProcessError(returncode=1, cmd="test_command")
    error.output = b"\xff\xfe"  # Invalid UTF-8
    wrapped_error = error_wrapper(error)
    assert "Failed to parse output." in str(wrapped_error)
    
    # Test output formatting with indentation
    error = subprocess.CalledProcessError(returncode=1, cmd="test_command")
    error.output = b"Line1\nLine2\nLine3"
    wrapped_error = error_wrapper(error)
    str_repr = str(wrapped_error)
    assert "    Line1" in str_repr
    assert "    Line2" in str_repr
    assert "    Line3" in str_repr


# LLM-generated content at query #4
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    error.output = b"test output"
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped)
    assert "test output" in str(wrapped)
    
    # Test with CalledProcessError without output
    error2 = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    error2.output = None
    wrapped2 = error_wrapper(error2)
    assert "No output was generated." in str(wrapped2)
    
    # Test with TimeoutExpired
    error3 = subprocess.TimeoutExpired(cmd="test_cmd", timeout=10)
    error3.output = b"timeout output"
    wrapped3 = error_wrapper(error3)
    assert isinstance(wrapped3, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped3)
    assert "timeout output" in str(wrapped3)
    
    # Test with non-subprocess exception (should return unchanged)
    regular_error = ValueError("test error")
    wrapped4 = error_wrapper(regular_error)
    assert wrapped4 is regular_error
    assert str(wrapped4) == "test error"
    
    # Test with Unicode output
    error5 = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    error5.output = b"test \xe9 output"  # non-UTF8 bytes
    wrapped5 = error_wrapper(error5)
    assert "Failed to parse output." in str(wrapped5)
    
    # Test that original attributes are preserved
    error6 = subprocess.CalledProcessError(returncode=42, cmd="special_cmd")
    error6.output = b"output"
    wrapped6 = error_wrapper(error6)
    assert wrapped6.returncode == 42
    assert wrapped6.cmd == "special_cmd"
    assert wrapped6.output == b"output"


# LLM-generated content at query #5
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(returncode=1, cmd="test_command")
    error.output = b"Test output line 1\nTest output line 2"
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    error_str = str(wrapped_error)
    assert "Command 'test_command' returned non-zero exit status 1." in error_str
    assert "Captured output:" in error_str
    assert "    Test output line 1" in error_str
    assert "    Test output line 2" in error_str

    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired(cmd="test_command", timeout=5)
    error.output = b"Timeout output"
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    error_str = str(wrapped_error)
    assert "Command 'test_command' timed out after 5 seconds" in error_str
    assert "Captured output:" in error_str
    assert "    Timeout output" in error_str

    # Test with no output
    error = subprocess.CalledProcessError(returncode=2, cmd="test_command")
    error.output = None
    wrapped_error = error_wrapper(error)
    error_str = str(wrapped_error)
    assert "No output was generated." in error_str

    # Test with empty output
    error = subprocess.CalledProcessError(returncode=3, cmd="test_command")
    error.output = b""
    wrapped_error = error_wrapper(error)
    error_str = str(wrapped_error)
    assert "No output was generated." in error_str

    # Test with non-subprocess exception (should return unchanged)
    original_error = ValueError("Test error")
    wrapped_error = error_wrapper(original_error)
    assert wrapped_error is original_error
    assert str(wrapped_error) == "Test error"

    # Test with Unicode decode error in output
    error = subprocess.CalledProcessError(returncode=4, cmd="test_command")
    error.output = b"\xff\xfe"  # Invalid UTF-8
    wrapped_error = error_wrapper(error)
    error_str = str(wrapped_error)
    assert "Failed to parse output." in error_str

    # Test that original exception attributes are preserved
    error = subprocess.CalledProcessError(returncode=5, cmd="test_command")
    error.output = b"Some output"
    wrapped_error = error_wrapper(error)
    assert wrapped_error.returncode == 5
    assert wrapped_error.cmd == "test_command"
    assert wrapped_error.output == b"Some output"

    # Test with multi-line output
    error = subprocess.CalledProcessError(returncode=6, cmd="test_command")
    error.output = b"Line 1\nLine 2\nLine 3"
    wrapped_error = error_wrapper(error)
    error_str = str(wrapped_error)
    assert "    Line 1" in error_str
    assert "    Line 2" in error_str
    assert "    Line 3" in error_str


# LLM-generated content at query #6
#--------------------------

```python
def test_run_command():
    # Test successful command execution without output
    result = run_command(["echo", "test"], return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None
    
    # Test successful command execution with output
    result = run_command(["echo", "test"], return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output
    
    # Test command with non-zero exit code
    result = run_command(["false"], return_output=True, ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None
    
    # Test command timeout with ignore_errors
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None
    
    # Test verbose mode
    result = run_command(["echo", "verbose_test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"verbose_test" in result.captured_output
    
    # Test with environment variables
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=test_value" in result.captured_output
    
    # Test with working directory
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output
    
    # Test shell command
    result = run_command("echo shell_test", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"shell_test" in result.captured_output
    
    # Test command with long output truncation
    long_output = "x" * 10000
    result = run_command(["echo", long_output], return_output=True, ignore_errors=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output
    
    # Test error propagation without ignore_errors
    import subprocess
    try:
        run_command(["false"])
        assert False, "Should have raised CalledProcessError"
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
        assert hasattr(e, 'output')
    
    # Test timeout propagation without ignore_errors
    try:
        run_command(["sleep", "2"], timeout=0.1)
        assert False, "Should have raised TimeoutExpired"
    except subprocess.TimeoutExpired as e:
        assert hasattr(e, 'output')
    
    # Test that error_wrapper modifies exception string representation
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        str_repr = str(e)
        assert "Captured output:" in str_repr or "No output was generated." in str_repr


# LLM-generated content at query #7
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(returncode=1, cmd="test_command")
    error.output = b"test output"
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped)
    assert "test output" in str(wrapped)
    
    # Test with CalledProcessError but no output
    error2 = subprocess.CalledProcessError(returncode=2, cmd="test_command2")
    error2.output = None
    wrapped2 = error_wrapper(error2)
    assert "No output was generated." in str(wrapped2)
    
    # Test with TimeoutExpired
    error3 = subprocess.TimeoutExpired(cmd="test_command3", timeout=5)
    error3.output = b"timeout output"
    wrapped3 = error_wrapper(error3)
    assert isinstance(wrapped3, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped3)
    
    # Test with other exception types (should return unchanged)
    value_error = ValueError("test error")
    wrapped4 = error_wrapper(value_error)
    assert wrapped4 is value_error
    assert str(wrapped4) == "test error"
    
    # Test with Unicode decode error in output
    error5 = subprocess.CalledProcessError(returncode=3, cmd="test_command5")
    error5.output = b'\xff\xfe'  # Invalid UTF-8
    wrapped5 = error_wrapper(error5)
    assert "Failed to parse output." in str(wrapped5)
    
    # Test that original attributes are preserved
    error6 = subprocess.CalledProcessError(returncode=4, cmd="test_command6")
    error6.output = b"some output"
    wrapped6 = error_wrapper(error6)
    assert wrapped6.returncode == 4
    assert wrapped6.cmd == "test_command6"
    assert wrapped6.output == b"some output"


# LLM-generated content at query #8
#--------------------------

```python
def test_run_command():
    # Test successful command execution without output
    result = run_command(["echo", "test"], return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None
    assert result.command == ["echo", "test"]

    # Test successful command execution with output
    result = run_command(["echo", "test"], return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output
    assert result.command == ["echo", "test"]

    # Test command with non-zero return code
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test command with timeout
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

    # Test verbose mode
    result = run_command(["echo", "verbose_test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"verbose_test" in result.captured_output

    # Test with custom environment
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=test_value" in result.captured_output

    # Test with custom working directory
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output

    # Test shell command
    result = run_command("echo shell_test", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"shell_test" in result.captured_output

    # Test error propagation without ignore_errors
    import subprocess
    try:
        run_command(["false"])
        assert False, "Should have raised CalledProcessError"
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
        assert hasattr(e, 'output')

    # Test timeout propagation without ignore_errors
    try:
        run_command(["sleep", "2"], timeout=0.1)
        assert False, "Should have raised TimeoutExpired"
    except subprocess.TimeoutExpired as e:
        assert hasattr(e, 'output')

    # Test long output truncation
    long_text = "x" * 10000
    result = run_command(["echo", long_text], ignore_errors=True, return_output=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output
    assert len(result.captured_output) < 10000

    # Test command with stderr redirection
    result = run_command(["ls", "/nonexistent"], ignore_errors=True, return_output=True)
    assert result.return_code != 0
    assert result.captured_output is not None
    assert len(result.captured_output) > 0


# LLM-generated content at query #9
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(returncode=1, cmd="test_command")
    error.output = b"test output\nline 2"
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped)
    assert "test output" in str(wrapped)
    assert "line 2" in str(wrapped)

    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired(cmd="test_command", timeout=5)
    error.output = b"timeout output"
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped)
    assert "timeout output" in str(wrapped)

    # Test with empty output
    error = subprocess.CalledProcessError(returncode=1, cmd="test_command")
    error.output = b""
    wrapped = error_wrapper(error)
    assert "No output was generated." in str(wrapped)

    # Test with None output
    error = subprocess.CalledProcessError(returncode=1, cmd="test_command")
    error.output = None
    wrapped = error_wrapper(error)
    assert "No output was generated." in str(wrapped)

    # Test with unicode decode error
    error = subprocess.CalledProcessError(returncode=1, cmd="test_command")
    error.output = b"\xff\xfe"  # Invalid UTF-8
    wrapped = error_wrapper(error)
    assert "Failed to parse output." in str(wrapped)

    # Test with other exception types (should return unchanged)
    original_exception = ValueError("test error")
    wrapped = error_wrapper(original_exception)
    assert wrapped is original_exception
    assert str(wrapped) == "test error"

    # Test with long output
    error = subprocess.CalledProcessError(returncode=1, cmd="test_command")
    error.output = b"line1\n" * 100
    wrapped = error_wrapper(error)
    str_repr = str(wrapped)
    assert "Captured output:" in str_repr
    assert "line1" in str_repr
    assert str_repr.count("line1") == 100

    # Test that __str__ method is properly added
    error = subprocess.CalledProcessError(returncode=1, cmd="test_command")
    error.output = b"test"
    wrapped = error_wrapper(error)
    assert hasattr(wrapped, "__str__")
    assert callable(wrapped.__str__)


# LLM-generated content at query #10
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError with output
    error = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    error.output = b"Error output line 1\nError output line 2"
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped)
    assert "    Error output line 1" in str(wrapped)
    assert "    Error output line 2" in str(wrapped)
    
    # Test with CalledProcessError without output
    error = subprocess.CalledProcessError(returncode=2, cmd="test_cmd")
    error.output = None
    wrapped = error_wrapper(error)
    assert "No output was generated." in str(wrapped)
    
    # Test with CalledProcessError with empty output
    error = subprocess.CalledProcessError(returncode=3, cmd="test_cmd")
    error.output = b""
    wrapped = error_wrapper(error)
    assert "No output was generated." in str(wrapped)
    
    # Test with TimeoutExpired with output
    error = subprocess.TimeoutExpired(cmd="test_cmd", timeout=10)
    error.output = b"Timeout output"
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped)
    
    # Test with TimeoutExpired without output
    error = subprocess.TimeoutExpired(cmd="test_cmd", timeout=10)
    error.output = None
    wrapped = error_wrapper(error)
    assert "No output was generated." in str(wrapped)
    
    # Test with non-subprocess exception (should return unchanged)
    original_error = ValueError("Test error")
    wrapped = error_wrapper(original_error)
    assert wrapped is original_error
    assert str(wrapped) == "Test error"
    
    # Test with Unicode decode error in output
    error = subprocess.CalledProcessError(returncode=4, cmd="test_cmd")
    error.output = b"\xff\xfe"  # Invalid UTF-8
    wrapped = error_wrapper(error)
    assert "Failed to parse output." in str(wrapped)
    
    # Test that original error attributes are preserved
    error = subprocess.CalledProcessError(returncode=5, cmd="original_cmd")
    error.output = b"Test output"
    wrapped = error_wrapper(error)
    assert wrapped.returncode == 5
    assert wrapped.cmd == "original_cmd"
    assert wrapped.output == b"Test output"


# LLM-generated content at query #11
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(returncode=1, cmd="test_command")
    error.output = b"test output"
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped)
    assert "test output" in str(wrapped)
    
    # Test with CalledProcessError and no output
    error = subprocess.CalledProcessError(returncode=1, cmd="test_command")
    error.output = None
    wrapped = error_wrapper(error)
    assert "No output was generated." in str(wrapped)
    
    # Test with CalledProcessError and empty output
    error = subprocess.CalledProcessError(returncode=1, cmd="test_command")
    error.output = b""
    wrapped = error_wrapper(error)
    assert "No output was generated." in str(wrapped)
    
    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired(cmd="test_command", timeout=10)
    error.output = b"timeout output"
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped)
    assert "timeout output" in str(wrapped)
    
    # Test with non-subprocess exception
    original_error = ValueError("test error")
    wrapped = error_wrapper(original_error)
    assert wrapped is original_error
    assert str(wrapped) == "test error"
    
    # Test with Unicode decode error in output
    error = subprocess.CalledProcessError(returncode=1, cmd="test_command")
    error.output = b"\xff\xfe"  # Invalid UTF-8
    wrapped = error_wrapper(error)
    assert "Failed to parse output." in str(wrapped)
    
    # Test that original attributes are preserved
    error = subprocess.CalledProcessError(returncode=42, cmd="special_cmd")
    error.output = b"error details"
    wrapped = error_wrapper(error)
    assert wrapped.returncode == 42
    assert wrapped.cmd == "special_cmd"
    assert wrapped.output == b"error details"


# LLM-generated content at query #12
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError with output
    error = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped)
    assert "error output" in str(wrapped)
    
    # Test with CalledProcessError without output
    error = subprocess.CalledProcessError(1, "test_command", output=None)
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped)
    
    # Test with CalledProcessError with empty output
    error = subprocess.CalledProcessError(1, "test_command", output=b"")
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped)
    
    # Test with TimeoutExpired with output
    error = subprocess.TimeoutExpired("test_command", 10, output=b"timeout output")
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped)
    assert "timeout output" in str(wrapped)
    
    # Test with TimeoutExpired without output
    error = subprocess.TimeoutExpired("test_command", 10, output=None)
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped)
    
    # Test with other exception types (should return unchanged)
    original_exception = ValueError("test error")
    wrapped = error_wrapper(original_exception)
    assert wrapped is original_exception
    assert str(wrapped) == "test error"
    
    # Test with CalledProcessError with unicode decode error
    error = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped)
    
    # Test that output is properly formatted with indentation
    error = subprocess.CalledProcessError(1, "test_command", output=b"line1\nline2\nline3")
    wrapped = error_wrapper(error)
    str_repr = str(wrapped)
    assert "    line1" in str_repr
    assert "    line2" in str_repr
    assert "    line3" in str_repr


# LLM-generated content at query #13
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped)
    assert "test output" in str(wrapped)
    
    # Test with CalledProcessError but no output
    error = subprocess.CalledProcessError(1, "test_command", output=None)
    wrapped = error_wrapper(error)
    assert "No output was generated." in str(wrapped)
    
    # Test with CalledProcessError and empty output
    error = subprocess.CalledProcessError(1, "test_command", output=b"")
    wrapped = error_wrapper(error)
    assert "No output was generated." in str(wrapped)
    
    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired("test_command", 10, output=b"timeout output")
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped)
    assert "timeout output" in str(wrapped)
    
    # Test with TimeoutExpired but no output
    error = subprocess.TimeoutExpired("test_command", 10, output=None)
    wrapped = error_wrapper(error)
    assert "No output was generated." in str(wrapped)
    
    # Test with other exception types (should return unchanged)
    original_error = ValueError("test error")
    wrapped = error_wrapper(original_error)
    assert wrapped is original_error
    assert str(wrapped) == "test error"
    
    # Test with CalledProcessError and Unicode decode error in output
    error = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped = error_wrapper(error)
    assert "Failed to parse output." in str(wrapped)
    
    # Test that output is properly formatted with indentation
    error = subprocess.CalledProcessError(1, "test_command", output=b"line1\nline2\nline3")
    wrapped = error_wrapper(error)
    assert "    line1" in str(wrapped)
    assert "    line2" in str(wrapped)
    assert "    line3" in str(wrapped)
    
    # Test that original error attributes are preserved
    error = subprocess.CalledProcessError(42, "my_command", output=b"error details")
    wrapped = error_wrapper(error)
    assert wrapped.returncode == 42
    assert wrapped.cmd == "my_command"
    assert wrapped.output == b"error details"


# LLM-generated content at query #14
#--------------------------

```python
def test_run_command():
    # Test successful command execution without output
    result = run_command(["echo", "test"], return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None
    
    # Test successful command execution with output
    result = run_command(["echo", "test"], return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output
    
    # Test command with non-zero return code
    result = run_command(["false"], return_output=True, ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None
    
    # Test command with verbose output
    result = run_command(["echo", "verbose_test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"verbose_test" in result.captured_output
    
    # Test command with environment variables
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=test_value" in result.captured_output
    
    # Test command with working directory
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output
    
    # Test command timeout
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None
    
    # Test command with shell
    result = run_command("echo shell_test", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"shell_test" in result.captured_output
    
    # Test command with string arguments
    result = run_command("echo string_test", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"string_test" in result.captured_output
    
    # Test error propagation without ignore_errors
    import subprocess
    try:
        run_command(["false"])
        assert False, "Should have raised CalledProcessError"
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
    
    # Test output truncation for long output
    long_text = "x" * 10000
    result = run_command(["echo", long_text], return_output=True, ignore_errors=True)
    assert result.return_code == 0
    assert b"truncated" in result.captured_output
    assert len(result.captured_output) < 10000
    
    # Test that output is captured on error
    try:
        run_command(["ls", "/nonexistent"])
        assert False, "Should have raised CalledProcessError"
    except subprocess.CalledProcessError as e:
        assert hasattr(e, 'output')
        assert b"No such file" in e.output or b"cannot access" in e.output


# LLM-generated content at query #15
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(returncode=1, cmd="test_command")
    error.output = b"Error output line 1\nError output line 2"
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_error)
    assert "Error output line 1" in str(wrapped_error)
    assert "Error output line 2" in str(wrapped_error)
    
    # Test with TimeoutExpired
    timeout_error = subprocess.TimeoutExpired(cmd="test_command", timeout=5)
    timeout_error.output = b"Timeout output"
    wrapped_timeout = error_wrapper(timeout_error)
    assert isinstance(wrapped_timeout, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_timeout)
    assert "Timeout output" in str(wrapped_timeout)
    
    # Test with no output
    error_no_output = subprocess.CalledProcessError(returncode=2, cmd="test_command")
    error_no_output.output = None
    wrapped_no_output = error_wrapper(error_no_output)
    assert "No output was generated." in str(wrapped_no_output)
    
    # Test with empty output
    error_empty = subprocess.CalledProcessError(returncode=3, cmd="test_command")
    error_empty.output = b""
    wrapped_empty = error_wrapper(error_empty)
    assert "No output was generated." in str(wrapped_empty)
    
    # Test with non-subprocess exception (should return unchanged)
    regular_error = ValueError("Regular error")
    wrapped_regular = error_wrapper(regular_error)
    assert wrapped_regular is regular_error
    assert str(wrapped_regular) == "Regular error"
    
    # Test with Unicode decode error in output
    error_binary = subprocess.CalledProcessError(returncode=4, cmd="test_command")
    error_binary.output = b"\xff\xfe\xfd"  # Invalid UTF-8
    wrapped_binary = error_wrapper(error_binary)
    assert "Failed to parse output." in str(wrapped_binary)
    
    # Test that original attributes are preserved
    error_with_attrs = subprocess.CalledProcessError(returncode=5, cmd="test_cmd")
    error_with_attrs.output = b"Test output"
    wrapped_with_attrs = error_wrapper(error_with_attrs)
    assert wrapped_with_attrs.returncode == 5
    assert wrapped_with_attrs.cmd == "test_cmd"
    assert wrapped_with_attrs.output == b"Test output"


# LLM-generated content at query #16
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(returncode=1, cmd="test_command")
    error.output = b"test output"
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped)
    assert "test output" in str(wrapped)
    
    # Test with CalledProcessError but no output
    error = subprocess.CalledProcessError(returncode=1, cmd="test_command")
    error.output = None
    wrapped = error_wrapper(error)
    assert "No output was generated." in str(wrapped)
    
    # Test with CalledProcessError and empty output
    error = subprocess.CalledProcessError(returncode=1, cmd="test_command")
    error.output = b""
    wrapped = error_wrapper(error)
    assert "No output was generated." in str(wrapped)
    
    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired(cmd="test_command", timeout=10)
    error.output = b"timeout output"
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped)
    assert "timeout output" in str(wrapped)
    
    # Test with TimeoutExpired but no output
    error = subprocess.TimeoutExpired(cmd="test_command", timeout=10)
    error.output = None
    wrapped = error_wrapper(error)
    assert "No output was generated." in str(wrapped)
    
    # Test with other exception types (should return unchanged)
    original_exception = ValueError("test error")
    wrapped = error_wrapper(original_exception)
    assert wrapped is original_exception
    assert str(wrapped) == "test error"
    
    # Test with CalledProcessError and Unicode decode error
    error = subprocess.CalledProcessError(returncode=1, cmd="test_command")
    # Create bytes that cannot be decoded as UTF-8
    error.output = b'\xff\xfe\x00\x00'
    wrapped = error_wrapper(error)
    assert "Failed to parse output." in str(wrapped)
    
    # Test that __str__ method is properly added
    error = subprocess.CalledProcessError(returncode=1, cmd="test_command")
    error.output = b"line1\nline2\nline3"
    wrapped = error_wrapper(error)
    str_representation = str(wrapped)
    assert "    line1" in str_representation
    assert "    line2" in str_representation
    assert "    line3" in str_representation


# LLM-generated content at query #17
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped)
    assert "test output" in str(wrapped)
    
    # Test with CalledProcessError and no output
    error = subprocess.CalledProcessError(1, "test_command", output=None)
    wrapped = error_wrapper(error)
    assert "No output was generated." in str(wrapped)
    
    # Test with CalledProcessError and empty output
    error = subprocess.CalledProcessError(1, "test_command", output=b"")
    wrapped = error_wrapper(error)
    assert "No output was generated." in str(wrapped)
    
    # Test with CalledProcessError and unicode output
    error = subprocess.CalledProcessError(1, "test_command", output="test output".encode('utf-8'))
    wrapped = error_wrapper(error)
    assert "Captured output:" in str(wrapped)
    assert "test output" in str(wrapped)
    
    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired("test_command", 10, output=b"timeout output")
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped)
    assert "timeout output" in str(wrapped)
    
    # Test with TimeoutExpired and no output
    error = subprocess.TimeoutExpired("test_command", 10, output=None)
    wrapped = error_wrapper(error)
    assert "No output was generated." in str(wrapped)
    
    # Test with other exception types (should return unchanged)
    original_error = ValueError("test error")
    wrapped = error_wrapper(original_error)
    assert wrapped is original_error
    assert str(wrapped) == "test error"
    
    # Test with CalledProcessError and output that fails to decode
    error = subprocess.CalledProcessError(1, "test_command", output=b'\xff\xfe')
    wrapped = error_wrapper(error)
    assert "Failed to parse output." in str(wrapped)
    
    # Test that original attributes are preserved
    error = subprocess.CalledProcessError(42, "test_command", output=b"output")
    wrapped = error_wrapper(error)
    assert wrapped.returncode == 42
    assert wrapped.cmd == "test_command"
    assert wrapped.output == b"output"
    
    # Test with TimeoutExpired timeout attribute preserved
    error = subprocess.TimeoutExpired("test_command", 30, output=b"output")
    wrapped = error_wrapper(error)
    assert wrapped.timeout == 30


# LLM-generated content at query #18
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError with output
    error = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped)
    assert "error output" in str(wrapped)
    
    # Test with CalledProcessError without output
    error = subprocess.CalledProcessError(1, "test_command", output=None)
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped)
    
    # Test with CalledProcessError with empty output
    error = subprocess.CalledProcessError(1, "test_command", output=b"")
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped)
    
    # Test with TimeoutExpired with output
    error = subprocess.TimeoutExpired("test_command", 10, output=b"timeout output")
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped)
    assert "timeout output" in str(wrapped)
    
    # Test with TimeoutExpired without output
    error = subprocess.TimeoutExpired("test_command", 10, output=None)
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped)
    
    # Test with other exception types (should return unchanged)
    original_exception = ValueError("test error")
    wrapped = error_wrapper(original_exception)
    assert wrapped is original_exception
    assert str(wrapped) == "test error"
    
    # Test with CalledProcessError with unicode decode error
    error = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped)
    
    # Test that output is properly formatted with indentation
    error = subprocess.CalledProcessError(1, "test_command", output=b"line1\nline2\nline3")
    wrapped = error_wrapper(error)
    wrapped_str = str(wrapped)
    assert "    line1" in wrapped_str
    assert "    line2" in wrapped_str
    assert "    line3" in wrapped_str


# LLM-generated content at query #19
#--------------------------

```python
def test_run_command():
    # Test successful command execution without output
    result = run_command(["echo", "test"], return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None
    
    # Test successful command execution with return_output=True
    result = run_command(["echo", "test"], return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output
    
    # Test command with non-zero exit code
    result = run_command(["false"], return_output=True, ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None
    
    # Test command with timeout
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None
    
    # Test verbose mode
    result = run_command(["echo", "verbose_test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"verbose_test" in result.captured_output
    
    # Test with environment variables
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=test_value" in result.captured_output
    
    # Test with working directory
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output
    
    # Test shell command with string
    result = run_command("echo shell_test", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"shell_test" in result.captured_output
    
    # Test error propagation without ignore_errors
    try:
        run_command(["false"])
        assert False, "Should have raised CalledProcessError"
    except subprocess.CalledProcessError:
        pass
    
    # Test output truncation for long output
    long_output = "x" * 10000
    result = run_command(["echo", long_output], return_output=True, ignore_errors=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output
    
    # Test that error_wrapper modifies exception string representation
    try:
        run_command(["false"], return_output=True)
    except subprocess.CalledProcessError as e:
        str_repr = str(e)
        assert "Captured output:" in str_repr or "No output was generated." in str_repr


# LLM-generated content at query #20
#--------------------------

```python
def test_run_command():
    # Test successful command execution without output capture
    result = run_command(["echo", "hello"])
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output is None
    
    # Test successful command with return_output=True
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output
    
    # Test command with non-zero exit code
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None
    
    # Test command with timeout
    result = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None
    
    # Test verbose mode
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output
    
    # Test with environment variables
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=test_value" in result.captured_output
    
    # Test with working directory
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output
    
    # Test shell command
    result = run_command("echo shell_test", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"shell_test" in result.captured_output
    
    # Test output truncation for long output
    long_output = "x" * 10000
    result = run_command(["echo", long_output], return_output=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output
    
    # Test error propagation without ignore_errors
    import subprocess
    try:
        run_command(["false"])
        assert False, "Should have raised CalledProcessError"
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
    
    # Test timeout propagation without ignore_errors
    try:
        run_command(["sleep", "10"], timeout=0.1)
        assert False, "Should have raised TimeoutExpired"
    except subprocess.TimeoutExpired:
        pass
    
    # Test that error_wrapper enhances exception messages
    try:
        run_command(["echo", "error"; "&&", "false"], shell=True)
    except subprocess.CalledProcessError as e:
        str_repr = str(e)
        assert "Captured output:" in str_repr or "No output was generated." in str_repr


# LLM-generated content at query #21
#--------------------------

```python
def test_run_command():
    import subprocess
    import tempfile
    import os
    import sys
    
    # Test 1: Successful command without return_output
    result = run_command(["echo", "test"])
    assert result.return_code == 0
    assert result.captured_output is None
    assert result.command == ["echo", "test"]
    
    # Test 2: Successful command with return_output=True
    result = run_command(["echo", "test"], return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output
    assert result.command == ["echo", "test"]
    
    # Test 3: Failed command (non-zero return code)
    result = run_command(["false"], return_output=True, ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None
    
    # Test 4: Failed command raises exception by default
    try:
        run_command(["false"])
        assert False, "Should have raised CalledProcessError"
    except subprocess.CalledProcessError:
        pass
    
    # Test 5: Command with timeout
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768  # Special code for timeout
    assert result.captured_output is not None
    
    # Test 6: Command with env parameter
    env = {"TEST_VAR": "test_value"}
    if sys.platform != "win32":
        result = run_command(["env"], env=env, return_output=True)
        assert b"TEST_VAR=test_value" in result.captured_output
    
    # Test 7: Command with cwd parameter
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert tmpdir.encode() in result.captured_output
    
    # Test 8: Command with shell=True
    result = run_command("echo shell_test", shell=True, return_output=True)
    assert b"shell_test" in result.captured_output
    
    # Test 9: Verbose mode (should not crash)
    result = run_command(["echo", "verbose_test"], verbose=True, return_output=True)
    assert b"verbose_test" in result.captured_output
    
    # Test 10: Long output truncation
    long_text = "x" * 10000
    result = run_command(["echo", long_text], return_output=True, ignore_errors=True)
    assert b"*** (previous output truncated) ***" in result.captured_output
    
    # Test 11: Command with string args (not list)
    result = run_command("echo string_args", shell=True, return_output=True)
    assert b"string_args" in result.captured_output
    
    # Test 12: Check error_wrapper integration
    try:
        run_command(["false"])
        assert False, "Should have raised CalledProcessError"
    except subprocess.CalledProcessError as e:
        assert "Captured output:" in str(e) or "No output was generated." in str(e)
    
    # Test 13: Timeout with error wrapper
    try:
        run_command(["sleep", "10"], timeout=0.01)
        assert False, "Should have raised TimeoutExpired"
    except subprocess.TimeoutExpired as e:
        assert "Captured output:" in str(e) or "No output was generated." in str(e)
    
    # Test 14: Return output when return code is non-zero
    result = run_command(["false"], return_output=True, ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None
    
    # Test 15: No output when return_output=False and successful
    result = run_command(["true"])
    assert result.return_code == 0
    assert result.captured_output is None


# LLM-generated content at query #22
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError with output
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"test output\nline 2")
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped)
    assert "test output" in str(wrapped)
    assert "line 2" in str(wrapped)
    
    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_cmd", output=None)
    wrapped = error_wrapper(err)
    assert "No output was generated." in str(wrapped)
    
    # Test with CalledProcessError with empty output
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"")
    wrapped = error_wrapper(err)
    assert "No output was generated." in str(wrapped)
    
    # Test with TimeoutExpired with output
    err = subprocess.TimeoutExpired("test_cmd", 10, output=b"timeout output")
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped)
    assert "timeout output" in str(wrapped)
    
    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_cmd", 10, output=None)
    wrapped = error_wrapper(err)
    assert "No output was generated." in str(wrapped)
    
    # Test with other exception types (should return unchanged)
    original_err = ValueError("test error")
    wrapped = error_wrapper(original_err)
    assert wrapped is original_err
    assert str(wrapped) == "test error"
    
    # Test with Unicode decode error in output
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"\xff\xfe")
    wrapped = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped)
    
    # Test that original attributes are preserved
    err = subprocess.CalledProcessError(2, ["cmd", "arg"], output=b"output")
    wrapped = error_wrapper(err)
    assert wrapped.returncode == 2
    assert wrapped.cmd == ["cmd", "arg"]
    assert wrapped.output == b"output"


# LLM-generated content at query #23
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError with output
    err = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    err.output = b"error output line 1\nerror output line 2"
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped)
    assert "error output line 1" in str(wrapped)
    assert "error output line 2" in str(wrapped)
    
    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    err.output = None
    wrapped = error_wrapper(err)
    assert "No output was generated." in str(wrapped)
    
    # Test with CalledProcessError with empty output
    err = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    err.output = b""
    wrapped = error_wrapper(err)
    assert "No output was generated." in str(wrapped)
    
    # Test with TimeoutExpired with output
    err = subprocess.TimeoutExpired(cmd="test_cmd", timeout=10)
    err.output = b"timeout output"
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped)
    assert "timeout output" in str(wrapped)
    
    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired(cmd="test_cmd", timeout=10)
    err.output = None
    wrapped = error_wrapper(err)
    assert "No output was generated." in str(wrapped)
    
    # Test with other exception types (should return unchanged)
    original_exception = ValueError("test error")
    wrapped = error_wrapper(original_exception)
    assert wrapped is original_exception
    assert str(wrapped) == "test error"
    
    # Test with Unicode decode error in output
    err = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    err.output = b"\xff\xfe"  # Invalid UTF-8
    wrapped = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped)
    
    # Test that __str__ method is properly added
    err = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    err.output = b"test output"
    wrapped = error_wrapper(err)
    str_repr = str(wrapped)
    assert "test output" in str_repr
    assert "Captured output:" in str_repr
    assert "    test output" in str_repr
    
    # Test output formatting with multiple lines
    err = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    err.output = b"line1\nline2\nline3"
    wrapped = error_wrapper(err)
    str_repr = str(wrapped)
    assert "    line1" in str_repr
    assert "    line2" in str_repr
    assert "    line3" in str_repr


# LLM-generated content at query #24
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError with output
    err = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    err.output = b"Error output line 1\nError output line 2"
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped)
    assert "Error output line 1" in str(wrapped)
    assert "Error output line 2" in str(wrapped)
    
    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    err.output = None
    wrapped = error_wrapper(err)
    assert "No output was generated." in str(wrapped)
    
    # Test with CalledProcessError with empty output
    err = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    err.output = b""
    wrapped = error_wrapper(err)
    assert "No output was generated." in str(wrapped)
    
    # Test with TimeoutExpired with output
    err = subprocess.TimeoutExpired(cmd="test_cmd", timeout=10)
    err.output = b"Timeout output line 1\nTimeout output line 2"
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped)
    assert "Timeout output line 1" in str(wrapped)
    
    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired(cmd="test_cmd", timeout=10)
    err.output = None
    wrapped = error_wrapper(err)
    assert "No output was generated." in str(wrapped)
    
    # Test with non-subprocess exception (should return unchanged)
    original_exc = ValueError("Test error")
    wrapped = error_wrapper(original_exc)
    assert wrapped is original_exc
    assert str(wrapped) == "Test error"
    
    # Test with Unicode decode error in output
    err = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    err.output = b"\xff\xfe"  # Invalid UTF-8
    wrapped = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped)
    
    # Test that original exception attributes are preserved
    err = subprocess.CalledProcessError(returncode=42, cmd="special_cmd")
    err.output = b"Special output"
    wrapped = error_wrapper(err)
    assert wrapped.returncode == 42
    assert wrapped.cmd == "special_cmd"
    assert wrapped.output == b"Special output"


# LLM-generated content at query #25
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(returncode=1, cmd="test_command")
    error.output = b"Test output line 1\nTest output line 2"
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    error_str = str(wrapped_error)
    assert "Command 'test_command' returned non-zero exit status 1." in error_str
    assert "Captured output:" in error_str
    assert "    Test output line 1" in error_str
    assert "    Test output line 2" in error_str

    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired(cmd="test_command", timeout=5)
    error.output = b"Timeout output"
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    error_str = str(wrapped_error)
    assert "Command 'test_command' timed out after 5 seconds" in error_str
    assert "Captured output:" in error_str
    assert "    Timeout output" in error_str

    # Test with no output
    error = subprocess.CalledProcessError(returncode=2, cmd="test_command")
    error.output = None
    wrapped_error = error_wrapper(error)
    error_str = str(wrapped_error)
    assert "No output was generated." in error_str

    # Test with empty output
    error = subprocess.CalledProcessError(returncode=3, cmd="test_command")
    error.output = b""
    wrapped_error = error_wrapper(error)
    error_str = str(wrapped_error)
    assert "No output was generated." in error_str

    # Test with Unicode decode error
    error = subprocess.CalledProcessError(returncode=4, cmd="test_command")
    error.output = b"\xff\xfe"  # Invalid UTF-8
    wrapped_error = error_wrapper(error)
    error_str = str(wrapped_error)
    assert "Failed to parse output." in error_str

    # Test with other exception types (should return unchanged)
    original_error = ValueError("Test error")
    wrapped_error = error_wrapper(original_error)
    assert wrapped_error is original_error
    assert str(wrapped_error) == "Test error"

    # Test with multi-line output
    error = subprocess.CalledProcessError(returncode=5, cmd="test_command")
    error.output = b"Line 1\nLine 2\nLine 3"
    wrapped_error = error_wrapper(error)
    error_str = str(wrapped_error)
    assert "    Line 1" in error_str
    assert "    Line 2" in error_str
    assert "    Line 3" in error_str

    # Test that original error attributes are preserved
    error = subprocess.CalledProcessError(returncode=6, cmd="test_cmd")
    error.output = b"output"
    wrapped_error = error_wrapper(error)
    assert wrapped_error.returncode == 6
    assert wrapped_error.cmd == "test_cmd"
    assert wrapped_error.output == b"output"


# LLM-generated content at query #26
#--------------------------

```python
def test_run_command():
    # Test successful command execution without output capture
    result = run_command(["echo", "hello"])
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output is None

    # Test successful command with return_output=True
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

    # Test command with non-zero exit code
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test command with timeout
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

    # Test command with environment variables
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=test_value" in result.captured_output

    # Test command with working directory
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output

    # Test verbose mode
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output

    # Test shell command
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

    # Test output truncation for long output
    long_output = "x" * 10000
    result = run_command(["echo", long_output], return_output=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output

    # Test error propagation when ignore_errors=False
    import subprocess
    try:
        run_command(["false"])
        assert False, "Should have raised CalledProcessError"
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
        assert e.output is not None

    # Test timeout error propagation when ignore_errors=False
    try:
        run_command(["sleep", "2"], timeout=0.1)
        assert False, "Should have raised TimeoutExpired"
    except subprocess.TimeoutExpired as e:
        assert e.output is not None

    # Test command with stderr redirection
    result = run_command(["sh", "-c", "echo error >&2"], return_output=True)
    assert result.return_code == 0
    assert b"error" in result.captured_output


# LLM-generated content at query #27
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError with output
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped)
    assert "error output" in str(wrapped)
    
    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command", output=None)
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped)
    
    # Test with TimeoutExpired with output
    err = subprocess.TimeoutExpired("test_command", 10, output=b"timeout output")
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped)
    assert "timeout output" in str(wrapped)
    
    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", 10, output=None)
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped)
    
    # Test with non-subprocess exception (should return unchanged)
    err = ValueError("test error")
    wrapped = error_wrapper(err)
    assert wrapped is err
    assert isinstance(wrapped, ValueError)
    assert str(wrapped) == "test error"
    
    # Test with Unicode output that can't be decoded
    err = subprocess.CalledProcessError(1, "test_command", output=b'\xff\xfe')
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped)
    
    # Test that original error attributes are preserved
    err = subprocess.CalledProcessError(42, "test_cmd", output=b"test")
    wrapped = error_wrapper(err)
    assert wrapped.returncode == 42
    assert wrapped.cmd == "test_cmd"
    assert wrapped.output == b"test"


# LLM-generated content at query #28
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError with output
    err = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    err.output = b"error line 1\nerror line 2"
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped)
    assert "    error line 1" in str(wrapped)
    assert "    error line 2" in str(wrapped)

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    err.output = None
    wrapped = error_wrapper(err)
    assert "No output was generated." in str(wrapped)

    # Test with CalledProcessError with empty output
    err = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    err.output = b""
    wrapped = error_wrapper(err)
    assert "No output was generated." in str(wrapped)

    # Test with TimeoutExpired with output
    err = subprocess.TimeoutExpired(cmd="test_cmd", timeout=10)
    err.output = b"timeout output"
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped)
    assert "    timeout output" in str(wrapped)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired(cmd="test_cmd", timeout=10)
    err.output = None
    wrapped = error_wrapper(err)
    assert "No output was generated." in str(wrapped)

    # Test with non-subprocess exception (should return unchanged)
    original_err = ValueError("test error")
    wrapped = error_wrapper(original_err)
    assert wrapped is original_err
    assert str(wrapped) == "test error"

    # Test with Unicode decode error in output
    err = subprocess.CalledProcessError(returncode=1, cmd="test_cmd")
    err.output = b"\xff\xfe"  # Invalid UTF-8
    wrapped = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped)

    # Test that original attributes are preserved
    err = subprocess.CalledProcessError(returncode=42, cmd="special_cmd")
    err.output = b"some output"
    wrapped = error_wrapper(err)
    assert wrapped.returncode == 42
    assert wrapped.cmd == "special_cmd"
    assert wrapped.output == b"some output"


# LLM-generated content at query #29
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped)
    assert "test output" in str(wrapped)
    
    # Test with CalledProcessError and no output
    error = subprocess.CalledProcessError(1, "test_command", output=None)
    wrapped = error_wrapper(error)
    assert "No output was generated." in str(wrapped)
    
    # Test with CalledProcessError and empty output
    error = subprocess.CalledProcessError(1, "test_command", output=b"")
    wrapped = error_wrapper(error)
    assert "No output was generated." in str(wrapped)
    
    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired("test_command", 10, output=b"timeout output")
    wrapped = error_wrapper(error)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped)
    assert "timeout output" in str(wrapped)
    
    # Test with TimeoutExpired and no output
    error = subprocess.TimeoutExpired("test_command", 10, output=None)
    wrapped = error_wrapper(error)
    assert "No output was generated." in str(wrapped)
    
    # Test with other exception types (should return unchanged)
    original_exception = ValueError("test error")
    wrapped = error_wrapper(original_exception)
    assert wrapped is original_exception
    assert str(wrapped) == "test error"
    
    # Test with Unicode decode error in output
    error = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped = error_wrapper(error)
    assert "Failed to parse output." in str(wrapped)
    
    # Test output formatting with multiple lines
    error = subprocess.CalledProcessError(1, "test_command", output=b"line1\nline2\nline3")
    wrapped = error_wrapper(error)
    assert "    line1" in str(wrapped)
    assert "    line2" in str(wrapped)
    assert "    line3" in str(wrapped)


