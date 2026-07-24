####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command with return_output=True
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

    # Test command with verbose=True
    result = run_command(["echo", "hello"], verbose=True)
    assert result.return_code == 0

    # Test command with ignore_errors=True and non-zero return code
    result = run_command(["ls", "/nonexistent"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

    # Test command with timeout and ignore_errors=True
    result = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768

    # Test command with environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True)
    assert result.return_code == 0

    # Test command with cwd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir)
        assert result.return_code == 0

    # Test command with Unicode output
    result = run_command(["echo", "héllo"], return_output=True)
    assert result.return_code == 0
    assert b"héllo" in result.captured_output

    # Test command with long output (truncation)
    long_output = "a" * (MAX_OUTPUT_LENGTH + 100)
    result = run_command(["echo", long_output], ignore_errors=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #2
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command with return_output=True
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

    # Test command with verbose=True
    result = run_command(["echo", "hello"], verbose=True)
    assert result.return_code == 0

    # Test command with ignore_errors=True
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

    # Test command with environment variables
    result = run_command(["printenv", "TEST_VAR"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"test_value" in result.captured_output

    # Test command with cwd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert bytes(tmpdir, 'utf-8') in result.captured_output

    # Test command with error_wrapper
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        run_command(["false"])
    assert "Captured output" in str(exc_info.value)

    # Test command with long output truncation
    long_output = "a" * (MAX_OUTPUT_LENGTH + 100)
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        run_command(["python", "-c", f"print('{long_output}')"], ignore_errors=False)
    assert b"*** (previous output truncated) ***" in exc_info.value.output


# LLM-generated content at query #3
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    error output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    timeout output"

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #4
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test command with non-zero return code
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1", shell=True)

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 2", shell=True, timeout=0.1)

    # Test command with ignore_errors=True
    result = run_command("exit 1", shell=True, ignore_errors=True)
    assert result.return_code == 1

    # Test command with timeout and ignore_errors=True
    result = run_command("sleep 2", shell=True, timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768

    # Test command with verbose=True
    result = run_command("echo 'Verbose test'", shell=True, verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"Verbose test" in result.captured_output

    # Test command with environment variables
    result = run_command("echo $TEST_VAR", shell=True, env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"test_value" in result.captured_output

    # Test command with working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", shell=True, cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert bytes(tmpdir, 'utf-8') in result.captured_output

    # Test command with long output truncation
    long_output = "a" * (MAX_OUTPUT_LENGTH + 1000)
    result = run_command(f"echo '{long_output}'", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len("*** (previous output truncated) ***\n")


# LLM-generated content at query #5
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #6
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #7
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=10)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "Test error"

    # Test with CalledProcessError and non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #8
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "test output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #9
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo hello")
    assert result.command == "echo hello"
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command with return_output=True
    result = run_command("echo hello", return_output=True)
    assert result.command == "echo hello"
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

    # Test command with verbose=True
    result = run_command("echo hello", verbose=True)
    assert result.command == "echo hello"
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command with non-zero return code
    result = run_command("exit 1", ignore_errors=True)
    assert result.command == "exit 1"
    assert result.return_code == 1
    assert result.captured_output is None

    # Test command with timeout
    result = run_command("sleep 2", timeout=1, ignore_errors=True)
    assert result.command == "sleep 2"
    assert result.return_code == -32768
    assert result.captured_output is not None

    # Test command with environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True, return_output=True)
    assert result.command == "echo $TEST_VAR"
    assert result.return_code == 0
    assert result.captured_output == b"test_value\n"

    # Test command with cwd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert result.command == "pwd"
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output

    # Test command with list args
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

    # Test command with ignore_errors=False and non-zero return code
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1", ignore_errors=False)

    # Test command with ignore_errors=False and timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 2", timeout=1, ignore_errors=False)


# LLM-generated content at query #10
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "test output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #11
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(1, "test_command")
    error.output = b"Error output"
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_error)
    assert "Error output" in str(wrapped_error)

    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired("test_command", 10)
    error.output = b"Timeout output"
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_error)
    assert "Timeout output" in str(wrapped_error)

    # Test with other exception types
    error = ValueError("Test error")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, ValueError)
    assert str(wrapped_error) == "Test error"

    # Test with no output
    error = subprocess.CalledProcessError(1, "test_command")
    error.output = None
    wrapped_error = error_wrapper(error)
    assert "No output was generated." in str(wrapped_error)

    # Test with non-UTF-8 output
    error = subprocess.CalledProcessError(1, "test_command")
    error.output = b'\xff\xfe'
    wrapped_error = error_wrapper(error)
    assert "Failed to parse output." in str(wrapped_error)


# LLM-generated content at query #12
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #13
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    error output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    timeout output"

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with UnicodeDecodeError
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #14
#--------------------------

```python
def test_error_wrapper():
    # Test with subprocess.CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    test output"

    # Test with subprocess.TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nNo output was generated."

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "test error"

    # Test with unicode decode error
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #15
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_error)
    assert "test output" in str(wrapped_error)

    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_error)

    # Test with non-subprocess error
    error = ValueError("test error")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, ValueError)
    assert str(wrapped_error) == "test error"

    # Test with CalledProcessError and non-UTF-8 output
    error = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_error)


# LLM-generated content at query #16
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None

    # Test with return_output=True
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello\n" in result.captured_output

    # Test with verbose=True
    result = run_command(["echo", "hello"], verbose=True)
    assert result.return_code == 0

    # Test command with non-zero return code
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["ls", "/nonexistent"])

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

    # Test with ignore_errors=True and non-zero return code
    result = run_command(["ls", "/nonexistent"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test with ignore_errors=True and timeout
    result = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

    # Test with custom environment
    result = run_command("echo $TEST_VAR", shell=True, env={"TEST_VAR": "test_value"})
    assert result.return_code == 0

    # Test with custom cwd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir)
        assert result.return_code == 0

    # Test with long output truncation
    long_output = "x" * (MAX_OUTPUT_LENGTH + 1000)
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        run_command(["python", "-c", f"print('{long_output}')"], ignore_errors=False)
    assert b"*** (previous output truncated) ***" in exc_info.value.output


# LLM-generated content at query #17
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    error output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nNo output was generated."

    # Test with other exception types
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #18
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1, capture_output=True)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with non-subprocess exception
    try:
        raise ValueError("test error")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, ValueError)
        assert str(wrapped) == "test error"


# LLM-generated content at query #19
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #20
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello"])
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command with return_output=True
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

    # Test command with verbose=True
    result = run_command(["echo", "hello"], verbose=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output is not None

    # Test command with ignore_errors=True
    result = run_command(["ls", "/nonexistent"], ignore_errors=True)
    assert result.command == ["ls", "/nonexistent"]
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

    # Test command with custom environment
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"TEST_VAR=test_value" in result.captured_output

    # Test command with custom working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert bytes(tmpdir, 'utf-8') in result.captured_output

    # Test command with shell=True
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.command == "echo hello"
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"


# LLM-generated content at query #21
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.1, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped) or "No output was generated" in str(wrapped)

    # Test with non-subprocess exception
    try:
        raise ValueError("test error")
    except ValueError as e:
        wrapped = error_wrapper(e)
        assert isinstance(wrapped, ValueError)
        assert str(wrapped) == "test error"


# LLM-generated content at query #22
#--------------------------

```python
def test_error_wrapper():
    # Test with subprocess.CalledProcessError
    error = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert str(wrapped_error) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    error output"

    # Test with subprocess.TimeoutExpired
    error = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert str(wrapped_error) == "Command 'test_command' timed out after 1 seconds.\nNo output was generated."

    # Test with non-subprocess exception
    error = ValueError("test error")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, ValueError)
    assert str(wrapped_error) == "test error"

    # Test with subprocess.CalledProcessError and non-UTF-8 output
    error = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert str(wrapped_error) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #23
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", return_output=True)
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "Hello, World!"

    # Test command with non-zero return code
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1

    # Test command with timeout
    result = run_command("sleep 10", timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768

    # Test command with environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.captured_output.decode('utf-8').strip() == "test_value"

    # Test command with working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert result.captured_output.decode('utf-8').strip() == tmpdir

    # Test verbose mode
    result = run_command("echo 'Verbose test'", verbose=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "Verbose test"

    # Test command as list
    result = run_command(["echo", "Hello, World!"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "Hello, World!"

    # Test exception handling
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1")

    # Test output truncation
    long_output = "a" * (MAX_OUTPUT_LENGTH + 1000)
    result = run_command(f"echo '{long_output}'", ignore_errors=True)
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len("*** (previous output truncated) ***\n")


# LLM-generated content at query #24
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello world"], return_output=True)
    assert result.command == ["echo", "hello world"]
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "hello world"

    # Test command with non-zero return code
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["ls", "/nonexistent_directory"])

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

    # Test command with ignore_errors=True
    result = run_command(["ls", "/nonexistent_directory"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test command with verbose=True
    with patch('builtins.print') as mock_print:
        run_command(["echo", "verbose test"], verbose=True)
        mock_print.assert_called()

    # Test command with environment variables
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"TEST_VAR=test_value" in result.captured_output

    # Test command with cwd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.captured_output.decode('utf-8').strip() == tmpdir

    # Test command with return_output=False
    result = run_command(["echo", "no output"], return_output=False)
    assert result.captured_output is None

    # Test command with shell=True
    result = run_command("echo shell_test", shell=True, return_output=True)
    assert result.captured_output.decode('utf-8').strip() == "shell_test"

    # Test command with long output (truncation)
    long_output = "a" * (MAX_OUTPUT_LENGTH + 1000)
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(["echo", long_output], 0)
        with patch('tempfile.TemporaryFile') as mock_tempfile:
            mock_file = mock_tempfile.return_value.__enter__.return_value
            mock_file.read.return_value = long_output.encode('utf-8')
            result = run_command(["echo", long_output], return_output=True)
            assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #25
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped_error) or "No output was generated" in str(wrapped_error)

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.001, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped_error) or "No output was generated" in str(wrapped_error)

    # Test with non-subprocess exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, ValueError)
        assert str(wrapped_error) == "Test error"


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #2
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    test output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    timeout output"

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with UnicodeDecodeError
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #3
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "test"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output.strip() == b"test"

    # Test command with non-zero return code
    result = run_command(["ls", "/nonexistent"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test command with timeout
    result = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

    # Test verbose mode
    with patch('builtins.print') as mock_print:
        run_command(["echo", "verbose"], verbose=True)
        assert mock_print.called

    # Test environment variables
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"TEST_VAR=test_value" in result.captured_output

    # Test working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert bytes(tmpdir, 'utf-8') in result.captured_output

    # Test command as string
    result = run_command("echo string_command", shell=True, return_output=True)
    assert result.captured_output.strip() == b"string_command"

    # Test error wrapper
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        run_command(["ls", "/nonexistent"])
    assert "Captured output:" in str(exc_info.value)


# LLM-generated content at query #4
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "test output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

    # Test with UnicodeDecodeError
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #5
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test command with error
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1")

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 10", timeout=0.1)

    # Test ignore_errors flag
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None

    # Test return_output flag
    result = run_command("echo 'Test'", return_output=True)
    assert result.captured_output is not None
    assert b"Test" in result.captured_output

    # Test verbose flag (just check it doesn't crash)
    result = run_command("echo 'Verbose'", verbose=True)
    assert result.return_code == 0

    # Test environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True, return_output=True)
    assert b"test_value" in result.captured_output

    # Test working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert bytes(tmpdir, 'utf-8') in result.captured_output

    # Test command as list
    result = run_command(["echo", "Hello"], return_output=True)
    assert b"Hello" in result.captured_output

    # Test long output truncation
    long_output = "x" * (MAX_OUTPUT_LENGTH + 1000)
    result = run_command(f"echo '{long_output}'", shell=True, return_output=True)
    assert b"*** (previous output truncated) ***" in result.captured_output
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + 32


# LLM-generated content at query #6
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None

    # Test command with return_output=True
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

    # Test command with verbose=True
    result = run_command(["echo", "hello"], verbose=True)
    assert result.return_code == 0

    # Test command with ignore_errors=True and non-zero return code
    result = run_command(["ls", "/nonexistent"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

    # Test command with timeout and ignore_errors=True
    result = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

    # Test command with environment variables
    result = run_command(["printenv", "TEST_VAR"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"test_value" in result.captured_output

    # Test command with cwd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert bytes(tmpdir, 'utf-8') in result.captured_output

    # Test command with UnicodeDecodeError handling
    result = run_command(["echo", "\xff"], verbose=True)
    assert result.return_code == 0

    # Test command with long output truncation
    long_output = "a" * (MAX_OUTPUT_LENGTH + 100)
    result = run_command(["echo", long_output], ignore_errors=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len(b"*** (previous output truncated) ***\n")


# LLM-generated content at query #7
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", return_output=True)
    assert result.return_code == 0
    assert b"Hello, World!" in result.captured_output

    # Test command with non-zero return code
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1

    # Test command with timeout
    result = run_command("sleep 10", timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768

    # Test verbose mode
    with patch('builtins.print') as mock_print:
        run_command("echo 'Verbose test'", verbose=True)
        assert mock_print.called

    # Test environment variables
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"test_value" in result.captured_output

    # Test working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test.txt")
        run_command(f"touch {test_file}", cwd=tmpdir)
        assert os.path.exists(test_file)

    # Test command as list
    result = run_command(["echo", "Hello, List!"], return_output=True)
    assert result.return_code == 0
    assert b"Hello, List!" in result.captured_output

    # Test error wrapper
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        run_command("exit 1")
    assert "Captured output" in str(exc_info.value)


# LLM-generated content at query #8
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    error = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_error)
    assert "Error output" in str(wrapped_error)

    # Test with TimeoutExpired
    error = subprocess.TimeoutExpired("test_command", timeout=10)
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_error)

    # Test with non-subprocess exception
    error = ValueError("test error")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, ValueError)
    assert str(wrapped_error) == "test error"

    # Test with CalledProcessError and non-UTF-8 output
    error = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_error)


# LLM-generated content at query #9
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    Error output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    Timeout output"

    # Test with non-subprocess exception
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "Test error"

    # Test with CalledProcessError and no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with TimeoutExpired and no output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nNo output was generated."

    # Test with CalledProcessError and non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #10
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "test output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

    # Test with UnicodeDecodeError
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #11
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    test output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    test output"

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #12
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "test"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output.strip() == b"test"

    # Test command with return code
    result = run_command(["sh", "-c", "exit 1"], return_output=True)
    assert result.return_code == 1
    assert result.captured_output is not None

    # Test verbose mode
    with patch("builtins.print") as mock_print:
        run_command(["echo", "verbose"], verbose=True)
        assert mock_print.called

    # Test environment variables
    result = run_command(["sh", "-c", "echo $TEST_VAR"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"test_value" in result.captured_output

    # Test working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert bytes(tmpdir, "utf-8") in result.captured_output

    # Test timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

    # Test ignore_errors
    result = run_command(["sh", "-c", "exit 1"], ignore_errors=True)
    assert result.return_code == 1

    # Test command as string
    result = run_command("echo string_command", shell=True, return_output=True)
    assert b"string_command" in result.captured_output

    # Test long output truncation
    long_output = "a" * (MAX_OUTPUT_LENGTH + 1000)
    result = run_command(["sh", "-c", f"echo {long_output}"], return_output=True)
    assert b"*** (previous output truncated) ***" in result.captured_output
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + 32

    # Test Unicode decode error handling
    result = run_command(["sh", "-c", "echo -e '\\xff\\xfe'"], return_output=True, verbose=True)
    assert result.captured_output is not None


# LLM-generated content at query #13
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'Hello, World!'", shell=True, return_output=True)
    assert result.command == "echo 'Hello, World!'"
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "Hello, World!"

    # Test command with non-zero return code
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1", shell=True)

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 2", shell=True, timeout=0.1)

    # Test command with ignore_errors=True
    result = run_command("exit 1", shell=True, ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None

    # Test command with verbose=True
    with pytest.capsys() as capsys:
        run_command("echo 'Verbose Test'", shell=True, verbose=True)
        captured = capsys.readouterr()
        assert "Verbose Test" in captured.out

    # Test command with environment variables
    result = run_command("echo $TEST_VAR", shell=True, env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.captured_output.decode('utf-8').strip() == "test_value"

    # Test command with working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", shell=True, cwd=tmpdir, return_output=True)
        assert result.captured_output.decode('utf-8').strip() == tmpdir

    # Test command with long output truncation
    long_output = "a" * (MAX_OUTPUT_LENGTH + 100)
    result = run_command(f"echo '{long_output}'", shell=True, return_output=True)
    assert b"*** (previous output truncated) ***" in result.captured_output
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len("*** (previous output truncated) ***\n")

    # Test command with return_output=False
    result = run_command("echo 'No Output'", shell=True, return_output=False)
    assert result.captured_output is None


# LLM-generated content at query #14
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    error output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nCaptured output:\n    timeout output"

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with no output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."

    # Test with UnicodeDecodeError
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #15
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    try:
        subprocess.run(["false"], check=True)
    except subprocess.CalledProcessError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.CalledProcessError)
        assert "Captured output" in str(wrapped_error) or "No output was generated" in str(wrapped_error)

    # Test with TimeoutExpired
    try:
        subprocess.run(["sleep", "10"], timeout=0.001, check=True)
    except subprocess.TimeoutExpired as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, subprocess.TimeoutExpired)
        assert "Captured output" in str(wrapped_error) or "No output was generated" in str(wrapped_error)

    # Test with non-subprocess exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        wrapped_error = error_wrapper(e)
        assert isinstance(wrapped_error, ValueError)
        assert str(wrapped_error) == "Test error"


# LLM-generated content at query #16
#--------------------------

```python
def test_error_wrapper():
    # Test with subprocess.CalledProcessError
    error = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert str(wrapped_error) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    Error output"

    # Test with subprocess.TimeoutExpired
    error = subprocess.TimeoutExpired("test_command", timeout=10)
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert str(wrapped_error) == "Command 'test_command' timed out after 10 seconds.\nNo output was generated."

    # Test with other exception types
    error = ValueError("test error")
    wrapped_error = error_wrapper(error)
    assert wrapped_error is error
    assert str(wrapped_error) == "test error"

    # Test with UnicodeDecodeError in output
    error = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert str(wrapped_error) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."

    # Test with no output
    error = subprocess.CalledProcessError(1, "test_command")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert str(wrapped_error) == "Command 'test_command' returned non-zero exit status 1.\nNo output was generated."


# LLM-generated content at query #17
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    error output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nNo output was generated."

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError and non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #18
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "Error output" in str(wrapped_err)

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"Timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "Timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "Test error"

    # Test with CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #19
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "hello world"], return_output=True)
    assert result.return_code == 0
    assert b"hello world" in result.captured_output

    # Test command with error
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["ls", "/nonexistent_directory"])

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

    # Test ignore_errors flag
    result = run_command(["ls", "/nonexistent_directory"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test verbose flag (capturing output)
    result = run_command(["echo", "verbose test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"verbose test" in result.captured_output

    # Test environment variables
    result = run_command("echo $TEST_VAR", shell=True, env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"test_value" in result.captured_output

    # Test working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test content")
        result = run_command(["cat", "test.txt"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert b"test content" in result.captured_output

    # Test command as string
    result = run_command("echo 'string command'", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"string command" in result.captured_output

    # Test return_output=False (default)
    result = run_command(["echo", "no output"])
    assert result.return_code == 0
    assert result.captured_output is None


# LLM-generated content at query #20
#--------------------------

```python
def test_error_wrapper():
    # Test with subprocess.CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "test output" in str(wrapped_err)

    # Test with subprocess.TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with subprocess.CalledProcessError and non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #21
#--------------------------

```python
def test_error_wrapper():
    # Test with subprocess.CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "test output" in str(wrapped_err)

    # Test with subprocess.TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with subprocess.CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with subprocess.TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

    # Test with subprocess.CalledProcessError with non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #22
#--------------------------

```python
def test_error_wrapper():
    # Test with subprocess.CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "error output" in str(wrapped_err)

    # Test with subprocess.TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with subprocess.CalledProcessError without output
    err = subprocess.CalledProcessError(1, "test_command")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

    # Test with subprocess.TimeoutExpired without output
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)


# LLM-generated content at query #23
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command("echo 'test'", return_output=True)
    assert result.return_code == 0
    assert b"test\n" in result.captured_output

    # Test command with error
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1")

    # Test command with timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 2", timeout=0.1)

    # Test ignore_errors flag
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1

    # Test verbose flag
    with patch('builtins.print') as mock_print:
        run_command("echo 'verbose'", verbose=True)
        mock_print.assert_called()

    # Test environment variables
    result = run_command("env", env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"TEST_VAR=test_value" in result.captured_output

    # Test working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert bytes(tmpdir, 'utf-8') in result.captured_output

    # Test command as list
    result = run_command(["echo", "test list"], return_output=True)
    assert b"test list\n" in result.captured_output

    # Test output truncation
    long_output = "a" * (MAX_OUTPUT_LENGTH + 100)
    result = run_command(f"echo '{long_output}'", return_output=True)
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len("*** (previous output truncated) ***\n")


# LLM-generated content at query #24
#--------------------------

```python
def test_error_wrapper():
    # Test with CalledProcessError
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    test output"

    # Test with TimeoutExpired
    err = subprocess.TimeoutExpired("test_command", timeout=1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nNo output was generated."

    # Test with non-subprocess exception
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

    # Test with CalledProcessError and non-UTF-8 output
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #25
#--------------------------

```python
def test_run_command():
    # Test successful command execution
    result = run_command(["echo", "test"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

    # Test command with non-zero return code
    result = run_command(["ls", "/nonexistent"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

    # Test command with timeout
    result = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

    # Test verbose mode
    result = run_command(["echo", "verbose"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"verbose\n"

    # Test environment variables
    result = run_command(["echo", "$TEST_VAR"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"test_value\n"

    # Test working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output

    # Test command as string
    result = run_command("echo string", shell=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"string\n"

    # Test error wrapper
    try:
        run_command(["ls", "/nonexistent"])
    except subprocess.CalledProcessError as e:
        assert "No such file or directory" in str(e)
        assert e.output is not None

    # Test output truncation
    long_output = "a" * (MAX_OUTPUT_LENGTH + 100)
    result = run_command(["echo", long_output], ignore_errors=True, return_output=True)
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len("*** (previous output truncated) ***\n")
    assert b"*** (previous output truncated) ***" in result.captured_output


