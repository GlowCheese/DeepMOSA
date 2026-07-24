####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```
def test_error_wrapper_with_called_process_error():
    err = subprocess.CalledProcessError(1, "cmd", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert "Captured output:\n    test output" in str(wrapped_err)

def test_error_wrapper_with_timeout_expired():
    err = subprocess.TimeoutExpired("cmd", 10, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert "Captured output:\n    timeout output" in str(wrapped_err)

def test_error_wrapper_with_other_exception():
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert str(wrapped_err) == "test error"

def test_error_wrapper_with_no_output():
    err = subprocess.CalledProcessError(1, "cmd", output=None)
    wrapped_err = error_wrapper(err)
    assert "No output was generated" in str(wrapped_err)

def test_error_wrapper_with_unicode_error():
    err = subprocess.CalledProcessError(1, "cmd", output=b"\xff")
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output" in str(wrapped_err)


# LLM-generated content at query #2
#--------------------------

```python
def test_error_wrapper_with_timeout_expired():
    import subprocess
    err = subprocess.TimeoutExpired(cmd="test", timeout=10)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)

def test_error_wrapper_with_called_process_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd="test")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

def test_error_wrapper_with_called_process_error_with_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd="test", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "test output" in str(wrapped_err)

def test_error_wrapper_with_other_exception():
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err == err
    assert str(wrapped_err) == "test error"


# LLM-generated content at query #3
#--------------------------

```python
def test_run_command_success():
    result = run_command("echo Hello", verbose=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"Hello\n"

def test_run_command_failure():
    result = run_command("invalid_command", ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_timeout():
    result = run_command("sleep 2", timeout=1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_return_output():
    result = run_command("echo Hello", return_output=True)
    assert result.captured_output == b"Hello\n"

def test_run_command_no_output():
    result = run_command("echo Hello")
    assert result.captured_output is None

def test_run_command_verbose():
    result = run_command("echo Hello", verbose=True)
    assert result.captured_output == b"Hello\n"

def test_run_command_cwd():
    result = run_command("pwd", cwd="/tmp", return_output=True)
    assert result.captured_output == b"/tmp\n"

def test_run_command_env():
    result = run_command("echo $TEST_ENV", env={"TEST_ENV": "test"}, return_output=True)
    assert result.captured_output == b"test\n"


# LLM-generated content at query #4
#--------------------------

```python
def test_run_command_with_return_output_true():
    result = run_command("echo Hello, World!", return_output=True)
    assert result.captured_output is not None


# LLM-generated content at query #5
#--------------------------

```python
def test_pipeline_returns_captured_output_when_return_output_is_true():
    result = run_command("echo 'Hello, World!'", return_output=True)
    assert result.captured_output is not None

def test_pipeline_returns_captured_output_when_return_code_is_nonzero():
    result = run_command("false")
    assert result.captured_output is not None

def test_pipeline_returns_captured_output_when_verbose_is_true():
    result = run_command("echo 'Hello, World!'", verbose=True)
    assert result.captured_output is not None


# LLM-generated content at query #6
#--------------------------

```
def test_error_wrapper_returns_non_subprocess_error_unchanged():
    class CustomError(Exception):
        pass
    
    custom_error = CustomError("test error")
    result = error_wrapper(custom_error)
    assert result is custom_error


# LLM-generated content at query #7
#--------------------------

```python
def test_run_command_truncates_long_output():
    # Simulate a case where the output length exceeds MAX_OUTPUT_LENGTH
    long_output = b"a" * (8192 + 1)
    truncated_output = b"*** (previous output truncated) ***\n" + long_output[-8192:]
    
    # Mock subprocess.run to raise CalledProcessError with long output
    def mock_subprocess_run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "mock_command", output=long_output)
    
    # Patch subprocess.run with the mock function
    import subprocess
    original_run = subprocess.run
    subprocess.run = mock_subprocess_run
    
    # Run the command with ignore_errors=True to capture the output
    result = run_command("mock_command", ignore_errors=True)
    
    # Assert that the output is truncated correctly
    assert result.captured_output == truncated_output
    
    # Restore the original subprocess.run
    subprocess.run = original_run


# LLM-generated content at query #8
#--------------------------

```python
def test_run_command_return_output_true():
    result = run_command("echo Hello", return_output=True)
    assert result.captured_output is not None

def test_run_command_return_code_nonzero():
    result = run_command("false", return_output=False)
    assert result.captured_output is not None

def test_run_command_verbose_true():
    result = run_command("echo Hello", verbose=True)
    assert result.captured_output is not None


# LLM-generated content at query #9
#--------------------------

def test_run_command_verbose_logging():
    args = ["echo", "hello"]
    env = {"TEST_ENV": "test_value"}
    cwd = "/tmp"
    timeout = 10.0
    verbose = True
    return_output = False
    ignore_errors = False
    result = run_command(args, env=env, cwd=cwd, timeout=timeout, verbose=verbose, return_output=return_output, ignore_errors=ignore_errors)
    assert isinstance(result, CommandResult)
    assert result.return_code == 0
    assert result.captured_output is not None


# LLM-generated content at query #10
#--------------------------

```python
def test_run_command_ignore_errors_with_timeout():
    result = run_command("sleep 10", timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None
    assert isinstance(result, CommandResult)


# LLM-generated content at query #11
#--------------------------

```python
def test_error_wrapper_non_subprocess_error():
    class CustomError(Exception):
        pass
    
    custom_err = CustomError("test error")
    result = error_wrapper(custom_err)
    assert result is custom_err


# LLM-generated content at query #12
#--------------------------

```
def test_error_wrapper_returns_same_error_if_not_subprocess_error():
    class CustomError(Exception):
        pass
    
    err = CustomError("test error")
    result = error_wrapper(err)
    assert result is err

def test_error_wrapper_wraps_called_process_error():
    err = subprocess.CalledProcessError(1, "cmd")
    result = error_wrapper(err)
    assert result is err
    assert isinstance(result, type(err))
    assert hasattr(result.__class__, "__str__")

def test_error_wrapper_wraps_timeout_expired():
    err = subprocess.TimeoutExpired("cmd", 10)
    result = error_wrapper(err)
    assert result is err
    assert isinstance(result, type(err))
    assert hasattr(result.__class__, "__str__")


# LLM-generated content at query #13
#--------------------------

```python
def test_run_command_success_no_output():
    result = run_command(["echo", "hello"], return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_error_no_output():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is None

def test_run_command_error_with_output():
    result = run_command(["false"], ignore_errors=True, return_output=True)
    assert result.return_code == 1
    assert result.captured_output == b""

def test_run_command_timeout_no_output():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is None

def test_run_command_timeout_with_output():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True, return_output=True)
    assert result.return_code == -32768
    assert result.captured_output.startswith(b"*** (previous output truncated) ***\n")

def test_run_command_verbose():
    result = run_command(["echo", "hello"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_env():
    result = run_command(["printenv", "TEST_ENV"], env={"TEST_ENV": "test"}, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.return_code == 0
    assert result.captured_output.rstrip() == b"/tmp"

def test_run_command_shell():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"


# LLM-generated content at query #14
#--------------------------

```python
def test_error_wrapper_non_subprocess_error():
    class CustomError(Exception):
        pass
    
    custom_err = CustomError("Test error")
    result = error_wrapper(custom_err)
    assert result is custom_err


# LLM-generated content at query #15
#--------------------------

def test_run_command_truncates_long_output():
    MAX_OUTPUT_LENGTH = 8192
    long_output = b"x" * (MAX_OUTPUT_LENGTH + 100)
    mock_process = subprocess.run
    mock_process.side_effect = subprocess.CalledProcessError(1, "cmd", output=long_output)
    result = run_command("cmd", ignore_errors=True)
    assert len(result.captured_output) == MAX_OUTPUT_LENGTH + len(b"*** (previous output truncated) ***\n")
    assert result.captured_output.startswith(b"*** (previous output truncated) ***\n")


# LLM-generated content at query #16
#--------------------------

```python
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

def test_run_command_failure():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode == 1
        assert e.output == b""

def test_run_command_timeout():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert e.timeout == 0.1
        assert e.output == b""

def test_run_command_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.command == ["false"]
    assert result.return_code == 1
    assert result.captured_output == b""

def test_run_command_verbose():
    result = run_command(["echo", "hello"], verbose=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_custom_env():
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"TEST_VAR=test_value" in result.captured_output

def test_run_command_custom_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.captured_output == b"/tmp\n"

def test_run_command_truncate_long_output():
    long_output = b"a" * 10000
    try:
        run_command(["echo", long_output.decode("utf-8")])
    except subprocess.CalledProcessError as e:
        assert len(e.output) <= 8192 + len("*** (previous output truncated) ***\n")


# LLM-generated content at query #17
#--------------------------

```python
def test_run_command_success():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_failure():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is None

def test_run_command_timeout():
    result = run_command(["sleep", "2"], timeout=1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command(["echo", "hello"], verbose=True)
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_return_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_non_zero_return_code():
    result = run_command(["false"], return_output=True, ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"/tmp\n"

def test_run_command_with_env():
    result = run_command(["env"], env={"TEST_ENV": "1"}, return_output=True)
    assert result.return_code == 0
    assert b"TEST_ENV=1" in result.captured_output


# LLM-generated content at query #18
#--------------------------

```python
def test_log_handles_unencodable_bytes():
    unencodable_output = b'\x80abc'
    log(unencodable_output.decode('utf-8'), timestamp=False, include_proc_id=False)


# LLM-generated content at query #19
#--------------------------

```python
def test_run_command_return_output_true():
    result = run_command("echo hello", return_output=True)
    assert result.captured_output is not None

def test_run_command_return_code_nonzero():
    result = run_command("false", ignore_errors=True)
    assert result.captured_output is not None

def test_run_command_verbose_true():
    result = run_command("echo hello", verbose=True)
    assert result.captured_output is not None


# LLM-generated content at query #20
#--------------------------

```python
def test_run_command_success():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_failure():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is None

def test_run_command_timeout():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose(capsys):
    run_command(["echo", "hello"], verbose=True)
    captured = capsys.readouterr()
    assert "hello" in captured.out

def test_run_command_return_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.captured_output == b"hello\n"

def test_run_command_cwd(tmp_path):
    (tmp_path / "test.txt").write_text("hello")
    result = run_command(["cat", "test.txt"], cwd=tmp_path, return_output=True)
    assert result.captured_output == b"hello"

def test_run_command_env():
    result = run_command(["sh", "-c", "echo $VAR"], env={"VAR": "hello"}, return_output=True)
    assert result.captured_output == b"hello\n"


# LLM-generated content at query #21
#--------------------------

```python
def test_run_command_with_unicode_decode_error():
    args = ["echo", "-n", "こんにちは"]
    result = run_command(args, verbose=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"こんにちは"


# LLM-generated content at query #22
#--------------------------

```
def test_run_command_no_unicode_decode_error():
    result = run_command(["echo", "hello"], return_output=True, verbose=True)
    assert isinstance(result, CommandResult)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"hello" in result.captured_output

def test_run_command_with_unicode_decode_error():
    # Create a binary output that cannot be decoded as UTF-8
    binary_output = b'\xff\xfe\xfd'
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(binary_output)
        tmp.flush()
        result = run_command(["cat", tmp.name], return_output=True, verbose=True)
    assert isinstance(result, CommandResult)
    assert result.return_code == 0
    assert result.captured_output == binary_output


# LLM-generated content at query #23
#--------------------------

```python
def test_log_with_verbose_flag():
    run_command("echo Hello", verbose=True)


# LLM-generated content at query #24
#--------------------------

```
def test_error_wrapper_with_called_process_error_with_output():
    err = subprocess.CalledProcessError(1, 'cmd', output=b'some output')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert 'Captured output:' in str(wrapped_err)
    assert 'some output' in str(wrapped_err)

def test_error_wrapper_with_called_process_error_without_output():
    err = subprocess.CalledProcessError(1, 'cmd', output=None)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert 'No output was generated' in str(wrapped_err)

def test_error_wrapper_with_timeout_expired_with_output():
    err = subprocess.TimeoutExpired('cmd', 1, output=b'some output')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert 'Captured output:' in str(wrapped_err)
    assert 'some output' in str(wrapped_err)

def test_error_wrapper_with_timeout_expired_without_output():
    err = subprocess.TimeoutExpired('cmd', 1, output=None)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert 'No output was generated' in str(wrapped_err)

def test_error_wrapper_with_non_subprocess_error():
    err = ValueError('some error')
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == 'some error'


# LLM-generated content at query #25
#--------------------------

```python
def test_error_wrapper_with_called_process_error():
    class MockCalledProcessError(subprocess.CalledProcessError):
        def __init__(self, returncode, cmd, output):
            super().__init__(returncode, cmd)
            self.output = output

    err = MockCalledProcessError(1, "cmd", b"output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, MockCalledProcessError)
    assert wrapped_err.__str__() == "Command 'cmd' returned non-zero exit status 1.\nCaptured output:\n    output"

def test_error_wrapper_with_timeout_expired():
    class MockTimeoutExpired(subprocess.TimeoutExpired):
        def __init__(self, cmd, timeout, output):
            super().__init__(cmd, timeout)
            self.output = output

    err = MockTimeoutExpired("cmd", 10, b"output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, MockTimeoutExpired)
    assert wrapped_err.__str__() == "Command 'cmd' timed out after 10 seconds\nCaptured output:\n    output"

def test_error_wrapper_with_other_error():
    err = ValueError("Some error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err == err
    assert wrapped_err.__str__() == "Some error"

def test_error_wrapper_with_no_output():
    class MockCalledProcessError(subprocess.CalledProcessError):
        def __init__(self, returncode, cmd):
            super().__init__(returncode, cmd)
            self.output = None

    err = MockCalledProcessError(1, "cmd")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, MockCalledProcessError)
    assert wrapped_err.__str__() == "Command 'cmd' returned non-zero exit status 1.\nNo output was generated."

def test_error_wrapper_with_unicode_encode_error():
    class MockCalledProcessError(subprocess.CalledProcessError):
        def __init__(self, returncode, cmd, output):
            super().__init__(returncode, cmd)
            self.output = output

    err = MockCalledProcessError(1, "cmd", b"\xff")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, MockCalledProcessError)
    assert wrapped_err.__str__() == "Command 'cmd' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #26
#--------------------------

```python
def test_run_command_success():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_failure():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode == 1

def test_run_command_timeout():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert e.timeout == 0.1

def test_run_command_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code == 1

def test_run_command_return_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.captured_output == b"hello\n"

def test_run_command_verbose():
    result = run_command(["echo", "hello"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_env():
    result = run_command(["env"], env={"TEST_ENV": "123"}, return_output=True)
    assert b"TEST_ENV=123" in result.captured_output

def test_run_command_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.captured_output.strip() == b"/tmp"


# LLM-generated content at query #27
#--------------------------

def test_run_command_success():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output


def test_run_command_failure():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0


def test_run_command_timeout():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768


def test_run_command_with_output():
    result = run_command(["echo", "test"], return_output=True)
    assert result.captured_output is not None
    assert b"test" in result.captured_output


def test_run_command_with_env():
    result = run_command(["printenv", "TEST_VAR"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"test_value" in result.captured_output


def test_run_command_verbose(capsys):
    run_command(["echo", "verbose_test"], verbose=True)
    captured = capsys.readouterr()
    assert "verbose_test" in captured.out


def test_run_command_invalid_command():
    try:
        run_command(["nonexistent_command"])
        assert False
    except subprocess.CalledProcessError:
        assert True


def test_run_command_with_cwd(tmp_path):
    (tmp_path / "testfile").write_text("content")
    result = run_command(["cat", "testfile"], cwd=tmp_path, return_output=True)
    assert b"content" in result.captured_output


# LLM-generated content at query #28
#--------------------------

```python
def test_error_wrapper_returns_same_error_when_not_subprocess_error():
    class CustomError(Exception):
        pass

    custom_error = CustomError("Test error")
    result = error_wrapper(custom_error)
    assert result is custom_error


# LLM-generated content at query #29
#--------------------------

```python
def test_run_command_without_return_output():
    result = run_command("echo 'test'", return_output=False)
    assert result.captured_output is None


# LLM-generated content at query #30
#--------------------------

```python
def test_run_command_success():
    result = run_command(["echo", "hello world"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"hello world\n"

def test_run_command_failure():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is None

def test_run_command_timeout():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_return_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_verbose(capsys):
    run_command(["echo", "verbose test"], verbose=True)
    captured = capsys.readouterr()
    assert "verbose test" in captured.out

def test_run_command_env():
    result = run_command(["printenv", "TEST_ENV"], env={"TEST_ENV": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"test_value\n"

def test_run_command_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"/tmp\n"


# LLM-generated content at query #31
#--------------------------

```python
def test_run_command_unicode_decode_error():
    args = ["echo", "test"]
    result = run_command(args, verbose=True, return_output=True)
    assert isinstance(result, CommandResult)
    assert result.return_code == 0
    assert result.captured_output is not None


# LLM-generated content at query #32
#--------------------------

```python
def test_log_with_unicode_decode_error():
    output = b"Hello \x80World"
    log(output.decode('utf-8'), timestamp=False, include_proc_id=False)


# LLM-generated content at query #33
#--------------------------

```python
def test_error_wrapper_returns_original_error_if_not_subprocess_error():
    err = ValueError("Some error")
    result = error_wrapper(err)
    assert result == err

def test_error_wrapper_wraps_subprocess_called_process_error():
    err = subprocess.CalledProcessError(1, "cmd", output=b"output")
    result = error_wrapper(err)
    assert result.__class__.__name__ == "CalledProcessError"
    assert hasattr(result, "__str__")

def test_error_wrapper_wraps_subprocess_timeout_expired_error():
    err = subprocess.TimeoutExpired("cmd", 1, output=b"output")
    result = error_wrapper(err)
    assert result.__class__.__name__ == "TimeoutExpired"
    assert hasattr(result, "__str__")


# LLM-generated content at query #34
#--------------------------

```python
def test_error_wrapper_with_called_process_error():
    error = subprocess.CalledProcessError(1, 'cmd', output=b'some output')
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert 'Captured output:' in str(wrapped_error)

def test_error_wrapper_with_timeout_expired():
    error = subprocess.TimeoutExpired('cmd', timeout=10, output=b'some output')
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert 'Captured output:' in str(wrapped_error)

def test_error_wrapper_with_other_exception():
    error = ValueError('Some error')
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, ValueError)
    assert str(wrapped_error) == 'Some error'

def test_error_wrapper_with_no_output():
    error = subprocess.CalledProcessError(1, 'cmd', output=None)
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert 'No output was generated.' in str(wrapped_error)

def test_error_wrapper_with_failed_output_decoding():
    error = subprocess.CalledProcessError(1, 'cmd', output=b'\xff')
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert 'Failed to parse output.' in str(wrapped_error)


# LLM-generated content at query #35
#--------------------------

```python
def test_error_wrapper_returns_same_error_when_not_subprocess_error():
    class CustomError(Exception):
        pass
    
    err = CustomError("test error")
    result = error_wrapper(err)
    assert result is err

def test_error_wrapper_returns_modified_error_when_subprocess_error():
    err = subprocess.CalledProcessError(1, "cmd")
    result = error_wrapper(err)
    assert isinstance(result, type(err))
    assert result.__class__.__name__ == err.__class__.__name__

def test_error_wrapper_preserves_error_attributes():
    err = subprocess.CalledProcessError(1, "cmd", output=b"test output")
    result = error_wrapper(err)
    assert result.returncode == 1
    assert result.cmd == "cmd"
    assert result.output == b"test output"


# LLM-generated content at query #36
#--------------------------

```python
def test_error_wrapper_returns_original_error_when_not_subprocess_error():
    class CustomError(Exception):
        pass

    custom_error = CustomError("Test error")
    result = error_wrapper(custom_error)
    assert result is custom_error

def test_error_wrapper_wraps_subprocess_called_process_error():
    subprocess_error = subprocess.CalledProcessError(returncode=1, cmd="test")
    result = error_wrapper(subprocess_error)
    assert isinstance(result, subprocess.CalledProcessError)
    assert result.__class__.__name__ == "CalledProcessError"

def test_error_wrapper_wraps_subprocess_timeout_expired():
    timeout_error = subprocess.TimeoutExpired(cmd="test", timeout=10)
    result = error_wrapper(timeout_error)
    assert isinstance(result, subprocess.TimeoutExpired)
    assert result.__class__.__name__ == "TimeoutExpired"


# LLM-generated content at query #37
#--------------------------

```python
def test_error_wrapper_with_called_process_error():
    err = subprocess.CalledProcessError(1, ['cmd'], output=b'output')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)

def test_error_wrapper_with_timeout_expired():
    err = subprocess.TimeoutExpired(['cmd'], 10, output=b'output')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)

def test_error_wrapper_with_other_exception():
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "test error"

def test_error_wrapper_with_no_output():
    err = subprocess.CalledProcessError(1, ['cmd'])
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

def test_error_wrapper_with_unicode_error():
    err = subprocess.CalledProcessError(1, ['cmd'], output=b'\xff')
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #38
#--------------------------

```python
def test_error_wrapper_with_called_process_error():
    err = subprocess.CalledProcessError(1, "cmd", output=b"sample output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:\n    sample output" in str(wrapped_err)

def test_error_wrapper_with_timeout_expired():
    err = subprocess.TimeoutExpired("cmd", 10, output=b"sample output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:\n    sample output" in str(wrapped_err)

def test_error_wrapper_with_other_exception():
    err = ValueError("Some error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err == err
    assert str(wrapped_err) == "Some error"


# LLM-generated content at query #39
#--------------------------

```
def test_error_wrapper_returns_err_when_not_called_process_error_or_timeout_expired():
    class CustomException(Exception):
        pass

    custom_exc = CustomException("Test exception")
    result = error_wrapper(custom_exc)
    assert result == custom_exc


# LLM-generated content at query #40
#--------------------------

```python
def test_error_wrapper_with_called_process_error():
    err = subprocess.CalledProcessError(1, "cmd", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert "Captured output:" in str(wrapped_err)
    assert "test output" in str(wrapped_err)

def test_error_wrapper_with_timeout_expired():
    err = subprocess.TimeoutExpired("cmd", 10, output=b"test output")
    wrapped_err = error_wrapper(err)
    assert "Captured output:" in str(wrapped_err)
    assert "test output" in str(wrapped_err)

def test_error_wrapper_with_other_exception():
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert str(wrapped_err) == "test error"

def test_error_wrapper_with_no_output():
    err = subprocess.CalledProcessError(1, "cmd")
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

def test_error_wrapper_with_unicode_decode_error():
    err = subprocess.CalledProcessError(1, "cmd", output=b"\xff")
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #41
#--------------------------

```python
def test_run_command_success():
    result = run_command(["echo", "hello"], verbose=True, return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_timeout():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.command == ["sleep", "2"]
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_error():
    result = run_command(["false"], ignore_errors=True)
    assert result.command == ["false"]
    assert result.return_code == 1
    assert result.captured_output is not None

def test_run_command_with_output():
    result = run_command(["echo", "world"], return_output=True)
    assert result.command == ["echo", "world"]
    assert result.return_code == 0
    assert result.captured_output == b"world\n"

def test_run_command_without_output():
    result = run_command(["true"])
    assert result.command == ["true"]
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_with_env():
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.command == ["env"]
    assert result.return_code == 0
    assert b"TEST_VAR=test_value" in result.captured_output

def test_run_command_with_cwd():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.command == ["pwd"]
        assert result.return_code == 0
        assert result.captured_output.decode('utf-8').strip() == tmpdir

def test_run_command_verbose():
    result = run_command(["echo", "verbose"], verbose=True)
    assert result.command == ["echo", "verbose"]
    assert result.return_code == 0
    assert result.captured_output is not None

def test_run_command_with_shell():
    result = run_command("echo shell", shell=True, return_output=True)
    assert result.command == "echo shell"
    assert result.return_code == 0
    assert result.captured_output == b"shell\n"


# LLM-generated content at query #42
#--------------------------

```
def test_error_wrapper_with_called_process_error():
    error = subprocess.CalledProcessError(1, 'cmd', output=b'output')
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert wrapped_error.__str__() == "Command 'cmd' returned non-zero exit status 1.\nCaptured output:\n    output"

def test_error_wrapper_with_timeout_expired():
    error = subprocess.TimeoutExpired('cmd', 10, output=b'output')
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert wrapped_error.__str__() == "Command 'cmd' timed out after 10 seconds\nCaptured output:\n    output"

def test_error_wrapper_with_other_exception():
    error = ValueError("test error")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, ValueError)
    assert wrapped_error.__str__() == "test error"

def test_error_wrapper_with_no_output():
    error = subprocess.CalledProcessError(1, 'cmd', output=None)
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert wrapped_error.__str__() == "Command 'cmd' returned non-zero exit status 1.\nNo output was generated."

def test_error_wrapper_with_failed_output_decoding():
    error = subprocess.CalledProcessError(1, 'cmd', output=b'\xff')
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert wrapped_error.__str__() == "Command 'cmd' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #43
#--------------------------

```python
def test_error_wrapper_returns_original_error_when_not_subprocess_error():
    class CustomError(Exception):
        pass

    err = CustomError("Custom error")
    result = error_wrapper(err)
    assert result is err


# LLM-generated content at query #44
#--------------------------

def test_run_command_output_truncation():
    MAX_OUTPUT_LENGTH = 8192
    long_output = b"a" * (MAX_OUTPUT_LENGTH + 1)
    args = ["echo", "test"]
    env = None
    cwd = None
    timeout = None
    verbose = False
    return_output = False
    ignore_errors = True
    kwargs = {}
    
    with tempfile.TemporaryFile() as f:
        f.write(long_output)
        f.seek(0)
        output = f.read()
        assert len(output) > MAX_OUTPUT_LENGTH


# LLM-generated content at query #45
#--------------------------

```python
def test_run_command_success():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_failure():
    result = run_command(["ls", "nonexistent"], ignore_errors=True)
    assert result.command == ["ls", "nonexistent"]
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_timeout():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.command == ["sleep", "2"]
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_with_output():
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_without_output():
    result = run_command(["echo", "test"], return_output=False)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_with_verbose(capsys):
    run_command(["echo", "verbose"], verbose=True)
    captured = capsys.readouterr()
    assert "'echo', 'verbose'" in captured.out

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.command == ["pwd"]
    assert result.return_code == 0
    assert result.captured_output.strip() == b"/tmp"

def test_run_command_with_env():
    result = run_command(["printenv", "TEST_ENV"], env={"TEST_ENV": "test_value"}, return_output=True)
    assert result.command == ["printenv", "TEST_ENV"]
    assert result.return_code == 0
    assert result.captured_output.strip() == b"test_value"


# LLM-generated content at query #46
#--------------------------

```python
def test_error_wrapper_returns_err_when_not_subprocess_error():
    err = ValueError("Test error")
    result = error_wrapper(err)
    assert result == err


# LLM-generated content at query #47
#--------------------------

```
def test_run_command_no_output_on_success():
    result = run_command(["echo", "hello"], return_output=False)
    assert result.captured_output is None


# LLM-generated content at query #48
#--------------------------

```python
def test_run_command_unicode_decode_error():
    output = b"invalid \xe9 utf-8"
    log_fn = lambda msg: None
    result = run_command("echo", return_output=True, verbose=True, _CONSOLE_LOG_FN=log_fn)
    assert result.captured_output == output


# LLM-generated content at query #49
#--------------------------

```
def test_error_wrapper_with_called_process_error():
    error = subprocess.CalledProcessError(1, 'cmd', output=b'test output')
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Captured output:\n    test output" in str(wrapped_error)

def test_error_wrapper_with_timeout_expired():
    error = subprocess.TimeoutExpired('cmd', 10, output=b'test output')
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert "Captured output:\n    test output" in str(wrapped_error)

def test_error_wrapper_with_other_exception():
    error = ValueError("test error")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, ValueError)
    assert str(wrapped_error) == "test error"

def test_error_wrapper_with_no_output():
    error = subprocess.CalledProcessError(1, 'cmd', output=None)
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_error)

def test_error_wrapper_with_unicode_error():
    error = subprocess.CalledProcessError(1, 'cmd', output=b'\xff')
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_error)


# LLM-generated content at query #50
#--------------------------

```
def test_run_command_unicode_decode_error_handling():
    args = ["echo", "test"]
    env = None
    cwd = None
    timeout = None
    verbose = True
    return_output = False
    ignore_errors = False
    kwargs = {}
    with tempfile.TemporaryFile() as mock_file:
        mock_file.write(b"\xff\xfe\xfd\xfc")  # Invalid UTF-8 bytes
        mock_file.seek(0)
        result = run_command(args, env=env, cwd=cwd, timeout=timeout, verbose=verbose, return_output=return_output, ignore_errors=ignore_errors, **kwargs)
        assert isinstance(result, CommandResult)


# LLM-generated content at query #51
#--------------------------

```
def test_run_command_return_output_true():
    result = run_command("echo test", return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"test" in result.captured_output

def test_run_command_return_code_nonzero():
    result = run_command("false", ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_verbose_true():
    result = run_command("echo verbose_test", verbose=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"verbose_test" in result.captured_output


# LLM-generated content at query #52
#--------------------------

def test_run_command_truncate_long_output():
    args = ["echo", "a" * 10000]
    result = run_command(args, ignore_errors=True, return_output=True)
    assert len(result.captured_output) <= 8192 + len(b"*** (previous output truncated) ***\n")
    assert b"truncated" in result.captured_output


# LLM-generated content at query #53
#--------------------------

```python
def test_run_command_verbose_logging():
    result = run_command("echo Hello, World!", verbose=True)
    assert result.return_code == 0


# LLM-generated content at query #54
#--------------------------

```python
def test_error_wrapper_with_called_process_error():
    err = subprocess.CalledProcessError(1, ['cmd'], output=b'output')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)

def test_error_wrapper_with_timeout_expired():
    err = subprocess.TimeoutExpired(['cmd'], 10, output=b'output')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)

def test_error_wrapper_with_other_exception():
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "test error"

def test_error_wrapper_with_no_output():
    err = subprocess.CalledProcessError(1, ['cmd'])
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

def test_error_wrapper_with_unicode_decode_error():
    err = subprocess.CalledProcessError(1, ['cmd'], output=b'\xff')
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #55
#--------------------------

```python
def test_error_wrapper_predicate_evaluates_to_false():
    class CustomError(Exception):
        pass

    custom_error = CustomError()
    result = error_wrapper(custom_error)
    assert result == custom_error


# LLM-generated content at query #56
#--------------------------

```python
def test_error_wrapper_non_subprocess_error():
    class CustomError(Exception):
        pass
    
    err = CustomError("test error")
    result = error_wrapper(err)
    assert result is err


# LLM-generated content at query #57
#--------------------------

```python
def test_run_command_success_no_output():
    result = run_command(["echo", "test"], return_output=False)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_success_with_output():
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_error_no_ignore_errors():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode == 1
        assert e.output == b""

def test_run_command_error_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.command == ["false"]
    assert result.return_code == 1
    assert result.captured_output == b""

def test_run_command_timeout_no_ignore_errors():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert e.timeout == 0.1
        assert isinstance(e.output, bytes)

def test_run_command_timeout_ignore_errors():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.command == ["sleep", "2"]
    assert result.return_code == -32768
    assert isinstance(result.captured_output, bytes)

def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.command == ["pwd"]
    assert result.return_code == 0
    assert result.captured_output.strip() == b"/tmp"

def test_run_command_with_env():
    result = run_command(["env"], env={"TEST_VAR": "test"}, return_output=True)
    assert result.command == ["env"]
    assert result.return_code == 0
    assert b"TEST_VAR=test" in result.captured_output


# LLM-generated content at query #58
#--------------------------

```
def test_error_wrapper_with_called_process_error():
    err = subprocess.CalledProcessError(1, "cmd", output=b"sample output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)

def test_error_wrapper_with_timeout_expired():
    err = subprocess.TimeoutExpired("cmd", 10, output=b"sample output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)

def test_error_wrapper_with_other_error():
    err = Exception("generic error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert "generic error" in str(wrapped_err)

def test_error_wrapper_with_no_output():
    err = subprocess.CalledProcessError(1, "cmd", output=None)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)


# LLM-generated content at query #59
#--------------------------

```python
def test_log_non_utf8_output():
    output = b"\x80abc"
    log(output.decode('utf-8'), timestamp=False, include_proc_id=False)


# LLM-generated content at query #60
#--------------------------

def test_run_command_success():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output


def test_run_command_error():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is None


def test_run_command_timeout():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None


def test_run_command_verbose():
    result = run_command(["echo", "verbose"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"verbose" in result.captured_output


def test_run_command_with_output():
    result = run_command(["echo", "output"], return_output=True)
    assert result.return_code == 0
    assert b"output" in result.captured_output


def test_run_command_without_output():
    result = run_command(["echo", "no output"])
    assert result.return_code == 0
    assert result.captured_output is None


def test_run_command_with_env():
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=test_value" in result.captured_output


def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.return_code == 0
    assert b"/tmp" in result.captured_output


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_run_command_success():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "hello"

def test_run_command_failure():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode == 1
        assert e.output is not None

def test_run_command_timeout():
    try:
        run_command(["sleep", "10"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert e.timeout == 0.1
        assert e.output is not None

def test_run_command_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None

def test_run_command_return_output():
    result = run_command(["echo", "test"], return_output=True)
    assert result.captured_output.decode('utf-8').strip() == "test"

def test_run_command_verbose():
    result = run_command(["echo", "verbose"], verbose=True)
    assert result.command == ["echo", "verbose"]
    assert result.return_code == 0

def test_run_command_cwd():
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        result = run_command(["pwd"], cwd=temp_dir, return_output=True)
        assert result.captured_output.decode('utf-8').strip() == temp_dir

def test_run_command_env():
    result = run_command(["env"], env={"TEST_ENV": "test_value"}, return_output=True)
    assert b"TEST_ENV=test_value" in result.captured_output

def test_run_command_truncate_output():
    long_output = "a" * 10000
    result = run_command(["echo", long_output], return_output=True)
    assert len(result.captured_output) <= 8192 + len("*** (previous output truncated) ***\n")

def test_run_command_unicode_output():
    result = run_command(["echo", "こんにちは"], return_output=True)
    assert result.captured_output.decode('utf-8').strip() == "こんにちは"


# LLM-generated content at query #2
#--------------------------

def test_run_command_ignore_errors_with_timeout():
    args = ["sleep", "10"]
    result = run_command(args, timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None


# LLM-generated content at query #3
#--------------------------

```
def test_run_command_verbose_logging():
    args = ["echo", "hello"]
    env = None
    cwd = None
    timeout = None
    verbose = True
    return_output = False
    ignore_errors = False
    result = run_command(args, env=env, cwd=cwd, timeout=timeout, verbose=verbose, return_output=return_output, ignore_errors=ignore_errors)
    assert result.return_code == 0
    assert result.captured_output is not None


# LLM-generated content at query #4
#--------------------------

```python
def test_run_command_return_output_true():
    result = run_command("echo 'Hello, World!'", return_output=True)
    assert result.captured_output is not None

def test_run_command_return_code_nonzero():
    result = run_command("false", ignore_errors=True)
    assert result.captured_output is not None

def test_run_command_verbose_true():
    result = run_command("echo 'Hello, World!'", verbose=True)
    assert result.captured_output is not None


# LLM-generated content at query #5
#--------------------------

```
def test_run_command_return_output_true():
    result = run_command("echo test", return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"test" in result.captured_output

def test_run_command_return_code_nonzero():
    result = run_command("false", ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_verbose_true():
    result = run_command("echo verbose_test", verbose=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"verbose_test" in result.captured_output


# LLM-generated content at query #6
#--------------------------

```python
def test_run_command_success_without_output():
    cmd = ["echo", "hello"]
    result = run_command(cmd, return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_success_with_output():
    cmd = ["echo", "hello"]
    result = run_command(cmd, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_failure_with_output():
    cmd = ["false"]
    result = run_command(cmd, return_output=True, ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_timeout_with_output():
    cmd = ["sleep", "2"]
    result = run_command(cmd, timeout=1, return_output=True, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose_logging():
    cmd = ["echo", "hello"]
    result = run_command(cmd, verbose=True)
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_with_custom_env():
    cmd = ["printenv", "TEST_ENV"]
    env = {"TEST_ENV": "test_value"}
    result = run_command(cmd, env=env, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"test_value\n"

def test_run_command_with_custom_cwd():
    cmd = ["pwd"]
    cwd = "/"
    result = run_command(cmd, cwd=cwd, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"/\n"


# LLM-generated content at query #7
#--------------------------

```
def test_run_command_unicode_decode_error_handling():
    args = ["echo", "-e", "\\x80\\x81"]
    result = run_command(args, verbose=True, return_output=True)
    assert isinstance(result, CommandResult)
    assert result.return_code == 0
    assert result.captured_output is not None


# LLM-generated content at query #8
#--------------------------

```python
def test_error_wrapper_with_CalledProcessError():
    err = subprocess.CalledProcessError(1, "cmd", output=b"some output")
    wrapped_err = error_wrapper(err)
    assert "Captured output:" in str(wrapped_err)
    assert "some output" in str(wrapped_err)

def test_error_wrapper_with_TimeoutExpired():
    err = subprocess.TimeoutExpired("cmd", timeout=10, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

def test_error_wrapper_with_other_exception():
    err = ValueError("some error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err == err
    assert "some error" in str(wrapped_err)

def test_error_wrapper_with_no_output():
    err = subprocess.CalledProcessError(1, "cmd", output=None)
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

def test_error_wrapper_with_failed_output_decoding():
    err = subprocess.CalledProcessError(1, "cmd", output=b"\xff")
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #9
#--------------------------

```python
def test_run_command_success():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_failure():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode == 1

def test_run_command_timeout():
    try:
        run_command(["sleep", "10"], timeout=0.1)
    except subprocess.TimeoutExpired:
        pass
    else:
        assert False, "Expected TimeoutExpired exception"

def test_run_command_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code == 1

def test_run_command_with_output():
    result = run_command(["echo", "test"], return_output=True)
    assert result.captured_output is not None
    assert b"test" in result.captured_output

def test_run_command_with_env():
    result = run_command(["printenv", "TEST_VAR"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"test_value" in result.captured_output

def test_run_command_invalid_command():
    try:
        run_command(["nonexistent_command"])
    except subprocess.CalledProcessError:
        pass
    else:
        assert False, "Expected CalledProcessError exception"

def test_run_command_verbose(capsys):
    run_command(["echo", "verbose"], verbose=True)
    captured = capsys.readouterr()
    assert "verbose" in captured.out

def test_run_command_cwd(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("test")
    result = run_command(["cat", "test.txt"], cwd=tmp_path, return_output=True)
    assert b"test" in result.captured_output


# LLM-generated content at query #10
#--------------------------

```
def test_run_command_unicode_decode_error_handling():
    mock_log = MagicMock()
    mock_subprocess_run = MagicMock(return_value=MagicMock(returncode=0))
    mock_tempfile = MagicMock()
    mock_tempfile.read.return_value = b'\xff\xfe\xfd'
    mock_tempfile.seek.return_value = None
    with patch('subprocess.run', mock_subprocess_run), \
         patch('tempfile.TemporaryFile', return_value=mock_tempfile), \
         patch('flutes.log.log', mock_log):
        run_command("echo hello", verbose=True)
        assert mock_log.call_count == 3


# LLM-generated content at query #11
#--------------------------

```python
def test_error_wrapper_returns_original_error_when_not_subprocess_error():
    err = ValueError("Test error")
    result = error_wrapper(err)
    assert result is err

def test_error_wrapper_wraps_subprocess_called_process_error():
    err = subprocess.CalledProcessError(1, "test_command")
    result = error_wrapper(err)
    assert isinstance(result, subprocess.CalledProcessError)

def test_error_wrapper_wraps_subprocess_timeout_expired():
    err = subprocess.TimeoutExpired("test_command", 1)
    result = error_wrapper(err)
    assert isinstance(result, subprocess.TimeoutExpired)


# LLM-generated content at query #12
#--------------------------

```python
def test_run_command_success():
    result = run_command("echo Hello, World!", return_output=True)
    assert result.command == "echo Hello, World!"
    assert result.return_code == 0
    assert result.captured_output == b"Hello, World!\n"

def test_run_command_failure():
    try:
        run_command("false")
    except subprocess.CalledProcessError as e:
        result = error_wrapper(e)
        assert isinstance(result, subprocess.CalledProcessError)
        assert result.returncode == 1
        assert result.output == b""

def test_run_command_timeout():
    try:
        run_command("sleep 2", timeout=0.1)
    except subprocess.TimeoutExpired as e:
        result = error_wrapper(e)
        assert isinstance(result, subprocess.TimeoutExpired)
        assert result.output is not None

def test_run_command_ignore_errors():
    result = run_command("false", ignore_errors=True)
    assert result.command == "false"
    assert result.return_code == 1
    assert result.captured_output == b""

def test_run_command_verbose():
    result = run_command("echo Hello, World!", verbose=True, return_output=True)
    assert result.command == "echo Hello, World!"
    assert result.return_code == 0
    assert result.captured_output == b"Hello, World!\n"

def test_run_command_with_env():
    env = {"TEST_ENV": "test_value"}
    result = run_command("echo $TEST_ENV", env=env, shell=True, return_output=True)
    assert result.command == "echo $TEST_ENV"
    assert result.return_code == 0
    assert result.captured_output == b"test_value\n"

def test_run_command_with_cwd(tmpdir):
    tmpdir.join("test_file.txt").write("Hello, World!")
    result = run_command("cat test_file.txt", cwd=str(tmpdir), return_output=True)
    assert result.command == "cat test_file.txt"
    assert result.return_code == 0
    assert result.captured_output == b"Hello, World!"


# LLM-generated content at query #13
#--------------------------

```python
def test_truncate_long_output():
    # Create a mock subprocess.CalledProcessError with long output
    mock_output = b"a" * (MAX_OUTPUT_LENGTH + 100)
    mock_error = subprocess.CalledProcessError(1, "mock_command", output=mock_output)
    
    # Mock tempfile behavior
    with tempfile.TemporaryFile() as mock_file:
        mock_file.write(mock_output)
        mock_file.seek(0)
        
        # Call the function with conditions that trigger line 32
        result = run_command("mock_command", ignore_errors=True)
        
        # Verify output was truncated
        assert len(result.captured_output) == MAX_OUTPUT_LENGTH + len(b"*** (previous output truncated) ***\n")
        assert result.captured_output.startswith(b"*** (previous output truncated) ***\n")


# LLM-generated content at query #14
#--------------------------

```python
def test_run_command_does_not_decode_non_utf8_output():
    result = run_command(["echo", "-n", "\xff"], return_output=True, verbose=True)
    assert result.captured_output == b"\xff"


# LLM-generated content at query #15
#--------------------------

def test_run_command_success():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output


def test_run_command_failure():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is None


def test_run_command_timeout():
    result = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None


def test_run_command_with_output():
    result = run_command(["echo", "test"], return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output


def test_run_command_verbose():
    result = run_command(["echo", "verbose"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"verbose" in result.captured_output


def test_run_command_with_env():
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=test_value" in result.captured_output


def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.return_code == 0
    assert b"/tmp" in result.captured_output


# LLM-generated content at query #16
#--------------------------

```python
def test_captured_output_truncated_when_exceeding_max_length():
    mock_output = b"a" * (8192 + 100)
    mock_file = object()
    mock_file.seek = lambda x: None
    mock_file.read = lambda: mock_output
    result = run_command(["echo", "test"], return_output=True, verbose=False, ignore_errors=True)
    assert len(result.captured_output) == 8192 + len(b"*** (previous output truncated) ***\n")


# LLM-generated content at query #17
#--------------------------

```python
def test_error_wrapper_with_called_process_error():
    err = subprocess.CalledProcessError(1, "cmd", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)

def test_error_wrapper_with_timeout_expired():
    err = subprocess.TimeoutExpired("cmd", 1, output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)

def test_error_wrapper_with_other_exception():
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err == err
    assert str(wrapped_err) == "test error"

def test_error_wrapper_with_no_output():
    err = subprocess.CalledProcessError(1, "cmd", output=None)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

def test_error_wrapper_with_failed_output_decoding():
    err = subprocess.CalledProcessError(1, "cmd", output=b"\xff")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #18
#--------------------------

def test_run_command_success():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output
    assert result.command == ["echo", "hello"]

def test_run_command_failure():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is None

def test_run_command_timeout():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command(["echo", "verbose"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"verbose" in result.captured_output

def test_run_command_with_env():
    result = run_command(["printenv", "TEST_VAR"], env={"TEST_VAR": "123"}, return_output=True)
    assert result.return_code == 0
    assert b"123" in result.captured_output

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.return_code == 0
    assert b"/tmp" in result.captured_output

def test_run_command_shell():
    result = run_command("echo $SHELL", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"/bin/" in result.captured_output

def test_run_command_output_truncation():
    long_output = b"a" * 10000
    with tempfile.NamedTemporaryFile() as f:
        f.write(long_output)
        f.flush()
        result = run_command(["cat", f.name], ignore_errors=True, return_output=True)
    assert result.return_code == 0
    assert b"truncated" in result.captured_output


# LLM-generated content at query #19
#--------------------------

Here are the test cases to ensure the predicate at line 25 evaluates to True:


# LLM-generated content at query #20
#--------------------------

```python
def test_error_wrapper_returns_same_error_for_non_subprocess_errors():
    class CustomError(Exception):
        pass
    
    custom_error = CustomError("Test error")
    wrapped_error = error_wrapper(custom_error)
    assert wrapped_error is custom_error

def test_error_wrapper_wraps_subprocess_called_process_error():
    error = subprocess.CalledProcessError(1, "command", output=b"output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_error)

def test_error_wrapper_wraps_subprocess_timeout_expired_error():
    error = subprocess.TimeoutExpired("command", 10, output=b"output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_error)

def test_error_wrapper_handles_unicode_decode_error():
    error = subprocess.CalledProcessError(1, "command", output=b"\xff\xff\xff")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_error)

def test_error_wrapper_handles_no_output():
    error = subprocess.CalledProcessError(1, "command", output=None)
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_error)


# LLM-generated content at query #21
#--------------------------

```python
def test_run_command_with_invalid_utf8_output():
    result = run_command(["echo", "-n", "\\xff"], return_output=True, verbose=True)
    assert result.captured_output == b"\\xff"


# LLM-generated content at query #22
#--------------------------

```python
def test_log_message_with_verbose_and_timestamp_false():
    args = ["echo", "Hello World"]
    cwd = "/tmp"
    verbose = True
    run_command(args, cwd=cwd, verbose=verbose)


# LLM-generated content at query #23
#--------------------------

```python
def test_error_wrapper_returns_err_when_not_subprocess_error():
    class CustomError(Exception):
        pass

    custom_err = CustomError("Test error")
    result = error_wrapper(custom_err)
    assert result == custom_err


# LLM-generated content at query #24
#--------------------------

```
def test_error_wrapper_predicate():
    class MockError(Exception):
        pass

    mock_error = MockError()
    result = error_wrapper(mock_error)
    assert result is mock_error


# LLM-generated content at query #25
#--------------------------

```python
def test_error_wrapper_with_called_process_error():
    err = subprocess.CalledProcessError(1, ['cmd'], output=b'some output')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert 'Captured output:' in str(wrapped_err)

def test_error_wrapper_with_timeout_expired():
    err = subprocess.TimeoutExpired(['cmd'], 10, output=b'timeout output')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert 'Captured output:' in str(wrapped_err)

def test_error_wrapper_with_other_exception():
    err = ValueError('test error')
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert 'test error' in str(wrapped_err)

def test_error_wrapper_with_no_output():
    err = subprocess.CalledProcessError(1, ['cmd'], output=None)
    wrapped_err = error_wrapper(err)
    assert 'No output was generated.' in str(wrapped_err)

def test_error_wrapper_with_unicode_decode_error():
    err = subprocess.CalledProcessError(1, ['cmd'], output=b'\xff')
    wrapped_err = error_wrapper(err)
    assert 'Failed to parse output.' in str(wrapped_err)


# LLM-generated content at query #26
#--------------------------

```python
def test_error_wrapper_returns_non_subprocess_error():
    class CustomError(Exception):
        pass

    custom_error = CustomError("Custom error")
    result = error_wrapper(custom_error)
    assert result == custom_error

def test_error_wrapper_returns_subprocess_error_with_output():
    process_error = subprocess.CalledProcessError(1, "cmd", output=b"output")
    result = error_wrapper(process_error)
    assert isinstance(result, subprocess.CalledProcessError)

def test_error_wrapper_returns_subprocess_error_without_output():
    process_error = subprocess.CalledProcessError(1, "cmd", output=None)
    result = error_wrapper(process_error)
    assert isinstance(result, subprocess.CalledProcessError)

def test_error_wrapper_returns_timeout_expired_error():
    timeout_error = subprocess.TimeoutExpired("cmd", 1)
    result = error_wrapper(timeout_error)
    assert isinstance(result, subprocess.TimeoutExpired)


# LLM-generated content at query #27
#--------------------------

```
def test_error_wrapper_returns_non_subprocess_errors_unchanged():
    class CustomError(Exception):
        pass
    
    custom_error = CustomError("test error")
    result = error_wrapper(custom_error)
    assert result is custom_error

def test_error_wrapper_wraps_called_process_error():
    try:
        raise subprocess.CalledProcessError(1, "cmd")
    except subprocess.CalledProcessError as e:
        original_str = str(e)
        wrapped = error_wrapper(e)
        assert str(wrapped) != original_str
        assert isinstance(wrapped, subprocess.CalledProcessError)

def test_error_wrapper_wraps_timeout_expired_error():
    try:
        raise subprocess.TimeoutExpired("cmd", 10)
    except subprocess.TimeoutExpired as e:
        original_str = str(e)
        wrapped = error_wrapper(e)
        assert str(wrapped) != original_str
        assert isinstance(wrapped, subprocess.TimeoutExpired)


# LLM-generated content at query #28
#--------------------------

```python
def test_error_wrapper_with_CalledProcessError():
    err = subprocess.CalledProcessError(1, 'cmd', output=b'error output')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)

def test_error_wrapper_with_TimeoutExpired():
    err = subprocess.TimeoutExpired('cmd', 10, output=b'timeout output')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)

def test_error_wrapper_with_other_exception():
    err = ValueError("Some error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err == err
    assert "Captured output:" not in str(wrapped_err)

def test_error_wrapper_with_no_output():
    err = subprocess.CalledProcessError(1, 'cmd', output=None)
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)


# LLM-generated content at query #29
#--------------------------

```python
def test_error_wrapper_with_CalledProcessError():
    err = subprocess.CalledProcessError(1, 'cmd', output=b'mock output')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)

def test_error_wrapper_with_TimeoutExpired():
    err = subprocess.TimeoutExpired('cmd', 10, output=b'mock output')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)

def test_error_wrapper_with_other_exception():
    err = ValueError('mock error')
    wrapped_err = error_wrapper(err)
    assert wrapped_err == err


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_run_command_success():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_failure():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0

def test_run_command_timeout():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768

def test_run_command_output_truncation():
    long_output = "a" * 10000
    result = run_command(["python", "-c", f"print('{long_output}')"], return_output=True, ignore_errors=True)
    assert b"truncated" in result.captured_output

def test_run_command_verbose():
    result = run_command(["echo", "verbose"], verbose=True, return_output=True)
    assert b"verbose" in result.captured_output

def test_run_command_env():
    result = run_command(["python", "-c", "import os; print(os.getenv('TEST_VAR'))"], env={"TEST_VAR": "test"}, return_output=True)
    assert b"test" in result.captured_output

def test_run_command_cwd(tmp_path):
    (tmp_path / "test.txt").write_text("content")
    result = run_command(["cat", "test.txt"], cwd=tmp_path, return_output=True)
    assert b"content" in result.captured_output

def test_run_command_shell():
    result = run_command("echo hello", shell=True, return_output=True)
    assert b"hello" in result.captured_output


# LLM-generated content at query #2
#--------------------------

```python
import subprocess
from flutes.run import run_command

def test_run_command_return_output_true():
    result = run_command("echo hello", return_output=True)
    assert result.captured_output is not None

def test_run_command_return_code_nonzero():
    result = run_command("exit 1", ignore_errors=True)
    assert result.captured_output is not None

def test_run_command_verbose_true():
    result = run_command("echo hello", verbose=True)
    assert result.captured_output is not None


# LLM-generated content at query #3
#--------------------------

```python
def test_log_non_utf8_output():
    non_utf8_output = b"\xff\xfe\xfd"
    run_command(["echo", "test"], verbose=True, return_output=True, ignore_errors=True, stdout=non_utf8_output)


# LLM-generated content at query #4
#--------------------------

```python
def test_captured_output_truncated_when_exceeding_max_length():
    output = b"a" * (MAX_OUTPUT_LENGTH + 100)
    truncated_output = b"*** (previous output truncated) ***\n" + output[-MAX_OUTPUT_LENGTH:]
    assert truncated_output == b"*** (previous output truncated) ***\n" + b"a" * MAX_OUTPUT_LENGTH


# LLM-generated content at query #5
#--------------------------

```python
def test_run_command_return_output_true():
    result = run_command("echo hello", return_output=True)
    assert result.captured_output is not None

def test_run_command_return_code_nonzero():
    result = run_command("false", ignore_errors=True)
    assert result.captured_output is not None

def test_run_command_verbose_true():
    result = run_command("echo hello", verbose=True)
    assert result.captured_output is not None


# LLM-generated content at query #6
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_failure():
    result = run_command(["ls", "nonexistent"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_timeout():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command(["echo", "verbose"], verbose=True)
    assert result.return_code == 0

def test_run_command_return_output():
    result = run_command(["echo", "output"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"output\n"

def test_run_command_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None


# LLM-generated content at query #7
#--------------------------

```python
def test_run_command_success():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output


def test_run_command_failure():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0


def test_run_command_timeout():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768


def test_run_command_output_truncation():
    long_output = b"a" * (8192 + 100)
    result = run_command(["python", "-c", f"print('{long_output.decode()}')"], ignore_errors=True, return_output=True)
    assert b"*** (previous output truncated) ***" in result.captured_output


def test_run_command_verbose():
    result = run_command(["echo", "verbose test"], verbose=True, return_output=True)
    assert b"verbose test" in result.captured_output


def test_run_command_env_vars():
    result = run_command(["printenv", "TEST_VAR"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"test_value" in result.captured_output


def test_run_command_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert b"/tmp" in result.captured_output


# LLM-generated content at query #8
#--------------------------

```python
def test_run_command_success():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"hello" in result.captured_output

def test_run_command_failure():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_timeout():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command(["echo", "verbose"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"verbose" in result.captured_output

def test_run_command_no_output():
    result = run_command(["true"])
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"/tmp" in result.captured_output

def test_run_command_with_env():
    result = run_command(["printenv", "TEST_ENV"], env={"TEST_ENV": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"test_value" in result.captured_output


# LLM-generated content at query #9
#--------------------------

```python
def test_run_command_with_verbose():
    result = run_command("echo Hello, World!", verbose=True)
    assert result.return_code == 0


# LLM-generated content at query #10
#--------------------------

def test_run_command_success():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output
    assert result.command == ["echo", "hello"]

def test_run_command_failure():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is None

def test_run_command_timeout():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command(["echo", "verbose"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"verbose" in result.captured_output

def test_run_command_with_env():
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=test_value" in result.captured_output

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.return_code == 0
    assert b"/tmp" in result.captured_output

def test_run_command_shell():
    result = run_command("echo $SHELL", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"/bin/bash" in result.captured_output or b"/bin/sh" in result.captured_output

def test_run_command_truncated_output():
    long_output = b"a" * (8192 + 100)
    result = run_command(["python", "-c", f"print('{long_output.decode()}')"], return_output=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #11
#--------------------------

```python
def test_run_command_success():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_failure():
    result = run_command(["false"], ignore_errors=True)
    assert result.command == ["false"]
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_timeout():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.command == ["sleep", "2"]
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose(capsys):
    run_command(["echo", "hello"], verbose=True)
    captured = capsys.readouterr()
    assert "hello" in captured.out

def test_run_command_return_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.captured_output == b"hello\n"

def test_run_command_with_env():
    env = {"TEST_ENV": "123"}
    result = run_command(["env"], env=env, return_output=True)
    assert b"TEST_ENV=123" in result.captured_output

def test_run_command_with_cwd():
    import os
    cwd = os.getcwd()
    result = run_command(["pwd"], cwd=cwd, return_output=True)
    assert result.captured_output.decode("utf-8").strip() == cwd

def test_run_command_truncate_output():
    long_output = "a" * 10000
    result = run_command(["echo", long_output], ignore_errors=True)
    assert len(result.captured_output) <= 8192 + len("*** (previous output truncated) ***\n")


# LLM-generated content at query #12
#--------------------------

```python
def test_run_command_unicode_decode_error_handling():
    output = b"\xff\xfe\xfd"  # Invalid UTF-8 bytes
    result = run_command("echo 'invalid'", return_output=True, verbose=True)
    assert result.captured_output == output


# LLM-generated content at query #13
#--------------------------

```python
def test_run_command_success():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_timeout():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert isinstance(e, subprocess.TimeoutExpired)

def test_run_command_error():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert isinstance(e, subprocess.CalledProcessError)

def test_run_command_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code == 1

def test_run_command_verbose():
    result = run_command(["echo", "hello"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_return_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_no_output():
    result = run_command(["true"])
    assert result.return_code == 0
    assert result.captured_output is None


# LLM-generated content at query #14
#--------------------------

```python
def test_predicate_evaluates_to_true_when_return_output_is_true():
    args = ["echo", "hello"]
    result = run_command(args, return_output=True)
    assert result.captured_output is not None

def test_predicate_evaluates_to_true_when_return_code_is_non_zero():
    args = ["false"]
    result = run_command(args)
    assert result.captured_output is not None

def test_predicate_evaluates_to_true_when_verbose_is_true():
    args = ["echo", "hello"]
    result = run_command(args, verbose=True)
    assert result.captured_output is not None


# LLM-generated content at query #15
#--------------------------

```python
def test_run_command_predicate_evaluates_to_true():
    args = ["echo", "test"]
    env = {}
    cwd = None
    timeout = None
    verbose = True
    return_output = True
    ignore_errors = False
    result = run_command(args, env=env, cwd=cwd, timeout=timeout, verbose=verbose, return_output=return_output, ignore_errors=ignore_errors)
    assert result.captured_output is not None


# LLM-generated content at query #16
#--------------------------

```python
def test_error_wrapper_with_called_process_error():
    err = subprocess.CalledProcessError(1, ['cmd'], output=b'some output')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:\n    some output" in str(wrapped_err)

def test_error_wrapper_with_timeout_expired():
    err = subprocess.TimeoutExpired(['cmd'], timeout=10, output=b'timeout output')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:\n    timeout output" in str(wrapped_err)

def test_error_wrapper_with_other_exception():
    err = ValueError("Some error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err == err
    assert str(wrapped_err) == "Some error"

def test_error_wrapper_with_no_output():
    err = subprocess.CalledProcessError(1, ['cmd'], output=None)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

def test_error_wrapper_with_unicode_error():
    err = subprocess.CalledProcessError(1, ['cmd'], output=b'\xff')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #17
#--------------------------

Here are the test cases to ensure the predicate at line 40 evaluates to True:


# LLM-generated content at query #18
#--------------------------

```python
def test_error_wrapper_non_subprocess_error():
    class CustomError(Exception):
        pass
    
    custom_error = CustomError("Test error")
    result = error_wrapper(custom_error)
    assert result is custom_error


# LLM-generated content at query #19
#--------------------------

```python
def test_log_non_utf8_output():
    output = b'\x80abc'
    log(output.decode('utf-8'), timestamp=False, include_proc_id=False)


# LLM-generated content at query #20
#--------------------------

```python
def test_run_command_success():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_failure():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode == 1

def test_run_command_timeout():
    try:
        run_command(["sleep", "10"], timeout=0.1)
    except subprocess.TimeoutExpired:
        pass
    else:
        assert False, "Expected TimeoutExpired exception"

def test_run_command_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code == 1

def test_run_command_with_output():
    result = run_command(["echo", "test"], return_output=True)
    assert result.captured_output is not None
    assert b"test" in result.captured_output

def test_run_command_with_env():
    result = run_command(["printenv", "TEST_VAR"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"test_value" in result.captured_output

def test_run_command_invalid_command():
    try:
        run_command(["nonexistent_command"])
    except subprocess.CalledProcessError:
        pass
    else:
        assert False, "Expected CalledProcessError exception"

def test_run_command_verbose(capsys):
    run_command(["echo", "verbose"], verbose=True)
    captured = capsys.readouterr()
    assert "verbose" in captured.out

def test_run_command_cwd(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("content")
    result = run_command(["cat", "test.txt"], cwd=tmp_path, return_output=True)
    assert b"content" in result.captured_output


# LLM-generated content at query #21
#--------------------------

```
def test_error_wrapper_returns_err_when_not_subprocess_error():
    class CustomException(Exception):
        pass

    custom_err = CustomException("test error")
    result = error_wrapper(custom_err)
    assert result is custom_err


# LLM-generated content at query #22
#--------------------------

```python
def test_error_wrapper_predicate_at_line_3():
    err = subprocess.CalledProcessError(1, "cmd")
    result = error_wrapper(err)
    assert isinstance(result, subprocess.CalledProcessError)


# LLM-generated content at query #23
#--------------------------

```python
def test_log_unicode_decode_error():
    output = b'\xc3\x28'  # Invalid UTF-8 sequence
    log(output.decode('utf-8'), timestamp=False, include_proc_id=False)


# LLM-generated content at query #24
#--------------------------

```python
def test_error_wrapper_with_called_process_error():
    class MockCalledProcessError(subprocess.CalledProcessError):
        def __init__(self, returncode, cmd, output):
            super().__init__(returncode, cmd)
            self.output = output

    err = MockCalledProcessError(1, "mock_command", b"mock_output")
    wrapped_err = error_wrapper(err)
    assert wrapped_err.__class__.__name__ == "CalledProcessError"
    assert "Captured output:" in str(wrapped_err)

def test_error_wrapper_with_timeout_expired():
    class MockTimeoutExpired(subprocess.TimeoutExpired):
        def __init__(self, cmd, timeout, output):
            super().__init__(cmd, timeout)
            self.output = output

    err = MockTimeoutExpired("mock_command", 10, b"mock_output")
    wrapped_err = error_wrapper(err)
    assert wrapped_err.__class__.__name__ == "TimeoutExpired"
    assert "Captured output:" in str(wrapped_err)

def test_error_wrapper_with_other_error():
    err = ValueError("mock_error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err == err


# LLM-generated content at query #25
#--------------------------

```python
def test_run_command_with_non_utf8_output():
    import subprocess
    import tempfile
    result = run_command(["echo", "-n", "\xff"], return_output=True, verbose=True)
    assert result.captured_output is not None
    assert isinstance(result.captured_output, bytes)


# LLM-generated content at query #26
#--------------------------

```
def test_run_command_no_unicode_decode_error():
    result = run_command(["echo", "hello"], return_output=True, verbose=True)
    assert isinstance(result, CommandResult)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"hello" in result.captured_output

def test_run_command_with_unicode_decode_error():
    # Create a binary output that cannot be decoded as UTF-8
    binary_output = b"\xff\xfe\xfd"
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(binary_output)
        tmp.flush()
        result = run_command(["cat", tmp.name], return_output=True, verbose=True)
    assert isinstance(result, CommandResult)
    assert result.return_code == 0
    assert result.captured_output == binary_output


# LLM-generated content at query #27
#--------------------------

```
def test_error_wrapper_wraps_called_process_error():
    import subprocess
    err = subprocess.CalledProcessError(1, "cmd")
    wrapped = error_wrapper(err)
    assert wrapped is err
    assert isinstance(wrapped, type(err))

def test_error_wrapper_wraps_timeout_expired():
    import subprocess
    err = subprocess.TimeoutExpired("cmd", 1)
    wrapped = error_wrapper(err)
    assert wrapped is err
    assert isinstance(wrapped, type(err))

def test_error_wrapper_does_not_wrap_other_exceptions():
    err = ValueError("test")
    wrapped = error_wrapper(err)
    assert wrapped is err


# LLM-generated content at query #28
#--------------------------

```python
def test_error_wrapper_with_called_process_error():
    err = subprocess.CalledProcessError(1, 'cmd', output=b'test output')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)

def test_error_wrapper_with_timeout_expired():
    err = subprocess.TimeoutExpired('cmd', 10, output=b'test output')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)

def test_error_wrapper_with_other_exception():
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

def test_error_wrapper_with_no_output():
    err = subprocess.CalledProcessError(1, 'cmd', output=None)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

def test_error_wrapper_with_failed_output_decoding():
    err = subprocess.CalledProcessError(1, 'cmd', output=b'\xff')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #29
#--------------------------

```python
def test_error_wrapper_returns_original_error_when_not_subprocess_error():
    err = ValueError("Test error")
    result = error_wrapper(err)
    assert result == err

def test_error_wrapper_returns_original_error_when_not_called_process_error_or_timeout_expired():
    err = RuntimeError("Test error")
    result = error_wrapper(err)
    assert result == err


# LLM-generated content at query #30
#--------------------------

```python
def test_error_wrapper_returns_non_subprocess_error():
    err = ValueError("Test error")
    result = error_wrapper(err)
    assert result == err

def test_error_wrapper_returns_subprocess_error_with_output():
    err = subprocess.CalledProcessError(1, "cmd", output=b"test output")
    result = error_wrapper(err)
    assert isinstance(result, type(err))
    assert result.__class__.__name__ == err.__class__.__name__

def test_error_wrapper_returns_subprocess_error_without_output():
    err = subprocess.CalledProcessError(1, "cmd", output=None)
    result = error_wrapper(err)
    assert isinstance(result, type(err))
    assert result.__class__.__name__ == err.__class__.__name__

def test_error_wrapper_returns_timeout_expired_error():
    err = subprocess.TimeoutExpired("cmd", timeout=10)
    result = error_wrapper(err)
    assert isinstance(result, type(err))
    assert result.__class__.__name__ == err.__class__.__name__


# LLM-generated content at query #31
#--------------------------

```python
def test_error_wrapper_non_subprocess_error():
    class CustomError(Exception):
        pass
    
    custom_err = CustomError("Test error")
    result = error_wrapper(custom_err)
    assert result is custom_err


# LLM-generated content at query #32
#--------------------------

```python
def test_error_wrapper_with_called_process_error():
    err = subprocess.CalledProcessError(1, ['cmd'], output=b'some output')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert 'Captured output:' in str(wrapped_err)

def test_error_wrapper_with_timeout_expired():
    err = subprocess.TimeoutExpired(['cmd'], 10, output=b'timeout output')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert 'Captured output:' in str(wrapped_err)

def test_error_wrapper_with_other_exception():
    err = ValueError("Some error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == "Some error"

def test_error_wrapper_with_no_output():
    err = subprocess.CalledProcessError(1, ['cmd'], output=None)
    wrapped_err = error_wrapper(err)
    assert 'No output was generated.' in str(wrapped_err)

def test_error_wrapper_with_unicode_decode_error():
    err = subprocess.CalledProcessError(1, ['cmd'], output=b'\xff')
    wrapped_err = error_wrapper(err)
    assert 'Failed to parse output.' in str(wrapped_err)


# LLM-generated content at query #33
#--------------------------

```python
def test_error_wrapper_wraps_called_process_error():
    err = subprocess.CalledProcessError(1, "cmd")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)

def test_error_wrapper_wraps_timeout_expired():
    err = subprocess.TimeoutExpired("cmd", 10)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)

def test_error_wrapper_does_not_wrap_other_exceptions():
    err = ValueError("Test error")
    wrapped_err = error_wrapper(err)
    assert wrapped_err == err


# LLM-generated content at query #34
#--------------------------

```python
def test_error_wrapper_returns_err_when_not_subprocess_error():
    class CustomException(Exception):
        pass

    custom_exception = CustomException("Custom error")
    result = error_wrapper(custom_exception)
    assert result == custom_exception


# LLM-generated content at query #35
#--------------------------

```python
def test_error_wrapper_with_called_process_error():
    err = subprocess.CalledProcessError(returncode=1, cmd="ls", output=b"file1\nfile2")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "    file1" in str(wrapped_err)
    assert "    file2" in str(wrapped_err)

def test_error_wrapper_with_timeout_expired():
    err = subprocess.TimeoutExpired(cmd="sleep", timeout=10, output=b"timeout")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "    timeout" in str(wrapped_err)

def test_error_wrapper_with_other_exception():
    err = ValueError("Some error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "Some error"

def test_error_wrapper_with_no_output():
    err = subprocess.CalledProcessError(returncode=1, cmd="ls", output=None)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

def test_error_wrapper_with_failed_output_decoding():
    err = subprocess.CalledProcessError(returncode=1, cmd="ls", output=b"\xff")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #36
#--------------------------

```python
def test_error_wrapper_predicate_false():
    err = Exception("Test exception")
    result = error_wrapper(err)
    assert result == err


# LLM-generated content at query #37
#--------------------------

```python
def test_error_wrapper_with_called_process_error():
    err = subprocess.CalledProcessError(1, ['cmd'], output=b'some output')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert 'Captured output:' in str(wrapped_err)

def test_error_wrapper_with_timeout_expired():
    err = subprocess.TimeoutExpired(['cmd'], 10, output=b'some output')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert 'Captured output:' in str(wrapped_err)

def test_error_wrapper_with_other_exception():
    err = ValueError('test error')
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == 'test error'

def test_error_wrapper_with_no_output():
    err = subprocess.CalledProcessError(1, ['cmd'], output=None)
    wrapped_err = error_wrapper(err)
    assert 'No output was generated.' in str(wrapped_err)

def test_error_wrapper_with_unicode_decode_error():
    err = subprocess.CalledProcessError(1, ['cmd'], output=b'\xff')
    wrapped_err = error_wrapper(err)
    assert 'Failed to parse output.' in str(wrapped_err)


# LLM-generated content at query #38
#--------------------------

```python
def test_log_non_utf8_output():
    non_utf8_output = b'\x80abc'
    log(non_utf8_output.decode('utf-8'), timestamp=False, include_proc_id=False)


# LLM-generated content at query #39
#--------------------------

```
def test_run_command_verbose_logging():
    args = ["echo", "hello"]
    result = run_command(args, verbose=True)
    assert isinstance(result, CommandResult)
    assert result.return_code == 0
    assert result.captured_output is not None


# LLM-generated content at query #40
#--------------------------

```python
def test_run_command_success_without_output():
    result = run_command("echo Hello", verbose=False, return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None
    assert result.command == "echo Hello"

def test_run_command_success_with_output():
    result = run_command("echo Hello", verbose=False, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"Hello\n"
    assert result.command == "echo Hello"

def test_run_command_failure_with_output():
    result = run_command("exit 1", verbose=False, return_output=True, ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None
    assert result.command == "exit 1"

def test_run_command_timeout_with_output():
    result = run_command("sleep 2", timeout=1, verbose=False, return_output=True, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None
    assert result.command == "sleep 2"

def test_run_command_verbose():
    result = run_command("echo Hello", verbose=True, return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None
    assert result.command == "echo Hello"

def test_run_command_with_cwd():
    result = run_command("echo Hello", cwd="/tmp", verbose=False, return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None
    assert result.command == "echo Hello"

def test_run_command_with_env():
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "Hello"}, verbose=False, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"Hello\n"
    assert result.command == "echo $TEST_VAR"

def test_run_command_with_shell():
    result = run_command("echo Hello", shell=True, verbose=False, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"Hello\n"
    assert result.command == "echo Hello"


# LLM-generated content at query #41
#--------------------------

```python
def test_error_wrapper_returns_non_subprocess_error():
    err = ValueError("Some error")
    assert error_wrapper(err) == err

def test_error_wrapper_returns_subprocess_error():
    err = subprocess.CalledProcessError(returncode=1, cmd="ls")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)

def test_error_wrapper_returns_timeout_expired_error():
    err = subprocess.TimeoutExpired(cmd="ls", timeout=10)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)


# LLM-generated content at query #42
#--------------------------

```python
def test_return_output_true():
    result = run_command("echo hello", return_output=True)
    assert result.captured_output is not None

def test_return_code_nonzero():
    result = run_command("exit 1", ignore_errors=True)
    assert result.captured_output is not None

def test_verbose_true():
    result = run_command("echo hello", verbose=True)
    assert result.captured_output is not None


# LLM-generated content at query #43
#--------------------------

```python
def test_error_wrapper_returns_original_error_when_not_subprocess_error():
    class CustomError(Exception):
        pass

    custom_error = CustomError("Custom error message")
    result = error_wrapper(custom_error)
    assert result == custom_error


# LLM-generated content at query #44
#--------------------------

def test_log_unicode_decode_error():
    output = b"\xff\xfe\xfd"  # Invalid UTF-8 bytes
    log(output.decode('utf-8'), timestamp=False, include_proc_id=False)


# LLM-generated content at query #45
#--------------------------

```python
def test_error_wrapper_non_subprocess_error():
    class CustomError(Exception):
        pass
    
    custom_error = CustomError("test error")
    result = error_wrapper(custom_error)
    assert result is custom_error


# LLM-generated content at query #46
#--------------------------

```
def test_error_wrapper_returns_same_error_when_not_subprocess_error():
    class CustomError(Exception):
        pass

    custom_error = CustomError("test")
    result = error_wrapper(custom_error)
    assert result is custom_error

def test_error_wrapper_returns_wrapped_error_when_called_process_error():
    error = subprocess.CalledProcessError(1, "cmd")
    result = error_wrapper(error)
    assert result is error
    assert isinstance(result, type(error))

def test_error_wrapper_returns_wrapped_error_when_timeout_expired():
    error = subprocess.TimeoutExpired("cmd", 10)
    result = error_wrapper(error)
    assert result is error
    assert isinstance(result, type(error))


# LLM-generated content at query #47
#--------------------------

```python
def test_run_command_verbose_logs_command():
    args = ["echo", "hello"]
    result = run_command(args, verbose=True)
    assert isinstance(result, CommandResult)


# LLM-generated content at query #48
#--------------------------

```python
def test_run_command_success():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_failure():
    result = run_command(["false"], ignore_errors=True)
    assert result.command == ["false"]
    assert result.return_code == 1
    assert result.captured_output is None

def test_run_command_timeout():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.command == ["sleep", "2"]
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_return_output():
    result = run_command(["echo", "world"], return_output=True)
    assert result.command == ["echo", "world"]
    assert result.return_code == 0
    assert result.captured_output == b"world\n"

def test_run_command_verbose():
    result = run_command(["echo", "verbose"], verbose=True, return_output=True)
    assert result.command == ["echo", "verbose"]
    assert result.return_code == 0
    assert result.captured_output == b"verbose\n"

def test_run_command_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.command == ["false"]
    assert result.return_code == 1
    assert result.captured_output is None


# LLM-generated content at query #49
#--------------------------

```python
def test_log_handles_unicode_decode_error():
    args = ["echo", "test"]
    result = run_command(args, verbose=True, return_output=True)
    assert isinstance(result, CommandResult)
    assert result.command == args
    assert result.return_code == 0
    assert result.captured_output is not None


# LLM-generated content at query #50
#--------------------------

```python
def test_output_truncated_when_longer_than_max_output_length():
    args = "echo " + "a" * (8192 + 1)
    result = run_command(args, return_output=True, ignore_errors=True)
    assert len(result.captured_output) <= 8192 + len("*** (previous output truncated) ***\n")


# LLM-generated content at query #51
#--------------------------

```python
def test_error_wrapper_returns_custom_exception():
    import subprocess
    exc = subprocess.CalledProcessError(returncode=1, cmd="test")
    wrapped_exc = error_wrapper(exc)
    assert isinstance(wrapped_exc, subprocess.CalledProcessError)
    assert hasattr(wrapped_exc.__class__, "__str__")

def test_error_wrapper_returns_original_exception():
    exc = ValueError("test")
    wrapped_exc = error_wrapper(exc)
    assert wrapped_exc == exc
    assert not hasattr(wrapped_exc.__class__, "__str__")


# LLM-generated content at query #52
#--------------------------

```python
def test_error_wrapper_returns_same_error_for_non_subprocess_errors():
    class CustomError(Exception):
        pass

    custom_error = CustomError("Custom error message")
    wrapped_error = error_wrapper(custom_error)
    assert wrapped_error is custom_error

def test_error_wrapper_wraps_subprocess_called_process_error():
    error = subprocess.CalledProcessError(1, "cmd", output=b"output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.CalledProcessError)
    assert wrapped_error.__class__.__name__ == "CalledProcessError"

def test_error_wrapper_wraps_subprocess_timeout_expired():
    error = subprocess.TimeoutExpired("cmd", 10, output=b"output")
    wrapped_error = error_wrapper(error)
    assert isinstance(wrapped_error, subprocess.TimeoutExpired)
    assert wrapped_error.__class__.__name__ == "TimeoutExpired"


# LLM-generated content at query #53
#--------------------------

```python
def test_log_unicode_decode_error():
    output = b'\xc3\x28'  # Invalid UTF-8 sequence
    try:
        log(output.decode('utf-8'), timestamp=False, include_proc_id=False)
    except UnicodeDecodeError:
        pass


# LLM-generated content at query #54
#--------------------------

```python
def test_error_wrapper_with_called_process_error():
    err = subprocess.CalledProcessError(1, "cmd", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert "Captured output:" in str(wrapped_err)
    assert "test output" in str(wrapped_err)

def test_error_wrapper_with_timeout_expired():
    err = subprocess.TimeoutExpired("cmd", 10, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

def test_error_wrapper_with_other_exception():
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert str(wrapped_err) == "test error"

def test_error_wrapper_with_unicode_decode_error():
    err = subprocess.CalledProcessError(1, "cmd", output=b"\xff")
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)

def test_error_wrapper_with_no_output():
    err = subprocess.CalledProcessError(1, "cmd")
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)


# LLM-generated content at query #55
#--------------------------

```python
def test_error_wrapper_with_called_process_error():
    err = subprocess.CalledProcessError(1, "cmd", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)

def test_error_wrapper_with_timeout_expired():
    err = subprocess.TimeoutExpired("cmd", timeout=10, output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)

def test_error_wrapper_with_other_exception():
    err = ValueError("test error")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, ValueError)
    assert str(wrapped_err) == "test error"

def test_error_wrapper_with_no_output():
    err = subprocess.CalledProcessError(1, "cmd", output=None)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "No output was generated." in str(wrapped_err)

def test_error_wrapper_with_unicode_error():
    err = subprocess.CalledProcessError(1, "cmd", output=b"\xff")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #56
#--------------------------

```python
def test_run_command_success_no_output():
    result = run_command(["echo", "hello"], return_output=False)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"hello" in result.captured_output

def test_run_command_failure():
    result = run_command(["false"], ignore_errors=True)
    assert result.command == ["false"]
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_timeout():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.command == ["sleep", "2"]
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command(["echo", "hello"], verbose=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"hello" in result.captured_output

def test_run_command_with_env():
    result = run_command(["env"], env={"TEST_ENV": "123"}, return_output=True)
    assert result.command == ["env"]
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"TEST_ENV=123" in result.captured_output

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.command == ["pwd"]
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"/tmp" in result.captured_output

def test_run_command_with_shell():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.command == "echo hello"
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"hello" in result.captured_output


# LLM-generated content at query #57
#--------------------------

def test_run_command_success():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output
    assert result.command == ["echo", "hello"]

def test_run_command_failure():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is None

def test_run_command_timeout():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_with_output():
    result = run_command(["echo", "test"], return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output

def test_run_command_verbose(capsys):
    run_command(["echo", "verbose"], verbose=True)
    captured = capsys.readouterr()
    assert "verbose" in captured.out

def test_run_command_env():
    result = run_command(["printenv", "TEST_VAR"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"test_value" in result.captured_output

def test_run_command_cwd(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("content")
    result = run_command(["cat", "test.txt"], cwd=tmp_path, return_output=True)
    assert result.return_code == 0
    assert b"content" in result.captured_output


# LLM-generated content at query #58
#--------------------------

```python
def test_log_handles_unicode_decode_error():
    # Create a byte string that cannot be decoded as UTF-8
    non_utf8_bytes = b'\x80abc'
    # Mock the log function to capture its calls
    original_log = flutes.log.log
    logged_messages = []
    def mock_log(msg, **kwargs):
        logged_messages.append(msg)
    flutes.log.log = mock_log
    
    try:
        # Call the code path that would trigger the UnicodeDecodeError
        flutes.run.run_command(["echo", "test"], verbose=True, return_output=True)
        # Verify the mock was called with the raw bytes when decode fails
        assert any(isinstance(msg, str) and "b'" in msg for msg in logged_messages)
    finally:
        # Restore original log function
        flutes.log.log = original_log


# LLM-generated content at query #59
#--------------------------

```python
def test_error_wrapper_returns_input_when_not_subprocess_error():
    class CustomError(Exception):
        pass

    custom_error = CustomError("Custom error")
    result = error_wrapper(custom_error)
    assert result == custom_error


# LLM-generated content at query #60
#--------------------------

```
def test_run_command_return_output_true():
    result = run_command("echo test", return_output=True)
    assert result.captured_output is not None

def test_run_command_return_code_nonzero():
    result = run_command("false", ignore_errors=True)
    assert result.captured_output is not None

def test_run_command_verbose_true():
    result = run_command("echo test", verbose=True)
    assert result.captured_output is not None


# LLM-generated content at query #61
#--------------------------

```python
def test_run_command_success():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_failure():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode == 1

def test_run_command_timeout():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired:
        pass
    else:
        assert False, "Expected TimeoutExpired"

def test_run_command_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code == 1

def test_run_command_return_output():
    result = run_command(["echo", "world"], return_output=True)
    assert result.captured_output == b"world\n"

def test_run_command_verbose(capsys):
    run_command(["echo", "verbose"], verbose=True)
    captured = capsys.readouterr()
    assert "verbose" in captured.out

def test_run_command_truncate_output():
    long_output = "a" * 10000
    try:
        run_command(["echo", long_output], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert len(e.output) <= 8192 + len("*** (previous output truncated) ***\n")


# LLM-generated content at query #62
#--------------------------

```python
def test_error_wrapper_predicate():
    err = subprocess.CalledProcessError(1, "cmd")
    result = isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))
    assert result == True


# LLM-generated content at query #63
#--------------------------

```python
def test_error_wrapper_returns_non_subprocess_error_unchanged():
    class CustomError(Exception):
        pass

    custom_error = CustomError("Custom error message")
    result = error_wrapper(custom_error)
    assert result == custom_error


# LLM-generated content at query #64
#--------------------------

```
def test_error_wrapper_with_called_process_error():
    err = subprocess.CalledProcessError(1, 'cmd', output=b'some output')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert 'Captured output:' in str(wrapped_err)

def test_error_wrapper_with_timeout_expired():
    err = subprocess.TimeoutExpired('cmd', 10, output=b'timeout output')
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert 'Captured output:' in str(wrapped_err)

def test_error_wrapper_with_other_exception():
    err = ValueError('some error')
    wrapped_err = error_wrapper(err)
    assert wrapped_err is err
    assert str(wrapped_err) == 'some error'

def test_error_wrapper_with_no_output():
    err = subprocess.CalledProcessError(1, 'cmd', output=None)
    wrapped_err = error_wrapper(err)
    assert 'No output was generated.' in str(wrapped_err)

def test_error_wrapper_with_unicode_error_output():
    err = subprocess.CalledProcessError(1, 'cmd', output=b'\xff')
    wrapped_err = error_wrapper(err)
    assert 'Failed to parse output.' in str(wrapped_err)


# LLM-generated content at query #65
#--------------------------

def test_run_command_success():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output
    assert result.command == ["echo", "hello"]

def test_run_command_failure():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_timeout():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_with_output():
    result = run_command(["echo", "test"], return_output=True)
    assert result.captured_output is not None
    assert b"test" in result.captured_output

def test_run_command_without_output():
    result = run_command(["true"])
    assert result.captured_output is None

def test_run_command_verbose(capsys):
    run_command(["echo", "verbose"], verbose=True)
    captured = capsys.readouterr()
    assert "verbose" in captured.out

def test_run_command_env():
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"TEST_VAR=test_value" in result.captured_output

def test_run_command_cwd(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("test")
    result = run_command(["cat", "test.txt"], cwd=tmp_path, return_output=True)
    assert b"test" in result.captured_output

def test_run_command_shell():
    result = run_command("echo $0", shell=True, return_output=True)
    assert b"sh" in result.captured_output or b"bash" in result.captured_output


